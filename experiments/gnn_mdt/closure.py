"""Close the remaining MDT x GNN hypotheses with fair classical controls.

The suite has three deliberately different objectives:

``teacher``
    Distil a fixed MDT truncated-SVD embedding.  This tests fidelity, speed and
    inductive out-of-sample extension, but cannot establish an in-sample
    accuracy advantage over its own teacher.
``gae``
    Reconstruct the union of the per-view kNN graphs with an inner-product
    decoder.  This is a genuinely different unsupervised objective.
``dgi``
    Deep Graph Infomax-style feature corruption.  This is a second objective
    that does not inherit the SVD ceiling.

Every self-supervised objective is run with a message-passing-free MLP control
or frozen uniform view fusion.  Labels are used only after fitting, for AMI.
The implementation needs torch/scipy/sklearn only (no torch-geometric).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy.io
import scipy.sparse as sp
import torch
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist
from scipy.sparse.linalg import eigsh, svds
from scipy.stats import t as student_t
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class Split:
    train: list[np.ndarray]
    test: list[np.ndarray]
    y_train: np.ndarray
    y_test: np.ndarray
    name: str


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(path: str, name: str, split: float, seed: int,
               cap_train: int | None, cap_test: int | None) -> Split:
    # expanduser: configs point at ~/.cache/... because /tmp is wiped on reboot,
    # and neither Path() nor scipy expand "~" on their own.
    mat = scipy.io.loadmat(str(Path(path).expanduser() / f"{name}.mat"))
    raw = [np.asarray(x, dtype=np.float32) for x in mat["X"].ravel()]
    y = np.asarray(mat["y"] if "y" in mat else mat["Y"]).ravel()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    n_train = int(split * len(y))
    tr, te = order[:n_train], order[n_train:]
    if cap_train:
        tr = tr[:cap_train]
    if cap_test:
        te = te[:cap_test]
    train, test = [], []
    for view in raw:
        scaler = StandardScaler().fit(np.nan_to_num(view[tr]))
        train.append(np.nan_to_num(scaler.transform(view[tr])).astype(np.float32))
        test.append(np.nan_to_num(scaler.transform(view[te])).astype(np.float32))
    return Split(train, test, y[tr], y[te], name)


def make_synthetic(seed: int = 0, n_train: int = 180, n_test: int = 90,
                   n_clusters: int = 3) -> Split:
    """Small multi-view problem used by smoke tests and CI."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    y = np.arange(n) % n_clusters
    rng.shuffle(y)
    centers = rng.normal(size=(n_clusters, 4)) * 3
    latent = centers[y] + rng.normal(scale=.55, size=(n, 4))
    view1 = latent @ rng.normal(size=(4, 10)) + rng.normal(scale=.3, size=(n, 10))
    view2 = np.tanh(latent @ rng.normal(size=(4, 7))) + rng.normal(scale=.12, size=(n, 7))
    train, test = [], []
    for view in (view1, view2):
        scaler = StandardScaler().fit(view[:n_train])
        train.append(scaler.transform(view[:n_train]).astype(np.float32))
        test.append(scaler.transform(view[n_train:]).astype(np.float32))
    return Split(train, test, y[:n_train], y[n_train:], "synthetic")


def _row_normalize(matrix: sp.spmatrix) -> sp.csr_matrix:
    """Row-stochastic normalisation, safe against float32 denormal degrees.

    ``degree > 0`` is not a sufficient guard.  An out-of-sample row for a test
    point far from every training point has kernel weights ``exp(-d^2/2s^2)``
    that underflow toward the float32 denormal range, so its degree can be a
    *positive* value near 1e-40 whose float32 reciprocal overflows to ``inf``.
    That turns one row into ``nan`` and, because a single nan disqualifies an
    embedding, silently drops that whole method from the cell -- and it does so
    preferentially on the hardest datasets, which is a biased missingness, not
    noise.  Reciprocate in float64 and treat an unrepresentable degree as a
    disconnected row (embedding at the origin), which is the honest answer for
    a point with no usable affinity.
    """
    matrix = matrix.tocsr().astype(np.float32)
    degree = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)
    inv = np.zeros_like(degree)
    usable = degree > np.finfo(np.float32).tiny
    inv[usable] = 1.0 / degree[usable]
    return (sp.diags(inv.astype(np.float32)) @ matrix).tocsr()


def knn_transitions(x_train: np.ndarray, x_test: np.ndarray, knn: int,
                    sigma: float | None = None,
                    alpha: float = 0.0,
                    normalize: bool = True) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """MDT-compatible Gaussian kNN train and test-to-train transitions.

    The train graph includes the self-neighbour and is symmetrised before row
    normalisation, matching ``src.mdt_operators.transition_matrix``.  The OOS
    graph is directed because test rows only connect to training columns.

    ``normalize=False`` returns the raw symmetric kernels instead of the
    row-stochastic operators.  Only Multi-View Diffusion Maps needs this: it
    normalises its own block operator, so it must not receive pre-normalised
    blocks (see ``experiments.mvbench.bench.mvd``).

    ``alpha`` is the Coifman-Lafon density-normalisation exponent applied to the
    kernel before row normalisation: ``K_a[i, j] = K[i, j] / (d_i^a d_j^a)``.
    ``alpha=0`` is the plain row-stochastic operator used by every earlier MDT
    run; ``alpha=1`` removes the sampling density and leaves the Laplace-Beltrami
    diffusion, which is the reason to expect it to help hub-dominated views.
    """
    n = len(x_train)
    k_train = min(max(1, knn), n)
    fit = NearestNeighbors(n_neighbors=k_train).fit(x_train)
    d_train, i_train = fit.kneighbors(x_train)
    if sigma is None:
        positive = d_train[d_train > 0]
        sigma = float(np.median(positive)) if len(positive) else 1.0
    sigma = max(float(sigma), 1e-8)
    rows = np.repeat(np.arange(n), k_train)
    weights = np.exp(-(d_train.ravel() ** 2) / (2 * sigma ** 2)).astype(np.float32)
    graph = sp.csr_matrix((weights, (rows, i_train.ravel())), shape=(n, n))
    graph = graph.maximum(graph.T)

    k_test = min(k_train, n)
    d_test, i_test = fit.kneighbors(x_test, n_neighbors=k_test)
    test_rows = np.repeat(np.arange(len(x_test)), k_test)
    test_weights = np.exp(-(d_test.ravel() ** 2) / (2 * sigma ** 2)).astype(np.float32)
    bipartite = sp.csr_matrix(
        (test_weights, (test_rows, i_test.ravel())), shape=(len(x_test), n)
    )
    if alpha:
        degree = np.asarray(graph.sum(axis=1)).ravel()
        scale = sp.diags(np.power(np.maximum(degree, 1e-12), -float(alpha)))
        graph = (scale @ graph @ scale).tocsr()
        # Only the train-column factor is applied out of sample: the test row's
        # own density factor is constant along its row and cancels in the row
        # normalisation below, so estimating a test degree would change nothing.
        bipartite = (bipartite @ scale).tocsr()
    if not normalize:
        return graph.tocsr(), bipartite.tocsr()
    return _row_normalize(graph), _row_normalize(bipartite)


def build_graphs(split: Split, knn: int, alpha: float = 0.0,
                 normalize: bool = True) -> tuple[list[sp.csr_matrix], list[sp.csr_matrix]]:
    train, test = [], []
    for xtr, xte in zip(split.train, split.test):
        # Match the existing MDT experiments: a global median bandwidth,
        # estimated on at most 200 points, followed by kNN sparsification.
        distances = pdist(xtr[:200])
        sigma = float(np.quantile(distances, .5)) if len(distances) else 1.0
        p, b = knn_transitions(xtr, xte, knn, sigma, alpha, normalize)
        train.append(p)
        test.append(b)
    return train, test


def shuffle_graphs(train: Sequence[sp.csr_matrix], test: Sequence[sp.csr_matrix],
                   seed: int) -> tuple[list[sp.csr_matrix], list[sp.csr_matrix]]:
    """Relabel the nodes of every graph, leaving the features in place.

    This is the capacity-matched null for message passing: the model keeps all
    of its weights active and still aggregates over k neighbours with the same
    kernel weights, degrees and spectrum, but the neighbours belong to another
    node.  A variant that scores the same with shuffled graphs was never using
    graph structure, only the extra live parameters.
    """
    order = np.random.default_rng(seed).permutation(train[0].shape[0])
    # ponytail: one permutation shared by all views, matching the real setting
    # where every view indexes the same node set.
    return ([p[order][:, order].tocsr() for p in train],
            [b[:, order].tocsr() for b in test])


def trajectory(n_views: int, steps: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).dirichlet(np.ones(n_views), size=steps)


def mdt_operator(graphs: Sequence[sp.csr_matrix], weights: np.ndarray) -> sp.csr_matrix:
    steps = [sum(float(a) * p for a, p in zip(row, graphs)).tocsr() for row in weights]
    operator = steps[0]
    for step in steps[1:]:
        operator = (step @ operator).tocsr()
    return operator


def mdt_oos_operator(train_graphs: Sequence[sp.csr_matrix],
                     test_graphs: Sequence[sp.csr_matrix],
                     weights: np.ndarray) -> sp.csr_matrix:
    train_steps = [sum(float(a) * p for a, p in zip(row, train_graphs)).tocsr()
                   for row in weights[:-1]]
    last = sum(float(a) * p for a, p in zip(weights[-1], test_graphs)).tocsr()
    if not train_steps:
        return last
    prefix = train_steps[0]
    for step in train_steps[1:]:
        prefix = (step @ prefix).tocsr()
    return (last @ prefix).tocsr()


def svd_embedding(operator: sp.spmatrix, n_components: int,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rank = min(n_components + 1, min(operator.shape) - 1)
    u, values, vt = svds(operator.astype(np.float64), k=rank, which="LM",
                         random_state=seed)
    order = np.argsort(values)[::-1]
    u, values, vt = u[:, order], values[order], vt[order]
    kept = slice(1, min(n_components + 1, len(values)))
    return (u[:, kept] * values[kept]).astype(np.float32), values, vt.T


def consensus_embedding(graphs: Sequence[sp.csr_matrix],
                        test_graphs: Sequence[sp.csr_matrix], steps: int,
                        n_components: int, count: int,
                        seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Average MDT Gram targets and extend that consensus out of sample.

    For trajectory ``i``, ``F_i`` and ``F*_i`` are the train SVD factor and
    its Nyström test extension.  We eigendecompose the mean train Gram and
    extend with the matching mean cross-Gram.  This avoids pairing a consensus
    train embedding with the unrelated extension of a single trajectory.
    """
    n = graphs[0].shape[0]
    target = np.zeros((n, n), dtype=np.float64)
    cross = np.zeros((test_graphs[0].shape[0], n), dtype=np.float64)
    for index in range(count):
        weights = trajectory(len(graphs), steps, seed * 1000 + index)
        w = mdt_operator(graphs, weights)
        factor, _, right = svd_embedding(w, n_components, seed + index)
        test_factor = np.asarray(
            mdt_oos_operator(graphs, test_graphs, weights)
            @ right[:, 1:n_components + 1]
        )
        target += factor @ factor.T
        cross += test_factor @ factor.T
    target /= count
    cross /= count
    values, vectors = eigsh(sp.csr_matrix(target), k=min(n_components, n - 1), which="LA")
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    train = vectors * np.sqrt(np.clip(values, 0, None))
    test = cross @ vectors @ np.diag(1.0 / np.sqrt(np.clip(values, 1e-12, None)))
    return train.astype(np.float32), test.astype(np.float32)


def scipy_to_torch(matrix: sp.spmatrix, device: torch.device) -> torch.Tensor:
    coo = matrix.tocoo()
    index = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long, device=device)
    value = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(index, value, coo.shape, device=device).coalesce()


class MultiViewSAGE(torch.nn.Module):
    """Two-hop relation-aware GraphSAGE with a native sparse implementation."""

    def __init__(self, dimensions: Sequence[int], hidden: int, output: int,
                 message_passing: bool = True, learn_fusion: bool = False):
        super().__init__()
        self.message_passing = message_passing
        self.input = torch.nn.ModuleList(torch.nn.Linear(d, hidden) for d in dimensions)
        self.layer1 = torch.nn.Linear(2 * hidden, hidden)
        self.layer2 = torch.nn.Linear(2 * hidden, output)
        self.fusion_logits = torch.nn.Parameter(
            torch.zeros(len(dimensions)), requires_grad=learn_fusion
        )
        self.discriminator = torch.nn.Bilinear(output, output, 1, bias=False)
        # AlphaFold-style recycling: the previous cycle's embedding re-enters the
        # input representation.  Zero-initialised so the first cycle is exactly
        # the non-recycled model and the baseline stays nested inside this one.
        self.recycle = torch.nn.Linear(output, hidden)
        torch.nn.init.zeros_(self.recycle.weight)
        torch.nn.init.zeros_(self.recycle.bias)

    def _input(self, views: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return [torch.relu(layer(x)) for layer, x in zip(self.input, views)]

    def _fuse(self, values: Sequence[torch.Tensor]) -> torch.Tensor:
        weights = torch.softmax(self.fusion_logits, dim=0)
        return sum(weight * value for weight, value in zip(weights, values))

    def _aggregate(self, graphs: Sequence[torch.Tensor],
                   values: Sequence[torch.Tensor]) -> torch.Tensor:
        if not self.message_passing:
            return torch.zeros_like(values[0])
        return self._fuse([torch.sparse.mm(graph, value)
                           for graph, value in zip(graphs, values)])

    def forward_train(self, views: Sequence[torch.Tensor],
                      graphs: Sequence[torch.Tensor],
                      previous: torch.Tensor | None = None) -> tuple[torch.Tensor, tuple]:
        h0_views = self._input(views)
        h0 = self._fuse(h0_views)
        if previous is not None:
            h0 = h0 + self.recycle(previous)
        h1 = torch.relu(self.layer1(torch.cat((h0, self._aggregate(graphs, h0_views)), dim=1)))
        h1_views = [h1 for _ in graphs]
        z = self.layer2(torch.cat((h1, self._aggregate(graphs, h1_views)), dim=1))
        return z, (h0_views, h1_views)

    def forward_test(self, train_views: Sequence[torch.Tensor],
                     test_views: Sequence[torch.Tensor],
                     train_graphs: Sequence[torch.Tensor],
                     test_graphs: Sequence[torch.Tensor],
                     previous_train: torch.Tensor | None = None,
                     previous_test: torch.Tensor | None = None) -> torch.Tensor:
        train_h0_views = self._input(train_views)
        test_h0_views = self._input(test_views)
        train_h0, test_h0 = self._fuse(train_h0_views), self._fuse(test_h0_views)
        if previous_train is not None:
            train_h0 = train_h0 + self.recycle(previous_train)
        if previous_test is not None:
            test_h0 = test_h0 + self.recycle(previous_test)
        train_h1 = torch.relu(self.layer1(torch.cat(
            (train_h0, self._aggregate(train_graphs, train_h0_views)), dim=1)))
        if self.message_passing:
            test_agg0 = self._fuse([torch.sparse.mm(graph, value)
                                    for graph, value in zip(test_graphs, train_h0_views)])
        else:
            test_agg0 = torch.zeros_like(test_h0)
        test_h1 = torch.relu(self.layer1(torch.cat((test_h0, test_agg0), dim=1)))
        if self.message_passing:
            test_agg1 = self._fuse([torch.sparse.mm(graph, train_h1)
                                    for graph in test_graphs])
        else:
            test_agg1 = torch.zeros_like(test_h1)
        return self.layer2(torch.cat((test_h1, test_agg1), dim=1))


def union_edges(graphs: Sequence[sp.csr_matrix]) -> np.ndarray:
    union = sum((graph > 0).astype(np.int8) for graph in graphs).tocsr()
    union.setdiag(0)
    union.eliminate_zeros()
    row, col = union.nonzero()
    keep = row < col
    return np.column_stack((row[keep], col[keep])).astype(np.int64)


def gae_loss(z: torch.Tensor, positive: torch.Tensor,
             generator: torch.Generator) -> torch.Tensor:
    n, count = len(z), len(positive)
    negative = torch.randint(n, (count, 2), generator=generator, device=z.device)
    pos_logits = (z[positive[:, 0]] * z[positive[:, 1]]).sum(1) / math.sqrt(z.shape[1])
    neg_logits = (z[negative[:, 0]] * z[negative[:, 1]]).sum(1) / math.sqrt(z.shape[1])
    return (torch.nn.functional.softplus(-pos_logits).mean()
            + torch.nn.functional.softplus(neg_logits).mean())


def dgi_loss(model: MultiViewSAGE, positive: torch.Tensor,
             negative: torch.Tensor) -> torch.Tensor:
    summary = torch.sigmoid(positive.mean(0, keepdim=True)).expand_as(positive)
    pos_logits = model.discriminator(positive, summary).squeeze(1)
    neg_logits = model.discriminator(negative, summary).squeeze(1)
    return (torch.nn.functional.softplus(-pos_logits).mean()
            + torch.nn.functional.softplus(neg_logits).mean())


def fit_model(split: Split, train_graphs: Sequence[sp.csr_matrix],
              test_graphs: Sequence[sp.csr_matrix], target: np.ndarray,
              variant: str, hidden: int, epochs: int, learning_rate: float,
              weight_decay: float, patience: int, seed: int,
              edge_graphs: Sequence[sp.csr_matrix] | None = None,
              recycles: int = 1) -> tuple[np.ndarray, np.ndarray, dict]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objective = "teacher" if variant.startswith("teacher") else variant.split("_")[0]
    # Substring rather than suffix tests, so an arm can compose flags:
    # ``teacher_mlp_pair_recycle`` is feature-only, pair-target and recycled.
    message_passing = "mlp" not in variant
    learn_fusion = "learned" in variant
    pair_target = "pair" in variant
    cycles = max(1, recycles) if "recycle" in variant else 1
    output = target.shape[1]
    model = MultiViewSAGE([x.shape[1] for x in split.train], hidden, output,
                          message_passing, learn_fusion).to(device)
    xtr = [torch.tensor(x, device=device) for x in split.train]
    xte = [torch.tensor(x, device=device) for x in split.test]
    ptr = [scipy_to_torch(p, device) for p in train_graphs]
    pte = [scipy_to_torch(p, device) for p in test_graphs]
    teacher = torch.tensor(target, device=device)
    teacher = teacher / teacher.std(0, keepdim=True).clamp_min(1e-6)
    # The pair arm regresses the teacher's Gram matrix instead of its coordinates.
    # Coordinates are only defined up to rotation -- which is why fidelity needs
    # Procrustes -- whereas the Gram is gauge-free, so the student is no longer
    # asked to guess the teacher's arbitrary frame.  Scaled to unit RMS so the
    # loss magnitude, and hence the shared learning rate, stays comparable.
    gram = teacher @ teacher.T
    gram = gram / gram.pow(2).mean().sqrt().clamp_min(1e-6)
    # The GAE target must stay the true graph even when the aggregation graphs
    # are shuffled, otherwise the control changes the task as well as the
    # message source and the two effects are no longer separable.
    edges = torch.tensor(union_edges(edge_graphs if edge_graphs is not None else train_graphs),
                         dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    generator = torch.Generator(device=device).manual_seed(seed)
    best_loss, best_state, stale = float("inf"), None, 0
    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        # AlphaFold recycling: all but the last cycle run without gradients, so
        # the parameter count is unchanged and the extra cost is forward-only.
        previous = None
        with torch.no_grad():
            for _ in range(cycles - 1):
                previous, _ = model.forward_train(xtr, ptr, previous)
        z, _ = model.forward_train(xtr, ptr, previous)
        if objective == "teacher":
            loss = (torch.nn.functional.mse_loss(z @ z.T, gram) if pair_target
                    else torch.nn.functional.mse_loss(z, teacher))
        elif objective == "gae":
            loss = gae_loss(z, edges, generator)
        elif objective == "dgi":
            permutations = [torch.randperm(len(x), generator=generator, device=device) for x in xtr]
            corrupted, _ = model.forward_train([x[p] for x, p in zip(xtr, permutations)], ptr)
            loss = dgi_loss(model, z, corrupted)
        else:
            raise ValueError(f"unknown objective in {variant!r}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss - 1e-6:
            best_loss = value
            best_state = {key: item.detach().cpu().clone()
                          for key, item in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    train_seconds = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        previous_train, previous_test = None, None
        for _ in range(cycles - 1):
            previous_train, _ = model.forward_train(xtr, ptr, previous_train)
            previous_test = model.forward_test(xtr, xte, ptr, pte,
                                               previous_train, previous_test)
        ztr, _ = model.forward_train(xtr, ptr, previous_train)
        zte = model.forward_test(xtr, xte, ptr, pte, previous_train, previous_test)
    inference_seconds = time.perf_counter() - start
    info = {
        "loss": best_loss,
        "cycles": cycles,
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "fusion": torch.softmax(model.fusion_logits.detach().cpu(), 0).numpy().tolist(),
        "device": str(device),
    }
    return ztr.cpu().numpy(), zte.cpu().numpy(), info


def cluster_metrics(train_embedding: np.ndarray, test_embedding: np.ndarray,
                    y_train: np.ndarray, y_test: np.ndarray, clusters: int,
                    seed: int) -> dict[str, float]:
    scaler = StandardScaler().fit(train_embedding)
    train_embedding = scaler.transform(train_embedding)
    test_embedding = scaler.transform(test_embedding)
    km = KMeans(clusters, n_init=20, random_state=seed).fit(train_embedding)
    standalone = KMeans(clusters, n_init=20, random_state=seed).fit_predict(test_embedding)
    return {
        "train_ami": float(adjusted_mutual_info_score(y_train, km.labels_)),
        "test_ami": float(adjusted_mutual_info_score(y_test, standalone)),
        "inductive_ami": float(adjusted_mutual_info_score(y_test, km.predict(test_embedding))),
    }


def procrustes_fidelity(embedding: np.ndarray, target: np.ndarray) -> float:
    a = StandardScaler().fit_transform(embedding)
    b = StandardScaler().fit_transform(target)
    rotation, _ = orthogonal_procrustes(a, b)
    residual = np.square(a @ rotation - b).sum()
    return float(max(0.0, 1.0 - residual / max(np.square(b).sum(), 1e-12)))


def run_one(split: Split, cfg: dict, seed: int) -> list[dict]:
    clusters = len(np.unique(split.y_train))
    knn_cfg = cfg["mdt"].get("knn", "auto")
    knn = int(np.floor(np.log(len(split.y_train)))) if knn_cfg == "auto" else int(knn_cfg)
    alpha = float(cfg["mdt"].get("alpha", 0.0))
    graph_start = time.perf_counter()
    train_graphs, test_graphs = build_graphs(split, knn, alpha)
    graph_seconds = time.perf_counter() - graph_start
    weights = trajectory(len(train_graphs), int(cfg["mdt"]["steps"]), seed)

    svd_start = time.perf_counter()
    operator = mdt_operator(train_graphs, weights)
    svd_train, _, right = svd_embedding(operator, clusters, seed)
    svd_seconds = time.perf_counter() - svd_start
    oos = mdt_oos_operator(train_graphs, test_graphs, weights)
    nystrom = np.asarray(oos @ right[:, 1:clusters + 1], dtype=np.float32)
    classical = cluster_metrics(svd_train, nystrom, split.y_train, split.y_test, clusters, seed)
    classical.update({
        "dataset": split.name, "seed": seed, "method": "mdt_nystrom",
        "graph_seconds": graph_seconds, "train_seconds": svd_seconds,
        "inference_seconds": 0.0, "fidelity": 1.0,
    })
    rows = [classical]

    feature_train = np.concatenate(split.train, axis=1)
    feature_test = np.concatenate(split.test, axis=1)
    features = cluster_metrics(feature_train, feature_test, split.y_train, split.y_test,
                               clusters, seed)
    features.update({"dataset": split.name, "seed": seed, "method": "features",
                     "graph_seconds": 0.0, "train_seconds": 0.0,
                     "inference_seconds": 0.0, "fidelity": 0.0})
    rows.append(features)

    consensus_start = time.perf_counter()
    consensus, consensus_test = consensus_embedding(
        train_graphs, test_graphs, int(cfg["mdt"]["steps"]), clusters,
        int(cfg["mdt"]["trajectories"]), seed
    )
    consensus_metrics = cluster_metrics(consensus, consensus_test, split.y_train, split.y_test,
                                        clusters, seed)
    consensus_metrics.update({
        "dataset": split.name, "seed": seed, "method": "mdt_consensus",
        "graph_seconds": graph_seconds,
        "train_seconds": time.perf_counter() - consensus_start,
        "inference_seconds": 0.0, "fidelity": 0.0,
    })
    rows.append(consensus_metrics)

    gnn = cfg["gnn"]
    shuffled = shuffle_graphs(train_graphs, test_graphs, seed)
    for variant in gnn["variants"]:
        graphs = shuffled if variant.endswith("shuffled") else (train_graphs, test_graphs)
        ztr, zte, info = fit_model(
            split, graphs[0], graphs[1], svd_train, variant,
            int(gnn["hidden"]), int(gnn["epochs"]), float(gnn["learning_rate"]),
            float(gnn["weight_decay"]), int(gnn["patience"]), seed, train_graphs,
            int(gnn.get("recycles", 3))
        )
        result = cluster_metrics(ztr, zte, split.y_train, split.y_test, clusters, seed)
        result.update({
            "dataset": split.name, "seed": seed, "method": variant,
            "graph_seconds": graph_seconds, **info,
            "fidelity": procrustes_fidelity(ztr, svd_train),
        })
        rows.append(result)
    for row in rows:
        row["alpha"] = alpha
    return rows


def paired_effect(rows: Sequence[dict], candidate: str, baseline: str,
                  metric: str) -> tuple[float | None, float | None, int, int]:
    by_key = {(row["dataset"], row["seed"], row["method"]): row for row in rows}
    effects = []
    grouped: dict[str, list[float]] = {}
    for dataset, seed, method in by_key:
        if method != candidate:
            continue
        other = by_key.get((dataset, seed, baseline))
        if other:
            value = by_key[(dataset, seed, candidate)][metric] - other[metric]
            effects.append(value)
            grouped.setdefault(dataset, []).append(value)
    if not effects:
        return None, None, 0, 0
    # Seeds from the same dataset are not independent benchmark tasks.  The
    # interval therefore treats datasets as clusters and gives each dataset
    # equal weight, avoiding seed-level pseudo-replication.
    cluster_effects = np.asarray([np.mean(values) for values in grouped.values()])
    return (float(cluster_effects.mean()),
            float(cluster_effects.std(ddof=1) if len(cluster_effects) > 1 else 0.0),
            len(effects), len(cluster_effects))


def summarize(rows: Sequence[dict], min_effect: float = .02) -> dict:
    methods = sorted({row["method"] for row in rows})
    summary = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            key: {"mean": float(np.mean([row[key] for row in selected])),
                  "std": float(np.std([row[key] for row in selected]))}
            for key in ("train_ami", "test_ami", "inductive_ami", "train_seconds", "fidelity")
        }
    def decision(candidate: str, baseline: str, metric: str) -> dict:
        effect, std, count, groups = paired_effect(rows, candidate, baseline, metric)
        if effect is None:
            return {"effect": None, "std": None, "pairs": 0, "datasets": 0,
                    "verdict": "inconclusive"}
        critical = float(student_t.ppf(.975, groups - 1)) if groups > 1 else None
        half_width = critical * std / math.sqrt(groups) if critical is not None else None
        lower = effect - half_width if half_width is not None else None
        upper = effect + half_width if half_width is not None else None
        # A smaller pilot is useful diagnostically but must not pronounce a
        # fate. Five independent datasets is the minimum for a verdict.
        if groups < 5:
            verdict = "inconclusive"
        elif effect >= min_effect and lower is not None and lower > 0:
            verdict = "continue"
        elif upper is not None and upper < min_effect:
            verdict = "dead_end"
        else:
            verdict = "inconclusive"
        return {"effect": effect, "std": std, "pairs": count, "datasets": groups,
                "ci95": [lower, upper] if lower is not None else None,
                "verdict": verdict}

    summary["decisions"] = {
        "teacher_oos": decision("teacher_gnn", "mdt_nystrom", "test_ami"),
        "teacher_message_passing": decision("teacher_gnn", "teacher_mlp", "test_ami"),
        "teacher_fidelity": decision("teacher_gnn", "teacher_mlp", "fidelity"),
        "gae_vs_consensus": decision("gae_learned", "mdt_consensus", "train_ami"),
        "gae_message_passing": decision("gae_uniform", "gae_mlp", "train_ami"),
        # Capacity-matched null: same live weights, same degrees, wrong
        # neighbours.  Reading order matters -- gae_message_passing only means
        # "graph structure helps" if gae_graph_signal also clears min_effect.
        "gae_graph_signal": decision("gae_uniform", "gae_shuffled", "train_ami"),
        "teacher_graph_signal": decision("teacher_gnn", "teacher_shuffled", "test_ami"),
        "gae_learned_fusion": decision("gae_learned", "gae_uniform", "train_ami"),
        "dgi_vs_consensus": decision("dgi_learned", "mdt_consensus", "train_ami"),
        "dgi_learned_fusion": decision("dgi_learned", "dgi_uniform", "train_ami"),
        # AlphaFold-family arms, read against the surviving feature-only map and
        # against the training-free baseline it has to beat to matter.
        "recycle_vs_mlp": decision("teacher_mlp_recycle", "teacher_mlp", "inductive_ami"),
        "pair_vs_mlp": decision("teacher_mlp_pair", "teacher_mlp", "inductive_ami"),
        "pair_recycle_vs_mlp": decision("teacher_mlp_pair_recycle", "teacher_mlp",
                                        "inductive_ami"),
        "recycle_vs_nystrom": decision("teacher_mlp_recycle", "mdt_nystrom", "inductive_ami"),
        "pair_vs_nystrom": decision("teacher_mlp_pair", "mdt_nystrom", "inductive_ami"),
    }
    if "teacher_gnn" in summary and "mdt_nystrom" in summary:
        gnn_time = summary["teacher_gnn"]["train_seconds"]["mean"]
        svd_time = summary["mdt_nystrom"]["train_seconds"]["mean"]
        ratio = gnn_time / max(svd_time, 1e-12)
        summary["decisions"]["compute"] = {
            "gnn_over_svds_ratio": ratio,
            "verdict": "continue" if ratio < 1 else "dead_end",
        }
    return summary


def load_config(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/gnn_mdt/config.yml")
    parser.add_argument("--smoke", action="store_true",
                        help="run a short synthetic end-to-end check")
    parser.add_argument("--summarize", metavar="JSONL",
                        help="summarize an existing metrics JSONL file")
    parser.add_argument("--datasets", nargs="+", help="override configured datasets")
    parser.add_argument("--seeds", nargs="+", type=int, help="override configured seeds")
    parser.add_argument("--epochs", type=int, help="override configured epoch budget")
    parser.add_argument("--alpha", type=float,
                        help="Coifman-Lafon density-normalisation exponent for the views")
    parser.add_argument("--variants", nargs="+", help="override configured GNN variants")
    parser.add_argument("--output", help="override metrics JSONL path")
    parser.add_argument("--summary-output", help="write the aggregate summary as JSON")
    parser.add_argument("--resume", action="store_true",
                        help="append and skip dataset/seed pairs already complete")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.datasets:
        cfg["data"]["datasets"] = args.datasets
    if args.seeds:
        cfg["evaluation"]["seeds"] = args.seeds
    if args.epochs:
        cfg["gnn"]["epochs"] = args.epochs
    if args.alpha is not None:
        cfg["mdt"]["alpha"] = args.alpha
    if args.variants:
        cfg["gnn"]["variants"] = args.variants
    if args.output:
        cfg["evaluation"]["output"] = args.output
    if args.summarize:
        with open(args.summarize, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        report = summarize(rows, cfg["evaluation"]["min_effect"])
        if args.summary_output:
            Path(args.summary_output).write_text(json.dumps(report, indent=2) + "\n",
                                                 encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    if args.smoke:
        cfg["gnn"].update(epochs=20, patience=8, hidden=32)
        if not args.variants:
            cfg["gnn"]["variants"] = ["teacher_gnn", "gae_uniform", "gae_learned", "dgi_uniform"]
        cfg["mdt"]["trajectories"] = 3
        splits = [(0, make_synthetic(0))]
        output = Path("/tmp/gnn_mdt_smoke.jsonl")
    else:
        data = cfg["data"]
        splits = []
        for seed in cfg["evaluation"]["seeds"]:
            for name in data["datasets"]:
                splits.append((seed, load_split(
                    data["path"], name, float(data["split"]), seed,
                    data.get("cap_train"), data.get("cap_test")
                )))
        output = Path(cfg["evaluation"]["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    completed = set()
    expected = 3 + len(cfg["gnn"]["variants"])
    if args.resume and output.exists():
        with output.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        counts = {}
        for row in rows:
            key = (row["dataset"], row["seed"])
            counts[key] = counts.get(key, 0) + 1
        completed = {key for key, count in counts.items() if count >= expected}
    mode = "a" if args.resume else "w"
    with output.open(mode, encoding="utf-8") as handle:
        for seed, split in splits:
            if (split.name, seed) in completed:
                print(f"[resume] skipping {split.name} seed={seed}", flush=True)
                continue
            for row in run_one(split, cfg, seed):
                rows.append(row)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                print(json.dumps(row), flush=True)
    report = summarize(rows, cfg["evaluation"]["min_effect"])
    summary_output = Path(args.summary_output) if args.summary_output else output.with_name("summary.json")
    summary_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
