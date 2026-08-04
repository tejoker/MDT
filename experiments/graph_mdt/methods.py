"""Method grid for the second-formulation ledger (step 3).

Embedding methods, all label-free:
  mdt_*      sparse/implicit MDT (works at 24k nodes, no dense n^2)
  gcn|sage x gae|bgrl   2-layer GNN backbones, self-supervised objectives
  mlp_ssl    same capacity, BGRL-style objective, no graph (feature ceiling)
  distill    GNN -> MLP embedding distillation (graph needed at inference?)

Graph fusion for GNNs is the uniform mean of normalised adjacencies
(learned global fusion was closed by the first ledger). Checkpoint selection
uses the silhouette criterion only (pre-registered, label-free).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import LinearOperator, svds
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from experiments.gnn_mdt.closure import _row_normalize, trajectory

STEPS = 4
CONSENSUS_TRAJECTORIES = 8


# --- MDT, implicit sparse path -------------------------------------------

def _product_operator(views: list, weights: np.ndarray) -> LinearOperator:
    """W = S_T ... S_1 with S_i = sum_v w_iv P_v, never materialised."""
    steps = [[(float(w), views[v]) for v, w in enumerate(row) if w > 1e-12]
             for row in weights]

    def matvec(x):
        for step in steps:
            x = sum(w * (p @ x) for w, p in step)
        return x

    def rmatvec(x):
        for step in reversed(steps):
            x = sum(w * (p.T @ x) for w, p in step)
        return x

    n = views[0].shape[0]
    return LinearOperator((n, n), matvec=matvec, rmatvec=rmatvec,
                          dtype=np.float64)


def _svd_factor(operator: LinearOperator, n_components: int,
                seed: int) -> np.ndarray:
    from scipy.sparse.linalg._eigen.arpack import ArpackError
    rank = min(n_components + 1, operator.shape[0] - 1)
    try:
        # maxiter capped: degenerate operators (near-empty views) must fail
        # in seconds and be skipped, not burn 10*n ARPACK iterations
        u, values, _ = svds(operator, k=rank, which="LM", random_state=seed,
                            maxiter=300)
    except ArpackError:  # incl. NoConvergence and "no shifts" (error 3)
        return None  # degenerate operator; caller skips this trajectory
    order = np.argsort(values)[::-1]
    u, values = u[:, order], values[order]
    kept = slice(1, min(n_components + 1, len(values)))
    return (u[:, kept] * values[kept]).astype(np.float32)


def silhouette_criterion(embedding: np.ndarray, k: int, seed: int = 0) -> float:
    sample = np.random.default_rng(seed).choice(
        len(embedding), min(2000, len(embedding)), replace=False)
    sub = embedding[sample]
    pred = KMeans(k, n_init=4, random_state=seed).fit_predict(sub)
    if len(np.unique(pred)) < 2:
        return -1.0
    return float(silhouette_score(sub, pred))


def modularity(partition: np.ndarray, a: sp.csr_matrix) -> float:
    """Newman modularity of a hard partition on a symmetric weighted graph."""
    two_m = a.sum()
    if two_m == 0:
        return -1.0
    degrees = np.asarray(a.sum(axis=1)).ravel()
    q = 0.0
    for c in np.unique(partition):
        idx = np.where(partition == c)[0]
        within = a[idx][:, idx].sum()
        q += within / two_m - (degrees[idx].sum() / two_m) ** 2
    return float(q)


def modularity_criterion(embedding: np.ndarray, k: int, scorer: sp.csr_matrix,
                         seed: int = 0) -> float:
    """Label-free selection score: modularity of KMeans(k) on the union
    input graph. Replaces silhouette, which rewards collapsed diffusion
    embeddings (ACM placement: consensus NMI 0.63 vs silhouette-selected
    0.004 — see ledger amendment)."""
    pred = KMeans(k, n_init=4, random_state=seed).fit_predict(embedding)
    if len(np.unique(pred)) < 2:
        return -1.0
    return modularity(pred, scorer)


def mdt_consensus(views: list, n_components: int, seed: int = 0) -> np.ndarray:
    """Mean-Gram consensus via thin SVD of stacked factors (no dense n^2)."""
    factors = []
    for index in range(CONSENSUS_TRAJECTORIES):
        weights = trajectory(len(views), STEPS, seed * 1000 + index)
        factor = _svd_factor(_product_operator(views, weights),
                             n_components, seed + index)
        if factor is not None:
            factors.append(factor)
    stacked = np.hstack(factors) / np.sqrt(max(len(factors), 1))
    u, s, _ = np.linalg.svd(stacked, full_matrices=False)
    return (u[:, :n_components] * s[:n_components]).astype(np.float32)


def view_gates(x: np.ndarray, transitions: list) -> np.ndarray:
    """E1 heuristic gates: cosine(node features, view-v neighbour average),
    clipped at 0, normalised per node (uniform fallback). Shape (V, n)."""
    norms = np.linalg.norm(x, axis=1) + 1e-12
    gates = []
    for p in transitions:
        m = p @ x
        cos = (m * x).sum(axis=1) / ((np.linalg.norm(m, axis=1) + 1e-12) * norms)
        gates.append(np.clip(cos, 0.0, None))
    gates = np.asarray(gates)
    total = gates.sum(axis=0)
    dead = total < 1e-12
    gates[:, dead] = 1.0 / len(transitions)
    gates[:, ~dead] /= total[~dead]
    return gates


def _gated_operator(views: list, gates: np.ndarray,
                    weights: np.ndarray) -> LinearOperator:
    """W = S_T ... S_1, S_i = sum_v diag(w_iv) P_v with per-node weights
    w_iv proportional to alpha_iv * gate_v, renormalised per node so each
    step stays row-stochastic."""
    steps = []
    for row in weights:
        w = gates * row[:, None]                       # (V, n)
        w /= w.sum(axis=0, keepdims=True) + 1e-12
        steps.append([(w[v], views[v]) for v in range(len(views))])

    def matvec(x):
        for step in steps:
            x = sum((p @ x) * (w[:, None] if x.ndim > 1 else w)
                    for w, p in step)
        return x

    def rmatvec(x):
        for step in reversed(steps):
            x = sum(p.T @ ((w[:, None] if x.ndim > 1 else w) * x)
                    for w, p in step)
        return x

    n = views[0].shape[0]
    return LinearOperator((n, n), matvec=matvec, rmatvec=rmatvec,
                          dtype=np.float64)


def mdt_gated_consensus(views: list, gates: np.ndarray, n_components: int,
                        seed: int = 0) -> np.ndarray:
    factors = []
    for index in range(CONSENSUS_TRAJECTORIES):
        weights = trajectory(len(views), STEPS, seed * 1000 + index)
        factor = _svd_factor(_gated_operator(views, gates, weights),
                             n_components, seed + index)
        if factor is not None:
            factors.append(factor)
    if not factors:
        return None
    stacked = np.hstack(factors) / np.sqrt(len(factors))
    u, s, _ = np.linalg.svd(stacked, full_matrices=False)
    return (u[:, :n_components] * s[:n_components]).astype(np.float32)


def two_hop_view(a: sp.csr_matrix, cap: int = 64) -> sp.csr_matrix:
    """E3 derived view: binarised A^2, diagonal removed, rows capped."""
    a2 = (a @ a).tocsr()
    a2.data = np.ones_like(a2.data)
    a2.setdiag(0)
    a2.eliminate_zeros()
    return _cap_neighbors(a2, cap)


def mdt_select(views: list, n_components: int, k: int,
               scorer: sp.csr_matrix, seed: int = 0,
               count: int = 16, top: int = 1) -> np.ndarray:
    """Best-scoring trajectory (top=1) or mean-Gram consensus of the top `top`.

    top>1 answers "is consensus's win just averaging, or trajectory quality?":
    the first CONSENSUS_TRAJECTORIES random candidates are exactly the paths
    mdt_consensus averages, so top=8 is mdt_consensus with the random draw
    replaced by criterion-ranked selection from the same pool.
    """
    candidates = [trajectory(len(views), STEPS, seed * 1000 + i)
                  for i in range(count)]
    for view in range(len(views)):
        one_hot = np.zeros((STEPS, len(views)))
        one_hot[:, view] = 1.0
        candidates.append(one_hot)
    scored = []
    for index, weights in enumerate(candidates):
        factor = _svd_factor(_product_operator(views, weights), n_components,
                             seed + index)
        if factor is None:
            continue
        scored.append((modularity_criterion(factor, k, scorer, seed), factor))
    if not scored:
        return None
    scored.sort(key=lambda pair: -pair[0])
    if top == 1:
        return scored[0][1]
    kept = [factor for _, factor in scored[:top]]
    stacked = np.hstack(kept) / np.sqrt(len(kept))
    u, s, _ = np.linalg.svd(stacked, full_matrices=False)
    return (u[:, :n_components] * s[:n_components]).astype(np.float32)


# --- GNN grid --------------------------------------------------------------

def _sym_normalize(a: sp.csr_matrix) -> sp.csr_matrix:
    a = (a + sp.eye(a.shape[0], format="csr")).tocsr()
    d = np.asarray(a.sum(axis=1)).ravel()
    inv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    return (sp.diags(inv) @ a @ sp.diags(inv)).tocsr()


def _cap_neighbors(a: sp.csr_matrix, cap: int = 64) -> sp.csr_matrix:
    """Keep each row's top-`cap` weights (SAGE-style neighbour sampling,
    label-free). Full-batch message passing on 700+-degree metapath graphs
    is 10-100x slower for no measured benefit."""
    a = a.tocsr()
    if a.nnz <= cap * a.shape[0]:
        return a
    rows, cols, data = [], [], []
    for i in range(a.shape[0]):
        lo, hi = a.indptr[i], a.indptr[i + 1]
        idx = a.indices[lo:hi]
        val = a.data[lo:hi]
        if len(val) > cap:
            top = np.argpartition(val, -cap)[-cap:]
            idx, val = idx[top], val[top]
        rows.extend([i] * len(idx))
        cols.extend(idx)
        data.extend(val)
    capped = sp.csr_matrix((data, (rows, cols)), shape=a.shape)
    return capped.maximum(capped.T)


def fused_adjacency(graphs: dict) -> sp.csr_matrix:
    parts = [_sym_normalize(a) for _, a in sorted(graphs.items())]
    return _cap_neighbors(sum(parts).tocsr() * (1.0 / len(parts)))


def _torch_sparse(a: sp.csr_matrix) -> torch.Tensor:
    coo = a.tocoo()
    idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(coo.data,
                                                     dtype=torch.float32),
                                   coo.shape).coalesce()


class Encoder(torch.nn.Module):
    """2-layer GCN / SAGE / MLP over one fused adjacency."""

    def __init__(self, d_in: int, hidden: int, d_out: int, backbone: str):
        super().__init__()
        self.backbone = backbone
        if backbone == "sage":
            self.w1 = torch.nn.Linear(2 * d_in, hidden)
            self.w2 = torch.nn.Linear(2 * hidden, d_out)
        else:  # gcn or mlp
            self.w1 = torch.nn.Linear(d_in, hidden)
            self.w2 = torch.nn.Linear(hidden, d_out)
        self.norm = torch.nn.BatchNorm1d(hidden)  # prevents BGRL collapse

    def _layer(self, linear, x, adj):
        if self.backbone == "mlp":
            return linear(x)
        if self.backbone == "sage":
            return linear(torch.cat([x, torch.sparse.mm(adj, x)], dim=1))
        return linear(torch.sparse.mm(adj, x))

    def forward(self, x, adj):
        h = self.norm(torch.relu(self._layer(self.w1, x, adj)))
        return self._layer(self.w2, h, adj)


def gae_loss(z: torch.Tensor, edges: torch.Tensor,
             generator: torch.Generator) -> torch.Tensor:
    src, dst = edges
    negative = torch.randint(0, z.shape[0], (2, src.numel()),
                             generator=generator)
    pos = (z[src] * z[dst]).sum(dim=1)
    neg = (z[negative[0]] * z[negative[1]]).sum(dim=1)
    logits = torch.cat([pos, neg])
    target = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, target)


def _augment(x, adj_sp, drop_edge, mask_feat, generator):
    keep = torch.rand(adj_sp._nnz(), generator=generator) > drop_edge
    idx, val = adj_sp.indices()[:, keep], adj_sp.values()[keep]
    adj = torch.sparse_coo_tensor(idx, val, adj_sp.shape).coalesce()
    mask = (torch.rand(1, x.shape[1], generator=generator) > mask_feat).float()
    return x * mask, adj


def train_gnn(x: np.ndarray, adj: sp.csr_matrix, objective: str,
              backbone: str, dim: int, k: int, seed: int,
              scorer: sp.csr_matrix = None,
              hidden: int = 128, epochs: int = 150,
              lr: float = 0.003, check_every: int = 15) -> np.ndarray:
    """Train one grid cell, return the best-silhouette checkpoint embedding."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    xt = torch.tensor(x, dtype=torch.float32)
    at = _torch_sparse(adj)
    model = Encoder(x.shape[1], hidden, dim, backbone)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    if objective == "bgrl":
        import copy
        target = copy.deepcopy(model)
        for p in target.parameters():
            p.requires_grad_(False)
        predictor = torch.nn.Sequential(torch.nn.Linear(dim, hidden),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hidden, dim))
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=lr, weight_decay=1e-4)
    else:
        coo = sp.triu(adj, k=1).tocoo()
        edges = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)

    best, best_score = None, -np.inf
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        if objective == "gae":
            loss = gae_loss(model(xt, at), edges, generator)
        else:  # bgrl
            x1, a1 = _augment(xt, at, .3, .3, generator)
            x2, a2 = _augment(xt, at, .3, .3, generator)
            z1, z2 = model(x1, a1), model(x2, a2)
            with torch.no_grad():
                t1, t2 = target(x1, a1), target(x2, a2)
            cos = torch.nn.functional.cosine_similarity
            loss = (2 - cos(predictor(z1), t2.detach()).mean()
                      - cos(predictor(z2), t1.detach()).mean())
        loss.backward()
        optimizer.step()
        if objective == "bgrl":
            with torch.no_grad():
                for p_t, p_o in zip(target.parameters(), model.parameters()):
                    p_t.mul_(0.99).add_(p_o, alpha=0.01)
        if epoch % check_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                z = model(xt, at).numpy()
            score = (modularity_criterion(z, k, scorer, seed)
                     if scorer is not None else silhouette_criterion(z, k, seed))
            if score > best_score:
                best, best_score = z, score
    return best


def train_mlp_ssl(x: np.ndarray, dim: int, k: int, seed: int, **kw) -> np.ndarray:
    """Feature ceiling: BGRL objective with feature masking only, no graph."""
    empty = sp.csr_matrix((len(x), len(x)), dtype=np.float32)
    return train_gnn(x, empty, "bgrl", "mlp", dim, k, seed, **kw)


def distill_mlp(x: np.ndarray, teacher_z: np.ndarray, k: int, seed: int,
                hidden: int = 128, epochs: int = 300,
                lr: float = 0.003) -> np.ndarray:
    torch.manual_seed(seed)
    xt = torch.tensor(x, dtype=torch.float32)
    zt = torch.tensor(teacher_z, dtype=torch.float32)
    model = Encoder(x.shape[1], hidden, teacher_z.shape[1], "mlp")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(xt, None), zt)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return model(xt, None).numpy()
