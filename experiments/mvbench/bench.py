"""MDT against the multi-view diffusion literature, out of sample.

The MDT paper (arXiv:2512.01484, Tab. 4) compares MDT to AD, p-AD, ID, MVD,
CR-DIFF, COM-DIFF, GCCA and MVSC -- all *transductive*.  Every one of them
embeds the n points it was built from and has no story for an unseen point.
This benchmark asks the question the paper cannot: given a train/test split,
which multi-view operator still separates the classes on points that were
never in the operator?

Controls, all deliberately identical across methods:

* one kernel.  Per-view Gaussian kNN transitions from
  ``closure.build_graphs`` (median bandwidth on <=200 points, kNN
  sparsification, symmetrise, row-normalise).  This is *not* the paper's
  kernel (it uses ``sigma = max_i min_j dist`` and ``exp(-d^2/sigma)``), so
  absolute AMI here will not reproduce Tab. 4.  It is the kernel every prior
  MDT/GNN result in this repo used, which is what makes the numbers
  comparable to ``results/gnn_mdt``.
* one embedding.  Truncated SVD of the operator, top singular vector dropped.
* one out-of-sample rule.  Nystrom.  It needs each operator written as

      W = sum_v P_v S_v          =>       W* = sum_v B_v S_v

  with ``B_v`` the test->train bipartite transition of view v.  So every
  builder returns ``(W, {v: S_v})`` and the runner does the substitution.
  This is ``closure.mdt_oos_operator`` generalised, and it is exact: no
  method gets a different extension, so a win is a win about the operator.
* one clustering.  ``closure.cluster_metrics``: KMeans on the train
  embedding, reported three ways.  ``inductive_ami`` (train-fitted KMeans
  applied to the test embedding) is the primary OOD number -- it fails unless
  the extension lands test points in the *same* coordinate frame.
  ``test_ami`` (KMeans refit on the test embedding) is the weaker question of
  whether the extension has any cluster structure at all.

Read the results next to ``ENCOMPASSED``.  Paper Sec 3.5 proves AD and ID are
MDT trajectories, so "MDT beats AD" is partly MDT beating itself with a worse
trajectory.  The methods genuinely outside the framework are CR-DIFF and
COM-DIFF (they use transposes, so they are not products of row-stochastic
matrices), MVD (its operator is Vn x Vn) and GCCA (not a diffusion at all).

Provenance: ``mdt_cvx_rand`` here and ``mdt_nystrom`` in ``closure`` draw the
same trajectory from the same seed and apply the same Nystrom rule, but embed
with dense ``np.linalg.svd`` and sparse ``svds`` respectively.  Measured over
the six configured datasets x 2 seeds that gives corr 0.983, mean difference
-0.006 inductive AMI, and a worst cell of 0.13 (Wikipedia, where AMI is low
enough that the backends disagree).  So the two tables are comparable in
aggregate and must NOT be joined row by row.

Usage
-----
    python -m experiments.mvbench.bench --smoke                 # self-check
    python -m experiments.mvbench.bench -c experiments/mvbench/config.yml
    python -m experiments.mvbench.bench --summarize results/mvbench/metrics.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.sparse as sp
import torch
from kneed import KneeLocator
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence
from scipy.stats import t as student_t

from experiments.gnn_mdt.closure import (
    Split, build_graphs, cluster_metrics, load_config, load_split,
    make_synthetic, paired_effect, seed_everything, trajectory,
)
from experiments.mvbench.paper_datasets import (
    AVAILABLE as PAPER_DATASETS, load_paper_split,
)

# Normalisation inside the singular-entropy elbow that selects the diffusion
# time for ID and p-AD.  Paper Eq. 18 calls it "the Shannon entropy of the
# normalised singular value distribution", i.e. sum(s) == 1 -> "l1".  The
# reference implementation divides by the L2 norm instead, so the vector is not
# a probability and the elbow moves: on the paper's own shipped datasets the
# selected t differs on 4/4 (MSRC 12 vs 1, Yale 8 vs 10, 100Leaves 5 vs 7,
# Caltech101-7 10 vs 17 for l2 vs l1), and 12/12 on synthetic operators.
# "l2" is the default here on purpose -- ID and p-AD should be represented the
# way their diffusion time was actually computed when the paper was run, not
# the way it was written down.  Flip to "l1" for the stated formula.
SPECTRUM_NORM = "l2"

# What Q_CH is scored on, for MDT-DIRECT and MDT-BSC.  Paper Sec. 4.1 writes
# CH(X_v, C) with X_v the data view; the reference implementation passes
# X_preprocessed, i.e. the n x n row-stochastic operators.  Default is
# "operators" -- the code, not the equation -- for the same reason
# SPECTRUM_NORM is "l2": these variants should be represented the way the
# published numbers were actually produced.
#
# This is NOT cosmetic and picking "views" was a measurement error on our part.
# Measured over 19 datasets x 5 seeds, moving from views to operators lifts
# MDT-DIRECT 0.356 -> 0.415 (+0.059, 14/19) and MDT-BSC 0.330 -> 0.379
# (+0.049, 16/19, p=0.004).  Under "views" the searched variants lose to a plain
# random convex draw; under "operators" MDT-DIRECT beats it (+0.034, 14/19).
# The whole "trajectory search buys nothing" reading was an artifact of scoring
# the criterion on the wrong object.  Kept switchable so the divergence between
# the paper and its code stays measurable rather than assumed.
CH_TARGET = "operators"

# Which kernel every diffusion arm is built from.  "quantile" is this repo's
# (median bandwidth, exp(-d^2/2s^2), standardised views); "reference" is the MDT
# repo's get_kernel_matrix (max-min-nn bandwidth, exp(-d^2/bw), raw views), i.e.
# what produced the published Tab. 4.  They are not a rescale of one another:
# per-view exponent denominators differ by 0.6x to 43000x, so which views look
# informative changes.  Run both -- if the operator ordering survives the swap,
# that is worth more than either column alone.
KERNEL = "quantile"

# The paper's robustness parameter s (Fig. 7), used only by K-/L-MvMNIST.  This
# axis is the one that tests MDT's actual thesis: in K-MvMNIST only the second
# view degrades, so a method that can down-weight a bad view should decay more
# slowly than one that treats views equally.  It is also the only axis on which
# SpecRaGE's and GRAB-MDM's stated claims (noise robustness) can be tested at
# all -- the clean .mat datasets cannot.
NOISE_FACTOR = 0.5


# --- shared numerics -------------------------------------------------------

def _dense(graphs: Sequence[sp.spmatrix]) -> list[np.ndarray]:
    return [np.asarray(g.todense(), dtype=np.float64) for g in graphs]


def _chain(matrices: Sequence[np.ndarray], n: int) -> np.ndarray:
    """Left-to-right product, identity when empty."""
    out = np.eye(n)
    for m in matrices:
        out = out @ m
    return out


def _spectral_entropy(values: np.ndarray) -> float:
    values = np.abs(values[values != 0]).astype(np.float64)
    if not len(values):
        return 0.0
    values = values / (values.sum() if SPECTRUM_NORM == "l1"
                       else np.linalg.norm(values))
    return float(-np.sum(values * np.log(values)))


def _elbow_power(matrix: np.ndarray, max_power: int, spectral: bool) -> int:
    """Diffusion time from the elbow of the entropy decay (paper Sec. 4.2).

    ``spectral=True`` reads the eigenvalues, which satisfy
    ``eig(A^t) = eig(A)^t``, so one decomposition covers every power -- this is
    what ID does in the reference repo.  ``spectral=False`` reads singular
    values, which do *not* power for an asymmetric A, so p-AD really does need
    one SVD per candidate power.
    """
    powers = list(range(1, max_power + 1))
    if spectral:
        values = np.linalg.eigvals(matrix)
        curve = [_spectral_entropy(np.power(values, t)) for t in powers]
    else:
        curve, running = [], np.eye(len(matrix))
        for _ in powers:
            running = running @ matrix
            curve.append(_spectral_entropy(np.linalg.svd(running,
                                                         compute_uv=False)))
    knee = KneeLocator(powers, curve, curve="convex",
                       direction="decreasing").knee
    return int(knee) if knee else 1


# --- operators.  each returns (W, {view: suffix}) with W = sum_v P_v suffix_v

def ad(p: list[np.ndarray], **_) -> tuple:
    """Alternating Diffusion, Lederman-Talmon 2019 / Katz 2019.  P_1 ... P_V.

    An MDT trajectory: the length-V one-hot path that visits each view once.
    """
    suffix = _chain(p[1:], len(p[0]))
    return p[0] @ suffix, {0: suffix}


def powered_ad(p: list[np.ndarray], max_power: int = 25, **_) -> tuple:
    """p-AD, Kuchroo 2022.  AD raised to its own entropy-elbow power."""
    base = _chain(p, len(p[0]))
    power = _elbow_power(base, max_power, spectral=False)
    tail = _chain(p[1:], len(p[0])) @ np.linalg.matrix_power(base, power - 1)
    return np.linalg.matrix_power(base, power), {0: tail}


def integrated(p: list[np.ndarray], max_power: int = 50, **_) -> tuple:
    """Integrated Diffusion, Kuchroo 2022.  Denoise each view to its own
    elbow power, then alternate.  Also an MDT trajectory (paper Sec. 3.5)."""
    n = len(p[0])
    powers = [max(1, _elbow_power(view, max_power, spectral=True)) for view in p]
    parts = [np.linalg.matrix_power(view, t) for view, t in zip(p, powers)]
    rest = _chain(parts[1:], n)
    suffix = np.linalg.matrix_power(p[0], powers[0] - 1) @ rest
    return parts[0] @ rest, {0: suffix}


def composite(p: list[np.ndarray], **_) -> tuple:
    """COM-DIFF, Shnitzer 2018.  P_1 P_2^T + P_2 P_1^T.  Two views only, and
    outside MDT: the transpose is not row-stochastic."""
    if len(p) != 2:
        return None
    return p[0] @ p[1].T + p[1] @ p[0].T, {0: p[1].T, 1: p[0].T}


def cross(p: list[np.ndarray], iterations: int = 25, **_) -> tuple:
    """CR-DIFF, Wang 2012.  Q_v <- P_v (mean_{u != v} Q_u) P_v^T, iterated,
    then averaged.  Outside MDT (transposes, and backward operators).

    The suffix is the *last* iteration's inner factor, so the out-of-sample
    row is the same word in P with only the leftmost factor swapped.
    """
    views = len(p)
    if views < 2:
        return None
    state = [view.copy() for view in p]
    others = None
    for _ in range(iterations):
        total = sum(state)
        others = [(total - q) / (views - 1) for q in state]
        state = [np.clip(p[v] @ others[v] @ p[v].T, -1e6, 1e6)
                 for v in range(views)]
    return (sum(state) / views,
            {v: others[v] @ p[v].T / views for v in range(views)})


def uniform_fused(p: list[np.ndarray], steps: int = 4, **_) -> tuple:
    """No-trajectory ablation: the uniform mean operator, t steps.  MDT's
    whole claim is that *which* views, in *which* order, matters; if this ties
    MDT then the trajectory machinery bought nothing on this dataset."""
    mean = sum(p) / len(p)
    tail = np.linalg.matrix_power(mean, steps - 1) / len(p)
    return np.linalg.matrix_power(mean, steps), {v: tail for v in range(len(p))}


def _from_weights(p: list[np.ndarray], weights: np.ndarray) -> tuple:
    """MDT operator for an explicit weight trajectory, plus its suffix."""
    prefix = _chain([sum(float(a) * view for a, view in zip(row, p))
                     for row in weights[:-1]], len(p[0]))
    last = sum(float(a) * view for a, view in zip(weights[-1], p))
    return last @ prefix, {v: float(weights[-1][v]) * prefix
                           for v in range(len(p))}


def mdt_convex(p: list[np.ndarray], steps: int = 4, seed: int = 0, **_) -> tuple:
    """MDT-CVX-RAND (paper Def. 5/6): convex weights per step, drawn once.

    The draw is Dirichlet(1) via ``closure.trajectory``, which is what every
    earlier MDT run in this repo used.  The reference implementation instead
    normalises i.i.d. uniforms (``random_mdt``, ``distribution="pseudo-uniform"``),
    which concentrates near the simplex centre -- i.e. nearer to
    ``uniform_fused`` than a Dirichlet draw is.  Dirichlet is the harder,
    more diverse baseline; ``uniform_fused`` brackets the other end.
    """
    return _from_weights(p, trajectory(len(p), steps, seed))


def mdt_discrete(p: list[np.ndarray], steps: int = 4, seed: int = 0, **_) -> tuple:
    """MDT-RAND: one view per step, drawn uniformly (paper's PRR baseline).

    ``W = P_{i_t} ... P_{i_1}``, so the leftmost factor is the *last* pick.
    """
    picks = np.random.default_rng(seed).integers(0, len(p), steps)
    prefix = _chain([p[i] for i in picks[:-1][::-1]], len(p[0]))
    return p[picks[-1]] @ prefix, {int(picks[-1]): prefix}


def mdt_selected(p: list[np.ndarray], steps: int = 4, seed: int = 0,
                 components: int = 2, pool: int = 24, **_) -> tuple:
    """MDT at its best under a label-free criterion: rank a sampled pool of
    trajectories by silhouette of the train embedding and keep the winner.

    This is the arm that makes the table fair to MDT.  MDT-RAND/CVX-RAND are
    single draws, so a table without a *searched* variant compares the
    literature's tuned operators to an untuned MDT.  Silhouette rather than
    the paper's Calinski-Harabasz: on ten multi-view datasets CH ranked
    trajectories at Spearman rho ~0.25 against oracle AMI and silhouette at
    ~0.48 (``docs/GRAPH_MDT_RESEARCH_LEDGER.md``), and ranking a sampled pool beats
    beam search, which prunes good paths.  Same criterion, same pool logic as
    ``src.mdt_operators.select_trajectory``; reimplemented here only because
    that helper returns no suffix for the Nystrom step.
    """
    views, n = len(p), len(p[0])
    rng = np.random.default_rng(seed)
    candidates = [np.eye(views)[[v] * steps] for v in range(views)]  # stay in one view
    candidates += [trajectory(views, steps, seed * 1000 + i) for i in range(pool)]
    candidates += [np.eye(views)[rng.integers(0, views, steps)] for _ in range(pool)]

    best, best_score = None, -np.inf
    for weights in candidates:
        operator, suffix = _from_weights(p, weights)
        u, values, _ = np.linalg.svd(operator, full_matrices=False)
        embedding = u[:, 1:components + 1] * values[1:components + 1]
        if not np.all(np.isfinite(embedding)):
            continue
        score = silhouette_of(embedding, seed, components)
        if score > best_score:
            best_score, best = score, weights
    if best is None:                      # nothing clustered; fall back to a draw
        best = trajectory(views, steps, seed)
    return _from_weights(p, best)


def calinski_of(embedding: np.ndarray, views: list, seed: int,
                clusters: int) -> float:
    """The paper's internal quality index Q_CH: cluster the trajectory's
    embedding, then average Calinski-Harabasz over the data views.

    Paper Sec. 4.1 writes ``Q_CH(tau) = sum_v w_v CH(X_v, C_v(tau))`` with
    ``w_v = 1/V`` and ``X_v`` the data view.  Note the reference implementation
    passes ``X_preprocessed`` here, i.e. the n x n row-stochastic operators, so
    its CH is taken over diffusion-operator rows rather than the data.  The
    paper's formula is used here; that is a deliberate divergence from the
    reference code, unlike ``SPECTRUM_NORM`` where the code's behaviour is what
    produced the published diffusion times.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score
    predicted = KMeans(clusters, n_init=4, random_state=seed).fit_predict(embedding)
    if len(np.unique(predicted)) < 2:
        return -np.inf
    return float(np.mean([calinski_harabasz_score(view, predicted) for view in views]))


def _score_weights(p: list, weights: np.ndarray, views: list, components: int,
                   seed: int) -> float:
    """Q_CH of the trajectory `weights`, label-free."""
    operator, _ = _from_weights(p, weights)
    u, values, _ = np.linalg.svd(operator, full_matrices=False)
    embedding = u[:, 1:components + 1] * values[1:components + 1]
    if not np.all(np.isfinite(embedding)):
        return -np.inf
    return calinski_of(embedding, views, seed, components)


def mdt_cst(p: list[np.ndarray], steps: int = 4, seed: int = 0, views: list = None,
            **_) -> tuple:
    """MDT-CST (paper Tab. 2): convex weights learned by ADAM on the
    contrastive index Q_X (Eq. 19/20).

    The paper scopes this variant to manifold learning rather than clustering,
    so it is the one MDT arm here that is being asked to do a job its authors
    did not claim for it -- read it as a check on whether the contrastive index
    transfers, not as a broken promise.  Reuses the loss vendored in
    ``src.mdt_operators._mdt_contrastive``.

    Note the paper's Eq. 17 says ``argmax Q`` while Eq. 19 defines Q_X as a sum
    of negative log-probabilities, i.e. a loss; the reference code minimises it,
    and so does this.
    """
    from src.mdt_operators import _mdt_contrastive
    weights = _mdt_contrastive(p, steps, seed=seed, return_weights=True)
    return _from_weights(p, weights)


def mdt_direct(p: list[np.ndarray], steps: int = 4, seed: int = 0,
               components: int = 2, views: list = None, budget: int = 100,
               **_) -> tuple:
    """MDT-DIRECT (paper Tab. 2) -- the paper's best variant, top mean AMI on 5
    of the 9 datasets in its Tab. 4.  Without this arm, any statement about
    "MDT" in an out-of-sample table is really about random trajectories.

    DIRECT (Jones 2001, the paper's ref. [18]) over the convex weights, guided
    by Q_CH.  ``scipy.optimize.direct`` is that same algorithm, so the reference
    repo's ``gob`` dependency is not needed.  Bounds ``[0, 20]^(V*t)`` and
    ``budget=100`` function evaluations match the reference implementation, and
    the parameters are softmaxed per step, also as in the reference.
    """
    from scipy.optimize import direct, Bounds
    n_views = len(p)

    def to_weights(vector: np.ndarray) -> np.ndarray:
        logits = np.asarray(vector, dtype=np.float64).reshape(steps, n_views)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)

    def objective(vector: np.ndarray) -> float:
        score = _score_weights(p, to_weights(vector), views, components, seed)
        return -score if np.isfinite(score) else 1e12

    result = direct(objective, Bounds([0.0] * (steps * n_views),
                                      [20.0] * (steps * n_views)),
                    maxfun=budget, maxiter=budget)
    return _from_weights(p, to_weights(result.x))


def mdt_bsc(p: list[np.ndarray], steps: int = 4, seed: int = 0,
            components: int = 2, views: list = None, width: int = 2,
            max_depth: int = 12, **_) -> tuple:
    """MDT-BSC (paper Tab. 2): beam search over the *discrete* tree, Q_CH-scored.

    Depth is ``2 x`` the singular-entropy elbow of the expected operator, as in
    the reference implementation, capped at ``max_depth`` for cost.  Beam width
    2, also from the reference.

    The product is grown in paper order, ``W = W_d ... W_1`` (Def. 1), i.e. new
    factors multiply on the left.  The reference implementation instead does
    ``parent.path_operator @ op``, which grows on the right and therefore beam-
    prunes on what is a trajectory *suffix* in the paper's indexing.  The set of
    reachable operators is identical (path reversal is a bijection); the greedy
    prefix is not.
    """
    n_views, n = len(p), len(p[0])
    mean = sum(p) / n_views
    depth = min(max_depth, max(1, 2 * _elbow_power(mean, 25, spectral=True)))

    # Faithful to _mdt_tree_utils.BeamSearch.search: expand `depth` times, keep
    # the top `width` candidates at each level, and return the best node of the
    # FINAL beam -- so the returned trajectory always has length `depth`.  An
    # earlier version here tracked the best score across all depths and could
    # return a much shorter trajectory; that is a different algorithm and it
    # collapsed on 2-view data (0.128 AMI on L-MvMNIST vs ~0.3 for its peers).
    beam = [[]]                                   # each entry is a list of view ids
    for _ in range(depth):
        scored = []
        for path in beam:
            for view in range(n_views):
                candidate = path + [view]
                scored.append((_score_weights(p, np.eye(n_views)[candidate],
                                              views, components, seed), candidate))
        if not scored:
            break
        scored.sort(key=lambda pair: -pair[0])
        beam = [candidate for _, candidate in scored[:width]]
        best_path = beam[0]                       # best of the current level
    if not beam or not beam[0]:
        return _from_weights(p, trajectory(n_views, steps, seed))
    return _from_weights(p, np.eye(n_views)[beam[0]])


def silhouette_of(embedding: np.ndarray, seed: int, clusters: int = None) -> float:
    """Label-free trajectory score: silhouette of KMeans on the embedding."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    k = clusters or min(embedding.shape[1], len(embedding) - 1)
    if k < 2:
        return -1.0
    predicted = KMeans(k, n_init=4, random_state=seed).fit_predict(embedding)
    if len(np.unique(predicted)) < 2:
        return -1.0
    return float(silhouette_score(embedding, predicted))


OPERATORS = {
    "mdt_cvx_rand": mdt_convex,
    "mdt_rand": mdt_discrete,
    "mdt_selected": mdt_selected,
    "mdt_cst": mdt_cst,
    "mdt_direct": mdt_direct,
    "mdt_bsc": mdt_bsc,
    "ad": ad,
    "p_ad": powered_ad,
    "id": integrated,
    "com_diff": composite,
    "cr_diff": cross,
    "uniform_fused": uniform_fused,
}

# Paper Sec. 3.5: which compared methods the MDT operator space contains.
# A win over an encompassed method is a statement about trajectory choice,
# not about the framework.
ENCOMPASSED = {"mdt_cvx_rand": True, "mdt_rand": True, "mdt_selected": True,
               "mdt_cst": True, "mdt_direct": True, "mdt_bsc": True,
               "ad": True,
               "p_ad": True, "id": True, "uniform_fused": True,
               "com_diff": False, "cr_diff": False, "mvd": False,
               "gcca": False, "dgcca": False, "specrage": False,
               "features_matched": None}


# --- methods that do not fit the (W, suffix) shape -------------------------

def mvd(kernels: list[np.ndarray], oos_kernels: list[np.ndarray],
        components: int, seed: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Multi-View Diffusion Maps, Lindenbaum 2020.  Block cross-kernel
    operator on Vn nodes: block (i, j) = K_i K_j, zero diagonal, then row
    normalised.  The paper reports the view-1 block; so do we.

    This is the only compared method the MDT framework explicitly does *not*
    encompass (paper Sec. 3.5: composite operators larger than n x n).  Its
    out-of-sample row is the same block row with K_1 -> K*_1.
    """
    views, n = len(kernels), len(kernels[0])
    if views < 2 or views * n > 6000:      # ponytail: dense Vn x Vn eigsh
        return None                         # raise the cap if you have the RAM
    block = np.zeros((views * n, views * n))
    for i in range(views):
        for j in range(i + 1, views):
            cross_block = kernels[i] @ kernels[j]
            block[i * n:(i + 1) * n, j * n:(j + 1) * n] = cross_block
            block[j * n:(j + 1) * n, i * n:(i + 1) * n] = cross_block.T
    rows = block.sum(axis=1, keepdims=True)
    operator = block / np.where(rows == 0, 1.0, rows)

    rank = min(components + 1, views * n - 1)
    symmetric = 0.5 * (operator + operator.T)
    try:
        values, vectors = eigsh(symmetric, k=rank, which="LA", maxiter=5000)
    except (ArpackNoConvergence, ArpackError, RuntimeError):
        # The reference implementation catches exactly this and falls back to a
        # dense eigh (utilities/evaluate.py::get_embedding).  ARPACK stalls when
        # the block operator is near-degenerate, which the reference kernel makes
        # common: identity views give repeated unit eigenvalues.
        values, vectors = np.linalg.eigh(symmetric)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    kept = slice(1, components + 1)
    train = (vectors[:n, kept] * values[kept]).astype(np.float32)

    # test block row: only view-1 rows are needed, so only K*_1 is substituted
    test_row = np.zeros((len(oos_kernels[0]), views * n))
    for j in range(1, views):
        test_row[:, j * n:(j + 1) * n] = oos_kernels[0] @ kernels[j]
    rows = test_row.sum(axis=1, keepdims=True)
    test_row = test_row / np.where(rows == 0, 1.0, rows)
    return train, (test_row @ vectors[:, kept]).astype(np.float32)


def _gcca_views(train_views: list, test_views: list, components: int,
                fraction_var: float = .9) -> tuple[np.ndarray, np.ndarray]:
    """Generalised CCA (Horst 1961), SVD form, on arbitrary view matrices.

    Per view: whiten to a rank cut, stack the whitened scores, take the top-k
    left singular vectors of the stack as the common subspace, and read off
    per-view loadings to project unseen points.

    ``fraction_var`` is load-bearing, not a knob to ignore.  At full numerical
    rank a view with d >= n whitens to a basis of the whole sample space, every
    view's whitened block spans the same space, and the common subspace is
    arbitrary -- GCCA scored ~0.03 AMI on MSRC-v5 that way.  Keeping the
    components that carry 90% of the spectral energy is mvlearn's default and
    the reason its GCCA is a real baseline.
    """
    whitened, loadings = [], []
    for train_view in train_views:
        centred = train_view - train_view.mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(centred, full_matrices=False)
        energy = np.cumsum(s ** 2)
        rank = int(np.searchsorted(energy, fraction_var * energy[-1]) + 1) \
            if energy[-1] > 0 else 1
        rank = max(components, min(rank, len(s)))
        whitened.append(u[:, :rank])
        loadings.append((vt[:rank].T / np.maximum(s[:rank], 1e-12)))
    stacked = np.hstack(whitened)
    u, _, _ = np.linalg.svd(stacked, full_matrices=False)
    basis = u[:, :components]

    test = np.zeros((len(test_views[0]), components))
    for index, (train_view, test_view) in enumerate(zip(train_views, test_views)):
        centred = test_view - train_view.mean(axis=0, keepdims=True)
        scores = centred @ loadings[index]
        # least-squares map from this view's whitened scores to the common basis
        test += scores @ np.linalg.lstsq(whitened[index], basis, rcond=None)[0]
    return (basis.astype(np.float32),
            (test / len(train_views)).astype(np.float32))


def gcca(split: Split, components: int, **_) -> tuple[np.ndarray, np.ndarray]:
    """GCCA on the raw views -- the paper's non-diffusion competitor, and the
    only one that is inductive by construction: each view gets a linear map, so
    a test point needs no extension rule at all.

    Note the reference repo hands GCCA ``X_preprocessed``, i.e. the n x n
    transition matrices, not the data views.  That is why its GCCA looks weak.
    """
    return _gcca_views(split.train, split.test, components)


def dgcca(split: Split, components: int, seed: int = 0, hidden: int = 256,
          epochs: int = 150, lr: float = 1e-3, reg: float = 1e-3,
          **_) -> tuple[np.ndarray, np.ndarray]:
    """Deep GCCA (Benton et al. 2017) -- the deep + alignment + inductive
    corner of the design space, completing the grid next to GCCA (linear,
    alignment) and SpecRaGE (deep, spectral).  Worth running because plain
    linear GCCA already ties MDT here: if going nonlinear on the *inductive*
    side pulls ahead, the diffusion operator is not where the value is.

    One MLP per view, trained to maximise the sum of the top-k eigenvalues of
    ``M = sum_v P_v``, with ``P_v`` the projection onto view v's encoded column
    space.  The eigenvectors are recomputed under ``no_grad`` each step and the
    objective is ``tr(U^T M U)`` -- that is exactly Benton's gradient, and it
    keeps ``eigh`` out of the backward pass, which is where it is unstable.

    The common basis and the out-of-sample map are then the *same* linear GCCA
    step applied to the encoder outputs, so the extension rule is shared with
    the ``gcca`` arm and only the representation differs.
    """
    torch.manual_seed(seed)
    train = [torch.tensor(v, dtype=torch.float32) for v in split.train]
    test = [torch.tensor(v, dtype=torch.float32) for v in split.test]
    encoders = torch.nn.ModuleList(
        torch.nn.Sequential(torch.nn.Linear(v.shape[1], hidden), torch.nn.ReLU(),
                            torch.nn.Linear(hidden, max(components, 2)))
        for v in train)
    optimiser = torch.optim.Adam(encoders.parameters(), lr=lr, weight_decay=1e-5)
    rows = len(split.y_train)
    identity = torch.eye(max(components, 2))

    for _ in range(epochs):
        optimiser.zero_grad()
        total = torch.zeros(rows, rows)
        for encoder, view in zip(encoders, train):
            h = encoder(view)
            h = h - h.mean(dim=0, keepdim=True)
            covariance = h.T @ h / max(rows - 1, 1) + reg * identity
            total = total + h @ torch.linalg.solve(covariance, h.T) / max(rows - 1, 1)
        with torch.no_grad():
            _, vectors = torch.linalg.eigh(0.5 * (total + total.T))
            basis = vectors[:, -components:]
        loss = -torch.trace(basis.T @ total @ basis)
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        encoded_train = [encoder(view).numpy() for encoder, view in zip(encoders, train)]
        encoded_test = [encoder(view).numpy() for encoder, view in zip(encoders, test)]
    return _gcca_views(encoded_train, encoded_test, components)


# --- runner ----------------------------------------------------------------

def svd_extend(operator: np.ndarray, suffixes: dict, bipartite: list[np.ndarray],
               components: int) -> tuple[np.ndarray, np.ndarray]:
    """Truncated-SVD embedding plus its Nystrom extension."""
    u, values, vt = np.linalg.svd(operator, full_matrices=False)
    kept = slice(1, components + 1)
    train = (u[:, kept] * values[kept]).astype(np.float32)
    oos = sum(bipartite[v] @ suffix for v, suffix in suffixes.items())
    return train, np.asarray(oos @ vt.T[:, kept], dtype=np.float32)


def run_one(split: Split, cfg: dict, seed: int) -> list[dict]:
    clusters = len(np.unique(split.y_train))
    knn_cfg = cfg["mdt"].get("knn", "auto")
    knn = (int(np.floor(np.log(len(split.y_train)))) if knn_cfg == "auto"
           else int(knn_cfg))
    steps = int(cfg["mdt"]["steps"])

    graph_start = time.perf_counter()
    train_graphs, test_graphs = build_graphs(split, knn, recipe=KERNEL)
    graph_seconds = time.perf_counter() - graph_start
    p, bipartite = _dense(train_graphs), _dense(test_graphs)

    rows = []

    def record(name: str, train, test, seconds: float, graph: float = None):
        if train is None:
            print(f"  [skip] {name}", flush=True)
            return
        if not (np.all(np.isfinite(train)) and np.all(np.isfinite(test))):
            print(f"  [skip] {name}: non-finite embedding", flush=True)
            return
        metrics = cluster_metrics(train, test, split.y_train, split.y_test,
                                 clusters, seed)
        metrics.update({"dataset": split.name, "seed": seed, "method": name,
                        "ch_target": CH_TARGET, "kernel": KERNEL,
                        "noise_factor": NOISE_FACTOR,
                        # nominal is what was asked for; effective is what the
                        # cap_train/cap_test limits actually produced -- they
                        # differ (0.70 -> 0.67) on datasets above ~1140 rows.
                        "split_nominal": float(cfg["data"]["split"])
                        if "data" in cfg else None,
                        "split_effective": float(len(split.y_train))
                        / (len(split.y_train) + len(split.y_test)),
                        "graph_seconds": graph_seconds if graph is None else graph,
                        "train_seconds": seconds,
                        "encompassed_by_mdt": ENCOMPASSED.get(name)})
        rows.append(metrics)
        print(json.dumps(metrics), flush=True)

    for name, builder in OPERATORS.items():
        if name in cfg["evaluation"].get("skip", []):
            continue
        start = time.perf_counter()
        try:
            built = builder(p, steps=steps, seed=seed, components=clusters,
                            views=(split.train if CH_TARGET == "views" else p))
            if built is None:
                record(name, None, None, 0.0)
                continue
            operator, suffixes = built
            train, test = svd_extend(operator, suffixes, bipartite, clusters)
        except Exception as exc:                              # noqa: BLE001
            # A single arm failing numerically must not kill the suite: an
            # unhandled ArpackNoConvergence once aborted a 95-cell run on its
            # first cell.  Record it so the loss is visible, then continue.
            print(f"  [error] {name}: {type(exc).__name__}: {exc}", flush=True)
            rows.append({"dataset": split.name, "seed": seed, "method": name,
                         "kernel": KERNEL, "ch_target": CH_TARGET,
                         "error": f"{type(exc).__name__}: {exc}"})
            continue
        record(name, train, test, time.perf_counter() - start)

    if "mvd" not in cfg["evaluation"].get("skip", []):
        kernel_start = time.perf_counter()
        k_train, k_test = build_graphs(split, knn, normalize=False, recipe=KERNEL)
        kernel_seconds = time.perf_counter() - kernel_start
        start = time.perf_counter()
        built = mvd(_dense(k_train), _dense(k_test), clusters, seed)
        record("mvd", *(built or (None, None)),
               time.perf_counter() - start, kernel_seconds)

    for name, builder in (("gcca", gcca), ("dgcca", dgcca)):
        if name in cfg["evaluation"].get("skip", []):
            continue
        start = time.perf_counter()
        train, test = builder(split, clusters, seed=seed)
        record(name, train, test, time.perf_counter() - start, graph=0.0)

    # Concatenated raw features: the ceiling every fusion method must clear.
    #
    # Reported twice, because the naive version is NOT dimension-matched and that
    # flatters it: every diffusion arm is truncated to `clusters` components,
    # while `features` keeps all d.  On K-MvMNIST that is 10 vs 200 dims, and the
    # gap is the whole reason concatenation appeared to beat MDT there -- at a
    # matched 10 dims it scores 0.330 against MDT's 0.458.  `features_matched`
    # PCA-projects the concatenation to `clusters` dims so the comparison is
    # about fusion rather than about representational budget.
    stacked_train = np.concatenate(split.train, axis=1)
    stacked_test = np.concatenate(split.test, axis=1)
    features = cluster_metrics(stacked_train, stacked_test,
                               split.y_train, split.y_test, clusters, seed)
    features.update({"dataset": split.name, "seed": seed, "method": "features",
                     "ch_target": None, "kernel": KERNEL,
                     "graph_seconds": 0.0, "train_seconds": 0.0,
                     "encompassed_by_mdt": None})
    rows.append(features)
    print(json.dumps(features), flush=True)

    from sklearn.decomposition import PCA
    rank = min(clusters, stacked_train.shape[1], len(split.y_train) - 1)
    projection = PCA(n_components=max(1, rank), random_state=seed).fit(stacked_train)
    matched = cluster_metrics(projection.transform(stacked_train),
                              projection.transform(stacked_test),
                              split.y_train, split.y_test, clusters, seed)
    matched.update({"dataset": split.name, "seed": seed,
                    "method": "features_matched", "ch_target": None,
                    "kernel": KERNEL, "noise_factor": NOISE_FACTOR,
                    "graph_seconds": 0.0, "train_seconds": 0.0,
                    "encompassed_by_mdt": None})
    rows.append(matched)
    print(json.dumps(matched), flush=True)
    return rows


def dataset_effects(rows: Sequence[dict], candidate: str, baseline: str,
                    metric: str) -> np.ndarray:
    """Per-dataset mean of (candidate - baseline), the unit the intervals use.

    ``closure.paired_effect`` computes these to build its t-interval but only
    returns the moments.  They are needed whole for the two checks below, both
    of which exist because a t-interval over 19 heterogeneous datasets assumes
    those 19 numbers are roughly normal, and there is no reason they should be.
    """
    by_key = {(row["dataset"], row["seed"], row["method"]): row for row in rows}
    grouped: dict[str, list[float]] = {}
    for dataset, seed, method in by_key:
        if method != candidate:
            continue
        other = by_key.get((dataset, seed, baseline))
        if other:
            grouped.setdefault(dataset, []).append(
                by_key[(dataset, seed, candidate)][metric] - other[metric])
    return np.asarray([np.mean(values) for values in grouped.values()])


def bootstrap_interval(effects: np.ndarray, seed: int = 0,
                       draws: int = 20000) -> list[float] | None:
    """Percentile bootstrap over datasets -- no normality assumption."""
    if len(effects) < 3:
        return None
    rng = np.random.default_rng(seed)
    sample = rng.choice(effects, size=(draws, len(effects)), replace=True)
    means = sample.mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def sign_test(effects: np.ndarray) -> dict:
    """Exact two-sided sign test on how many datasets the baseline wins.

    Distribution-free: it only counts signs, so a single dataset with a huge
    effect cannot carry it.  That is the failure mode the mean is exposed to
    here -- on the six-dataset pilot, SpecRaGE's entire advantage over MDT came
    from Wikipedia alone.
    """
    wins = int(np.sum(effects > 0))
    losses = int(np.sum(effects < 0))
    total = wins + losses
    if total == 0:
        return {"wins": 0, "losses": 0, "p_sign": None}
    from math import comb
    extreme = min(wins, losses)
    tail = sum(comb(total, i) for i in range(extreme + 1)) / 2 ** total
    return {"wins": wins, "losses": losses, "ties": int(len(effects) - total),
            "p_sign": float(min(1.0, 2 * tail))}


def summarize(rows: Sequence[dict], baseline: str = "mdt_cvx_rand",
              min_effect: float = .02) -> dict:
    methods = sorted({row["method"] for row in rows})
    summary: dict = {"baseline": baseline, "methods": {}}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        summary["methods"][method] = {
            "encompassed_by_mdt": selected[0].get("encompassed_by_mdt"),
            "runs": len(selected),
            **{key: {"mean": float(np.mean([row[key] for row in selected])),
                     "std": float(np.std([row[key] for row in selected]))}
               for key in ("train_ami", "test_ami", "inductive_ami",
                           "train_seconds")},
        }

    def decision(candidate: str, metric: str) -> dict:
        effect, std, pairs, groups = paired_effect(rows, baseline, candidate, metric)
        if effect is None:
            return {"effect": None, "datasets": 0, "verdict": "inconclusive"}
        critical = float(student_t.ppf(.975, groups - 1)) if groups > 1 else None
        half = critical * std / math.sqrt(groups) if critical else None
        lower = effect - half if half is not None else None
        upper = effect + half if half is not None else None
        # Same rule as the GNN ledger: five independent datasets minimum, and
        # an effect has to clear min_effect, not merely exclude zero.
        if groups < 5:
            verdict = "inconclusive"
        elif effect >= min_effect and lower is not None and lower > 0:
            verdict = "mdt_wins"
        elif upper is not None and upper < min_effect:
            verdict = "no_mdt_advantage"
        else:
            verdict = "inconclusive"
        # Two-sided paired t on the cluster means against H0: effect == 0.  The
        # verdict above uses min_effect, not this p; it is reported so the
        # family-wise correction below has something to correct.
        statistic = effect / (std / math.sqrt(groups)) if std > 0 and groups > 1 else None
        p_value = (float(2 * student_t.sf(abs(statistic), groups - 1))
                   if statistic is not None else None)
        effects = dataset_effects(rows, baseline, candidate, metric)
        return {"effect": effect, "std": std, "pairs": pairs, "datasets": groups,
                "ci95": [lower, upper] if lower is not None else None,
                "ci95_bootstrap": bootstrap_interval(effects),
                "sign_test": sign_test(effects),
                "p_raw": p_value, "verdict": verdict}

    summary["decisions"] = {
        f"{baseline}_vs_{method}": {
            metric: decision(method, metric)
            for metric in ("inductive_ami", "test_ami", "train_ami")
        }
        for method in methods if method != baseline
    }
    # Holm-Bonferroni across the whole family, per metric.  One baseline is
    # compared to every other method at once, so uncorrected 95% intervals
    # would be the wrong thing to quote: with 13 competitors the chance that at
    # least one crosses zero by luck is ~1 - 0.95^13 = 49%.  Holm is uniformly
    # more powerful than Bonferroni and needs no independence assumption, which
    # matters because the arms share a split, a kernel and an embedding.
    for metric in ("inductive_ami", "test_ami", "train_ami"):
        family = [(key, block[metric]) for key, block in summary["decisions"].items()
                  if block[metric].get("p_raw") is not None]
        family.sort(key=lambda pair: pair[1]["p_raw"])
        running = 0.0
        for index, (_, block) in enumerate(family):
            adjusted = min(1.0, (len(family) - index) * block["p_raw"])
            running = max(running, adjusted)      # enforce monotonicity
            block["p_holm"] = running
            block["significant_holm_05"] = bool(running < .05)
        for _, block in summary["decisions"].items():
            block[metric].setdefault("p_holm", None)
            block[metric].setdefault("significant_holm_05", None)
        summary.setdefault("family_size", {})[metric] = len(family)
    # PRR (paper Eq. 21) on the OOD metric, so the table is readable next to the
    # paper's Fig. 6 -- above 1 beats the neutral MDT baseline.
    #
    # Computed as a ratio of means, NOT the paper's mean of per-cell ratios.
    # At 19 datasets the latter is unusable: AMI is chance-adjusted so it can be
    # ~0 or negative, and a single cell where the baseline lands near zero sends
    # its ratio to +-inf and drags the average with it (it produced PRR of -0.85
    # and 0.03 for methods whose mean AMI was within 0.05 of the baseline's).
    # A ratio of means has no such pole. Cells are still paired first, so the
    # two methods are always averaged over exactly the same cells.
    by_key = {(row["dataset"], row["seed"], row["method"]): row for row in rows}
    for method in methods:
        pairs = [(by_key[(d, s, method)]["inductive_ami"],
                  by_key[(d, s, baseline)]["inductive_ami"])
                 for (d, s, m) in by_key if m == method and (d, s, baseline) in by_key]
        denominator = float(np.mean([b for _, b in pairs])) if pairs else 0.0
        summary["methods"][method]["prr_inductive"] = (
            float(np.mean([a for a, _ in pairs]) / denominator)
            if pairs and abs(denominator) > 1e-3 else None)
        summary["methods"][method]["prr_paired_cells"] = len(pairs)
    return summary


def _self_check() -> None:
    """The Nystrom rule is only valid if W == sum_v P_v S_v for every builder.
    Verify that identity numerically, then run the whole suite end to end."""
    split = make_synthetic(0, n_train=120, n_test=60, n_clusters=3)
    train_graphs, test_graphs = build_graphs(split, 5)
    p, bipartite = _dense(train_graphs), _dense(test_graphs)
    for name, builder in OPERATORS.items():
        built = builder(p, steps=4, seed=0, components=3, views=split.train)
        assert built is not None, name
        operator, suffixes = built
        rebuilt = sum(p[v] @ suffix for v, suffix in suffixes.items())
        error = np.abs(operator - rebuilt).max() / max(np.abs(operator).max(), 1e-12)
        assert error < 1e-8, f"{name}: W != sum_v P_v S_v (rel err {error:.2e})"
        train, test = svd_extend(operator, suffixes, bipartite, 3)
        assert train.shape == (120, 3) and test.shape == (60, 3), name
        assert np.all(np.isfinite(train)) and np.all(np.isfinite(test)), name
        print(f"  suffix identity OK  {name:14s} rel_err={error:.1e}")
    for name, builder in (("gcca", gcca), ("dgcca", dgcca)):
        train, test = builder(split, 3, seed=0)
        assert train.shape == (120, 3) and test.shape == (60, 3), name
        assert np.all(np.isfinite(test)), f"{name} produced non-finite test scores"
        print(f"  inductive map OK   {name}")
    print("self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default="experiments/mvbench/config.yml")
    parser.add_argument("--smoke", action="store_true",
                        help="suffix-identity self-check plus a synthetic run")
    parser.add_argument("--summarize", metavar="JSONL")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--skip", nargs="+", help="method names to leave out")
    parser.add_argument("--split", type=float,
                        help="train fraction; sweep it to test OOS sensitivity")
    parser.add_argument("--cap-train", type=int, help="override data.cap_train")
    parser.add_argument("--cap-test", type=int, help="override data.cap_test")
    parser.add_argument("--noise-factor", type=float,
                        help="paper Fig. 7 parameter s; affects K-/L-MvMNIST only")
    parser.add_argument("--kernel", choices=("quantile", "reference"),
                        help="reference = the MDT repo's get_kernel_matrix recipe; "
                             "it implies raw (unstandardised) views")
    parser.add_argument("--ch-target", choices=("views", "operators"),
                        help="what Q_CH scores: the paper's data views, or the "
                             "reference implementation's transition matrices")
    parser.add_argument("--output")
    parser.add_argument("--summary-output")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        _self_check()
        cfg = {"mdt": {"steps": 4, "knn": "auto"},
               "evaluation": {"skip": args.skip or [], "min_effect": .02}}
        run_one(make_synthetic(0, n_train=120, n_test=60), cfg, 0)
        return

    cfg = load_config(args.config)
    if args.datasets:
        cfg["data"]["datasets"] = args.datasets
    if args.seeds:
        cfg["evaluation"]["seeds"] = args.seeds
    if args.skip:
        cfg["evaluation"]["skip"] = args.skip
    if args.split:
        cfg["data"]["split"] = args.split
    if args.ch_target:
        globals()["CH_TARGET"] = args.ch_target
    if args.kernel:
        globals()["KERNEL"] = args.kernel
    if args.noise_factor is not None:
        globals()["NOISE_FACTOR"] = args.noise_factor
    if args.cap_train:
        cfg["data"]["cap_train"] = args.cap_train
    if args.cap_test:
        cfg["data"]["cap_test"] = args.cap_test
    if args.output:
        cfg["evaluation"]["output"] = args.output

    if args.summarize:
        with open(args.summarize, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        report = summarize(rows, cfg["evaluation"].get("baseline", "mdt_cvx_rand"),
                           cfg["evaluation"]["min_effect"])
        if args.summary_output:
            Path(args.summary_output).write_text(json.dumps(report, indent=2) + "\n",
                                                encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    data = cfg["data"]
    output = Path(cfg["evaluation"]["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, completed = [], set()
    if args.resume and output.exists():
        with output.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        completed = {(row["dataset"], row["seed"]) for row in rows}
    with output.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for seed in cfg["evaluation"]["seeds"]:
            for name in data["datasets"]:
                if (name, seed) in completed:
                    print(f"[resume] skipping {name} seed={seed}", flush=True)
                    continue
                seed_everything(seed)
                # The paper's constructed datasets (K-/L-MvMNIST, Olivetti) are
                # not .mat files; they are built in paper_datasets with every
                # fitted transform estimated on the training rows only.
                if name in PAPER_DATASETS:
                    split = load_paper_split(name, float(data["split"]), seed,
                                             data.get("cap_train"),
                                             data.get("cap_test"),
                                             standardize=(KERNEL != "reference"),
                                             noise_factor=NOISE_FACTOR)
                else:
                    split = load_split(data["path"], name, float(data["split"]), seed,
                                       data.get("cap_train"), data.get("cap_test"),
                                       standardize=(KERNEL != "reference"))
                print(f"[{name} seed={seed}] n_train={len(split.y_train)} "
                      f"n_test={len(split.y_test)} views={len(split.train)}",
                      flush=True)
                for row in run_one(split, cfg, seed):
                    rows.append(row)
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
    report = summarize(rows, cfg["evaluation"].get("baseline", "mdt_cvx_rand"),
                       cfg["evaluation"]["min_effect"])
    target = (Path(args.summary_output) if args.summary_output
              else output.with_name("summary.json"))
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["decisions"], indent=2))


if __name__ == "__main__":
    main()
