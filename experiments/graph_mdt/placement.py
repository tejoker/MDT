"""Step 1b: place MDT-with-graph-views in the published clustering tables.

Full-graph protocol (all nodes, no split) to match the multiplex clustering
literature (O2MAC / DMGI lineages): KMeans on the MDT embedding, metrics
ACC / macro-F1 / NMI / ARI / AMI, mean over 5 KMeans seeds.

Arms per dataset (all label-free):
  feat   MDT on the feature-kNN transition only
  graph  MDT on the natural adjacency transitions only
  all    MDT on feature + adjacency transitions jointly

Fusion modes:
  consensus  mean Gram over 8 random Dirichlet trajectories (locked recipe)
  select     best of 16 sampled trajectories by silhouette (the lab's
             label-free select_trajectory criterion, lightweight port)

Feature graph is a cosine kNN connectivity graph (k = floor(log n)): the
Gaussian-euclidean graph of the gnn_mdt harness collapses on high-dim BoW
features (ACM homophily 0.36 vs 0.71 cosine — see independence screen).

Usage: python -m experiments.graph_mdt.placement [dataset ...]
Appends rows to results/graph_mdt/placement.jsonl and prints them as json.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             f1_score, normalized_mutual_info_score,
                             silhouette_score)

from experiments.gnn_mdt.closure import _row_normalize, svd_embedding, trajectory
from experiments.graph_mdt import datasets
from experiments.graph_mdt.independence import knn_graph

STEPS = 4
CONSENSUS_TRAJECTORIES = 8
SELECT_TRAJECTORIES = 16
KMEANS_SEEDS = [0, 1, 2, 3, 4]
OUTPUT = "results/graph_mdt/placement.jsonl"


def feature_transition(features: np.ndarray) -> sp.csr_matrix:
    k = max(1, int(np.log(len(features))))
    return _row_normalize(knn_graph(features, k))


def graph_transition(a: sp.csr_matrix) -> sp.csr_matrix:
    return _row_normalize(a + sp.eye(a.shape[0], format="csr"))


def _operator(dense_views: list, weights: np.ndarray) -> np.ndarray:
    operator = None
    for row in weights:
        step = sum(float(a) * g for a, g in zip(row, dense_views))
        operator = step if operator is None else step @ operator
    return operator


def _embed(operator: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    from scipy.sparse.linalg._eigen.arpack import ArpackError
    try:
        factor, _, _ = svd_embedding(sp.csr_matrix(operator), n_components, seed)
    except ArpackError:  # incl. NoConvergence and "no shifts" (error 3)
        return None  # degenerate operator; caller skips this candidate
    return factor


def _silhouette(embedding: np.ndarray, k: int, seed: int = 0) -> float:
    sample = np.random.default_rng(seed).choice(
        len(embedding), min(2000, len(embedding)), replace=False)
    sub = embedding[sample]
    pred = KMeans(k, n_init=4, random_state=seed).fit_predict(sub)
    if len(np.unique(pred)) < 2:
        return -1.0
    return float(silhouette_score(sub, pred))


def consensus_embedding_full(dense_views: list, n_components: int,
                             seed: int = 0) -> np.ndarray:
    n = dense_views[0].shape[0]
    target = np.zeros((n, n), dtype=np.float64)
    used = 0
    for index in range(CONSENSUS_TRAJECTORIES):
        weights = trajectory(len(dense_views), STEPS, seed * 1000 + index)
        factor = _embed(_operator(dense_views, weights), n_components, seed + index)
        if factor is None:
            continue
        target += factor @ factor.T
        used += 1
    target /= max(used, 1)
    values, vectors = eigsh(target, k=min(n_components, n - 1), which="LA")
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    return (vectors * np.sqrt(np.clip(values, 0, None))).astype(np.float32)


def selected_embedding_full(dense_views: list, n_components: int, k: int,
                            scorer, seed: int = 0) -> np.ndarray:
    """Sample Dirichlet + one-hot trajectories, keep the best modularity on
    the union input graph (label-free). Silhouette was the original
    criterion but rewards collapsed diffusion embeddings — see ledger."""
    from experiments.graph_mdt.methods import modularity_criterion
    rng = np.random.default_rng(seed)
    candidates = [trajectory(len(dense_views), STEPS, seed * 1000 + i)
                  for i in range(SELECT_TRAJECTORIES)]
    for view in range(len(dense_views)):  # pure single-view paths
        one_hot = np.zeros((STEPS, len(dense_views)))
        one_hot[:, view] = 1.0
        candidates.append(one_hot)
    best, best_score = None, -np.inf
    for weights in candidates:
        factor = _embed(_operator(dense_views, weights), n_components,
                        int(rng.integers(1 << 30)))
        if factor is None:
            continue
        score = modularity_criterion(factor, k, scorer, seed)
        if score > best_score:
            best, best_score = factor, score
    return best


def clustering_accuracy(labels: np.ndarray, pred: np.ndarray) -> tuple:
    classes, clusters = np.unique(labels), np.unique(pred)
    cost = np.zeros((len(clusters), len(classes)))
    for i, c in enumerate(clusters):
        for j, k in enumerate(classes):
            cost[i, j] = -np.sum((pred == c) & (labels == k))
    rows, cols = linear_sum_assignment(cost)
    mapping = {clusters[r]: classes[c] for r, c in zip(rows, cols)}
    mapped = np.array([mapping[p] for p in pred])
    return float((mapped == labels).mean()), float(
        f1_score(labels, mapped, average="macro"))


def evaluate(embedding: np.ndarray, labels: np.ndarray) -> dict:
    k = len(np.unique(labels))
    metrics = {m: [] for m in ("acc", "f1", "nmi", "ari", "ami")}
    for seed in KMEANS_SEEDS:
        pred = KMeans(k, n_init=10, random_state=seed).fit_predict(embedding)
        acc, f1 = clustering_accuracy(labels, pred)
        metrics["acc"].append(acc)
        metrics["f1"].append(f1)
        metrics["nmi"].append(normalized_mutual_info_score(labels, pred))
        metrics["ari"].append(adjusted_rand_score(labels, pred))
        metrics["ami"].append(adjusted_mutual_info_score(labels, pred))
    return {m: round(float(np.mean(v)), 4) for m, v in metrics.items()} | {
        f"{m}_std": round(float(np.std(v)), 4) for m, v in metrics.items()}


def run_dataset(name: str, dims: list) -> list:
    features, graphs, labels = datasets.load(name)
    k = len(np.unique(labels))
    arms = {"feat": [feature_transition(features)],
            "graph": [graph_transition(a) for _, a in sorted(graphs.items())]}
    arms["all"] = arms["feat"] + arms["graph"]
    scorer = (arms["feat"][0] > 0).astype(np.float32)
    for _, a in sorted(graphs.items()):
        scorer = scorer.maximum((a > 0).astype(np.float32))
    scorer = scorer.maximum(scorer.T).tocsr()
    rows = []
    for arm, views in arms.items():
        dense = [np.asarray(v.todense(), dtype=np.float32) for v in views]
        for dim in dims:
            for mode in ("consensus", "select"):
                start = time.time()
                if mode == "consensus":
                    embedding = consensus_embedding_full(dense, dim)
                else:
                    embedding = selected_embedding_full(dense, dim, k, scorer)
                row = {"dataset": name, "arm": arm, "mode": mode, "dim": dim,
                       "views": len(views),
                       "seconds": round(time.time() - start, 1)}
                row |= evaluate(embedding, labels)
                rows.append(row)
                print(json.dumps(row), flush=True)
        del dense
    return rows


def main(names: list) -> None:
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "a") as f:
        for name in names:
            try:
                classes = len(np.unique(datasets.load(name)[2]))
                for row in run_dataset(name, dims=[classes, 32]):
                    f.write(json.dumps(row) + "\n")
                    f.flush()
            except Exception as exc:  # noqa: BLE001 - keep other datasets alive
                print(f"{name}: FAILED {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["acm", "imdb", "dblp"])
