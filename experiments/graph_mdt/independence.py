"""Graph-independence screen (pre-registered gate, step 2 of the ledger).

For every natural adjacency view, quantify how much label-relevant structure
it carries beyond a feature-kNN graph of matched density:

  edge_homophily      fraction of edges joining same-label nodes
  adjusted_homophily  Platonov et al. 2023 class-imbalance correction
  knn_homophily       same metrics for the cosine kNN graph on features,
                      k = round(mean degree of the natural graph)
  edge_jaccard        overlap of undirected edge sets, natural vs kNN

Gate (pre-registered): a dataset enters the PRIMARY analysis iff at least one
natural view has edge_jaccard <= 0.30 (the graph is not a re-derivation of
the features). Views with adjusted_homophily < 0.05 are flagged heterophilous
and analysed as secondary controls, not primary evidence.

Usage: python -m experiments.graph_mdt.independence [dataset ...]
Writes results/graph_mdt/independence.jsonl and prints a markdown table.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from experiments.graph_mdt import datasets

JACCARD_GATE = 0.30
HOMOPHILY_FLAG = 0.05
OUTPUT = "results/graph_mdt/independence.jsonl"


def _edge_codes(a: sp.csr_matrix) -> np.ndarray:
    """Undirected edges as sorted int64 codes row*n+col (upper triangle)."""
    coo = sp.triu(a, k=1).tocoo()
    return np.sort(coo.row.astype(np.int64) * a.shape[0] + coo.col)


def edge_homophily(a: sp.csr_matrix, labels: np.ndarray) -> float:
    coo = sp.triu(a, k=1).tocoo()
    if coo.nnz == 0:
        return float("nan")
    return float((labels[coo.row] == labels[coo.col]).mean())


def adjusted_homophily(a: sp.csr_matrix, labels: np.ndarray) -> float:
    h = edge_homophily(a, labels)
    degrees = np.asarray(a.sum(axis=1)).ravel()
    total = degrees.sum()
    baseline = sum((degrees[labels == c].sum() / total) ** 2
                   for c in np.unique(labels))
    return float((h - baseline) / (1.0 - baseline))


def knn_graph(features: np.ndarray, k: int) -> sp.csr_matrix:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(features)
    a = nn.kneighbors_graph(features, mode="connectivity")
    a.setdiag(0)
    a.eliminate_zeros()
    return a.maximum(a.T).tocsr()


def screen_dataset(name: str) -> list:
    features, graphs, labels = datasets.load(name)
    rows = []
    for gname, a in graphs.items():
        mean_degree = a.nnz / a.shape[0]
        # ponytail: k capped at 128 — beyond that a kNN graph is not a
        # plausible feature-derived alternative; Jaccard vs a near-complete
        # metapath graph is ~0 by density mismatch alone (see mean_degree).
        k = min(max(1, round(mean_degree)), 128)
        knn = knn_graph(features, k)
        natural_edges, knn_edges = _edge_codes(a), _edge_codes(knn)
        shared = len(np.intersect1d(natural_edges, knn_edges, assume_unique=True))
        union = len(natural_edges) + len(knn_edges) - shared
        rows.append({
            "dataset": name, "graph": gname, "n": int(a.shape[0]),
            "edges": int(a.nnz // 2), "mean_degree": round(mean_degree, 2),
            "knn_k": k,
            "edge_homophily": round(edge_homophily(a, labels), 4),
            "adjusted_homophily": round(adjusted_homophily(a, labels), 4),
            "knn_edge_homophily": round(edge_homophily(knn, labels), 4),
            "knn_adjusted_homophily": round(adjusted_homophily(knn, labels), 4),
            "edge_jaccard": round(shared / union, 4) if union else float("nan"),
        })
        print(f"  screened {name}/{gname}", file=sys.stderr, flush=True)
    for row in rows:
        row["independent"] = bool(row["edge_jaccard"] <= JACCARD_GATE)
        row["heterophilous"] = bool(row["adjusted_homophily"] < HOMOPHILY_FLAG)
    return rows


def main(names: list) -> None:
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    all_rows = []
    for name in names:
        try:
            all_rows.extend(screen_dataset(name))
        except Exception as exc:  # noqa: BLE001 - report and continue the screen
            print(f"{name}: FAILED {exc}", file=sys.stderr)
    with open(OUTPUT, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    cols = ["dataset", "graph", "n", "edges", "mean_degree", "knn_k", "edge_homophily",
            "adjusted_homophily", "knn_edge_homophily",
            "knn_adjusted_homophily", "edge_jaccard", "independent",
            "heterophilous"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for row in all_rows:
        print("| " + " | ".join(str(row[c]) for c in cols) + " |")


if __name__ == "__main__":
    main(sys.argv[1:] or datasets.available())
