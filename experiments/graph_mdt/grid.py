"""Step 3 grid runner: all arms x primary datasets x 5 seeds, full-graph AMI.

Arms (pre-registered):
  mdt_consensus_all / mdt_select_all   lab baseline, graph views included
  mdt_feat / mdt_graph                 decomposition readout (select mode)
  gcn_gae, gcn_bgrl, sage_gae, sage_bgrl   the pivotal GNN cells
  mlp_ssl                              feature ceiling, no graph
  distill_mlp                          best-silhouette GNN -> MLP

Resume: existing (dataset, arm, seed) rows in the output are skipped.

Usage: python -m experiments.graph_mdt.grid [dataset ...]
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.graph_mdt import datasets, methods
from experiments.graph_mdt.placement import (evaluate, feature_transition,
                                             graph_transition)

SEEDS = [0, 1, 2, 3, 4]
GNN_ARMS = [("gcn", "gae"), ("gcn", "bgrl"), ("sage", "gae"), ("sage", "bgrl")]
OUTPUT = "results/graph_mdt/grid.jsonl"
DEFAULT_DATASETS = ["acm", "imdb", "cora", "citeseer", "dblp",
                    "amazon_ratings", "minesweeper", "roman_empire"]


def existing_rows(path: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                done.add((row["dataset"], row["arm"], row["seed"]))
    return done


def run_dataset(name: str, out, done: set) -> None:
    features, graphs, labels = datasets.load(name)
    k = len(np.unique(labels))
    dim = 32  # uniform embedding dim for every arm; KMeans still uses k
    x = StandardScaler().fit_transform(features).astype(np.float32)
    feat_view = feature_transition(features)
    # GRAPH_MDT_VIEW_CAP=64: cap MDT view degrees like the GNN fused graph
    # (symmetric treatment); used for dblp where PATAP makes svds intractable
    cap = int(os.environ.get("GRAPH_MDT_VIEW_CAP", "0"))
    raw_views = [methods._cap_neighbors(a, cap) if cap else a
                 for _, a in sorted(graphs.items())]
    graph_views = [graph_transition(a) for a in raw_views]
    fused = methods.fused_adjacency(graphs)
    # label-free selection scorer: binary union of every input graph
    scorer = (feat_view > 0).astype(np.float32)
    for _, a in sorted(graphs.items()):
        scorer = scorer.maximum((a > 0).astype(np.float32))
    scorer = scorer.maximum(scorer.T).tocsr()

    def emit(arm, seed, embedding, seconds):
        if embedding is None:
            print(f"{name}/{arm}/seed{seed}: no valid embedding, skipped",
                  flush=True)
            return
        row = {"dataset": name, "arm": arm, "seed": seed, "dim": dim,
               "seconds": round(seconds, 1)} | evaluate(embedding, labels)
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps(row), flush=True)

    # E1: gates from self-loop-free transitions on raw features
    from experiments.gnn_mdt.closure import _row_normalize
    gate_transitions = [feat_view] + [_row_normalize(a) for a in raw_views]
    gates = methods.view_gates(features.astype(np.float64), gate_transitions)
    # secondary sharpened variant (beta=4, fixed a priori, no sweep)
    sharp_gates = gates ** 4
    sharp_gates /= sharp_gates.sum(axis=0, keepdims=True) + 1e-12
    # E3: binarised capped A^2 of every natural view
    a2_views = [graph_transition(methods.two_hop_view(a)) for a in raw_views]

    for seed in SEEDS:
        mdt_arms = {
            "mdt_consensus_all": lambda: methods.mdt_consensus(
                [feat_view] + graph_views, dim, seed),
            "mdt_select_all": lambda: methods.mdt_select(
                [feat_view] + graph_views, dim, k, scorer, seed),
            # selection-vs-diversity probe: same pool as select, top 8 averaged
            "mdt_top8_all": lambda: methods.mdt_select(
                [feat_view] + graph_views, dim, k, scorer, seed, top=8),
            "mdt_feat": lambda: methods.mdt_select(
                [feat_view], dim, k, scorer, seed),
            "mdt_graph": lambda: methods.mdt_select(
                graph_views, dim, k, scorer, seed),
            "mdt_gated": lambda: methods.mdt_gated_consensus(
                [feat_view] + graph_views, gates, dim, seed),
            "mdt_gated_sharp": lambda: methods.mdt_gated_consensus(
                [feat_view] + graph_views, sharp_gates, dim, seed),
            "mdt_a2": lambda: methods.mdt_consensus(
                [feat_view] + graph_views + a2_views, dim, seed),
        }
        for arm, fn in mdt_arms.items():
            if (name, arm, seed) in done:
                continue
            start = time.time()
            emit(arm, seed, fn(), time.time() - start)

        best_gnn, best_score = None, -np.inf
        for backbone, objective in GNN_ARMS:
            arm = f"{backbone}_{objective}"
            if (name, arm, seed) in done:
                continue
            start = time.time()
            z = methods.train_gnn(x, fused, objective, backbone, dim, k, seed,
                                  scorer=scorer)
            emit(arm, seed, z, time.time() - start)
            score = methods.modularity_criterion(z, k, scorer, seed)
            if score > best_score:
                best_gnn, best_score = z, score

        if (name, "kmeans_feat", seed) not in done:
            emit("kmeans_feat", seed, x, 0.0)

        if (name, "mlp_ssl", seed) not in done:
            start = time.time()
            emit("mlp_ssl", seed,
                 methods.train_mlp_ssl(x, dim, k, seed, scorer=scorer),
                 time.time() - start)

        if (name, "distill_mlp", seed) not in done and best_gnn is not None:
            start = time.time()
            emit("distill_mlp", seed, methods.distill_mlp(x, best_gnn, k, seed),
                 time.time() - start)


def main(names: list) -> None:
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    done = existing_rows(OUTPUT)
    with open(OUTPUT, "a") as out:
        for name in names:
            try:
                run_dataset(name, out, done)
            except Exception as exc:  # noqa: BLE001 - isolate dataset failures
                print(f"{name}: FAILED {exc}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_DATASETS)
