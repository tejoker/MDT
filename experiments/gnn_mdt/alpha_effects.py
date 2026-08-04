"""Paired alpha-family effects on the MDT out-of-sample arms.

Each alpha arm is a separate metrics file produced by ``closure.py --alpha a``
on identical datasets, seeds and splits, so a row pairs with its alpha=0 twin on
``(dataset, seed, method)``.  The estimator is the one the automated summary
uses: average the seeds inside a dataset first, then take a paired 95% Student
interval across the dataset means.

    python -m experiments.gnn_mdt.alpha_effects results/gnn_mdt/alpha_a*_metrics.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from experiments.gnn_mdt.closure import paired_effect, student_t
import math

METRICS = ("test_ami", "inductive_ami", "train_ami")
METHODS = ("mdt_nystrom", "teacher_mlp", "teacher_gnn", "mdt_consensus")


def load(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                # paired_effect keys on the method name, so the alpha arm has to
                # live in that name for the pairing to be an alpha contrast.
                row["method"] = f"{row['method']}@a{row['alpha']:g}"
                rows.append(row)
    return rows


def interval(rows: list[dict], candidate: str, baseline: str, metric: str) -> dict:
    effect, std, pairs, groups = paired_effect(rows, candidate, baseline, metric)
    if effect is None:
        return {"effect": None, "pairs": 0, "datasets": 0}
    half = (float(student_t.ppf(.975, groups - 1)) * std / math.sqrt(groups)
            if groups > 1 else None)
    index = {(row["dataset"], row["seed"], row["method"]): row for row in rows}
    wins = sum(1 for (dataset, seed, method), row in index.items()
               if method == candidate
               and (dataset, seed, baseline) in index
               and row[metric] > index[(dataset, seed, baseline)][metric])
    return {"effect": effect, "ci95": None if half is None else [effect - half, effect + half],
            "pairs": pairs, "datasets": groups, "wins": wins}


def main() -> None:
    rows = load(sys.argv[1:] or ["results/gnn_mdt/alpha_a0.0_metrics.jsonl"])
    alphas = sorted({row["alpha"] for row in rows})
    report: dict = {"alphas": alphas, "means": {}, "effects": {}}
    for method in METHODS:
        for alpha in alphas:
            name = f"{method}@a{alpha:g}"
            selected = [row for row in rows if row["method"] == name]
            if selected:
                report["means"][name] = {
                    metric: sum(row[metric] for row in selected) / len(selected)
                    for metric in METRICS
                }
        for alpha in alphas[1:]:
            for metric in METRICS:
                key = f"{method} a{alpha:g}-a{alphas[0]:g} {metric}"
                report["effects"][key] = interval(
                    rows, f"{method}@a{alpha:g}", f"{method}@a{alphas[0]:g}", metric)
    # The graph term of the decomposition must be re-read at every alpha: alpha
    # changes the message source, so a closed GNN-minus-MLP verdict at alpha=0
    # says nothing about alpha=1.
    for alpha in alphas:
        for metric in METRICS:
            for candidate, baseline in (("teacher_gnn", "teacher_mlp"),
                                        ("teacher_mlp", "mdt_nystrom"),
                                        ("teacher_gnn", "mdt_nystrom")):
                key = f"{candidate}-{baseline} @a{alpha:g} {metric}"
                report["effects"][key] = interval(
                    rows, f"{candidate}@a{alpha:g}", f"{baseline}@a{alpha:g}", metric)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
