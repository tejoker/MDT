"""Code-enforced verdicts for the second-formulation ledger (step 5).

Frozen rule (pre-registered): pair per (dataset, seed), average seeds within
dataset, 95% t-interval across dataset means. continue iff mean >= +0.02 and
lower > 0; dead_end iff upper < +0.02; else inconclusive. No verdict from
fewer than 5 primary datasets. Secondary: across-dataset win rate.

Usage: python -m experiments.graph_mdt.verdicts results/graph_mdt/grid.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import t as student_t

MIN_EFFECT = 0.02
PRIMARY = ["acm", "dblp", "imdb", "cora", "citeseer", "amazon_ratings"]
CONTROLS = ["minesweeper", "roman_empire"]
METRIC = "ami"

GNN_ARMS = ["gcn_gae", "gcn_bgrl", "sage_gae", "sage_bgrl"]
COMPARISONS = (
    # pivotal: GNN vs MDT given the same inputs, both label-free MDT modes
    [(g, "mdt_select_all") for g in GNN_ARMS]
    + [(g, "mdt_consensus_all") for g in GNN_ARMS]
    # feature references: learned SSL ceiling and raw KMeans floor
    + [(g, "mlp_ssl") for g in GNN_ARMS]
    + [(g, "kmeans_feat") for g in GNN_ARMS]
    # fusion readout and inference-time graph need
    + [("mdt_select_all", "mdt_feat"), ("mdt_select_all", "mdt_graph"),
       ("mdt_consensus_all", "mdt_feat"), ("mdt_consensus_all", "mdt_graph"),
       ("distill_mlp", "mdt_select_all")]
    # enhancement round E1/E3 (pre-registered)
    + [("mdt_gated", "mdt_consensus_all"), ("mdt_a2", "mdt_consensus_all"),
       ("mdt_a2", "gcn_gae"), ("mdt_gated_sharp", "mdt_consensus_all")]
)


def load_rows(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def paired(rows: list, candidate: str, baseline: str,
           datasets: list) -> dict:
    values = defaultdict(dict)
    for row in rows:
        if row["dataset"] in datasets:
            values[(row["dataset"], row["seed"])][row["arm"]] = row[METRIC]
    per_dataset = defaultdict(list)
    for (dataset, _), arms in values.items():
        if candidate in arms and baseline in arms:
            per_dataset[dataset].append(arms[candidate] - arms[baseline])
    means = {d: float(np.mean(v)) for d, v in per_dataset.items()}
    n = len(means)
    if n == 0:
        return {"n_datasets": 0, "verdict": "no_data"}
    effects = np.array(list(means.values()))
    mean = float(effects.mean())
    result = {"candidate": candidate, "baseline": baseline,
              "n_datasets": n, "effect": round(mean, 4),
              "per_dataset": {d: round(v, 4) for d, v in means.items()},
              "wins": int((effects > 0).sum())}
    if n < 5:
        result |= {"verdict": "inconclusive", "reason": "fewer than 5 datasets"}
        return result
    half = float(student_t.ppf(.975, n - 1) * effects.std(ddof=1) / np.sqrt(n))
    lower, upper = mean - half, mean + half
    result["ci95"] = [round(lower, 4), round(upper, 4)]
    if mean >= MIN_EFFECT and lower > 0:
        result["verdict"] = "continue"
    elif upper < MIN_EFFECT:
        result["verdict"] = "dead_end"
    else:
        result["verdict"] = "inconclusive"
    return result


def main(path: str) -> None:
    rows = load_rows(path)
    report = {"primary": [], "controls": []}
    for candidate, baseline in COMPARISONS:
        report["primary"].append(paired(rows, candidate, baseline, PRIMARY))
        control = paired(rows, candidate, baseline, CONTROLS)
        control.pop("verdict", None)  # controls never issue verdicts
        report["controls"].append(control)
    print(json.dumps(report, indent=1))
    out = path.replace(".jsonl", "_verdicts.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/graph_mdt/grid.jsonl")
