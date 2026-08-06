"""SpecRaGE as an mvbench arm -- the strongest published inductive competitor.

SpecRaGE (Yacobi et al., TMLR 2025, arXiv:2411.02138,
github.com/shaham-lab/SpecRaGE) learns a *parametric* map that approximates the
joint diagonalisation of the per-view graph Laplacians, with a fusion module
that reweights views by quality.  It is the closest published relative of this
repo's MDT -> DDM pipeline: same goal (a generalisable multi-view spectral
embedding), one stage instead of two, and it needs no Nystrom rule because the
network itself extends out of sample.  So it is the arm that can actually
threaten the mvbench result, and it is run here on the *identical* split.

Run it against the vendored upstream checkout:

    git clone https://github.com/shaham-lab/SpecRaGE
    python -m experiments.mvbench.specrage_arm --repo /path/to/SpecRaGE \
        -c experiments/mvbench/config.yml

Fairness notes, all of them load-bearing:

* Same split.  ``closure.load_split`` provides train/test, so SpecRaGE sees
  exactly the rows every other arm saw -- not the upstream ``_data.py``
  loaders, which use a different ratio (0.8), a fixed seed 42, and for
  Handwritten only views 0 and 2 of 6.
* Same metrics.  ``closure.cluster_metrics`` on ``predict(train)`` and
  ``predict(test)``, so ``inductive_ami`` means what it means everywhere else.
* Unsupervised.  ``SpecRaGETrainer.train`` stores ``labels`` but the loss is
  ``SpecRaGELoss(Ws, Y)`` over per-view affinities only -- verified by reading
  it.  Labels are passed because ``MultiViewDataset`` wants them, and are
  never differentiated through.
* No leakage.  Upstream caches AE and siamese weights under
  ``weights/{dataset}{i}_*.pth``.  ``fit`` trains them on train rows and saves;
  ``predict`` then *loads* them.  That is only true if the cache key is unique
  per cell -- with a shared key, cell 2 would reuse cell 1's encoder, and with
  no cache at all ``predict`` would retrain the autoencoder on the test rows.
  This module therefore keys every cache on ``{dataset}_s{seed}``.
* Batch size is scaled, and it must be.  ``_get_data_loader`` uses
  ``drop_last=True`` on a 0.9 subsample, so the configured 1024 yields *zero*
  batches whenever ``n_train < 1138``: the model would report a loss of 0.0 and
  train on nothing.  Every scaled hyperparameter is written into the output row
  under ``config_scaled`` so the deviation from upstream is auditable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

from experiments.gnn_mdt.closure import (
    cluster_metrics, load_config, load_split, make_synthetic, seed_everything,
)


def _import_specrage(repo: str):
    """Put the upstream package on the path, stubbing what will not build."""
    source = Path(repo) / "src"
    if not source.is_dir():
        raise SystemExit(f"no src/ under {repo} -- clone shaham-lab/SpecRaGE first")
    # annoy needs a C++ toolchain and is only reached under use_approx /
    # is_sparse_graph, both False in every upstream config and here.  Stub it so
    # the import succeeds and any *actual* use is a loud failure, not a silent
    # fallback to a different neighbour graph.
    if "annoy" not in sys.modules:
        stub = types.ModuleType("annoy")
        def _unavailable(*_a, **_k):
            raise RuntimeError("annoy is stubbed: set use_approx/is_sparse_graph False")
        stub.AnnoyIndex = _unavailable
        sys.modules["annoy"] = stub
    sys.path.insert(0, str(source))
    import matplotlib
    matplotlib.use("Agg")           # upstream imports pyplot at module scope
    from _model import SpecRaGE     # noqa: E402
    return SpecRaGE


def build_config(name: str, seed: int, dimensions: list[int], clusters: int,
                 n_train: int, attempt: int = 0) -> tuple[dict, dict]:
    """Upstream handwritten.json, with only the size-dependent knobs rescaled."""
    usable = int(0.9 * n_train)                     # random_split inside upstream
    validation = n_train - usable                   # the other 10%
    batch = max(4, min(1024, usable))
    # The neighbour count is bounded by the *validation* batch, not the train
    # batch: ``validate`` calls the same ``_get_affinity_matrix``, and
    # ``compute_scale`` indexes ``Dis[:, scale_k - 1]`` on a matrix with only
    # ``min(n_neighbors + 1, validation)`` columns.  Upstream's 30 crashes with
    # IndexError on any split whose 10% validation slice is smaller than that.
    neighbors = max(2, min(30, validation - 2, batch // 8))
    scaled = {"batch_size": batch, "n_neighbors": neighbors,
              "scale_k": max(2, min(20, neighbors)),
              "siamese_neighbors": max(1, min(3, batch // 16)),
              "usable_train_rows": usable, "validation_rows": validation}
    encoder = [{"hidden_dim1": 512, "hidden_dim2": 512, "hidden_dim3": 2048,
                "output_dim": clusters} for _ in dimensions]
    network = [{"n_layers": 5, "hidden_dim1": 1024, "hidden_dim2": 1024,
                "hidden_dim3": 512, "output_dim": clusters} for _ in dimensions]
    config = {
        # Unique cache key: no leakage between cells, and a retry must not
        # reload the diverged attempt's cached AE/siamese weights.
        "dataset": f"{name}_s{seed}" + (f"_a{attempt}" if attempt else ""),
        "n_views": len(dimensions),
        "should_use_ae": True,
        "should_use_siamese": True,
        "datatypes": ["vector"] * len(dimensions),
        "n_clusters": clusters,
        "is_sparse_graph": False,
        "ae": {"architectures": encoder, "epochs": 100, "n_samples": 70000,
               "lr": 1e-3, "lr_decay": .1, "min_lr": 1e-7, "patience": 5,
               "batch_size": min(256, batch)},
        "siamese": {"architectures": network, "epochs": 20, "n_samples": 5000,
                    "lr": 1e-3, "lr_decay": .1, "min_lr": 1e-7, "patience": 5,
                    "n_neighbors": scaled["siamese_neighbors"],
                    "use_approx": False, "batch_size": min(128, batch)},
        "spectral": {"architectures": network, "epochs": 35, "lr": 1e-3,
                     "lr_decay": .1, "min_lr": 1e-8, "batch_size": batch,
                     "n_neighbors": neighbors, "scale_k": scaled["scale_k"],
                     "is_local_scale": False, "temperture": 5, "patience": 5},
    }
    return config, scaled


def _fit_once(SpecRaGE, split, seed: int, clusters: int, train, test,
              config) -> tuple:
    started = time.perf_counter()
    model = SpecRaGE(n_clusters=clusters, config=config)
    model.fit(list(train), torch.from_numpy(split.y_train.astype(np.int64)))
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    # predict() mutates the list it is given when should_use_ae is False, and
    # re-embeds through the cached AE when True -- pass fresh copies both times.
    embedded_test = model.predict(list(test))
    embedded_train = model.predict(list(train))
    return embedded_train, embedded_test, fit_seconds, time.perf_counter() - started


def run_cell(SpecRaGE, split, seed: int, attempts: int = 3) -> dict:
    """Fit and extend, retrying a diverged fit with a fresh initialisation.

    On high-dimensional sparse text views the SpectralNet-style
    orthonormalisation (a Cholesky of the batch output Gram) can fail on the
    very first batch: the loss is ``nan`` from epoch 1, not a slow divergence.
    Measured on BBCSport (3183/3203-dim), 3 of 10 seeds diverge this way.

    Retrying with a different init is what any user of the method would do, and
    dropping those cells instead would handicap the competitor with biased
    missingness -- the same failure this benchmark already fixed on its own side
    (see ``closure._row_normalize``).  ``attempts_used`` and ``diverged`` are
    recorded on every row so the robustness cost stays visible instead of being
    laundered away by the retry.
    """
    clusters = len(np.unique(split.y_train))
    train = [torch.from_numpy(np.ascontiguousarray(v)) for v in split.train]
    test = [torch.from_numpy(np.ascontiguousarray(v)) for v in split.test]

    for attempt in range(attempts):
        config, scaled = build_config(split.name, seed, [v.shape[1] for v in train],
                                     clusters, len(split.y_train), attempt)
        torch.manual_seed(seed + 1000 * attempt)
        embedded_train, embedded_test, fit_seconds, inference_seconds = _fit_once(
            SpecRaGE, split, seed, clusters, train, test, config)
        if np.all(np.isfinite(embedded_train)) and np.all(np.isfinite(embedded_test)):
            row = cluster_metrics(embedded_train, embedded_test, split.y_train,
                                  split.y_test, clusters, seed)
            row.update({"dataset": split.name, "seed": seed, "method": "specrage",
                        "graph_seconds": 0.0, "train_seconds": fit_seconds,
                        "inference_seconds": inference_seconds,
                        "encompassed_by_mdt": False, "config_scaled": scaled,
                        "attempts_used": attempt + 1, "diverged": attempt > 0})
            return row
        print(f"  [retry] {split.name} seed={seed} attempt {attempt + 1} diverged "
              f"(nan loss from epoch 1)", flush=True)
    return {"dataset": split.name, "seed": seed, "method": "specrage",
            "error": f"non-finite embedding after {attempts} inits",
            "config_scaled": scaled, "attempts_used": attempts, "diverged": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="clone of shaham-lab/SpecRaGE")
    parser.add_argument("-c", "--config", default="experiments/mvbench/config.yml")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default="results/mvbench/specrage.jsonl")
    parser.add_argument("--append", action="store_true",
                        help="append instead of truncating (for re-running cells)")
    parser.add_argument("--workdir", default="/tmp/specrage_work",
                        help="upstream writes weights/ relative to the cwd")
    args = parser.parse_args()

    SpecRaGE = _import_specrage(args.repo)
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.workdir)                      # isolate the weights/ cache

    if args.smoke:
        seed_everything(0)
        print(json.dumps(run_cell(SpecRaGE, make_synthetic(0, 240, 120), 0), indent=2))
        return

    root = Path(__file__).resolve().parents[2]
    cfg = load_config(str(root / args.config))
    data = cfg["data"]
    datasets = args.datasets or data["datasets"]
    seeds = args.seeds or cfg["evaluation"]["seeds"]
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("a" if args.append else "w", encoding="utf-8") as handle:
        for seed in seeds:
            for name in datasets:
                seed_everything(seed)
                split = load_split(data["path"], name, float(data["split"]), seed,
                                   data.get("cap_train"), data.get("cap_test"))
                print(f"[specrage] {name} seed={seed} n_train={len(split.y_train)} "
                      f"views={len(split.train)}", flush=True)
                try:
                    row = run_cell(SpecRaGE, split, seed)
                except Exception as exc:                        # noqa: BLE001
                    row = {"dataset": name, "seed": seed, "method": "specrage",
                           "error": f"{type(exc).__name__}: {exc}"}
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
