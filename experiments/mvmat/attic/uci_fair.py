"""Isolate the UCI crash: run each fair-baseline piece with flushing + gc, so we see the exact
step that dies. best-of-8 (random) first (answers the gated-vs-fair question), MDT-Cst in try/except."""
import gc, traceback
import numpy as np
from scipy.spatial.distance import pdist
from experiments.mvmat.t_sweep import load, ami_of
from src.mdt_operators import mdt_operator_from_views

Xs, y, k = load('UCI', 1500); n = len(y); knn = int(np.log(n)) + 1
sig = [np.quantile(pdist(v[:200]), 0.5) for v in Xs]
print(f"UCI n={n} k={k} knn={knn} views={len(Xs)} dims={[v.shape[1] for v in Xs]}", flush=True)

best = []
for s in [0, 1, 2, 3, 4]:
    cand = []
    for j in range(8):
        W = mdt_operator_from_views(Xs, sig, 4, 'random', knn, seed=s * 100 + j)
        cand.append(ami_of(W, k, y)); del W; gc.collect()
    best.append(max(cand))
    print(f"  seed {s}: best-of-8={max(cand):.3f} mean={np.mean(cand):.3f}", flush=True)
print(f"UCI best-of-8(fair) = {np.mean(best):.3f}+/-{np.std(best):.3f}", flush=True)

try:
    cst = [ami_of(mdt_operator_from_views(Xs, sig, 4, 'contrastive', knn, seed=s), k, y) for s in [0, 1, 2]]
    print(f"UCI MDT-Cst(fair) = {np.mean(cst):.3f}+/-{np.std(cst):.3f}", flush=True)
except Exception:
    print("MDT-Cst FAILED:", flush=True); traceback.print_exc()
