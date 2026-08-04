"""Is the UCI 'gated beats fixed' win real, or just beating the weak random-trajectory strawman?
Compare gated (0.865 UCI / 0.518 MSRC, from t_sweep_ms) against FAIR classical operators in the
SAME harness: single random trajectory (weak), best-of-8 random (fair), and MDT-Cst learned scalar
weights (fair). 5 seeds. If gated < best-of-8 or < MDT-Cst, the 'win' was a strawman artifact."""
import numpy as np
from scipy.spatial.distance import pdist
from experiments.mvmat.t_sweep import load, ami_of
from src.mdt_operators import mdt_operator_from_views

SEEDS = [0, 1, 2, 3, 4]


def st(x):
    a = np.array(x); return f"{a.mean():.3f}+/-{a.std():.3f}"


def main():
    for name, N in [('MSRC-v5', None), ('UCI', 1500)]:
        Xs, y, k = load(name, N); n = len(y); knn = int(np.log(n)) + 1
        sig = [np.quantile(pdist(v[:200]), 0.5) for v in Xs]
        rand = [ami_of(mdt_operator_from_views(Xs, sig, 4, 'random', knn, seed=s), k, y) for s in SEEDS]
        best8 = []
        for s in SEEDS:
            cand = [ami_of(mdt_operator_from_views(Xs, sig, 4, 'random', knn, seed=s * 100 + j), k, y)
                    for j in range(8)]
            best8.append(max(cand))
        cst = [ami_of(mdt_operator_from_views(Xs, sig, 4, 'contrastive', knn, seed=s), k, y) for s in SEEDS]
        print(f"{name:10s} n={n} k={k} | random(weak)={st(rand)} | best-of-8(fair)={st(best8)} | "
              f"MDT-Cst(fair)={st(cst)}", flush=True)


if __name__ == '__main__':
    main()
