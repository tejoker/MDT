"""Multi-seed error bars on the two configs that mattered: short product (t=2) and gated
(lam=1.0), vs fixed+random, on the datasets where learned looked competitive (UCI win, MSRC).
5 seeds vary model init + trajectory; data subsample fixed (seed 0). Answers: is the UCI
'learned beats fixed MDT' real or single-seed noise?"""
import numpy as np
from scipy.spatial.distance import pdist
from experiments.mvmat.t_sweep import load, train, ami_of
from src.mdt_operators import mdt_operator_from_views

SEEDS = [0, 1, 2, 3, 4]


def stats(xs):
    a = np.array(xs)
    return f"{a.mean():.3f}+/-{a.std():.3f}"


def main():
    for name, N in [('MSRC-v5', None), ('UCI', 1500)]:
        Xs, y, k = load(name, N); n = len(y); knn = int(np.log(n)) + 1
        sig = [np.quantile(pdist(v[:200]), 0.5) for v in Xs]
        ref = [ami_of(mdt_operator_from_views(Xs, sig, 4, 'random', knn, seed=s), k, y) for s in SEEDS]
        lt2 = [train(Xs, y, k, 2, seed=s)['amiN'] for s in SEEDS]
        glam = [train(Xs, y, k, 4, gate=True, lam=1.0, seed=s)['amiN'] for s in SEEDS]
        print(f"{name:10s} n={n} k={k} | fixed+random={stats(ref)} | learned t=2={stats(lt2)} | "
              f"gated lam=1={stats(glam)}", flush=True)


if __name__ == '__main__':
    main()
