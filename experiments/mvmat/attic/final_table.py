"""One clean table replacing Tables 7+8. All 4 collapse datasets, 5 seeds.
- t-sweep now includes t=1 (does minimal diffusion win all the way?).
- learned column = best t over {1,2,3,4}, multi-seed.
- fair best-of-8 baseline + weak random + gated (lam=1), all multi-seed.
Runs SEQUENTIALLY (one dataset at a time) to avoid the OOM that killed concurrent runs."""
import gc
import numpy as np
from scipy.spatial.distance import pdist
from experiments.mvmat.t_sweep import load, train, ami_of
from src.mdt_operators import mdt_operator_from_views

SEEDS = [0, 1, 2, 3, 4]


def st(x):
    a = np.array(x); return f"{a.mean():.3f}+/-{a.std():.3f}"


def main():
    for name, N in [('MSRC-v5', None), ('Caltech101-7', None), ('OutdoorScene', 1500), ('UCI', 1500)]:
        Xs, y, k = load(name, N); n = len(y); knn = int(np.log(n)) + 1
        sig = [np.quantile(pdist(v[:200]), 0.5) for v in Xs]

        rand = [ami_of(mdt_operator_from_views(Xs, sig, 4, 'random', knn, seed=s), k, y) for s in SEEDS]
        best8 = []
        for s in SEEDS:
            cand = [ami_of(mdt_operator_from_views(Xs, sig, 4, 'random', knn, seed=s * 100 + j), k, y)
                    for j in range(8)]
            best8.append(max(cand)); gc.collect()

        lt = {t: train(Xs, y, k, t, seed=0)['amiN'] for t in (1, 2, 3, 4)}
        bt = max(lt, key=lt.get)
        learned = [train(Xs, y, k, bt, seed=s)['amiN'] for s in SEEDS]
        gated = [train(Xs, y, k, 4, gate=True, lam=1.0, seed=s)['amiN'] for s in SEEDS]

        print(f"\n=== {name} n={n} k={k} ===", flush=True)
        print(f"  t-sweep(seed0): t1={lt[1]:.3f} t2={lt[2]:.3f} t3={lt[3]:.3f} t4={lt[4]:.3f}  -> best t={bt}", flush=True)
        print(f"  random(weak)={st(rand)} | best-of-8(fair)={st(best8)} | "
              f"learned(t{bt})={st(learned)} | gated(l1)={st(gated)}", flush=True)
        gc.collect()


if __name__ == '__main__':
    main()
