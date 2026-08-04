"""Full multi-seed t-sweep (t=1..4) so the whole table is internally consistent (all mean+-std).
Baselines (random, best-of-8) and gated already multi-seeded in final_table.py; this fills the
learned t=2,3,4 columns at 5 seeds. Sequential, one dataset at a time (avoid OOM)."""
import gc
import numpy as np
from experiments.mvmat.t_sweep import load, train

SEEDS = [0, 1, 2, 3, 4]


def st(x):
    a = np.array(x); return f"{a.mean():.3f}+/-{a.std():.3f}"


def main():
    for name, N in [('MSRC-v5', None), ('Caltech101-7', None), ('OutdoorScene', 1500), ('UCI', 1500)]:
        Xs, y, k = load(name, N); n = len(y)
        cols = []
        for t in (1, 2, 3, 4):
            vals = [train(Xs, y, k, t, seed=s)['amiN'] for s in SEEDS]
            cols.append(f"t{t}={st(vals)}")
            gc.collect()
        print(f"{name:12s} n={n} k={k} | " + " | ".join(cols), flush=True)
        gc.collect()


if __name__ == '__main__':
    main()
