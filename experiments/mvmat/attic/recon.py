"""Reconcile S4 (single-seed) with S5 (multi-seed). Does the paper-protocol MDT-consensus
actually LEAD the deterministic AD/ID/MVD baselines once consensus is multi-seeded? AD/ID/MVD
are deterministic operators (stable, 20-KMeans-averaged); only the MDT trajectory entries are
seed-dependent. Print consensus mean+/-std over 5 seeds next to AD/ID/MVD and the S5 enhanced
number, and flag where consensus does NOT beat AD."""
import numpy as np, scipy.io
from functools import reduce
from reproduce_paper import get_kernel_matrix, get_embedding, ami_runs, knee_power, MVD

DATA = '/tmp/Multi-view-datasets'
SEEDS = range(5)
ENH = {'MSRC-v5': 0.768, '100Leaves': 0.895, 'Caltech101-7': 0.578}   # S5 enhanced (oracle/sel)


def main():
    for name in ['MSRC-v5', '100Leaves', 'Caltech101-7']:
        m = scipy.io.loadmat(f'{DATA}/{name}.mat'); Xo = m['X'].ravel()
        y = np.asarray(m['y'] if 'y' in m else m['Y']).ravel(); k = len(np.unique(y))
        Xv = [np.asarray(v).astype(float) for v in Xo]
        P = [get_kernel_matrix(v, True) for v in Xv]
        Kun = [get_kernel_matrix(v, False) for v in Xv]
        prod = reduce(lambda a, b: a @ b, P)
        ad = ami_runs(get_embedding(prod, k), y, k)
        idd = ami_runs(get_embedding(reduce(lambda a, b: a @ b,
              [np.linalg.matrix_power(p, knee_power(p, eigen=True)) for p in P]), k), y, k)
        mvd = ami_runs(get_embedding(MVD(Kun), k)[:len(y)], y, k)
        t = max(knee_power(prod), 1)
        cons = []
        for s in SEEDS:
            rng = np.random.default_rng(s); ops = []
            for _ in range(15):
                seq = rng.integers(0, len(P), t)
                ops.append(reduce(lambda a, b: b @ a, [P[i] for i in seq]))
            cons.append(ami_runs(get_embedding(np.mean(ops, 0), k), y, k))
        cm, cs = float(np.mean(cons)), float(np.std(cons))
        best_baseline = max(ad, idd, mvd)
        leads = "LEADS" if cm - cs > best_baseline else ("ties" if cm + cs > best_baseline else "BELOW")
        print(f"{name:13s} k={k} t={t} | AD {ad:.3f} ID {idd:.3f} MVD {mvd:.3f} | "
              f"consensus {cm:.3f}+/-{cs:.3f} [{leads} best-baseline {best_baseline:.3f}] | "
              f"S5-enhanced {ENH[name]:.3f}", flush=True)


if __name__ == '__main__':
    main()
