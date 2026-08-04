"""The honest ceiling. knn and normalization are high-impact, dataset-dependent, and
COUPLED -- but tuning them to AMI leaks labels. So select (knn, norm) by an UNSUPERVISED
criterion (silhouette of the consensus embedding) and check it lands near the AMI oracle.
If silhouette-selected ~= oracle, the gain is real and deployable; if not, it is
unreachable without the answer key. Fixed: local-scaling bandwidth, quality-weighted
consensus, M=30, t=knee (both already shown to be the right settings / flat knobs)."""
import numpy as np, sys
from functools import reduce
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI, silhouette_score
sys.path.insert(0, '/home/nicolasbigeard/internshipborelli/deep-diffusion-maps/experiments/mvmat')
from enhance_operator import kernel, consensus_op, view_quality, load, SEEDS
from reproduce_paper import get_embedding, knee_power

R = 10


def eval_cfg(Xv, knn, norm, qw, t, k, y):
    P = [kernel(v, 'local', norm, knn) for v in Xv]
    w = view_quality(P, k) if qw else None
    amis, sils = [], []
    for s in SEEDS:
        E = get_embedding(consensus_op(P, t, 30, s, w), k)
        labs = [KMeans(k, n_init=1, random_state=r).fit_predict(E) for r in range(R)]
        amis.append(np.mean([AMI(y, l) for l in labs]))
        l0 = labs[0]; sils.append(silhouette_score(E, l0) if len(set(l0)) > 1 else -1.0)
    return float(np.mean(amis)), float(np.mean(sils))


def main():
    names = sys.argv[1:] or ['MSRC-v5', '100Leaves', 'Caltech101-7', 'Handwritten', 'UCI']
    for name in names:
        Xv, y, k = load(name); n = len(y); base = max(2, int(np.floor(np.log(n))))
        Pdef = [kernel(v, 'local', 'row') for v in Xv]
        t = max(knee_power(reduce(lambda a, b: a @ b, Pdef)), 1)
        knns = sorted({base, base + 5, 2 * base, 18})
        print(f"\n=== {name}  n={n} k={k}  t={t}  knn grid={knns} ===", flush=True)
        grid = []
        for knn in knns:
            for norm in ('row', 'sym'):
                for qw in (False, True):
                    ami, sil = eval_cfg(Xv, knn, norm, qw, t, k, y)
                    tag = 'qw' if qw else 'un'
                    grid.append((knn, norm, tag, ami, sil))
                    print(f"  knn={knn:3d} {norm:3s} {tag}  AMI {ami:.3f}  silhouette {sil:+.3f}", flush=True)
        pick = max(grid, key=lambda g: g[4]); oracle = max(grid, key=lambda g: g[3])
        gap = oracle[3] - pick[3]
        print(f"  -> SELECTED by silhouette: knn={pick[0]} {pick[1]} {pick[2]}  AMI {pick[3]:.3f}", flush=True)
        print(f"  -> ORACLE (max AMI):       knn={oracle[0]} {oracle[1]} {oracle[2]}  AMI {oracle[3]:.3f}"
              f"   leakage gap {gap:+.3f}", flush=True)


if __name__ == '__main__':
    main()
