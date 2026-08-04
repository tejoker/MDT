"""CORRECTED operator enhancement on the CANONICAL paper kernel (get_kernel_matrix:
kNN graph WITH self-loops, max-of-row-min bandwidth, symmetrise, row-normalise). The
earlier enhance_operator.kernel dropped the self-loop (diagonal=1 lazy-walk term), giving
a strawman baseline (MSRC consensus 0.592 vs the true 0.684). Here baseline and every
enhanced variant share kernel2(), and the silhouette selector ranges over the FULL grid
INCLUDING the paper baseline (global bw) -- so a reliable selector can never regress below
it. Reports: paper baseline, silhouette-selected (deployable), AMI oracle, leakage gap."""
import numpy as np, sys
from functools import reduce
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI, silhouette_score
sys.path.insert(0, '/home/nicolasbigeard/internshipborelli/deep-diffusion-maps/experiments/mvmat')
from enhance_operator import consensus_op, view_quality, load, SEEDS
from reproduce_paper import get_embedding, ami_runs, knee_power

R = 10


def kernel2(x, scaling='global', norm='row', knn=None):
    """Canonical paper kernel (self-loops kept), with local-scaling and sym-norm options."""
    D = squareform(pdist(x, 'euclidean')); n = len(x)
    if knn is None:
        knn = int(np.floor(np.log(n)))
    K = NearestNeighbors(n_neighbors=knn, metric='precomputed').fit(D).kneighbors_graph(D, mode='distance').tocsr()
    if scaling == 'global':
        bw = np.max([row[row > 0].min() for row in D])
        K.data = np.exp(-K.data ** 2 / bw)            # self entries (dist 0) -> exp(0)=1
    else:                                             # local: sigma_i = k-th NN distance (max kept)
        sig = np.array([K.data[K.indptr[i]:K.indptr[i + 1]].max() if K.indptr[i + 1] > K.indptr[i] else 1.0
                        for i in range(n)])
        sig[sig <= 0] = 1.0
        C = K.tocoo(); C.data = np.exp(-C.data ** 2 / (sig[C.row] * sig[C.col])); K = C.tocsr()
    K = (K + K.T) / 2
    A = np.asarray(K.todense())
    if norm == 'row':
        rs = A.sum(1); rs[rs == 0] = 1; return A / rs[:, None]
    d = np.sqrt(A.sum(1)); d[d == 0] = 1; return A / np.outer(d, d)


def eval_cfg(Xv, scaling, knn, norm, qw, t, k, y):
    P = [kernel2(v, scaling, norm, knn) for v in Xv]
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
        Xv, y, k = load(name); n = len(y); base = int(np.floor(np.log(n)))
        Pdef = [kernel2(v, 'global', 'row') for v in Xv]
        t = max(knee_power(reduce(lambda a, b: a @ b, Pdef)), 1)
        knns = sorted({base, base + 5, max(2 * base, 14)})
        print(f"\n=== {name}  n={n} k={k} t={t}  base_knn={base} grid={knns} ===", flush=True)
        # paper baseline (this MUST match reproduce_paper / recon ~0.684 etc.)
        b_ami, b_sil = eval_cfg(Xv, 'global', base, 'row', False, t, k, y)
        print(f"  paper baseline (global,row,knn={base},uniform)  AMI {b_ami:.3f}  sil {b_sil:+.3f}", flush=True)
        grid = [(base, 'global', 'row', False, b_ami, b_sil)]
        for sc in ('global', 'local'):
            for knn in knns:
                for norm in ('row', 'sym'):
                    for qw in (False, True):
                        if sc == 'global' and knn == base and norm == 'row' and not qw:
                            continue                       # already have the baseline
                        a, s = eval_cfg(Xv, sc, knn, norm, qw, t, k, y)
                        grid.append((knn, sc, norm, qw, a, s))
                        tag = f"{sc[:3]} knn={knn} {norm} {'qw' if qw else 'un'}"
                        print(f"  {tag:24s}  AMI {a:.3f}  sil {s:+.3f}", flush=True)
        pick = max(grid, key=lambda g: g[5]); oracle = max(grid, key=lambda g: g[4])
        gap = oracle[4] - pick[4]; delta = pick[4] - b_ami
        print(f"  -> SELECTED(sil): {pick[1]} knn={pick[0]} {pick[2]} {'qw' if pick[3] else 'un'}"
              f"  AMI {pick[4]:.3f}   gain vs baseline {delta:+.3f}", flush=True)
        print(f"  -> ORACLE: AMI {oracle[4]:.3f}   leakage gap {gap:+.3f}", flush=True)


if __name__ == '__main__':
    main()
