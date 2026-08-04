"""Can we reproduce the MDT paper's clustering AMI? Replicates the paper's EXACT
protocol (from the MDT repo): kernel = kNN graph with knn=floor(log N), bandwidth =
max over rows of the min nonzero distance, exp(-d^2/bw), symmetrised, row-normalised;
embedding = truncated svds dropping the top component; AMI averaged over R runs of
KMeans into #clusters; FULL dataset. Methods: AD (t=1 product), MVD (block op on
unnormalised kernels), ID (per-view spectral-entropy knee power), MDT-Rand (random
length-t trajectory, t from the singular-entropy knee)."""
import numpy as np
import scipy.io
import scipy.sparse as sp
from functools import reduce
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI
from kneed import KneeLocator


def get_kernel_matrix(x, normalize=True):                       # verbatim port of MDT repo
    D = squareform(pdist(x, "euclidean"))
    bandwidth = np.max([row[row > 0].min() for row in D])
    knn = int(np.floor(np.log(len(x))))
    nbrs = NearestNeighbors(n_neighbors=knn, metric="precomputed").fit(D)
    K = nbrs.kneighbors_graph(D, mode="distance")
    K.data = np.exp(-K.data ** 2 / bandwidth)
    K = (K + K.T) / 2
    if normalize:
        rs = np.asarray(K.sum(1)).ravel(); rs[rs == 0] = 1
        K = sp.diags(1.0 / rs) @ K
    return np.asarray(K.todense())


def get_embedding(P, k):                                        # MDT repo svds embedding
    from scipy.sparse.linalg import svds
    try:
        U, s, _ = svds(P, k=k + 1); U, s = U[:, ::-1], s[::-1]
    except Exception:
        U, s, _ = np.linalg.svd(P, full_matrices=False)
    return U[:, 1:k + 1] * s[1:k + 1]


def spectral_entropy(vals):
    v = np.abs(vals); v = v / (v.sum() + 1e-12); v = v[v > 0]
    return float(-(v * np.log(v)).sum())


def knee_power(P, max_t=25, eigen=False):
    rng = range(1, max_t)
    if eigen:
        ev = np.linalg.eigvals(P)
        ents = [spectral_entropy(np.power(ev, t)) for t in rng]
    else:
        ents = [spectral_entropy(np.linalg.svd(np.linalg.matrix_power(P, t), compute_uv=False)) for t in rng]
    kn = KneeLocator(list(rng), ents, curve="convex", direction="decreasing")
    return kn.elbow or 1


def ami_runs(E, y, k, R=20):
    return float(np.mean([AMI(y, KMeans(k, n_init=1, random_state=r).fit_predict(E)) for r in range(R)]))


def MVD(Kun):
    n = len(Kun); m = Kun[0].shape[0]; op = np.zeros((n * m, n * m))
    for i in range(n):
        for j in range(i + 1, n):
            b = Kun[i] @ Kun[j]
            op[i * m:(i + 1) * m, j * m:(j + 1) * m] = b; op[j * m:(j + 1) * m, i * m:(i + 1) * m] = b.T
    rs = op.sum(1, keepdims=True); rs[rs == 0] = 1
    return op / rs


def main():
    rng = np.random.default_rng(0)
    for name in ['MSRC-v5', '100Leaves', 'Caltech101-7']:
        m = scipy.io.loadmat(f'/tmp/Multi-view-datasets/{name}.mat'); Xo = m['X'].ravel()
        y = np.asarray(m['y'] if 'y' in m else m['Y']).ravel(); k = len(np.unique(y))
        Xv = [np.asarray(v).astype(float) for v in Xo]
        P = [get_kernel_matrix(v, normalize=True) for v in Xv]
        Kun = [get_kernel_matrix(v, normalize=False) for v in Xv]

        ad = ami_runs(get_embedding(reduce(lambda a, b: a @ b, P), k), y, k)
        idd = ami_runs(get_embedding(reduce(lambda a, b: a @ b, [np.linalg.matrix_power(p, knee_power(p, eigen=True)) for p in P]), k), y, k)
        mvd = ami_runs(get_embedding(MVD(Kun), k)[:len(y)], y, k)
        # MDT-Rand: random length-t one-hot trajectory, t from knee of the product
        t = max(knee_power(reduce(lambda a, b: a @ b, P)), 1)
        seq = rng.integers(0, len(P), t)
        W = reduce(lambda a, b: b @ a, [P[i] for i in seq])
        mr = ami_runs(get_embedding(W, k), y, k)
        # MDT-consensus under the SAME paper protocol: average of M random trajectory operators
        ops = []
        for _ in range(15):
            s = rng.integers(0, len(P), t)
            ops.append(reduce(lambda a, b: b @ a, [P[i] for i in s]))
        cons = ami_runs(get_embedding(np.mean(ops, 0), k), y, k)
        print(f"{name:13s} (n={len(y)},k={k}) | AD {ad:.3f} | ID {idd:.3f} | MVD {mvd:.3f} | "
              f"MDT-Rand {mr:.3f} | MDT-consensus {cons:.3f}", flush=True)


if __name__ == '__main__':
    main()
