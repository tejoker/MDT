"""Push the MDT operator W to its ceiling. Ablates the four operator levers on the
EXACT paper protocol (reproduce_paper.py helpers): kernel bandwidth (global max-of-
row-min-dist vs Zelnik-Manor local self-tuning), normalization (row-stochastic D^-1 K
vs symmetric D^-1/2 K D^-1/2), view fusion (uniform-random trajectory vs quality-
weighted), consensus size M. Stochastic configs run over SEEDS seeds -> mean +/- std
so a gain has to clear the noise floor (the single-seed lesson). Reports AMI over R
KMeans runs per config, finds the best combo per dataset, then sweeps M to show the
ceiling is flat."""
import numpy as np, scipy.io, sys
from functools import reduce
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI, silhouette_score

sys.path.insert(0, '/home/nicolasbigeard/internshipborelli/deep-diffusion-maps/experiments/mvmat')
from reproduce_paper import get_embedding, ami_runs, knee_power  # paper protocol verbatim

DATA = '/tmp/Multi-view-datasets'
SEEDS = [0, 1, 2]
R = 12


def kernel(x, scaling='global', norm='row', knn=None):
    """Per-view transition. scaling: global (paper) | local (Zelnik-Manor). norm: row | sym."""
    D = squareform(pdist(x, 'euclidean')); n = len(x)
    if knn is None:
        knn = max(2, int(np.floor(np.log(n))))
    G = NearestNeighbors(n_neighbors=knn, metric='precomputed').fit(D).kneighbors_graph(D, mode='distance').toarray()
    mask = G > 0
    if scaling == 'global':
        bw = np.max([row[row > 0].min() for row in D])
        K = np.where(mask, np.exp(-G ** 2 / bw), 0.0)
    else:                                              # local self-tuning: sigma_i = k-th NN dist
        sig = np.array([row[row > 0].max() if mask[i].any() else 1.0 for i, row in enumerate(G)])
        sig[sig == 0] = 1.0
        K = np.where(mask, np.exp(-G ** 2 / np.outer(sig, sig)), 0.0)
    K = (K + K.T) / 2
    if norm == 'row':
        rs = K.sum(1); rs[rs == 0] = 1; return K / rs[:, None]
    d = np.sqrt(K.sum(1)); d[d == 0] = 1; return K / np.outer(d, d)   # symmetric


def consensus_op(P, t, M, seed, w=None):
    """Average over M random length-t trajectory products. w: view-sampling probs (None=uniform)."""
    rng = np.random.default_rng(seed); V = len(P); ops = []
    for _ in range(M):
        s = rng.choice(V, size=t, p=w)
        ops.append(reduce(lambda a, b: b @ a, [P[i] for i in s]))
    return np.mean(ops, 0)


def view_quality(P, k):
    """Per-view single-view embedding silhouette -> softmax sampling weights."""
    q = []
    for p in P:
        E = get_embedding(p, k); lab = KMeans(k, n_init=3, random_state=0).fit_predict(E)
        q.append(silhouette_score(E, lab) if len(set(lab)) > 1 else -1.0)
    q = np.array(q); e = np.exp((q - q.max()) / 0.1); return e / e.sum()


def stoch(name, P, t, k, y, M, w=None):
    """Mean +/- std AMI of a consensus config over SEEDS."""
    a = [ami_runs(get_embedding(consensus_op(P, t, M, s, w), k), y, k, R) for s in SEEDS]
    return name, float(np.mean(a)), float(np.std(a))


def load(name):
    m = scipy.io.loadmat(f'{DATA}/{name}.mat')
    Xo = m['X'].ravel(); y = np.asarray(m['y'] if 'y' in m else m['Y']).ravel()
    Xv = [np.asarray(v).astype(float) for v in Xo]
    return Xv, y, len(np.unique(y))


def main():
    names = sys.argv[1:] or ['MSRC-v5', '100Leaves', 'Caltech101-7', 'Handwritten', 'UCI']
    for name in names:
        Xv, y, k = load(name)
        # build the four kernel families once
        P = {(sc, nm): [kernel(v, sc, nm) for v in Xv] for sc in ('global', 'local') for nm in ('row', 'sym')}
        t = max(knee_power(reduce(lambda a, b: a @ b, P[('global', 'row')])), 1)
        w = view_quality(P[('local', 'sym')], k)
        print(f"\n=== {name}  n={len(y)} k={k} views={len(Xv)} t={t}  qweights={np.round(w,2)} ===", flush=True)

        # AD reference (product of all views) under baseline vs best kernel
        for tag, key in [('AD global+row', ('global', 'row')), ('AD local+sym', ('local', 'sym'))]:
            ad = ami_runs(get_embedding(reduce(lambda a, b: a @ b, P[key]), k), y, k, R)
            print(f"  {tag:22s} AMI {ad:.3f}", flush=True)

        # consensus ablation (stochastic -> mean+/-std)
        rows = [
            stoch('B  global+row  uniform', P[('global', 'row')], t, k, y, 15),
            stoch('E1 local +row  uniform', P[('local', 'row')], t, k, y, 15),
            stoch('E2 global+sym  uniform', P[('global', 'sym')], t, k, y, 15),
            stoch('E3 local +sym  uniform', P[('local', 'sym')], t, k, y, 15),
            stoch('E4 local +sym  qweight', P[('local', 'sym')], t, k, y, 15, w),
        ]
        for nm, mu, sd in rows:
            print(f"  {nm:26s} AMI {mu:.3f} +/- {sd:.3f}", flush=True)

        # ceiling: best uniform kernel family, sweep M to show plateau
        best = max(rows[:4], key=lambda r: r[1])
        bestkey = {'B  global+row  uniform': ('global', 'row'), 'E1 local +row  uniform': ('local', 'row'),
                   'E2 global+sym  uniform': ('global', 'sym'), 'E3 local +sym  uniform': ('local', 'sym')}[best[0]]
        print(f"  -- ceiling sweep on {best[0].split()[1]}+{best[0].split()[2]} --", flush=True)
        for M in (15, 30, 60, 120):
            nm, mu, sd = stoch(f'M={M}', P[bestkey], t, k, y, M)
            print(f"     M={M:4d}  AMI {mu:.3f} +/- {sd:.3f}", flush=True)


if __name__ == '__main__':
    main()
