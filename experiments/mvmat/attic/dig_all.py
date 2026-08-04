"""Three digs into DDM/MDT.
 DIG 1: is the spectral gap the master variable? gap vs whether bigger embedding-dim hurts.
 DIG 2: can an UNSUPERVISED per-view quality proxy (silhouette) down-weight the clean-but-useless view
        so quality-weighted fusion beats uniform on heterogeneous data? (or is the failure fundamental?)
 DIG 3: where does DDM's sqrt(pi) weighting earn its keep? pi non-uniformity vs (pi-weighted - raw) AMI."""
import numpy as np, scipy.io, sys
from functools import reduce
from numpy.linalg import svd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
sys.path.insert(0, '/home/nicolasbigeard/internshipborelli/deep-diffusion-maps/experiments/mvmat')
from corrected_enhance import kernel2
from reproduce_paper import get_embedding, ami_runs

CAP = 800
D1 = ['MSRC-v5', 'Yale', 'BBCSport', 'Caltech101-7', 'Handwritten', 'UCI',
      'Wikipedia-test', 'NUS-WIDE', 'Cora', 'CiteSeer', 'Prokaryotic', 'OutdoorScene']
D2 = ['MSRC-v5', 'Handwritten', 'Wikipedia-test', 'Cora', 'NUS-WIDE', 'Prokaryotic', 'CiteSeer']


def load(name):
    m = scipy.io.loadmat(f'/tmp/Multi-view-datasets/{name}.mat')
    Xv = [np.asarray(v).astype(float) for v in m['X'].ravel()]; y = np.asarray(m['y']).ravel()
    if len(y) > CAP:
        idx = np.random.default_rng(0).choice(len(y), CAP, replace=False)
        Xv = [v[idx] for v in Xv]; y = y[idx]
    return Xv, y, len(np.unique(y))


def traj(P, t=4):
    seq = np.random.default_rng(0).integers(0, len(P), t)
    return reduce(lambda a, b: b @ a, [P[i] for i in seq])


def stat(W, it=200):
    n = len(W); v = np.ones(n) / n; Wt = W.T
    for _ in range(it):
        v = Wt @ v; s = v.sum(); v = v / (s if s else 1.0)
    return np.clip(v, 1e-12, None)


def dig1():
    print("### DIG 1 — spectral gap vs dim-effect (does a sharp gap predict that bigger d is safe?)", flush=True)
    print(f"{'dataset':13s} {'k':>3s} {'gap sk/sk+1':>12s} {'energy':>7s} {'AMI@k':>6s} {'AMI@4k':>7s} {'dim_eff':>8s}", flush=True)
    gaps, effs = [], []
    for name in D1:
        try:
            Xv, y, k = load(name); P = [kernel2(v, 'global', 'row') for v in Xv]
            W = traj(P); U, s, _ = svd(W)
            gap = s[k] / max(s[k + 1], 1e-12)
            energy = (s[1:k + 1] ** 2).sum() / max((s ** 2).sum() - s[0] ** 2, 1e-12)
            d2 = min(4 * k, len(y) - 2)
            a1 = ami_runs(get_embedding(W, k), y, k, 3); a2 = ami_runs(get_embedding(W, d2), y, k, 3)
            gaps.append(gap); effs.append(a2 - a1)
            print(f"{name:13s} {k:3d} {gap:12.3f} {energy:7.3f} {a1:6.3f} {a2:7.3f} {a2-a1:+8.3f}", flush=True)
        except Exception as e:
            print(f"{name:13s} ERR {e}", flush=True)
    print(f"-> corr(gap, dim_effect) = {spearmanr(gaps, effs).correlation:+.2f}  "
          f"(positive => sharper gap, bigger d less harmful). gaps range {min(gaps):.2f}-{max(gaps):.2f}\n", flush=True)


def dig2():
    print("### DIG 2 — view-quality-aware fusion: can silhouette down-weight the useless view?", flush=True)
    for name in D2:
        try:
            Xv, y, k = load(name); P = [kernel2(v, 'global', 'row') for v in Xv]
            sils, amis = [], []
            for Pv in P:
                E = get_embedding(Pv, k); lab = KMeans(k, n_init=3, random_state=0).fit_predict(E)
                sils.append(silhouette_score(E, lab) if len(set(lab)) > 1 else -1.0)
                amis.append(ami_runs(E, y, k, 3))
            sils = np.array(sils); w = np.exp((sils - sils.max()) / 0.1); w /= w.sum()
            au = ami_runs(get_embedding(sum(P) / len(P), k), y, k, 3)
            aq = ami_runs(get_embedding(sum(wi * Pi for wi, Pi in zip(w, P)), k), y, k, 3)
            pv = "  ".join(f"sil{s:+.2f}/AMI{a:.2f}" for s, a in zip(sils, amis))
            print(f"{name:13s} [{pv}]", flush=True)
            print(f"{'':13s}  uniform {au:.3f} | qual-weighted {aq:.3f} | best-single {max(amis):.3f}"
                  f"  ({'QW HELPS' if aq > au + 0.02 else 'no gain'})", flush=True)
        except Exception as e:
            print(f"{name:13s} ERR {e}", flush=True)
    print("-> silhouette-vs-AMI per view: if the useless (low-AMI) view also has LOW silhouette, the proxy",
          "\n   sees it and QW helps; if the useless view has HIGH silhouette (clean-but-useless), no unsup proxy sees it.\n", flush=True)


def dig3():
    print("### DIG 3 — where does sqrt(pi) weighting earn its keep? (pi non-uniformity vs pi-weighted - raw)", flush=True)
    print(f"{'dataset':13s} {'pi_cv':>7s} {'raw AMI':>8s} {'pi-wt AMI':>10s} {'delta':>7s}", flush=True)
    cvs, deltas = [], []
    for name in D1:
        try:
            Xv, y, k = load(name); P = [kernel2(v, 'global', 'row') for v in Xv]
            W = traj(P); pi = stat(W); cv = float(pi.std() / pi.mean())
            raw = ami_runs(get_embedding(W, k), y, k, 3)
            sq = np.sqrt(pi); A = sq[:, None] * W * (1.0 / sq)[None, :]
            G = A @ A.T - np.outer(sq, sq)
            wv, Q = np.linalg.eigh(G); idx = np.argsort(wv)[::-1][:k]
            F = (1.0 / sq)[:, None] * Q[:, idx] * np.sqrt(np.clip(wv[idx], 0, None))
            piw = ami_runs(F, y, k, 3)
            cvs.append(cv); deltas.append(piw - raw)
            print(f"{name:13s} {cv:7.2f} {raw:8.3f} {piw:10.3f} {piw-raw:+7.3f}", flush=True)
        except Exception as e:
            print(f"{name:13s} ERR {e}", flush=True)
    print(f"-> corr(pi_cv, |delta|) = {spearmanr(cvs, np.abs(deltas)).correlation:+.2f}  "
          f"(positive => pi-weighting matters more where pi is non-uniform)", flush=True)


if __name__ == '__main__':
    dig1(); dig2(); dig3()
