"""Gap #1: multi-seed the paper's only positive findings.
(a) consensus vs single trajectory (is the gain > noise?);
(b) silhouette vs Calinski-Harabasz as a selection criterion (is rho gap robust?)."""
import numpy as np
import scipy.io
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import silhouette_score as SIL, calinski_harabasz_score as CH
from src.mdt_operators import transition_matrix, _mdt_operator, _trajectory


def load(name, seed, Ncap=800):
    m = scipy.io.loadmat(f'/tmp/Multi-view-datasets/{name}.mat'); Xo = m['X'].ravel()
    y = np.asarray(m['y'] if 'y' in m else m['Y']).ravel(); k = len(np.unique(y))
    N = min(len(y), Ncap)
    idx = np.random.default_rng(seed).choice(len(y), N, replace=False)
    Xs = [np.nan_to_num(StandardScaler().fit_transform(np.asarray(v)[idx].astype(float))).astype('float32') for v in Xo]
    return Xs, y[idx], k


def gram(W):
    n = len(W); p = np.ones(n) / n
    for _ in range(200):
        p = p @ W; p /= p.sum()
    sp = np.sqrt(np.maximum(p, 1e-12)); A = sp[:, None] * W / sp[None, :]
    return A @ A.T - np.outer(sp, sp)


def emb(T, k):
    w, V = np.linalg.eigh((T + T.T) / 2); idx = np.argsort(w)[::-1][:k]
    return V[:, idx] * np.sqrt(np.clip(w[idx], 0, None))


def km(E, k):
    return KMeans(k, n_init=8, random_state=0).fit_predict(E)


def f(x):
    return float(np.nanmean(x)), float(np.nanstd(x))


def consensus(name, M=15, seeds=5, knn=7):
    sm, cm = [], []
    for seed in range(seeds):
        Xs, y, k = load(name, seed)
        sig = [np.quantile(pdist(v[:200]), 0.5) for v in Xs]
        P = [transition_matrix(v, s, knn) for v, s in zip(Xs, sig)]
        Ts = [gram(_mdt_operator(_trajectory(len(P), 4, 'random', seed * 100 + m), P)) for m in range(M)]
        sm.append(np.mean([AMI(y, km(emb(T, k), k)) for T in Ts]))
        cm.append(AMI(y, km(emb(np.mean(Ts, 0), k), k)))
    return f(sm), f(cm)


def criterion(names, M=30, seeds=2, knn=7):
    rc, rs = [], []
    for name in names:
        for seed in range(seeds):
            Xs, y, k = load(name, seed)
            sig = [np.quantile(pdist(v[:200]), 0.5) for v in Xs]
            P = [transition_matrix(v, s, knn) for v, s in zip(Xs, sig)]
            ch, sil, ami = [], [], []
            for m in range(M):
                E = emb(gram(_mdt_operator(_trajectory(len(P), 4, 'random', seed * 1000 + m), P)), k)
                lab = km(E, k); ch.append(CH(E, lab)); sil.append(SIL(E, lab)); ami.append(AMI(y, lab))
            rc.append(spearmanr(ch, ami)[0]); rs.append(spearmanr(sil, ami)[0])
    return f(rc), f(rs)


if __name__ == '__main__':
    print("=== Consensus vs single trajectory (5 seeds, mean±std) ===", flush=True)
    for name in ['MSRC-v5', 'OutdoorScene', 'Handwritten', '100Leaves']:
        s, c = consensus(name)
        print(f"{name:12s} | single {s[0]:.3f}±{s[1]:.3f} | consensus {c[0]:.3f}±{c[1]:.3f} | gain {c[0]-s[0]:+.3f}", flush=True)
    print("=== Selection criterion rho vs AMI (over dataset×seed, mean±std) ===", flush=True)
    rc, rs = criterion(['MSRC-v5', 'OutdoorScene', 'Handwritten', 'UCI', '100Leaves'])
    print(f"CH rho = {rc[0]:.3f}±{rc[1]:.3f} | Silhouette rho = {rs[0]:.3f}±{rs[1]:.3f}", flush=True)
