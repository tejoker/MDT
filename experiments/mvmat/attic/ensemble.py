"""Consensus over MDT trajectories via the DDM Gram target.

Does a consensus geometry (average of the DDM targets over many sampled
trajectories) improve over a single trajectory -- and over the best single one
we could hope to select? If yes, the DDM encoder can distill it for OOS; if no,
the idea is dead. This is the label-free core test (no training).
"""
import argparse
import yaml
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import silhouette_score as SIL

from experiments.mvmat.load_data import get_data
from src.mdt_operators import transition_matrix, _mdt_operator, _trajectory


def pi_sym(W):
    n = len(W); p = np.ones(n) / n
    for _ in range(300):
        p = p @ W; p /= p.sum()
    sp = np.sqrt(np.maximum(p, 1e-12))
    return sp[:, None] * W / sp[None, :], sp


def gram_target(W):
    A, sp = pi_sym(W)
    return A @ A.T - np.outer(sp, sp)


def embed(T, k):
    w, V = np.linalg.eigh((T + T.T) / 2)
    idx = np.argsort(w)[::-1][:k]
    return V[:, idx] * np.sqrt(np.clip(w[idx], 0, None))


def km(E, k):
    return KMeans(k, n_init=10, random_state=0).fit_predict(E)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('-c', '--config', required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    k, t, knn = cfg['mdt']['n_components'], cfg['mdt']['steps'], cfg['mdt'].get('knn', 7)
    M = cfg['mdt'].get('search', 30)
    data = get_data(**cfg['data'])
    X, y = data['train'], data['train_color']
    sig = [np.quantile(pdist(v.reshape(len(v), -1)), 0.5) for v in X]
    P = [transition_matrix(v, s, knn) for v, s in zip(X, sig)]

    Ts = [gram_target(_mdt_operator(_trajectory(len(P), t, 'random', s), P)) for s in range(M)]
    singles = [AMI(y, km(embed(T, k), k)) for T in Ts]
    sil = [SIL(e, km(e, k)) for e in (embed(T, k) for T in Ts)]
    ami_sel = singles[int(np.argmax(sil))]                 # silhouette-selected single

    Tbar = np.mean(Ts, axis=0)                              # consensus (avg DDM targets)
    ami_consensus = AMI(y, km(embed(Tbar, k), k))

    print(f"{cfg['data']['name']:13s} | single: mean={np.mean(singles):.3f} max={max(singles):.3f} "
          f"sil-sel={ami_sel:.3f} | CONSENSUS={ami_consensus:.3f}")


if __name__ == '__main__':
    main()
