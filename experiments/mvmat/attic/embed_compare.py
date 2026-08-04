"""Controlled comparison of two MDT embedding readouts on the SAME operators:
  (a) MDT-shipped plain SVD of W, top component dropped;
  (b) DDM-style pi-symmetrized Gram, A = Pi^1/2 W Pi^-1/2, embed AAᵀ − √π√πᵀ.
Mean clustering AMI over M random trajectories isolates the embedding-formula effect.
"""
import argparse
import yaml
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI

from experiments.mvmat.load_data import get_data
from src.mdt_operators import transition_matrix, _mdt_operator, _trajectory


def km(E, k):
    return KMeans(k, n_init=10, random_state=0).fit_predict(E)


def embed_plain(W, k):
    U, s, _ = np.linalg.svd(W, full_matrices=False)
    return U[:, 1:k + 1] * s[1:k + 1]


def embed_pisym(W, k):
    n = len(W); p = np.ones(n) / n
    for _ in range(300):
        p = p @ W; p /= p.sum()
    sp = np.sqrt(np.maximum(p, 1e-12))
    A = sp[:, None] * W / sp[None, :]
    T = A @ A.T - np.outer(sp, sp)
    w, V = np.linalg.eigh((T + T.T) / 2)
    idx = np.argsort(w)[::-1][:k]
    return V[:, idx] * np.sqrt(np.clip(w[idx], 0, None))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('-c', '--config', required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    k, t, knn = cfg['mdt']['n_components'], cfg['mdt']['steps'], cfg['mdt'].get('knn', 7)
    M = cfg['mdt'].get('search', 30)
    data = get_data(**cfg['data'])
    X, y = data['train'], data['train_color']
    sig = [np.quantile(pdist(v.reshape(len(v), -1)), 0.5) for v in X]
    P = [transition_matrix(v, s, knn) for v, s in zip(X, sig)]

    Ws = [_mdt_operator(_trajectory(len(P), t, 'random', s), P) for s in range(M)]
    plain = [AMI(y, km(embed_plain(W, k), k)) for W in Ws]
    pisym = [AMI(y, km(embed_pisym(W, k), k)) for W in Ws]
    print(f"{cfg['data']['name']:13s} | plain-SVD(MDT)={np.mean(plain):.3f}  "
          f"pi-sym(DDM)={np.mean(pisym):.3f}  delta={np.mean(pisym)-np.mean(plain):+.3f}")


if __name__ == '__main__':
    main()
