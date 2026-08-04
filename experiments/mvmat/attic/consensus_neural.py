"""Faithful neural SVD (DDM, sqrt-pi Gram matching) of the MDT consensus operator.

Consensus operator W̄ = mean_m W_m (the paper's E[W] mixture) is itself
row-stochastic, so it has a well-defined stationary pi̅ and MVDiffusionLoss
applies directly -- no collapse. Compares single-trajectory mean, classical
eig of the consensus, and the faithful neural SVD of the consensus.
"""
import argparse
import yaml
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI

from experiments.mvmat.load_data import get_data
from experiments.utils.models import build_mv_encoder
from src.mdt_operators import transition_matrix, _mdt_operator, _trajectory
from src.mvdiffusionloss import MVDiffusionLoss


def km(E, k):
    return KMeans(k, n_init=10, random_state=0).fit_predict(E)


def gram_target(W):
    n = len(W); p = np.ones(n) / n
    for _ in range(300):
        p = p @ W; p /= p.sum()
    sp = np.sqrt(np.maximum(p, 1e-12))
    A = sp[:, None] * W / sp[None, :]
    return A @ A.T - np.outer(sp, sp)


def eig_embed(T, k):
    w, V = np.linalg.eigh((T + T.T) / 2)
    idx = np.argsort(w)[::-1][:k]
    return V[:, idx] * np.sqrt(np.clip(w[idx], 0, None))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('-c', '--config', required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    k, t, knn = cfg['mdt']['n_components'], cfg['mdt']['steps'], cfg['mdt'].get('knn', 7)
    M = cfg['mdt'].get('search', 20)
    data = get_data(**cfg['data'])
    X, y = data['train'], data['train_color']
    sig = [np.quantile(pdist(v.reshape(len(v), -1)), 0.5) for v in X]
    P = [transition_matrix(v, s, knn) for v, s in zip(X, sig)]

    Ws = [_mdt_operator(_trajectory(len(P), t, 'random', s), P) for s in range(M)]
    ami_single = np.mean([AMI(y, km(eig_embed(gram_target(W), k), k)) for W in Ws])
    Wbar = np.mean(Ws, axis=0)                                   # E[W] consensus operator
    ami_eig = AMI(y, km(eig_embed(gram_target(Wbar), k), k))     # classical consensus

    enc = build_mv_encoder([v.shape[1:] for v in X], units=256, n_components=k, use_bn=True)
    enc.compile(loss=MVDiffusionLoss(Wbar), optimizer=tf.keras.optimizers.Adam(0.01))
    enc.fit(X, np.arange(len(y)), epochs=600, batch_size=min(500, len(y)), shuffle=True, verbose=0)
    ami_neural = AMI(y, km(enc.predict(X, verbose=0), k))

    print(f"{cfg['data']['name']:13s} | single-mean={ami_single:.3f}  consensus eig={ami_eig:.3f}  "
          f"consensus NEURAL-SVD(faithful)={ami_neural:.3f}")


if __name__ == '__main__':
    main()
