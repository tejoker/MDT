"""Neural SVD the DDM way (Gram-matching, Eckart-Young) on the MDT consensus.

DDM computes the diffusion embedding with a network by minimising
    ‖ F Fᵀ − T ‖²   (F = encoder output),
whose optimum is the truncated SVD of T (Eckart-Young) -- no spectral
decomposition, no Rayleigh/orthonormality objective. Here T = consensus target
T̄ = mean_m(A_m A_mᵀ − √π√πᵀ). This is the correct "neural SVD" for this
ecosystem; compared against exact eig(T̄).
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


class GramLoss(tf.keras.losses.Loss):
    def __init__(self, G):
        super().__init__(); self.G = tf.constant(G, tf.float32)

    def call(self, y_true, y_pred):
        ids = tf.cast(y_true, tf.int32)
        G = tf.gather(tf.gather(self.G, ids), ids, axis=1)
        return tf.reduce_mean(tf.square(tf.matmul(y_pred, y_pred, transpose_b=True) - G))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('-c', '--config', required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    k, t, knn = cfg['mdt']['n_components'], cfg['mdt']['steps'], cfg['mdt'].get('knn', 7)
    M = cfg['mdt'].get('search', 20)
    data = get_data(**cfg['data'])
    X, y = data['train'], data['train_color']
    sig = [np.quantile(pdist(v.reshape(len(v), -1)), 0.5) for v in X]
    P = [transition_matrix(v, s, knn) for v, s in zip(X, sig)]

    Tbar = np.mean([gram_target(_mdt_operator(_trajectory(len(P), t, 'random', s), P)) for s in range(M)], axis=0)
    ami_eig = AMI(y, km(eig_embed(Tbar, k), k))
    Tn = Tbar / np.linalg.norm(Tbar, 2)               # unit spectral scale (conditioning)

    enc = build_mv_encoder([v.shape[1:] for v in X], units=256, n_components=k, use_bn=True)
    enc.compile(loss=GramLoss(Tn), optimizer=tf.keras.optimizers.Adam(0.01))
    enc.fit(X, np.arange(len(y)), epochs=600, batch_size=min(500, len(y)), shuffle=True, verbose=0)
    ami_neural = AMI(y, km(enc.predict(X, verbose=0), k))

    print(f"{cfg['data']['name']:13s} | consensus eig(SVD)={ami_eig:.3f}  "
          f"NEURAL-SVD (DDM Gram-match)={ami_neural:.3f}")


if __name__ == '__main__':
    main()
