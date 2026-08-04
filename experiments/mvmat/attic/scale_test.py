"""Scaling test: at growing N, where is the cost, and does neural SVD help?

For each N (subsample of MNIST-10k): build one MDT operator, then compare
classical eig of its DDM Gram target vs the faithful neural SVD (MVDiffusionLoss
encoder). Times each stage to see whether the SVD is even the bottleneck -- MDT's
operator is a dense product of N x N matrices, so building it is itself O(N^3).
"""
import time
import numpy as np
import scipy.io
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI

from experiments.utils.models import build_mv_encoder
from src.mdt_operators import transition_matrix, _mdt_operator, _trajectory
from src.mvdiffusionloss import MVDiffusionLoss

PATH = '/tmp/Multi-view-datasets/MNIST-10k.mat'
K = 10


def km(E):
    return KMeans(K, n_init=10, random_state=0).fit_predict(E)


def gram_target(W):
    n = len(W); p = np.ones(n) / n
    for _ in range(200):
        p = p @ W; p /= p.sum()
    sp = np.sqrt(np.maximum(p, 1e-12))
    A = sp[:, None] * W / sp[None, :]
    return A @ A.T - np.outer(sp, sp)


def eig_embed(T, k):
    w, V = np.linalg.eigh((T + T.T) / 2)
    idx = np.argsort(w)[::-1][:k]
    return V[:, idx] * np.sqrt(np.clip(w[idx], 0, None))


def main():
    m = scipy.io.loadmat(PATH); Xo = m['X'].ravel(); y = np.asarray(m['y']).ravel()
    rng = np.random.default_rng(0)
    for N in (2000, 4000, 6000, 8000):
        idx = rng.choice(len(y), N, replace=False)
        Xs = [StandardScaler().fit_transform(np.asarray(v)[idx].astype(float)).astype('float32') for v in Xo]
        yy = y[idx]
        knn = int(np.log(N)) + 1

        t0 = time.time()
        P = [transition_matrix(v, np.quantile(np.linalg.norm(v[:300, None] - v[None, :300], axis=2), 0.5), knn) for v in Xs]
        W = _mdt_operator(_trajectory(len(P), 4, 'random', 0), P)
        t_build = time.time() - t0

        t0 = time.time(); E = eig_embed(gram_target(W), K); t_eig = time.time() - t0
        ami_eig = AMI(yy, km(E))

        t0 = time.time()
        enc = build_mv_encoder([v.shape[1:] for v in Xs], units=256, n_components=K, use_bn=True)
        enc.compile(loss=MVDiffusionLoss(W), optimizer=tf.keras.optimizers.Adam(0.01))
        enc.fit(Xs, np.arange(N), epochs=150, batch_size=512, shuffle=True, verbose=0)
        t_neural = time.time() - t0
        ami_neural = AMI(yy, km(enc.predict(Xs, verbose=0)))

        print(f"N={N:5d} | build={t_build:5.1f}s eig={t_eig:6.1f}s neural={t_neural:6.1f}s | "
              f"AMI eig={ami_eig:.3f} neural={ami_neural:.3f}", flush=True)


if __name__ == '__main__':
    main()
