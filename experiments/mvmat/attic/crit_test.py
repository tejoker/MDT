"""Two critical tests.
(A) Confound: is the neural 'win' a better decomposition, or feature-bypass?
    Compare features-only KMeans, eig of a GOOD operator (best of several, via
    truncated svds), and the neural SVD on that same good operator. If neural ~ eig
    on a good operator -> faithful; if features-only ~ neural -> operator adds little.
(B) Real scaling baseline: replace dense eigh with truncated svds on the SPARSE
    operator action (the proper scalable classical method) and time vs the neural net.
"""
import time
import argparse
import numpy as np
import scipy.io
import torch
from scipy.sparse.linalg import svds, LinearOperator
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI

import experiments.mvmat.sparse_neural_svd as S


def km(E, k):
    return KMeans(k, n_init=10, random_state=0).fit_predict(E)


def load(name, N, seed=0):
    m = scipy.io.loadmat(f'/tmp/Multi-view-datasets/{name}.mat'); Xo = m['X'].ravel()
    y = np.asarray(m['y'] if 'y' in m else m['Y']).ravel(); k = len(np.unique(y))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), N, replace=False) if N < len(y) else np.arange(len(y))
    Xs = [np.nan_to_num(StandardScaler().fit_transform(np.asarray(v)[idx].astype(float))).astype('float32') for v in Xo]
    return Xs, y[idx], k


def trajectory(V, seed):
    return np.asarray(torch.softmax(torch.tensor(np.random.default_rng(seed).standard_normal((S.T, V)), dtype=torch.float64), 1))


def lin_op(Ps, a):
    N = Ps[0].shape[0]; Pst = [P.T.tocsr() for P in Ps]
    def mv(x):
        out = x
        for s in range(a.shape[0]):
            out = sum(a[s, v] * (Ps[v] @ out) for v in range(len(Ps)))
        return out
    def rmv(x):
        out = x
        for s in reversed(range(a.shape[0])):
            out = sum(a[s, v] * (Pst[v] @ out) for v in range(len(Ps)))
        return out
    return LinearOperator((N, N), matvec=mv, rmatvec=rmv)


def svds_embed(Ps, a, k):
    u, sv, _ = svds(lin_op(Ps, a), k=k + 1)
    u, sv = u[:, ::-1], sv[::-1]
    return u[:, 1:k + 1] * sv[1:k + 1]


def part_A(name, N):
    Xs, yy, k = load(name, N)
    feat = AMI(yy, km(np.concatenate(Xs, 1), k))                 # features-only, no operator
    knn = int(np.log(N)) + 1
    Ps = [S.sparse_transition(v, knn) for v in Xs]
    eigs = [(AMI(yy, km(svds_embed(Ps, trajectory(len(Xs), s), k), k)), s) for s in range(8)]
    rand_mean = np.mean([e for e, _ in eigs])
    best_ami, best_seed = max(eigs)
    a = trajectory(len(Xs), best_seed)
    Pts = [S.torch_sparse(P.T) for P in Ps]
    neural = AMI(yy, km(S.neural_svd(Xs, Ps, Pts, torch.tensor(a, dtype=torch.float32), k), k))
    print(f"[A] {name:9s} N={N} k={k} | features-only={feat:.3f} | eig random-mean={rand_mean:.3f} "
          f"eig BEST-op={best_ami:.3f} | neural on best-op={neural:.3f}", flush=True)


def part_B(name, sizes):
    for N in sizes:
        Xs, yy, k = load(name, N)
        knn = int(np.log(N)) + 1
        Ps = [S.sparse_transition(v, knn) for v in Xs]
        Pts = [S.torch_sparse(P.T) for P in Ps]
        a = trajectory(len(Xs), 0)
        t0 = time.time(); E = svds_embed(Ps, a, k); t_svds = time.time() - t0
        ami_svds = AMI(yy, km(E, k))
        t0 = time.time(); emb = S.neural_svd(Xs, Ps, Pts, torch.tensor(a, dtype=torch.float32), k); t_n = time.time() - t0
        ami_n = AMI(yy, km(emb, k))
        print(f"[B] {name} N={N:5d} | svds(sparse) {t_svds:6.2f}s AMI={ami_svds:.3f} | "
              f"neural {t_n:6.1f}s AMI={ami_n:.3f}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', default='A')
    args = ap.parse_args()
    if args.part == 'A':
        part_A('MNIST-4', 4000); part_A('MNIST-10k', 4000); part_A('ALOI', 4000)
    else:
        part_B('MNIST-10k', [2000, 4000, 8000, 10000])
