"""T2 deep-dig: multi-seed ceiling test. Is neural ≈ eig_best (the SVD ceiling)
robust across seeds, and does the operator beat features-only with error bars?"""
import numpy as np
import torch
from sklearn.metrics import adjusted_mutual_info_score as AMI
import experiments.mvmat.crit_test as C
import experiments.mvmat.sparse_neural_svd as S


def run(name, N, seeds=5):
    F, B, Ne = [], [], []
    for seed in range(seeds):
        Xs, yy, k = C.load(name, N, seed)
        F.append(AMI(yy, C.km(np.concatenate(Xs, 1), k)))
        knn = int(np.log(N)) + 1
        Ps = [S.sparse_transition(v, knn) for v in Xs]
        eigs = [(AMI(yy, C.km(C.svds_embed(Ps, C.trajectory(len(Xs), s), k), k)), s) for s in range(8)]
        bami, bs = max(eigs); B.append(bami)
        a = C.trajectory(len(Xs), bs); Pts = [S.torch_sparse(P.T) for P in Ps]
        Ne.append(AMI(yy, C.km(S.neural_svd(Xs, Ps, Pts, torch.tensor(a, dtype=torch.float32), k), k)))
    m = lambda x: (float(np.mean(x)), float(np.std(x)))
    return m(F), m(B), m(Ne)


if __name__ == '__main__':
    print("=== Multi-seed ceiling (5 seeds): features-only | eig_best | neural ===", flush=True)
    for name, N in [('MSRC-v5', 210), ('MNIST-10k', 4000), ('ALOI', 4000)]:
        f, b, n = run(name, N)
        print(f"{name:11s} | feat {f[0]:.3f}±{f[1]:.3f} | eig_best {b[0]:.3f}±{b[1]:.3f} | neural {n[0]:.3f}±{n[1]:.3f}", flush=True)
