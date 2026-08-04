"""T3 deep-dig: multi-seed svds-vs-neural accuracy parity + timing, to confirm the
~300x cost gap and equal accuracy are robust, not single-run flukes."""
import time
import numpy as np
import scipy.io
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI
import experiments.mvmat.sparse_neural_svd as S
import experiments.mvmat.crit_test as C


def run(name, N, seeds=3):
    m = scipy.io.loadmat(f'/tmp/Multi-view-datasets/{name}.mat'); Xo = m['X'].ravel()
    y = np.asarray(m['y']).ravel(); k = len(np.unique(y))
    sa, na, st, nt = [], [], [], []
    for seed in range(seeds):
        idx = np.random.default_rng(seed).choice(len(y), N, replace=False); yy = y[idx]
        Xs = [np.nan_to_num(StandardScaler().fit_transform(np.asarray(v)[idx].astype(float))).astype('float32') for v in Xo]
        knn = int(np.log(N)) + 1
        Ps = [S.sparse_transition(v, knn) for v in Xs]; Pts = [S.torch_sparse(P.T) for P in Ps]
        a = C.trajectory(len(Xs), seed)
        t0 = time.time(); E = C.svds_embed(Ps, a, k); st.append(time.time() - t0)
        sa.append(AMI(yy, KMeans(k, n_init=10, random_state=0).fit_predict(E)))
        t0 = time.time(); emb = S.neural_svd(Xs, Ps, Pts, torch.tensor(a, dtype=torch.float32), k); nt.append(time.time() - t0)
        na.append(AMI(yy, KMeans(k, n_init=10, random_state=0).fit_predict(emb)))
    g = lambda x: (float(np.mean(x)), float(np.std(x)))
    return g(sa), g(na), float(np.mean(st)), float(np.mean(nt))


if __name__ == '__main__':
    print("=== T3 multi-seed (3 seeds): svds vs neural ===", flush=True)
    for N in (4000, 8000):
        sa, na, stime, ntime = run('MNIST-10k', N)
        print(f"N={N} | svds AMI {sa[0]:.3f}±{sa[1]:.3f} ({stime:.2f}s) | "
              f"neural AMI {na[0]:.3f}±{na[1]:.3f} ({ntime:.1f}s) | speedup {ntime/stime:.0f}x", flush=True)
