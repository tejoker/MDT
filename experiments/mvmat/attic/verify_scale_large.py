"""Close the 'scale only N=1e4' limitation for the compute claim. Synthetic
multi-view Gaussian blobs at N up to 1e5; build the sparse MDT operator and time
truncated svds on its sparse action. If svds stays cheap at 1e5, the linear-scaling
claim holds beyond the real datasets (which top out at ~1e4)."""
import time
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI
import experiments.mvmat.sparse_neural_svd as S
import experiments.mvmat.crit_test as C


def synth(N, k=10, V=3, d=20, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, N)
    centers = [rng.standard_normal((k, d)) * 4 for _ in range(V)]
    Xs = [(centers[v][y] + rng.standard_normal((N, d))).astype('float32') for v in range(V)]
    return Xs, y, k


if __name__ == '__main__':
    print("=== Synthetic scale: truncated svds on the sparse MDT operator ===", flush=True)
    for N in (10000, 50000, 100000):
        Xs, y, k = synth(N)
        knn = int(np.log(N)) + 1
        t0 = time.time()
        P = [S.sparse_transition(v, knn) for v in Xs]
        t_build = time.time() - t0
        a = C.trajectory(len(Xs), 0)
        t0 = time.time(); E = C.svds_embed(P, a, k); t_svds = time.time() - t0
        ami = AMI(y, KMeans(k, n_init=10, random_state=0).fit_predict(E))
        print(f"N={N:6d} | kernels(sparse) {t_build:6.1f}s | svds {t_svds:6.2f}s | AMI={ami:.3f}", flush=True)
