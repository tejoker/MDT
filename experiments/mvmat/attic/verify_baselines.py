"""Close the 'no MVD/AD/ID baseline' limitation. Vendored from the MDT repo's
competitors/ (pure numpy): Alternating Diffusion, Integrated Diffusion, Multi-View
Diffusion. Compared against our MDT-consensus, multi-seed, embedding->KMeans->AMI."""
import numpy as np
import scipy.io
from functools import reduce
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI
from src.mdt_operators import transition_matrix, _mdt_operator, _trajectory


def load(name, seed, Ncap=600):
    m = scipy.io.loadmat(f'/tmp/Multi-view-datasets/{name}.mat'); Xo = m['X'].ravel()
    y = np.asarray(m['y'] if 'y' in m else m['Y']).ravel(); k = len(np.unique(y))
    N = min(len(y), Ncap)
    idx = np.random.default_rng(seed).choice(len(y), N, replace=False)
    Xs = [np.nan_to_num(StandardScaler().fit_transform(np.asarray(v)[idx].astype(float))).astype('float64') for v in Xo]
    return Xs, y[idx], k


def km(E, k):
    return KMeans(k, n_init=8, random_state=0).fit_predict(E)


def embed_svd(W, k):
    U, s, _ = np.linalg.svd(W, full_matrices=False)
    return U[:, 1:k + 1] * s[1:k + 1]


def gram(W):
    n = len(W); p = np.ones(n) / n
    for _ in range(150):
        p = p @ W; p /= p.sum()
    sp = np.sqrt(np.maximum(p, 1e-12)); A = sp[:, None] * W / sp[None, :]
    return A @ A.T - np.outer(sp, sp)


def eig_embed(T, k):
    w, V = np.linalg.eigh((T + T.T) / 2); idx = np.argsort(w)[::-1][:k]
    return V[:, idx] * np.sqrt(np.clip(w[idx], 0, None))


# --- vendored competitors (fixed power t=4 in place of the knee heuristic) ---
def AD(P, t=4):
    return np.linalg.matrix_power(reduce(lambda a, b: a @ b, P), t)


def ID(P, t=4):
    return reduce(lambda a, b: a @ b, [np.linalg.matrix_power(p, t) for p in P])


def MVD(P):
    n = len(P); m = P[0].shape[0]; op = np.zeros((n * m, n * m))
    for i in range(n):
        for j in range(i + 1, n):
            b = P[i] @ P[j]
            op[i * m:(i + 1) * m, j * m:(j + 1) * m] = b
            op[j * m:(j + 1) * m, i * m:(i + 1) * m] = b.T
    rs = op.sum(1, keepdims=True); rs[rs == 0] = 1
    return op / rs


def consensus(P, M=12, seed=0):
    return np.mean([gram(_mdt_operator(_trajectory(len(P), 4, 'random', seed * 100 + mm), P)) for mm in range(M)], 0)


def run(name, seeds=3, knn=6):
    res = {m: [] for m in ['AD', 'ID', 'MVD', 'MDT-consensus']}
    for seed in range(seeds):
        Xs, y, k = load(name, seed)
        sig = [np.quantile(pdist(v[:200]), 0.5) for v in Xs]
        P = [transition_matrix(v, s, knn) for v, s in zip(Xs, sig)]
        res['AD'].append(AMI(y, km(embed_svd(AD(P), k), k)))
        res['ID'].append(AMI(y, km(embed_svd(ID(P), k), k)))
        Wmv = MVD(P); E = embed_svd(Wmv, k)[:len(y)]      # first view's block
        res['MVD'].append(AMI(y, km(E, k)))
        res['MDT-consensus'].append(AMI(y, km(eig_embed(consensus(P, seed=seed), k), k)))
    return res


if __name__ == '__main__':
    print("=== Field baselines vs MDT-consensus (3 seeds, AMI mean±std) ===", flush=True)
    for name in ['MSRC-v5', 'Handwritten', 'OutdoorScene', 'UCI']:
        r = run(name)
        cells = "  ".join(f"{m} {np.mean(v):.3f}±{np.std(v):.3f}" for m, v in r.items())
        print(f"{name:12s} | {cells}", flush=True)
