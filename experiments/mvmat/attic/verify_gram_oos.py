"""Gap #2: multi-seed the Gram-matching encoder OOS (Table 4 was single-run).
Is 'Nystrom beats the Gram-matching encoder' real, or also within noise?"""
import numpy as np
import scipy.io
import tensorflow as tf
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI

from experiments.utils.models import build_mv_encoder
from src.mvdiffusionloss import MVDiffusionLoss
from src.mdt_operators import transition_matrix, oos_transition, _mdt_operator, _trajectory


def km(E, k):
    return KMeans(k, n_init=10, random_state=0).fit_predict(E)


def run(name, ntr, nte, seed, t=4):
    m = scipy.io.loadmat(f'/tmp/Multi-view-datasets/{name}.mat'); Xo = m['X'].ravel()
    y = np.asarray(m['y'] if 'y' in m else m['Y']).ravel(); k = len(np.unique(y))
    perm = np.random.default_rng(seed).permutation(len(y))
    tr, te = perm[:ntr], perm[ntr:ntr + nte]
    Xtr = [np.nan_to_num(StandardScaler().fit_transform(np.asarray(v)[tr].astype(float))).astype('float32') for v in Xo]
    Xte = [np.nan_to_num(StandardScaler().fit_transform(np.asarray(v)[te].astype(float))).astype('float32') for v in Xo]
    ytr, yte = y[tr], y[te]
    knn = int(np.log(ntr)) + 1
    sig = [np.quantile(pdist(v[:200]), 0.5) for v in Xtr]
    P = [transition_matrix(v, s, knn) for v, s in zip(Xtr, sig)]
    Pnew = [oos_transition(a, b, s, knn) for a, b, s in zip(Xtr, Xte, sig)]
    a = _trajectory(len(P), t, 'random', seed)
    W = _mdt_operator(a, P)
    U, s_, Vt = np.linalg.svd(W, full_matrices=False); V = Vt.T

    weighted = [np.einsum('v,vnm->nm', a[s], np.stack(P)) for s in range(t)]
    M = np.eye(ntr) if t == 1 else weighted[0].copy()
    for i in range(1, t - 1):
        M = weighted[i] @ M
    W_oos = np.einsum('v,vnm->nm', a[t - 1], np.stack(Pnew)) @ M
    ami_nys = AMI(yte, km(W_oos @ V[:, 1:k + 1], k))

    enc = build_mv_encoder([v.shape[1:] for v in Xtr], units=256, n_components=k, use_bn=True)
    enc.compile(loss=MVDiffusionLoss(W), optimizer=tf.keras.optimizers.Adam(0.01))
    enc.fit(Xtr, np.arange(ntr), epochs=600, batch_size=min(500, ntr), shuffle=True, verbose=0)
    ami_deep = AMI(yte, km(enc.predict(Xte, verbose=0), k))
    return ami_nys, ami_deep


if __name__ == '__main__':
    print("=== Gram-matching encoder OOS, 5 seeds (mean±std): Nystrom vs deep(Gram) ===", flush=True)
    for name, ntr, nte in [('Handwritten', 1400, 600), ('Wikipedia', 2000, 800)]:
        nys, deep = [], []
        for seed in range(3):
            n, d = run(name, ntr, nte, seed)
            nys.append(n); deep.append(d)
        print(f"{name:12s} | Nys {np.mean(nys):.3f}±{np.std(nys):.3f} | deep(Gram) {np.mean(deep):.3f}±{np.std(deep):.3f}", flush=True)
