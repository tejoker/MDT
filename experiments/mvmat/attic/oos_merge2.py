"""Merged OOS table (Scenario 3), FAITHFUL: Nystrom vs the paper's own DDM encoder
(build_mv_encoder + MVDiffusionLoss, the sqrt(pi)-scaled Gram loss), 5 seeds, 6 datasets.
Same encoder as Table 2. N capped for speed. Prints mean+-std per dataset."""
import os, sys, numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI
sys.path.insert(0, '/home/nicolasbigeard/internshipborelli/deep-diffusion-maps')
from experiments.mvmat.load_data import get_data
from experiments.utils.models import build_mv_encoder
from src.mvdiffusionloss import MVDiffusionLoss
from src.mdt_operators import transition_matrix, oos_transition, _mdt_operator, _trajectory

DATA = '/tmp/Multi-view-datasets'
DATASETS = ['MSRC-v5', 'Handwritten', 'Wikipedia', 'UCI', 'OutdoorScene', 'Caltech101-7']
T, SEEDS, CAP_TR, CAP_TE = 4, 5, 800, 400


def km(E, k):
    return KMeans(k, n_init=10, random_state=0).fit_predict(E)


def sig(v):
    return float(np.quantile(pdist(v[:200]), 0.5))


def embed_svd(W, k):
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    return U[:, 1:k + 1] * s[1:k + 1], Vt.T


def one(name, seed):
    d = get_data(path=DATA, name=name, split=0.7, seed=seed)
    Xtr = [np.asarray(v)[:CAP_TR] for v in d['train']]
    Xte = [np.asarray(v)[:CAP_TE] for v in d['test']]
    ytr, yte = np.asarray(d['train_color'])[:CAP_TR], np.asarray(d['test_color'])[:CAP_TE]
    k = len(np.unique(ytr)); n = len(ytr); knn = int(np.log(n)) + 1
    sigmas = [sig(v) for v in Xtr]
    P = [transition_matrix(v, s, knn) for v, s in zip(Xtr, sigmas)]
    Pnew = [oos_transition(vtr, vte, s, knn) for vtr, vte, s in zip(Xtr, Xte, sigmas)]
    a = _trajectory(len(P), T, 'random', seed)
    W = _mdt_operator(a, P)
    _, Vmat = embed_svd(W, k)
    # Nystrom OOS
    weighted = [np.einsum('v,vnm->nm', a[s], np.stack(P)) for s in range(T)]
    M = np.eye(n) if T == 1 else weighted[0].copy()
    for i in range(1, T - 1):
        M = weighted[i] @ M
    W_oos = np.einsum('v,vnm->nm', a[T - 1], np.stack(Pnew)) @ M
    ami_nys = AMI(yte, km(W_oos @ Vmat[:, 1:k + 1], k))
    # Deep OOS (paper's encoder)
    tf.keras.utils.set_random_seed(seed)
    enc = build_mv_encoder([v.shape[1:] for v in Xtr], units=256, n_components=k, use_bn=True)
    enc.compile(loss=MVDiffusionLoss(W), optimizer=tf.keras.optimizers.Adam(0.01))
    enc.fit(Xtr, np.arange(n), epochs=600, batch_size=min(500, n), shuffle=True, verbose=0)
    ami_deep = AMI(yte, km(enc.predict(Xte, verbose=0), k))
    return ami_nys, ami_deep


def main():
    print(f"{'Dataset':13s} {'Nystrom':>14s} {'deep(Gram)':>14s}", flush=True)
    for name in DATASETS:
        nys, dp = [], []
        for seed in range(SEEDS):
            try:
                x, y = one(name, seed); nys.append(x); dp.append(y)
            except Exception as e:
                print(f"  {name} seed{seed} ERR {type(e).__name__}: {e}", flush=True)
        if nys:
            print(f"{name:13s} {np.mean(nys):.3f}+-{np.std(nys):.3f}  {np.mean(dp):.3f}+-{np.std(dp):.3f}",
                  flush=True)


if __name__ == '__main__':
    main()
