import numpy as np
import tensorflow as tf
from itertools import product
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import calinski_harabasz_score as CH
from sklearn.metrics import silhouette_score as SIL
from scipy.stats import spearmanr

from experiments.utils.experiments import load_config, get_sigma
from experiments.utils.models import build_mv_encoder
from experiments.mvmat.load_data import get_data
from src.mdt_operators import transition_matrix, oos_transition, _mdt_operator, _trajectory
from src.mvdiffusionloss import MVDiffusionLoss


def embed_svd(W, k):
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    return U[:, 1:k + 1] * s[1:k + 1], Vt.T


def km(E, k):
    return KMeans(k, n_init=10, random_state=0).fit_predict(E)


def candidate_paths(V, t, K, cap=700):
    """Discrete tree paths (one-hot per step) + K convex trajectories."""
    paths = []
    if V ** t <= cap:                       # whole tree
        for combo in product(range(V), repeat=t):
            tr = np.zeros((t, V)); tr[np.arange(t), combo] = 1.0; paths.append(tr)
    else:                                   # sample the tree
        for seed in range(200):
            c = np.random.default_rng(seed).integers(0, V, t)
            tr = np.zeros((t, V)); tr[np.arange(t), c] = 1.0; paths.append(tr)
    paths += [_trajectory(V, t, 'random', seed) for seed in range(K)]   # convex
    return paths


def main():
    cfg, _ = load_config()
    k, t, knn = cfg['mdt']['n_components'], cfg['mdt']['steps'], cfg['mdt'].get('knn')
    data = get_data(**cfg['data'])
    Xtr, Xte, ytr, yte = data['train'], data['test'], data['train_color'], data['test_color']
    sigmas = [get_sigma(v, cfg['mdt']['quantile']) for v in Xtr]
    P = [transition_matrix(v, s, knn) for v, s in zip(Xtr, sigmas)]
    Pnew = [oos_transition(vtr, vte, s, knn) for vtr, vte, s in zip(Xtr, Xte, sigmas)]
    V = len(P)

    # ---- BEST PATH: search the discrete tree, score by CH / silhouette / oracle-AMI ----
    cands = candidate_paths(V, t, cfg['mdt'].get('search', 20))
    ch, sil, ami, Ws = [], [], [], []
    for tr in cands:
        W = _mdt_operator(tr, P); E, _ = embed_svd(W, k); lab = km(E, k)
        ch.append(CH(E, lab)); sil.append(SIL(E, lab)); ami.append(AMI(ytr, lab)); Ws.append((tr, W))
    ch, sil, ami = np.array(ch), np.array(sil), np.array(ami)
    i_ch, i_sil, i_or = ch.argmax(), sil.argmax(), ami.argmax()
    print(f"[best path] paths={len(cands)} | oracle-AMI={ami[i_or]:.3f} | CH-pick={ami[i_ch]:.3f} "
          f"sil-pick={ami[i_sil]:.3f} | mean={ami.mean():.3f} | rho(CH,AMI)={spearmanr(ch,ami)[0]:+.2f} "
          f"rho(sil,AMI)={spearmanr(sil,ami)[0]:+.2f}")

    # extend the CH-selected operator (honest, label-free pipeline)
    traj_star, W = Ws[i_ch]
    E_L, Vmat = embed_svd(W, k)

    # ---- Nystrom OOS ----
    weighted = [np.einsum('v,vnm->nm', traj_star[s], np.stack(P)) for s in range(t)]
    M = np.eye(len(P[0])) if t == 1 else weighted[0].copy()
    for i in range(1, t - 1):
        M = weighted[i] @ M
    W_oos = np.einsum('v,vnm->nm', traj_star[t - 1], np.stack(Pnew)) @ M
    ami_nys = AMI(yte, km(W_oos @ Vmat[:, 1:k + 1], k))

    # ---- Deep OOS ----
    enc = build_mv_encoder([v.shape[1:] for v in Xtr], n_components=k, **cfg['encoder']['architecture'])
    enc.compile(loss=MVDiffusionLoss(W), optimizer=tf.keras.optimizers.Adam(**cfg['encoder']['optimizer']))
    enc.fit(Xtr, np.arange(len(ytr)), **cfg['encoder']['training'])
    ami_deep = AMI(yte, km(enc.predict(Xte, verbose=0), k))

    # ---- full-recompute ceiling ----
    Pall = [transition_matrix(np.vstack([a, b]), s, knn) for a, b, s in zip(Xtr, Xte, sigmas)]
    E_all, _ = embed_svd(_mdt_operator(traj_star, Pall), k)
    ami_full = AMI(yte, km(E_all[len(ytr):], k))

    print(f"[OOS AMI / {len(yte)} pts]  Nystrom={ami_nys:.3f}  Deep={ami_deep:.3f}  ceiling={ami_full:.3f}  "
          f"-> {'DEEP' if ami_deep>ami_nys+0.01 else 'NYSTROM' if ami_nys>ami_deep+0.01 else 'tie'}")


if __name__ == '__main__':
    main()
