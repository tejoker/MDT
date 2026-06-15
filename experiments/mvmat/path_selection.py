"""Which unsupervised criterion best selects an MDT trajectory (the 'best path')?

For a candidate pool of discrete-tree paths (+ convex), we score each by 5
label-free criteria and by the oracle AMI, then report, per criterion:
  rho   = Spearman correlation with AMI (higher = better predictor)
  regret= oracle_AMI - AMI(path the criterion picks)   (0 = picks the best)
All criteria oriented so higher = better.
"""
import numpy as np
from itertools import product
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_mutual_info_score as AMI,
                             calinski_harabasz_score as CH,
                             silhouette_score as SIL,
                             davies_bouldin_score as DB)
from scipy.stats import spearmanr

from experiments.utils.experiments import load_config, get_sigma
from experiments.mvmat.load_data import get_data
from src.mdt_operators import transition_matrix, _mdt_operator, _trajectory


def embed_svd(W, k):
    U, s, _ = np.linalg.svd(W, full_matrices=False)
    return U[:, 1:k + 1] * s[1:k + 1]


def contrastive_quality(W, masks):
    """MDT contrastive Q (eq 19-20): per-view NLL over kernel neighbours; return -loss."""
    Wc = np.clip(W, -20, 20)
    ex = np.exp(Wc); np.fill_diagonal(ex, 0.0)
    denom = np.clip(ex.sum(1, keepdims=True), 1e-12, None)
    logp = np.log(np.clip(ex / denom, 1e-12, None))
    loss = np.mean([-logp[m].sum() for m in masks]) / W.shape[0]
    return -loss


def spectral_entropy(W):
    s = np.linalg.svd(W, compute_uv=False); s = s[s > 0]; s = s / s.sum()
    return -np.sum(s * np.log(s))     # raw entropy; oriented as -entropy below


def candidates(V, t, K, cap=700):
    paths = []
    if V ** t <= cap:
        for c in product(range(V), repeat=t):
            tr = np.zeros((t, V)); tr[np.arange(t), c] = 1.0; paths.append(tr)
    else:
        for seed in range(300):
            c = np.random.default_rng(seed).integers(0, V, t)
            tr = np.zeros((t, V)); tr[np.arange(t), c] = 1.0; paths.append(tr)
    paths += [_trajectory(V, t, 'random', seed) for seed in range(K)]
    return paths


def main():
    cfg, _ = load_config()
    k, t, knn = cfg['mdt']['n_components'], cfg['mdt']['steps'], cfg['mdt'].get('knn')
    data = get_data(**cfg['data'])
    X, y = data['train'], data['train_color']
    sigmas = [get_sigma(v, cfg['mdt']['quantile']) for v in X]
    P = [transition_matrix(v, s, knn) for v, s in zip(X, sigmas)]
    masks = [p > 0 for p in P]
    V = len(P)

    crit = {'CH': [], 'SIL': [], '-DB': [], 'Qc': [], '-Sent': []}
    ami = []
    for tr in candidates(V, t, cfg['mdt'].get('search', 30)):
        W = _mdt_operator(tr, P); E = embed_svd(W, k); lab = KMeans(k, n_init=5, random_state=0).fit_predict(E)
        crit['CH'].append(CH(E, lab)); crit['SIL'].append(SIL(E, lab)); crit['-DB'].append(-DB(E, lab))
        crit['Qc'].append(contrastive_quality(W, masks)); crit['-Sent'].append(-spectral_entropy(W))
        ami.append(AMI(y, lab))
    ami = np.array(ami); name = cfg['data']['name']
    out = f"{name:13s} paths={len(ami):4d} oracle={ami.max():.3f} mean={ami.mean():.3f} |"
    for c, vals in crit.items():
        vals = np.array(vals); rho = spearmanr(vals, ami)[0]; reg = ami.max() - ami[vals.argmax()]
        out += f" {c}:rho={rho:+.2f},reg={reg:.3f} |"
    print(out)


if __name__ == '__main__':
    main()
