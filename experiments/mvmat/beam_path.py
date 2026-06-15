"""Silhouette-guided beam search for the best MDT discrete path.

Beam search over the trajectory tree: keep the top-B partial paths by an
unsupervised criterion at each depth, extend by every view, repeat to depth t.
Compares Silhouette- vs CH-guided beams against the exhaustive oracle and the
random-trajectory mean, reporting AMI and number of operator evaluations.
"""
import numpy as np
from itertools import product
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_mutual_info_score as AMI,
                             calinski_harabasz_score as CH,
                             silhouette_score as SIL)

from experiments.utils.experiments import load_config, get_sigma
from experiments.mvmat.load_data import get_data
from src.mdt_operators import transition_matrix, _mdt_operator


def embed_svd(W, k):
    U, s, _ = np.linalg.svd(W, full_matrices=False)
    return U[:, 1:k + 1] * s[1:k + 1]


def onehot(path, V):
    tr = np.zeros((len(path), V)); tr[np.arange(len(path)), path] = 1.0
    return tr


def beam_search(P, t, k, B, crit):
    V = len(P); evals = [0]

    def score(path):
        evals[0] += 1
        E = embed_svd(_mdt_operator(onehot(path, V), P), k)
        lab = KMeans(k, n_init=3, random_state=0).fit_predict(E)
        return (SIL if crit == 'sil' else CH)(E, lab)

    beam = sorted(([score([v]), [v]] for v in range(V)), key=lambda x: -x[0])[:B]
    for _ in range(2, t + 1):
        cand = [[score(p + [v]), p + [v]] for _, p in beam for v in range(V)]
        beam = sorted(cand, key=lambda x: -x[0])[:B]
    return beam[0][1], evals[0]


def main():
    cfg, _ = load_config()
    k, t, knn, B = cfg['mdt']['n_components'], cfg['mdt']['steps'], cfg['mdt'].get('knn'), 5
    data = get_data(**cfg['data'])
    X, y = data['train'], data['train_color']
    sigmas = [get_sigma(v, cfg['mdt']['quantile']) for v in X]
    P = [transition_matrix(v, s, knn) for v, s in zip(X, sigmas)]
    V = len(P)

    def ami_of(path):
        E = embed_svd(_mdt_operator(onehot(path, V), P), k)
        return AMI(y, KMeans(k, n_init=10, random_state=0).fit_predict(E))

    res = {}
    for crit in ('sil', 'CH'):
        path, ev = beam_search(P, t, k, B, crit)
        res[crit] = (ami_of(path), ev)

    # exhaustive oracle + mean (only if the tree is small enough)
    n_paths = V ** t
    if n_paths <= 1500:
        amis = [ami_of(list(c)) for c in product(range(V), repeat=t)]
        oracle, mean, ne = max(amis), float(np.mean(amis)), len(amis)
    else:
        rng = np.random.default_rng(0)
        amis = [ami_of(list(rng.integers(0, V, t))) for _ in range(300)]
        oracle, mean, ne = max(amis), float(np.mean(amis)), -300  # -ve = sampled estimate

    print(f"{cfg['data']['name']:13s} V={V} | beam-SIL={res['sil'][0]:.3f}(ev={res['sil'][1]}) "
          f"beam-CH={res['CH'][0]:.3f}(ev={res['CH'][1]}) | oracle={oracle:.3f}(ev={ne}) mean={mean:.3f}")


if __name__ == '__main__':
    main()
