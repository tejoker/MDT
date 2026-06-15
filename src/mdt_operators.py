import numpy as np
from itertools import product
from sklearn.metrics import pairwise_distances


def transition_matrix(X_v: np.ndarray, sigma: float, knn: int = None) -> np.ndarray:
    """Row-stochastic Gaussian transition matrix P = D^-1 K for a single view.

    Pass `knn` to sparsify K to a k-nearest-neighbour graph (recommended for the
    'contrastive' trajectory, whose positive pairs are the non-zero entries).
    """
    d2 = pairwise_distances(X_v.reshape(len(X_v), -1), metric='sqeuclidean', n_jobs=-1)
    K = np.exp(-d2 / (2 * sigma ** 2))
    if knn is not None:
        keep = np.argsort(d2, axis=1)[:, :knn]
        mask = np.zeros_like(K)
        np.put_along_axis(mask, keep, 1.0, axis=1)
        K *= np.maximum(mask, mask.T)
    return K / K.sum(axis=1, keepdims=True)


def oos_transition(X_train: np.ndarray, X_new: np.ndarray, sigma: float, knn: int = None) -> np.ndarray:
    """Row-stochastic transition of out-of-sample points to the train set (n_new x m).

    Each new point's affinities to the m train points, knn-sparsified and
    normalised the same way as `transition_matrix`. Used to build the operator
    row of an unseen point for Nystrom-style extension.
    """
    a = X_train.reshape(len(X_train), -1)
    b = X_new.reshape(len(X_new), -1)
    d2 = pairwise_distances(b, a, metric='sqeuclidean', n_jobs=-1)
    K = np.exp(-d2 / (2 * sigma ** 2))
    if knn is not None:
        keep = np.argsort(d2, axis=1)[:, :knn]
        mask = np.zeros_like(K)
        np.put_along_axis(mask, keep, 1.0, axis=1)
        K *= mask
    return K / K.sum(axis=1, keepdims=True)


def _mdt_operator(trajectory: np.ndarray, P: list) -> np.ndarray:
    """MDT operator W^(t) = W_t...W_1 with W_i = sum_v traj[i,v] P_v (left product).

    Identical construction to mdt.mdt_utils.mdt_operator in the MDT reference repo.
    """
    weighted = np.einsum('tk,knm->tnm', trajectory, np.stack(P, axis=0))
    W = weighted[0]
    for i in range(1, len(weighted)):
        W = weighted[i] @ W
    return W


def _trajectory(n_views: int, t: int, method: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if method == 'random':       # convex combination per step (Dirichlet weights)
        return rng.dirichlet(np.ones(n_views), size=t)
    if method == 'circulant':    # cycle through the views, one per step
        return np.eye(n_views)[[i % n_views for i in range(t)]]
    raise ValueError(f"unknown discrete method: {method!r}")


def _mdt_contrastive(P: list, t: int, steps: int = 80, lr: float = 0.05, seed: int = 0):
    """Learn convex view-fusion weights by ADAM on a contrastive loss (MDT-Cst).

    Vendored from github.com/Gwendal-Debaussart/mixed-diffusion-trajectory
    (mdt/mdt_contrastive.py). Positive pairs per view = non-zero kernel entries,
    so views must be knn-sparsified. torch-only; no external package needed.
    """
    import torch
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Xt = torch.as_tensor(np.asarray(P), dtype=torch.float32, device=device)  # (k, n, n)
    idx = [torch.as_tensor(p > 0, dtype=torch.bool, device=device) for p in P]
    eye = torch.eye(Xt.shape[1], device=device)
    A = torch.rand(t, len(P), dtype=torch.float32, requires_grad=True, device=device)
    opt = torch.optim.Adam([A], lr=lr)

    def operator(weights):
        w = torch.einsum('tk,knm->tnm', weights, Xt)
        W = w[0]
        for i in range(1, len(w)):
            W = w[i] @ W
        return W

    def contrastive_loss(W):
        W = torch.nan_to_num(W, nan=0., posinf=20., neginf=-20.).clamp(-20., 20.)
        exW = torch.exp(W)
        D = (exW * (1 - eye)).sum(1).clamp_min(1e-12)
        logp = torch.log((exW / D.unsqueeze(1)).clamp_min(1e-12))
        return -sum(logp[m].sum() / len(idx) for m in idx) / W.shape[0]

    best, best_loss = A.detach().clone(), float('inf')
    for _ in range(steps):
        opt.zero_grad()
        W = operator(torch.softmax(A, dim=1))
        if not torch.isfinite(W).all():
            continue
        loss = contrastive_loss(W)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_([A], 1.0)
        opt.step()
        if loss.item() < best_loss:
            best_loss, best = loss.item(), A.detach().clone()

    weights = torch.softmax(best, dim=1).cpu().numpy()
    return _mdt_operator(weights, P)


def mdt_operator_from_views(Xs, sigmas, t: int, method: str = 'contrastive',
                            knn: int = None, seed: int = 0) -> np.ndarray:
    """Build an MDT operator W^(t) from a list of data views.

    method: 'random' | 'circulant' (numpy-only) |
            'contrastive' (learned convex view fusion via ADAM, needs torch).
    """
    P = [transition_matrix(Xv, s, knn) for Xv, s in zip(Xs, sigmas)]
    if method in ('random', 'circulant'):
        return _mdt_operator(_trajectory(len(P), t, method, seed), P)
    if method == 'contrastive':
        return _mdt_contrastive(P, t, seed=seed)
    raise ValueError(f"unknown method: {method!r}")


def select_trajectory(P, t: int, k: int, n_samples: int = 200, seed: int = 0):
    """Select the best MDT trajectory by silhouette over a sampled pool of paths.

    Silhouette is a markedly better trajectory selector than the Calinski-Harabasz
    index used by MDT-Direct (mean Spearman rho with oracle AMI ~0.48 vs ~0.25 over
    10 multi-view datasets), and full-pool ranking beats beam search, which prunes
    away good paths. Candidates are the discrete one-hot tree (exhaustive if small,
    else `n_samples` random paths) plus a few convex trajectories.

    Parameters
    ----------
    P : list of (n, n) row-stochastic per-view transition matrices.
    t : trajectory length. k : embedding dimension (e.g. number of clusters).
    n_samples : pool size when the tree is too large to enumerate.

    Returns
    -------
    (W, trajectory, silhouette) : the chosen operator, its weight trajectory, and score.
    Falls back to a random trajectory if no candidate yields a valid clustering.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    V = len(P)
    rng = np.random.default_rng(seed)
    if V ** t <= n_samples:
        paths = [_onehot(c, V) for c in product(range(V), repeat=t)]
    else:
        paths = [_onehot(rng.integers(0, V, t), V) for _ in range(n_samples)]
    paths += [rng.dirichlet(np.ones(V), size=t) for _ in range(max(1, n_samples // 10))]

    best_score, best_W, best_traj = -np.inf, None, None
    for traj in paths:
        W = _mdt_operator(traj, P)
        U, s, _ = np.linalg.svd(W, full_matrices=False)
        E = U[:, 1:k + 1] * s[1:k + 1]
        labels = KMeans(k, n_init=5, random_state=0).fit_predict(E)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(E, labels)
        if score > best_score:
            best_score, best_W, best_traj = score, W, traj
    if best_W is None:                       # no candidate clustered; fall back
        best_traj = paths[0]; best_W = _mdt_operator(best_traj, P)
    return best_W, best_traj, best_score


def _onehot(path, V):
    traj = np.zeros((len(path), V))
    traj[np.arange(len(path)), np.asarray(path)] = 1.0
    return traj
