"""Merged OOS table (Scenario 3): Nystrom vs the SAME sqrt(pi)-Gram encoder as Table 2,
over 5 seeds, on all 6 datasets. One consistent encoder everywhere (no feat/aff swap)."""
import numpy as np, scipy.io, torch, torch.nn as nn, sys
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI
sys.path.insert(0, '/home/nicolasbigeard/internshipborelli/deep-diffusion-maps')
from src.mdt_operators import transition_matrix, oos_transition, _mdt_operator, _trajectory

DATA = '/tmp/Multi-view-datasets'
DATASETS = ['MSRC-v5', 'Handwritten', 'Wikipedia', 'UCI', 'OutdoorScene', 'Caltech101-7']
T, SEEDS = 4, 5


def km(E, k):
    return KMeans(k, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(E))


def gram_target(W):                                   # sqrt(pi)-symmetrized Gram (DDM target)
    n = len(W); p = np.ones(n) / n
    for _ in range(300):
        p = p @ W; p /= p.sum()
    sp = np.sqrt(np.maximum(p, 1e-12))
    A = sp[:, None] * W / sp[None, :]
    return A @ A.T - np.outer(sp, sp)


def gram_encoder_oos(Xtr, Xte, G, k, seed, epochs=600, lr=0.01):   # min ||FFᵀ - G||, apply to test
    torch.manual_seed(seed)
    xt = torch.tensor(np.concatenate(Xtr, 1), dtype=torch.float32)
    xe = torch.tensor(np.concatenate(Xte, 1), dtype=torch.float32)
    Gt = torch.tensor(G / max(np.linalg.norm(G, 2), 1e-12), dtype=torch.float32)
    net = nn.Sequential(nn.Linear(xt.shape[1], 256), nn.ReLU(),
                        nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, k))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        F = net(xt)
        (((F @ F.T) - Gt) ** 2).mean().backward()
        opt.step()
    with torch.no_grad():
        return net(xe).numpy()


def one(name, seed):
    m = scipy.io.loadmat(f'{DATA}/{name}.mat'); Xo = m['X'].ravel()
    y = np.asarray(m['y'] if 'y' in m else m['Y']).ravel(); k = len(np.unique(y))
    N = len(y); perm = np.random.default_rng(seed).permutation(N)
    n_tr = min(600, N // 2); n_te = min(N - n_tr, 400)
    tr, te = perm[:n_tr], perm[n_tr:n_tr + n_te]
    Xtr = [np.nan_to_num(StandardScaler().fit_transform(np.asarray(v)[tr].astype(float))) for v in Xo]
    Xte = [np.nan_to_num(StandardScaler().fit_transform(np.asarray(v)[te].astype(float))) for v in Xo]
    ytr, yte = y[tr], y[te]
    knn = int(np.log(n_tr)) + 1
    sig = [np.quantile(pdist(v[:200]), 0.5) for v in Xtr]
    P = [transition_matrix(v, s, knn) for v, s in zip(Xtr, sig)]
    Pnew = [oos_transition(vtr, vte, s, knn) for vtr, vte, s in zip(Xtr, Xte, sig)]
    a = _trajectory(len(P), T, 'random', seed)
    W = _mdt_operator(a, P)
    U, s_, Vt = np.linalg.svd(W, full_matrices=False); Vmat = Vt.T
    # Nystrom OOS (last step uses the train->test transition Pnew)
    weighted = [np.einsum('v,vnm->nm', a[sidx], np.stack(P)) for sidx in range(T)]
    M = np.eye(n_tr) if T == 1 else weighted[0].copy()
    for i in range(1, T - 1):
        M = weighted[i] @ M
    W_oos = np.einsum('v,vnm->nm', a[T - 1], np.stack(Pnew)) @ M
    ami_nys = AMI(yte, km(W_oos @ Vmat[:, 1:k + 1], k))
    # Gram encoder OOS (same target, applied to test features)
    ami_deep = AMI(yte, km(gram_encoder_oos(Xtr, Xte, gram_target(W), k, seed), k))
    return ami_nys, ami_deep


def main():
    print(f"{'Dataset':13s} {'Nystrom':>14s} {'deep(Gram)':>14s}", flush=True)
    for name in DATASETS:
        nys, dp = [], []
        for seed in range(SEEDS):
            try:
                a, b = one(name, seed); nys.append(a); dp.append(b)
            except Exception as e:
                print(f"  {name} seed{seed} ERR {e}", flush=True)
        if nys:
            print(f"{name:13s} {np.mean(nys):.3f}+-{np.std(nys):.3f}  {np.mean(dp):.3f}+-{np.std(dp):.3f}",
                  flush=True)


if __name__ == '__main__':
    main()
