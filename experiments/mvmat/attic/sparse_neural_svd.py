"""Scalable neural SVD of the MDT operator via sparse operator-action.

Never materialises W or G (N x N). Per-view transitions are sparse kNN graphs;
the t-step MDT operator is applied by sparse matvecs only. A multi-view net F =
f_θ(x) is QR-orthonormalised each step (this -- not a soft penalty -- is what
prevents the Ψ=0 collapse) and trained to maximise Σ_j ‖Wᵀ Q_j‖² , i.e. to span
the top-k left-singular subspace of W. Ordered components are recovered post-hoc
from the (k+1)x(k+1) matrix QᵀWWᵀQ. Cost per step is O(N·knn·t·k), linear in N.

Compares against classical eig (built densely, only where feasible) for accuracy,
and times both to expose the scaling.
"""
import argparse
import time
import numpy as np
import scipy.io
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI

PATH = '/tmp/Multi-view-datasets/MNIST-10k.mat'
K = 10
T = 4


def sparse_transition(X, knn):
    nn = NearestNeighbors(n_neighbors=knn + 1).fit(X)
    dist, idx = nn.kneighbors(X)
    n = len(X)
    rows = np.repeat(np.arange(n), knn)
    cols = idx[:, 1:].ravel()
    d = dist[:, 1:].ravel()
    dpos = d[d > 0]
    sigma = (np.median(dpos) if dpos.size else 1.0) + 1e-9   # ignore duplicate-point zeros
    w = np.exp(-(d ** 2) / (2 * sigma ** 2))
    Km = sp.csr_matrix((w, (rows, cols)), shape=(n, n))
    Km = Km.maximum(Km.T)
    deg = np.asarray(Km.sum(1)).ravel()
    iso = deg == 0
    if iso.any():                       # isolated nodes get a self-loop (avoid zero rows)
        Km = Km + sp.diags(iso.astype(float))
        deg = np.asarray(Km.sum(1)).ravel()
    deg[deg == 0] = 1
    return (sp.diags(1.0 / deg) @ Km).tocsr()


def torch_sparse(P):
    P = P.tocoo()
    return torch.sparse_coo_tensor(np.vstack([P.row, P.col]), P.data.astype('float64'), P.shape).coalesce()


def apply_op(mats, a, F):                       # (S_t...S_1) F, S_s = Σ_v a[s,v] P_v
    out = F
    for s in range(a.shape[0]):
        out = sum(a[s, v] * torch.sparse.mm(mats[v], out) for v in range(len(mats)))
    return out


class MVNet(torch.nn.Module):
    def __init__(self, dims, k, h=128):
        super().__init__()
        self.b = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Linear(h, h)) for d in dims])
        self.head = torch.nn.Sequential(torch.nn.Linear(h * len(dims), h), torch.nn.ReLU(), torch.nn.Linear(h, k))

    def forward(self, Xs):
        return self.head(torch.cat([bi(x) for bi, x in zip(self.b, Xs)], 1))


def neural_svd(Xs, Ps, Pts, a, d, steps=250):                 # float64 throughout; d = embedding dim
    Xt = [torch.tensor(v, dtype=torch.float64) for v in Xs]
    net = MVNet([v.shape[1] for v in Xs], d + 1).double()
    a = a.double()
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(steps):
        opt.zero_grad()
        Q, _ = torch.linalg.qr(net(Xt))                       # N x (d+1), orthonormal
        s2 = (apply_op(Pts, a.flip(0), Q) ** 2).sum(0)        # Wᵀ Q column energies = σ²
        (-s2.sum()).backward(); opt.step()
    with torch.no_grad():
        Q, _ = torch.linalg.qr(net(Xt))
        WtQ = apply_op(Pts, a.flip(0), Q)                     # N x (d+1)
        G = WtQ.T @ WtQ                                       # (d+1)x(d+1) = Qᵀ W Wᵀ Q
        ev, O = torch.linalg.eigh(G)
        order = torch.argsort(ev, descending=True)
        emb = (Q @ O[:, order]) * torch.sqrt(torch.clamp(ev[order], min=0))
    return np.nan_to_num(emb[:, 1:d + 1].numpy())             # drop top (constant) mode


def classical(Ps, a, k):
    W = np.eye(Ps[0].shape[0])
    for s in range(a.shape[0]):
        S = sum(a[s, v].item() * Ps[v] for v in range(len(Ps)))
        W = S @ W
    W = np.asarray(W.todense()) if sp.issparse(W) else W
    n = len(W); p = np.ones(n) / n
    for _ in range(200):
        p = p @ W; p /= p.sum()
    spi = np.sqrt(np.maximum(p, 1e-12))
    A = spi[:, None] * W / spi[None, :]
    Tg = A @ A.T - np.outer(spi, spi)
    w, V = np.linalg.eigh((Tg + Tg.T) / 2)
    idx = np.argsort(w)[::-1][1:k + 1]
    return np.nan_to_num(V[:, idx] * np.sqrt(np.clip(w[idx], 0, None)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='MNIST-10k')
    ap.add_argument('--sizes', default='2000,4000,8000,10000')
    ap.add_argument('--dim', type=int, default=0)             # embedding dim; 0 = #classes
    args = ap.parse_args()
    m = scipy.io.loadmat(f'/tmp/Multi-view-datasets/{args.name}.mat')
    Xo = m['X'].ravel(); y = np.asarray(m['y'] if 'y' in m else m['Y']).ravel()
    k = len(np.unique(y)); sizes = [int(s) for s in args.sizes.split(',')]
    d = args.dim if args.dim else k                           # decouple embedding dim from #clusters
    print(f"=== {args.name}: N_total={len(y)} V={len(Xo)} classes={k} embed_dim={d} ===", flush=True)
    rng = np.random.default_rng(0)
    a = torch.softmax(torch.tensor(rng.standard_normal((T, len(Xo))), dtype=torch.float32), 1)
    for N in sizes:
        if N > len(y):
            continue
        idx = rng.choice(len(y), N, replace=False) if N < len(y) else np.arange(len(y))
        Xs = [np.nan_to_num(StandardScaler().fit_transform(np.asarray(v)[idx].astype(float))).astype('float32') for v in Xo]
        yy = y[idx]; knn = int(np.log(N)) + 1
        Ps = [sparse_transition(v, knn) for v in Xs]
        Pts = [torch_sparse(P.T) for P in Ps]

        t0 = time.time()
        emb = neural_svd(Xs, Ps, Pts, a, d)
        t_n = time.time() - t0
        ami_n = AMI(yy, KMeans(k, n_init=10, random_state=0).fit_predict(emb))

        if N <= 8000:
            t0 = time.time()
            E = classical(Ps, a, d)
            t_c = time.time() - t0
            ami_c = AMI(yy, KMeans(k, n_init=10, random_state=0).fit_predict(E))
            cstr = f"classical={t_c:6.1f}s AMI={ami_c:.3f}"
        else:
            cstr = "classical=SKIPPED (dense N x N too costly)"

        print(f"N={N:5d} | neural(sparse)={t_n:6.1f}s AMI={ami_n:.3f} | {cstr}", flush=True)


if __name__ == '__main__':
    main()
