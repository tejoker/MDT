"""Does end-to-end learning fail because of a BUILD error (dense kernel) rather than fundamentally?
Faithful rebuild vs the dense DeepMDT:
  - SPARSE kNN kernel (K = mask (x) exp(-gamma d^2), row-normalised) -- MDT's actual operator, not dense.
  - identity-initialised per-view projection (starts at raw-feature distances, not a random 32-d scramble).
  - contrastive loss with a logit temperature tau (unstick the near-uniform-logit flat loss).
Sweep t=1..4 and tau; compare to fixed best-of-8 MDT (same harness). If the faithful build holds
structure at t>1 and approaches fixed MDT -> the dense negative result was a build artifact.
Sequential, one dataset at a time."""
import gc
import numpy as np
import torch
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
from experiments.mvmat.t_sweep import load, ami_of, effrank
from src.mdt_operators import mdt_operator_from_views


def knn_mask(X, knn):
    idx = NearestNeighbors(n_neighbors=knn + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    n = len(X); m = np.zeros((n, n), bool)
    np.put_along_axis(m, idx, True, 1)
    return m                                          # directed kNN; weights symmetrised in _P (MDT: (K+K^T)/2)


class SparseMDT(torch.nn.Module):
    def __init__(self, dims, t, masks, use_proj=True):
        super().__init__()
        self.maskf = [torch.tensor(m, dtype=torch.float32) for m in masks]
        self.log_gamma = torch.nn.Parameter(torch.zeros(len(dims)))
        self.a = torch.nn.Parameter(torch.zeros(t, len(dims)))
        self.proj = None
        if use_proj:
            self.proj = torch.nn.ModuleList([torch.nn.Linear(d, d) for d in dims])
            with torch.no_grad():
                for lin, d in zip(self.proj, dims):
                    lin.weight.copy_(torch.eye(d)); lin.bias.zero_()   # identity init

    def _P(self, X, v):
        z = self.proj[v](X) if self.proj is not None else X
        d2 = torch.cdist(z, z) ** 2
        K = torch.exp(-torch.exp(self.log_gamma[v]) * d2) * self.maskf[v]   # SPARSE (masked kNN)
        K = (K + K.t()) / 2                                                 # MDT symmetrisation (K+K^T)/2
        return K / K.sum(1, keepdim=True).clamp_min(1e-12)

    def operator(self, Xs):
        P = [self._P(X, v) for v, X in enumerate(Xs)]
        steps = torch.einsum('tv,vnm->tnm', torch.softmax(self.a, 1), torch.stack(P))
        W = steps[0]
        for s in range(1, steps.shape[0]):
            W = steps[s] @ W
        return W

    @torch.no_grad()
    def init_bandwidth(self, Xs):
        # MDT max-min bandwidth: bw = max_j (min_{i!=j} dist_v);  kernel exp(-d^2 / bw) => gamma = 1/bw
        for v, X in enumerate(Xs):
            z = self.proj[v](X) if self.proj is not None else X
            D = torch.cdist(z, z)
            D = D + torch.eye(D.shape[0]) * D.max()          # mask self (0) before per-row min
            bw = D.min(1).values.max().clamp_min(1e-6)
            self.log_gamma[v] = -torch.log(bw)


def contrastive(W, lossmasks, tau):
    N = W.shape[0]
    Wc = torch.clamp(torch.nan_to_num(W), -20, 20) * tau
    ex = torch.exp(Wc) * (1 - torch.eye(N))
    logp = torch.log((ex / ex.sum(1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-12))
    return -sum(logp[m].sum() for m in lossmasks) / (len(lossmasks) * N)


def train_sparse(Xs, y, k, t, tau=1.0, use_proj=True, sparse=True, steps=200, lr=0.01, seed=0):
    torch.manual_seed(seed)
    Xt = [torch.tensor(v, dtype=torch.float32) for v in Xs]
    knn = int(np.floor(np.log(len(y))))              # MDT: knn = floor(log N)
    kmasks = [knn_mask(v, knn) for v in Xs]          # kNN structure
    n = len(y); full = ~np.eye(n, dtype=bool)
    opmasks = kmasks if sparse else [full for _ in Xs]   # OPERATOR: sparse kNN vs dense (all pairs)
    lossmasks = [torch.tensor(m) for m in kmasks]        # contrastive positives ALWAYS kNN (isolate kernel)
    model = SparseMDT([v.shape[1] for v in Xs], t, opmasks, use_proj)
    model.init_bandwidth(Xt)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    with torch.no_grad():
        W0 = model.operator(Xt).numpy()
    l0 = None
    for _ in range(steps):
        opt.zero_grad()
        W = model.operator(Xt)
        loss = contrastive(W, lossmasks, tau)
        if l0 is None:
            l0 = float(loss.item())
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    with torch.no_grad():
        Wd = model.operator(Xt).numpy(); lN = float(contrastive(model.operator(Xt), lossmasks, tau))
    return dict(amiN=ami_of(Wd, k, y), ami0=ami_of(W0, k, y), l0=l0, lN=lN, erN=effrank(Wd))


def main():
    for name, N in [('MSRC-v5', None), ('OutdoorScene', 1500), ('UCI', 1500)]:
        Xs, y, k = load(name, N); n = len(y); knn = int(np.log(n)) + 1
        sig = [np.quantile(pdist(v[:200]), 0.5) for v in Xs]
        best8 = max(ami_of(mdt_operator_from_views(Xs, sig, 4, 'random', knn, seed=j), k, y) for j in range(8))
        print(f"\n=== {name} n={n} k={k} | [ref] fixed best-of-8 = {best8:.3f} ===", flush=True)
        for t in (1, 2, 3, 4):
            r = train_sparse(Xs, y, k, t, tau=1.0)
            print(f"  sparse t={t} tau=1   AMI={r['amiN']:.3f} (init {r['ami0']:.3f}) "
                  f"loss {r['l0']:.2f}->{r['lN']:.2f} ({100*(r['lN']-r['l0'])/max(abs(r['l0']),1e-9):+.1f}%) "
                  f"effrank={r['erN']:.0f}", flush=True)
        for tau in (5.0, 20.0):
            r = train_sparse(Xs, y, k, 4, tau=tau)
            print(f"  sparse t=4 tau={tau:<4} AMI={r['amiN']:.3f} (init {r['ami0']:.3f}) "
                  f"loss {r['l0']:.2f}->{r['lN']:.2f} ({100*(r['lN']-r['l0'])/max(abs(r['l0']),1e-9):+.1f}%) "
                  f"effrank={r['erN']:.0f}", flush=True)
        gc.collect()


if __name__ == '__main__':
    main()
