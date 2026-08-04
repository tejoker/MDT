"""Neural SVD of the MDT consensus operator vs exact SVD/eig.

Learns the top-k eigenfunctions of the consensus DDM target
    T̄ = mean_m (A_m A_mᵀ − √π_m√π_mᵀ)
with a multi-view net F = f_θ(x) by the Rayleigh objective
    max tr(Fᵀ T̄ F)  s.t.  FᵀF = I,
i.e. loss = −tr(Fᵀ T̄ F) + ρ‖FᵀF − I‖².  Compared to exact eig(T̄) and to the
single-trajectory mean. Tests whether a parametric neural SVD enhances accuracy
(expectation: it matches eig, since both recover the same top-k subspace).
"""
import argparse
import yaml
import numpy as np
import torch
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI

from experiments.mvmat.load_data import get_data
from src.mdt_operators import transition_matrix, _mdt_operator, _trajectory


def km(E, k):
    return KMeans(k, n_init=10, random_state=0).fit_predict(E)


def gram_target(W):
    n = len(W); p = np.ones(n) / n
    for _ in range(300):
        p = p @ W; p /= p.sum()
    sp = np.sqrt(np.maximum(p, 1e-12))
    A = sp[:, None] * W / sp[None, :]
    return A @ A.T - np.outer(sp, sp)


def eig_embed(T, k):
    w, V = np.linalg.eigh((T + T.T) / 2)
    idx = np.argsort(w)[::-1][:k]
    return V[:, idx] * np.sqrt(np.clip(w[idx], 0, None))


class MVNet(torch.nn.Module):
    def __init__(self, dims, k, h=128):
        super().__init__()
        self.branch = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Linear(h, h)) for d in dims])
        self.head = torch.nn.Sequential(torch.nn.Linear(h * len(dims), h), torch.nn.ReLU(), torch.nn.Linear(h, k))

    def forward(self, Xs):
        return self.head(torch.cat([b(x) for b, x in zip(self.branch, Xs)], dim=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('-c', '--config', required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    k, t, knn = cfg['mdt']['n_components'], cfg['mdt']['steps'], cfg['mdt'].get('knn', 7)
    M = cfg['mdt'].get('search', 20)
    data = get_data(**cfg['data'])
    X, y = data['train'], data['train_color']
    sig = [np.quantile(pdist(v.reshape(len(v), -1)), 0.5) for v in X]
    P = [transition_matrix(v, s, knn) for v, s in zip(X, sig)]

    Ts = [gram_target(_mdt_operator(_trajectory(len(P), t, 'random', s), P)) for s in range(M)]
    singles = [AMI(y, km(eig_embed(T, k), k)) for T in Ts]
    Tbar = np.mean(Ts, axis=0)
    ami_eig = AMI(y, km(eig_embed(Tbar, k), k))                       # exact consensus

    # neural SVD of Tbar
    torch.manual_seed(0)
    Tn = torch.tensor(Tbar / np.linalg.norm(Tbar, 2), dtype=torch.float32)   # unit spectral scale
    Xt = [torch.tensor(v, dtype=torch.float32) for v in X]
    net = MVNet([v.shape[1] for v in X], k)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    I = torch.eye(k)
    for _ in range(400):
        opt.zero_grad()
        F = net(Xt)
        quad = (F * (Tn @ F)).sum()
        orth = ((F.T @ F - I) ** 2).sum()
        loss = -quad + 10.0 * orth
        loss.backward(); opt.step()
    with torch.no_grad():
        ami_nsvd = AMI(y, km(net(Xt).numpy(), k))

    print(f"{cfg['data']['name']:13s} | single-mean={np.mean(singles):.3f}  "
          f"consensus eig(SVD)={ami_eig:.3f}  consensus NEURAL-SVD={ami_nsvd:.3f}")


if __name__ == '__main__':
    main()
