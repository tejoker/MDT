"""Deep MDT-Cst: learn the multi-view diffusion geometry end-to-end.

Unlike the imitative encoder (which reproduces a frozen operator), here the
network is UPSTREAM of the MDT operator: small per-view projections + learnable
bandwidths define the kernels, learnable convex weights fuse the trajectory, and
the whole operator W = S_t...S_1 is a differentiable layer trained on the MDT
contrastive loss (eq 19-20). Ablation vs fixed-kernel baselines tells us whether
a *learned* geometry beats a *fixed* one.
"""
import argparse
import yaml
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.neighbors import NearestNeighbors

from experiments.mvmat.load_data import get_data
from src.mdt_operators import mdt_operator_from_views


def knn_masks(Xs, knn):
    masks = []
    for X in Xs:
        idx = NearestNeighbors(n_neighbors=knn + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
        m = np.zeros((len(X), len(X)), dtype=bool)
        np.put_along_axis(m, idx, True, axis=1)
        masks.append(m)
    return masks


def embed_ami(W, k, y):
    U, s, _ = np.linalg.svd(W, full_matrices=False)
    E = U[:, 1:k + 1] * s[1:k + 1]
    return AMI(y, KMeans(k, n_init=10, random_state=0).fit_predict(E))


class DeepMDT(torch.nn.Module):
    def __init__(self, dims, t, r=32):
        super().__init__()
        self.proj = torch.nn.ModuleList([torch.nn.Linear(d, r) for d in dims])
        self.log_gamma = torch.nn.Parameter(torch.zeros(len(dims)))
        self.a = torch.nn.Parameter(torch.zeros(t, len(dims)))

    def operator(self, Xs):
        P = []
        for v, X in enumerate(Xs):
            z = self.proj[v](X)
            d2 = torch.cdist(z, z) ** 2
            K = torch.exp(-torch.exp(self.log_gamma[v]) * d2)
            P.append(K / K.sum(1, keepdim=True).clamp_min(1e-12))
        steps = torch.einsum('tv,vnm->tnm', torch.softmax(self.a, 1), torch.stack(P))
        W = steps[0]
        for s in range(1, steps.shape[0]):
            W = steps[s] @ W
        return W

    @torch.no_grad()
    def init_bandwidth(self, Xs):                       # set gamma_v = 1/median(dist)^2
        for v, X in enumerate(Xs):
            d2 = torch.cdist(self.proj[v](X), self.proj[v](X)) ** 2
            med = d2[d2 > 0].median().clamp_min(1e-6)
            self.log_gamma[v] = -torch.log(med)


def contrastive(W, masks):
    Wc = torch.clamp(torch.nan_to_num(W), -20, 20)
    ex = torch.exp(Wc) * (1 - torch.eye(W.shape[0], device=W.device))
    logp = torch.log((ex / ex.sum(1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-12))
    return -sum(logp[m].sum() for m in masks) / (len(masks) * W.shape[0])


def main():
    p = argparse.ArgumentParser(); p.add_argument('-c', '--config', required=True)
    cfg = yaml.safe_load(open(p.parse_args().config))
    k, t, knn = cfg['mdt']['n_components'], cfg['mdt']['steps'], cfg['mdt'].get('knn', 7)
    data = get_data(**cfg['data'])
    Xs, y = data['train'], data['train_color']
    sigmas = [np.quantile(np.linalg.norm(X[:200, None] - X[None, :200], axis=2), 0.5) for X in Xs]

    # baselines (fixed kernels): random fusion, and MDT-Cst (learned scalar weights)
    ami_rand = embed_ami(mdt_operator_from_views(Xs, sigmas, t, 'random', knn), k, y)
    ami_cst = embed_ami(mdt_operator_from_views(Xs, sigmas, t, 'contrastive', knn), k, y)

    # deep fusion: learned kernels + learned weights
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    Xt = [torch.tensor(X, dtype=torch.float32, device=dev) for X in Xs]
    masks = [torch.tensor(m, device=dev) for m in knn_masks(Xs, knn)]
    model = DeepMDT([X.shape[1] for X in Xs], t).to(dev)
    model.init_bandwidth(Xt)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for step in range(150):
        opt.zero_grad()
        loss = contrastive(model.operator(Xt), masks)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    with torch.no_grad():
        ami_deep = embed_ami(model.operator(Xt).cpu().numpy(), k, y)

    print(f"{cfg['data']['name']:13s} | fixed+random={ami_rand:.3f}  fixed+learned(MDT-Cst)={ami_cst:.3f}  "
          f"LEARNED kernel+weights={ami_deep:.3f}")


if __name__ == '__main__':
    main()
