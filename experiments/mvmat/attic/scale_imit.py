"""Parametric-smoothing-at-scale: does imit-Gram beat truncated svds on the SAME operator, and does
the edge grow with N? Same fixed MDT operator per N (best-of-4 random, t=4); classical = truncated
svds embedding (ami_of); neural = sqrt(pi)-Gram MLP reproducing it. MNIST-10k subsamples. If neural-svds
grows with N -> parametric smoothing is the one honest regime where imit > SVD."""
import numpy as np, sys, gc
import torch, torch.nn as nn
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI
sys.path.insert(0, '/home/nicolasbigeard/internshipborelli/deep-diffusion-maps')
from experiments.mvmat.t_sweep import load, ami_of
from src.mdt_operators import mdt_operator_from_views


def stationary(W, it=200):
    n = len(W); v = np.ones(n) / n; Wt = W.T
    for _ in range(it):
        v = Wt @ v; s = v.sum(); v = v / (s if s else 1.0)
    return np.clip(v, 1e-12, None)


def neural_ami(X, W, y, k, seed=0, epochs=700):
    torch.manual_seed(seed)
    sq = np.sqrt(stationary(W))
    A = sq[:, None] * W * (1.0 / sq)[None, :]
    G = torch.tensor(A @ A.T - np.outer(sq, sq), dtype=torch.float32)
    Xt = torch.tensor(X, dtype=torch.float32); St = torch.tensor(sq, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(Xt.shape[1], 256), nn.BatchNorm1d(256), nn.ReLU(),
                        nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, k))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        opt.zero_grad(); F = net(Xt); SF = St[:, None] * F
        (((SF @ SF.T - G) ** 2).mean()).backward(); opt.step(); sch.step()
    return AMI(y, KMeans(k, n_init=10, random_state=0).fit_predict(net(Xt).detach().numpy()))


def main():
    for N in [1500, 3000, 5000]:
        Xs, y, k = load('MNIST-10k', N); n = len(y); knn = int(np.log(n)) + 1
        sig = [np.quantile(pdist(v[:200]), 0.5) for v in Xs]
        best, Wstar = -1.0, None
        for j in range(4):
            W = mdt_operator_from_views(Xs, sig, 4, 'random', knn, seed=j)
            a = ami_of(W, k, y)
            if a > best:
                best, Wstar = a, W
        Xall = np.concatenate(Xs, axis=1).astype('float32')
        im = neural_ami(Xall, Wstar, y, k)
        print(f"N={n:5d} k={k} | svds(classical) {best:.3f} | imit-Gram(neural) {im:.3f} | "
              f"neural-svds {im-best:+.3f}", flush=True)
        gc.collect()


if __name__ == '__main__':
    main()
