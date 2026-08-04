"""Imitative Gram-match column for Table 8 (tab:collapse-ablation), on its EXACT setup:
same load(name,N), same fixed best-of-8 operator (mdt_operator_from_views t=4 'random',
median bandwidth). Reproduces the table's 'fixed MDT' column, then Gram-matches that operator
with the imitative encoder -> a fair 'imitative' column showing it ties fixed (unlike the
end-to-end trained columns that collapse)."""
import numpy as np, sys
import torch, torch.nn as nn
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI
sys.path.insert(0, '/home/nicolasbigeard/internshipborelli/deep-diffusion-maps')
from experiments.mvmat.t_sweep import load, ami_of
from src.mdt_operators import mdt_operator_from_views


def stationary(W, it=300):
    n = len(W); v = np.ones(n) / n; Wt = W.T
    for _ in range(it):
        v = Wt @ v; s = v.sum(); v = v / (s if s else 1.0)
    return np.clip(v, 1e-12, None)


def neural_ami(X, W, y, k, seed, epochs=800):
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
    F = net(Xt).detach().numpy()
    return AMI(y, KMeans(k, n_init=10, random_state=0).fit_predict(F))


def main():
    for name, N in [('Caltech101-7', None)]:
        Xs, y, k = load(name, N); n = len(y); knn = int(np.log(n)) + 1
        sig = [np.quantile(pdist(v[:200]), 0.5) for v in Xs]
        best, Wstar = -1.0, None
        for j in range(8):
            W = mdt_operator_from_views(Xs, sig, 4, 'random', knn, seed=j)
            a = ami_of(W, k, y)
            if a > best:
                best, Wstar = a, W
        Xall = np.concatenate(Xs, axis=1).astype('float32')
        imit = [neural_ami(Xall, Wstar, y, k, s) for s in (0, 1)]
        print(f"{name:13s} n={n} k={k} | fixed best-of-8 {best:.3f} | "
              f"imitative(Gram) {np.mean(imit):.3f}+/-{np.std(imit):.3f}", flush=True)


if __name__ == '__main__':
    main()
