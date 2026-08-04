"""Imitative Gram-match t-sweep (t=1..4), parallel to the end-to-end t-sweep in Table 7.
For each dataset and each t: fixed(t) = best-of-8 random length-t MDT operator (same recipe as
final_table.py / Table 7); imit(t) = the sqrt(pi)-Gram encoder trained to REPRODUCE that fixed
operator. Shows imit(t) tracks fixed(t) at every length (it imitates a fixed operator), unlike the
end-to-end operator which collapses. Same load(name,N) as Table 7."""
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


def neural_ami(X, W, y, k, seed=0, epochs=700):
    torch.manual_seed(seed)
    sq = np.sqrt(stationary(W))
    G = torch.tensor(sq[:, None] * W * (1.0 / sq)[None, :] @ (sq[:, None] * W * (1.0 / sq)[None, :]).T
                     - np.outer(sq, sq), dtype=torch.float32)
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
    for name, N in [('MSRC-v5', None), ('Caltech101-7', None), ('OutdoorScene', 1500), ('UCI', 1500)]:
        Xs, y, k = load(name, N); n = len(y); knn = int(np.log(n)) + 1
        sig = [np.quantile(pdist(v[:200]), 0.5) for v in Xs]
        Xall = np.concatenate(Xs, axis=1).astype('float32')
        out = []
        for t in (1, 2, 3, 4):
            best, Wstar = -1.0, None
            for j in range(8):
                W = mdt_operator_from_views(Xs, sig, t, 'random', knn, seed=j)
                a = ami_of(W, k, y)
                if a > best:
                    best, Wstar = a, W
            im = neural_ami(Xall, Wstar, y, k)
            out.append((t, best, im))
        s = "  ".join(f"t={t}: fix {b:.3f}/imit {im:.3f}" for t, b, im in out)
        print(f"{name:13s} n={n} k={k} | {s}", flush=True)


if __name__ == '__main__':
    main()
