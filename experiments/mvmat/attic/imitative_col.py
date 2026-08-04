"""Imitative Gram-matching encoder vs fixed MDT, on the 4 collapse datasets, to add a column to
tab:collapse-fair showing that the IMITATIVE model reproduces fixed MDT (unlike the end-to-end one).
For each dataset: build the best-of-8 fixed operator (canonical kernel2), report its SVD AMI (fixed),
then train the sqrt(pi)-Gram MLP to reproduce it, report neural AMI (imitative). Multi-seed."""
import numpy as np, scipy.io, sys
import torch, torch.nn as nn
from functools import reduce
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, '/home/nicolasbigeard/internshipborelli/deep-diffusion-maps/experiments/mvmat')
from corrected_enhance import kernel2
from reproduce_paper import get_embedding, ami_runs, knee_power

CAP = 1500


def stationary(W, it=300):
    n = len(W); v = np.ones(n) / n; Wt = W.T
    for _ in range(it):
        v = Wt @ v; s = v.sum(); v = v / (s if s else 1.0)
    return np.clip(v, 1e-12, None)


def neural_ami(X, W, y, k, seed, epochs=800):
    torch.manual_seed(seed)
    sq = np.sqrt(stationary(W))
    G = torch.tensor(sq[:, None] * W * (1.0 / sq)[None, :] @ (sq[:, None] * W * (1.0 / sq)[None, :]).T
                     - np.outer(sq, sq), dtype=torch.float32)
    Xt = torch.tensor(StandardScaler().fit_transform(X), dtype=torch.float32)
    St = torch.tensor(sq, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(Xt.shape[1], 256), nn.BatchNorm1d(256), nn.ReLU(),
                        nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, k))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        opt.zero_grad(); F = net(Xt); SF = St[:, None] * F
        (((SF @ SF.T - G) ** 2).mean()).backward(); opt.step(); sched.step()
    return ami_runs(net(Xt).detach().numpy(), y, k, 10)


def main():
    for name in ['MSRC-v5', 'Caltech101-7', 'OutdoorScene', 'UCI']:
        m = scipy.io.loadmat(f'/tmp/Multi-view-datasets/{name}.mat')
        Xv = [np.asarray(v).astype(float) for v in m['X'].ravel()]
        y = np.asarray(m['y']).ravel()
        if len(y) > CAP:
            idx = np.random.default_rng(0).choice(len(y), CAP, replace=False)
            Xv = [v[idx] for v in Xv]; y = y[idx]
        k = len(np.unique(y)); V = len(Xv)
        P = [kernel2(v, 'global', 'row') for v in Xv]
        Xall = np.concatenate([StandardScaler().fit_transform(v) for v in Xv], axis=1)
        t = max(knee_power(reduce(lambda a, b: a @ b, P)), 1)
        rng = np.random.default_rng(0); best, Wstar = -1.0, None
        for _ in range(8):
            seq = rng.integers(0, V, t)
            W = reduce(lambda a, b: b @ a, [P[i] for i in seq])
            a = ami_runs(get_embedding(W, k), y, k)
            if a > best:
                best, Wstar = a, W
        imit = [neural_ami(Xall, Wstar, y, k, s) for s in (0, 1)]
        print(f"{name:13s} n={len(y)} k={k} | fixed best-of-8 {best:.3f} | "
              f"imitative(Gram) {np.mean(imit):.3f}+/-{np.std(imit):.3f} | gap {np.mean(imit)-best:+.3f}", flush=True)


if __name__ == '__main__':
    main()
