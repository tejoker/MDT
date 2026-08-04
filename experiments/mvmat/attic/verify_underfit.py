"""Is the learned-vs-fixed AMI gap a WALL or just UNDERFIT? Escalate encoder effort and show the
gap shrink toward 0 (never below -> ceiling). Since W=U S V^T, the embedding U*S = W V is LINEAR in
the operator's rows, so a net fed the OPERATOR ROW can reach the SVD exactly; a net fed raw features
must learn features->embedding from scratch. Configs: (a) raw feats small/short (baseline),
(b) raw feats big/long, (c) operator-row input. Report AMI (mean+/-std, 3 seeds), subspace cos, gap
vs fixed, and final Gram loss. Faithful sqrt(pi)-Gram loss throughout. MSRC t=1 best view + knee-t."""
import numpy as np, scipy.io, sys
import torch, torch.nn as nn
from functools import reduce
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, '/home/nicolasbigeard/internshipborelli/deep-diffusion-maps/experiments/mvmat')
from corrected_enhance import kernel2
from reproduce_paper import get_embedding, ami_runs, knee_power

SEEDS = [0, 1, 2]


def stationary(W, it=300):
    n = len(W); v = np.ones(n) / n; Wt = W.T
    for _ in range(it):
        v = Wt @ v; s = v.sum(); v = v / (s if s else 1.0)
    return np.clip(v, 1e-12, None)


def train(Inp, W, k, seed, hidden, layers, epochs, lr=1e-2):
    torch.manual_seed(seed)
    sq = np.sqrt(stationary(W))
    A = sq[:, None] * W * (1.0 / sq)[None, :]
    G = torch.tensor(A @ A.T - np.outer(sq, sq), dtype=torch.float32)
    Xt = torch.tensor(StandardScaler().fit_transform(Inp), dtype=torch.float32)
    St = torch.tensor(sq, dtype=torch.float32)
    mods = [nn.Linear(Xt.shape[1], hidden), nn.BatchNorm1d(hidden), nn.ReLU()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU()]
    mods += [nn.Linear(hidden, k)]
    net = nn.Sequential(*mods)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    last = 0.0
    for _ in range(epochs):
        opt.zero_grad(); F = net(Xt); SF = St[:, None] * F
        loss = ((SF @ SF.T - G) ** 2).mean(); loss.backward(); opt.step(); sched.step()
        last = float(loss)
    return net(Xt).detach().numpy(), last


def subcos(A, B):
    Qa, _ = np.linalg.qr(A); Qb, _ = np.linalg.qr(B)
    return float(np.mean(np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0, 1)))


def main():
    m = scipy.io.loadmat('/tmp/Multi-view-datasets/MSRC-v5.mat')
    Xv = [np.asarray(v).astype(float) for v in m['X'].ravel()]
    y = np.asarray(m['y']).ravel(); k = len(np.unique(y)); V = len(Xv)
    P = [kernel2(v, 'global', 'row') for v in Xv]
    Xall = np.concatenate([StandardScaler().fit_transform(v) for v in Xv], axis=1)
    vb = int(np.argmax([ami_runs(get_embedding(P[v], k), y, k) for v in range(V)]))
    t = max(knee_power(reduce(lambda a, b: a @ b, P)), 1)
    cases = [(f"t=1 best view v{vb}", P[vb], Xv[vb]),
             (f"knee-t (t={t})", reduce(lambda a, b: b @ a,
              [P[i] for i in np.random.default_rng(0).integers(0, V, t)]), Xall)]
    for cname, W, Xfeat in cases:
        Efix = get_embedding(W, k); fx = ami_runs(Efix, y, k)
        print(f"\n=== {cname}  fixed-MDT AMI {fx:.3f} ===", flush=True)
        configs = [("raw feats 256x2 1500ep", Xfeat, 256, 2, 1500),
                   ("raw feats 512x3 6000ep", Xfeat, 512, 3, 6000),
                   ("operator-row 512x2 4000ep", W.copy(), 512, 2, 4000)]
        for label, Inp, h, L, ep in configs:
            amis, cos, gl = [], [], []
            for s in SEEDS:
                F, loss = train(Inp, W, k, s, h, L, ep)
                amis.append(ami_runs(F, y, k, 10)); cos.append(subcos(Efix, F)); gl.append(loss)
            print(f"  {label:28s} AMI {np.mean(amis):.3f}+/-{np.std(amis):.3f}  "
                  f"gap {np.mean(amis)-fx:+.3f}  cos {np.mean(cos):.3f}  gramloss {np.mean(gl):.2e}", flush=True)


if __name__ == '__main__':
    main()
