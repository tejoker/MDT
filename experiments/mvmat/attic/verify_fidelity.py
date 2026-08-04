"""Is the learned encoder 'not replicating well', or is the AMI gap just KMeans noise?
AMI is an indirect, noisy proxy. The DIRECT fidelity test is the subspace overlap between the
learned k-dim embedding and the fixed-MDT SVD subspace (cosines of principal angles: 1.0 = identical
subspace). Report, multi-seed on MSRC t=1 best view + knee-t: fixed AMI, neural AMI (mean+/-std),
and subspace mean/min cos. If mean-cos ~1 the encoder DOES replicate the SVD and the AMI wobble is
KMeans; if cos is low it genuinely fails to replicate."""
import numpy as np, scipy.io, sys
import torch, torch.nn as nn
from functools import reduce
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, '/home/nicolasbigeard/internshipborelli/deep-diffusion-maps/experiments/mvmat')
from corrected_enhance import kernel2
from reproduce_paper import get_embedding, ami_runs, knee_power

SEEDS = [0, 1, 2, 3]


def stationary(W, it=300):
    n = len(W); v = np.ones(n) / n; Wt = W.T
    for _ in range(it):
        v = Wt @ v; s = v.sum(); v = v / (s if s else 1.0)
    return np.clip(v, 1e-12, None)


def neural_embed(X, W, k, seed, epochs=1500):
    torch.manual_seed(seed)
    sq = np.sqrt(stationary(W))
    G = torch.tensor(sq[:, None] * W * (1.0 / sq)[None, :] @ (sq[:, None] * W * (1.0 / sq)[None, :]).T
                     - np.outer(sq, sq), dtype=torch.float32)
    Xt = torch.tensor(StandardScaler().fit_transform(X), dtype=torch.float32)
    St = torch.tensor(sq, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(Xt.shape[1], 256), nn.BatchNorm1d(256), nn.ReLU(),
                        nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, k))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad(); F = net(Xt); SF = St[:, None] * F
        (((SF @ SF.T - G) ** 2).mean()).backward(); opt.step()
    return net(Xt).detach().numpy()


def subspace_cos(A, B):
    """Cosines of principal angles between column spaces of A, B (both N x k)."""
    Qa, _ = np.linalg.qr(A); Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1))), float(np.min(np.clip(s, 0, 1)))


def main():
    m = scipy.io.loadmat('/tmp/Multi-view-datasets/MSRC-v5.mat')
    Xv = [np.asarray(v).astype(float) for v in m['X'].ravel()]
    y = np.asarray(m['y']).ravel(); k = len(np.unique(y)); V = len(Xv)
    P = [kernel2(v, 'global', 'row') for v in Xv]
    Xall = np.concatenate([StandardScaler().fit_transform(v) for v in Xv], axis=1)
    vb = int(np.argmax([ami_runs(get_embedding(P[v], k), y, k) for v in range(V)]))
    t = max(knee_power(reduce(lambda a, b: a @ b, P)), 1)
    cases = [(f"t=1 best view (v{vb})", P[vb], Xv[vb]),
             (f"knee-t (t={t})", reduce(lambda a, b: b @ a,
              [P[i] for i in np.random.default_rng(0).integers(0, V, t)]), Xall)]
    print(f"MSRC k={k}  ({len(SEEDS)} seeds)\n"
          f"{'case':22s} {'fixedAMI':>9s} {'neuralAMI':>16s} {'subspace cos(mean/min)':>24s}", flush=True)
    for name, W, X in cases:
        Efix = get_embedding(W, k); fx = ami_runs(Efix, y, k)
        amis, mc, nc = [], [], []
        for s in SEEDS:
            F = neural_embed(X, W, k, s)
            amis.append(ami_runs(F, y, k, 10))
            mean_c, min_c = subspace_cos(Efix, F); mc.append(mean_c); nc.append(min_c)
        print(f"{name:22s} {fx:9.3f} {np.mean(amis):7.3f}+/-{np.std(amis):.3f}   "
              f"mean {np.mean(mc):.3f}  min {np.mean(nc):.3f}", flush=True)


if __name__ == '__main__':
    main()
