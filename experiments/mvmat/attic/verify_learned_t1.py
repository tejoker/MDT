"""Does the LEARNED encoder replicate fixed MDT at t=1? The faithful DDM encoder Gram-matches
a fixed operator's target G = A A^T - sqrt(pi)sqrt(pi)^T (A = Pi^1/2 W Pi^-1/2); by Eckart-Young
its optimum IS that operator's truncated SVD, for ANY operator including a t=1 single view. So a
well-trained encoder must MATCH fixed-MDT's t=1 AMI; a shortfall = optimisation/underfit, never a
better method (ceiling). Test on MSRC: fixed-SVD AMI vs neural-SVD AMI for the best single view
(t=1), convex t=1, and the knee-t trajectory."""
import numpy as np, scipy.io, sys
import torch, torch.nn as nn
from functools import reduce
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, '/home/nicolasbigeard/internshipborelli/deep-diffusion-maps/experiments/mvmat')
from corrected_enhance import kernel2
from reproduce_paper import get_embedding, ami_runs, knee_power


def stationary(W, it=300):
    n = len(W); v = np.ones(n) / n; Wt = W.T
    for _ in range(it):
        v = Wt @ v; s = v.sum(); v = v / (s if s else 1.0)
    return np.clip(v, 1e-12, None)


def neural_svd_ami(X, W, y, k, epochs=1500, seed=0):
    """Faithful DDM neural SVD: MLP(features)->k trained on the sqrt(pi)-scaled Gram loss."""
    torch.manual_seed(seed)
    sq = np.sqrt(stationary(W))
    A = sq[:, None] * W * (1.0 / sq)[None, :]
    G = torch.tensor(A @ A.T - np.outer(sq, sq), dtype=torch.float32)
    Xt = torch.tensor(StandardScaler().fit_transform(X), dtype=torch.float32)
    St = torch.tensor(sq, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(Xt.shape[1], 256), nn.BatchNorm1d(256), nn.ReLU(),
                        nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, k))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        F = net(Xt); SF = St[:, None] * F
        loss = ((SF @ SF.T - G) ** 2).mean()
        loss.backward(); opt.step()
    return ami_runs(net(Xt).detach().numpy(), y, k, 10)


def main():
    m = scipy.io.loadmat('/tmp/Multi-view-datasets/MSRC-v5.mat')
    Xv = [np.asarray(v).astype(float) for v in m['X'].ravel()]
    y = np.asarray(m['y']).ravel(); k = len(np.unique(y)); V = len(Xv)
    P = [kernel2(v, 'global', 'row') for v in Xv]
    Xall = np.concatenate([StandardScaler().fit_transform(v) for v in Xv], axis=1)
    svv = [ami_runs(get_embedding(P[v], k), y, k) for v in range(V)]
    vb = int(np.argmax(svv))
    t = max(knee_power(reduce(lambda a, b: a @ b, P)), 1)

    cases = [
        (f"t=1 best single view (v{vb})", P[vb], Xv[vb]),
        ("t=1 convex fusion (mean P_v)", sum(P) / V, Xall),
        ("AD (product all views)",       reduce(lambda a, b: a @ b, P), Xall),
        (f"knee-t trajectory (t={t})",   reduce(lambda a, b: b @ a, [P[i] for i in
            np.random.default_rng(0).integers(0, V, t)]), Xall),
    ]
    print(f"MSRC k={k} V={V} best-view=v{vb}\n{'case':34s} {'fixed-SVD':>10s} {'neural-SVD':>11s} {'gap':>7s}", flush=True)
    for name, W, X in cases:
        fx = ami_runs(get_embedding(W, k), y, k)
        nn_ = neural_svd_ami(X, W, y, k)
        print(f"{name:34s} {fx:10.3f} {nn_:11.3f} {nn_-fx:+7.3f}", flush=True)


if __name__ == '__main__':
    main()
