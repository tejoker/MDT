"""Probe: is the Gram encoder's OOS collapse real, or just my under-regularized MLP?
Add BatchNorm (what the paper's build_mv_encoder uses) and validate IN-SAMPLE first
(must approach the SVD ceiling) before trusting any OOS number."""
import numpy as np, scipy.io, torch, torch.nn as nn, sys
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI
sys.path.insert(0, '/home/nicolasbigeard/internshipborelli/deep-diffusion-maps')
from src.mdt_operators import transition_matrix, oos_transition, _mdt_operator, _trajectory

DATA = '/tmp/Multi-view-datasets'; T = 4


def km(E, k):
    return KMeans(k, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(E))


def gram_target(W):
    n = len(W); p = np.ones(n) / n
    for _ in range(300):
        p = p @ W; p /= p.sum()
    sp = np.sqrt(np.maximum(p, 1e-12)); A = sp[:, None] * W / sp[None, :]
    return A @ A.T - np.outer(sp, sp)


def eig_embed(T_, k):
    w, V = np.linalg.eigh((T_ + T_.T) / 2); idx = np.argsort(w)[::-1][:k]
    return V[:, idx] * np.sqrt(np.clip(w[idx], 0, None))


def enc_bn(din, k):
    return nn.Sequential(nn.Linear(din, 256), nn.BatchNorm1d(256), nn.ReLU(),
                         nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, k))


def gram_fit(Xtr, Xte, G, k, seed, epochs=600, lr=0.01):
    torch.manual_seed(seed)
    xt = torch.tensor(np.concatenate(Xtr, 1), dtype=torch.float32)
    xe = torch.tensor(np.concatenate(Xte, 1), dtype=torch.float32)
    Gt = torch.tensor(G / max(np.linalg.norm(G, 2), 1e-12), dtype=torch.float32)
    net = enc_bn(xt.shape[1], k); opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for _ in range(epochs):
        opt.zero_grad(); F = net(xt)
        (((F @ F.T) - Gt) ** 2).mean().backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(xt).numpy(), net(xe).numpy()


def run(name, seed=0):
    m = scipy.io.loadmat(f'{DATA}/{name}.mat'); Xo = m['X'].ravel()
    y = np.asarray(m['y'] if 'y' in m else m['Y']).ravel(); k = len(np.unique(y))
    N = len(y); perm = np.random.default_rng(seed).permutation(N)
    n_tr = min(600, N // 2); n_te = min(N - n_tr, 400)
    tr, te = perm[:n_tr], perm[n_tr:n_tr + n_te]
    Xtr = [np.nan_to_num(StandardScaler().fit_transform(np.asarray(v)[tr].astype(float))) for v in Xo]
    Xte = [np.nan_to_num(StandardScaler().fit_transform(np.asarray(v)[te].astype(float))) for v in Xo]
    ytr, yte = y[tr], y[te]
    knn = int(np.log(n_tr)) + 1
    sig = [np.quantile(pdist(v[:200]), 0.5) for v in Xtr]
    P = [transition_matrix(v, s, knn) for v, s in zip(Xtr, sig)]
    Pnew = [oos_transition(vtr, vte, s, knn) for vtr, vte, s in zip(Xtr, Xte, sig)]
    a = _trajectory(len(P), T, 'random', seed); W = _mdt_operator(a, P)
    G = gram_target(W)
    ceil = AMI(ytr, km(eig_embed(G, k), k))                       # in-sample SVD ceiling
    Ftr, Fte = gram_fit(Xtr, Xte, G, k, seed)
    g_in = AMI(ytr, km(Ftr, k)); g_out = AMI(yte, km(Fte, k))     # gram in-sample / OOS
    print(f"{name:13s} k={k} n_tr={n_tr} | SVD-ceiling(in) {ceil:.3f} | "
          f"Gram in-sample {g_in:.3f} | Gram OOS {g_out:.3f}", flush=True)


if __name__ == '__main__':
    for nm in ['MSRC-v5', 'Handwritten', 'UCI']:
        run(nm)
