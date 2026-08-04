"""T6 deep-dig: instrument the end-to-end collapse instead of inferring it from AMI.
Logs contrastive loss AND AMI over training (did the loss go DOWN while AMI died?),
measures the operator's effective rank and projected-feature std at each checkpoint
(does it go rank-1 / do features collapse?), and tests whether a feature-variance
regulariser prevents the collapse (structural vs fixable)."""
import numpy as np
import scipy.io
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI
import experiments.mvmat.deep_fusion as DF


def load(name, N=None, seed=0):
    m = scipy.io.loadmat(f'/tmp/Multi-view-datasets/{name}.mat'); Xo = m['X'].ravel()
    y = np.asarray(m['y'] if 'y' in m else m['Y']).ravel(); k = len(np.unique(y))
    idx = (np.arange(len(y)) if (N is None or N >= len(y))
           else np.random.default_rng(seed).choice(len(y), N, replace=False))
    Xs = [np.nan_to_num(StandardScaler().fit_transform(np.asarray(v)[idx].astype(float))).astype('float32') for v in Xo]
    return Xs, y[idx], k


def embed(W, k):
    U, s, _ = np.linalg.svd(W, full_matrices=False)
    return U[:, 1:k + 1] * s[1:k + 1]


def train(Xs, y, k, steps=200, reg=0.0, seed=0, lr=0.01):
    torch.manual_seed(seed)
    Xt = [torch.tensor(v, dtype=torch.float32) for v in Xs]
    masks = [torch.tensor(m) for m in DF.knn_masks(Xs, int(np.log(len(y))) + 1)]
    model = DF.DeepMDT([v.shape[1] for v in Xs], 4); model.init_bandwidth(Xt)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log = []
    for step in range(steps):
        opt.zero_grad()
        W = model.operator(Xt)
        closs = DF.contrastive(W, masks)
        loss = closs
        if reg > 0:
            vp = sum(torch.relu(1 - model.proj[v](Xt[v]).std(0)).mean() for v in range(len(Xt)))
            loss = loss + reg * vp
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 40 == 0 or step == steps - 1:
            with torch.no_grad():
                Wd = W.detach().numpy()
                ami = AMI(y, KMeans(k, n_init=5, random_state=0).fit_predict(embed(Wd, k)))
                sv = np.linalg.svd(Wd, compute_uv=False); sv = sv[sv > 0]; sv = sv / sv.sum()
                effrank = float(np.exp(-(sv * np.log(sv)).sum()))     # spectral-entropy effective rank
                fstd = float(np.mean([model.proj[v](Xt[v]).std().item() for v in range(len(Xt))]))
                log.append((step, float(closs.item()), ami, effrank, fstd))
    return log


if __name__ == '__main__':
    for name, N in [('MSRC-v5', None), ('Caltech101-7', None), ('OutdoorScene', 1500), ('UCI', 1500)]:
        Xs, y, k = load(name, N)
        log = train(Xs, y, k, reg=0.0)
        s0, l0, a0, r0, f0 = log[0]; sN, lN, aN, rN, fN = log[-1]
        print(f"{name:12s} (n={len(y)},k={k}) | loss {l0:.3f}->{lN:.3f} ({100*(lN-l0)/l0:+.1f}%) | "
              f"AMI {a0:.3f}->{aN:.3f} | effrank {r0:.0f}->{rN:.0f} | fstd {f0:.2f}->{fN:.2f}", flush=True)
    # fixability: does longer training / lower lr unstick the loss on Caltech?
    Xs, y, k = load('Caltech101-7', None)
    long = train(Xs, y, k, steps=1000, reg=0.0)
    print(f"Caltech 1000-step | loss {long[0][1]:.3f}->{long[-1][1]:.3f} | AMI {long[0][2]:.3f}->{long[-1][2]:.3f} "
          f"(if loss still flat -> the stall is not just too-few-steps)", flush=True)
