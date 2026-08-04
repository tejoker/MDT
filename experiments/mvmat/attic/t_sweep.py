"""
(1) short product length t (2-3) so W doesn't diffuse to the stationary (uniform) op;
(2) make the effective length learnable+regularizable via per-step laziness gates
    S_s = g_s*(mix P_v) + (1-g_s)*I,  loss + lambda*sum(sigmoid(gate))  (ridge-like on length).
Compare learned-geometry AMI at t=2,3,4 and gated-t=4 against fixed+random (the Table-6 baseline).
Report AMI, contrastive-loss movement (did gradient exist?), and effective rank (wash-out proxy)."""
import numpy as np, scipy.io, torch
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI
import experiments.mvmat.deep_fusion as DF
from src.mdt_operators import mdt_operator_from_views


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


def ami_of(W, k, y):
    return AMI(y, KMeans(k, n_init=10, random_state=0).fit_predict(embed(W, k)))


def effrank(W):
    sv = np.linalg.svd(W, compute_uv=False); sv = sv[sv > 0]; sv = sv / sv.sum()
    return float(np.exp(-(sv * np.log(sv)).sum()))


class GatedMDT(DF.DeepMDT):
    def __init__(self, dims, t, r=32):
        super().__init__(dims, t, r)
        self.gate = torch.nn.Parameter(torch.zeros(t))     # sigmoid(0)=0.5 start

    def operator(self, Xs):
        P = []
        for v, X in enumerate(Xs):
            z = self.proj[v](X)
            d2 = torch.cdist(z, z) ** 2
            K = torch.exp(-torch.exp(self.log_gamma[v]) * d2)
            P.append(K / K.sum(1, keepdim=True).clamp_min(1e-12))
        steps = torch.einsum('tv,vnm->tnm', torch.softmax(self.a, 1), torch.stack(P))
        g = torch.sigmoid(self.gate); N = steps.shape[1]; I = torch.eye(N)
        W = g[0] * steps[0] + (1 - g[0]) * I
        for s in range(1, steps.shape[0]):
            W = (g[s] * steps[s] + (1 - g[s]) * I) @ W
        return W


def train(Xs, y, k, t, steps=200, lr=0.01, gate=False, lam=0.0, seed=0):
    torch.manual_seed(seed)
    Xt = [torch.tensor(v, dtype=torch.float32) for v in Xs]
    masks = [torch.tensor(m) for m in DF.knn_masks(Xs, int(np.log(len(y))) + 1)]
    model = (GatedMDT if gate else DF.DeepMDT)([v.shape[1] for v in Xs], t)
    model.init_bandwidth(Xt)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    with torch.no_grad():
        W0 = model.operator(Xt).numpy()
    l0 = None
    for step in range(steps):
        opt.zero_grad()
        W = model.operator(Xt)
        closs = DF.contrastive(W, masks)
        if l0 is None:
            l0 = float(closs.item())
        loss = closs + (lam * torch.sigmoid(model.gate).sum() if gate and lam > 0 else 0.0)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    with torch.no_grad():
        W = model.operator(Xt); lN = float(DF.contrastive(W, masks).item()); Wd = W.numpy()
        gmean = float(torch.sigmoid(model.gate).mean()) if gate else float('nan')
    return dict(amiN=ami_of(Wd, k, y), l0=l0, lN=lN, erN=effrank(Wd), gmean=gmean)


def main():
    for name, N in [('MSRC-v5', None), ('Caltech101-7', None), ('OutdoorScene', 1500), ('UCI', 1500)]:
        Xs, y, k = load(name, N); n = len(y)
        knn = int(np.log(n)) + 1
        sig = [np.quantile(pdist(v[:200]), 0.5) for v in Xs]
        ref4 = ami_of(mdt_operator_from_views(Xs, sig, 4, 'random', knn), k, y)
        print(f"\n=== {name}  n={n} k={k} | fixed+random(t=4) AMI={ref4:.3f} ===", flush=True)
        for t in (2, 3, 4):
            r = train(Xs, y, k, t)
            print(f"  learned t={t}      AMI={r['amiN']:.3f}  loss {r['l0']:.2f}->{r['lN']:.2f} "
                  f"({100*(r['lN']-r['l0'])/r['l0']:+.1f}%)  effrank={r['erN']:.0f}", flush=True)
        for lam in (0.1, 1.0):
            r = train(Xs, y, k, 4, gate=True, lam=lam)
            print(f"  gated t=4 lam={lam:<4} AMI={r['amiN']:.3f}  loss {r['l0']:.2f}->{r['lN']:.2f} "
                  f"({100*(r['lN']-r['l0'])/r['l0']:+.1f}%)  effrank={r['erN']:.0f}  gate_mean={r['gmean']:.2f}", flush=True)


if __name__ == '__main__':
    main()
