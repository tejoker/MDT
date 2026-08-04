"""User's instinct: MDT's Internal Quality index (contrastive-Q, Eq 19-20) has a DDM
connection. Q is a contrastive NLL on the raw operator entries -> weak (paper rho=0.20)
because the diffused operator has near-uniform rows (the same uniform-logit problem that
stalls end-to-end). DDM says the right quantity is the pi-weighted spectral structure of
G = A A^T - sqrt(pi)sqrt(pi)^T, A = Pi^1/2 W Pi^-1/2. Test whether a DDM spectral index
(top-k energy ratio, or eigengap at k) selects trajectories better than contrastive-Q and
silhouette. Report Spearman rho AND selection regret (oracle_AMI - selected_AMI), multi-seed.
Canonical self-loop kernel (kernel2) -- no strawman."""
import numpy as np, sys
from functools import reduce
from scipy.stats import spearmanr
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
sys.path.insert(0, '/home/nicolasbigeard/internshipborelli/deep-diffusion-maps/experiments/mvmat')
from corrected_enhance import kernel2
from enhance_operator import load
from reproduce_paper import get_embedding, ami_runs, knee_power

POOL, SEEDS = 40, [0, 1, 2]


def contrastive_Q(W, masks):
    """MDT Eq 19-20: per-view NLL of kNN neighbours under softmax(W). Return -loss (higher=better)."""
    Wc = np.clip(W, -20, 20); ex = np.exp(Wc); np.fill_diagonal(ex, 0.0)
    denom = np.clip(ex.sum(1, keepdims=True), 1e-12, None)
    logp = np.log(np.clip(ex / denom, 1e-12, None))
    return -np.mean([-logp[m].sum() for m in masks]) / W.shape[0]


def _energy_gap(M, k, fro):
    kk = min(k + 2, len(M) - 1)
    try:
        s = np.sort(svds(M, k=kk, return_singular_vectors=False))[::-1]
    except Exception:
        s = np.linalg.svd(M, compute_uv=False)[:kk]
    total = max(fro - s[0] ** 2, 1e-12)                      # drop trivial top mode (s[0]~1)
    energy = float(np.sum(s[1:k + 1] ** 2) / total)
    gap = float(s[k] / max(s[k + 1], 1e-12)) if len(s) > k + 1 else float(s[k])
    return energy, gap


def ddm_spectral(W, k):
    """pi-weighted (DDM) spectral index of A=Pi^1/2 W Pi^-1/2 vs the plain-W control (no pi).
    Returns (energy_pi, gap_pi, energy_plain). If pi==plain, the win is 'spectral', not 'DDM'."""
    n = len(W); v = np.ones(n) / n; Wt = W.T
    for _ in range(200):
        v = Wt @ v; sv = v.sum(); v = v / (sv if sv else 1.0)
    pi = np.clip(v, 1e-12, None); sq = np.sqrt(pi)
    A = sq[:, None] * W * (1.0 / sq)[None, :]
    e_pi, g_pi = _energy_gap(A, k, float(np.sum(A * A)))
    e_plain, _ = _energy_gap(W, k, float(np.sum(W * W)))     # control: no pi-weighting
    return e_pi, g_pi, e_plain


def main():
    names = sys.argv[1:] or ['MSRC-v5', 'UCI', 'Caltech101-7', 'Handwritten']
    for name in names:
        Xv, y, k = load(name); V = len(Xv)
        P = [kernel2(v, 'global', 'row') for v in Xv]
        masks = [np.logical_and(p > 0, ~np.eye(len(p), dtype=bool)) for p in P]   # true kNN neighbours (paper J_i)
        t = max(knee_power(reduce(lambda a, b: a @ b, P)), 1)
        rows = {'contrastive': [], 'silhouette': [], 'energy(pi)': [], 'gap(pi)': [], 'energy(plain)': []}
        regret = {kk: [] for kk in rows}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            amis, cq, sil, en, gp, enp = [], [], [], [], [], []
            for _ in range(POOL):
                seq = rng.integers(0, V, t)
                W = reduce(lambda a, b: b @ a, [P[i] for i in seq])
                E = get_embedding(W, k)
                amis.append(ami_runs(E, y, k, 5))
                cq.append(contrastive_Q(W, masks))
                lab = KMeans(k, n_init=1, random_state=0).fit_predict(E)
                sil.append(silhouette_score(E, lab) if len(set(lab)) > 1 else -1.0)
                e, g, ep = ddm_spectral(W, k); en.append(e); gp.append(g); enp.append(ep)
            amis = np.array(amis); best = amis.max()
            for tag, idx in [('contrastive', cq), ('silhouette', sil), ('energy(pi)', en),
                             ('gap(pi)', gp), ('energy(plain)', enp)]:
                idx = np.array(idx)
                rows[tag].append(spearmanr(idx, amis).correlation)
                regret[tag].append(best - amis[int(np.argmax(idx))])
        print(f"\n=== {name}  k={k} t={t} pool={POOL} ===", flush=True)
        print(f"  {'index':12s} {'rho(idx,AMI)':>16s} {'selection regret':>18s}", flush=True)
        for tag in rows:
            r = np.array(rows[tag]); rg = np.array(regret[tag])
            print(f"  {tag:12s} {np.nanmean(r):+.2f} +/- {np.nanstd(r):.2f}   "
                  f"{np.mean(rg):.3f} +/- {np.std(rg):.3f}", flush=True)


if __name__ == '__main__':
    main()
