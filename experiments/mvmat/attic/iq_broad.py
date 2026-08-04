"""Broad MDT trajectory-selector study: does a SPECTRAL internal-quality index (top-k
energy ratio / eigengap of the operator) select trajectories better than MDT's contrastive-Q
(Eq 19-20) and silhouette, across many datasets? (The pi-weighting of DDM was shown inert:
pi ~ uniform on diffused operators, so energy(pi)==energy(plain); we use plain-W spectral.)
Metric that matters = selection REGRET (oracle_AMI - AMI of the argmax-index trajectory);
also Spearman rho. Per-dataset mean+/-std over seeds, plus an AGGREGATE over datasets with
real structure (oracle AMI >= 0.35 -- no selector can work below that). Canonical self-loop
kernel. n capped at 2000 by subsample to keep the large sets runnable."""
import numpy as np, scipy.io, sys, os
from functools import reduce
from scipy.stats import spearmanr
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI, silhouette_score
sys.path.insert(0, '/home/nicolasbigeard/internshipborelli/deep-diffusion-maps/experiments/mvmat')
from corrected_enhance import kernel2
from reproduce_paper import get_embedding, ami_runs, knee_power

POOL, SEEDS, CAP = 30, [0, 1, 2], 2000
DIM = int(os.environ.get('IQ_DIM', '0'))          # embedding dim; 0 -> use k (cluster count, = Table 7)
T_OVR = int(os.environ.get('IQ_T', '0'))          # trajectory length override; 0 -> knee-selected
DATASETS = ['MSRC-v5', 'Yale', '3Sources', 'BBCSport', 'Prokaryotic', 'Movies', 'Wikipedia-test',
            'ProteinFold', 'WebKB', 'Reuters-1200', 'Reuters-1500', 'Caltech101-7', '100Leaves',
            'Handwritten', 'UCI', 'NUS-WIDE', 'OutdoorScene', 'Cora', 'Wikipedia', 'ACM', 'CiteSeer']


def load2(name):
    m = scipy.io.loadmat(f'/tmp/Multi-view-datasets/{name}.mat')
    Xv = [np.asarray(v).astype(float) for v in m['X'].ravel()]
    y = np.asarray(m['y']).ravel(); n = len(y)
    if n > CAP:
        idx = np.random.default_rng(0).choice(n, CAP, replace=False)
        Xv = [v[idx] for v in Xv]; y = y[idx]
    return Xv, y, len(np.unique(y))


def contrastive_Q(W, masks):
    Wc = np.clip(W, -20, 20); ex = np.exp(Wc); np.fill_diagonal(ex, 0.0)
    denom = np.clip(ex.sum(1, keepdims=True), 1e-12, None)
    logp = np.log(np.clip(ex / denom, 1e-12, None))
    return -np.mean([-logp[m].sum() for m in masks]) / W.shape[0]


def spectral(W, k):
    kk = min(k + 2, len(W) - 1)
    try:
        s = np.sort(svds(W, k=kk, return_singular_vectors=False))[::-1]
    except Exception:
        s = np.linalg.svd(W, compute_uv=False)[:kk]
    total = max(float(np.sum(W * W)) - s[0] ** 2, 1e-12)
    energy = float(np.sum(s[1:k + 1] ** 2) / total)                 # top-k(=d) energy, drop trivial mode
    gap = float(s[k] / max(s[k + 1], 1e-12)) if len(s) > k + 1 else float(s[k])
    return energy, gap


def main():
    names = sys.argv[1:] or DATASETS
    agg = {t: {'rho': [], 'regret': []} for t in ['contrastive', 'silhouette', 'energy', 'gap']}
    wins = {t: 0 for t in agg}; structured = 0
    for name in names:
        try:
            Xv, y, k = load2(name)
        except Exception as e:
            print(f"{name:14s} LOAD-ERR {e}", flush=True); continue
        V = len(Xv); P = [kernel2(v, 'global', 'row') for v in Xv]
        masks = [np.logical_and(p > 0, ~np.eye(len(p), dtype=bool)) for p in P]
        t = T_OVR or max(knee_power(reduce(lambda a, b: a @ b, P)), 1)
        d = min(DIM, len(y) - 3) if DIM else k          # embedding dim (Table 7 uses k); clustering stays k
        res = {tt: {'rho': [], 'regret': []} for tt in agg}; oracles = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            amis, cq, sil, en, gp = [], [], [], [], []
            for _ in range(POOL):
                seq = rng.integers(0, V, t)
                W = reduce(lambda a, b: b @ a, [P[i] for i in seq])
                E = get_embedding(W, d); amis.append(ami_runs(E, y, k, 5))
                cq.append(contrastive_Q(W, masks))
                lab = KMeans(k, n_init=1, random_state=0).fit_predict(E)
                sil.append(silhouette_score(E, lab) if len(set(lab)) > 1 else -1.0)
                e, g = spectral(W, d); en.append(e); gp.append(g)
            amis = np.array(amis); best = amis.max(); oracles.append(best)
            for tt, idx in [('contrastive', cq), ('silhouette', sil), ('energy', en), ('gap', gp)]:
                res[tt]['rho'].append(spearmanr(np.array(idx), amis).correlation)
                res[tt]['regret'].append(best - amis[int(np.argmax(idx))])
        omean = float(np.mean(oracles))
        tag = "structured" if omean >= 0.35 else "NO-STRUCTURE"
        print(f"\n=== {name}  n={len(y)} k={k} dim={d} t={t}  oracleAMI~{omean:.2f} [{tag}] ===", flush=True)
        best_regret, best_tt = 1e9, None
        for tt in agg:
            r = np.nanmean(res[tt]['rho']); rg = float(np.mean(res[tt]['regret']))
            print(f"  {tt:12s} rho {r:+.2f}  regret {rg:.3f}", flush=True)
            if omean >= 0.35:
                agg[tt]['rho'].append(r); agg[tt]['regret'].append(rg)
                if rg < best_regret: best_regret, best_tt = rg, tt
        if omean >= 0.35:
            structured += 1; wins[best_tt] += 1
    print(f"\n===== AGGREGATE over {structured} structured datasets (oracleAMI>=0.35) =====", flush=True)
    print(f"  {'index':12s} {'mean rho':>10s} {'mean regret':>12s} {'#best-regret':>13s}", flush=True)
    for tt in agg:
        print(f"  {tt:12s} {np.mean(agg[tt]['rho']):+.3f}   {np.mean(agg[tt]['regret']):.3f}"
              f"      {wins[tt]}/{structured}", flush=True)


if __name__ == '__main__':
    main()
