"""Test the failure hypothesis: does the spectral-energy index fail exactly when a view is
CLEAN-BUT-USELESS (high energy, low single-view AMI)? Try heterogeneous / cross-modal datasets.
For each: (a) per-view (dim, spectral energy, single-view AMI) -> spot a clean-but-useless view;
(b) selector rho for Qc/sil/en over a trajectory pool + energy regret -> does energy fail?
Hypothesis: energy fails (rho<=0) iff per-view energy does NOT track per-view AMI (a high-energy low-AMI view)."""
import numpy as np, scipy.io, sys
from functools import reduce
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
sys.path.insert(0, '/home/nicolasbigeard/internshipborelli/deep-diffusion-maps/experiments/mvmat')
from corrected_enhance import kernel2
from iq_broad import contrastive_Q, spectral
from reproduce_paper import get_embedding, ami_runs

CAP, POOL, SEEDS, T = 1000, 25, range(3), 5
CANDIDATES = ['Wikipedia', 'NUS-WIDE', 'Cora', 'CiteSeer', 'Prokaryotic']   # heterogeneous / cross-modal


def load(name):
    m = scipy.io.loadmat(f'/tmp/Multi-view-datasets/{name}.mat')
    Xv = [np.asarray(v).astype(float) for v in m['X'].ravel()]; y = np.asarray(m['y']).ravel()
    if len(y) > CAP:
        idx = np.random.default_rng(0).choice(len(y), CAP, replace=False)
        Xv = [v[idx] for v in Xv]; y = y[idx]
    return Xv, y, len(np.unique(y))


def main():
    for name in CANDIDATES:
        try:
            Xv, y, k = load(name)
        except Exception as e:
            print(f"{name}: LOAD-ERR {e}\n", flush=True); continue
        P = [kernel2(v, 'global', 'row') for v in Xv]
        # (a) per-view energy vs single-view AMI
        print(f"=== {name}  n={len(y)} k={k} views={len(Xv)} ===", flush=True)
        pv = []
        for vi, Pvi in enumerate(P):
            e = spectral(Pvi, k)[0]; a = ami_runs(get_embedding(Pvi, k), y, k, 3)
            pv.append((e, a))
            print(f"   view {vi} dim={Xv[vi].shape[1]:5d}  energy={e:.3f}  single-view AMI={a:.3f}", flush=True)
        # does energy track AMI across views? (Spearman over the few views)
        es = [p[0] for p in pv]; as_ = [p[1] for p in pv]
        cross = spearmanr(es, as_).correlation if len(P) > 2 else (np.sign((es[0]-es[1])*(as_[0]-as_[1])) if len(P) == 2 else np.nan)
        clean_useless = any(e > 0.5 * max(es) and a < 0.5 * max(as_ + [1e-9]) for e, a in pv)  # a high-energy low-AMI view?
        # (b) selector rho over a trajectory pool
        masks = [np.logical_and(p > 0, ~np.eye(len(p), dtype=bool)) for p in P]
        rc, rs, re, rg = [], [], [], []
        for s in SEEDS:
            rng = np.random.default_rng(s); amis, cq, sil, en = [], [], [], []
            for _ in range(POOL):
                W = reduce(lambda A, B: B @ A, [P[i] for i in rng.integers(0, len(P), T)])
                E = get_embedding(W, k); amis.append(ami_runs(E, y, k, 3))
                cq.append(contrastive_Q(W, masks))
                lab = KMeans(k, n_init=1, random_state=0).fit_predict(E)
                sil.append(silhouette_score(E, lab) if len(set(lab)) > 1 else -1.0)
                en.append(spectral(W, k)[0])
            amis = np.array(amis); best = amis.max()
            rc.append(spearmanr(cq, amis).correlation); rs.append(spearmanr(sil, amis).correlation)
            re.append(spearmanr(en, amis).correlation); rg.append(best - amis[int(np.argmax(en))])
        print(f"   selector rho (3 seeds): Qc {np.nanmean(rc):+.2f}  sil {np.nanmean(rs):+.2f}  "
              f"en {np.nanmean(re):+.2f}   | energy regret {np.mean(rg):.3f}", flush=True)
        print(f"   -> clean-but-useless view present? {clean_useless} | energy-tracks-AMI-across-views (sign/rho) {cross:+.2f} "
              f"| ENERGY {'FAILS' if np.nanmean(re) < 0.10 else 'works'}\n", flush=True)


if __name__ == '__main__':
    main()
