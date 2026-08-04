"""Deep dive on the Internal-Quality index (Table 9), FAST version. Is spectral energy's advantage
ROBUST or noise? 3 seeds x pool 25 per dataset, n<=1000, fixed t=5 (avoids the costly knee_power).
Prints per-dataset rho mean+/-std as it goes; then PAIRED (energy-others) per (dataset,seed), aggregate
win-rates, and the Wikipedia failure mechanism (per-view spectral energy vs per-view AMI)."""
import numpy as np, scipy.io, sys
from functools import reduce
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
sys.path.insert(0, '/home/nicolasbigeard/internshipborelli/deep-diffusion-maps/experiments/mvmat')
from corrected_enhance import kernel2
from iq_broad import contrastive_Q, spectral
from reproduce_paper import get_embedding, ami_runs

SEEDS, POOL, CAP, T = range(3), 25, 1000, 5
STRUCT = ['MSRC-v5', 'Yale', 'BBCSport', 'Caltech101-7', 'Handwritten', 'UCI', 'Wikipedia-test']


def load(name):
    m = scipy.io.loadmat(f'/tmp/Multi-view-datasets/{name}.mat')
    Xv = [np.asarray(v).astype(float) for v in m['X'].ravel()]; y = np.asarray(m['y']).ravel()
    if len(y) > CAP:
        idx = np.random.default_rng(0).choice(len(y), CAP, replace=False)
        Xv = [v[idx] for v in Xv]; y = y[idx]
    return Xv, y, len(np.unique(y))


def eval_pool(P, masks, y, k, seed):
    rng = np.random.default_rng(seed); V = len(P)
    amis, cq, sil, en, gp = [], [], [], [], []
    for _ in range(POOL):
        W = reduce(lambda a, b: b @ a, [P[i] for i in rng.integers(0, V, T)])
        E = get_embedding(W, k); amis.append(ami_runs(E, y, k, 3))
        cq.append(contrastive_Q(W, masks))
        lab = KMeans(k, n_init=1, random_state=0).fit_predict(E)
        sil.append(silhouette_score(E, lab) if len(set(lab)) > 1 else -1.0)
        e, g = spectral(W, k); en.append(e); gp.append(g)
    amis = np.array(amis); best = amis.max()
    return {tag: (spearmanr(np.array(idx), amis).correlation, float(best - amis[int(np.argmax(idx))]))
            for tag, idx in [('Qc', cq), ('sil', sil), ('en', en), ('gap', gp)]}


def main():
    rho = {t: {d: [] for d in STRUCT} for t in ['Qc', 'sil', 'en', 'gap']}
    print(f"PER-DATASET Spearman rho (mean+/-std, {len(list(SEEDS))} seeds, pool {POOL}, n<={CAP}, t={T})")
    print(f"{'dataset':14s} {'Qc':>12s} {'sil':>12s} {'energy':>12s}", flush=True)
    for d in STRUCT:
        Xv, y, k = load(d)
        P = [kernel2(v, 'global', 'row') for v in Xv]
        masks = [np.logical_and(p > 0, ~np.eye(len(p), dtype=bool)) for p in P]
        for s in SEEDS:
            o = eval_pool(P, masks, y, k, s)
            for tg in rho:
                rho[tg][d].append(o[tg][0])
        f = lambda tg: f"{np.nanmean(rho[tg][d]):+.2f}+/-{np.nanstd(rho[tg][d]):.2f}"
        print(f"{d:14s} {f('Qc'):>12s} {f('sil'):>12s} {f('en'):>12s}", flush=True)

    ena = np.array([rho['en'][d] for d in STRUCT]).ravel()
    cqa = np.array([rho['Qc'][d] for d in STRUCT]).ravel()
    sla = np.array([rho['sil'][d] for d in STRUCT]).ravel()
    gpa = np.array([rho['gap'][d] for d in STRUCT]).ravel()
    n = len(ena)
    print(f"\nPAIRED over {n} (dataset,seed) points (energy rho minus other's rho):", flush=True)
    print(f"  vs contrastive: {np.nanmean(ena-cqa):+.3f} +/- {np.nanstd(ena-cqa):.3f}  (energy higher {100*np.mean(ena>cqa):.0f}%)")
    print(f"  vs silhouette : {np.nanmean(ena-sla):+.3f} +/- {np.nanstd(ena-sla):.3f}  (energy higher {100*np.mean(ena>sla):.0f}%)")
    print(f"  vs gap        : {np.nanmean(ena-gpa):+.3f} +/- {np.nanstd(ena-gpa):.3f}  (energy higher {100*np.mean(ena>gpa):.0f}%)")
    winner = np.nanargmax(np.vstack([cqa, sla, ena, gpa]), 0); nm = ['Qc', 'sil', 'en', 'gap']
    print("\nAGGREGATE mean rho: " + "  ".join(f"{a} {np.nanmean(b):+.2f}" for a, b in zip(nm, [cqa, sla, ena, gpa])))
    print("top-ranker win-rate: " + "  ".join(f"{nm[i]} {100*np.mean(winner==i):.0f}%" for i in range(4)))
    print(f"energy rho > 0 in {100*np.mean(ena>0):.0f}% of {n} points", flush=True)

    print("\nWIKIPEDIA-TEST mechanism (per view: high energy != good clustering?):", flush=True)
    Xv, y, k = load('Wikipedia-test')
    for vi, v in enumerate(Xv):
        Pv = kernel2(v, 'global', 'row')
        print(f"  view {vi} dim={v.shape[1]:4d}: spectral energy {spectral(Pv, k)[0]:.3f} | "
              f"single-view AMI {ami_runs(get_embedding(Pv, k), y, k):.3f}", flush=True)


if __name__ == '__main__':
    main()
