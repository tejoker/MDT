"""Make best-of-8 FAIR: select the trajectory with an UNSUPERVISED criterion
(no labels), not by AMI. Report the AMI of the picked trajectory and the regret
vs the oracle (AMI-selected) best. Two unsupervised selectors: spectral energy
(our IQ-index winner) and embedding silhouette. Same operators/paths as crit_test."""
import numpy as np, torch
from scipy.sparse.linalg import svds
from sklearn.metrics import adjusted_mutual_info_score as AMI, silhouette_score
import experiments.mvmat.crit_test as C
import experiments.mvmat.sparse_neural_svd as S


def energy(Ps, a, k, extra=8):
    # top-k spectral energy among top-(k+extra), dropping the stationary mode.
    # consistent proxy across trajectories -> valid for RANKING (denominator truncated).
    _, sv, _ = svds(C.lin_op(Ps, a), k=k + extra)
    sv = np.sort(sv)[::-1][1:]                 # drop stationary singular value
    return float((sv[:k] ** 2).sum() / max((sv ** 2).sum(), 1e-12))


def run(name, N):
    Xs, yy, k = C.load(name, N)
    knn = int(np.log(N)) + 1
    Ps = [S.sparse_transition(v, knn) for v in Xs]
    amis, ens, sils = [], [], []
    for s in range(8):
        a = C.trajectory(len(Xs), s)
        E = C.svds_embed(Ps, a, k); lab = C.km(E, k)
        amis.append(AMI(yy, lab))
        ens.append(energy(Ps, a, k))
        sils.append(silhouette_score(E, lab) if len(set(lab)) > 1 else -1.0)
    amis = np.array(amis)
    oracle = amis.max(); rand = amis.mean()
    fair_en = amis[int(np.argmax(ens))]; fair_sil = amis[int(np.argmax(sils))]
    print(f"{name:9s} N={N} k={k} | random-mean {rand:.3f} | oracle(AMI) {oracle:.3f} "
          f"| fair-energy {fair_en:.3f} (regret {oracle-fair_en:.3f}) "
          f"| fair-sil {fair_sil:.3f} (regret {oracle-fair_sil:.3f})", flush=True)
    print(f"          per-traj AMI {np.round(amis,3).tolist()}", flush=True)


if __name__ == '__main__':
    for name, N in [('MNIST-4', 4000), ('MNIST-10k', 4000), ('ALOI', 4000)]:
        run(name, N)
