"""Neural on the FAIR path: energy-select the best-of-8 trajectory (unsupervised),
then run the neural SVD on THAT operator -> a fully deployable pipeline. Also run
neural on the oracle path to confirm it reproduces the current table's value."""
import numpy as np, torch
from scipy.sparse.linalg import svds
from sklearn.metrics import adjusted_mutual_info_score as AMI
import experiments.mvmat.crit_test as C
import experiments.mvmat.sparse_neural_svd as S


def energy(Ps, a, k, extra=8):
    _, sv, _ = svds(C.lin_op(Ps, a), k=k + extra)
    sv = np.sort(sv)[::-1][1:]
    return float((sv[:k] ** 2).sum() / max((sv ** 2).sum(), 1e-12))


def run(name, N):
    Xs, yy, k = C.load(name, N)
    knn = int(np.log(N)) + 1
    Ps = [S.sparse_transition(v, knn) for v in Xs]
    Pts = [S.torch_sparse(P.T) for P in Ps]
    amis, ens = [], []
    for s in range(8):
        a = C.trajectory(len(Xs), s)
        amis.append(AMI(yy, C.km(C.svds_embed(Ps, a, k), k)))
        ens.append(energy(Ps, a, k))
    amis = np.array(amis)
    fair_seed = int(np.argmax(ens)); oracle_seed = int(np.argmax(amis))
    a_fair = torch.tensor(C.trajectory(len(Xs), fair_seed), dtype=torch.float32)
    a_orac = torch.tensor(C.trajectory(len(Xs), oracle_seed), dtype=torch.float32)
    nf = AMI(yy, C.km(S.neural_svd(Xs, Ps, Pts, a_fair, k), k))
    no = AMI(yy, C.km(S.neural_svd(Xs, Ps, Pts, a_orac, k), k))
    print(f"{name:9s} k={k} | eig fair(energy) {amis[fair_seed]:.3f} | neural FAIR {nf:.3f} "
          f"| eig oracle {amis[oracle_seed]:.3f} | neural oracle {no:.3f}", flush=True)


if __name__ == '__main__':
    for name, N in [('MNIST-4', 4000), ('MNIST-10k', 4000), ('ALOI', 4000)]:
        run(name, N)
