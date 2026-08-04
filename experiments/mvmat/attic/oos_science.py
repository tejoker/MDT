"""Why does the deep encoder lose to Nystrom out-of-sample? Hypothesis tests.

All three methods extend the SAME train embedding E_train (plain SVD of the
operator); the ONLY thing that varies is how a test point is mapped to it:
  Nystrom        : linear projection of the test point's affinity row.
  deep-features  : MLP regressed E from raw features (what the paper's encoder sees).
  deep-affinity  : MLP regressed E from the affinity row (what Nystrom sees).

Exp B (sample sweep): vary N_train; does (Nystrom - deep-features) shrink? -> H2.
Exp C (input swap):   does deep-affinity match Nystrom while deep-features loses? -> H3.
"""
import argparse
import numpy as np
import scipy.io
import torch
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI

from src.mdt_operators import transition_matrix, oos_transition, _mdt_operator, _trajectory


def km(E, k):
    E = StandardScaler().fit_transform(E)
    return KMeans(k, n_init=10, random_state=0).fit_predict(E)


def regress(Xtr, Etr, Xte, epochs=500, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    sc = StandardScaler().fit(Xtr)
    Xtr_t = torch.tensor(sc.transform(Xtr), dtype=torch.float32)
    Xte_t = torch.tensor(sc.transform(Xte), dtype=torch.float32)
    Et = torch.tensor(StandardScaler().fit_transform(Etr), dtype=torch.float32)
    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, Etr.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((net(Xtr_t) - Et) ** 2).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        return net(Xte_t).numpy()


def run(name, n_train, n_test, t=4, seed=0, views=None):
    m = scipy.io.loadmat(f'/tmp/Multi-view-datasets/{name}.mat'); Xo = m['X'].ravel()
    if views is not None:
        Xo = Xo[list(views)]
    y = np.asarray(m['y'] if 'y' in m else m['Y']).ravel(); k = len(np.unique(y))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    tr, te = perm[:n_train], perm[n_train:n_train + n_test]
    Xtr = [np.nan_to_num(StandardScaler().fit_transform(np.asarray(v)[tr].astype(float))) for v in Xo]
    Xte = [np.nan_to_num(StandardScaler().fit_transform(np.asarray(v)[te].astype(float))) for v in Xo]
    ytr, yte = y[tr], y[te]
    knn = int(np.log(n_train)) + 1
    sig = [np.quantile(pdist(v[:200]), 0.5) for v in Xtr]
    P = [transition_matrix(v, s, knn) for v, s in zip(Xtr, sig)]
    Pnew = [oos_transition(vtr, vte, s, knn) for vtr, vte, s in zip(Xtr, Xte, sig)]
    a = _trajectory(len(P), t, 'random', seed)

    W = _mdt_operator(a, P)
    U, s_, Vt = np.linalg.svd(W, full_matrices=False)
    E_train = U[:, 1:k + 1] * s_[1:k + 1]
    Vmat = Vt.T

    # Nystrom OOS
    weighted = [np.einsum('v,vnm->nm', a[s], np.stack(P)) for s in range(t)]
    M = np.eye(n_train) if t == 1 else weighted[0].copy()
    for i in range(1, t - 1):
        M = weighted[i] @ M
    W_oos = np.einsum('v,vnm->nm', a[t - 1], np.stack(Pnew)) @ M       # n_test x n_train
    E_nys = W_oos @ Vmat[:, 1:k + 1]

    # deep-from-features  and  deep-from-affinity
    Xtr_feat = np.concatenate(Xtr, 1); Xte_feat = np.concatenate(Xte, 1)
    E_feat = regress(Xtr_feat, E_train, Xte_feat, seed=seed)
    E_aff = regress(W, E_train, W_oos, seed=seed)                      # input = affinity rows

    return (k,
            AMI(yte, km(E_nys, k)),
            AMI(yte, km(E_feat, k)),
            AMI(yte, km(E_aff, k)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--exp', default='C')
    exp = ap.parse_args().exp
    if exp == 'C':
        print("=== Exp C (input swap): Nystrom | deep-features | deep-affinity ===", flush=True)
        for name, ntr, nte in [('MSRC-v5', 147, 63), ('Wikipedia', 2000, 800), ('Handwritten', 1400, 600)]:
            k, nys, feat, aff = run(name, ntr, nte)
            print(f"{name:12s} (n_tr={ntr}) | Nystrom={nys:.3f}  deep-feat={feat:.3f}  deep-aff={aff:.3f}", flush=True)
    else:
        print("=== Exp B (sample sweep, Handwritten, test=600): Nystrom vs deep-features ===", flush=True)
        for ntr in (200, 400, 800, 1400):
            k, nys, feat, aff = run('Handwritten', ntr, 600)
            print(f"N_train={ntr:5d} | Nystrom={nys:.3f}  deep-feat={feat:.3f}  gap={nys-feat:+.3f}", flush=True)
