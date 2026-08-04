"""Does t=1 match 'fixed MDT'? Pin down exactly what t=1 is. A length-1 MDT trajectory
is _mdt_operator(onehot([v]), P) = P_v (a SINGLE view), OR a convex step sum_v a_v P_v.
Neither equals the multi-view AD baseline (product of ALL views). Show: (1) my kernel2
single-view op == reproduce_paper canonical kernel op (same AMI) -> no kernel bug; (2) the
MDT one-hot length-1 operator == that single view exactly; (3) AD / convex fusion are
DIFFERENT (multi-view) quantities; (4) at the embedding dim MDT actually uses (d=k) it is
stable -- the d=128 collapse is a dimension artifact, not a t=1 failure."""
import numpy as np, scipy.io, sys
from functools import reduce
sys.path.insert(0, '/home/nicolasbigeard/internshipborelli/deep-diffusion-maps/experiments/mvmat')
sys.path.insert(0, '/home/nicolasbigeard/internshipborelli/deep-diffusion-maps/src')
from corrected_enhance import kernel2
from reproduce_paper import get_kernel_matrix, get_embedding, ami_runs
from mdt_operators import _mdt_operator, _onehot


def main():
    m = scipy.io.loadmat('/tmp/Multi-view-datasets/MSRC-v5.mat')
    Xv = [np.asarray(v).astype(float) for v in m['X'].ravel()]
    y = np.asarray(m['y']).ravel(); k = len(np.unique(y)); V = len(Xv)
    for label, P in [('kernel2   ', [kernel2(v, 'global', 'row') for v in Xv]),
                     ('reproduce ', [get_kernel_matrix(v, True) for v in Xv])]:
        sv = [ami_runs(get_embedding(P[v], k), y, k) for v in range(V)]          # each single view, d=k
        onehot_v0 = ami_runs(get_embedding(_mdt_operator(_onehot([0], V), P), k), y, k)  # MDT len-1 traj on view0
        identical = np.allclose(_mdt_operator(_onehot([0], V), P), P[0])
        convex = ami_runs(get_embedding(sum(P) / V, k), y, k)                    # convex uniform fusion, t=1
        ad = ami_runs(get_embedding(reduce(lambda a, b: a @ b, P), k), y, k)     # AD: product of all views
        print(f"{label} single-view AMIs(d=k) {np.round(sv,3)} | MDT onehot(v0)={onehot_v0:.3f} "
              f"(op==P0? {identical}) | convex-t1={convex:.3f} | AD(prod all)={ad:.3f}", flush=True)
    # dimension stability of the single best view (view with max AMI) at d=k vs d=64 vs d=128
    P = [kernel2(v, 'global', 'row') for v in Xv]
    v_best = int(np.argmax([ami_runs(get_embedding(P[v], k), y, k) for v in range(V)]))
    print("\ndim stability of the SINGLE best view (t=1, no trajectory):", flush=True)
    for d in (k, 32, 64, 128):
        a = ami_runs(get_embedding(P[v_best], d), y, k)
        print(f"   d={d:3d}  AMI {a:.3f}", flush=True)


if __name__ == '__main__':
    main()
