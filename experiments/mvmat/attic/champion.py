"""Champion operator (local-scaling bandwidth + row-norm + quality-weighted consensus,
M=30) and the ceiling proof: sweep the two remaining knobs (knn, trajectory length t)
to show the champion sits on a FLAT plateau -> no room left in the W = trajectory-of-
fused-transitions family. Multi-seed mean +/- std throughout."""
import numpy as np, sys
from functools import reduce
sys.path.insert(0, '/home/nicolasbigeard/internshipborelli/deep-diffusion-maps/experiments/mvmat')
from enhance_operator import kernel, consensus_op, view_quality, load, SEEDS, R
from reproduce_paper import get_embedding, ami_runs, knee_power


def champ(P, t, k, y, M=30):
    w = view_quality(P, k)
    a = [ami_runs(get_embedding(consensus_op(P, t, M, s, w), k), y, k, R) for s in SEEDS]
    return float(np.mean(a)), float(np.std(a))


def main():
    names = sys.argv[1:] or ['MSRC-v5', '100Leaves', 'Caltech101-7', 'Handwritten', 'UCI']
    for name in names:
        Xv, y, k = load(name); n = len(y)
        base = max(2, int(np.floor(np.log(n))))
        Pdef = [kernel(v, 'local', 'row') for v in Xv]
        t0 = max(knee_power(reduce(lambda a, b: a @ b, Pdef)), 1)
        mu, sd = champ(Pdef, t0, k, y)
        print(f"\n=== {name}  n={n} k={k}  champion(local+row+qweight, knn={base}, t={t0}, M=30): "
              f"AMI {mu:.3f} +/- {sd:.3f} ===", flush=True)
        print("  knn robustness:", flush=True)
        for knn in sorted({base, base + 3, base * 2, 15, 20}):
            P = [kernel(v, 'local', 'row', knn) for v in Xv]
            m, s = champ(P, t0, k, y); print(f"     knn={knn:3d}  AMI {m:.3f} +/- {s:.3f}", flush=True)
        print("  trajectory-length robustness:", flush=True)
        for t in sorted({max(1, t0 - 2), t0, t0 + 2, t0 + 4}):
            m, s = champ(Pdef, t, k, y); print(f"     t={t:3d}  AMI {m:.3f} +/- {s:.3f}", flush=True)


if __name__ == '__main__':
    main()
