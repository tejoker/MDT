"""T4/T5 deep-dig: multi-seed OOS (error bars on the MSRC win) + per-view Wikipedia."""
import numpy as np
import experiments.mvmat.oos_science as O


def agg(name, ntr, nte, views=None, seeds=5):
    nys, feat, aff = [], [], []
    for s in range(seeds):
        _, n, f, a = O.run(name, ntr, nte, seed=s, views=views)
        nys.append(n); feat.append(f); aff.append(a)
    m = lambda x: (float(np.mean(x)), float(np.std(x)))
    return m(nys), m(feat), m(aff)


if __name__ == '__main__':
    print("=== Multi-seed OOS (mean±std, 5 seeds): Nystrom | deep-feat | deep-aff ===", flush=True)
    for name, ntr, nte in [('MSRC-v5', 147, 63), ('Wikipedia', 2000, 800), ('Handwritten', 1400, 600)]:
        n, f, a = agg(name, ntr, nte)
        print(f"{name:12s} | Nys {n[0]:.3f}±{n[1]:.3f} | feat {f[0]:.3f}±{f[1]:.3f} | aff {a[0]:.3f}±{a[1]:.3f}", flush=True)
    print("=== Per-view Wikipedia (which view makes raw features a weak input): deep-feat ===", flush=True)
    for vi, lab in [((0,), 'view0 img(128d)'), ((1,), 'view1 text(10d)'), ((0, 1), 'both')]:
        n, f, a = agg('Wikipedia', 2000, 800, views=vi, seeds=3)
        print(f"  {lab:16s} | Nys {n[0]:.3f} | deep-feat {f[0]:.3f}", flush=True)
