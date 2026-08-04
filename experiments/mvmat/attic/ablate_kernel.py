"""Clean ablation: does the SPARSE kNN kernel beat the DENSE kernel for end-to-end learning,
holding EVERYTHING else fixed (identity init, same bandwidth, same t, same tau, same seed,
same kNN contrastive target)? Only the OPERATOR mask toggles. This isolates the kernel effect;
the earlier dense-vs-sparse comparison was confounded by projection-init and temperature."""
import gc
import numpy as np
from experiments.mvmat.t_sweep import load
from experiments.mvmat.sparse_e2e import train_sparse


def main():
    for name, N in [('MSRC-v5', None), ('OutdoorScene', 1500), ('UCI', 1500)]:
        Xs, y, k = load(name, N)
        print(f"\n=== {name} n={len(y)} k={k} ===", flush=True)
        for t in (1, 4):
            for tau in (1.0, 20.0):
                sp = train_sparse(Xs, y, k, t, tau=tau, sparse=True)
                de = train_sparse(Xs, y, k, t, tau=tau, sparse=False)
                print(f"  t={t} tau={tau:<4} | SPARSE trained={sp['amiN']:.3f} (init {sp['ami0']:.3f}) "
                      f"| DENSE trained={de['amiN']:.3f} (init {de['ami0']:.3f}) "
                      f"| delta(sparse-dense)={sp['amiN']-de['amiN']:+.3f}", flush=True)
                gc.collect()
        gc.collect()


if __name__ == '__main__':
    main()
