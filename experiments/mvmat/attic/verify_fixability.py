"""Is the end-to-end stall fundamental or just bad optimisation? Sweep lr (and longer
training) on Caltech; if the contrastive loss stays flat for every lr, the stall is
not merely a step-count / learning-rate issue."""
import numpy as np
import experiments.mvmat.verify_collapse as VC

if __name__ == '__main__':
    Xs, y, k = VC.load('Caltech101-7', None)
    print("=== Fixability: Caltech, 600 steps, lr sweep (loss start->end, AMI start->end) ===", flush=True)
    for lr in [0.001, 0.01, 0.05, 0.2]:
        log = VC.train(Xs, y, k, steps=600, reg=0.0, lr=lr)
        l0, lN = log[0][1], log[-1][1]; a0, aN = log[0][2], log[-1][2]
        print(f"  lr={lr:<5} | loss {l0:.3f}->{lN:.3f} ({100*(lN-l0)/l0:+.1f}%) | AMI {a0:.3f}->{aN:.3f}", flush=True)
