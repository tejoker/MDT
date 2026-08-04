"""Why is the end-to-end loss landscape flat? Hypothesis: the t-step diffused
operator W has tiny, near-uniform entries, so the contrastive loss (logits = exp(W))
sits in a flat region -> vanishing gradient -> stall. Tests:
  1. logit scale: |W| and softmax(W) row peakedness (uniform => flat).
  2. per-group gradient norms (which parameters get any signal).
  3. loss sensitivity to a logit temperature tau (does sharpening make L respond?).
  4. intervention: train with temperature tau -> does it unstick the loss + recover AMI?
If a large tau unsticks it, the stall is a logit-scale problem, not lr/steps."""
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI
import experiments.mvmat.deep_fusion as DF
import experiments.mvmat.verify_collapse as VC


def contrastive_temp(W, masks, tau):
    Wc = torch.clamp(torch.nan_to_num(W), -20, 20) * tau
    ex = torch.exp(Wc) * (1 - torch.eye(W.shape[0]))
    logp = torch.log((ex / ex.sum(1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-12))
    return -sum(logp[m].sum() for m in masks) / (len(masks) * W.shape[0])


def main(name='Caltech101-7'):
    Xs, y, k = VC.load(name, None)
    Xt = [torch.tensor(v, dtype=torch.float32) for v in Xs]
    masks = [torch.tensor(m) for m in DF.knn_masks(Xs, int(np.log(len(y))) + 1)]
    torch.manual_seed(0)
    model = DF.DeepMDT([v.shape[1] for v in Xs], 4); model.init_bandwidth(Xt)
    n = len(y)

    print(f"=== {name} (n={n}, k={k}) landscape at init ===", flush=True)
    W = model.operator(Xt); Wd = W.detach().numpy()
    sm = np.exp(Wd - Wd.max(1, keepdims=True)); sm /= sm.sum(1, keepdims=True)
    print(f"  W entries: mean|W_ij|={np.abs(Wd).mean():.2e}  per-row(max-min)={np.mean(Wd.max(1)-Wd.min(1)):.2e}", flush=True)
    print(f"  softmax(W) rows: mean max-prob={sm.max(1).mean():.5f}  (uniform = 1/n = {1/n:.5f})", flush=True)

    loss = DF.contrastive(W, masks); model.zero_grad(); loss.backward()
    g = {'proj': 0., 'log_gamma': 0., 'a': 0.}
    for nm, p in model.named_parameters():
        if p.grad is None:
            continue
        key = 'proj' if 'proj' in nm else ('log_gamma' if 'log_gamma' in nm else 'a')
        g[key] += float(p.grad.norm())
    print(f"  grad norms: proj={g['proj']:.2e}  log_gamma={g['log_gamma']:.2e}  a={g['a']:.2e}", flush=True)

    print("  loss vs logit temperature (fixed W -- does sharpening make L respond?):", flush=True)
    for tau in [1, 5, 20, 100]:
        print(f"    L(tau={tau:3d}) = {float(contrastive_temp(W, masks, tau)):.4f}", flush=True)

    print("=== intervention: train 200 steps with temperature tau ===", flush=True)
    for tau in [1, 20, 100]:
        torch.manual_seed(0)
        m2 = DF.DeepMDT([v.shape[1] for v in Xs], 4); m2.init_bandwidth(Xt)
        opt = torch.optim.Adam(m2.parameters(), lr=0.01)
        l0 = None
        for step in range(200):
            opt.zero_grad(); W2 = m2.operator(Xt); L = contrastive_temp(W2, masks, tau)
            if step == 0:
                l0 = float(L)
            L.backward(); torch.nn.utils.clip_grad_norm_(m2.parameters(), 1.0); opt.step()
        with torch.no_grad():
            Wn = m2.operator(Xt)
            E = VC.embed(Wn.detach().numpy(), k)
            ami = AMI(y, KMeans(k, n_init=5, random_state=0).fit_predict(E))
            lN = float(contrastive_temp(Wn, masks, tau))
        print(f"  tau={tau:3d}: loss {l0:.4f}->{lN:.4f} ({100*(lN-l0)/abs(l0):+.1f}%) | final AMI={ami:.3f}", flush=True)


if __name__ == '__main__':
    main()
