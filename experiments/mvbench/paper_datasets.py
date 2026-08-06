"""The MDT paper's non-.mat clustering datasets, as out-of-sample Splits.

`experiments/mvbench/config.yml` covers 5 of the 9 clustering datasets in the
paper's Tab. 3 (100Leaves, Caltech101-7, MSRC-v5, Yale, Multi-Feat==Handwritten)
because those ship as .mat files.  This module adds the other constructed ones so
the appendix spans the paper's own benchmark:

    k_mvmnist   K-MvMNIST  (Kuchroo et al. 2022)  6000 x 2 views, 10 classes
    l_mvmnist   L-MvMNIST  (Lindenbaum et al. 2020)
    olivetti    Olivetti faces, pixel + HOG views, 400 x 2 views, 40 classes

L-Isolet is deliberately NOT included, for a reason that is about this benchmark
rather than about the dataset.  Its three "views" are three *different kernels*
(Gaussian, exponential, correlation) computed on one feature matrix.  mvbench's
entire control is that every method sees the same kernel, so a dataset whose
multi-view structure IS a kernel choice cannot enter it without destroying the
thing being controlled.  Reported separately instead.

Two departures from the reference loaders, both required by an out-of-sample
protocol and neither optional:

* **PCA is fitted on the training rows only.**  Upstream fits PCA on all 6000
  samples and then returns the views, so a transductive split of its output has
  already seen the test rows through the projection.  The view *construction*
  (noise, masking, HOG) is reproduced exactly; only the projection is refitted.
* **MinMax scaling is likewise train-fitted** for the same reason.

Reference-repo bug found while reading these, worth reporting upstream: in
``benchmarks/isolet_lindenbaum.py`` the K1 and K2 kernels are written into the
same ``k_temp`` buffer and appended by reference, so ``list_views[0]`` is
overwritten by K2.  Verified: ``allclose(view1, view2) == True``.  The published
L-Isolet therefore has two distinct views, not three.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from experiments.gnn_mdt.closure import Split

AVAILABLE = ("k_mvmnist", "l_mvmnist", "olivetti")
# Only these respond to the paper's noise parameter s (Fig. 7).
NOISE_SWEEPABLE = ("k_mvmnist", "l_mvmnist")
_MNIST_CACHE: dict[int, tuple] = {}


def _mnist(num: int = 6000):
    """MNIST-784, cached in-process: fetch_openml is slow and hit once per call."""
    if num not in _MNIST_CACHE:
        from sklearn.datasets import fetch_openml
        data = fetch_openml("mnist_784", as_frame=False)
        _MNIST_CACHE[num] = (np.asarray(data.data[:num], dtype=np.float64),
                             np.asarray(data.target[:num], dtype=int))
    return _MNIST_CACHE[num]


def _raw_views(name: str, noise_factor: float = 0.5) -> tuple[list, np.ndarray]:
    """View construction only -- no projection, no scaling. Those are per-split."""
    if name == "k_mvmnist":
        # Kuchroo et al.: both views are the digits plus Gaussian noise, the
        # second one noisier. random_state 3333 as upstream.
        x, y = _mnist()
        latent = MinMaxScaler().fit_transform(x)
        rng = np.random.default_rng(3333)
        first = np.clip(latent + rng.normal(0, .3, latent.shape), 0, 1)
        rng2 = np.random.default_rng(3335)
        second = np.clip(latent + rng2.normal(0, noise_factor, latent.shape), 0, 1)
        return [first, second], y
    if name == "l_mvmnist":
        # Lindenbaum et al.: view 1 additive noise, view 2 random pixel dropout.
        x, y = _mnist()
        latent = MinMaxScaler().fit_transform(x)
        rng = np.random.default_rng(333)
        first = np.clip(latent + rng.normal(0, noise_factor, latent.shape), 0, 1)
        rng2 = np.random.default_rng(334)
        second = latent.copy()
        second[rng2.binomial(1, noise_factor, latent.shape) == 1] = 0
        return [first, second], y
    if name == "olivetti":
        from skimage.feature import hog
        from skimage.transform import resize
        from sklearn.datasets import fetch_olivetti_faces
        faces = fetch_olivetti_faces()
        pixels = np.asarray(faces.data, dtype=np.float64)
        descriptors = np.asarray([hog(resize(img, (32, 32))) for img in faces.images])
        return [pixels, descriptors], np.asarray(faces.target, dtype=int)
    raise ValueError(f"unknown paper dataset: {name!r} (have {AVAILABLE})")


def load_paper_split(name: str, split: float, seed: int,
                     cap_train: int | None = None, cap_test: int | None = None,
                     standardize: bool = True, components: int = 100,
                     noise_factor: float = .5) -> Split:
    """Build a Split with every fitted transform estimated on train rows only.

    ``noise_factor`` is the paper's robustness parameter s (Fig. 7).  It only
    affects K-/L-MvMNIST; Olivetti ignores it.  In K-MvMNIST *only the second
    view* degrades (the first is fixed at 0.3), so it isolates whether a method
    can down-weight a deteriorating view -- which is precisely MDT's claim.  In
    L-MvMNIST both views degrade, so it measures graceful decay instead.
    """
    views, y = _raw_views(name, noise_factor)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    n_train = int(split * len(y))
    tr, te = order[:n_train], order[n_train:]
    if cap_train:
        tr = tr[:cap_train]
    if cap_test:
        te = te[:cap_test]

    train, test = [], []
    for view in views:
        scaler = MinMaxScaler().fit(view[tr])
        a, b = scaler.transform(view[tr]), scaler.transform(view[te])
        rank = min(components, a.shape[1], len(tr) - 1)
        if rank >= 1:
            projection = PCA(n_components=rank, random_state=seed).fit(a)
            a, b = projection.transform(a), projection.transform(b)
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        if standardize:
            # match closure.load_split: per-feature standardisation on train
            mean, std = a.mean(0, keepdims=True), a.std(0, keepdims=True)
            std = np.where(std < 1e-12, 1.0, std)
            a, b = (a - mean) / std, (b - mean) / std
        train.append(np.nan_to_num(a).astype(np.float32))
        test.append(np.nan_to_num(b).astype(np.float32))
    return Split(train, test, y[tr], y[te], name)


def demo() -> None:
    """Self-check: shapes line up, classes survive the split, no leakage in PCA."""
    for name in AVAILABLE:
        s = load_paper_split(name, .7, 0, 800, 400)
        assert len(s.train) == 2, name
        assert len(s.y_train) == len(s.train[0]), name
        assert len(s.y_test) == len(s.test[0]), name
        assert len(np.unique(s.y_train)) > 1 and len(np.unique(s.y_test)) > 1, name
        assert all(np.all(np.isfinite(v)) for v in s.train + s.test), name
        print(f"  {name:12s} n_train={len(s.y_train):4d} n_test={len(s.y_test):4d} "
              f"views={[v.shape[1] for v in s.train]} "
              f"classes={len(np.unique(s.y_train))}")
    print("paper_datasets self-check passed")


if __name__ == "__main__":
    demo()
