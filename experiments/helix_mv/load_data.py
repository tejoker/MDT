import numpy as np


def _helix_b(num_samples: int):
    """Helix-B (Lindenbaum et al., 2020): two 3D helix views of the same points."""
    a = np.linspace(0, 2 * np.pi, num_samples)
    b = (a + 0.5 * np.pi) % (2 * np.pi)
    views = []
    for i in (a, b):
        t = np.zeros((num_samples, 3))
        t[:, 0] = np.cos(5 * i)
        t[:, 1] = np.sin(5 * i)
        t[:, 2] = i
        views.append(4 * t)

    return views, a


def get_data(npoints: int = 1500, split: float = 0.5, seed: int = 123):
    """Multi-view Helix-B split into train/test.

    Returns a dict with 'train'/'test' as lists of V view arrays (same objects
    across views) plus the helix parameter as a color for plotting. The MDT
    operator is built on the train views; test points are extended out-of-sample
    by the encoder.
    """
    views, color = _helix_b(npoints)
    perm = np.random.default_rng(seed).permutation(npoints)
    n_train = int(npoints * split)
    tr, te = perm[:n_train], perm[n_train:]

    return {
        'train': [v[tr].astype('float32') for v in views],
        'test': [v[te].astype('float32') for v in views],
        'train_color': color[tr],
        'test_color': color[te],
    }
