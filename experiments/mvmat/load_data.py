import numpy as np
import scipy.io
from sklearn.preprocessing import StandardScaler


def get_data(path: str, name: str, split: float = 0.7, seed: int = 0, standardize: bool = True):
    """Load a multi-view .mat dataset (github.com/ChuanbinZhang/Multi-view-datasets).

    The .mat holds X = object array of V views, each (n, d_v), and y = labels.
    Returns train/test lists of views (same objects across views) plus labels.
    The scaler is fit on train only, so the test split is a true out-of-sample set.
    """
    m = scipy.io.loadmat(f"{path}/{name}.mat")
    views = [np.asarray(v, dtype='float32') for v in m['X'].ravel()]
    y = np.asarray(m['y'] if 'y' in m else m['Y']).ravel()

    perm = np.random.default_rng(seed).permutation(len(y))
    n_train = int(len(y) * split)
    tr, te = perm[:n_train], perm[n_train:]

    data = {'train': [], 'test': []}
    for v in views:
        v_tr, v_te = v[tr], v[te]
        if standardize:
            sc = StandardScaler().fit(v_tr)
            v_tr, v_te = sc.transform(v_tr), sc.transform(v_te)
        data['train'].append(v_tr.astype('float32'))
        data['test'].append(v_te.astype('float32'))
    data['train_color'], data['test_color'] = y[tr], y[te]

    return data
