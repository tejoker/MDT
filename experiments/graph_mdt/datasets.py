"""Unified loaders for graph-view datasets.

Every dataset is normalised to the same triple:
  features  : float32 array (n, d)         -- node feature view
  graphs    : dict name -> csr (n, n)      -- natural adjacency views, binary,
                                              symmetric, no self-loops
  labels    : int array (n,)

Raw sources (DMGI pickles, .mat, npz) are converted once and cached as
compressed npz under CACHE_DIR so the 2 GB pickles are never loaded twice.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat

RAW_DIR = os.environ.get("GRAPH_MDT_RAW", "/tmp/multiplex-data")
CACHE_DIR = os.environ.get("GRAPH_MDT_CACHE", "/tmp/multiplex-data/cache")


def _clean_adjacency(a) -> sp.csr_matrix:
    """Binary, symmetric, zero-diagonal csr from any dense/sparse square input."""
    a = sp.csr_matrix(a)
    a.data = np.ones_like(a.data)
    a = a.maximum(a.T)
    a.setdiag(0)
    a.eliminate_zeros()
    return a.astype(np.float32)


def _labels_1d(y) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] > 1:  # one-hot
        y = y.argmax(axis=1)
    return y.ravel().astype(np.int64)


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.npz")


def _save_cache(name: str, features, graphs, labels) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    payload = {"features": features.astype(np.float32), "labels": labels,
               "graph_names": np.array(sorted(graphs))}
    for gname, a in graphs.items():
        a = a.tocsr()
        payload[f"g_{gname}_data"] = a.data
        payload[f"g_{gname}_indices"] = a.indices
        payload[f"g_{gname}_indptr"] = a.indptr
    np.savez_compressed(_cache_path(name), **payload)


def _load_cache(name: str):
    z = np.load(_cache_path(name), allow_pickle=False)
    n = z["labels"].shape[0]
    graphs = {}
    for gname in z["graph_names"]:
        gname = str(gname)
        graphs[gname] = sp.csr_matrix(
            (z[f"g_{gname}_data"], z[f"g_{gname}_indices"], z[f"g_{gname}_indptr"]),
            shape=(n, n))
    return z["features"], graphs, z["labels"]


# --- raw converters -------------------------------------------------------

def _convert_acm():
    m = loadmat(os.path.join(RAW_DIR, "data", "acm.mat"))
    graphs = {"PAP": _clean_adjacency(m["PAP"]), "PLP": _clean_adjacency(m["PLP"])}
    return np.asarray(m["feature"], dtype=np.float32), graphs, _labels_1d(m["label"])


def _convert_dmgi_pickle(fname, graph_keys=None):
    with open(os.path.join(RAW_DIR, "data", fname), "rb") as f:
        d = pickle.load(f)
    if graph_keys is None:  # auto-detect: every square (n, n) array is a view
        n = np.asarray(d["label"]).shape[0]
        graph_keys = [k for k, v in d.items()
                      if hasattr(v, "shape") and v.shape == (n, n)]
    graphs = {k: _clean_adjacency(d[k]) for k in graph_keys}
    features = d["feature"]
    if sp.issparse(features):
        features = features.toarray()
    return np.asarray(features, dtype=np.float32), graphs, _labels_1d(d["label"])


def _convert_heterophilous(fname):
    z = np.load(os.path.join(RAW_DIR, "heterophilous", fname))
    n = z["node_labels"].shape[0]
    e = z["edges"]
    a = sp.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    return (z["node_features"].astype(np.float32),
            {"graph": _clean_adjacency(a)}, z["node_labels"].astype(np.int64))


def _convert_mv_mat(name, adjacency_views, feature_view):
    """Graph datasets from /tmp/Multi-view-datasets (X = object array of views)."""
    m = loadmat(f"/tmp/Multi-view-datasets/{name}.mat")
    views = list(m["X"].ravel())
    key = "y" if "y" in m else "Y"
    graphs = {f"A{i}": _clean_adjacency(views[i]) for i in adjacency_views}
    features = np.asarray(views[feature_view], dtype=np.float32)
    return features, graphs, _labels_1d(m[key])


_CONVERTERS = {
    "acm": _convert_acm,
    "dblp": lambda: _convert_dmgi_pickle("dblp.pkl", DMGI_GRAPH_KEYS["dblp"]),
    "imdb": lambda: _convert_dmgi_pickle("imdb.pkl", DMGI_GRAPH_KEYS["imdb"]),
    "amazon": lambda: _convert_dmgi_pickle("amazon.pkl"),
    "roman_empire": lambda: _convert_heterophilous("roman_empire.npz"),
    "amazon_ratings": lambda: _convert_heterophilous("amazon_ratings.npz"),
    "minesweeper": lambda: _convert_heterophilous("minesweeper.npz"),
    "cora": lambda: _convert_mv_mat("Cora", adjacency_views=[0], feature_view=1),
    "citeseer": lambda: _convert_mv_mat("CiteSeer", adjacency_views=[0], feature_view=1),
}

# DMGI-lineage pickles: DBLP-7907 papers, IMDB-3550 movies.
DMGI_GRAPH_KEYS = {"dblp": ["PAP", "PPrefP", "PTP", "PATAP"],
                   "imdb": ["MAM", "MDM"]}


def load(name: str):
    """Return (features, graphs, labels) for a dataset, converting once."""
    if not os.path.exists(_cache_path(name)):
        features, graphs, labels = _CONVERTERS[name]()
        _save_cache(name, features, graphs, labels)
    return _load_cache(name)


def available() -> list:
    return sorted(_CONVERTERS)


if __name__ == "__main__":
    for name in available():
        try:
            features, graphs, labels = load(name)
            gdesc = ", ".join(f"{k}:{int(a.nnz // 2)}e" for k, a in graphs.items())
            print(f"{name}: n={len(labels)} d={features.shape[1]} "
                  f"classes={len(np.unique(labels))} graphs[{gdesc}]")
        except Exception as exc:  # noqa: BLE001 - smoke report, keep going
            print(f"{name}: FAILED {exc}")
