"""k nearest neighbors (self included): exact blockwise GEMM or approximate (pynndescent)."""
import numpy as np

EXACT_MAX_N = 20_000  # knn_method="auto": exact up to this n, else approximate


def _self_first(idx, dist):
    """Ensure each row lists itself first (distance 0), then neighbors by increasing distance."""
    n, k = idx.shape
    rows = np.arange(n)[:, None]
    is_self = idx == rows
    missing = ~is_self.any(1)
    idx[missing, -1], dist[missing, -1] = np.arange(n)[missing], 0.0
    key = np.where(idx == rows, -1.0, dist)
    order = np.argsort(key, axis=1, kind="stable")
    idx, dist = np.take_along_axis(idx, order, 1), np.take_along_axis(dist, order, 1)
    dist[:, 0] = 0.0
    return idx, dist


def knn_exact(X, k, block=1024):
    """Blockwise squared distances ||xi||^2 + ||xj||^2 - 2 xi.xj on centered float32 data
    (negatives from round-off set to 0); argpartition for the k smallest per row."""
    Xc = (X - X.mean(0)).astype(np.float32)
    sq = (Xc * Xc).sum(1)
    n = len(Xc)
    idx = np.empty((n, k), dtype=np.int64)
    for s in range(0, n, block):
        e = min(s + block, n)
        d2 = sq[s:e, None] + sq[None, :] - 2 * Xc[s:e] @ Xc.T
        np.maximum(d2, 0, out=d2)
        d2[np.arange(e - s), np.arange(s, e)] = -1  # self always selected (duplicates)
        idx[s:e] = np.argpartition(d2, k - 1, axis=1)[:, :k]
    dist = np.linalg.norm(X[idx] - X[:, None], axis=-1)  # selected distances in float64
    return _self_first(idx, dist)


def knn_approx(X, k, seed=0):
    try:
        from pynndescent import NNDescent
    except ImportError as e:
        raise ImportError("knn_method='approx' needs the optional dependency pynndescent") from e
    idx, dist = NNDescent(X, n_neighbors=k, random_state=seed).neighbor_graph
    return _self_first(idx.astype(np.int64), dist.astype(np.float64))


def knn(X, k=20, method="auto", seed=0):
    """Neighborhoods (n, k) of the rows of X (unique rows: the caller deduplicates): column 0 is the cell itself, then its k - 1 nearest neighbors.
    Returns (idx, dist)."""
    if method == "auto":
        method = "exact" if len(X) <= EXACT_MAX_N else "approx"
    if method == "exact":
        return knn_exact(X, k)
    if method == "approx":
        return knn_approx(X, k, seed)
    raise ValueError(f"unknown knn_method {method!r}")
