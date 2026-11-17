"""Local tangent bases: top-2 principal directions of each centered neighborhood."""
import numpy as np

ROUTES = ("auto", "gram", "cov")


def tangent_bases(X, idx, chunk=1000, route="auto"):
    """T (n, d, 2) float32 from neighborhoods idx (n, k).

    gram: eigh of the k x k Gram matrix Z Z^T, t = Z^T u / sqrt(lambda);
    cov:  eigh of the d x d matrix Z^T Z. "auto": gram iff k < d.
    Z = neighborhood rows centered by their mean.
    """
    if route not in ROUTES:
        raise ValueError(f"unknown tangent route {route!r}")
    n, d = X.shape
    k = idx.shape[1]
    if route == "auto":
        route = "gram" if k < d else "cov"
    T = np.empty((n, d, 2), dtype=np.float32)
    for s in range(0, n, chunk):
        Z = X[idx[s:s + chunk]]                                 # (b, k, d)
        Z = Z - Z.mean(1, keepdims=True)
        Zt = np.swapaxes(Z, 1, 2)
        if route == "gram":
            lam, U = np.linalg.eigh(Z @ Zt)                     # ascending
            lam, U = lam[:, [-1, -2]], U[:, :, [-1, -2]]
            T[s:s + chunk] = Zt @ U / np.sqrt(np.maximum(lam, 1e-300))[:, None, :]
        else:
            _, U = np.linalg.eigh(Zt @ Z)
            T[s:s + chunk] = U[:, :, [-1, -2]]
    return T
