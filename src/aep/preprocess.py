"""Optional PCA into the working space."""
import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class PCATransform:
    """Maps raw rows to the working space. Identity (no centering) when pca=False."""
    mean: np.ndarray | None = None
    components: np.ndarray | None = None  # (d, D)
    explained_variance_ratio: np.ndarray | None = None

    @property
    def enabled(self):
        return self.components is not None

    def transform(self, X_raw):
        X_raw = np.asarray(X_raw, dtype=np.float64)
        if not self.enabled:
            return X_raw
        return (X_raw - self.mean) @ self.components.T

    @property
    def variance_retained(self):
        return float(self.explained_variance_ratio.sum()) if self.enabled else 1.0


PCA_SOLVERS = ("auto", "randomized", "full")
RANDOMIZED_MIN_DIM = 500  # solver="auto": randomized iff min(n, D) > this, else exact


def top_svd(Xc, m, seed=0, solver="auto"):
    """Top-m singular values s (m,) and right singular vectors Vt (m, D) of the centered matrix Xc.

    "randomized": sklearn's randomized_svd (Halko et al. 2011; n_oversamples 10, n_iter "auto",
    random_state seed), O(n D m), approximate; falls back to the exact SVD when
    m + 10 >= min(n, D), where it saves nothing. "full": exact np.linalg.svd, O(n D min(n, D)).
    "auto" (default): randomized iff min(n, D) > RANDOMIZED_MIN_DIM. For narrow data the exact SVD
    is both faster and exact (48,430 x 30: 0.04 s vs 0.65 s); for wide data randomized is much
    faster (20,000 x 2,000: 1.6 s vs 11.3 s)."""
    if solver not in PCA_SOLVERS:
        raise ValueError(f"unknown pca solver {solver!r}; expected one of {PCA_SOLVERS}")
    if solver == "auto":
        solver = "randomized" if min(Xc.shape) > RANDOMIZED_MIN_DIM else "full"
    if solver == "randomized" and m + 10 < min(Xc.shape):
        from sklearn.utils.extmath import randomized_svd
        _, s, Vt = randomized_svd(Xc, m, random_state=seed)
        return s, Vt
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return s[:m], Vt[:m]


def fit_working_space(X_raw, pca=True, n_pca=None, seed=0, solver="auto"):
    """Return (X working space, PCATransform). PCA via top_svd(solver); explained-variance ratios
    use the exact total variance ||X - mean||_F^2."""
    X_raw = np.asarray(X_raw, dtype=np.float64)
    if X_raw.ndim != 2:
        raise ValueError(f"expected 2D data matrix, got shape {X_raw.shape}")
    D = X_raw.shape[1]
    if not pca:
        log.info("working space: raw input (no PCA), d=%d", D)
        return X_raw.copy(), PCATransform()
    if n_pca is None:
        n_pca = min(D, 10)
    if not 3 <= n_pca <= D:
        raise ValueError(f"n_pca must satisfy 3 <= n_pca <= D={D}, got {n_pca}")
    mean = X_raw.mean(0)
    Xc = X_raw - mean
    if solver == "auto":
        solver = "randomized" if min(Xc.shape) > RANDOMIZED_MIN_DIM else "full"
    s, Vt = top_svd(Xc, n_pca, seed, solver)
    tf = PCATransform(mean=mean, components=Vt, explained_variance_ratio=s**2 / np.square(Xc).sum())
    log.info("working space: PCA (%s), d=%d of D=%d, variance retained %.4f", solver, n_pca, D,
             tf.variance_retained)
    return tf.transform(X_raw), tf


def top_pcs(X, m, seed=0, solver="auto"):
    """Scores on the top-m principal components of X (top_svd(solver))."""
    Xc = X - X.mean(0)
    _, Vt = top_svd(Xc, m, seed, solver)
    return Xc @ Vt.T


def encode_replicates(replicate, n):
    """(labels (R,), codes (n,) in 0..R-1) from per-cell replicate labels; None = one replicate.
    Validates length n and n_r >= 2 for every replicate."""
    if replicate is None:
        return np.array([0]), np.zeros(n, dtype=np.int64)
    replicate = np.asarray(replicate).ravel()
    if len(replicate) != n:
        raise ValueError(f"replicate has length {len(replicate)}, expected n = {n}")
    labels, codes = np.unique(replicate, return_inverse=True)
    sizes = np.bincount(codes.ravel(), minlength=len(labels))
    if (sizes < 2).any():
        raise ValueError(f"every replicate needs n_r >= 2; got {dict(zip(labels[sizes < 2].tolist(), sizes[sizes < 2].tolist()))}")
    return labels, codes.ravel().astype(np.int64)


def replicate_units(X, codes, k, knn_method="auto", seed=0):
    """Preprocessing within each replicate, on its unique rows.

    Returns (Xu (n_u, d), g (n,), rep_u (n_u,), idx (n_u, k), T (n_u, d, 2)): the unique rows of all
    replicates stacked in replicate order, the unique-row index of each cell (unique across
    replicates, so identical rows in different replicates are distinct units), the replicate of
    each unique row, the within-replicate k-NN neighborhoods (global unique-row indices) and the
    tangent bases."""
    from aep.knn import knn
    from aep.tangents import tangent_bases

    R = int(codes.max()) + 1
    g = np.empty(len(X), dtype=np.int64)
    Xus, reps, idxs, Ts = [], [], [], []
    offset = 0
    for r in range(R):
        cells = np.flatnonzero(codes == r)
        Xu_r, g_r = np.unique(X[cells], axis=0, return_inverse=True)
        if k > len(Xu_r):
            raise ValueError(f"k = {k} exceeds the {len(Xu_r)} unique rows of replicate {r}")
        idx_r, _ = knn(Xu_r, k, knn_method, seed)
        Ts.append(tangent_bases(Xu_r, idx_r))
        g[cells] = g_r.ravel() + offset
        Xus.append(Xu_r)
        reps.append(np.full(len(Xu_r), r, dtype=np.int64))
        idxs.append(idx_r + offset)
        offset += len(Xu_r)
    return (np.concatenate(Xus), g, np.concatenate(reps), np.concatenate(idxs),
            np.concatenate(Ts))
