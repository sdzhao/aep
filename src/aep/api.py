"""Public entry point: fit_aep."""
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from aep.losses import mean_normalizer
from aep.model import AEPModel
from aep.preprocess import encode_replicates, fit_working_space, replicate_units
from aep.train import ReplicateBatchSampler, TrainConfig, setup_torch, train

log = logging.getLogger(__name__)


def fit_aep(X_raw, replicate=None, pca=True, n_pca=None, pca_solver="auto", k=20,
            knn_method="auto", stress_power=2.0, lambda_rank=1.0, lambda_s=5.0, tau=1e-12,
            jacobian_route="auto", steps=5000, batch_size=256, lr=1e-3, seed=0, run_name=None,
            out_dir="results"):
    """Fit an AEP map f_theta: R^d -> R^3 to the rows of X_raw and embed them.

    Parameters
    ----------
    X_raw : array (n, D)
        Points (e.g. cells) in rows, features in columns.
    replicate : array of n labels, optional
        Replicate (sample/batch) of each row. One map is shared by all replicates; neighbors,
        tangent planes, duplicate detection, stress pairs and normalizers are within-replicate,
        and each training batch comes from one replicate (drawn with probability n_r / n).
        None: all rows form one replicate.
    pca : bool, default True
        Project onto the top principal components (fit on all rows) before fitting.
    n_pca : int, optional
        Number of PCs d (3 <= d <= D); default min(D, 10).
    pca_solver : {"auto", "full", "randomized"}, default "auto"
        "auto": exact SVD if min(n, D) <= 500, otherwise randomized SVD (seeded).
    k : int, default 20
        Neighborhood size (the point plus its k - 1 nearest neighbors) for the tangent planes.
    knn_method : {"auto", "exact", "approx"}, default "auto"
        "auto": exact if a replicate has <= 20,000 unique rows, otherwise NN-descent.
    stress_power : float, default 2.0
        p in the stress pair weights w_ij = delta_ij^-p (p = 2: relative / elastic stress).
    lambda_rank : float, default 1.0
        Weight of the rank penalty sigma_3(J)^2, which keeps the image a 2D surface.
    lambda_s : float, default 5.0
        Weight of the normalized stress (distance preservation within batches).
    tau : float, default 1e-12
        Numerical guard in the isometry term.
    jacobian_route : {"auto", "explicit", "factorized"}, default "auto"
        How Jacobians are computed (identical results; "auto" picks the faster one).
    steps : int, default 5000
        Minibatch updates (Adam, cosine learning-rate decay, EMA of weights).
    batch_size : int, default 256
        Rows per batch; all within-batch pairs enter the stress.
    lr : float, default 1e-3
        Initial learning rate.
    seed : int, default 0
        Seed for initialization, batch sampling, randomized SVD and NN-descent.
    run_name : str, optional
        If given, save {out_dir}/logs/{run_name}.csv (training log),
        {out_dir}/embeddings/{run_name}.npy (Y) and {out_dir}/embeddings/{run_name}.pt (model).
    out_dir : str or Path, default "results"
        Output folder used with run_name.

    Returns
    -------
    Y : array (n, 3)
        AEP coordinates of the rows of X_raw.
    model : AEPModel
        The fitted map; model(X_new) embeds new rows (same D features), model.save(path) and
        AEPModel.load(path) store and restore it.
    info : dict
        Settings, PCA variance retained, per-replicate sizes and normalizers, timings, and the
        loss terms of the last batch (info["final"]).
    """
    if stress_power < 0:
        raise ValueError(f"stress_power must be >= 0, got {stress_power}")
    t0 = time.perf_counter()
    cfg = TrainConfig(stress_power=stress_power, lambda_rank=lambda_rank, lambda_s=lambda_s,
                      tau=tau, jacobian_route=jacobian_route, steps=steps, batch_size=batch_size,
                      lr=lr, seed=seed)
    setup_torch(seed)
    X, transform = fit_working_space(X_raw, pca=pca, n_pca=n_pca, seed=seed, solver=pca_solver)
    labels, codes = encode_replicates(replicate, len(X))
    # preprocessing within each replicate, on its unique rows; g maps each cell to its unique row
    Xu, g, _, _, T = replicate_units(X, codes, k, knn_method, seed)
    sampler = ReplicateBatchSampler(codes, batch_size, np.random.default_rng(seed))
    nubar = mean_normalizer(Xu, g, stress_power, sampler)
    t_pre = time.perf_counter() - t0
    log.info("R=%d, n_u=%d, k=%d, p=%g, tau=%.3g, nu_bar=%s, lambda_rank=%.4g, lambda_s=%.4g, %.1fs",
             len(labels), len(Xu), k, stress_power, tau, np.array2string(nubar, precision=4),
             lambda_rank, lambda_s, t_pre)

    out = Path(out_dir)
    log_path = out / "logs" / f"{run_name}.csv" if run_name else None
    net, tinfo = train(X, T, g, nubar, sampler, cfg, log_path=log_path)
    model = AEPModel(net, transform)
    Y = model.embed_working(X)
    if run_name:
        (out / "embeddings").mkdir(parents=True, exist_ok=True)
        np.save(out / "embeddings" / f"{run_name}.npy", Y)
        model.save(out / "embeddings" / f"{run_name}.pt")

    info = {
        "settings": {"pca": pca, "n_pca": X.shape[1] if pca else None,
                     "pca_solver": pca_solver if pca else None, "k": k,
                     "knn_method": knn_method, **asdict(cfg)},
        "n_unique": len(Xu),
        "space": "pca" if transform.enabled else "raw",
        "variance_retained": transform.variance_retained if transform.enabled else None,
        "tau": tau,
        "replicates": labels.tolist(),
        "n_r": sampler.n_r.tolist(),
        "nu_bar": nubar.tolist(),   # per replicate
        "preprocess_time": t_pre,
        **tinfo,
    }
    info["total_time"] = time.perf_counter() - t0
    return Y, model, info
