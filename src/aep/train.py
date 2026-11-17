"""Training loop: ARAP isometry + rank penalty + power-weighted stress (no gradient clipping), Adam
with cosine decay, EMA of parameters."""
import copy
import csv
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from aep.cotangent import jacobian_terms, select_route
from aep.losses import TAU, stress, term_iso, term_rank
from aep.model import SurfaceMLP

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    stress_power: float = 2.0   # p in w_ij = delta_ij^-p
    lambda_rank: float = 1.0    # rank penalty mu_3 = sigma_3(J)^2, relative to term 1 (weight 1)
    lambda_s: float = 5.0       # power-weighted stress, relative to term 1
    tau: float = TAU            # absolute guard in the closed-form term 1 and on d_ij^2
    jacobian_route: str = "auto"
    steps: int = 5000
    batch_size: int = 256
    lr: float = 1e-3
    ema_decay: float = 0.999
    log_every: int = 250
    seed: int = 0


class ReplicateBatchSampler:
    """Single-replicate batches: replicate r with probability n_r / n, then B_r = min(B, n_r) of its
    cells uniformly without replacement; all within-batch pairs are used. With one replicate no
    replicate is drawn, so the random stream is that of plain uniform batches of all n cells."""

    def __init__(self, codes, batch_size, rng):
        codes = np.asarray(codes)
        self.R = int(codes.max()) + 1
        self.cells = [np.flatnonzero(codes == r) for r in range(self.R)]
        self.n_r = np.array([len(c) for c in self.cells])
        self.prob = self.n_r / self.n_r.sum()
        self.B = np.minimum(batch_size, self.n_r)
        self.rng = rng

    def sample_from(self, r):
        c = self.cells[r]
        return c[self.rng.choice(len(c), self.B[r], replace=False)]

    def sample(self):
        """(replicate r, cell indices of the batch)."""
        r = 0 if self.R == 1 else int(self.rng.choice(self.R, p=self.prob))
        return r, self.sample_from(r)


def setup_torch(seed):
    """float32 and fixed seeds; thread counts are left at library defaults."""
    torch.set_default_dtype(torch.float32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def step_terms(net, xb, Tb, gb, p, nubar, tau, route):
    """One batch: (term 1 mean, term 2 mean, term 3), all unweighted (lambda = 1).
    Tb: tangent bases of the batch's unique rows gb."""
    Y, m, C = jacobian_terms(net, xb, Tb, route)
    return term_iso(m, tau).mean(), term_rank(C).mean(), stress(Y, xb, gb, p, nubar, tau)


def train(X, T, g, nubar, sampler, cfg: TrainConfig, log_path=None):
    """Fit f_theta on working-space X (n, d), with tangent bases T (n_u, d, 2) of the unique rows,
    g (n,) the unique-row index of each cell and nubar (R,) the per-replicate stress normalizers,
    drawing single-replicate batches from `sampler`. Returns (EMA net, info)."""
    n, d = X.shape
    Xt = torch.as_tensor(X, dtype=torch.float32)
    Tt = torch.as_tensor(T, dtype=torch.float32)
    gt = torch.as_tensor(g, dtype=torch.int64)
    net = SurfaceMLP(d)
    route = select_route(d, net.width, cfg.jacobian_route)
    info = {"d": d, "n": n, "threads": torch.get_num_threads(), "jacobian_route": route,
            "batch_size": int(sampler.B.max())}

    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)
    ema = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(cfg.ema_decay))
    lam = np.array([1.0, cfg.lambda_rank, cfg.lambda_s])

    columns = ["step", "iso", "rank", "stress", "total", "wall_s"]
    rows = []
    writer = None
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = open(log_path, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(columns)
    t_main = time.perf_counter()
    try:
        for step in range(1, cfg.steps + 1):
            r, b = sampler.sample()
            b = torch.as_tensor(b)
            gb = gt[b]
            iso, rank, s = step_terms(net, Xt[b], Tt[gb], gb, cfg.stress_power, float(nubar[r]),
                                      cfg.tau, route)
            opt.zero_grad()
            (iso + cfg.lambda_rank * rank + cfg.lambda_s * s).backward()
            opt.step()
            sched.step()
            ema.update_parameters(net)
            terms = np.array([iso.item(), rank.item(), s.item()])
            if step % cfg.log_every == 0 or step == cfg.steps:
                row = [step, *terms, float(lam @ terms), time.perf_counter() - t_main]
                rows.append(row)
                if writer:
                    writer.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in row])
                    fh.flush()
                log.info("step %5d  iso %.4f  rank %.4f  stress %.4f  total %.4f  (%.0fs)",
                         step, *row[1:])
    finally:
        if writer:
            fh.close()
    info["train_time"] = time.perf_counter() - t_main
    info["step_time"] = info["train_time"] / cfg.steps
    info["log"] = rows
    info["log_columns"] = columns
    info["final"] = {"iso": float(terms[0]), "rank": float(terms[1]), "stress": float(terms[2]),
                     "total": float(lam @ terms)}  # final training batch
    final = copy.deepcopy(ema.module).eval()
    return final, info
