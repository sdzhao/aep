"""Term 1 ARAP tangent isometry (closed form), term 2 rank penalty mu_3, term 3 power-weighted
stress, and the per-replicate stress normalizers nu_bar_r."""
import numpy as np
import torch

TAU = 1e-12   # guard in the closed-form term 1 and on coincident outputs in term 3 (absolute)


def term_iso(m, tau=TAU):
    """ARAP energy 1/2 sum_k (s_k - 1)^2 per row, s_k = singular values of [m1 m2] (B, 3, 2):
    1/2 (a + c - 2p + 2) with q = sqrt(|m1 x m2|^2 + tau) = s1 s2 and
    p = sqrt(a + c + 2q + tau) = s1 + s2 (no eigensolver, no SVD). The guard keeps the gradient
    finite when [m1 m2] is rank-deficient or zero."""
    m1, m2 = m[..., 0], m[..., 1]
    ac = (m1 * m1).sum(-1) + (m2 * m2).sum(-1)
    q = torch.sqrt(torch.linalg.cross(m1, m2).pow(2).sum(-1) + tau)
    p = torch.sqrt(ac + 2 * q + tau)
    return 0.5 * (ac - 2 * p + 2)


def term_rank(C):
    """mu_3 = smallest eigenvalue of C = J J^T (B, 3, 3) = sigma_3(J)^2 per row, from batched
    eigvalsh (round-off negatives set to 0). Eigenvalue-only gradients need no eigen-gap division."""
    return torch.linalg.eigvalsh(0.5 * (C + C.transpose(1, 2)))[:, 0].clamp_min(0)


def sq_pair_dists(Z):
    """||z_i - z_j||^2 by direct differences (B x B x dim; no Gram-trick cancellation for near pairs)."""
    return (Z[:, None, :] - Z[None, :, :]).pow(2).sum(-1)


def stress(Y, Xb, g, p, nubar, tau=TAU):
    """Unbiased within-batch estimate of 1/nu sum_{i<j} w_ij (d_ij - delta_ij)^2 (lambda_s = 1),
    delta_ij = ||x_i - x_j|| on the batch inputs Xb (B, d), w_ij = delta_ij^-p,
    nu = N nubar; the diagonal and duplicate pairs (same unique row, g_i = g_j) are masked.
    Targets, mask and weights carry no gradient; d_ij is differentiated by autograd."""
    B = len(Y)
    with torch.no_grad():
        mask = torch.triu(torch.ones(B, B, dtype=torch.bool), 1) & (g[:, None] != g[None, :])
        delta = torch.where(mask, sq_pair_dists(Xb), 1.0).sqrt()
        w = torch.where(mask, delta.double().pow(-p), 0.0)      # float64: no overflow for small delta
    d = torch.where(mask, sq_pair_dists(Y).clamp_min(tau), 1.0).sqrt()
    total = (w * (d - delta).pow(2).double()).sum()
    return (total / (nubar * B * (B - 1) / 2)).to(Y.dtype)


def mean_normalizer(Xu, g, p, sampler, M=100):
    """Per-replicate nu_bar_r = nu_r / N_r (R,), each estimated as the mean of
    omega_ij = delta_ij^(2 - p) over all within-batch pairs of M batches of replicate r drawn with
    the training sampler (sampler.sample_from(r)), on the unique rows Xu (n_u, d) indexed by the
    unique-row index g (n,); duplicate pairs (g_i = g_j) contribute 0. Float64."""
    Xu = np.asarray(Xu, dtype=np.float64)
    nubar = np.empty(sampler.R)
    for r in range(sampler.R):
        total, count = 0.0, 0
        for _ in range(M):
            gb = g[sampler.sample_from(r)]
            iu = np.triu_indices(len(gb), 1)
            nz = gb[iu[0]] != gb[iu[1]]
            delta = np.linalg.norm(Xu[gb[iu[0][nz]]] - Xu[gb[iu[1][nz]]], axis=1)
            total += (delta ** (2 - p)).sum()
            count += len(iu[0])
        nubar[r] = total / count
    return nubar
