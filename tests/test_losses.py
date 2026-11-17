import numpy as np
import pytest
import torch

from aep.losses import term_iso, term_rank


def _m_cases(seed=0):
    """[m1 m2] (B, 3, 2): random, near-isotropic, scaled orthonormal and rank-deficient (rank 1)."""
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.standard_normal((5, 3, 2)))[0]
    return np.concatenate([rng.standard_normal((20, 3, 2)),
                           Q * np.array([1.0, 1.0 + 1e-4]),
                           Q * rng.uniform(0.1, 10, (5, 1, 1)),
                           rng.standard_normal((5, 3, 1)) * rng.standard_normal((5, 1, 2))])


def _svd_ref(m):
    """1/2 sum (s - 1)^2 and 1/2 ||m - R*||^2, R* = polar factor (orthogonal Procrustes)."""
    U, s, Vt = np.linalg.svd(m, full_matrices=False)
    R = U @ Vt
    return 0.5 * ((s - 1) ** 2).sum(1), 0.5 * np.linalg.norm(m - R, axis=(1, 2)) ** 2


def test_arap_closed_form_matches_svd_and_procrustes():
    """Test 1: closed form = 1/2 sum (s_k - 1)^2 = 1/2 ||J T - R*||^2 (random, near-isotropic,
    rank-deficient)."""
    m = _m_cases()
    iso = term_iso(torch.as_tensor(m)).numpy()
    ref_s, ref_R = _svd_ref(m)
    np.testing.assert_allclose(iso, ref_s, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(iso, ref_R, rtol=1e-5, atol=1e-5)


def _fd(m):
    h = 1e-6
    fd = np.zeros_like(m)
    for idx in np.ndindex(m.shape):
        e = np.zeros_like(m)
        e[idx] = h
        fd[idx] = (term_iso(torch.as_tensor((m + e)[None])) - term_iso(torch.as_tensor((m - e)[None]))).item() / (2 * h)
    return fd


@pytest.mark.parametrize("case", ["rank_deficient", "zero", "random", "isometric"])
def test_arap_gradient_finite_and_matches_finite_differences(case):
    """Test 1: gradient wrt [m1 m2] finite at rank-deficient and zero J T; matches central
    differences at random full rank and at J T = R (isometric)."""
    rng = np.random.default_rng(2)
    m = {"rank_deficient": rng.standard_normal((3, 1)) * rng.standard_normal((1, 2)),
         "zero": np.zeros((3, 2)), "random": rng.standard_normal((3, 2)),
         "isometric": np.linalg.qr(rng.standard_normal((3, 2)))[0]}[case]
    mt = torch.as_tensor(m[None]).requires_grad_()
    g, = torch.autograd.grad(term_iso(mt).sum(), mt)
    assert torch.isfinite(g).all()
    if case in ("random", "isometric"):
        np.testing.assert_allclose(g[0].numpy(), _fd(m), rtol=1e-5, atol=1e-7)


def test_guard_effect():
    """Test 1: default tau matches SVD to 1e-5 relative for s >= 1e-2; a larger tau overshoots
    p = s1 + s2 by at most sqrt(tau)/s1 at rank-deficient J T."""
    rng = np.random.default_rng(3)
    U = np.linalg.qr(rng.standard_normal((50, 3, 2)))[0]
    s = rng.uniform(1e-2, 3, (50, 2))
    m = U * s[:, None, :]
    np.testing.assert_allclose(term_iso(torch.as_tensor(m)).numpy(), _svd_ref(m)[0], rtol=1e-5)
    tau = 1e-4
    m = rng.standard_normal((20, 3, 1)) * rng.standard_normal((20, 1, 2))   # rank 1
    s1 = np.linalg.svd(m, compute_uv=False)[:, 0]
    # l = 1/2 (a + c - 2p + 2) with a + c exact: the overshoot of p is (exact l - guarded l)
    over = _svd_ref(m)[0] - term_iso(torch.as_tensor(m), tau).numpy()
    assert (over >= -1e-12).all() and (over <= np.sqrt(tau) / s1 + tau).all()   # + O(tau)


def test_rank_eigenvalue():
    """Test 1: mu_3 from batched eigvalsh = sigma_3(J)^2 (random J)."""
    rng = np.random.default_rng(0)
    J = rng.standard_normal((20, 3, 7))
    rank = term_rank(torch.as_tensor(J @ np.swapaxes(J, 1, 2))).numpy()
    np.testing.assert_allclose(rank, np.linalg.svd(J, compute_uv=False)[:, 2] ** 2, rtol=1e-5)
