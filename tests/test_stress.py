import numpy as np
import pytest
import torch

from aep.losses import mean_normalizer, stress
from aep.train import ReplicateBatchSampler


def _data(seed=0, n=80, dups=0):
    """X (n, 5), a fixed embedding Y = f(X); the last `dups` rows duplicate earlier rows (in X and Y)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    Y = np.tanh(X @ rng.standard_normal((5, 3)))
    if dups:
        src = rng.choice(n - dups, dups, replace=False)
        X[n - dups:], Y[n - dups:] = X[src], Y[src]
    return torch.as_tensor(X), torch.as_tensor(Y)


def _pairs(X, Y):
    iu = torch.triu_indices(len(X), len(X), 1)
    delta = (X[iu[0]] - X[iu[1]]).norm(dim=1)
    d = (Y[iu[0]] - Y[iu[1]]).norm(dim=1)
    return delta, d


def _true_nubar(X, p):
    delta, _ = _pairs(X, X)
    nz = delta > 0
    return (delta[nz] ** (2 - p)).sum().item() / len(delta)


def _ids(X):
    """Unique-row index g of each row."""
    return torch.as_tensor(np.unique(X.numpy(), axis=0, return_inverse=True)[1].ravel())


@pytest.mark.parametrize("p", [0.0, 1.0, 2.0, 3.0, 4.0])
def test_term3_is_weighted_mean_of_relative_errors(p):
    """Test 3: whole data as one batch with the true normalizer, term 3 =
    sum omega e^2 / sum omega, omega = delta^(2 - p), e = (d - delta) / delta."""
    X, Y = _data()
    delta, d = _pairs(X, Y)
    omega, e = delta ** (2 - p), (d - delta) / delta
    value = stress(Y, X, _ids(X), p, _true_nubar(X, p))
    torch.testing.assert_close(value, (omega * e**2).sum() / omega.sum())


def test_special_cases():
    """Test 3: p = 0, 1, 2 give normalized raw stress, Sammon's stress and the mean of e^2."""
    X, Y = _data(1)
    delta, d = _pairs(X, Y)
    raw = ((d - delta) ** 2).sum() / (delta**2).sum()
    sammon = ((d - delta) ** 2 / delta).sum() / delta.sum()
    elastic = (((d - delta) / delta) ** 2).mean()
    for p, ref in ((0.0, raw), (1.0, sammon), (2.0, elastic)):
        torch.testing.assert_close(stress(Y, X, _ids(X), p, _true_nubar(X, p)), ref)


@pytest.mark.parametrize("p", [2.0, 3.0, 4.0])
def test_duplicates_finite(p):
    """Test 3: duplicated rows, default tau, float32: loss and gradient finite; duplicate pairs
    contribute nothing."""
    X, Y = _data(2, n=60, dups=6)
    ids = _ids(X)
    X, Y = X.float(), Y.float().requires_grad_()
    value = stress(Y, X, ids, p, _true_nubar(X.double(), p))
    g, = torch.autograd.grad(value, Y)
    assert torch.isfinite(value) and torch.isfinite(g).all()
    # coincident outputs of distinct inputs: also finite
    Y2 = Y.detach().clone()
    Y2[1] = Y2[0]
    Y2.requires_grad_()
    value = stress(Y2, X, ids, p, 1.0)
    g, = torch.autograd.grad(value, Y2)
    assert torch.isfinite(value) and torch.isfinite(g).all()


def test_normalizer_estimate():
    """The sampled nu_bar (M batches, duplicates contribute 0) is close to the all-pairs value."""
    X, _ = _data(3, n=300, dups=10)
    Xu, g = np.unique(X.numpy(), axis=0, return_inverse=True)
    sampler = ReplicateBatchSampler(np.zeros(len(X), dtype=int), 50, np.random.default_rng(0))
    est, = mean_normalizer(Xu, g.ravel(), 1.5, sampler, M=400)
    assert est == pytest.approx(_true_nubar(X, 1.5), rel=0.03)
