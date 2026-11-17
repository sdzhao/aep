import pytest
import torch
from torch.func import functional_call, jacrev, vmap

from aep.cotangent import jacobian, jacobian_terms, select_route
from aep.losses import term_iso, term_rank
from aep.model import SurfaceMLP


def _setup(d, B=8, seed=0):
    torch.manual_seed(seed)
    net = SurfaceMLP(d).double()
    x = torch.randn(B, d, dtype=torch.float64)
    T = torch.linalg.qr(torch.randn(B, d, 2, dtype=torch.float64)).Q
    return net, x, T


def _jacrev(net, x, params=None):
    params = dict(net.named_parameters()) if params is None else params
    return vmap(jacrev(lambda xi: functional_call(net, params, (xi[None],))[0]))(x)


@pytest.mark.parametrize("d", [5, 80])          # d < h and d > h (h = 64)
def test_jacobian_matches_jacrev_and_finite_differences(d):
    """Test 2a: U1 W1 matches torch.func.jacrev and central finite differences."""
    net, x, _ = _setup(d)
    J = jacobian(net, x)
    torch.testing.assert_close(J, _jacrev(net, x))
    h = 1e-6
    with torch.no_grad():
        for k in range(d):
            e = torch.zeros(d, dtype=torch.float64)
            e[k] = h
            fd = (net(x + e) - net(x - e)) / (2 * h)
            torch.testing.assert_close(J[:, :, k], fd, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("route", ["explicit", "factorized"])
@pytest.mark.parametrize("d", [5, 80])
def test_tangent_derivatives_and_gram(d, route):
    """Test 2b: both routes give [m1 m2] = J T and C = J J^T."""
    net, x, T = _setup(d)
    Y, m, C = jacobian_terms(net, x, T, route)
    J = _jacrev(net, x)
    torch.testing.assert_close(Y, net(x))
    torch.testing.assert_close(m, J @ T)
    torch.testing.assert_close(C, J @ J.transpose(1, 2))


def test_auto_route_selects_explicit_iff_d_le_h():
    assert select_route(5, 64) == select_route(64, 64) == "explicit"
    assert select_route(65, 64) == "factorized"
    assert select_route(5, 64, "factorized") == "factorized"
    with pytest.raises(ValueError):
        select_route(5, 64, "bogus")


@pytest.mark.parametrize("route", ["explicit", "factorized"])
@pytest.mark.parametrize("d", [5, 80])
def test_parameter_gradients_match_jacrev_reference(d, route):
    """Test 2c: parameter gradients of terms 1-2 match a jacrev
    (double-backward) reference, for both routes."""
    net, x, T = _setup(d)

    def loss_cotangent():
        _, m, C = jacobian_terms(net, x, T, route)
        return term_iso(m).mean() + term_rank(C).mean()

    def loss_jacrev():
        J = _jacrev(net, x)
        return term_iso(J @ T).mean() + term_rank(J @ J.transpose(1, 2)).mean()

    J = _jacrev(net, x).detach()
    mu = torch.linalg.eigvalsh(J @ J.transpose(1, 2))
    assert (mu[:, 1] > 1.01 * mu[:, 0]).all() and (mu[:, 0] > 0).all()   # non-degenerate
    params = list(net.parameters())
    g1 = torch.autograd.grad(loss_cotangent(), params, allow_unused=True)
    g2 = torch.autograd.grad(loss_jacrev(), params, allow_unused=True)
    for p, a, b in zip(params, g1, g2):  # the output bias does not enter J: no/zero gradient
        a = torch.zeros_like(p) if a is None else a
        b = torch.zeros_like(p) if b is None else b
        torch.testing.assert_close(a, b)
    assert any(a is not None and a.abs().max() > 0 for a in g1)
