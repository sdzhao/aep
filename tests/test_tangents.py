import numpy as np
import pytest

from aep.knn import knn
from aep.tangents import tangent_bases


@pytest.mark.parametrize("d, k", [(30, 10), (5, 20)])     # k < d and k >= d
def test_gram_and_cov_routes_same_projector(d, k):
    """Test 4: the Gram and covariance routes give the same projector T T^T."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, d)) * np.linspace(4, 0.5, d)
    idx, _ = knn(X, k)
    Tg = tangent_bases(X, idx, chunk=128, route="gram")
    Tc = tangent_bases(X, idx, chunk=128, route="cov")
    Pg = Tg @ np.swapaxes(Tg, 1, 2)
    Pc = Tc @ np.swapaxes(Tc, 1, 2)
    np.testing.assert_allclose(Pg, Pc, atol=1e-4)


def test_duplicates_identical():
    """Test 4: duplicate rows receive identical T (preprocessing on unique rows)."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((300, 6))
    X[250:] = X[:50]
    Xu, g = np.unique(X, axis=0, return_inverse=True)
    g = g.ravel()
    idx, _ = knn(Xu, 20)
    T = tangent_bases(Xu, idx)
    np.testing.assert_array_equal(T[g[250:]], T[g[:50]])
