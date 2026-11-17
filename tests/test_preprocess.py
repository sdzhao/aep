import numpy as np
import pytest

from aep.preprocess import fit_working_space, top_pcs


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return rng.standard_normal((300, 12)) @ rng.standard_normal((12, 12))


def test_default_n_pca(data):
    X, tf = fit_working_space(data)
    assert X.shape[1] == 10
    X5, _ = fit_working_space(data[:, :5])
    assert X5.shape[1] == 5
    assert 0 < tf.variance_retained <= 1


def test_no_pca_is_identity(data):
    X, tf = fit_working_space(data, pca=False)
    np.testing.assert_array_equal(X, data)
    np.testing.assert_array_equal(tf.transform(data[:7]), data[:7])


@pytest.mark.parametrize("n_pca", [2, 13, 0])
def test_invalid_n_pca_raises(data, n_pca):
    with pytest.raises(ValueError):
        fit_working_space(data, n_pca=n_pca)


def test_stored_transform_reproduces_projection(data):
    train, new = data[:250], data[250:]
    X, tf = fit_working_space(train, n_pca=4)
    np.testing.assert_allclose(tf.transform(train), X, atol=1e-10)
    # same transform applied to new rows: (x - mean) @ V^T
    np.testing.assert_allclose(tf.transform(new), (new - train.mean(0)) @ tf.components.T)


def _gapped(n=2000, D=200, k=10, seed=0):
    """Data with k strong directions and a clear spectral gap to the noise."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    scale = np.r_[np.linspace(20, 10, k), np.full(D - k, 0.5)]
    return (rng.standard_normal((n, D)) * scale) @ Q.T + 3.0


def test_randomized_matches_full():
    """Randomized PCA (default) spans the same top-k subspace as the exact SVD and reports the same
    variance retained."""
    X = _gapped()
    Xr, tr = fit_working_space(X, n_pca=10, solver="randomized")
    Xf, tf = fit_working_space(X, n_pca=10, solver="full")
    Pr, Pf = tr.components.T @ tr.components, tf.components.T @ tf.components
    assert np.abs(Pr - Pf).max() < 1e-6
    assert tr.variance_retained == pytest.approx(tf.variance_retained, rel=1e-8)
    np.testing.assert_allclose(tr.explained_variance_ratio, tf.explained_variance_ratio, rtol=1e-6)
    np.testing.assert_allclose(tr.transform(X), Xr)                  # stored transform reproduces it
    np.testing.assert_allclose(np.abs(Xr), np.abs(Xf), atol=1e-4)    # equal up to sign per PC
    np.testing.assert_allclose(np.abs(top_pcs(X, 10, solver="randomized")),
                               np.abs(top_pcs(X, 10, solver="full")), atol=1e-4)


def test_randomized_is_seeded():
    X = _gapped(seed=1)
    a = fit_working_space(X, n_pca=5, seed=3, solver="randomized")[1].components
    b = fit_working_space(X, n_pca=5, seed=3, solver="randomized")[1].components
    np.testing.assert_array_equal(a, b)


def test_unknown_solver_raises():
    with pytest.raises(ValueError):
        fit_working_space(_gapped(n=50, D=20), n_pca=5, solver="arpack")


def test_auto_solver_choice(monkeypatch):
    """auto: exact SVD for narrow data (identical to solver="full"), randomized for wide data."""
    import aep.preprocess as pp
    X = _gapped(n=600, D=40)
    np.testing.assert_array_equal(fit_working_space(X, n_pca=10)[1].components,
                                  fit_working_space(X, n_pca=10, solver="full")[1].components)
    monkeypatch.setattr(pp, "RANDOMIZED_MIN_DIM", 30)
    np.testing.assert_array_equal(fit_working_space(X, n_pca=10)[1].components,
                                  fit_working_space(X, n_pca=10, solver="randomized")[1].components)
