"""Test 5 (pooled PCA) and test 6 (replicates)."""
import numpy as np
import pytest
import torch

from aep import fit_aep
from aep.losses import mean_normalizer, stress
from aep.preprocess import encode_replicates, fit_working_space, replicate_units
from aep.train import ReplicateBatchSampler


def _data(seed=0, n=(120, 60, 90), d=6):
    """Three replicates of different sizes; the first 10 rows of replicate 1 copy rows of
    replicate 0 (identical rows across replicates), and replicate 2 has 8 within-replicate duplicates."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((sum(n), d)) * np.linspace(3, 0.5, d)
    labels = np.repeat(np.array(["a", "b", "c"]), n)
    X[n[0]:n[0] + 10] = X[:10]
    s = n[0] + n[1]
    X[s + 82:s + 90] = X[s:s + 8]
    return X, labels


def test_single_replicate_matches_none():
    """replicate=None and all labels equal give the same training and embedding."""
    X, _ = _data()
    kw = {"steps": 30, "batch_size": 64, "seed": 3}
    Y0, _, i0 = fit_aep(X, **kw)
    Y1, _, i1 = fit_aep(X, replicate=np.full(len(X), "s1"), **kw)
    np.testing.assert_array_equal(Y0, Y1)
    assert [r[:-1] for r in i0["log"]] == [r[:-1] for r in i1["log"]]   # all but wall time
    assert i0["nu_bar"] == i1["nu_bar"]


def test_single_replicate_matches_global_unique_rows():
    """With one replicate, preprocessing equals the single-sample route on np.unique(X)."""
    X, _ = _data()
    _, codes = encode_replicates(None, len(X))
    Xu, g, rep_u, _, _ = replicate_units(X, codes, 20)
    Xu0, g0 = np.unique(X, axis=0, return_inverse=True)
    np.testing.assert_array_equal(Xu, Xu0)
    np.testing.assert_array_equal(g, g0.ravel())
    assert (rep_u == 0).all()


def test_batches_single_replicate_with_size_probabilities():
    X, labels = _data()
    _, codes = encode_replicates(labels, len(X))
    sampler = ReplicateBatchSampler(codes, 100, np.random.default_rng(0))
    counts = np.zeros(3)
    for _ in range(6000):
        r, b = sampler.sample()
        assert (codes[b] == r).all() and len(b) == min(100, sampler.n_r[r])
        assert len(np.unique(b)) == len(b)
        counts[r] += 1
    np.testing.assert_allclose(counts / counts.sum(), sampler.n_r / sampler.n_r.sum(), atol=0.02)


def test_knn_within_replicate_and_cross_replicate_duplicates_distinct():
    X, labels = _data()
    _, codes = encode_replicates(labels, len(X))
    Xu, g, rep_u, idx, T = replicate_units(X, codes, 20)
    assert (rep_u[idx] == rep_u[:, None]).all()          # all neighbors from the own replicate
    assert (rep_u[g] == codes).all()
    n0 = 120
    # identical rows in replicates a and b: different units, different neighborhoods
    assert (g[:10] != g[n0:n0 + 10]).all()
    np.testing.assert_array_equal(Xu[g[:10]], Xu[g[n0:n0 + 10]])
    # within-replicate duplicates share the unit (and so T)
    s = 180
    np.testing.assert_array_equal(g[s + 82:s + 90], g[s:s + 8])
    assert len(Xu) == len(X) - 8 and T.shape == (len(Xu), X.shape[1], 2)


def test_normalizers_from_own_replicate():
    """nu_bar_r equals the estimate from replicate r alone (same sampler stream per replicate)."""
    X, labels = _data()
    X[labels == "b"] *= 3.0                               # replicates on different scales
    _, codes = encode_replicates(labels, len(X))
    Xu, g, _, _, _ = replicate_units(X, codes, 20)
    p = 1.0
    nubar = mean_normalizer(Xu, g, p, ReplicateBatchSampler(codes, 40, np.random.default_rng(0)), M=300)
    for r in range(3):
        cells = codes == r
        D = np.linalg.norm(X[cells][:, None] - X[cells][None], axis=-1)
        iu = np.triu_indices(cells.sum(), 1)
        true = (D[iu][D[iu] > 0] ** (2 - p)).sum() / len(iu[0])
        assert nubar[r] == pytest.approx(true, rel=0.05)
    assert nubar[1] == pytest.approx(3 * nubar[0], rel=0.1)


def test_term3_never_uses_cross_replicate_pairs():
    """Batches are single-replicate, so term 3 on a batch only sees within-replicate pairs: its
    value equals the stress computed on that replicate's cells alone."""
    X, labels = _data()
    _, codes = encode_replicates(labels, len(X))
    _, g, _, _, _ = replicate_units(X, codes, 20)
    sampler = ReplicateBatchSampler(codes, 1000, np.random.default_rng(1))   # B_r = n_r
    for _ in range(10):
        r, b = sampler.sample()
        assert set(b) == set(np.flatnonzero(codes == r))
        Y = torch.as_tensor(np.tanh(X[b, :3]))
        v = stress(Y, torch.as_tensor(X[b]), torch.as_tensor(g[b]), 2.0, 1.0)
        assert torch.isfinite(v)


def test_validation():
    X, labels = _data()
    with pytest.raises(ValueError):
        encode_replicates(labels[:-1], len(X))
    bad = labels.copy()
    bad[0] = "z"                                          # a replicate with one cell
    with pytest.raises(ValueError):
        encode_replicates(bad, len(X))
    _, codes = encode_replicates(labels, len(X))
    with pytest.raises(ValueError):
        replicate_units(X, codes, 61)                     # k > n_u of replicate b (60)


def test_pca_pooled():
    """Test 5: PCA is fit on all pooled cells, each weighted equally."""
    X, labels = _data()
    _, tf = fit_working_space(X, n_pca=4)
    np.testing.assert_allclose(tf.mean, X.mean(0))
    _, _, info = fit_aep(X, replicate=labels, n_pca=4, steps=5, batch_size=32)
    assert info["n_r"] == [120, 60, 90] and info["replicates"] == ["a", "b", "c"]
    assert len(info["nu_bar"]) == 3
