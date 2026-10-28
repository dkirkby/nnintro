import numpy as np
import pytest

from nnintro import datasets


def test_tilted_plane_is_exactly_its_stated_formula():
    ds = datasets.tilted_plane()
    expected = 0.6 * ds.x[:, 0] - 0.4 * ds.x[:, 1] + 0.2
    assert np.array_equal(ds.y, expected)


def test_samples_lie_inside_the_declared_domain():
    ds = datasets.tilted_plane()
    (lo1, hi1), (lo2, hi2) = ds.domain
    assert ds.x[:, 0].min() >= lo1 and ds.x[:, 0].max() <= hi1
    assert ds.x[:, 1].min() >= lo2 and ds.x[:, 1].max() <= hi2


def test_the_sample_is_fixed_across_calls():
    a, b = datasets.tilted_plane(), datasets.tilted_plane()
    assert np.array_equal(a.x, b.x)
    assert np.array_equal(a.is_test, b.is_test)


def test_the_split_is_a_permutation_not_the_head():
    ds = datasets.tilted_plane()
    n = len(ds.y)
    assert ds.is_test.sum() == round(n * datasets.TEST_FRACTION)
    # A head split would put every test index in the final fifth.
    assert np.flatnonzero(ds.is_test).min() < n * 0.8
    assert len(ds.train[1]) + len(ds.test[1]) == n


def test_analytic_target_agrees_with_the_sampled_targets():
    ds = datasets.tilted_plane()
    assert np.allclose(datasets.analytic_target(ds.id, ds.x), ds.y)


def test_unknown_dataset_is_rejected():
    with pytest.raises(KeyError):
        datasets.build("no-such-dataset")
