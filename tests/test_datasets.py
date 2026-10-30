import numpy as np
import pytest

from nnintro import datasets


def test_tilted_plane_is_exactly_its_stated_formula():
    ds = datasets.tilted_plane()
    expected = 0.6 * ds.x[:, 0] - 0.4 * ds.x[:, 1] + 0.2
    assert np.array_equal(ds.y, expected)


@pytest.mark.parametrize("dataset_id", list(datasets.BUILDERS))
def test_samples_lie_inside_the_declared_domain(dataset_id):
    # The Gaussian generators are truncated by rejection rather than
    # clipped, so this also checks that nothing was piled onto the frame:
    # a clipped tail would sit exactly on a bound.
    ds = datasets.build(dataset_id)
    (lo1, hi1), (lo2, hi2) = ds.domain
    assert ds.x[:, 0].min() >= lo1 and ds.x[:, 0].max() <= hi1
    assert ds.x[:, 1].min() >= lo2 and ds.x[:, 1].max() <= hi2
    assert not np.isin(ds.x, [lo1, hi1, lo2, hi2]).any()


@pytest.mark.parametrize("dataset_id", list(datasets.BUILDERS))
def test_every_dataset_is_fixed_labelled_and_split(dataset_id):
    a, b = datasets.build(dataset_id), datasets.build(dataset_id)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)
    assert np.array_equal(a.is_test, b.is_test)
    assert a.label and a.description and a.task in {"regression", "classification"}
    assert a.is_test.sum() == round(len(a.y) * datasets.TEST_FRACTION)


@pytest.mark.parametrize("dataset_id", ["gaussian-clusters", "xor-quadrants"])
def test_classification_targets_are_balanced_class_labels(dataset_id):
    ds = datasets.build(dataset_id)
    assert set(np.unique(ds.y)) == {0.0, 1.0}
    assert ds.y.sum() == len(ds.y) / 2
    # Stochastic in x given the class, so an arbitrary point has no label to
    # report and the explorer must not invent one (PLAN.md section 3).
    assert ds.has_analytic_target is False
    with pytest.raises(KeyError):
        datasets.analytic_target(ds.id, ds.x)


def test_the_classifiers_have_irreducible_error():
    """Overlap is the point: it is what makes a train/held-out gap mean
    something, since neither regression generator has any noise at all."""
    clusters = datasets.build("gaussian-clusters")
    ideal = (clusters.x.sum(axis=1) > 0).astype(float)
    assert 0.02 < (ideal != clusters.y).mean() < 0.2

    quadrants = datasets.build("xor-quadrants")
    ideal = ((quadrants.x[:, 0] > 0) == (quadrants.x[:, 1] > 0)).astype(float)
    assert 0.02 < (ideal != quadrants.y).mean() < 0.25


def test_xor_is_not_linearly_separable():
    """The teaching purpose of the dataset: the 2 -> 1 and identity variants
    fail on it by construction, and that imperfect fit is the asset."""
    ds = datasets.build("xor-quadrants")
    # Each class occupies two opposite quadrants, so no half-plane holds one
    # of them: both classes appear on both sides of either axis.
    for axis in (0, 1):
        for side in (ds.x[:, axis] > 0, ds.x[:, axis] < 0):
            assert set(np.unique(ds.y[side])) == {0.0, 1.0}


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


@pytest.mark.parametrize("dataset_id", ["tilted-plane", "curved-surface"])
def test_analytic_target_agrees_with_the_sampled_targets(dataset_id):
    ds = datasets.build(dataset_id)
    assert ds.has_analytic_target is True
    assert np.allclose(datasets.analytic_target(ds.id, ds.x), ds.y)


def test_the_curved_surface_is_exactly_its_stated_formula():
    ds = datasets.curved_surface()
    expected = np.sin(np.pi * ds.x[:, 0]) * np.cos(np.pi * ds.x[:, 1])
    assert np.array_equal(ds.y, expected)


def test_unknown_dataset_is_rejected():
    with pytest.raises(KeyError):
        datasets.build("no-such-dataset")
