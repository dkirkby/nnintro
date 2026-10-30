"""Curation: the step that makes a trained model a teaching asset.

The criteria are about legibility, not accuracy, and they have to hold for
every architecture and activation in the catalog -- including the ones with
no hidden layer to diagnose and the ones that cannot fit their dataset.
"""

import numpy as np
import torch

from nnintro import datasets
from nnintro.model import DenseNet
from nnintro.train import (
    SELECTION_CRITERIA,
    TrainConfig,
    _unit_report,
    curate,
    diagnose_units,
    train,
)


def _with_parameters(sizes, activation, layers):
    """A model with parameters set by hand, so a diagnostic has a known answer."""
    model = DenseNet(sizes, activation, seed=0)
    with torch.no_grad():
        for layer, (w, b) in zip(model.layers, layers):
            layer.weight.copy_(torch.as_tensor(w, dtype=torch.float64))
            layer.bias.copy_(torch.as_tensor(b, dtype=torch.float64))
    return model


def test_every_hidden_layer_is_diagnosed_not_just_the_first():
    ds = datasets.build("tilted-plane")
    result = train(ds, TrainConfig(sizes=[2, 2, 4, 1], activation="relu", steps=50))
    units = diagnose_units(result.model, ds)
    assert [(u.layer, u.unit) for u in units] == [
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
    ]


def test_positive_fraction_locates_the_units_hyperplane():
    # Unit 0: z = x1, so positive on exactly half of a symmetric domain.
    # Unit 1: z = x1 + 10, so positive everywhere -- no transition to see.
    # Unit 2: z = x1 - 10, so never positive: a permanently dead ReLU.
    model = _with_parameters(
        [2, 3, 1],
        "relu",
        [([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], [0.0, 10.0, -10.0]), ([[1.0, 1.0, 1.0]], [0.0])],
    )
    ds = datasets.build("tilted-plane")
    units = diagnose_units(model, ds)
    # The grid is 41 x 41 and the test is strict, so the x1 = 0 column --
    # one column in 41, exactly on the kink -- counts as not positive.
    assert units[0].positive_fraction == 20 / 41
    assert units[1].positive_fraction == 1.0
    assert units[2].positive_fraction == 0.0
    # Variation catches the dead one, which is what the criterion is for.
    assert units[2].relative_variation == 0.0
    assert units[0].relative_variation > SELECTION_CRITERIA["minRelativeVariation"]


def test_a_dead_relu_is_rejected_and_a_crossing_unit_is_not():
    ds = datasets.build("tilted-plane")
    config = TrainConfig(sizes=[2, 3, 1], activation="relu", steps=1)

    dead = _with_parameters(
        [2, 3, 1],
        "relu",
        [([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], [0.0, 0.0, -10.0]), ([[1.0, 1.0, 1.0]], [0.0])],
    )
    report = _unit_report(_fake_result(dead, config), ds)
    assert report["checks"]["noDeadUnits"] is False
    assert report["accepted"] is False

    alive = _with_parameters(
        [2, 3, 1],
        "relu",
        [([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [0.0, 0.0, 0.0]), ([[1.0, 1.0, 1.0]], [0.0])],
    )
    report = _unit_report(_fake_result(alive, config), ds)
    assert report["checks"] == {
        "noVestigialUnits": True,
        "noDeadUnits": True,
        "hasTransitionInDomain": True,
    }
    assert report["accepted"] is True


def test_a_saturated_tanh_is_dead_too():
    """What `relative_variation` buys over the old active-fraction test: a
    tanh pinned at +1 everywhere is active by any ReLU-shaped measure."""
    model = _with_parameters(
        [2, 2, 1],
        "tanh",
        [([[0.0, 0.0], [1.0, 0.0]], [40.0, 0.0]), ([[1.0, 1.0]], [0.0])],
    )
    ds = datasets.build("tilted-plane")
    units = diagnose_units(model, ds)
    assert np.isclose(units[0].relative_variation, 0.0)
    report = _unit_report(_fake_result(model, TrainConfig(sizes=[2, 2, 1], activation="tanh")), ds)
    assert report["checks"]["noDeadUnits"] is False


def test_identity_is_not_asked_for_a_transition_it_cannot_have():
    ds = datasets.build("tilted-plane")
    # Both units always positive, so no transition crosses the domain. For
    # ReLU that is a rejection; for identity there is nothing to find.
    layers = [([[1.0, 0.0], [0.0, 1.0]], [10.0, 10.0]), ([[1.0, 1.0]], [0.0])]
    for activation, expected in (("relu", False), ("identity", True)):
        model = _with_parameters([2, 2, 1], activation, layers)
        config = TrainConfig(sizes=[2, 2, 1], activation=activation)
        report = _unit_report(_fake_result(model, config), ds)
        assert report["checks"]["hasTransitionInDomain"] is expected


def test_a_network_with_no_hidden_layer_passes_vacuously():
    ds = datasets.build("xor-quadrants")
    config = TrainConfig(sizes=[2, 1], activation="identity", steps=20)
    result = train(ds, config)
    report = _unit_report(result, ds)
    assert report["units"] == []
    assert report["accepted"] is True


def test_curation_prefers_training_loss_and_records_what_it_cost():
    """Selecting on the held-out loss would bias the one number that is
    supposed to say something about generalization."""
    ds = datasets.build("curved-surface")
    config = TrainConfig(sizes=[2, 4, 1], activation="relu", steps=300)
    result, report = curate(ds, config, attempts=4)
    assert report["accepted"] is True
    assert report["bestTrainLossAccepted"] == result.train_loss
    assert report["bestTrainLossOverall"] <= result.train_loss
    assert report["accepted_count"] + report["rejected_count"] == 4 - report["diverged_count"]
    assert report["criteria"] == SELECTION_CRITERIA


def _fake_result(model, config):
    """A TrainResult for a hand-built model: only the model and config are read."""
    from nnintro.train import TrainResult

    return TrainResult(model=model, config=config, train_loss=0.0, test_loss=0.0, losses=[])
