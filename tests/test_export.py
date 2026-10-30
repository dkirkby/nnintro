import json

import numpy as np
import pytest
import torch

from nnintro import datasets, export
from nnintro.cli import _model_from_payload
from nnintro.model import mean_loss
from nnintro.train import TrainConfig, train


@pytest.fixture(scope="module")
def payload():
    ds = datasets.tilted_plane()
    result = train(ds, TrainConfig(sizes=[2, 4, 1], activation="relu", steps=200))
    return export.build_checkpoint(result, ds)


def test_weights_are_exported_destination_first(payload):
    first = payload["parameters"][0]
    assert np.array(first["W"]).shape == (4, 2)  # [out][in]
    assert np.array(first["b"]).shape == (4,)
    assert payload["architecture"]["weightLayout"].startswith("W[out][in]")


def test_golden_probes_reproduce_from_parameters(payload):
    """The stored parameters alone must regenerate every stored probe."""
    model = _model_from_payload(payload)
    for probe in payload["goldenProbes"]:
        trace = model.trace(torch.as_tensor(probe["x"], dtype=torch.float64))
        assert float(trace.output.detach()) == pytest.approx(probe["output"], abs=1e-12)
        for stored, got in zip(probe["act"], trace.act):
            assert got.tolist() == pytest.approx(stored, abs=1e-12)


def test_input_sensitivity_matches_finite_differences(payload):
    model = _model_from_payload(payload)
    for probe in payload["goldenProbes"]:
        if probe["id"] == "relu-kink":
            continue  # no derivative exists there; see the convention test
        x = np.array(probe["x"])
        for axis, stored in enumerate(probe["inputSensitivity"]):
            step = 1e-6
            shift = np.zeros(2)
            shift[axis] = step
            with torch.no_grad():
                hi = float(model(torch.as_tensor(x + shift)))
                lo = float(model(torch.as_tensor(x - shift)))
            assert (hi - lo) / (2 * step) == pytest.approx(stored, abs=1e-6)


def test_split_gradient_differentiates_the_mean_loss(payload):
    """Not a mean of per-example gradients' magnitudes: the real thing."""
    ds = datasets.tilted_plane()
    model = _model_from_payload(payload)
    x, y = (torch.as_tensor(a) for a in ds.train)
    loss = mean_loss(model(x), y, ds.task)
    grads = export._parameter_gradient(model, loss)
    stored = payload["splitGradients"]["train"]["gradient"]
    for (w, b), layer in zip(grads, stored):
        assert np.allclose(w.numpy(), layer["W"], rtol=0, atol=1e-12)
        assert np.allclose(b.numpy(), layer["b"], rtol=0, atol=1e-12)


def test_split_gradient_matches_finite_differences(payload):
    """An independent check on one scalar parameter of each layer."""
    ds = datasets.tilted_plane()
    model = _model_from_payload(payload)
    x, y = (torch.as_tensor(a) for a in ds.train)
    step = 1e-7
    for index, layer in enumerate(model.layers):
        stored = payload["splitGradients"]["train"]["gradient"][index]["W"][0][0]
        with torch.no_grad():
            original = float(layer.weight[0, 0])
            layer.weight[0, 0] = original + step
            hi = float(mean_loss(model(x), y, ds.task))
            layer.weight[0, 0] = original - step
            lo = float(mean_loss(model(x), y, ds.task))
            layer.weight[0, 0] = original
        assert (hi - lo) / (2 * step) == pytest.approx(stored, rel=1e-4, abs=1e-9)


def test_the_kink_probe_sits_exactly_on_a_kink(payload):
    """phi'(0) = 0 is a convention, so the fixture must be exactly at 0."""
    probe = next(p for p in payload["goldenProbes"] if p["id"] == "relu-kink")
    assert 0.0 in probe["pre"][1]
    # The kinked unit is off, and the convention is recorded in the asset.
    index = probe["pre"][1].index(0.0)
    assert probe["act"][1][index] == 0.0
    assert payload["architecture"]["reluDerivativeAtZero"] == 0.0


def test_the_kink_probe_is_reproducible_in_the_browsers_arithmetic(payload):
    """Every ReLU gate must read the same summed either way.

    The browser starts a unit's sum from the bias and adds the weighted
    inputs in index order; torch sums however its gemv does. Those differ in
    the last bit, which is nothing at all except exactly on a kink -- where
    it is the difference between phi'(z) being 0 and being 1, and so a
    sensitivity that differs in the first digit. The probe exists to pin
    that convention, so it is only a fixture if both arithmetics agree about
    every unit, not just the one it was solved for.
    """
    probe = next(p for p in payload["goldenProbes"] if p["id"] == "relu-kink")
    x = probe["x"]
    activations = list(x)
    for layer, stored in zip(payload["parameters"], probe["pre"][1:]):
        ours = []
        for row, bias in zip(layer["W"], layer["b"]):
            total = bias
            for w, a in zip(row, activations):
                total += w * a
            ours.append(total)
        for mine, theirs in zip(ours, stored):
            assert (mine > 0) == (theirs > 0)
            assert (mine == 0.0) == (theirs == 0.0)
        activations = [max(0.0, v) for v in ours]


def test_the_inactive_probe_actually_has_inactive_units(payload):
    probe = next(p for p in payload["goldenProbes"] if p["id"] == "most-inactive")
    assert any(a == 0.0 for a in probe["act"][1])


def test_an_unlabeled_probe_gets_no_loss_and_no_gradient():
    """A point with no defined target must not be given an invented one."""
    ds = datasets.tilted_plane()
    result = train(ds, TrainConfig(sizes=[2, 4, 1], steps=10))
    probe = export._probe(
        result.model, ds, np.array([0.1, 0.2]), None, "unlabeled", "no target"
    )
    assert probe["target"] is None
    assert probe["pointLoss"] is None
    assert probe["pointGradient"] is None
    assert probe["inputSensitivity"] is not None  # sensitivity is still defined


def test_content_hash_covers_the_content(payload):
    assert export.content_hash(payload) == payload["contentHash"]
    tampered = json.loads(json.dumps(payload))
    tampered["parameters"][0]["b"][0] += 1.0
    assert export.content_hash(tampered) != tampered["contentHash"]


def test_non_finite_values_fail_the_export(payload):
    broken = json.loads(json.dumps(payload))
    broken["parameters"][0]["b"][0] = float("nan")
    with pytest.raises(FloatingPointError):
        export._assert_finite(broken)


def test_the_target_range_is_carried_to_the_browser(payload):
    """The colour scale's ends travel with the asset, not with the draw.

    The browser cannot evaluate the generator over the domain -- it has the
    formula but not the search -- so the range it paints has to arrive in
    the checkpoint and in the manifest entry the controls read.
    """
    declared = datasets.tilted_plane().target_range
    assert payload["dataset"]["targetRange"] == list(declared)
    manifest = export.build_manifest([payload])
    assert manifest["datasets"][0]["targetRange"] == list(declared)
    # The sampled extreme falls short of it, which is the whole reason the
    # range is declared rather than measured.
    assert max(abs(y) for y in payload["dataset"]["samples"]["y"]) < max(
        abs(end) for end in declared
    )


def test_manifest_lists_every_checkpoint(payload):
    manifest = export.build_manifest([payload])
    assert [c["checkpointId"] for c in manifest["checkpoints"]] == [
        payload["checkpointId"]
    ]
    assert manifest["checkpoints"][0]["url"].endswith(".json")
