"""The browser asset contract: checkpoints, golden probes, and the manifest.

A checkpoint carries everything the explorer needs to reproduce a model
exactly, plus *golden probes*: fully worked forward states and derivatives
computed here by PyTorch autograd. The browser recomputes all of it in
JavaScript and the frontend tests assert agreement (PLAN.md section 12).

Golden gradients are validation fixtures, never a runtime substitute: the
whole point of the explorer is that a parameter edit invalidates them, so
the browser always recomputes after an edit.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from . import SCHEMA_VERSION
from .datasets import Dataset, analytic_target
from .model import LOSSES, OUTPUT_TRANSFORMS, DenseNet, mean_loss, point_loss
from .train import TrainResult


def checkpoint_id(dataset_id: str, sizes: list[int], activation: str) -> str:
    """A stable identifier naming every dimension that selects a checkpoint."""
    arch = "-".join(str(s) for s in sizes)
    # A network with no hidden layer has no hidden activation to name.
    suffix = activation if len(sizes) > 2 else "none"
    return f"{dataset_id}__{arch}__{suffix}"


def _parameters(model: DenseNet) -> list[dict]:
    """Layer parameters in canonical [out, in] orientation."""
    return [
        {
            "W": layer.weight.detach().tolist(),  # [out][in]
            "b": layer.bias.detach().tolist(),  # [out]
        }
        for layer in model.layers
    ]


def _grad_to_lists(grads: list[tuple[torch.Tensor, torch.Tensor]]) -> list[dict]:
    return [{"W": w.tolist(), "b": b.tolist()} for w, b in grads]


def _parameter_gradient(
    model: DenseNet, loss: torch.Tensor
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    params = [p for layer in model.layers for p in (layer.weight, layer.bias)]
    grads = torch.autograd.grad(loss, params, retain_graph=True)
    return [(grads[i], grads[i + 1]) for i in range(0, len(grads), 2)]


def _input_sensitivity(model: DenseNet, x: torch.Tensor) -> torch.Tensor:
    """d(output)/dx at a single point, with the output being the raw scalar.

    For a classification model this is the logit sensitivity; the browser
    converts to probability sensitivity and labels which one it shows.
    """
    x = x.detach().clone().requires_grad_(True)
    output = model(x)
    (grad,) = torch.autograd.grad(output, x)
    return grad


def _kink_point(model: DenseNet, domain) -> np.ndarray | None:
    """A point where one first-layer preactivation is exactly zero.

    ReLU has no derivative there. The explorer discloses its convention
    (zero at z = 0) rather than hiding the discontinuity, so the fixture set
    pins that convention on a point that actually sits on the kink -- found
    exactly rather than by search, by solving w . x + b = 0 along x2.
    """
    if model.activation != "relu":
        return None
    weight = model.layers[0].weight.detach().numpy()
    bias = model.layers[0].bias.detach().numpy()
    (lo1, hi1), (lo2, hi2) = domain
    for unit in range(weight.shape[0]):
        w1, w2 = weight[unit]
        if w2 == 0:
            continue
        for x1 in np.linspace(lo1, hi1, 21):
            x2 = -(w1 * x1 + bias[unit]) / w2
            if lo2 <= x2 <= hi2:
                point = np.array([x1, x2], dtype=np.float64)
                # Confirm it is exactly zero in the arithmetic that matters.
                z = float(weight[unit] @ point + bias[unit])
                if z == 0.0:
                    return point
    return None


def _probe(
    model: DenseNet,
    dataset: Dataset,
    x: np.ndarray,
    target: float | None,
    probe_id: str,
    label: str,
) -> dict:
    """One fully worked example: state, loss, and both kinds of derivative."""
    xt = torch.as_tensor(x, dtype=torch.float64)
    trace = model.trace(xt)
    output = trace.output

    record: dict = {
        "id": probe_id,
        "label": label,
        "x": xt.tolist(),
        # Index 0 is the input layer, whose preactivation and activation are
        # both the input itself.
        "pre": [t.tolist() for t in trace.pre],
        "act": [t.tolist() for t in trace.act],
        "output": float(output.detach()),
        "inputSensitivity": _input_sensitivity(model, xt).tolist(),
    }
    if dataset.task == "classification":
        record["probability"] = float(torch.sigmoid(output.detach()))

    if target is None:
        # A free probe in a dataset with no analytic target has no loss and
        # no loss gradient; it is not given an invented label.
        record["target"] = None
        record["pointLoss"] = None
        record["pointGradient"] = None
    else:
        yt = torch.as_tensor(target, dtype=torch.float64)
        loss = point_loss(output, yt, dataset.task)
        record["target"] = float(yt)
        # The gradient is taken first: `loss` must keep its graph until
        # autograd has used it.
        record["pointGradient"] = _grad_to_lists(_parameter_gradient(model, loss))
        record["pointLoss"] = float(loss.detach())
    return record


def _golden_probes(model: DenseNet, dataset: Dataset) -> list[dict]:
    """A fixture set chosen to exercise the cases that break implementations."""
    probes: list[dict] = []
    train_index = int(np.flatnonzero(~dataset.is_test)[0])
    test_index = int(np.flatnonzero(dataset.is_test)[0])
    probes.append(
        _probe(
            model,
            dataset,
            dataset.x[train_index],
            float(dataset.y[train_index]),
            "train-sample",
            f"training sample {train_index}",
        )
    )
    probes.append(
        _probe(
            model,
            dataset,
            dataset.x[test_index],
            float(dataset.y[test_index]),
            "test-sample",
            f"held-out sample {test_index}",
        )
    )
    # The origin: all preactivations reduce to the biases, which makes a
    # transposed weight matrix invisible here and obvious everywhere else.
    origin = np.zeros(2, dtype=np.float64)
    probes.append(
        _probe(
            model,
            dataset,
            origin,
            float(analytic_target(dataset.id, origin))
            if dataset.has_analytic_target
            else None,
            "origin",
            "the origin, where every preactivation is its bias",
        )
    )
    # An asymmetric free probe: the case that catches a transposed W.
    free = np.array([0.73, -0.41], dtype=np.float64)
    probes.append(
        _probe(
            model,
            dataset,
            free,
            float(analytic_target(dataset.id, free))
            if dataset.has_analytic_target
            else None,
            "free-probe",
            "an asymmetric free probe",
        )
    )
    # A point with the most inactive ReLU units, where a downstream ripple
    # is blocked rather than propagated.
    if model.activation == "relu":
        grid = np.stack(
            np.meshgrid(np.linspace(-1, 1, 41), np.linspace(-1, 1, 41)), axis=-1
        ).reshape(-1, 2)
        with torch.no_grad():
            hidden = model.trace(torch.as_tensor(grid)).act[1]
        dead = int(np.argmax((hidden.numpy() == 0.0).sum(axis=1)))
        point = grid[dead].astype(np.float64)
        probes.append(
            _probe(
                model,
                dataset,
                point,
                float(analytic_target(dataset.id, point))
                if dataset.has_analytic_target
                else None,
                "most-inactive",
                "the grid point with the most inactive hidden units",
            )
        )
        kink = _kink_point(model, dataset.domain)
        if kink is not None:
            probes.append(
                _probe(
                    model,
                    dataset,
                    kink,
                    float(analytic_target(dataset.id, kink))
                    if dataset.has_analytic_target
                    else None,
                    "relu-kink",
                    "exactly on a ReLU kink, where phi'(0) = 0 by convention",
                )
            )
    return probes


def _split_gradients(model: DenseNet, dataset: Dataset) -> dict:
    """Mean-loss and its gradient, per split.

    The gradient OF THE MEAN LOSS, not a mean of per-example gradient
    magnitudes: those are different quantities and only this one is the
    slope the explorer claims it is.
    """
    out: dict = {}
    for name, (x, y) in (("train", dataset.train), ("test", dataset.test)):
        xt = torch.as_tensor(x)
        yt = torch.as_tensor(y)
        loss = mean_loss(model(xt), yt, dataset.task)
        out[name] = {
            "count": int(len(y)),
            "meanLoss": float(loss.detach()),
            "gradient": _grad_to_lists(_parameter_gradient(model, loss)),
        }
    return out


def content_hash(payload: dict) -> str:
    """SHA-256 over the canonical JSON, excluding the hash field itself."""
    body = {k: v for k, v in payload.items() if k != "contentHash"}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_checkpoint(
    result: TrainResult, dataset: Dataset, selection: dict | None = None
) -> dict:
    model = result.model
    config = result.config
    payload: dict = {
        "schemaVersion": SCHEMA_VERSION,
        "checkpointId": checkpoint_id(dataset.id, config.sizes, config.activation),
        "dataset": {
            "id": dataset.id,
            "label": dataset.label,
            "task": dataset.task,
            "description": dataset.description,
            "generatorVersion": dataset.generator_version,
            "domain": [list(d) for d in dataset.domain],
            "inputNames": list(dataset.input_names),
            "targetName": dataset.target_name,
            "hasAnalyticTarget": dataset.has_analytic_target,
            # No preprocessing: the generators already emit the common
            # domain, so reported derivatives are in original input units
            # and need no chain-rule correction.
            "preprocessing": None,
            "samples": {
                "x": dataset.x.tolist(),
                "y": dataset.y.tolist(),
                "isTest": dataset.is_test.tolist(),
            },
        },
        "architecture": {
            "layerSizes": list(config.sizes),
            "hiddenActivation": config.activation if len(config.sizes) > 2 else None,
            "outputTransform": OUTPUT_TRANSFORMS[dataset.task],
            "loss": LOSSES[dataset.task],
            # Spelled out so no reader has to infer it from the arrays.
            "weightLayout": "W[out][in], b[out]; row-major, destination-first",
            "reluDerivativeAtZero": 0.0 if config.activation == "relu" else None,
        },
        "parameters": _parameters(model),
        "training": {
            **config.as_metadata(),
            "torchVersion": torch.__version__,
            "dtype": "float64",
            "finalTrainLoss": result.train_loss,
            "finalTestLoss": result.test_loss,
        },
        "metrics": {"trainLoss": result.train_loss, "testLoss": result.test_loss},
        # How this checkpoint was chosen among training attempts, and what
        # each hidden unit actually contributes. Recorded so that a curated
        # asset can be audited rather than taken on trust.
        "selection": selection,
        "splitGradients": _split_gradients(model, dataset),
        "goldenProbes": _golden_probes(model, dataset),
    }
    _assert_finite(payload)
    payload["contentHash"] = content_hash(payload)
    return payload


def _assert_finite(payload: dict) -> None:
    """Every exported number must be finite; a NaN is a failed export."""

    def walk(node, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")
        elif isinstance(node, float) and not np.isfinite(node):
            raise FloatingPointError(f"non-finite value at {path}: {node}")

    walk(payload, "$")


def write_checkpoint(payload: dict, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{payload['checkpointId']}.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return path


def build_manifest(payloads: list[dict]) -> dict:
    """Lists the configurations the explorer may offer, and where each lives.

    A control must not be able to reach a configuration that is missing
    here: the browser enumerates this, rather than assuming coverage.
    """
    # One entry per dataset, so the UI can label and group its dataset
    # control without opening 19 checkpoints to find out what they share.
    seen: dict[str, dict] = {}
    for p in payloads:
        spec = p["dataset"]
        seen.setdefault(
            spec["id"],
            {
                "id": spec["id"],
                "label": spec["label"],
                "task": spec["task"],
                "description": spec["description"],
                "generatorVersion": spec["generatorVersion"],
                "hasAnalyticTarget": spec["hasAnalyticTarget"],
                "sampleCount": len(spec["samples"]["y"]),
            },
        )
    return {
        "schemaVersion": SCHEMA_VERSION,
        "datasets": list(seen.values()),
        "checkpoints": [
            {
                "checkpointId": p["checkpointId"],
                "dataset": p["dataset"]["id"],
                "task": p["dataset"]["task"],
                "layerSizes": p["architecture"]["layerSizes"],
                "hiddenActivation": p["architecture"]["hiddenActivation"],
                "url": f"{p['checkpointId']}.json",
                "contentHash": p["contentHash"],
                "metrics": p["metrics"],
            }
            for p in payloads
        ],
    }


def publish(src: Path, dest: Path) -> list[Path]:
    """Copy the working output to the tracked snapshot the site builds from.

    This is the only thing that should write `dest`: hand-editing it makes
    the published site disagree with the pipeline that produced it.
    """
    if not src.is_dir():
        raise FileNotFoundError(f"nothing to publish: {src} does not exist")
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for path in sorted(src.glob("*.json")):
        target = dest / path.name
        shutil.copyfile(path, target)
        written.append(target)
    return written
