"""Offline training of one catalog checkpoint.

`TrainConfig` is the single source of truth for a run and is serialized
verbatim into the exported metadata: adding a knob means adding a field here
and a flag in `cli.py`, so that every published checkpoint records exactly
what produced it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from .datasets import Dataset
from .model import DenseNet, mean_loss


@dataclass(frozen=True)
class TrainConfig:
    sizes: list[int]
    activation: str = "relu"
    steps: int = 4000
    learning_rate: float = 0.05
    optimizer: str = "adam"
    init_seed: int = 0
    # Full-batch gradient descent over the training split. The datasets are
    # a few hundred points and the networks have tens of parameters, so
    # minibatching would add a knob and a noise source for no benefit.

    def as_metadata(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainResult:
    model: DenseNet
    config: TrainConfig
    train_loss: float
    test_loss: float
    losses: list[float]


def train(dataset: Dataset, config: TrainConfig) -> TrainResult:
    """Train one model to convergence and report both split losses."""
    if config.sizes[0] != dataset.x.shape[1]:
        raise ValueError(
            f"architecture input {config.sizes[0]} does not match dataset "
            f"width {dataset.x.shape[1]}"
        )
    x_train, y_train = (torch.as_tensor(a) for a in dataset.train)
    x_test, y_test = (torch.as_tensor(a) for a in dataset.test)

    model = DenseNet(config.sizes, config.activation, seed=config.init_seed)
    if config.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    else:
        raise KeyError(f"unknown optimizer {config.optimizer!r}")

    losses: list[float] = []
    for _ in range(config.steps):
        opt.zero_grad(set_to_none=True)
        loss = mean_loss(model(x_train), y_train, dataset.task)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    with torch.no_grad():
        train_loss = float(mean_loss(model(x_train), y_train, dataset.task))
        test_loss = float(mean_loss(model(x_test), y_test, dataset.task))
    if not np.isfinite([train_loss, test_loss]).all():
        raise FloatingPointError(
            f"training diverged: train={train_loss}, test={test_loss}"
        )
    return TrainResult(
        model=model,
        config=config,
        train_loss=train_loss,
        test_loss=test_loss,
        losses=losses,
    )


@dataclass(frozen=True)
class UnitDiagnostics:
    """What one hidden unit actually contributes, over the whole domain."""

    unit: int
    # |output weight| relative to the largest in its layer.
    relative_output_weight: float
    # Fraction of a domain grid on which the unit is active (a > 0).
    active_fraction: float


def diagnose_units(model: DenseNet, dataset: Dataset, resolution: int = 41) -> list[UnitDiagnostics]:
    """Measure each first hidden layer unit's reach across the input domain."""
    (lo1, hi1), (lo2, hi2) = dataset.domain
    grid = np.stack(
        np.meshgrid(
            np.linspace(lo1, hi1, resolution), np.linspace(lo2, hi2, resolution)
        ),
        axis=-1,
    ).reshape(-1, 2)
    with torch.no_grad():
        hidden = model.trace(torch.as_tensor(grid)).act[1].numpy()
        downstream = model.layers[1].weight.detach().numpy()
    magnitude = np.abs(downstream).max(axis=0)
    largest = magnitude.max() if magnitude.max() > 0 else 1.0
    return [
        UnitDiagnostics(
            unit=unit,
            relative_output_weight=float(magnitude[unit] / largest),
            active_fraction=float((hidden[:, unit] > 0).mean()),
        )
        for unit in range(hidden.shape[1])
    ]


# A checkpoint is only useful for teaching if its hidden units are visible in
# the figure. These thresholds reject the degenerate fits that plain training
# reaches most often: a network that routes everything through one unit and
# leaves the rest with ~1e-6 output weights fits the tilted plane perfectly
# and demonstrates nothing (PLAN.md section 8 on curating examples).
SELECTION_CRITERIA = {
    "maxTrainLoss": 1e-4,
    # No vestigial units: every unit carries real weight downstream.
    "minRelativeOutputWeight": 0.05,
    # No permanently dead units.
    "minActiveFraction": 0.02,
    # At least one unit whose ReLU kink crosses the visible domain, so that
    # "a ReLU can block a downstream change" is demonstrable at all.
    "minUnitsWithKinkInDomain": 1,
    "kinkActiveFractionRange": [0.05, 0.95],
}


def _selection_report(result: TrainResult, dataset: Dataset) -> dict:
    units = diagnose_units(result.model, dataset)
    lo, hi = SELECTION_CRITERIA["kinkActiveFractionRange"]
    kinked = [u for u in units if lo <= u.active_fraction <= hi]
    checks = {
        "fits": result.train_loss <= SELECTION_CRITERIA["maxTrainLoss"],
        "noVestigialUnits": all(
            u.relative_output_weight >= SELECTION_CRITERIA["minRelativeOutputWeight"]
            for u in units
        ),
        "noDeadUnits": all(
            u.active_fraction >= SELECTION_CRITERIA["minActiveFraction"] for u in units
        ),
        "hasKinkInDomain": len(kinked)
        >= SELECTION_CRITERIA["minUnitsWithKinkInDomain"],
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "unitsWithKinkInDomain": [u.unit for u in kinked],
        "units": [asdict(u) for u in units],
    }


def curate(
    dataset: Dataset, config: TrainConfig, attempts: int = 24
) -> tuple[TrainResult, dict]:
    """Train repeatedly and keep the best checkpoint that is worth showing.

    Training alone does not produce a useful teaching asset: it produces a
    model that fits. Selection is a separate, recorded step, and the report
    it returns is exported with the checkpoint so the choice is auditable
    rather than implicit.
    """
    candidates: list[tuple[TrainResult, dict]] = []
    rejected = 0
    for attempt in range(attempts):
        trial = TrainConfig(**{**asdict(config), "init_seed": config.init_seed + attempt})
        try:
            result = train(dataset, trial)
        except FloatingPointError:
            rejected += 1
            continue
        report = _selection_report(result, dataset)
        if report["accepted"]:
            candidates.append((result, report))
        else:
            rejected += 1

    if not candidates:
        raise RuntimeError(
            f"no checkpoint met the selection criteria in {attempts} attempts; "
            f"criteria: {SELECTION_CRITERIA}"
        )
    # Among the acceptable ones, prefer the best held-out loss.
    result, report = min(candidates, key=lambda pair: pair[0].test_loss)
    report = {
        **report,
        "criteria": SELECTION_CRITERIA,
        "attempts": attempts,
        "accepted_count": len(candidates),
        "rejected_count": rejected,
        "chosen_init_seed": result.config.init_seed,
    }
    return result, report
