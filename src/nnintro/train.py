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
    """What one hidden unit actually contributes, over the whole domain.

    Three measurements, each the generalization of a ReLU-specific question
    to the whole activation family:

    - `relative_output_weight` answers "does anything downstream read it",
      as a fraction of the largest such weight in its own layer.
    - `relative_variation` answers "does it carry any signal": the standard
      deviation of its activation over a domain grid, again relative to the
      largest in its layer. A dead ReLU, a saturated tanh and a unit whose
      incoming weights are ~0 all score ~0 here, which is why this replaced
      the old active-fraction test.
    - `positive_fraction` answers "where is its transition": the fraction of
      the grid on which its preactivation is positive, i.e. which side of
      the unit's own hyperplane the input falls. For ReLU that is exactly
      the active fraction, and a value strictly between 0 and 1 means the
      kink crosses the visible domain.
    """

    layer: int  # 1-based: the hidden layer this unit belongs to
    unit: int
    relative_output_weight: float
    relative_variation: float
    positive_fraction: float


def _domain_grid(dataset: Dataset, resolution: int) -> np.ndarray:
    (lo1, hi1), (lo2, hi2) = dataset.domain
    return np.stack(
        np.meshgrid(
            np.linspace(lo1, hi1, resolution), np.linspace(lo2, hi2, resolution)
        ),
        axis=-1,
    ).reshape(-1, 2)


def diagnose_units(
    model: DenseNet, dataset: Dataset, resolution: int = 41
) -> list[UnitDiagnostics]:
    """Measure every hidden unit's reach across the input domain.

    Every hidden layer, not just the first: a vestigial unit in the second
    hidden layer is just as invisible in the figure, and the catalog now has
    two-hidden-layer entries.
    """
    grid = _domain_grid(dataset, resolution)
    with torch.no_grad():
        trace = model.trace(torch.as_tensor(grid))
        weights = [layer.weight.detach().numpy() for layer in model.layers]

    out: list[UnitDiagnostics] = []
    # Layers 1..n-1 of the trace are the hidden ones; the last is the output.
    for layer in range(1, len(model.layers)):
        pre = trace.pre[layer].numpy()
        act = trace.act[layer].numpy()
        downstream = np.abs(weights[layer]).max(axis=0)
        variation = act.std(axis=0)
        largest_weight = downstream.max() or 1.0
        largest_variation = variation.max() or 1.0
        for unit in range(act.shape[1]):
            out.append(
                UnitDiagnostics(
                    layer=layer,
                    unit=unit,
                    relative_output_weight=float(downstream[unit] / largest_weight),
                    relative_variation=float(variation[unit] / largest_variation),
                    positive_fraction=float((pre[:, unit] > 0).mean()),
                )
            )
    return out


# A checkpoint is only useful for teaching if its hidden units are visible in
# the figure. These thresholds reject the degenerate fits that plain training
# reaches most often: a network that routes everything through one unit and
# leaves the rest with ~1e-6 output weights fits the tilted plane perfectly
# and demonstrates nothing (PLAN.md section 8 on curating examples).
SELECTION_CRITERIA = {
    # No vestigial units: every hidden unit carries real weight downstream.
    "minRelativeOutputWeight": 0.05,
    # No dead or saturated units: every hidden unit's activation varies.
    "minRelativeVariation": 0.05,
    # At least one hidden unit whose transition crosses the visible domain,
    # so that "a ReLU can block a downstream change" is demonstrable at all.
    # Identity has no transition to find, so the check does not apply to it.
    "minUnitsWithTransitionInDomain": 1,
    "transitionFractionRange": [0.05, 0.95],
}

def _unit_report(result: TrainResult, dataset: Dataset) -> dict:
    """Judge one attempt on whether its figure would be worth looking at."""
    units = diagnose_units(result.model, dataset)
    lo, hi = SELECTION_CRITERIA["transitionFractionRange"]
    crossing = [u for u in units if lo <= u.positive_fraction <= hi]
    has_hidden = len(result.config.sizes) > 2
    checks = {
        "noVestigialUnits": all(
            u.relative_output_weight >= SELECTION_CRITERIA["minRelativeOutputWeight"]
            for u in units
        ),
        "noDeadUnits": all(
            u.relative_variation >= SELECTION_CRITERIA["minRelativeVariation"]
            for u in units
        ),
        # Vacuous without a hidden layer, and meaningless for identity, whose
        # units have no transition anywhere.
        "hasTransitionInDomain": (
            not has_hidden
            or result.config.activation == "identity"
            or len(crossing) >= SELECTION_CRITERIA["minUnitsWithTransitionInDomain"]
        ),
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "unitsWithTransitionInDomain": [
            {"layer": u.layer, "unit": u.unit} for u in crossing
        ],
        "units": [asdict(u) for u in units],
    }


def curate(
    dataset: Dataset, config: TrainConfig, attempts: int = 256
) -> tuple[TrainResult, dict]:
    """Train repeatedly and keep the best checkpoint that is worth showing.

    Training alone does not produce a useful teaching asset: it produces a
    model that fits. Selection is a separate, recorded step, and the report
    it returns is exported with the checkpoint so the choice is auditable
    rather than implicit.

    The criteria are about *legibility*, not accuracy: they reject fits
    whose figure would show nothing, and among the ones that survive the
    best training loss wins. Accuracy cannot be a criterion across this
    catalog, because the configurations that cannot fit their dataset -- a
    linear model on XOR, a 2-2-1 on the curved surface -- are intended
    teaching assets rather than failures (PLAN.md section 3).

    `attempts` has to be this large for the catalog to come out complete.
    The hardest entry is the tilted plane at 2-4-4-1 with ReLU: eight
    hidden units fitting a two-parameter plane leaves most of them with
    nothing to do, so only about one attempt in fifty survives
    `noDeadUnits` and `noVestigialUnits`, and its first acceptable seed is
    the 114th. 256 is not that bound plus a rounding -- it is the bound
    plus room for the acceptance pattern to shift under a different numpy
    or torch, since at 128 that entry would rest on two seeds. Lowering it
    leaves holes, which is a reportable failure rather than a silent one.

    The objective is the *training* loss, deliberately not the held-out
    loss. The explorer displays both, and choosing the attempt with the
    best held-out loss would bias that number below its training
    counterpart, turning the one quantity that is supposed to say something
    about generalization into an artefact of selection.
    """
    results: list[TrainResult] = []
    diverged = 0
    for attempt in range(attempts):
        trial = TrainConfig(**{**asdict(config), "init_seed": config.init_seed + attempt})
        try:
            results.append(train(dataset, trial))
        except FloatingPointError:
            diverged += 1

    if not results:
        raise RuntimeError(f"every one of {attempts} attempts diverged")

    judged = [(r, _unit_report(r, dataset)) for r in results]
    candidates = [pair for pair in judged if pair[1]["accepted"]]
    best_overall = min(r.train_loss for r in results)

    if not candidates:
        raise RuntimeError(
            f"no checkpoint met the selection criteria in {attempts} attempts; "
            f"best train loss {best_overall:.3e}; "
            f"rejected by: {_failure_tally(judged)}"
        )
    result, report = min(candidates, key=lambda pair: pair[0].train_loss)
    report = {
        **report,
        "criteria": SELECTION_CRITERIA,
        "attempts": attempts,
        # What legibility cost: the best training loss any attempt reached,
        # against the best among those worth showing.
        "bestTrainLossOverall": best_overall,
        "bestTrainLossAccepted": result.train_loss,
        "accepted_count": len(candidates),
        "rejected_count": len(results) - len(candidates),
        "diverged_count": diverged,
        "rejected_by": _failure_tally(judged),
        "chosen_init_seed": result.config.init_seed,
    }
    return result, report


def _failure_tally(judged: list[tuple[TrainResult, dict]]) -> dict:
    """How many attempts each check rejected, for the failure message."""
    tally: dict[str, int] = {}
    for _, report in judged:
        for name, passed in report["checks"].items():
            if not passed:
                tally[name] = tally.get(name, 0) + 1
    return tally
