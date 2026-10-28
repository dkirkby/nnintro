"""Dataset generators for the explorer catalog.

Every dataset has two input features on a common domain of [-1, 1]^2 and one
scalar target, so that every architecture in the catalog can consume every
dataset and input-space plots are directly interpretable (PLAN.md section 3).

Samples are fixed and seeded: a dataset's identity includes its generator
version, so changing the recipe means a new catalog entry, not a silent
change to an existing one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GENERATOR_VERSION = 1

# The fraction of samples assigned to the held-out split.
TEST_FRACTION = 0.2


@dataclass(frozen=True)
class Dataset:
    """A fixed, seeded sample of a generator, with its split membership."""

    id: str
    task: str  # "regression" | "classification"
    x: np.ndarray  # (n, 2) float64, inside `domain`
    y: np.ndarray  # (n,) float64 targets
    is_test: np.ndarray  # (n,) bool, True for the held-out split
    domain: tuple[tuple[float, float], tuple[float, float]]
    input_names: tuple[str, str]
    target_name: str
    description: str
    # True when the generator defines a target at every point of the domain,
    # so a freely dragged probe can be given a target and a point loss.
    has_analytic_target: bool
    generator_version: int = GENERATOR_VERSION

    @property
    def train(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x[~self.is_test], self.y[~self.is_test]

    @property
    def test(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x[self.is_test], self.y[self.is_test]


def _split(n: int, rng: np.random.Generator) -> np.ndarray:
    """Assign a fixed held-out split: a seeded permutation, not the head."""
    n_test = int(round(n * TEST_FRACTION))
    is_test = np.zeros(n, dtype=bool)
    is_test[rng.permutation(n)[:n_test]] = True
    return is_test


def tilted_plane(num_samples: int = 400, seed: int = 20261027) -> Dataset:
    """y = 0.6 x1 - 0.4 x2 + 0.2, sampled uniformly on [-1, 1]^2.

    The teaching purpose is that every weight and the bias have an
    immediately interpretable role: a trained model's parameters can be read
    straight off against the generating coefficients.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(num_samples, 2))
    y = 0.6 * x[:, 0] - 0.4 * x[:, 1] + 0.2
    return Dataset(
        id="tilted-plane",
        task="regression",
        x=x,
        y=y,
        is_test=_split(num_samples, rng),
        domain=((-1.0, 1.0), (-1.0, 1.0)),
        input_names=("x1", "x2"),
        target_name="y",
        description="y = 0.6 x1 - 0.4 x2 + 0.2",
        has_analytic_target=True,
    )


def analytic_target(dataset_id: str, x: np.ndarray) -> np.ndarray:
    """The generator's target at arbitrary points, for free probes."""
    if dataset_id == "tilted-plane":
        return 0.6 * x[..., 0] - 0.4 * x[..., 1] + 0.2
    raise KeyError(f"no analytic target for dataset {dataset_id!r}")


BUILDERS = {"tilted-plane": tilted_plane}


def build(dataset_id: str) -> Dataset:
    try:
        builder = BUILDERS[dataset_id]
    except KeyError:
        raise KeyError(
            f"unknown dataset {dataset_id!r}; have {sorted(BUILDERS)}"
        ) from None
    return builder()
