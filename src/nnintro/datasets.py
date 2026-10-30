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
    # The short name a control wears, as opposed to `description`, which is
    # the formula or recipe. The pipeline owns both: the UI should not be
    # title-casing identifiers to guess at either.
    label: str
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
        label="Tilted plane",
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


def _blobs(
    centers: np.ndarray,
    spread: float,
    counts: np.ndarray,
    domain,
    rng: np.random.Generator,
) -> np.ndarray:
    """Isotropic Gaussian blobs, truncated to the domain by rejection.

    Rejection rather than clipping: clipping would pile the tail of every
    blob onto the frame of the input map, which reads as a feature of the
    data and is an artefact of the plot's edge.
    """
    (lo1, hi1), (lo2, hi2) = domain
    out = []
    for center, count in zip(centers, counts):
        kept = np.empty((0, 2), dtype=np.float64)
        while len(kept) < count:
            draw = rng.normal(center, spread, size=(2 * int(count) + 16, 2))
            inside = (
                (draw[:, 0] >= lo1)
                & (draw[:, 0] <= hi1)
                & (draw[:, 1] >= lo2)
                & (draw[:, 1] <= hi2)
            )
            kept = np.concatenate([kept, draw[inside]])
        out.append(kept[:count])
    return np.concatenate(out)


def curved_surface(num_samples: int = 400, seed: int = 20261028) -> Dataset:
    """y = sin(pi x1) cos(pi x2), sampled uniformly on [-1, 1]^2.

    The teaching purpose is approximation: no affine map comes close, so the
    hidden layer's job is visible in the gap between the surface and the
    target. The generator is noiseless, so train and held-out losses agree
    by construction and the only error is representational.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(num_samples, 2))
    y = np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
    return Dataset(
        id="curved-surface",
        label="Curved surface",
        task="regression",
        x=x,
        y=y,
        is_test=_split(num_samples, rng),
        domain=((-1.0, 1.0), (-1.0, 1.0)),
        input_names=("x1", "x2"),
        target_name="y",
        description="y = sin(pi x1) cos(pi x2)",
        has_analytic_target=True,
    )


def gaussian_clusters(num_samples: int = 400, seed: int = 20261029) -> Dataset:
    """Two Gaussian clusters, one per class, placed along the diagonal.

    The teaching purpose is a linear decision boundary and a probability:
    one weight vector and one bias are enough, and the output is read as
    p rather than as a value. The clusters overlap, so some points are
    unavoidably misclassified -- the irreducible error that makes a
    train/held-out gap mean something here.
    """
    rng = np.random.default_rng(seed)
    domain = ((-1.0, 1.0), (-1.0, 1.0))
    half = num_samples // 2
    counts = np.array([num_samples - half, half])
    centers = np.array([[-0.45, -0.45], [0.45, 0.45]])
    x = _blobs(centers, 0.38, counts, domain, rng)
    y = np.concatenate([np.zeros(counts[0]), np.ones(counts[1])])
    order = rng.permutation(num_samples)
    x, y = x[order], y[order]
    return Dataset(
        id="gaussian-clusters",
        label="Gaussian clusters",
        task="classification",
        x=x,
        y=y,
        is_test=_split(num_samples, rng),
        domain=domain,
        input_names=("x1", "x2"),
        target_name="class",
        description="two overlapping Gaussian clusters, one per class",
        has_analytic_target=False,
    )


def xor_quadrants(num_samples: int = 400, seed: int = 20261030) -> Dataset:
    """Four Gaussian blobs, one per quadrant, labelled by XOR of the signs.

    The teaching purpose is that a problem can need a hidden layer at all:
    no single line separates these classes, so the 2 -> 1 and identity
    variants fail on it by construction. That imperfect fit is the asset,
    not a defect (PLAN.md section 3).
    """
    rng = np.random.default_rng(seed)
    domain = ((-1.0, 1.0), (-1.0, 1.0))
    centers = np.array([[0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]])
    # Class 1 where the signs agree, which is the XOR of "x is negative".
    labels = np.array([1.0, 1.0, 0.0, 0.0])
    base, extra = divmod(num_samples, 4)
    counts = np.full(4, base)
    counts[:extra] += 1
    x = _blobs(centers, 0.30, counts, domain, rng)
    y = np.repeat(labels, counts)
    order = rng.permutation(num_samples)
    x, y = x[order], y[order]
    return Dataset(
        id="xor-quadrants",
        label="XOR quadrants",
        task="classification",
        x=x,
        y=y,
        is_test=_split(num_samples, rng),
        domain=domain,
        input_names=("x1", "x2"),
        target_name="class",
        description="XOR quadrants: class 1 where the two inputs share a sign",
        has_analytic_target=False,
    )


def analytic_target(dataset_id: str, x: np.ndarray) -> np.ndarray:
    """The generator's target at arbitrary points, for free probes."""
    if dataset_id == "tilted-plane":
        return 0.6 * x[..., 0] - 0.4 * x[..., 1] + 0.2
    if dataset_id == "curved-surface":
        return np.sin(np.pi * x[..., 0]) * np.cos(np.pi * x[..., 1])
    # The classification datasets are stochastic in x given the class, so an
    # arbitrary point has no label to report (PLAN.md section 3).
    raise KeyError(f"no analytic target for dataset {dataset_id!r}")


# Declaration order is the order the UI offers them: the two regression
# datasets first, simplest first, then the two classifiers.
BUILDERS = {
    "tilted-plane": tilted_plane,
    "curved-surface": curved_surface,
    "gaussian-clusters": gaussian_clusters,
    "xor-quadrants": xor_quadrants,
}


def build(dataset_id: str) -> Dataset:
    try:
        builder = BUILDERS[dataset_id]
    except KeyError:
        raise KeyError(
            f"unknown dataset {dataset_id!r}; have {sorted(BUILDERS)}"
        ) from None
    return builder()
