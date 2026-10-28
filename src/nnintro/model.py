"""The dense network, its activations, and its analytic derivatives.

Canonical parameter storage is PyTorch's: layer `l` holds `W` with shape
[out, in] and `b` with shape [out]. The browser stores the same orientation,
and the SVG adapter is the only place that translates to the renderer's
source-first edge indexing (PLAN.md section 7).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

# Hidden activations offered by the catalog. Identity is the linear baseline
# that makes "depth without nonlinearity is still affine" demonstrable.
ACTIVATIONS = ("relu", "tanh", "identity")

# Output transforms, keyed by task.
OUTPUT_TRANSFORMS = {"regression": "identity", "classification": "sigmoid"}
LOSSES = {"regression": "squared-error", "classification": "bce-with-logits"}


def activation_fn(name: str):
    if name == "relu":
        return torch.relu
    if name == "tanh":
        return torch.tanh
    if name == "identity":
        return lambda z: z
    raise KeyError(f"unknown activation {name!r}; have {list(ACTIVATIONS)}")


@dataclass(frozen=True)
class Forward:
    """Every intermediate state of one forward pass, in layer order.

    `pre[l]` is the preactivation z of layer l and `act[l]` its activation a.
    The output layer's activation is its preactivation: the output transform
    (sigmoid for classification) is applied when a probability is displayed,
    never inside the layer, so the logit stays available.
    """

    pre: list[torch.Tensor]
    act: list[torch.Tensor]

    @property
    def output(self) -> torch.Tensor:
        """The network's scalar output: a regression prediction or a logit."""
        return self.pre[-1][..., 0]


class DenseNet(nn.Module):
    """A fully connected `sizes[0] -> ... -> sizes[-1]` network.

    CPU float64 throughout. The browser recomputes everything in JavaScript
    doubles, and the golden fixtures are only a meaningful check if both
    sides use the same precision — float32 training would put a 1e-7 floor
    under a comparison that should hold to 1e-12.
    """

    def __init__(
        self,
        sizes: list[int],
        activation: str = "relu",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if len(sizes) < 2 or any(s < 1 for s in sizes):
            raise ValueError(f"need at least an input and an output layer: {sizes}")
        if activation not in ACTIVATIONS:
            raise KeyError(f"unknown activation {activation!r}")
        self.sizes = list(sizes)
        self.activation = activation
        self.phi = activation_fn(activation)
        # Seeded in a forked RNG so constructing a model never perturbs the
        # caller's global stream (the convention in ../ntk and ../infoplane).
        # nn.Linear runs its own default init on construction, so the layers
        # have to be built inside the fork as well, not just re-initialized
        # inside it.
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.manual_seed(seed)
            self.layers = nn.ModuleList(
                nn.Linear(a, b, dtype=torch.float64)
                for a, b in zip(sizes[:-1], sizes[1:])
            )
            self._init_parameters()

    def _init_parameters(self) -> None:
        """Activation-aware initialization; biases start at zero.

        He scaling for ReLU, Xavier for tanh and identity. The browser's
        "randomize" uses the same rule with randomized biases, but records
        the resulting parameters rather than only the seed.
        """
        for index, layer in enumerate(self.layers):
            fan_in = layer.in_features
            is_output = index == len(self.layers) - 1
            if is_output or self.activation != "relu":
                std = (1.0 / fan_in) ** 0.5
            else:
                std = (2.0 / fan_in) ** 0.5
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            nn.init.zeros_(layer.bias)

    def trace(self, x: torch.Tensor) -> Forward:
        """Forward pass retaining every preactivation and activation."""
        pre: list[torch.Tensor] = [x]
        act: list[torch.Tensor] = [x]
        last = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            z = layer(act[-1])
            pre.append(z)
            act.append(z if index == last else self.phi(z))
        return Forward(pre=pre, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trace(x).output


def point_loss(output: torch.Tensor, target: torch.Tensor, task: str) -> torch.Tensor:
    """Per-example loss. Regression: (yhat - y)^2. Classification: BCE.

    No factor of one half anywhere, and the dataset loss is the arithmetic
    mean of this over the chosen split -- stated once here so that nothing
    downstream has to guess the convention (PLAN.md section 8).
    """
    if task == "regression":
        return (output - target) ** 2
    if task == "classification":
        # From logits: numerically stable at extreme values, unlike
        # sigmoid-then-log.
        return torch.nn.functional.binary_cross_entropy_with_logits(
            output, target, reduction="none"
        )
    raise KeyError(f"unknown task {task!r}")


def mean_loss(output: torch.Tensor, target: torch.Tensor, task: str) -> torch.Tensor:
    return point_loss(output, target, task).mean()
