import torch

from nnintro.model import DenseNet, mean_loss, point_loss


def test_trace_exposes_every_layer_state():
    model = DenseNet([2, 4, 1], "relu", seed=0)
    trace = model.trace(torch.zeros(2, dtype=torch.float64))
    # Index 0 is the input layer, then one entry per Linear.
    assert len(trace.pre) == 3
    assert [len(t) for t in trace.act] == [2, 4, 1]


def test_the_output_layer_is_not_squashed():
    """The last layer's activation is its preactivation: the logit stays raw."""
    model = DenseNet([2, 4, 1], "relu", seed=0)
    trace = model.trace(torch.tensor([0.3, -0.7], dtype=torch.float64))
    assert torch.equal(trace.act[-1], trace.pre[-1])


def test_weights_are_destination_first():
    model = DenseNet([2, 4, 1], "relu", seed=0)
    assert tuple(model.layers[0].weight.shape) == (4, 2)  # [out, in]
    assert tuple(model.layers[0].bias.shape) == (4,)


def test_construction_does_not_perturb_the_global_rng():
    torch.manual_seed(1234)
    before = torch.randn(3, dtype=torch.float64)
    torch.manual_seed(1234)
    DenseNet([2, 4, 1], "relu", seed=7)
    after = torch.randn(3, dtype=torch.float64)
    assert torch.equal(before, after)


def test_relu_derivative_at_zero_is_zero_by_convention():
    z = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    torch.relu(z).backward()
    assert float(z.grad) == 0.0


def test_an_identity_network_is_affine():
    """Depth without a nonlinearity collapses to a single affine map."""
    model = DenseNet([2, 4, 3, 1], "identity", seed=3)
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.6, -0.3]], dtype=torch.float64)
    y = model(x)
    predicted = y[0] + 0.6 * (y[1] - y[0]) + (-0.3) * (y[2] - y[0])
    assert torch.allclose(y[3], predicted, atol=1e-12)


def test_regression_point_loss_has_no_factor_of_one_half():
    loss = point_loss(
        torch.tensor(3.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
        "regression",
    )
    assert float(loss) == 4.0


def test_classification_loss_is_stable_at_extreme_logits():
    logit = torch.tensor(-800.0, dtype=torch.float64)
    target = torch.tensor(1.0, dtype=torch.float64)
    loss = point_loss(logit, target, "classification")
    assert torch.isfinite(loss) and float(loss) > 700


def test_mean_loss_is_the_arithmetic_mean_of_point_losses():
    out = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    y = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    assert float(mean_loss(out, y, "regression")) == (1 + 4 + 9) / 3
