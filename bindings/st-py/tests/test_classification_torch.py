"""Independent float64 PyTorch oracle; torch is never part of the implementation."""
import pytest

torch = pytest.importorskip("torch")
import spiraltorch as st

pytestmark = pytest.mark.skipif(getattr(st, "_rs", None) is None, reason="native extension required")


def native(values):
    rows, cols = values.shape
    return st.Tensor(rows, cols, values.detach().flatten().tolist())


def flat(value):
    return [item for row in value.tolist() for item in row]


@pytest.mark.parametrize("shape", [(1, 1), (7, 3), (4, 257)])
@pytest.mark.parametrize("smoothing", [0.0, 0.13, 1.0])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_cross_entropy_value_and_vjp_match_float64_pytorch(shape, smoothing, reduction):
    generator = torch.Generator().manual_seed(71)
    # Round inputs once to the native f32 transport, then use float64 for the oracle.
    values = torch.randn(shape, generator=generator, dtype=torch.float32).double().requires_grad_()
    labels = torch.randint(shape[1], (shape[0],), generator=generator)
    if shape[0] > 1:
        labels[1] = -100
    reference = torch.nn.functional.cross_entropy(values, labels, reduction=reduction, label_smoothing=smoothing)
    seed = torch.randn(reference.shape, generator=generator, dtype=torch.float32).double()
    reference.backward(seed)
    leaf = st.AutogradTensor.variable(native(values))
    output = leaf.cross_entropy_with_logits(labels.tolist(), reduction=reduction, label_smoothing=smoothing)
    assert flat(output.value()) == pytest.approx(reference.detach().flatten().tolist(), rel=3e-6, abs=1e-7)
    native_seed = native(seed.reshape(-1, 1))
    gradient = output.vector_jacobian_product(leaf, native_seed)
    assert flat(gradient) == pytest.approx(values.grad.flatten().tolist(), rel=3e-5, abs=1e-7)
    output.backward(native_seed)
    assert flat(leaf.grad()) == flat(gradient)


@pytest.mark.parametrize("scale", [1.0, 50.0, 1e6])
def test_log_softmax_and_vjp_match_float64_pytorch(scale):
    generator = torch.Generator().manual_seed(11)
    values = (torch.randn((5, 37), generator=generator, dtype=torch.float32) * scale).double().requires_grad_()
    seed = torch.randn((5, 37), generator=generator, dtype=torch.float32).double()
    reference = torch.nn.functional.log_softmax(values, dim=1)
    reference.backward(seed)
    logits = native(values)
    assert flat(logits.row_log_softmax()) == pytest.approx(reference.detach().flatten().tolist(), rel=2e-6, abs=1e-7)
    assert flat(logits.row_log_softmax_backward(native(seed))) == pytest.approx(values.grad.flatten().tolist(), rel=3e-5, abs=1e-7)
