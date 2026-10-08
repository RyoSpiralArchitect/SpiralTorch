import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not hasattr(st.nn, "WaveGate"), reason="native NN bindings required"
)


def tensor(value):
    value = torch.as_tensor(value, dtype=torch.float32)
    return st.Tensor(tuple(value.shape), data=value.flatten().tolist())


def state(layer):
    return {name: value.tolist() for name, value in layer.state_dict()}


def test_wave_gate_raw_vjp_matches_torch_without_accumulation():
    layer = st.nn.WaveGate("wave", 2, -1.0, 0.5)
    x = torch.tensor(
        [[9e-5, -7.5e-5], [3e-5, 6e-5]], dtype=torch.float64, requires_grad=True
    )
    gate = torch.tensor([[14000.0, -11000.0]], dtype=torch.float64, requires_grad=True)
    bias = torch.tensor([[0.15, -0.05]], dtype=torch.float64, requires_grad=True)
    seed = torch.tensor([[0.35, -0.2], [-0.1, 0.3]], dtype=torch.float64)
    layer.load_state_dict([("wave::gate", tensor(gate)), ("wave::bias", tensor(bias))])

    def saturate(value):
        limit = 10000.0
        outside = (
            value.sign()
            * limit
            * (1.0 - 0.05 * (value.abs() - limit) / (value.abs() + limit))
        )
        return torch.where(value.abs() <= limit, value, outside)

    affine = saturate(x * saturate(gate) + bias)
    norm = affine.norm(dim=-1, keepdim=True)
    expected = affine * (norm.tanh() / norm)
    expected_grads = torch.autograd.grad((expected * seed).sum(), (x, gate, bias))
    actual = layer.forward(tensor(x))
    grads = layer.vjp(tensor(x), tensor(seed))
    torch.testing.assert_close(
        torch.tensor(actual.tolist()).double(), expected, rtol=2e-5, atol=2e-7
    )
    for actual, expected in zip(grads, expected_grads):
        torch.testing.assert_close(
            torch.tensor(actual.tolist()).double(), expected, rtol=3e-4, atol=2e-9
        )
    before = state(layer)
    layer.apply_step(0.01)
    assert state(layer) == before


def test_wave_gate_module_policy_and_repeatable_state_restore():
    layer = st.nn.WaveGate("wave", 2, -1.0, 0.5)
    x = tensor([[0.2, -0.3], [0.4, 0.5]])
    seed = tensor([[0.1, -0.2], [0.3, 0.4]])
    before = state(layer)
    _, dg, db = layer.vjp(x, seed)
    layer.backward(x, seed)
    layer.apply_step(0.1)
    after = state(layer)
    for name, derivative in [("wave::gate", dg), ("wave::bias", db)]:
        expected = torch.tensor(before[name]) - 0.05 * torch.tensor(derivative.tolist())
        torch.testing.assert_close(torch.tensor(after[name]), expected)
    clone = st.nn.WaveGate("wave", 2, -1.0, 0.5)
    clone.load_state_dict(layer.state_dict())
    layer.zero_accumulators()
    for current in (layer, clone):
        current.backward(x, seed)
        current.apply_step(0.1)
    assert state(layer) == state(clone)


def test_wave_gate_vjp_empty_and_invalid_upstream():
    layer = st.nn.WaveGate("wave", 2, -1.0, 0.5)
    empty = st.Tensor((0, 2), data=[])
    dx, dg, db = layer.vjp(empty, empty)
    assert dx.shape() == (0, 2)
    assert dg.tolist() == [[0.0, 0.0]] and db.tolist() == [[0.0, 0.0]]
    x = tensor([[0.2, -0.3]])
    before = state(layer)
    with pytest.raises((ValueError, RuntimeError)):
        layer.backward(x, tensor([[float("nan"), 0.5]]))
    layer.apply_step(0.1)
    assert state(layer) == before
