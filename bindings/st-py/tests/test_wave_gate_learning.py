import copy
import io
import json
from math import tanh as math_tanh

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not hasattr(st, "WaveGateKernel"), reason="native WaveGate kernel required"
)


def reference(x, gate, bias, curvature=-0.7, saturation=1.0, porosity=0.2):
    def saturate(value):
        outside = (
            value.sign()
            * saturation
            * (
                1
                - (porosity * 0.25)
                * (value.abs() - saturation)
                / (value.abs() + saturation)
            )
        )
        return torch.where(value.abs() <= saturation, value, outside)

    affine = saturate(x * saturate(gate) + bias)
    norm = affine.norm(dim=-1, keepdim=True)
    return affine * (torch.tanh(norm / (-curvature) ** 0.5) / norm)


def test_public_surface():
    for name in (
        "WaveGateKernel",
        "WaveGateLearningBatch",
        "WaveGateAdapter",
        "wave_gate_autograd",
    ):
        assert name in st.__all__ and name in dir(st)
        assert callable(getattr(st, name))


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_nd_forward_and_parameter_sum_match_independent_torch(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    x = torch.tensor(
        [[[0.2, -0.3], [1.5, 0.5]], [[-0.4, 2.0], [0.6, -0.7]]],
        device=device,
        requires_grad=True,
    )
    gate = torch.tensor([1.4, -1.1], device=device, requires_grad=True)
    bias = torch.tensor([0.15, -0.05], device=device, requires_grad=True)
    seed = torch.arange(8, dtype=torch.float32, device=device).reshape_as(x) / 11 - 0.3
    kernel = st.WaveGateKernel(curvature=-0.7, saturation=1.0, porosity=0.2)
    expected_inputs = [
        value.detach().cpu().double().requires_grad_() for value in (x, gate, bias)
    ]
    expected = reference(*expected_inputs)
    expected_gradients = torch.autograd.grad(
        (expected * seed.cpu().double()).sum(), expected_inputs
    )
    actual = st.wave_gate_autograd(x, gate, bias, kernel=kernel)
    (actual * seed).sum().backward()
    torch.testing.assert_close(
        actual.detach().cpu().double(), expected, rtol=3e-5, atol=3e-7
    )
    for parameter, gradient in zip((x, gate, bias), expected_gradients):
        torch.testing.assert_close(
            parameter.grad.cpu().double(), gradient, rtol=3e-4, atol=3e-7
        )


def test_snapshot_ownership_and_repeatability():
    kernel = st.WaveGateKernel()
    x, gate, bias = [0.2, -0.3], [1.4, -1.1], [0.1, -0.2]
    snapshot = kernel.forward(x, gate, bias, 1, 2)
    expected = snapshot.vjp([0.3, -0.1])
    output = snapshot.output
    x[:] = [float("nan")] * 2
    gate[:] = bias[:] = [0.0] * 2
    output[:] = [float("nan")] * 2
    del kernel
    assert snapshot.vjp([0.3, -0.1]) == expected
    assert all(torch.isfinite(torch.tensor(snapshot.output)))
    with pytest.raises(ValueError):
        snapshot.vjp([float("nan"), 0.0])
    with pytest.raises(ValueError):
        snapshot.vjp([0.0])


def test_conditioning_is_snapshot_local_optional_and_does_not_change_gradients():
    kernel = st.WaveGateKernel(curvature=-1.0, saturation=10.0)
    x = torch.tensor([[0.3, 0.4]], requires_grad=True)
    gate = torch.ones(2, requires_grad=True)
    bias = torch.zeros(2, requires_grad=True)
    output, report = st.wave_gate_autograd(
        x, gate, bias, kernel=kernel, return_conditioning=True
    )
    before = torch.autograd.grad(output.sum(), (x, gate, bias))
    plain = st.wave_gate_autograd(x, gate, bias, kernel=kernel)
    after = torch.autograd.grad(plain.sum(), (x, gate, bias))
    assert torch.equal(output, plain)
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    assert report["dimensionless_norm_mean"] == pytest.approx(0.5)
    assert report["relative_radial_gain_mean"] == pytest.approx(1 - math_tanh(0.5) ** 2)
    assert report["relative_tangential_gain_mean"] == pytest.approx(
        math_tanh(0.5) / 0.5
    )
    snapshot = kernel.forward([0.3, 0.4], [1.0, 1.0], [0.0, 0.0], 1, 2)
    assert json.loads(snapshot.conditioning_json()) == report
    report["relative_radial_gain_mean"] = 123
    assert json.loads(snapshot.conditioning_json())["relative_radial_gain_mean"] < 1
    adapter = st.WaveGateAdapter(2)
    result, identity = adapter.forward_with_conditioning(x)
    assert torch.equal(result, x) and identity["relative_radial_gain_mean"] == 1
    adapter.strength = 0
    assert adapter.forward_with_conditioning(x) == (x, None)


def test_identity_start_pointwise_causality_empty_and_noncontiguous():
    adapter = st.WaveGateAdapter(3, strength=0.3)
    assert adapter.execution_backend == "rust_f32_cpu"
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(1, 2) / 20
    assert not x.is_contiguous()
    assert torch.equal(adapter(x), x)
    adapter(x).sum().backward()
    assert adapter.gate.grad.abs().sum() > 0 and adapter.bias.grad.abs().sum() > 0
    with torch.no_grad():
        adapter.gate.fill_(0.5)
    changed = x.clone()
    changed[:, 2:] += 20
    assert torch.equal(adapter(x)[:, :2], adapter(changed)[:, :2])
    empty = torch.empty(0, 3, requires_grad=True)
    adapter.zero_grad()
    adapter(empty).sum().backward()
    assert empty.grad.shape == (0, 3)
    assert torch.equal(adapter.gate.grad, torch.zeros(3))
    adapter.strength = 0
    assert adapter(x) is x


def test_recipe_changes_do_not_reinterpret_backward_and_inplace_is_rejected():
    adapter = st.WaveGateAdapter(2, strength=0.2)
    x = torch.tensor([[0.2, -0.3]], requires_grad=True)
    result = adapter(x)
    state = adapter.get_extra_state()
    state["kernel"]["curvature"] = -4.0
    adapter.set_extra_state(state)
    result.sum().backward()
    torch.testing.assert_close(adapter.gate.grad, x.detach()[0] * 0.2)
    torch.testing.assert_close(adapter.bias.grad, torch.full((2,), 0.2))
    y = adapter(x)
    with torch.no_grad():
        adapter.gate.add_(0.1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.sum().backward()
    accepted = adapter.get_extra_state()
    broken = copy.deepcopy(accepted)
    broken["kernel"]["curvature"] = float("nan")
    with pytest.raises(ValueError):
        adapter.set_extra_state(broken)
    assert adapter.get_extra_state() == accepted


def test_invalid_input_shape_dtype_and_budget():
    adapter = st.WaveGateAdapter(2, max_values=4)
    for dtype in (torch.float16, torch.float64):
        with pytest.raises(TypeError, match="float32"):
            adapter(torch.ones(1, 2, dtype=dtype))
    with pytest.raises(ValueError, match="budget"):
        adapter(torch.ones(3, 2))
    with pytest.raises(ValueError):
        adapter(torch.tensor([[float("inf"), 0.0]]))
    with pytest.raises(ValueError, match="vector"):
        st.wave_gate_autograd(torch.ones(2, 2), torch.ones(2, 2), torch.zeros(2))
    for curvature in (0, 1, float("nan"), -float("inf")):
        with pytest.raises(ValueError):
            st.WaveGateKernel(curvature=curvature)
    with pytest.raises(ValueError):
        st.WaveGateKernel().forward([], [], [], 0, 0)


def test_adam_learning_and_checkpoint_reproduce_next_update():
    torch.manual_seed(29)
    x = torch.randn(32, 3) * 0.4
    teacher = st.WaveGateAdapter(3, strength=0.3, curvature=-0.7, porosity=0.2)
    with torch.no_grad():
        teacher.gate.copy_(torch.tensor([0.5, -0.4, 0.3]))
        teacher.bias.copy_(torch.tensor([0.1, -0.1, 0.05]))
        target = teacher(x)
    student = st.WaveGateAdapter(3, strength=0.3, curvature=-0.7, porosity=0.2)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.03)
    initial = (student(x) - target).square().mean().item()
    for _ in range(80):
        optimizer.zero_grad()
        (student(x) - target).square().mean().backward()
        optimizer.step()
    assert (student(x) - target).square().mean().item() < initial * 0.01
    stream = io.BytesIO()
    torch.save(student.state_dict(), stream)
    stream.seek(0)
    restored = st.WaveGateAdapter(3)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    resumed = torch.optim.Adam(restored.parameters(), lr=0.03)
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for model, opt in ((student, optimizer), (restored, resumed)):
        opt.zero_grad()
        (model(x) - target).square().mean().backward()
        opt.step()
    assert torch.equal(student.gate, restored.gate)
    assert torch.equal(student.bias, restored.bias)
    assert torch.equal(student(x), restored(x))


def test_hf_causal_loss_reaches_gate_and_bias_with_frozen_base():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(23)
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=16,
            n_positions=8,
            n_embd=8,
            n_layer=1,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    ).eval()
    model.requires_grad_(False)
    base = [(parameter, parameter.detach().clone()) for parameter in model.parameters()]
    ids = torch.tensor([[1, 2, 3, 1, 2, 3]])
    baseline = model(ids).logits.detach().clone()
    adapter = st.WaveGateAdapter(8, strength=0.2)
    model.transformer.h[0].mlp = torch.nn.Sequential(
        model.transformer.h[0].mlp, adapter
    )
    assert torch.equal(model(ids).logits.detach(), baseline)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.03)
    losses = []
    for _ in range(8):
        optimizer.zero_grad()
        loss = model(ids, labels=ids).loss
        loss.backward()
        for parameter in adapter.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert all(
        parameter.grad is None and torch.equal(parameter, saved)
        for parameter, saved in base
    )
    assert not torch.equal(model(ids).logits.detach(), baseline)
