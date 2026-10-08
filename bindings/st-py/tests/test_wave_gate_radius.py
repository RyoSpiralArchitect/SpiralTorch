import copy
import io
import math

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not hasattr(st, "WaveGateKernel"), reason="native WaveGate required"
)


def reference(x, gate, bias, log_radius):
    def saturate(v):
        outside = v.sign() * (1 - 0.05 * (v.abs() - 1) / (v.abs() + 1))
        return torch.where(v.abs() <= 1, v, outside)

    z = saturate(x * saturate(gate) + bias)
    norm = z.norm(dim=-1, keepdim=True)
    radius = log_radius.exp()
    return radius * torch.tanh(norm / (0.7**0.5 * radius)) * z / norm


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("log_radius", [-2.0, 0.0, 2.0])
def test_four_vjps_match_independent_torch(device, log_radius):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    values = [
        torch.tensor([[[0.2, -0.3], [1.5, 0.5]], [[-0.4, 2.0], [0.6, -0.7]]]),
        torch.tensor([1.4, -1.1]),
        torch.tensor([0.15, -0.05]),
        torch.tensor(log_radius),
    ]
    inputs = [v.to(device).requires_grad_() for v in values]
    expected_inputs = [v.double().requires_grad_() for v in values]
    seed = torch.arange(8, dtype=torch.float32).reshape_as(values[0]) / 11 - 0.3
    expected = reference(*expected_inputs)
    gradients = torch.autograd.grad((expected * seed).sum(), expected_inputs)
    actual, report = st.wave_gate_autograd(
        *inputs[:3],
        log_radius=inputs[3],
        kernel=st.WaveGateKernel(curvature=-0.7, porosity=0.2),
        return_conditioning=True,
    )
    actual_gradients = torch.autograd.grad((actual * seed.to(device)).sum(), inputs)
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=3e-5, atol=3e-7)
    for actual_gradient, expected_gradient in zip(actual_gradients, gradients):
        torch.testing.assert_close(
            actual_gradient.cpu().double(), expected_gradient, rtol=3e-4, atol=3e-7
        )
    assert report["projection_radius"] == pytest.approx(math.exp(log_radius))
    assert report["schema"] == "spiraltorch.wave_gate_conditioning.v2"


def test_identity_origin_gain_radius_starts_stationary_then_learns_and_resumes():
    torch.manual_seed(41)
    x = torch.randn(32, 3) * 0.4
    legacy = st.WaveGateAdapter(3, curvature=-0.7)
    adapter = st.WaveGateAdapter(
        3, curvature=-0.7, log_radius=0.8, learnable_radius=True
    )
    assert torch.equal(adapter(x), x)
    legacy(x).sum().backward()
    adapter(x).sum().backward()
    torch.testing.assert_close(
        legacy.gate.grad, adapter.gate.grad, rtol=1e-6, atol=1e-7
    )
    assert adapter.log_radius.grad.item() == 0
    target = 0.9 * x
    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.02)
    initial = (adapter(x) - target).square().mean().item()
    for _ in range(30):
        optimizer.zero_grad()
        (adapter(x) - target).square().mean().backward()
        optimizer.step()
    assert torch.isfinite(adapter.log_radius.grad)
    assert adapter.log_radius.grad.abs() > 0
    assert adapter.log_radius.item() != pytest.approx(0.8)
    assert (adapter(x) - target).square().mean().item() < initial
    stream = io.BytesIO()
    torch.save(
        {"adapter": adapter.state_dict(), "optimizer": optimizer.state_dict()}, stream
    )
    stream.seek(0)
    saved = torch.load(stream, weights_only=True)
    restored = st.WaveGateAdapter(3, learnable_radius=True)
    restored.load_state_dict(saved["adapter"])
    resumed = torch.optim.Adam(restored.parameters(), lr=0.02)
    resumed.load_state_dict(saved["optimizer"])
    for model, opt in ((adapter, optimizer), (restored, resumed)):
        opt.zero_grad()
        (model(x) - target).square().mean().backward()
        opt.step()
    for a, b in zip(adapter.parameters(), restored.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(optimizer.state.values(), resumed.state.values()):
        assert all(torch.equal(a[key], b[key]) for key in a)
    fixed = st.WaveGateAdapter(3, log_radius=0.0)
    with pytest.raises(ValueError, match="incompatible"):
        fixed.set_extra_state(adapter.get_extra_state())
    assert "log_radius" in dict(fixed.named_buffers())
    assert "log_radius" not in dict(fixed.named_parameters())
    assert "log_radius" not in legacy.state_dict()


def test_radius_shape_domain_snapshot_and_inplace_guards():
    kernel = st.WaveGateKernel()
    x, gate, bias = torch.ones(1, 2), torch.ones(2), torch.zeros(2)
    for invalid in [
        torch.ones(1),
        torch.tensor(float("nan")),
        torch.tensor(100.0),
        torch.tensor(-100.0),
    ]:
        with pytest.raises(ValueError):
            st.wave_gate_autograd(x, gate, bias, log_radius=invalid)
    with pytest.raises(TypeError):
        st.wave_gate_autograd(x, gate, bias, log_radius=0.0)
    batch = kernel.forward_with_log_radius(
        [0.2, -0.3], [1.4, -1.1], [0.1, 0.2], 1, 2, 0.5
    )
    expected = batch.vjp_with_log_radius([0.1, 0.2])
    assert batch.vjp([0.1, 0.2]) == expected[:3]
    assert batch.vjp_with_log_radius([0.1, 0.2]) == expected
    adapter = st.WaveGateAdapter(2, learnable_radius=True)
    output = adapter(x)
    with torch.no_grad():
        adapter.log_radius.add_(0.1)
    with pytest.raises(RuntimeError, match="inplace"):
        output.sum().backward()
    empty = torch.empty(0, 2, requires_grad=True)
    adapter(empty).sum().backward()
    assert adapter.log_radius.grad.item() == 0
    accepted = copy.deepcopy(adapter.get_extra_state())
    invalid = copy.deepcopy(accepted)
    invalid["learnable_radius"] = 1
    with pytest.raises(ValueError):
        adapter.set_extra_state(invalid)
    assert adapter.get_extra_state() == accepted


def test_hf_loss_updates_radius_after_identity_start_with_frozen_base():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(23)
    model = (
        transformers.GPT2LMHeadModel(
            transformers.GPT2Config(
                vocab_size=16,
                n_positions=8,
                n_embd=8,
                n_layer=1,
                n_head=2,
                resid_pdrop=0,
                embd_pdrop=0,
                attn_pdrop=0,
            )
        )
        .eval()
        .requires_grad_(False)
    )
    base = [(p, p.detach().clone()) for p in model.parameters()]
    adapter = st.WaveGateAdapter(8, strength=0.2, learnable_radius=True)
    model.transformer.h[0].mlp = torch.nn.Sequential(
        model.transformer.h[0].mlp, adapter
    )
    ids = torch.tensor([[1, 2, 3, 1, 2, 3]])
    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.03)
    losses, radius_gradients = [], []
    for _ in range(8):
        optimizer.zero_grad()
        loss = model(ids, labels=ids).loss
        loss.backward()
        losses.append(loss.item())
        radius_gradients.append(adapter.log_radius.grad.item())
        optimizer.step()
    assert radius_gradients[0] == 0 and any(abs(v) > 0 for v in radius_gradients[1:])
    assert all(math.isfinite(v) for v in radius_gradients)
    assert losses[-1] < losses[0]
    assert all(p.grad is None and torch.equal(p, saved) for p, saved in base)
