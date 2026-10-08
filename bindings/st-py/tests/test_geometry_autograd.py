"""Real Rust-backed gradients and learning, not mocked geometric reports."""

import copy
import io

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not hasattr(st, "ToposResonatorKernel"),
    reason="native geometric kernel is not built",
)


def test_public_imports_are_discoverable():
    for name in (
        "ToposResonatorKernel",
        "ToposResonatorAdapter",
        "topos_resonator_autograd",
    ):
        assert name in st.__all__ and name in dir(st)
        assert callable(getattr(st, name))


@pytest.mark.parametrize("input_scale", [1.0, 20.0])
def test_native_vjp_matches_finite_difference_and_broadcast_gradient(input_scale):
    kernel = st.ToposResonatorKernel(
        coupling=0.35, iterations=6, saturation=1.0, porosity=0.2
    )
    x = (torch.tensor([[0.2, -0.4], [0.6, 0.3]]) * input_scale).requires_grad_()
    gate = torch.tensor([0.3, -0.2], requires_grad=True)
    seed = torch.tensor([[0.4, -0.7], [-0.3, 0.8]])
    y = st.topos_resonator_autograd(x, gate, kernel=kernel)
    (y * seed).sum().backward()
    gx, gg = kernel.backward(
        x.detach().flatten().tolist(),
        gate.detach().expand_as(x).flatten().tolist(),
        seed.flatten().tolist(),
        2,
        2,
    )
    torch.testing.assert_close(x.grad, torch.tensor(gx).reshape_as(x), rtol=0, atol=0)
    torch.testing.assert_close(
        gate.grad, torch.tensor(gg).reshape_as(x).sum(0), rtol=0, atol=0
    )
    for parameter in (x, gate):
        expected = torch.empty_like(parameter)
        for i in range(parameter.numel()):
            with torch.no_grad():
                old = parameter.flatten()[i].item()
                # Relative perturbations avoid f32 cancellation on saturated tails.
                h = 1e-3 * max(1.0, abs(old))
                parameter.flatten()[i] = old + h
                plus = (
                    (
                        st.topos_resonator_autograd(x, gate, kernel=kernel).double()
                        * seed.double()
                    )
                    .sum()
                    .item()
                )
                parameter.flatten()[i] = old - h
                minus = (
                    (
                        st.topos_resonator_autograd(x, gate, kernel=kernel).double()
                        * seed.double()
                    )
                    .sum()
                    .item()
                )
                parameter.flatten()[i] = old
            expected.flatten()[i] = (plus - minus) / (2 * h)
        torch.testing.assert_close(parameter.grad, expected, rtol=2e-3, atol=3e-5)


def test_identity_initialization_and_zero_strength_control():
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 20
    adapter = st.ToposResonatorAdapter(4, strength=0.25)
    assert adapter.execution_backend == "rust_f32_cpu"
    assert torch.equal(adapter(x), x)
    adapter(x).sum().backward()
    assert adapter.gate.grad.abs().sum().item() > 0
    with torch.no_grad():
        adapter.gate.fill_(0.5)
    assert not torch.equal(adapter(x), x)
    adapter.strength = 0.0
    assert adapter(x) is x


def test_pointwise_causality_empty_batch_and_noncontiguous_input():
    adapter = st.ToposResonatorAdapter(3)
    with torch.no_grad():
        adapter.gate.copy_(torch.tensor([0.1, 0.2, 0.3]))
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(1, 2) / 20
    assert not x.is_contiguous()
    before = adapter(x)
    changed = x.clone()
    changed[:, 2:] += 20
    assert torch.equal(before[:, :2], adapter(changed)[:, :2])
    empty = torch.empty(0, 3, requires_grad=True)
    adapter(empty).sum().backward()
    assert empty.grad.shape == (0, 3)


def test_invalid_inputs_and_stale_backward_fail_explicitly():
    adapter = st.ToposResonatorAdapter(2, max_values=4)
    for x in (
        torch.ones(1, 2, dtype=torch.float64),
        torch.ones(1, 2, dtype=torch.float16),
    ):
        with pytest.raises(TypeError, match="float32"):
            adapter(x)
    with pytest.raises(ValueError, match="budget"):
        adapter(torch.ones(3, 2))
    with pytest.raises(ValueError):
        adapter(torch.tensor([[float("nan"), 0.0]]))
    with pytest.raises(ValueError):
        st.ToposResonatorKernel(coupling=1.0)
    x = torch.ones(1, 2, requires_grad=True)
    y = adapter(x)
    with torch.no_grad():
        x.add_(0.1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.sum().backward()


def test_gate_learning_and_checkpoint_reproduce_next_update():
    torch.manual_seed(19)
    x = torch.randn(32, 3) * 0.4
    teacher = st.ToposResonatorAdapter(3, strength=0.3, coupling=0.4, iterations=5)
    with torch.no_grad():
        teacher.gate.copy_(torch.tensor([0.6, -0.4, 0.3]))
        target = teacher(x)
    student = st.ToposResonatorAdapter(3, strength=0.3, coupling=0.4, iterations=5)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.03)
    initial = (student(x) - target).square().mean().item()
    for _ in range(60):
        optimizer.zero_grad()
        loss = (student(x) - target).square().mean()
        loss.backward()
        optimizer.step()
    assert (student(x) - target).square().mean().item() < initial * 0.02
    restored = st.ToposResonatorAdapter(3)
    stream = io.BytesIO()
    torch.save(student.state_dict(), stream)
    stream.seek(0)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.03)
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for layer, opt in ((student, optimizer), (restored, restored_optimizer)):
        opt.zero_grad()
        (layer(x) - target).square().mean().backward()
        opt.step()
    assert torch.equal(restored.gate, student.gate)
    assert torch.equal(restored(x), student(x))


def test_checkpoint_recipe_change_does_not_reinterpret_live_backward():
    adapter = st.ToposResonatorAdapter(2)
    x = torch.tensor([[0.2, 0.3]], requires_grad=True)
    result = adapter(x)
    state = adapter.get_extra_state()
    state["kernel"]["coupling"] = 0.8
    adapter.set_extra_state(state)
    result.sum().backward()
    old_gain = sum(0.25**i for i in range(4))
    torch.testing.assert_close(adapter.gate.grad, x.detach()[0] * 0.1 * old_gain)
    accepted = adapter.get_extra_state()
    broken = copy.deepcopy(accepted)
    broken["kernel"]["coupling"] = 1.0
    with pytest.raises(ValueError):
        adapter.set_extra_state(broken)
    assert adapter.get_extra_state() == accepted


def test_tiny_hf_lm_loss_updates_geometry_without_unfreezing_base():
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
    )
    model.requires_grad_(False)
    model.eval()
    ids = torch.tensor([[1, 2, 3, 1, 2, 3]])
    baseline = model(ids).logits.detach().clone()
    adapter = st.ToposResonatorAdapter(8, strength=0.2)
    model.transformer.h[0].mlp = torch.nn.Sequential(
        model.transformer.h[0].mlp, adapter
    )
    assert torch.equal(model(ids).logits.detach(), baseline)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.03)
    for _ in range(4):
        optimizer.zero_grad()
        loss = model(ids, labels=ids).loss
        assert torch.isfinite(loss)
        loss.backward()
        assert (
            adapter.gate.grad is not None and adapter.gate.grad.abs().sum().item() > 0
        )
        optimizer.step()
    assert adapter.gate.detach().abs().sum().item() > 0
    assert not torch.equal(model(ids).logits.detach(), baseline)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


def test_gpu_tensor_transport_keeps_device_and_gradients():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        pytest.skip("no accelerator for explicit CPU-bridge transport test")
    cpu = st.ToposResonatorAdapter(2, porosity=0.2)
    with torch.no_grad():
        cpu.gate.copy_(torch.tensor([0.3, -0.2]))
    gpu = st.ToposResonatorAdapter(2)
    gpu.load_state_dict(copy.deepcopy(cpu.state_dict()))
    gpu.to(device)
    x = torch.tensor([[0.2, -0.4], [4.0, -3.0]], requires_grad=True)
    remote_x = x.detach().to(device).requires_grad_()
    expected, actual = cpu(x), gpu(remote_x)
    assert actual.device.type == device
    assert gpu.execution_backend == "rust_f32_cpu"
    expected.sum().backward()
    actual.sum().backward()
    torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(remote_x.grad.cpu(), x.grad)
    torch.testing.assert_close(gpu.gate.grad.cpu(), cpu.gate.grad)
