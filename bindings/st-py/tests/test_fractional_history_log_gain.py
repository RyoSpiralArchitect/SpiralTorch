from array import array
import copy
import itertools
import math

import pytest

torch = pytest.importorskip("torch")
import spiraltorch as st
import spiraltorch.fractional_autograd as bridge


def reference(x, alpha, log_gain, length):
    """Independent ordinary Torch math, used only as a correctness reference."""
    a = alpha.double()
    coefficients = [torch.ones_like(a)]
    for k in range(1, length):
        coefficients.append(coefficients[-1] * (k - 1 - a) / k)
    past = torch.stack(coefficients[1:])
    past = log_gain.double().exp() * past / past.norm()
    output = torch.zeros_like(x, dtype=torch.float64)
    for k, value in enumerate(past, 1):
        if k < x.shape[1]:
            output = output + value * torch.cat((torch.zeros_like(x[:, :k]), x[:, :-k]), 1).double()
    return output.float()


@pytest.mark.parametrize("buffers", [False, True])
@pytest.mark.parametrize("needs", list(itertools.product([False, True], repeat=3))[1:])
def test_selective_gradients_match_independent_torch_math(monkeypatch, buffers, needs):
    monkeypatch.setattr(bridge, "_buffer_transport_available", lambda: buffers)
    x = torch.linspace(-.9, 1.1, 48).reshape(2, 8, 3).transpose(0, 1).contiguous().transpose(0, 1)
    variables = [x.requires_grad_(needs[0]), torch.tensor(2., requires_grad=needs[1]),
                 torch.tensor(.3, requires_grad=needs[2])]
    y = st.fractional_gl_history_log_gain_autograd(*variables, axis=1, kernel=st.FractionalGlKernel(kernel_len=6))
    expected = reference(*variables, 6)
    assert torch.allclose(y, expected, atol=2e-7, rtol=2e-6)
    requested = [p for p in variables if p.requires_grad]
    upstream = x.detach().cos()
    actual = torch.autograd.grad(y, requested, upstream)
    control = torch.autograd.grad(expected, requested, upstream)
    assert all(torch.allclose(a, b, atol=2e-6, rtol=3e-6) for a, b in zip(actual, control))


@pytest.mark.parametrize("buffers", [False, True])
def test_forward_ad_joint_and_scalar_only_directions(monkeypatch, buffers):
    monkeypatch.setattr(bridge, "_buffer_transport_available", lambda: buffers)
    x = torch.linspace(-.4, .8, 24).reshape(2, 4, 3)
    alpha, gain = torch.tensor(.7), torch.tensor(.2)
    dx, da, dg = x.cos(), torch.tensor(.3), torch.tensor(-.4)
    kernel = st.FractionalGlKernel(kernel_len=5)
    native = kernel.forward_history_log_gain(x.flatten().tolist(), list(x.shape), 1, alpha.item(), gain.item())
    with torch.autograd.forward_ad.dual_level():
        values = [torch.autograd.forward_ad.make_dual(v, d) for v, d in zip((x, alpha, gain), (dx, da, dg))]
        dual = st.fractional_gl_history_log_gain_autograd(*values, axis=1, kernel=kernel)
        _, tangent = torch.autograd.forward_ad.unpack_dual(dual)
    expected = torch.tensor(native.jvp(dx.flatten().tolist(), da.item(), dg.item())).reshape_as(x)
    assert torch.equal(tangent, expected)
    with torch.autograd.forward_ad.dual_level():
        dual_gain = torch.autograd.forward_ad.make_dual(gain, torch.tensor(1.))
        _, tangent = torch.autograd.forward_ad.unpack_dual(
            st.fractional_gl_history_log_gain_autograd(x, alpha, dual_gain, axis=1, kernel=kernel))
    assert torch.equal(tangent, torch.tensor(native.output).reshape_as(x))


def test_buffer_snapshots_are_owned_and_transport_matches():
    kernel = st.FractionalGlKernel(kernel_len=4)
    x = array("f", [1., 2., 3., 4.])
    batch = kernel.forward_history_log_gain_buffer(x, [4], 0, .7, .2)
    listed = kernel.forward_history_log_gain(list(x), [4], 0, .7, .2)
    x[0] = 50.
    output = batch.output_buffer()
    view = memoryview(output).cast("f")
    assert list(view) == batch.output == listed.output
    view[1] = 100.
    assert batch.output == listed.output
    upstream = array("f", [.1, -.2, .3, -.4])
    dx, da, dg = batch.vjp_buffer(upstream)
    assert (list(memoryview(dx).cast("f")), da, dg) == listed.vjp(list(upstream))
    assert batch.vjp_parameters_buffer(upstream) == (da, dg)
    assert batch.vjp_alpha_buffer(upstream) == da
    assert batch.vjp_log_gain_buffer(upstream) == dg
    assert list(memoryview(batch.jvp_buffer(upstream, .2, -.1)).cast("f")) == listed.jvp(list(upstream), .2, -.1)
    for bad in (array("d", [1.] * 4), memoryview(upstream)[::2], b"bad"):
        with pytest.raises((ValueError, TypeError, BufferError)):
            batch.vjp_log_gain_buffer(bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 100., -200.])
def test_unrepresentable_gain_fails_even_for_empty_history(bad):
    for length in (1, 5):
        kernel = st.FractionalGlKernel(kernel_len=length)
        with pytest.raises(ValueError, match="log-gain"):
            kernel.forward_history_log_gain([1.], [1], 0, .7, bad)


@pytest.mark.parametrize("bad", [.3, True, torch.tensor([.3]), torch.tensor(.3, dtype=torch.float64)])
def test_autograd_rejects_non_scalar_f32_gain(bad):
    with pytest.raises(TypeError, match="log_gain"):
        st.fractional_gl_history_log_gain_autograd(torch.ones(2, 3), torch.tensor(.7), bad, axis=1)


def test_autograd_requests_only_finite_components():
    kernel = st.FractionalGlKernel(kernel_len=2)
    x = torch.tensor([torch.finfo(torch.float32).max, 0.], requires_grad=True)
    y = st.fractional_gl_history_log_gain_autograd(x, torch.tensor(1.), torch.tensor(0.), axis=0, kernel=kernel)
    assert torch.equal(torch.autograd.grad(y, x, torch.tensor([0., 2.]))[0], torch.tensor([-2., 0.]))
    gain = torch.tensor(math.log(2.), requires_grad=True)
    y = st.fractional_gl_history_log_gain_autograd(torch.zeros(2), torch.tensor(1.), gain, axis=0, kernel=kernel)
    assert torch.autograd.grad(y, gain, torch.tensor([0., torch.finfo(torch.float32).max]))[0] == 0


def test_adapter_identity_recipe_separation_and_dtype_guards():
    adapter = st.FractionalGainHistoryAdapter(3, initial_alpha=2., initial_gain=5**.5, kernel_len=5)
    assert sum(p.numel() for p in adapter.parameters()) == 8
    x = torch.ones(2, 4, 3)
    assert torch.equal(adapter(x), x)
    assert adapter.get_extra_state()["schema"] == "spiraltorch.fractional_gain_history_adapter.v1"
    for old in (st.FractionalHistoryAdapter(3), st.FractionalL2HistoryAdapter(3)):
        with pytest.raises((ValueError, RuntimeError)):
            old.load_state_dict(adapter.state_dict())
        with pytest.raises((ValueError, RuntimeError)):
            adapter.load_state_dict(old.state_dict())
    adapter.log_gain.data = adapter.log_gain.data.double()
    with pytest.raises(TypeError, match="log_gain"):
        adapter(x)


def test_frozen_hf_learning_and_exact_next_update_resume():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(191)
    torch.set_num_threads(2)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=32, n_positions=8, n_embd=8, n_layer=1, n_head=2,
        resid_pdrop=0, attn_pdrop=0, embd_pdrop=0, use_cache=False,
    )).eval().requires_grad_(False)
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    adapter = st.FractionalGainHistoryAdapter(8, initial_alpha=2., initial_gain=5**.5, kernel_len=5)
    initial = {k: p.detach().clone() for k, p in adapter.named_parameters()}
    original = model.transformer.h[0].mlp
    model.transformer.h[0].mlp = torch.nn.Sequential(original, adapter)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=.01)
    tokens = torch.arange(12).reshape(2, 6) % 32

    def update(module, opt):
        opt.zero_grad()
        loss = model(tokens, labels=tokens).loss
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        opt.step()
        return float(loss.detach())

    losses = [update(adapter, optimizer) for _ in range(12)]
    assert all(math.isfinite(value) for value in losses)
    assert all(not torch.equal(p, initial[k]) for k, p in adapter.named_parameters())
    saved = copy.deepcopy(adapter.state_dict()), copy.deepcopy(optimizer.state_dict())
    next_loss = update(adapter, optimizer)
    restored = st.FractionalGainHistoryAdapter(8, kernel_len=5)
    restored.load_state_dict(saved[0])
    resumed = torch.optim.Adam(restored.parameters(), lr=.01)
    resumed.load_state_dict(saved[1])
    model.transformer.h[0].mlp = torch.nn.Sequential(original, restored)
    assert update(restored, resumed) == next_loss
    for p, q in zip(adapter.parameters(), restored.parameters()):
        assert torch.equal(p, q)
        assert all(torch.equal(value, resumed.state[q][key]) for key, value in optimizer.state[p].items())
    model.transformer.h[0].mlp = original
    assert all(torch.equal(p, before[k]) and p.grad is None for k, p in model.named_parameters())
