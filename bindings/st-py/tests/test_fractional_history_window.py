from array import array
import copy
import itertools
import math

import pytest

torch = pytest.importorskip("torch")
import spiraltorch as st
import spiraltorch.fractional_autograd as bridge


def reference(x, alpha, log_gain, length, window):
    """Independent Torch reference, never a production backend."""
    a = alpha.double()
    coefficients = [torch.ones_like(a)]
    for lag in range(1, length):
        coefficients.append(coefficients[-1] * (lag - 1 - a) / lag)
    past = torch.stack(coefficients[1:])
    normalized = log_gain.double().exp() * past / past.norm()
    output = x.double() * 0
    for lag in range(*window):
        if lag < x.shape[1]:
            output = output + normalized[lag - 1] * torch.cat((torch.zeros_like(x[:, :lag]), x[:, :-lag]), 1).double()
    return output.float()


@pytest.mark.parametrize("buffers", [False, True])
@pytest.mark.parametrize("needs", list(itertools.product([False, True], repeat=3))[1:])
@pytest.mark.parametrize("window", [(1, 3), (3, 8)])
def test_window_gradients_match_independent_full_normalization(monkeypatch, buffers, needs, window):
    monkeypatch.setattr(bridge, "_buffer_transport_available", lambda: buffers)
    for order in (.2, 2.):
        x = torch.linspace(-.7, .9, 60).reshape(2, 10, 3).transpose(0, 1).contiguous().transpose(0, 1)
        values = [x.requires_grad_(needs[0]), torch.tensor(order, requires_grad=needs[1]),
                  torch.tensor(.3, requires_grad=needs[2])]
        actual = st.fractional_gl_history_log_gain_autograd(*values, axis=1,
                    kernel=st.FractionalGlKernel(kernel_len=8), lag_window=window)
        expected = reference(*values, 8, window)
        assert torch.allclose(actual, expected, atol=3e-7, rtol=3e-6)
        requested = [v for v in values if v.requires_grad]
        a = torch.autograd.grad(actual, requested, x.detach().cos())
        b = torch.autograd.grad(expected, requested, x.detach().cos())
        assert all(torch.allclose(left, right, atol=3e-6, rtol=3e-6) for left, right in zip(a, b))


@pytest.mark.parametrize("buffers", [False, True])
@pytest.mark.parametrize("window", [(1, 3), (3, 8)])
def test_window_joint_jvp_and_angular_composition(monkeypatch, buffers, window):
    monkeypatch.setattr(bridge, "_buffer_transport_available", lambda: buffers)
    x, angle, gain = torch.linspace(-.4, .8, 60).reshape(2, 10, 3), torch.tensor(-.2), torch.tensor(.2)
    dx, da, dg = x.cos(), torch.tensor(.1), torch.tensor(-.3)
    kernel = st.FractionalGlKernel(kernel_len=8)
    with torch.autograd.forward_ad.dual_level():
        dual_x, dual_angle, dual_gain = [torch.autograd.forward_ad.make_dual(v, d)
                                       for v, d in zip((x, angle, gain), (dx, da, dg))]
        result = st.fractional_gl_history_log_gain_autograd(dual_x,
            st.fractional_gl_angle_autograd(dual_angle), dual_gain, axis=1, kernel=kernel, lag_window=window)
        _, actual = torch.autograd.forward_ad.unpack_dual(result)
    _, expected = torch.autograd.functional.jvp(
        lambda x, a, g: reference(x, 1 + 2*a.tan(), g, 8, window), (x, angle, gain), (dx, da, dg))
    assert torch.allclose(actual, expected, atol=2e-6, rtol=3e-6)


def test_native_window_buffer_ownership_and_full_window_compatibility():
    kernel = st.FractionalGlKernel(kernel_len=8)
    x = array("f", [1., -.2, .4, -.8, .3, .6, -.1, .7])
    all_lags = kernel.forward_history_log_gain_window(x, [8], 0, .7, .2, 1, 8)
    original = kernel.forward_history_log_gain(x, [8], 0, .7, .2)
    assert all_lags.output == original.output
    assert all_lags.vjp(x) == original.vjp(x)
    assert all_lags.jvp(x, .2, -.3) == original.jvp(x, .2, -.3)
    short = kernel.forward_history_log_gain_window_buffer(x, [8], 0, .7, .2, 1, 3)
    listed = kernel.forward_history_log_gain_window(x, [8], 0, .7, .2, 1, 3)
    x[0] = 99
    view = memoryview(short.output_buffer()).cast("f")
    assert list(view) == short.output == listed.output
    view[1] = 88
    assert short.output == listed.output
    dx, da, dg = short.vjp_buffer(x)
    assert (list(memoryview(dx).cast("f")), da, dg) == listed.vjp(x)


@pytest.mark.parametrize("window", [(0, 3), (4, 3), (1, 9), (-1, 3), (True, 3), (.5, 3), "1:3", (), (1,)])
def test_bad_windows_are_rejected_by_api_and_adapter(window):
    with pytest.raises((TypeError, ValueError, OverflowError)):
        st.fractional_gl_history_log_gain_autograd(torch.ones(1, 4, 2), torch.tensor(.7),
            torch.tensor(0.), axis=1, kernel=st.FractionalGlKernel(kernel_len=8), lag_window=window)
    with pytest.raises((TypeError, ValueError, OverflowError)):
        st.FractionalAngleGainHistoryAdapter(2, kernel_len=8, lag_window=window)


@pytest.mark.parametrize("kind", [st.FractionalGainHistoryAdapter, st.FractionalAngleGainHistoryAdapter])
def test_window_recipe_cannot_silently_replace_full_or_other_window(kind):
    source = kind(3, kernel_len=8, lag_window=(1, 3))
    saved = copy.deepcopy(source.state_dict())
    for target in (kind(3, kernel_len=8), kind(3, kernel_len=8, lag_window=(3, 8))):
        with pytest.raises(ValueError):
            target.load_state_dict(saved)
        with pytest.raises(ValueError):
            source.load_state_dict(target.state_dict())
    restored = kind(3, lag_window=(1, 3))
    restored.load_state_dict(saved)
    assert restored.get_extra_state() == source.get_extra_state()
    assert restored.lag_window == (1, 3)
    bad = copy.deepcopy(saved)
    bad["_extra_state"]["kernel"]["kernel_len"] = 2
    with pytest.raises(ValueError, match="window"):
        restored.load_state_dict(bad)
    assert restored.get_extra_state() == source.get_extra_state()
    assert set(kind(3).get_extra_state()) == {"schema", "features", "strength", "kernel"}


def test_tail_learning_needs_active_initial_support_when_feature_gates_are_zero():
    x = torch.arange(1, 13, dtype=torch.float32).reshape(1, 6, 2)
    dormant = st.FractionalAngleGainHistoryAdapter(2, kernel_len=8, lag_window=(3, 8))
    history = dormant._history(x, dormant._alpha_tensor())
    assert torch.count_nonzero(history) == 0
    assert torch.autograd.grad(history.sum(), dormant.history_angle)[0] != 0
    dormant(x).sum().backward()
    # A nonzero operator differential alone cannot cross a zero feature gate.
    assert torch.count_nonzero(dormant.gate.grad) == 0
    assert dormant.history_angle.grad == dormant.log_gain.grad == 0
    active = st.FractionalAngleGainHistoryAdapter(2, initial_angle=-.2, kernel_len=8, lag_window=(3, 8))
    assert torch.equal(active(x), x)
    active(x).sum().backward()
    assert torch.count_nonzero(active.gate.grad) == 2


@pytest.mark.parametrize("kind", [st.FractionalGainHistoryAdapter, st.FractionalAngleGainHistoryAdapter])
@pytest.mark.parametrize("window", [(1, 3), (3, 8)])
def test_window_learns_in_hf_and_resumes_exactly_without_training_the_base(kind, window):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(193)
    torch.set_num_threads(2)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=32, n_positions=8, n_embd=8, n_layer=1, n_head=2,
        resid_pdrop=0, attn_pdrop=0, embd_pdrop=0, use_cache=False,
    )).eval().requires_grad_(False)
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    options = {"initial_angle": -.2} if kind is st.FractionalAngleGainHistoryAdapter else {"initial_alpha": .65}
    adapter = kind(8, kernel_len=8, lag_window=window, **options)
    initial = {k: p.detach().clone() for k, p in adapter.named_parameters()}
    assert sum(p.numel() for p in adapter.parameters()) == 18
    original = model.transformer.h[0].mlp
    model.transformer.h[0].mlp = torch.nn.Sequential(original, adapter)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=.01)
    tokens = torch.arange(16).reshape(2, 8) % 32

    def update(module, opt):
        opt.zero_grad()
        loss = model(tokens, labels=tokens).loss
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        opt.step()
        return float(loss.detach())

    losses = [update(adapter, optimizer) for _ in range(12)]
    assert all(math.isfinite(v) for v in losses)
    assert all(not torch.equal(p, initial[k]) for k, p in adapter.named_parameters())
    saved = copy.deepcopy(adapter.state_dict()), copy.deepcopy(optimizer.state_dict())
    next_loss = update(adapter, optimizer)
    restored = kind(8, kernel_len=8, lag_window=window)
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
