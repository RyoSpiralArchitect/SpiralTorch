"""Independent Torch references for Rust-owned normalized history, not a backend."""

from array import array
import copy

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")


def oracle(value, alpha, *, axis, kernel_len, gain):
    x, a = value.double().movedim(axis, 0), alpha.double()
    if kernel_len == 1 or x.shape[0] == 1:
        return (value.double() * 0 + a * 0).float()
    taps = [torch.zeros_like(a)]
    for k in range(1, kernel_len):
        terms = torch.stack([(j - a) / (j + 1) for j in range(k)])
        taps.append(terms.prod())
    taps = torch.stack(taps)
    taps = gain * taps / torch.linalg.vector_norm(taps)
    return torch.stack([
        x[t] * 0 + a * 0 + sum(taps[k] * x[t-k]
                              for k in range(1, min(t + 1, kernel_len)))
        for t in range(x.shape[0])
    ]).movedim(0, axis).float()


@pytest.mark.parametrize("order", [.01, .45, 1., 2., 4.])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("buffers", [False, True])
def test_output_and_both_gradients_match_torch(order, axis, buffers, monkeypatch):
    import spiraltorch.fractional_autograd as bridge

    monkeypatch.setattr(bridge, "_buffer_transport_available", lambda: buffers)
    x = torch.linspace(-.8, .9, 40).reshape(2, 5, 4).transpose(0, 2).requires_grad_()
    a = torch.tensor(order, requires_grad=True)
    kernel = st.FractionalGlKernel(kernel_len=7, step=.7)
    y = st.fractional_gl_history_l2_autograd(x, a, axis=axis, kernel=kernel, gain=1.5)
    reference = oracle(x, a, axis=axis, kernel_len=7, gain=1.5)
    torch.testing.assert_close(y, reference, rtol=3e-6, atol=3e-6)
    observed = torch.autograd.grad(y, (x, a), x.detach().cos())
    expected = torch.autograd.grad(reference, (x, a), x.detach().cos())
    for actual, wanted in zip(observed, expected):
        torch.testing.assert_close(actual, wanted, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("buffers", [False, True])
def test_forward_ad_and_selective_gradients(buffers, monkeypatch):
    import spiraltorch.fractional_autograd as bridge

    monkeypatch.setattr(bridge, "_buffer_transport_available", lambda: buffers)
    x = torch.linspace(-.7, .8, 18).reshape(2, 3, 3)
    a, dx, da = torch.tensor(2.), x.cos(), torch.tensor(.2)
    kernel = st.FractionalGlKernel(kernel_len=5, step=.4)
    operation = lambda x, a: st.fractional_gl_history_l2_autograd(
        x, a, axis=1, kernel=kernel, gain=.5)
    with torch.autograd.forward_ad.dual_level():
        y = operation(torch.autograd.forward_ad.make_dual(x, dx),
                      torch.autograd.forward_ad.make_dual(a, da))
        _, tangent = torch.autograd.forward_ad.unpack_dual(y)
    eps = 1e-3
    reference = (oracle(x+eps*dx, a+eps*da, axis=1, kernel_len=5, gain=.5)
                 - oracle(x-eps*dx, a-eps*da, axis=1, kernel_len=5, gain=.5)) / (2*eps)
    torch.testing.assert_close(tangent, reference, rtol=2e-3, atol=3e-4)
    both_x, both_a = x.clone().requires_grad_(), a.clone().requires_grad_()
    joint = torch.autograd.grad(operation(both_x, both_a).sum(), (both_x, both_a))
    only_x, only_a = x.clone().requires_grad_(), a.clone().requires_grad_()
    assert torch.equal(torch.autograd.grad(operation(only_x, a).sum(), only_x)[0], joint[0])
    assert torch.equal(torch.autograd.grad(operation(x, only_a).sum(), only_a)[0], joint[1])


def test_native_buffers_are_bit_exact_and_independently_owned():
    kernel = st.FractionalGlKernel(kernel_len=5)
    source = array("f", [.2, .4, .3, -.5, 1., -.1])
    saved = kernel.forward_history_l2_buffer(source, [2, 3], 1, 2., 1.5)
    listed = kernel.forward_history_l2(list(source), [2, 3], 1, 2., 1.5)
    source[0] = 99
    assert saved.output_buffer() == listed.output_buffer()
    upstream = array("f", [1.] * 6)
    gx, ga = saved.vjp_buffer(upstream)
    assert list(memoryview(gx).cast("f")) == listed.vjp(list(upstream))[0]
    assert ga == listed.vjp_alpha(list(upstream))
    assert saved.vjp_input_buffer(upstream) == gx
    assert saved.vjp_alpha_buffer(upstream) == ga
    assert list(memoryview(saved.jvp_buffer(upstream, .2)).cast("f")) == listed.jvp(list(upstream), .2)
    output = saved.output_buffer()
    memoryview(output).cast("f")[1] = 99
    assert saved.output == listed.output
    for bad, error in [(array("d", [1.] * 6), TypeError),
                       (memoryview(source)[::2], BufferError)]:
        with pytest.raises(error):
            kernel.forward_history_l2_buffer(bad, [2, 3], 1, 2.)


def test_tiny_order_and_one_tap_do_not_create_spurious_gradients():
    tiny = torch.nextafter(torch.tensor(0.), torch.tensor(1.)).item()
    x = [1., 0., 0., 0., 0., 0., 0.]
    kernel = st.FractionalGlKernel(kernel_len=7)
    batch = kernel.forward_history_l2(x, [7], 0, tiny)
    nearby = kernel.forward_history_l2(x, [7], 0, 1e-8)
    torch.testing.assert_close(torch.tensor(batch.output), torch.tensor(nearby.output),
                               rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(torch.tensor(batch.jvp([0.] * 7, 1.)),
                               torch.tensor(nearby.jvp([0.] * 7, 1.)), rtol=1e-6, atol=1e-7)
    for alpha in [tiny, 1e-30, .1, 1., 4., 1e38]:
        one = st.FractionalGlKernel(kernel_len=2, step=tiny).forward_history_l2(x, [7], 0, alpha)
        assert one.output == [0., -1., 0., 0., 0., 0., 0.]
        assert one.vjp_alpha([1.] * 7) == 0


def test_step_cancels_and_prefixes_use_the_declared_kernel():
    observed = []
    for step in [1e-40, .7, 1., 1e38]:
        kernel = st.FractionalGlKernel(kernel_len=7, step=step)
        batch = kernel.forward_history_l2([1.] + [0.] * 6, [7], 0, 4., 1.5)
        assert sum(c*c for c in batch.output) == pytest.approx(1.5**2, rel=2e-7)
        short = kernel.forward_history_l2([1., 0.], [2], 0, 4., 1.5)
        assert short.output == batch.output[:2]
        observed.append((batch.output, batch.vjp([.5] * 7), batch.jvp([.2] * 7, .3)))
    assert all(item == observed[0] for item in observed)


@pytest.mark.parametrize("shape,kernel_len", [((2, 3, 4), 1), ((2, 1, 4), 8)])
def test_empty_history_has_zero_differentials(shape, kernel_len):
    x, a = torch.ones(shape, requires_grad=True), torch.tensor(1e38, requires_grad=True)
    kernel = st.FractionalGlKernel(kernel_len=kernel_len, step=.1)
    y = st.fractional_gl_history_l2_autograd(x, a, axis=1, kernel=kernel)
    gx, ga = torch.autograd.grad(y.sum(), (x, a))
    assert torch.count_nonzero(y) == torch.count_nonzero(gx) == ga == 0


@pytest.mark.parametrize("gain", [0., -1., float("nan"), float("inf"), 1e-50, 1e50])
def test_invalid_gain_rejected_even_for_empty_history(gain):
    kernel = st.FractionalGlKernel(kernel_len=1)
    with pytest.raises(ValueError):
        kernel.forward_history_l2([1.], [1], 0, 1., gain)
    with pytest.raises(ValueError):
        st.FractionalL2HistoryAdapter(1, gain=gain)


@pytest.mark.parametrize("gain", [True, "1", torch.tensor(1., requires_grad=True)])
def test_python_gain_is_a_constant_number(gain):
    with pytest.raises(TypeError):
        st.fractional_gl_history_l2_autograd(torch.ones(2), torch.tensor(1.), axis=0, gain=gain)


def test_adapter_identity_causality_and_checkpoint_contract():
    adapter = st.FractionalL2HistoryAdapter(3, initial_alpha=2., gain=1.5, kernel_len=7, step=.6)
    x = torch.linspace(-1, 1, 24).reshape(2, 4, 3)
    assert sum(p.numel() for p in adapter.parameters()) == 7
    assert torch.equal(adapter(x), x)
    with torch.no_grad():
        adapter.gate.fill_(.3)
        adapter.local_gate.fill_(-.4)
    expected = (x + .1 * adapter.local_gate.tanh() * x
                + .1 * adapter.gate.tanh() * oracle(x, adapter.log_alpha.exp(), axis=1,
                                                  kernel_len=7, gain=1.5))
    torch.testing.assert_close(adapter(x), expected, rtol=2e-6, atol=2e-6)
    changed = x.clone()
    changed[0, 2:] = 90
    changed[1] = -80
    assert torch.equal(adapter(x)[0, :2], adapter(changed)[0, :2])
    state = copy.deepcopy(adapter.state_dict())
    restored = st.FractionalL2HistoryAdapter(3, kernel_len=2, gain=.5)
    restored.load_state_dict(state)
    assert restored.get_extra_state() == adapter.get_extra_state()
    assert restored.gain == 1.5 and torch.equal(adapter(x), restored(x))
    for raw in [st.FractionalMemoryAdapter(3), st.FractionalHistoryAdapter(3)]:
        with pytest.raises(ValueError, match="incompatible"):
            raw.load_state_dict(state)
        with pytest.raises(ValueError, match="incompatible"):
            restored.load_state_dict(raw.state_dict())
    before = restored.get_extra_state()
    for change in [{"gain": float("nan")}, {"schema": "raw"}, {"kernel": {"kernel_len": 0}}]:
        with pytest.raises(ValueError):
            restored.set_extra_state({**before, **change})
        assert restored.get_extra_state() == before
