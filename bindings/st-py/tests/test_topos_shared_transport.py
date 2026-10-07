"""Unexpanded feature gates share the Rust recurrence and wide row-sum contract."""

from array import array
from concurrent.futures import ThreadPoolExecutor
import math

import pytest
import spiraltorch as st


def packed(values):
    return array("f", values).tobytes()


@pytest.mark.parametrize("shape", [(0, 3), (1, 1), (7, 17), (256, 768)])
@pytest.mark.parametrize("iterations,porosity", [(1, 0.), (5, .3), (64, 1.)])
def test_shared_list_buffer_matches_expanded_tape_and_wide_reduction(shape, iterations, porosity):
    rows, features = shape
    x = array("f", ((i * 37 % 127) / 53 - 1 for i in range(rows * features)))
    gate = array("f", ((i % 7 - 3) * .7 for i in range(features)))
    dy = array("f", ((i * 13 % 31) / 31 - .5 for i in range(len(x))))
    kernel = st.ToposResonatorKernel(iterations=iterations, porosity=porosity)
    legacy = kernel.capture_buffer(x, gate * rows, *shape)
    dx, expanded_dg = legacy.vjp(dy)
    sums = [0.] * features
    for i, value in enumerate(expanded_dg):
        sums[i % features] += value
    expected = packed(dx), packed(sums)
    assert legacy.gate_layout == "elementwise" and legacy.gate_values == len(x)
    for buffers in (False, True):
        capture = kernel.capture_shared_rows_buffer if buffers else kernel.capture_shared_rows
        forward = kernel.forward_shared_rows_buffer if buffers else kernel.forward_shared_rows
        batch = capture(x, gate, *shape)
        assert batch.gate_layout == "shared_rows" and batch.gate_values == features
        assert batch.output_buffer() == legacy.output_buffer()
        output = forward(x, gate, *shape)
        assert (bytes(output) if buffers else packed(output)) == batch.output_buffer()
        assert batch.audit_json() == legacy.audit_json()
        assert batch.vjp_buffer(dy) == expected
        assert tuple(packed(g) for g in batch.vjp(dy)) == expected


def test_shared_ownership_concurrent_pullbacks_and_recovery():
    kernel = st.ToposResonatorKernel(porosity=.3)
    source = array("f", [.2, -.3, .8, 1.2, -.9, .1])
    x, gate = memoryview(source), memoryview(source)[3:]
    batch = kernel.capture_shared_rows_buffer(x, gate, 2, 3)
    dy = array("f", [.3] * 6)
    output, gradients = batch.output_buffer(), batch.vjp_buffer(dy)
    source[:] = array("f", [float("nan")] * 6)
    del kernel
    batch.output_buffer()[:] = bytes(len(output))
    batch.vjp_buffer(dy)[1][:] = bytes(12)
    for bad in (array("f"), array("f", [float("nan")] * 6), dy * 2):
        with pytest.raises(ValueError):
            batch.vjp_buffer(bad)
    with ThreadPoolExecutor(max_workers=4) as workers:
        assert all(g == gradients for g in workers.map(lambda _: batch.vjp_buffer(dy), range(12)))
    assert batch.output_buffer() == output


@pytest.mark.parametrize("buffers", [False, True])
def test_shared_shape_budget_and_finite_guards(buffers):
    kernel = st.ToposResonatorKernel(max_values=4)
    for name in ("capture_shared_rows", "forward_shared_rows"):
        method = getattr(kernel, name + ("_buffer" if buffers else ""))
        for x, gate, rows, features in (
            ([], [], 0, 0), ([], [1.] * 5, 0, 5),
            ([1.] * 5, [1.], 5, 1), ([1.] * 4, [1.] * 4, 2, 2),
            ([], [math.nan], 0, 1), ([math.inf], [1.], 1, 1),
            ([1.], [1.], 2**63, 2**63),
        ):
            with pytest.raises((ValueError, OverflowError)):
                method(array("f", x), array("f", gate), rows, features)
        if buffers:
            for bad in (b"1234", array("d", [1.])):
                with pytest.raises(TypeError):
                    method(array("f", [1.]), bad, 1, 1)
            with pytest.raises(BufferError):
                method(array("f", [1.] * 4), memoryview(array("f", [1.] * 4))[::2], 2, 2)


def test_wide_sum_cancels_without_silently_accepting_overflow():
    maximum = float.fromhex("0x1.fffffep+127")
    kernel = st.ToposResonatorKernel(coupling=0., iterations=1)
    batch = kernel.capture_shared_rows([maximum, maximum, -maximum, -maximum, 1.], [0.], 5, 1)
    assert batch.vjp([1.] * 5)[1] == [1.]
    with pytest.raises(ValueError, match="grad_gate_sum"):
        batch.vjp([1., 1., 0., 0., 0.])
    assert batch.vjp([1.] * 5)[1] == [1.]


@pytest.mark.parametrize("gate_shape", [(3,), (1, 3), (1, 1, 3), (), (2, 1, 1), (1, 4, 3)])
@pytest.mark.parametrize("buffers", [False, True])
def test_torch_dispatch_transport_and_broadcast_gradient(monkeypatch, gate_shape, buffers):
    torch = pytest.importorskip("torch")
    from spiraltorch import geometry_autograd as bridge
    monkeypatch.setattr(bridge, "_buffer_transport_available", lambda: buffers)
    x = torch._neg_view(torch.linspace(-1.3, 1.4, 24).reshape(2, 3, 4).transpose(1, 2)).requires_grad_()
    gate = torch._neg_view(torch.full(gate_shape, .8)).requires_grad_()
    kernel = st.ToposResonatorKernel(porosity=.3)
    shared = gate_shape in ((3,), (1, 3), (1, 1, 3))
    transport_name = "_buffer_values" if buffers else "_values"
    original = getattr(bridge, transport_name)
    calls = []

    def observe(value):
        calls.append(value.numel())
        return original(value)

    monkeypatch.setattr(bridge, transport_name, observe)
    y = st.topos_resonator_autograd(x, gate, kernel=kernel)
    assert calls == [24, 3 if shared else 24]
    assert y.grad_fn.snapshot.gate_layout == ("shared_rows" if shared else "elementwise")
    assert y.grad_fn.snapshot.gate_values == (3 if shared else 24)
    dy = torch.linspace(-.5, .6, 24).reshape_as(x)
    dx, dg = torch.autograd.grad(y, (x, gate), dy)
    assert calls == [24, 3 if shared else 24, 24]
    legacy = kernel.capture(x.detach().flatten().tolist(), gate.detach().expand_as(x).flatten().tolist(), 8, 3)
    lx, lg = legacy.vjp(dy.flatten().tolist())
    torch.testing.assert_close(dx, torch.tensor(lx).reshape_as(x), rtol=0, atol=0)
    gradient = torch.tensor(lg).reshape_as(x)
    reference = gradient.double().sum_to_size(gate.shape).float() if shared else gradient.sum_to_size(gate.shape)
    torch.testing.assert_close(dg, reference, rtol=0, atol=0)
    # Legacy f32 accumulation is numerically close, not promised bit-identical.
    torch.testing.assert_close(dg, gradient.sum_to_size(gate.shape), rtol=5e-5, atol=3e-6)
    calls.clear()
    with torch.no_grad():
        plain = st.topos_resonator_autograd(x, gate, kernel=kernel)
    assert calls == [24, 3 if shared else 24]
    assert torch.equal(plain, y)


def test_empty_budget_and_gate_version_checks():
    torch = pytest.importorskip("torch")
    x, gate = torch.empty(0, 3, requires_grad=True), torch.ones(3, requires_grad=True)
    y = st.topos_resonator_autograd(x, gate)
    dx, dg = torch.autograd.grad(y.sum(), (x, gate))
    assert dx.shape == x.shape and torch.equal(dg, torch.zeros_like(gate))
    with pytest.raises(ValueError, match="budget"):
        st.topos_resonator_autograd(x, gate, kernel=st.ToposResonatorKernel(max_values=2))
    y = st.topos_resonator_autograd(torch.ones(2, 3), gate)
    with torch.no_grad():
        gate.add_(.1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.sum().backward()
