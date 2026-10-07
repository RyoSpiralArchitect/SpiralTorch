"""Rust-owned Topos tape: exact legacy derivatives, independent ownership."""

from array import array
from concurrent.futures import ThreadPoolExecutor
import gc
import json
from pathlib import Path
import subprocess
import sys

import pytest
import spiraltorch as st


def packed(values):
    return array("f", values).tobytes()


@pytest.mark.parametrize("shape", [(0, 3), (1, 1), (7, 17), (256, 768)])
@pytest.mark.parametrize("iterations,coupling", [(1, 0.), (4, .25), (16, .75)])
@pytest.mark.parametrize("porosity", [0., .2])
def test_list_buffer_capture_and_recomputed_bits_match(shape, iterations, coupling, porosity):
    rows, features = shape
    x = array("f", ((i * 37 % 127) / 53 - 1 for i in range(rows * features)))
    gate = array("f", ((i % 7 - 3) * .7 for i in range(len(x))))
    dy = array("f", ((i * 13 % 31) / 31 - .5 for i in range(len(x))))
    kernel = st.ToposResonatorKernel(iterations=iterations, coupling=coupling, porosity=porosity)
    expected = packed(kernel.forward(list(x), list(gate), *shape))
    gradients = tuple(packed(g) for g in kernel.backward(list(x), list(gate), list(dy), *shape))
    batch = kernel.capture_buffer(x, gate, *shape)
    listed = kernel.capture(list(x), list(gate), *shape)
    assert type(batch.output_buffer()) is bytearray
    assert batch.output_buffer() == packed(batch.output) == expected == kernel.forward_buffer(x, gate, *shape)
    assert batch.audit_json() == listed.audit_json()
    assert isinstance(json.loads(batch.audit_json()), dict)
    assert batch.vjp_buffer(dy) == gradients == kernel.backward_buffer(x, gate, dy, *shape)
    assert tuple(packed(g) for g in batch.vjp(list(dy))) == gradients


def test_owned_tape_survives_mutation_drop_and_concurrent_vjps():
    kernel = st.ToposResonatorKernel(porosity=.2)
    x, gate, dy = array("f", [0., -0., 1e-40, -.3, .5, 1.3]), array("f", [1.] * 6), array("f", [.3] * 6)
    batch = kernel.capture_buffer(memoryview(b"!" + x.tobytes())[1:].cast("f"), memoryview(gate).toreadonly(), 2, 3)
    output, gradients = batch.output_buffer(), batch.vjp_buffer(dy)
    for source in (x, gate):
        source[:] = array("f", [float("nan")] * len(source))
        source.append(4.)
    del kernel
    batch.output_buffer()[:] = bytes(len(output))
    batch.vjp_buffer(dy)[0][:] = bytes(len(output))
    assert batch.output_buffer() == output
    with ThreadPoolExecutor(max_workers=4) as workers:
        assert all(g == gradients for g in workers.map(lambda _: batch.vjp_buffer(dy), range(12)))
    survivors = batch.output_buffer(), batch.vjp_buffer(dy)
    del batch
    gc.collect()
    assert survivors == (output, gradients)


@pytest.mark.parametrize("buffers", [False, True])
def test_capture_of_aliased_sources_remains_independent_after_failure_and_drop(buffers):
    kernel = st.ToposResonatorKernel(porosity=.2)
    values = array("f", [0., -0., 1e-40, -.3, .5, 1.3])
    if not buffers:
        values = list(values)
    capture = kernel.capture_buffer if buffers else kernel.capture
    batch = capture(values, values, 2, 3)
    upstream = array("f", [.3] * 6)
    expected_output, expected_audit = batch.output_buffer(), batch.audit_json()
    expected_gradients = batch.vjp_buffer(upstream)
    values[:] = array("f", [float("nan")] * 6) if buffers else [float("nan")] * 6
    values.append(4.)
    with pytest.raises(ValueError):
        capture(values, values, 1, 7)
    del capture, kernel, values
    gc.collect()
    assert batch.output_buffer() == expected_output
    assert batch.audit_json() == expected_audit
    assert batch.vjp_buffer(upstream) == expected_gradients


@pytest.mark.parametrize("bad", [b"1234", bytearray(4), array("d", [1.]), array("i", [1])])
def test_buffers_reject_non_f32(bad):
    kernel, good = st.ToposResonatorKernel(), array("f", [1.])
    for method in (kernel.capture_buffer, kernel.forward_buffer):
        for args in ((bad, good), (good, bad)):
            with pytest.raises(TypeError, match="native-endian float32"):
                method(*args, 1, 1)
    with pytest.raises(TypeError, match="native-endian float32"):
        kernel.capture_buffer(good, good, 1, 1).vjp_buffer(bad)


def test_shape_domain_layout_budget_and_failed_vjp_do_not_corrupt_tape():
    kernel = st.ToposResonatorKernel(max_values=4)
    x = array("f", [1., 2., 3., 4.])
    for method in (kernel.capture_buffer, kernel.forward_buffer):
        for bad in (memoryview(x)[::2], memoryview(x)[::-1]):
            with pytest.raises(BufferError, match="contiguous"):
                method(bad, x, 2, 2)
        for rows, cols in ((1, 2), (0, 0), (2**63, 2**63)):
            with pytest.raises((ValueError, OverflowError)):
                method(x, x, rows, cols)
        with pytest.raises(ValueError, match="budget"):
            method(x * 2, x * 2, 4, 2)
        with pytest.raises(ValueError):
            method(x, x[:1], 2, 2)
        for invalid in (float("nan"), float("inf"), -float("inf")):
            bad = array("f", [invalid] * 4)
            for inputs in ((bad, x), (x, bad)):
                with pytest.raises(ValueError):
                    method(*inputs, 2, 2)
    batch = kernel.capture_buffer(x, x, 2, 2)
    expected = batch.vjp_buffer(x)
    for bad in (x[:3], x * 2, array("f", [float("nan")] * 4)):
        with pytest.raises(ValueError):
            batch.vjp_buffer(bad)
    assert batch.vjp_buffer(x) == expected
    batch = kernel.capture([1.], [0.], 1, 1)
    with pytest.raises(ValueError):
        batch.vjp([3.4028234663852886e38])
    assert batch.vjp([1.]) == kernel.backward([1.], [0.], [1.], 1, 1)


def test_endian_and_fortran_layout_rejected():
    np = pytest.importorskip("numpy")
    kernel = st.ToposResonatorKernel()
    good = np.ones((2, 3), dtype=np.float32)
    for bad, error in ((np.asfortranarray(good), BufferError),
                       (good.astype(">f4" if sys.byteorder == "little" else "<f4"), TypeError)):
        with pytest.raises(error):
            kernel.capture_buffer(bad, good, 2, 3)


@pytest.mark.parametrize("empty", [False, True])
def test_broadcast_negative_views_and_captured_backward_without_reupload(monkeypatch, empty):
    torch = pytest.importorskip("torch")
    from spiraltorch import geometry_autograd as bridge
    value = torch._neg_view(torch.linspace(-1.3, 1.4, 24).reshape(2, 3, 4).transpose(1, 2))
    if empty:
        value = value[:0]
    results = []
    for buffers in (False, True):
        monkeypatch.setattr(bridge, "_buffer_transport_available", lambda: buffers)
        inputs = (value.detach().requires_grad_(), torch.tensor([1.4, -.8, .3], requires_grad=True))
        output = st.topos_resonator_autograd(*inputs)
        assert output.grad_fn.buffer_transport is buffers
        assert isinstance(output.grad_fn.snapshot, st.ToposResonatorLearningBatch)
        original = bridge._buffer_values if buffers else bridge._values
        calls = []

        def observe(tensor):
            calls.append(tensor.shape)
            return original(tensor)

        with monkeypatch.context() as scoped:
            scoped.setattr(bridge, "_buffer_values" if buffers else "_values", observe)
            gradients = torch.autograd.grad(output.sum(), inputs)
        assert calls == [value.shape]
        results.append((output.detach(), *gradients))
    for a, b in zip(*results):
        assert a.contiguous().numpy().tobytes() == b.contiguous().numpy().tobytes()


def test_optional_numpy_and_native_public_imports():
    code = """
import sys
sys.path.insert(0, sys.argv[1])
sys.modules['numpy'] = None
import torch
import spiraltorch as st
from spiraltorch import geometry_autograd as bridge
assert 'ToposResonatorLearningBatch' in st.__all__
assert not bridge._buffer_transport_available()
adapter = st.ToposResonatorAdapter(2)
adapter(torch.tensor([[.1, .2]])).sum().backward()
assert torch.isfinite(adapter.gate.grad).all()
"""
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", code,
                                str(Path(st.__file__).resolve().parent.parent)],
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("buffers", [False, True])
def test_inference_does_not_capture_learning_tape(monkeypatch, buffers):
    torch = pytest.importorskip("torch")
    from spiraltorch import geometry_autograd as bridge
    monkeypatch.setattr(bridge, "_buffer_transport_available", lambda: buffers)

    def unexpected(*args):
        raise AssertionError("inference captured an autograd tape")

    monkeypatch.setattr(bridge._ToposResonatorFunction, "apply", unexpected)
    x, gate = torch.tensor([[.2, -.3]]), torch.tensor([.5, .8])
    kernel = st.ToposResonatorKernel()
    expected = packed(kernel.forward(x.flatten().tolist(), gate.tolist(), 1, 2))
    assert st.topos_resonator_autograd(x, gate).numpy().tobytes() == expected
    x.requires_grad_()
    gate.requires_grad_()
    for mode in (torch.no_grad, torch.inference_mode):
        with mode():
            output = st.topos_resonator_autograd(x, gate)
            assert not output.requires_grad and output.numpy().tobytes() == expected
