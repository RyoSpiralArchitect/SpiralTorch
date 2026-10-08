"""Owned bulk buffers preserve legacy and radius WaveGate learning contracts."""

from array import array
from concurrent.futures import ThreadPoolExecutor
import gc
from pathlib import Path
import struct
import subprocess
import sys

import pytest
import spiraltorch as st


def packed(values):
    return array("f", values).tobytes()


def capture(kernel, values, gate, bias, rows, cols, radius, buffers):
    name = "forward" if radius is None else "forward_with_log_radius"
    args = (values, gate, bias, rows, cols)
    return getattr(kernel, name + ("_buffer" if buffers else ""))(*args, *([] if radius is None else [radius]))


@pytest.mark.parametrize("radius", [None, -2., 0., 2.])
@pytest.mark.parametrize("shape", [(0, 3), (1, 1), (32, 65), (256, 768)])
def test_output_and_joint_gradients_are_bit_exact(radius, shape):
    rows, cols = shape
    x = array("f", ((i * 37 % 127) / 53 - 1 for i in range(rows * cols)))
    gate = array("f", ((i % 7 - 3) * .7 for i in range(cols)))
    bias = array("f", ((i % 5 - 2) * .1 for i in range(cols)))
    dy = array("f", ((i * 13 % 31) / 31 - .5 for i in range(rows * cols)))
    kernel = st.WaveGateKernel(curvature=-.7, porosity=.2)
    reference = capture(kernel, list(x), list(gate), list(bias), rows, cols, radius, False)
    saved = capture(kernel, x, gate, bias, rows, cols, radius, True)
    assert type(saved.output_buffer()) is bytearray
    assert saved.output_buffer() == packed(reference.output)
    assert saved.conditioning_json() == reference.conditioning_json()
    assert saved.vjp_buffer(dy) == tuple(packed(g) for g in reference.vjp(list(dy)))
    if radius is not None:
        actual = saved.vjp_with_log_radius_buffer(dy)
        expected = reference.vjp_with_log_radius(list(dy))
        assert actual[:3] == tuple(packed(g) for g in expected[:3])
        assert struct.pack("=f", actual[3]) == struct.pack("=f", expected[3])


@pytest.mark.parametrize("radius", [None, 0.7])
def test_snapshot_ownership_readonly_unaligned_and_concurrent_vjps(radius):
    x = array("f", [0., -0., 1e-40, -.3, .5, 1.3])
    gate, bias = array("f", [1.4, -.8, .3]), array("f", [0., -.1, .2])
    dy = array("f", [.3] * 6)
    kernel = st.WaveGateKernel()
    saved = capture(kernel, memoryview(b"!" + x.tobytes())[1:].cast("f"),
                    memoryview(gate).toreadonly(), bias, 2, 3, radius, True)
    expected, gradients = saved.output_buffer(), saved.vjp_buffer(dy)
    for source in (x, gate, bias):
        source[:] = array("f", [0.] * len(source))
        source.append(4.)
    output = saved.output_buffer()
    output[:] = bytes(len(output))
    changed = saved.vjp_buffer(dy)
    changed[0][:] = bytes(len(changed[0]))
    assert saved.output_buffer() == expected
    with ThreadPoolExecutor(max_workers=4) as workers:
        assert all(result == gradients for result in workers.map(lambda _: saved.vjp_buffer(dy), range(12)))
    survivors = saved.output_buffer(), saved.vjp_buffer(dy)
    del saved, kernel
    gc.collect()
    assert survivors == (expected, gradients)


@pytest.mark.parametrize("bad", [b"1234", bytearray(4), array("d", [1.]), array("i", [1])])
def test_all_buffers_check_f32_format(bad):
    kernel = st.WaveGateKernel()
    good = array("f", [1.])
    for index in range(3):
        args = [good, good, good]
        args[index] = bad
        with pytest.raises(TypeError, match="native-endian float32"):
            kernel.forward_buffer(*args, 1, 1)
    saved = kernel.forward_with_log_radius_buffer(good, good, good, 1, 1, .5)
    for method in (saved.vjp_buffer, saved.vjp_with_log_radius_buffer):
        with pytest.raises(TypeError, match="native-endian float32"):
            method(bad)


def test_layout_shape_domain_and_budget_guards():
    kernel = st.WaveGateKernel(max_values=4)
    x = array("f", [1., 2., 3., 4.])
    gate = array("f", [1., 1.])
    for view in (memoryview(x)[::2], memoryview(x)[::-1]):
        with pytest.raises(BufferError, match="contiguous"):
            kernel.forward_buffer(view, gate, gate, 1, 2)
    for rows, cols in [(1, 2), (2, 1), (0, 0), (2**63, 2**63)]:
        with pytest.raises((ValueError, OverflowError)):
            kernel.forward_buffer(x, gate, gate, rows, cols)
    with pytest.raises(ValueError, match="budget"):
        kernel.forward_buffer(x * 2, gate, gate, 4, 2)
    with pytest.raises(ValueError):
        kernel.forward_buffer(x, gate[:1], gate, 2, 2)
    saved = kernel.forward_buffer(x, gate, gate, 2, 2)
    with pytest.raises(ValueError):
        saved.vjp_with_log_radius_buffer(x)
    for bad in (array("f"), x[:3], x + gate):
        with pytest.raises(ValueError):
            saved.vjp_buffer(bad)
    for bad in (float("nan"), float("inf"), -float("inf")):
        for index in range(3):
            args = [x, gate, gate]
            args[index] = array("f", [bad] * len(args[index]))
            with pytest.raises(ValueError):
                kernel.forward_buffer(*args, 2, 2)
        with pytest.raises(ValueError):
            saved.vjp_buffer(array("f", [bad] * 4))
        with pytest.raises(ValueError):
            kernel.forward_with_log_radius_buffer(x, gate, gate, 2, 2, bad)


def test_numpy_endian_and_fortran_inputs_are_rejected():
    np = pytest.importorskip("numpy")
    kernel = st.WaveGateKernel()
    gate = np.ones(3, dtype=np.float32)
    with pytest.raises(BufferError, match="C-contiguous"):
        kernel.forward_buffer(np.asfortranarray(np.ones((2, 3), dtype=np.float32)), gate, gate, 2, 3)
    with pytest.raises(TypeError, match="native-endian"):
        kernel.forward_buffer(np.ones(6, dtype=">f4" if sys.byteorder == "little" else "<f4"), gate, gate, 2, 3)


@pytest.mark.parametrize("radius", [None, 0.7])
@pytest.mark.parametrize("empty", [False, True])
def test_torch_transport_handles_negative_views_empty_and_all_gradients(monkeypatch, radius, empty):
    torch = pytest.importorskip("torch")
    from spiraltorch import geometry_autograd as bridge
    value = torch._neg_view(torch.linspace(-1.3, 1.4, 24).reshape(2, 3, 4).transpose(1, 2))
    if empty:
        value = value[:0]
    results = []
    for buffers in (False, True):
        monkeypatch.setattr(bridge, "_buffer_transport_available", lambda: buffers)
        inputs = [value.detach().requires_grad_(), torch.tensor([1.4, -.8, .3], requires_grad=True),
                  torch.tensor([.1, -.1, .2], requires_grad=True)]
        if radius is not None:
            inputs.append(torch.tensor(radius, requires_grad=True))
        output = st.wave_gate_autograd(*inputs[:3], log_radius=inputs[3] if radius is not None else None)
        assert output.grad_fn.buffer_transport is buffers
        results.append((output.detach(), *torch.autograd.grad(output.sum(), inputs)))
    for actual, expected in zip(*results):
        assert actual.contiguous().numpy().tobytes() == expected.contiguous().numpy().tobytes()


def test_torch_sequence_fallback_without_numpy():
    code = """
import sys
sys.path.insert(0, sys.argv[1])
sys.modules['numpy'] = None
import torch
import spiraltorch as st
from spiraltorch import geometry_autograd as bridge
assert not bridge._buffer_transport_available()
for radius in (None, .5):
    adapter = st.WaveGateAdapter(2, log_radius=radius, learnable_radius=radius is not None)
    result = adapter(torch.tensor([[.1, .2]]))
    result.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in adapter.parameters())
"""
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", code,
                                str(Path(st.__file__).resolve().parent.parent)],
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
