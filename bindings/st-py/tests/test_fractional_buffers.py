"""Bulk transport preserves Rust semantics and never aliases a saved snapshot."""

from array import array
from concurrent.futures import ThreadPoolExecutor
import gc
from pathlib import Path
import struct
import subprocess
import sys
import textwrap

import pytest
import spiraltorch as st


def packed(values):
    return array("f", values).tobytes()


@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_native_buffer_forward_and_all_differentials_are_bit_exact(history, axis):
    shape = [2, 3, 65]
    size = 390
    values = array("f", ((i * 37 % 127) / 97 - .65 for i in range(size)))
    direction = array("f", ((i * 13 % 31) / 31 - .5 for i in range(size)))
    nd = memoryview(values).cast("B").cast("f", shape)
    kernel = st.FractionalGlKernel(kernel_len=5, step=.7)
    name = "forward_history" if history else "forward"
    reference = getattr(kernel, name)(list(values), shape, axis, .6)
    saved = getattr(kernel, name + "_buffer")(nd, shape, axis, .6)
    assert type(saved.output_buffer()) is bytearray
    assert saved.output_buffer() == packed(reference.output)
    dx, da = saved.vjp_buffer(direction)
    expected_dx, expected_da = reference.vjp(list(direction))
    assert dx == saved.vjp_input_buffer(direction) == packed(expected_dx)
    assert struct.pack("=f", da) == struct.pack("=f", expected_da)
    assert da == saved.vjp_alpha_buffer(direction)
    assert saved.jvp_buffer(direction, .3) == packed(reference.jvp(list(direction), .3))


@pytest.mark.parametrize("history", [False, True])
def test_buffer_copies_accept_readonly_unaligned_and_signed_subnormal_values(history):
    values = array("f", [0., -0., struct.unpack("=f", struct.pack("=I", 1))[0], -.25, 1.])
    kernel = st.FractionalGlKernel(kernel_len=8, step=.7)
    name = "forward_history" if history else "forward"
    expected = packed(getattr(kernel, name)(list(values), [5], 0, 1.).output)
    unaligned = memoryview(b"!" + values.tobytes())[1:].cast("f")
    for view in (memoryview(values).toreadonly(), unaligned):
        assert getattr(kernel, name + "_buffer")(view, [5], 0, 1.).output_buffer() == expected


def test_buffer_owners_and_snapshots_are_independent_and_exports_are_released():
    source = array("f", [1., 2., 3., 4.])
    direction = array("f", [1., -.2, .3, -.4])
    saved = st.FractionalGlKernel(kernel_len=3).forward_buffer(source, [4], 0, .6)
    output, gradients = saved.output_buffer(), saved.vjp_buffer(direction)
    expected = bytes(output)
    source[:] = array("f", [0.] * 4)
    source.append(5.)
    direction.append(0.)
    output[:] = bytes(len(output))
    gradients[0][:] = bytes(len(gradients[0]))
    assert saved.output_buffer() == expected
    survivors = saved.output_buffer(), saved.vjp_input_buffer(array("f", [1.] * 4))
    del saved, source, direction
    gc.collect()
    assert survivors[0] == expected
    assert len(survivors[1]) == 16


@pytest.mark.parametrize("bad", [b"1234", bytearray(4), array("d", [1.]), array("i", [1])])
def test_buffers_reject_untyped_bytes_and_non_f32_formats(bad):
    with pytest.raises(TypeError, match="native-endian float32"):
        st.FractionalGlKernel().forward_buffer(bad, [1], 0, .5)


def test_buffer_layout_lengths_shapes_and_budgets_are_checked():
    values = array("f", [1., 2., 3., 4.])
    with pytest.raises(BufferError, match="C-contiguous"):
        st.FractionalGlKernel().forward_buffer(memoryview(values)[::2], [2], 0, .5)
    with pytest.raises(BufferError, match="C-contiguous"):
        st.FractionalGlKernel().forward_buffer(memoryview(values)[::-1], [4], 0, .5)
    with pytest.raises(ValueError, match="budget"):
        st.FractionalGlKernel(kernel_len=1, max_values=1, max_products=1).forward_buffer(values, [4], 0, .5)
    kernel = st.FractionalGlKernel()
    for shape, axis in (([3], 0), ([1] * 17, 0), ([4], 1), ([0], 0)):
        with pytest.raises(ValueError):
            kernel.forward_buffer(values, shape, axis, .5)
    snapshot = kernel.forward_buffer(values, [4], 0, .5)
    for operation in (snapshot.vjp_buffer, snapshot.vjp_input_buffer, snapshot.vjp_alpha_buffer,
                      lambda value: snapshot.jvp_buffer(value, .2)):
        for bad in (array("f"), array("f", [1.] * 3), array("f", [1.] * 5)):
            with pytest.raises(ValueError):
                operation(bad)


def test_numpy_endianness_and_fortran_layout_are_not_silently_reinterpreted():
    np = pytest.importorskip("numpy")
    kernel = st.FractionalGlKernel()
    nonnative = np.array([1., 2.], dtype=">f4" if sys.byteorder == "little" else "<f4")
    with pytest.raises(TypeError, match="native-endian"):
        kernel.forward_buffer(nonnative, [2], 0, .5)
    with pytest.raises(BufferError, match="C-contiguous"):
        kernel.forward_buffer(np.asfortranarray(np.ones((2, 3), dtype=np.float32)), [2, 3], 1, .5)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_buffer_nonfinite_data_and_tangents_reach_rust_validation(bad):
    kernel = st.FractionalGlKernel(kernel_len=1, step=.1)
    for method in (kernel.forward_buffer, kernel.forward_history_buffer):
        with pytest.raises(ValueError):
            method(array("f", [bad]), [1], 0, .5)
    saved = kernel.forward_history_buffer(array("f", [1.]), [1], 0, 100.)
    assert saved.output_buffer() == packed([0.])
    for method in (saved.vjp_buffer, saved.vjp_input_buffer, saved.vjp_alpha_buffer,
                   lambda value: saved.jvp_buffer(value, 0.)):
        with pytest.raises(ValueError):
            method(array("f", [bad]))
    with pytest.raises(ValueError):
        saved.jvp_buffer(array("f", [0.]), bad)


def test_buffer_snapshots_support_concurrent_readonly_pullbacks():
    values = array("f", [i / 31 for i in range(64)])
    saved = st.FractionalGlKernel(kernel_len=8).forward_buffer(values, [8, 8], 0, .5)
    expected = saved.vjp_buffer(values)
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(lambda _: saved.vjp_buffer(values), range(12)))
    assert all(result == expected for result in results)


@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_torch_buffer_and_list_routes_preserve_reverse_and_forward_ad(history, axis, monkeypatch):
    torch = pytest.importorskip("torch")
    from spiraltorch import fractional_autograd as bridge
    if not bridge._buffer_transport_available():
        pytest.skip("Torch/NumPy buffer interop is not installed")
    operation = st.fractional_gl_history_autograd if history else st.fractional_gl_autograd
    kernel = st.FractionalGlKernel(kernel_len=5, step=.7)
    value = torch._neg_view(torch.linspace(-.8, .9, 24).reshape(2, 3, 4).transpose(0, 2))

    def run(buffers):
        monkeypatch.setattr(bridge, "_buffer_transport_available", lambda: buffers)
        x = value.detach().requires_grad_()
        alpha = torch.tensor(.6, requires_grad=True)
        y = operation(x, alpha, axis=axis, kernel=kernel)
        assert y.grad_fn.buffer_transport is buffers
        gradients = torch.autograd.grad(y, (x, alpha), x.detach().cos())
        with torch.autograd.forward_ad.dual_level():
            dx = torch.autograd.forward_ad.make_dual(value, value.cos())
            da = torch.autograd.forward_ad.make_dual(alpha.detach(), torch.tensor(.3))
            primal, tangent = torch.autograd.forward_ad.unpack_dual(operation(dx, da, axis=axis, kernel=kernel))
        return y.detach(), *gradients, primal, tangent

    reference, actual = run(False), run(True)
    for left, right in zip(reference, actual):
        assert torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                           right.contiguous().reshape(-1).view(torch.uint8))


def test_torch_output_keeps_buffer_export_alive_without_aliasing_snapshot():
    torch = pytest.importorskip("torch")
    from spiraltorch import fractional_autograd as bridge
    saved = st.FractionalGlKernel().forward_buffer(array("f", [1., 2.]), [2], 0, .5)
    raw = saved.output_buffer()
    output = bridge._transport_output(raw, torch.zeros(2), True)
    with pytest.raises(BufferError):
        raw.extend(b"0000")
    output[0] = 42.
    assert saved.output[0] == 1.
    del saved, raw
    gc.collect()
    assert output[0] == 42.


def test_torch_sequence_fallback_without_numpy_in_a_fresh_process():
    code = textwrap.dedent("""
        import sys
        sys.path.insert(0, sys.argv[1])
        sys.modules['numpy'] = None
        import torch
        import spiraltorch as st
        from spiraltorch import fractional_autograd as bridge
        assert not bridge._buffer_transport_available()
        value = torch.tensor([1., 2., 3.], requires_grad=True)
        alpha = torch.tensor(.5, requires_grad=True)
        result = st.fractional_gl_autograd(value, alpha, axis=0)
        assert result.grad_fn.buffer_transport is False
        gradients = torch.autograd.grad(result.sum(), (value, alpha))
        assert all(torch.isfinite(v).all() for v in gradients)
    """)
    completed = subprocess.run([sys.executable, "-I", "-c", code,
                                str(Path(st.__file__).resolve().parent.parent)],
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
