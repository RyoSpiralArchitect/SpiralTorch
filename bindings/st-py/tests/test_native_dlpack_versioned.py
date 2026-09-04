"""Real native/NumPy/PyTorch interoperability, not the source-only Tensor stub."""

import ctypes as ct
import gc
import importlib
import weakref

import numpy as np
import pytest

import spiraltorch as st


class DLDevice(ct.Structure):
    _fields_ = [("device_type", ct.c_int32), ("device_id", ct.c_int32)]


class DLDataType(ct.Structure):
    _fields_ = [("code", ct.c_uint8), ("bits", ct.c_uint8), ("lanes", ct.c_uint16)]


class DLTensor(ct.Structure):
    _fields_ = [
        ("data", ct.c_void_p), ("device", DLDevice), ("ndim", ct.c_int32),
        ("dtype", DLDataType), ("shape", ct.POINTER(ct.c_int64)),
        ("strides", ct.POINTER(ct.c_int64)), ("byte_offset", ct.c_uint64),
    ]


class DLManagedTensor(ct.Structure):
    _fields_ = [("dl_tensor", DLTensor), ("manager_ctx", ct.c_void_p), ("deleter", ct.c_void_p)]


class DLPackVersion(ct.Structure):
    _fields_ = [("major", ct.c_uint32), ("minor", ct.c_uint32)]


class DLManagedTensorVersioned(ct.Structure):
    _fields_ = [
        ("version", DLPackVersion), ("manager_ctx", ct.c_void_p),
        ("deleter", ct.c_void_p), ("flags", ct.c_uint64), ("dl_tensor", DLTensor),
    ]


_capsule_name = ct.pythonapi.PyCapsule_GetName
_capsule_name.argtypes = [ct.py_object]
_capsule_name.restype = ct.c_char_p
_capsule_pointer = ct.pythonapi.PyCapsule_GetPointer
_capsule_pointer.argtypes = [ct.py_object, ct.c_char_p]
_capsule_pointer.restype = ct.c_void_p
_numpy_versioned = int(np.__version__.split(".")[0]) >= 2


def snapshot(capsule):
    name = _capsule_name(capsule)
    assert name in (b"dltensor", b"dltensor_versioned")
    versioned = name == b"dltensor_versioned"
    header_type = DLManagedTensorVersioned if versioned else DLManagedTensor
    header = ct.cast(_capsule_pointer(capsule, name), ct.POINTER(header_type)).contents
    tensor = header.dl_tensor
    assert tensor.ndim == 2
    return {
        "name": name,
        "version": (header.version.major, header.version.minor) if versioned else None,
        "flags": header.flags if versioned else None,
        "shape": tuple(tensor.shape[i] for i in range(2)),
        "data": tensor.data,
        "offset": tensor.byte_offset,
        "device": (tensor.device.device_type, tensor.device.device_id),
        "dtype": (tensor.dtype.code, tensor.dtype.bits, tensor.dtype.lanes),
    }


@pytest.fixture(autouse=True, scope="module")
def require_native_tensor():
    native = importlib.import_module("spiraltorch.spiraltorch")
    assert isinstance(st.Tensor(1, 1, [0.0]), native.Tensor)


@pytest.mark.parametrize("max_version", [None, (0, 8), (1, 0), (1, 3), (2, 0)])
def test_native_export_negotiates_the_abi(max_version):
    tensor = st.Tensor(1, 2, [3., 5.])
    capsule = tensor.__dlpack__(max_version=max_version, copy=False)
    meta = snapshot(capsule)
    versioned = max_version is not None and max_version >= (1, 0)
    assert meta["name"] == (b"dltensor_versioned" if versioned else b"dltensor")
    assert meta["version"] == ((1, 0) if versioned else None)
    assert meta["flags"] == (0 if versioned else None)
    assert meta["device"] == (1, 0)
    assert meta["dtype"] == (2, 32, 1)
    restored = st.from_dlpack(capsule)
    assert _capsule_name(capsule) == (b"used_dltensor_versioned" if versioned else b"used_dltensor")
    with pytest.raises(ValueError, match="unconsumed"):
        st.from_dlpack(capsule)
    del tensor, capsule
    gc.collect()
    assert restored.tolist() == [[3., 5.]]


@pytest.mark.parametrize("copy", [None, False, True])
@pytest.mark.parametrize("max_version", [None, (1, 0)])
def test_native_export_copy_policy(copy, max_version):
    tensor = st.Tensor(1, 2, [3., 5.])
    shared = tensor.to_dlpack()
    exported = tensor.__dlpack__(copy=copy, max_version=max_version)
    meta = snapshot(exported)
    assert (meta["data"] != snapshot(shared)["data"]) == (copy is True)
    if max_version is not None:
        assert meta["flags"] == (2 if copy is True else 0)
    assert st.from_dlpack(exported).tolist() == [[3., 5.]]


def test_numpy_producer_negotiates_and_outlives_python_owner():
    array = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    owner = weakref.ref(array)
    pointer = array.__array_interface__["data"][0]
    tensor = st.from_dlpack(array)
    exported = tensor.__dlpack__(max_version=(1, 0))
    meta = snapshot(exported)
    assert meta["data"] + meta["offset"] == pointer
    del array, tensor
    gc.collect()
    assert owner() is not None
    restored = st.from_dlpack(exported)
    assert restored.tolist() == [[1., 2.], [3., 4.]]
    del restored, exported
    gc.collect()
    assert owner() is None


@pytest.mark.parametrize("consume", [False, True])
def test_shared_capsule_releases_numpy_owner_once_finished(consume):
    array = np.array([[1., 2.]], dtype=np.float32)
    owner = weakref.ref(array)
    tensor = st.from_dlpack(array)
    capsule = tensor.__dlpack__(max_version=(1, 0))
    restored = st.from_dlpack(capsule) if consume else None
    del array, tensor, capsule
    gc.collect()
    assert (owner() is not None) == consume
    del restored
    gc.collect()
    assert owner() is None


@pytest.mark.skipif(not _numpy_versioned, reason="NumPy 1.x cannot export read-only DLPack")
def test_readonly_import_preserves_flags_and_detaches_before_rust_mutation():
    array = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    array.flags.writeable = False
    tensor = st.from_dlpack(array)
    capsule = tensor.__dlpack__(max_version=(1, 0), copy=False)
    assert snapshot(capsule)["flags"] == 1
    assert snapshot(capsule)["data"] == array.__array_interface__["data"][0]
    view = np.from_dlpack(tensor, copy=False)
    assert not view.flags.writeable
    assert np.shares_memory(array, view)
    with pytest.raises(BufferError, match="requires a copy"):
        tensor.__dlpack__(copy=False)
    legacy = tensor.to_dlpack()
    assert snapshot(legacy)["data"] != array.__array_interface__["data"][0]
    tensor.add_row_inplace([10., 20.])
    assert tensor.tolist() == [[11., 22.], [13., 24.]]
    assert array.tolist() == [[1., 2.], [3., 4.]]
    assert view.tolist() == array.tolist()
    assert st.from_dlpack(capsule).tolist() == array.tolist()


@pytest.mark.skipif(not _numpy_versioned, reason="NumPy 1.x has no explicit DLPack copy parameter")
@pytest.mark.parametrize("readonly", [False, True])
def test_numpy_explicit_copy_is_independent_and_respects_consumer_policy(readonly):
    array = np.array([[1., 2.]], dtype=np.float32)
    array.flags.writeable = not readonly
    tensor = st.from_dlpack(array)
    control = np.from_dlpack(array, copy=True)
    capsule = tensor.__dlpack__(max_version=(1, 0), copy=True)
    assert snapshot(capsule)["flags"] == 2  # IS_COPIED, without READ_ONLY.
    assert snapshot(capsule)["data"] != array.__array_interface__["data"][0]
    native_copy = st.from_dlpack(capsule)
    native_copy.add_row_inplace([10., 20.])
    assert native_copy.tolist() == [[11., 22.]]
    copied = np.from_dlpack(tensor, copy=True)
    # Some NumPy 2.x consumers expose even their own explicit copies read-only.
    assert copied.flags.writeable == control.flags.writeable
    assert not np.shares_memory(array, copied)
    np.testing.assert_array_equal(copied, [[1., 2.]])
    if copied.flags.writeable:
        copied[0, 0] = 42.
    else:
        with pytest.raises(ValueError, match="read-only"):
            copied[0, 0] = 42.
    assert tensor.tolist() == [[1., 2.]]
    array.flags.writeable = True
    array[0, 1] = 99.
    assert copied[0, 1] == control[0, 1] == 2.


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (2, 0), (1, 4), (4, 1)])
def test_numpy_empty_and_singleton_shapes(shape):
    array = np.zeros(shape, dtype=np.float32)
    tensor = st.from_dlpack(array)
    assert tensor.shape() == shape
    result = np.from_dlpack(tensor)
    assert result.shape == shape
    np.testing.assert_array_equal(result, array)
    if not array.size:
        capsule = tensor.__dlpack__(max_version=(1, 0))
        assert snapshot(capsule)["data"] is None


@pytest.mark.parametrize("array", [
    np.zeros((2, 2), dtype=np.float64),
    np.zeros((2, 2), dtype=np.int32),
    np.zeros((2, 3), dtype=np.float32).T,
    np.zeros((2, 3), dtype=np.float32)[:, ::-1],
    np.zeros((4,), dtype=np.float32),
])
def test_invalid_numpy_capsules_are_consumed_even_on_error(array):
    capsule = array.__dlpack__(max_version=(1, 0)) if _numpy_versioned else array.__dlpack__()
    with pytest.raises(ValueError):
        st.from_dlpack(capsule)
    with pytest.raises(ValueError, match="unconsumed"):
        st.from_dlpack(capsule)


def test_legacy_python_producer_body_runs_only_once():
    calls = []

    class LegacyProducer:
        def __dlpack__(self, *, stream=None):
            calls.append(stream)
            return st.Tensor(1, 1, [7.]).to_dlpack()

    assert st.from_dlpack(LegacyProducer()).tolist() == [[7.]]
    assert calls == [None]


def test_rejected_capsule_releases_its_numpy_producer():
    array = np.array([[1., 2.]], dtype=np.float64)
    owner = weakref.ref(array)
    capsule = array.__dlpack__()
    with pytest.raises(ValueError, match="f32"):
        st.from_dlpack(capsule)
    del array
    gc.collect()
    assert owner() is None, "the used capsule must not retain a rejected producer"


@pytest.mark.parametrize("error", [
    RuntimeError("producer-failed"),
    TypeError("producer-failed"),
    TypeError("got an unexpected keyword argument 'max_version'"),
])
def test_errors_inside_python_producers_are_never_retried(error):
    calls = []

    class BrokenProducer:
        def __dlpack__(self, **kwargs):
            calls.append(kwargs)
            raise error

    with pytest.raises(type(error)) as caught:
        st.from_dlpack(BrokenProducer())
    assert caught.value is error
    assert calls == [{"stream": None, "max_version": (1, 0)}]


def test_producer_descriptor_errors_are_preserved():
    class BrokenDescriptor:
        @property
        def __dlpack__(self):
            raise RuntimeError("descriptor-failed")

    with pytest.raises(RuntimeError, match="descriptor-failed"):
        st.from_dlpack(BrokenDescriptor())


@pytest.mark.parametrize("device", [(2, 0), (1, 1), (15, 0)])
def test_export_rejects_unsupported_device_without_silent_transfer(device):
    tensor = st.Tensor(1, 1, [1.])
    with pytest.raises(BufferError, match="CPU device"):
        tensor.__dlpack__(max_version=(1, 0), dl_device=device, copy=True)
    assert tensor.__dlpack_device__() == (1, 0)


def test_torch_storage_is_shared_but_rust_mutation_detaches():
    torch = pytest.importorskip("torch")
    assert hasattr(torch, "Tensor"), "this test requires real PyTorch, not a stub"
    source = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float32)
    tensor = st.from_dlpack(source)
    shared = torch.from_dlpack(tensor)
    assert source.data_ptr() == shared.data_ptr()
    tensor.add_row_inplace([10., 20.])
    assert source.tolist() == [[1., 2.], [3., 4.]]
    assert tensor.tolist() == [[11., 22.], [13., 24.]]
    del source, tensor
    gc.collect()
    assert shared.tolist() == [[1., 2.], [3., 4.]]
