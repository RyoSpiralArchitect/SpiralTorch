from __future__ import annotations

import pytest


def test_tensor_dlpack_roundtrip(stub_spiraltorch_with_numpy) -> None:
    np = pytest.importorskip("numpy")
    if not hasattr(np, "from_dlpack") or not hasattr(np.ndarray, "__dlpack__"):
        pytest.skip("NumPy lacks DLPack support")

    Tensor = stub_spiraltorch_with_numpy.Tensor
    source = Tensor(shape=(2, 3), data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    capsule = source.to_dlpack()
    restored = Tensor.from_dlpack(capsule)

    assert restored.shape() == source.shape()
    assert restored.backend == "numpy"
    assert restored.tolist() == source.tolist()

    identity = Tensor(
        shape=(3, 3),
        data=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    product = restored.matmul(identity)
    assert product.tolist() == source.tolist()

    array_from_dlpack = np.from_dlpack(source.__dlpack__())
    assert array_from_dlpack.shape == (2, 3)
    assert array_from_dlpack.tolist() == source.tolist()


def test_tensor_dlpack_unavailable(stub_spiraltorch_with_numpy) -> None:
    Tensor = stub_spiraltorch_with_numpy.Tensor
    message = Tensor.DLPACK_UNAVAILABLE_MESSAGE

    original_cells: list[tuple[object, object]] = []

    def override_freevar(func, name: str, value):
        closure = func.__closure__
        if closure is None:
            pytest.fail(f"{func.__qualname__} does not capture {name}")
        mapping = dict(zip(func.__code__.co_freevars, closure))
        if name not in mapping:
            pytest.fail(f"{func.__qualname__} does not have freevar {name}")
        cell = mapping[name]
        original_cells.append((cell, cell.cell_contents))
        cell.cell_contents = value

    try:
        for method in (
            Tensor.from_dlpack,
            Tensor.to_dlpack,
            Tensor.__dlpack__,
            Tensor.__dlpack_device__,
        ):
            override_freevar(method, "NUMPY_AVAILABLE", False)
            override_freevar(method, "_np", None)

        with pytest.raises(RuntimeError) as from_error:
            Tensor.from_dlpack(object())
        assert message in str(from_error.value)

        tensor = Tensor(shape=(1, 1), data=[[1.0]], backend="python")

        with pytest.raises(RuntimeError) as to_error:
            tensor.to_dlpack()
        assert message in str(to_error.value)

        with pytest.raises(RuntimeError) as method_error:
            tensor.__dlpack__()
        assert message in str(method_error.value)

        with pytest.raises(RuntimeError) as device_error:
            tensor.__dlpack_device__()
        assert message in str(device_error.value)
    finally:
        for cell, original in original_cells:
            setattr(cell, "cell_contents", original)
