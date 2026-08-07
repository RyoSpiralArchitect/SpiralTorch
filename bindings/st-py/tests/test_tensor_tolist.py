from __future__ import annotations

import importlib
import sys
import types
from typing import Optional

import pytest


def _ensure_torch_stub() -> None:
    if "torch" in sys.modules:
        return
    torch_stub = types.ModuleType("torch")
    torch_stub.autograd = types.SimpleNamespace(Function=object)
    sys.modules["torch"] = torch_stub


def _load_native_module() -> Optional[types.ModuleType]:
    _ensure_torch_stub()
    try:
        native = importlib.import_module("spiraltorch")
    except ModuleNotFoundError:
        return None
    return native if hasattr(native, "Tensor") else None


_NATIVE_MODULE = _load_native_module()


def _expected_matrix(rows: int, cols: int) -> list[list[float]]:
    data = [float(index + 1) for index in range(rows * cols)]
    return [data[r * cols : (r + 1) * cols] for r in range(rows)]


def _expected_range_matrix() -> list[list[float]]:
    return [
        [float(value) for value in range(start, start + 3)]
        for start in (0, 3)
    ]


@pytest.mark.parametrize(
    "rows, cols",
    [
        (0, 0),
        (0, 3),
        (3, 0),
        (2, 3),
    ],
)
def test_stub_tensor_tolist_matches_expected(
    rows: int, cols: int, stub_spiraltorch
) -> None:
    module = stub_spiraltorch
    payload = [float(index + 1) for index in range(rows * cols)]
    tensor = module.Tensor(rows, cols, payload, backend="python")  # type: ignore[attr-defined]
    assert tensor.tolist() == _expected_matrix(rows, cols)

    if "numpy" in module.available_stub_backends():  # type: ignore[attr-defined]
        tensor_np = module.Tensor(rows, cols, payload, backend="numpy")  # type: ignore[attr-defined]
        assert tensor_np.tolist() == _expected_matrix(rows, cols)


def test_stub_tensor_tolist_from_range_is_nested(stub_spiraltorch) -> None:
    module = stub_spiraltorch
    expected = _expected_range_matrix()

    tensor_default = module.Tensor(2, 3, range(6))  # type: ignore[attr-defined]
    assert tensor_default.tolist() == expected

    tensor_python = module.Tensor(2, 3, range(6), backend="python")  # type: ignore[attr-defined]
    assert tensor_python.tolist() == expected
    assert tensor_python.tolist() == tensor_default.tolist()

    if "numpy" in module.available_stub_backends():  # type: ignore[attr-defined]
        tensor_numpy = module.Tensor(2, 3, range(6), backend="numpy")  # type: ignore[attr-defined]
        assert tensor_numpy.tolist() == expected
        assert tensor_numpy.tolist() == tensor_default.tolist()


def test_stub_tensor_tolist_range_backend_parity(stub_spiraltorch) -> None:
    module = stub_spiraltorch
    expected = [
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
    ]

    results = [module.Tensor(2, 3, range(6)).tolist()]  # type: ignore[attr-defined]
    results.append(module.Tensor(2, 3, range(6), backend="python").tolist())  # type: ignore[attr-defined]

    if "numpy" in module.available_stub_backends():  # type: ignore[attr-defined]
        results.append(
            module.Tensor(2, 3, range(6), backend="numpy").tolist()  # type: ignore[attr-defined]
        )

    for matrix in results:
        assert matrix == expected
        assert all(isinstance(row, list) for row in matrix)


def test_stub_tensor_tolist_uses_python_scalars(stub_spiraltorch) -> None:
    module = stub_spiraltorch

    tensor_python = module.Tensor(1, 3, [1, 2, 3], backend="python")  # type: ignore[attr-defined]
    python_result = tensor_python.tolist()
    assert all(
        isinstance(value, float) for row in python_result for value in row
    )

    if "numpy" in module.available_stub_backends():  # type: ignore[attr-defined]
        tensor_numpy = module.Tensor(1, 3, [1, 2, 3], backend="numpy")  # type: ignore[attr-defined]
        numpy_result = tensor_numpy.tolist()
        assert all(
            isinstance(value, float) for row in numpy_result for value in row
        )


@pytest.mark.parametrize(
    "rows, cols",
    [
        (0, 0),
        (0, 3),
        (3, 0),
        (2, 3),
    ],
)
def test_stub_and_native_tolist_agree(
    rows: int, cols: int, stub_spiraltorch
) -> None:
    native = _NATIVE_MODULE
    if native is None:
        pytest.skip("Native SpiralTorch extension is unavailable")

    module = stub_spiraltorch
    payload = [float(index + 1) for index in range(rows * cols)]

    stub_tensor = module.Tensor(rows, cols, payload, backend="python")  # type: ignore[attr-defined]
    native_tensor = native.Tensor(payload, shape=(rows, cols))

    expected = _expected_matrix(rows, cols)
    assert stub_tensor.tolist() == expected
    assert native_tensor.tolist() == expected

    if "numpy" in module.available_stub_backends():  # type: ignore[attr-defined]
        stub_numpy = module.Tensor(rows, cols, payload, backend="numpy")  # type: ignore[attr-defined]
        assert stub_numpy.tolist() == expected
        assert stub_numpy.tolist() == native_tensor.tolist()
