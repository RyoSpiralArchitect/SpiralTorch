from __future__ import annotations

import importlib
import pathlib
import sys
import types
import warnings
from typing import Optional

import pytest


def _ensure_torch_stub() -> None:
    if "torch" in sys.modules:
        return
    torch_stub = types.ModuleType("torch")
    torch_stub.autograd = types.SimpleNamespace(Function=object)
    sys.modules["torch"] = torch_stub


def _load_stub_module() -> types.ModuleType:
    root = pathlib.Path(__file__).resolve().parents[3]
    source_path = root / "spiraltorch" / "__init__.py"
    source = source_path.read_text()

    for marker in ("\n_load_native_package()", "\ndel _load_native_package"):
        head, sep, _ = source.rpartition(marker)
        if sep:
            source = head

    module = types.ModuleType("_spiraltorch_stub_for_tests")
    module.__file__ = str(source_path)
    module.__package__ = "spiraltorch"
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        module._install_stub_bindings(  # type: ignore[attr-defined]
            module,
            ModuleNotFoundError("spiraltorch", name="spiraltorch.spiraltorch"),
        )
    return module


def _load_native_module() -> Optional[types.ModuleType]:
    _ensure_torch_stub()
    try:
        native = importlib.import_module("spiraltorch")
    except ModuleNotFoundError:
        return None
    return native if hasattr(native, "Tensor") else None


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
def test_stub_tensor_tolist_matches_expected(rows: int, cols: int) -> None:
    module = _load_stub_module()
    payload = [float(index + 1) for index in range(rows * cols)]
    tensor = module.Tensor(rows, cols, payload, backend="python")  # type: ignore[attr-defined]
    assert tensor.tolist() == _expected_matrix(rows, cols)

    if "numpy" in module.available_stub_backends():  # type: ignore[attr-defined]
        tensor_np = module.Tensor(rows, cols, payload, backend="numpy")  # type: ignore[attr-defined]
        assert tensor_np.tolist() == _expected_matrix(rows, cols)


def test_stub_tensor_tolist_from_range_is_nested() -> None:
    module = _load_stub_module()
    expected = _expected_range_matrix()

    tensor_python = module.Tensor(2, 3, range(6), backend="python")  # type: ignore[attr-defined]
    assert tensor_python.tolist() == expected

    if "numpy" in module.available_stub_backends():  # type: ignore[attr-defined]
        tensor_numpy = module.Tensor(2, 3, range(6), backend="numpy")  # type: ignore[attr-defined]
        assert tensor_numpy.tolist() == expected


@pytest.mark.parametrize(
    "rows, cols",
    [
        (0, 0),
        (0, 3),
        (3, 0),
        (2, 3),
    ],
)
def test_stub_and_native_tolist_agree(rows: int, cols: int) -> None:
    native = _load_native_module()
    if native is None:
        pytest.skip("Native SpiralTorch extension is unavailable")

    module = _load_stub_module()
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
