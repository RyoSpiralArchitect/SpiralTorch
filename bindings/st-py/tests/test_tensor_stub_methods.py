from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys
import types
from array import array

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


@pytest.fixture
def stub_spiraltorch(monkeypatch: pytest.MonkeyPatch):
    module_names = (
        "spiraltorch",
        "spiraltorch.spiraltorch",
        "spiraltorch.spiraltorch_native",
        "spiraltorch_native",
    )
    for name in module_names:
        monkeypatch.delitem(sys.modules, name, raising=False)

    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.autograd = types.SimpleNamespace(Function=object)
        monkeypatch.setitem(sys.modules, "torch", torch_stub)

    spec = importlib.util.spec_from_file_location(
        "spiraltorch", REPO_ROOT / "spiraltorch" / "__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, "spiraltorch", module)
    spec.loader.exec_module(module)
    if hasattr(module, "_install_stub_bindings"):
        module._install_stub_bindings(module, ModuleNotFoundError("spiraltorch"))
    return module


@pytest.fixture
def shim_spiraltorch(monkeypatch: pytest.MonkeyPatch):
    module_names = (
        "spiraltorch",
        "spiraltorch.spiraltorch",
        "spiraltorch.spiraltorch_native",
        "spiraltorch_native",
    )
    for name in module_names:
        monkeypatch.delitem(sys.modules, name, raising=False)

    monkeypatch.syspath_prepend(str(REPO_ROOT))

    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.autograd = types.SimpleNamespace(Function=object)
        monkeypatch.setitem(sys.modules, "torch", torch_stub)

    module = importlib.import_module("spiraltorch")
    if hasattr(module, "_install_stub_bindings"):
        module._install_stub_bindings(module, ModuleNotFoundError("spiraltorch"))
    return module


def test_tensor_factory_methods_shapes(stub_spiraltorch) -> None:
    Tensor = stub_spiraltorch.Tensor

    zeros = Tensor.zeros(2, 3)
    assert zeros.shape() == (2, 3)
    assert zeros.rows == 2
    assert zeros.cols == 3

    random_normal = Tensor.randn(1, 4, seed=123)
    assert random_normal.shape() == (1, 4)

    random_uniform = Tensor.rand(3, 1, seed=456)
    assert random_uniform.shape() == (3, 1)


def test_tensor_python_backend_operations(stub_spiraltorch) -> None:
    Tensor = stub_spiraltorch.Tensor
    data = array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    tensor = Tensor._from_python_array(2, 3, data)
    assert tensor.backend == "python"

    reshaped = tensor.reshape(3, 2)
    assert reshaped.shape() == (3, 2)
    assert reshaped.backend == "python"
    assert reshaped.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    transposed = tensor.transpose()
    assert transposed.shape() == (3, 2)
    assert transposed.backend == "python"
    assert transposed.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

    sums = tensor.sum_axis0()
    assert sums == [5.0, 7.0, 9.0]

    row_sums = tensor.sum_axis1()
    assert row_sums == [6.0, 15.0]


def test_tensor_cat_rows_shapes(stub_spiraltorch) -> None:
    Tensor = stub_spiraltorch.Tensor
    first = Tensor.zeros(1, 2)
    second = Tensor.zeros(2, 2)

    combined = Tensor.cat_rows([first, second])
    assert combined.shape() == (3, 2)

    python_first = Tensor._from_python_array(1, 2, array("d", [1.0, 2.0]))
    python_second = Tensor._from_python_array(1, 2, array("d", [3.0, 4.0]))
    python_combined = Tensor.cat_rows([python_first, python_second])
    assert python_combined.shape() == (2, 2)
    assert python_combined.backend == "python"


def test_shim_tensor_constructor_variants(shim_spiraltorch) -> None:
    Tensor = shim_spiraltorch.Tensor

    direct = Tensor(2, 3, [1, 2, 3, 4, 5, 6])
    assert direct.shape() == (2, 3)

    nested = Tensor(2, 3, [[1, 2, 3], [4, 5, 6]])
    assert nested.shape() == (2, 3)

    keyword_data = Tensor((2, 3), data=[1, 2, 3, 4, 5, 6])
    assert keyword_data.shape() == (2, 3)

    zero_sized = Tensor(0, 3, [])
    assert zero_sized.shape() == (0, 3)
    assert zero_sized.tolist() == []
