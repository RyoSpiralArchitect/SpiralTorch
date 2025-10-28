import builtins
import importlib.util
import pathlib
import sys
import types

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


@pytest.fixture
def stub_spiraltorch_no_numpy(monkeypatch: pytest.MonkeyPatch):
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

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "numpy":
            raise ModuleNotFoundError("No module named 'numpy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

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


def test_tensor_factories_python_backend(stub_spiraltorch_no_numpy) -> None:
    Tensor = stub_spiraltorch_no_numpy.Tensor

    zeros = Tensor.zeros(2, 3)
    assert zeros.shape() == (2, 3)
    assert zeros.rows == 2
    assert zeros.cols == 3
    assert zeros.backend == "python"
    assert zeros.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    normal = Tensor.randn(1, 4, seed=7)
    assert normal.shape() == (1, 4)
    assert normal.backend == "python"

    uniform = Tensor.rand(3, 2, min=-1.0, max=1.0, seed=11)
    assert uniform.shape() == (3, 2)
    assert uniform.backend == "python"
    for row in uniform.tolist():
        for value in row:
            assert -1.0 <= value <= 1.0


def test_tensor_basic_ops_python_backend(stub_spiraltorch_no_numpy) -> None:
    Tensor = stub_spiraltorch_no_numpy.Tensor
    base = Tensor(shape=(2, 2), data=[[1.0, 2.0], [3.0, 4.0]], backend="python")

    reshaped = base.reshape(1, 4)
    assert reshaped.shape() == (1, 4)
    assert reshaped.backend == "python"

    transposed = base.transpose()
    assert transposed.shape() == (2, 2)
    assert transposed.backend == "python"
    assert transposed.tolist() == [[1.0, 3.0], [2.0, 4.0]]

    axis0 = base.sum_axis0()
    assert axis0 == [4.0, 6.0]

    axis1 = base.sum_axis1()
    assert axis1 == [3.0, 7.0]


def test_tensor_cat_rows_python_backend(stub_spiraltorch_no_numpy) -> None:
    Tensor = stub_spiraltorch_no_numpy.Tensor

    first = Tensor(shape=(1, 2), data=[[1.0, 2.0]], backend="python")
    second = Tensor(shape=(2, 2), data=[[3.0, 4.0], [5.0, 6.0]], backend="python")

    combined = Tensor.cat_rows([first, second])
    assert combined.shape() == (3, 2)
    assert combined.backend == "python"
    assert combined.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    subclass = type("CustomTensor", (Tensor,), {})
    custom_first = subclass(shape=(1, 2), data=[[0.0, 1.0]], backend="python")
    custom_second = subclass(shape=(1, 2), data=[[2.0, 3.0]], backend="python")
    custom_combined = subclass.cat_rows([custom_first, custom_second])
    assert isinstance(custom_combined, subclass)
    assert custom_combined.shape() == (2, 2)
    assert custom_combined.backend == "python"
    assert custom_combined.tolist() == [[0.0, 1.0], [2.0, 3.0]]
