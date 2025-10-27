from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.autograd = types.SimpleNamespace(Function=object)
    sys.modules["torch"] = torch_stub

for module_name in (
    "spiraltorch",
    "spiraltorch.spiraltorch",
    "spiraltorch.spiraltorch_native",
    "spiraltorch_native",
):
    sys.modules.pop(module_name, None)

spec = importlib.util.spec_from_file_location(
    "spiraltorch", REPO_ROOT / "spiraltorch" / "__init__.py"
)
st = importlib.util.module_from_spec(spec)
sys.modules["spiraltorch"] = st
assert spec.loader is not None
spec.loader.exec_module(st)

if not hasattr(st, "Tensor") and hasattr(st, "_install_stub_bindings"):
    st._install_stub_bindings(st, ModuleNotFoundError("spiraltorch"))


def test_tensor_from_shape_tuple_produces_zeros() -> None:
    tensor = st.Tensor((2, 3))
    assert tensor.shape() == (2, 3)
    assert tensor.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_tensor_from_nested_iterable_infers_shape() -> None:
    tensor = st.Tensor([[1, 2, 3], [4, 5, 6]])
    assert tensor.shape() == (2, 3)
    assert tensor.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_tensor_from_iterable_with_shape_keyword() -> None:
    tensor = st.Tensor(range(6), shape=(2, 3))
    assert tensor.shape() == (2, 3)
    assert tensor.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


def test_tensor_accepts_existing_tensor_instances() -> None:
    base = st.Tensor([[7, 8], [9, 10]])
    clone = st.Tensor(base)
    assert clone.shape() == base.shape()
    assert clone.tolist() == base.tolist()
