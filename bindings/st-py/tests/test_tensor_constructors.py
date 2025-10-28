from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
from typing import Iterable

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

import pytest


def _install_stub_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubTensorBase:
        """Very small stand-in for the native tensor type used in tests."""

        matmul_simd_prepacked = None  # Signal that SIMD fast paths are absent.

        def __new__(cls, rows: int, cols: int, data: Iterable[float] | None = None):
            self = super().__new__(cls)
            self._rows = int(rows)
            self._cols = int(cols)
            if self._rows < 0 or self._cols < 0:
                raise ValueError("tensor dimensions must be non-negative")

            if data is None:
                values = [0.0] * (self._rows * self._cols)
            else:
                values = [float(value) for value in data]
                if len(values) != self._rows * self._cols:
                    raise ValueError(
                        "data length does not match matrix dimensions",
                    )

            self._values = values
            return self

        def shape(self) -> tuple[int, int]:
            return self._rows, self._cols

        @property
        def rows(self) -> int:
            return self._rows

        @property
        def cols(self) -> int:
            return self._cols

        def tolist(self) -> list[list[float]]:
            rows, cols = self._rows, self._cols
            if rows == 0:
                return []
            if cols == 0:
                return [[] for _ in range(rows)]
            return [
                self._values[index : index + cols]
                for index in range(0, rows * cols, cols)
            ]

        def matmul(
            self, other: "_StubTensorBase", *, backend: str | None = None, out=None
        ):
            if not isinstance(other, _StubTensorBase):
                raise TypeError("matmul expects another Tensor instance")
            if self._cols != other._rows:
                raise ValueError("inner dimensions do not match for matmul")

            rows, cols, inner = self._rows, other._cols, self._cols
            if rows == 0 or cols == 0 or inner == 0:
                return type(self)(rows, cols, [])

            result = [0.0] * (rows * cols)
            left = self._values
            right = other._values
            for i in range(rows):
                for k in range(inner):
                    lhs = left[i * inner + k]
                    if lhs == 0.0:
                        continue
                    for j in range(cols):
                        result[i * cols + j] += lhs * right[k * cols + j]
            return type(self)(rows, cols, result)

        @staticmethod
        def storage_token(_: "_StubTensorBase") -> int:
            return 0

    stub_native = types.ModuleType("spiraltorch.spiraltorch")
    stub_native.Tensor = _StubTensorBase
    stub_native.PyTensor = _StubTensorBase

    monkeypatch.setitem(sys.modules, "spiraltorch.spiraltorch", stub_native)
    monkeypatch.setitem(sys.modules, "spiraltorch.spiraltorch_native", stub_native)
    monkeypatch.setitem(sys.modules, "spiraltorch_native", stub_native)

    zspace_stub = types.ModuleType("spiraltorch.zspace_inference")
    _ZSPACE_CLASSES = [
        "ZMetrics",
        "ZSpaceDecoded",
        "ZSpaceInference",
        "ZSpacePosterior",
        "ZSpacePartialBundle",
        "ZSpaceTelemetryFrame",
        "ZSpaceInferencePipeline",
    ]
    for name in _ZSPACE_CLASSES:
        setattr(zspace_stub, name, type(name, (), {}))

    _ZSPACE_FUNCTIONS = [
        "inference_to_mapping",
        "inference_to_zmetrics",
        "prepare_trainer_step_payload",
        "canvas_partial_from_snapshot",
        "canvas_coherence_partial",
        "elliptic_partial_from_telemetry",
        "coherence_partial_from_diagnostics",
        "decode_zspace_embedding",
        "blend_zspace_partials",
        "infer_canvas_snapshot",
        "infer_canvas_transformer",
        "infer_coherence_diagnostics",
        "infer_coherence_from_sequencer",
        "infer_canvas_with_coherence",
        "infer_with_partials",
        "infer_from_partial",
        "weights_partial_from_dlpack",
        "weights_partial_from_compat",
        "infer_weights_from_dlpack",
        "infer_weights_from_compat",
    ]
    for name in _ZSPACE_FUNCTIONS:
        setattr(zspace_stub, name, lambda *args, **kwargs: None)

    monkeypatch.setitem(sys.modules, "spiraltorch.zspace_inference", zspace_stub)

    elliptic_stub = types.ModuleType("spiraltorch.elliptic")
    elliptic_stub.EllipticWarpFunction = type("EllipticWarpFunction", (), {})
    elliptic_stub.elliptic_warp_autograd = lambda *args, **kwargs: None
    elliptic_stub.elliptic_warp_features = lambda *args, **kwargs: None
    elliptic_stub.elliptic_warp_partial = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "spiraltorch.elliptic", elliptic_stub)


@pytest.fixture
def spiraltorch_module(monkeypatch: pytest.MonkeyPatch):
    """Load the Python Tensor front-end with a lightweight stub backend.

    The real bindings expect to inherit from the compiled ``Tensor`` type.  For
    unit tests we emulate the minimal surface required by the constructor logic
    so that shape inference runs in pure Python.
    """

    for name in list(sys.modules):
        if name == "spiraltorch" or name.startswith("spiraltorch."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    _install_stub_environment(monkeypatch)

    impl_path = pathlib.Path(__file__).resolve().parents[1] / "spiraltorch" / "__init__.py"
    spec = importlib.util.spec_from_file_location("spiraltorch", impl_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader  # narrow type checkers
    monkeypatch.setitem(sys.modules, "spiraltorch", module)
    spec.loader.exec_module(module)
    return module


def test_tensor_from_shape_tuple_produces_zeros(spiraltorch_module) -> None:
    st = spiraltorch_module
    tensor = st.Tensor((2, 3))
    assert tensor.shape() == (2, 3)
    assert tensor.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_tensor_from_nested_iterable_infers_shape(spiraltorch_module) -> None:
    st = spiraltorch_module
    tensor = st.Tensor([[1, 2, 3], [4, 5, 6]])
    assert tensor.shape() == (2, 3)
    assert tensor.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_tensor_from_iterable_with_shape_keyword(spiraltorch_module) -> None:
    st = spiraltorch_module
    tensor = st.Tensor(range(6), shape=(2, 3))
    assert tensor.shape() == (2, 3)
    assert tensor.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


def test_tensor_accepts_existing_tensor_instances(spiraltorch_module) -> None:
    st = spiraltorch_module
    base = st.Tensor([[7, 8], [9, 10]])
    clone = st.Tensor(base)
    assert clone.shape() == base.shape()
    assert clone.tolist() == base.tolist()


def test_tensor_supports_zero_dimensional_shapes(spiraltorch_module) -> None:
    st = spiraltorch_module

    zero_by_three = st.Tensor(0, 3, [])
    assert zero_by_three.shape() == (0, 3)
    assert zero_by_three.tolist() == []

    three_by_zero = st.Tensor([[], [], []])
    assert three_by_zero.shape() == (3, 0)
    assert three_by_zero.tolist() == [[], [], []]

    explicit_zero = st.Tensor([], rows=0, cols=0)
    assert explicit_zero.shape() == (0, 0)
    assert explicit_zero.tolist() == []

    zero_square = zero_by_three.matmul(st.Tensor(3, 0, []))
    assert zero_square.shape() == (0, 0)
    assert zero_square.tolist() == []

    zero_self_product = explicit_zero.matmul(explicit_zero)
    assert zero_self_product.shape() == (0, 0)
    assert zero_self_product.tolist() == []


def test_labeled_tensor_supports_zero_sized_axes(spiraltorch_module) -> None:
    st = spiraltorch_module

    left = st.tensor([], axes=(st.Axis("rows"), st.Axis("shared")))
    right = st.tensor([], axes=(st.Axis("shared"), st.Axis("cols")))

    assert left.shape == (0, 0)
    assert right.shape == (0, 0)
    assert left.tolist() == []
    assert right.tolist() == []
    assert left.axes[0].size == 0
    assert left.axes[1].size == 0
    assert right.axes[0].size == 0
    assert right.axes[1].size == 0

    product = left @ right
    assert product.shape == (0, 0)
    assert product.tolist() == []
    assert product.axes[0].size == 0
    assert product.axes[1].size == 0
