from __future__ import annotations

import importlib.util
import math
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


def _manual_attention(
    queries: list[list[float]],
    keys: list[list[float]],
    values: list[list[float]],
    *,
    contexts: int,
    sequence: int,
    scale: float,
    z_bias: list[list[float]] | None = None,
    attn_bias: list[list[float]] | None = None,
) -> list[list[float]]:
    head_dim = len(queries[0]) if queries else 0
    result: list[list[float]] = []
    for context in range(contexts):
        for query_idx in range(sequence):
            logits: list[float] = []
            for key_idx in range(sequence):
                query_row = queries[context * sequence + query_idx]
                key_row = keys[context * sequence + key_idx]
                dot = sum(q * k for q, k in zip(query_row, key_row))
                logit = dot * scale
                if z_bias is not None:
                    logit += z_bias[context][key_idx]
                if attn_bias is not None:
                    logit += attn_bias[context * sequence + query_idx][key_idx]
                logits.append(logit)
            max_logit = max(logits)
            exps = [math.exp(value - max_logit) for value in logits]
            denom = sum(exps)
            weights = [exp / denom if denom > 0.0 else 0.0 for exp in exps]
            accum = [0.0] * head_dim
            for weight, key_idx in zip(weights, range(sequence)):
                value_row = values[context * sequence + key_idx]
                for dim in range(head_dim):
                    accum[dim] += weight * value_row[dim]
            result.append(accum)
    return result


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


def test_tensor_subclass_preserves_type(stub_spiraltorch) -> None:
    class CustomTensor(stub_spiraltorch.Tensor):
        pass

    tensor = CustomTensor(shape=(2, 2), data=[[1.0, 2.0], [3.0, 4.0]], backend="python")

    reshaped = tensor.reshape(1, 4)
    assert type(reshaped) is CustomTensor

    transposed = tensor.transpose()
    assert type(transposed) is CustomTensor


@pytest.mark.parametrize("backend", ["python", "numpy"])
def test_row_softmax_stub_backends(stub_spiraltorch, backend: str) -> None:
    if backend == "numpy" and not getattr(stub_spiraltorch, "NUMPY_AVAILABLE", False):
        pytest.skip("NumPy backend not available in stub")

    Tensor = stub_spiraltorch.Tensor
    rows = [[1.0, -1.5, 0.5], [0.0, 0.25, -0.25]]
    backend_hint = "python" if backend == "python" else "numpy"
    tensor = Tensor(shape=(2, 3), data=rows, backend=backend_hint)

    softmax = tensor.row_softmax(backend=backend)
    assert softmax.shape() == (2, 3)
    expected_backend = "numpy" if backend == "numpy" else "python"
    assert softmax.backend == expected_backend
    for row in softmax.tolist():
        assert math.isclose(sum(row), 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_labeled_tensor_row_softmax_preserves_axes(stub_spiraltorch) -> None:
    Tensor = stub_spiraltorch.Tensor
    Axis = stub_spiraltorch.Axis
    LabeledTensor = stub_spiraltorch.LabeledTensor

    base = Tensor(shape=(2, 2), data=[[1.0, 0.0], [0.0, 1.0]], backend="python")
    labeled = LabeledTensor(base, [Axis("rows"), Axis("cols")])

    result = labeled.row_softmax(backend="python")
    assert isinstance(result, LabeledTensor)
    assert result.axis_names() == ("rows", "cols")
    for row in result.tolist():
        assert math.isclose(sum(row), 1.0, rel_tol=1e-6, abs_tol=1e-6)


@pytest.mark.parametrize("backend", ["python", "numpy"])
def test_scaled_dot_attention_stub_matches_manual(stub_spiraltorch, backend: str) -> None:
    if backend == "numpy" and not getattr(stub_spiraltorch, "NUMPY_AVAILABLE", False):
        pytest.skip("NumPy backend not available in stub")

    Tensor = stub_spiraltorch.Tensor
    contexts, sequence, head_dim = 1, 2, 2
    backend_hint = "python" if backend == "python" else "numpy"

    queries = [[1.0, 0.0], [0.0, 1.0]]
    keys = [[1.0, 0.0], [0.0, 1.0]]
    values = [[1.0, 2.0], [3.0, 4.0]]
    z_bias = [[0.2, -0.1]]
    attn_bias = [[0.05, -0.05], [-0.02, 0.02]]

    q_tensor = Tensor(shape=(contexts * sequence, head_dim), data=queries, backend=backend_hint)
    k_tensor = Tensor(shape=(contexts * sequence, head_dim), data=keys, backend=backend_hint)
    v_tensor = Tensor(shape=(contexts * sequence, head_dim), data=values, backend=backend_hint)
    z_tensor = Tensor(shape=(contexts, sequence), data=z_bias, backend=backend_hint)
    attn_tensor = Tensor(
        shape=(contexts * sequence, sequence), data=attn_bias, backend=backend_hint
    )

    result = q_tensor.scaled_dot_attention(
        k_tensor,
        v_tensor,
        contexts=contexts,
        sequence=sequence,
        scale=1.0,
        z_bias=z_tensor,
        attn_bias=attn_tensor,
        backend=backend,
    )

    expected = _manual_attention(
        queries,
        keys,
        values,
        contexts=contexts,
        sequence=sequence,
        scale=1.0,
        z_bias=z_bias,
        attn_bias=attn_bias,
    )

    assert result.shape() == (contexts * sequence, head_dim)
    expected_backend = "numpy" if backend == "numpy" else "python"
    assert result.backend == expected_backend
    for actual, manual in zip(result.tolist(), expected):
        assert actual == pytest.approx(manual, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("backend", ["python", "numpy"])
def test_tensor_add_sub_scale_backends(stub_spiraltorch, backend: str) -> None:
    if backend == "numpy" and not getattr(stub_spiraltorch, "NUMPY_AVAILABLE", False):
        pytest.skip("NumPy backend not available in stub")

    Tensor = stub_spiraltorch.Tensor
    backend_hint = "python" if backend == "python" else "numpy"

    data_a = [[1.0, 2.0], [3.0, 4.0]]
    data_b = [[0.5, -0.5], [1.5, -1.5]]
    tensor_a = Tensor(shape=(2, 2), data=data_a, backend=backend_hint)
    tensor_b = Tensor(shape=(2, 2), data=data_b, backend=backend_hint)

    added = tensor_a.add(tensor_b)
    subbed = tensor_a.sub(tensor_b)
    scaled = tensor_a.scale(2.0)

    expected_backend = "numpy" if backend == "numpy" else "python"
    assert added.backend == expected_backend
    assert subbed.backend == expected_backend
    assert scaled.backend == expected_backend

    expected_add = [[1.5, 1.5], [4.5, 2.5]]
    expected_sub = [[0.5, 2.5], [1.5, 5.5]]
    expected_scale = [[2.0, 4.0], [6.0, 8.0]]

    for actual, expected_row in zip(added.tolist(), expected_add):
        assert actual == pytest.approx(expected_row, rel=1e-9, abs=1e-9)
    for actual, expected_row in zip(subbed.tolist(), expected_sub):
        assert actual == pytest.approx(expected_row, rel=1e-9, abs=1e-9)
    for actual, expected_row in zip(scaled.tolist(), expected_scale):
        assert actual == pytest.approx(expected_row, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("backend", ["python", "numpy"])
def test_tensor_inplace_ops_backends(stub_spiraltorch, backend: str) -> None:
    if backend == "numpy" and not getattr(stub_spiraltorch, "NUMPY_AVAILABLE", False):
        pytest.skip("NumPy backend not available in stub")

    Tensor = stub_spiraltorch.Tensor
    backend_hint = "python" if backend == "python" else "numpy"

    base = Tensor(shape=(2, 2), data=[[1.0, 2.0], [3.0, 4.0]], backend=backend_hint)
    other = Tensor(shape=(2, 2), data=[[0.5, 1.0], [1.5, 2.0]], backend=backend_hint)

    base.add_scaled_(other, 0.5)
    after_add = base.tolist()
    expected_after_add = [[1.25, 2.5], [3.75, 5.0]]
    for actual, expected_row in zip(after_add, expected_after_add):
        assert actual == pytest.approx(expected_row, rel=1e-9, abs=1e-9)

    base.add_row_inplace([1.0, -1.0])
    after_bias = base.tolist()
    expected_after_bias = [[2.25, 1.5], [4.75, 4.0]]
    for actual, expected_row in zip(after_bias, expected_after_bias):
        assert actual == pytest.approx(expected_row, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("backend", ["python", "numpy"])
def test_tensor_squared_norm_and_hyperbolic(stub_spiraltorch, backend: str) -> None:
    if backend == "numpy" and not getattr(stub_spiraltorch, "NUMPY_AVAILABLE", False):
        pytest.skip("NumPy backend not available in stub")

    Tensor = stub_spiraltorch.Tensor
    backend_hint = "python" if backend == "python" else "numpy"

    tensor = Tensor(shape=(2, 2), data=[[0.3, 0.4], [0.0, 0.0]], backend=backend_hint)
    assert math.isclose(tensor.squared_l2_norm(), 0.3**2 + 0.4**2, rel_tol=1e-9, abs_tol=1e-9)

    projected = tensor.project_to_poincare(-1.0)
    for row in projected.tolist():
        norm = math.sqrt(sum(value * value for value in row))
        assert norm <= 1.0 + 1e-9

    other = Tensor(shape=(2, 2), data=[[0.1, 0.2], [0.0, 0.0]], backend=backend_hint)
    projected_other = other.project_to_poincare(-1.0)
    distance = projected.hyperbolic_distance(projected_other, -1.0)
    assert distance >= 0.0
    assert math.isclose(projected.hyperbolic_distance(projected, -1.0), 0.0, abs_tol=1e-9)


def test_hyperbolic_methods_validate_curvature(stub_spiraltorch) -> None:
    Tensor = stub_spiraltorch.Tensor
    tensor = Tensor(shape=(1, 2), data=[[0.0, 0.0]], backend="python")

    with pytest.raises(ValueError):
        tensor.project_to_poincare(0.0)
    with pytest.raises(ValueError):
        tensor.hyperbolic_distance(tensor, 0.0)
