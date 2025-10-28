from __future__ import annotations

import math
import pathlib
import sys
import types

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


@pytest.fixture
def spiraltorch_stub(monkeypatch: pytest.MonkeyPatch):
    """Materialize the pure-Python SpiralTorch stub bindings for testing."""

    stub_path = REPO_ROOT / "spiraltorch" / "__init__.py"
    source = stub_path.read_text()
    prefix, _, _ = source.partition("\n_load_native_package()")

    module = types.ModuleType("spiraltorch_stub_test")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(prefix, str(stub_path), "exec"), module.__dict__)
    module._install_stub_bindings(module, ModuleNotFoundError("spiraltorch"))
    return module


def _reference_attention(
    queries: list[float],
    keys: list[float],
    values: list[float],
    *,
    contexts: int,
    sequence: int,
    head_dim: int,
    scale: float,
    z_bias: list[float] | None,
    attn_bias: list[float] | None,
) -> list[list[float]]:
    output: list[list[float]] = []
    for context in range(contexts):
        context_offset = context * sequence
        for query_idx in range(sequence):
            query_row = context_offset + query_idx
            logits: list[float] = []
            for key_idx in range(sequence):
                key_row = context_offset + key_idx
                query_offset = query_row * head_dim
                key_offset = key_row * head_dim
                dot = 0.0
                for dim in range(head_dim):
                    dot += queries[query_offset + dim] * keys[key_offset + dim]
                logit = dot * scale
                if z_bias is not None:
                    logit += z_bias[context_offset + key_idx]
                if attn_bias is not None:
                    logit += attn_bias[query_row * sequence + key_idx]
                logits.append(logit)
            if logits:
                max_logit = max(logits)
                exp_values = [math.exp(value - max_logit) for value in logits]
                denom = sum(exp_values)
                if denom > 0.0:
                    weights = [value / denom for value in exp_values]
                else:
                    weights = [0.0] * len(exp_values)
            else:
                weights = []
            accum = [0.0] * head_dim
            for key_idx, weight in enumerate(weights):
                key_row = context_offset + key_idx
                key_offset = key_row * head_dim
                for dim in range(head_dim):
                    accum[dim] += weight * values[key_offset + dim]
            output.append(accum)
    return output


def _assert_matrix_close(actual: list[list[float]], expected: list[list[float]], *, rel: float = 1e-7, abs: float = 1e-7) -> None:
    for lhs, rhs in zip(actual, expected):
        assert lhs == pytest.approx(rhs, rel=rel, abs=abs)


def test_row_softmax_stub_matches_cpu_backend(spiraltorch_stub) -> None:
    st = spiraltorch_stub
    tensor = st.Tensor([[1.0, -1.0, 0.5], [0.0, 0.25, -0.75]])
    auto = tensor.row_softmax()
    cpu = tensor.row_softmax(backend="cpu")
    for row in auto.tolist():
        assert pytest.approx(1.0, rel=1e-6, abs=1e-6) == sum(row)
    _assert_matrix_close(auto.tolist(), cpu.tolist())


def test_tensor_arithmetic_and_scaling(spiraltorch_stub) -> None:
    st = spiraltorch_stub
    lhs = st.Tensor(2, 2, [1.0, 2.0, 3.0, 4.0])
    rhs = st.Tensor(2, 2, [0.5, -1.0, 1.5, 2.0])
    added = lhs.add(rhs)
    subtracted = lhs.sub(rhs)
    scaled = lhs.scale(2.0)
    _assert_matrix_close(added.tolist(), [[1.5, 1.0], [4.5, 6.0]])
    _assert_matrix_close(subtracted.tolist(), [[0.5, 3.0], [1.5, 2.0]])
    _assert_matrix_close(scaled.tolist(), [[2.0, 4.0], [6.0, 8.0]])


def test_add_scaled_and_row_bias_inplace(spiraltorch_stub) -> None:
    st = spiraltorch_stub
    base = st.Tensor(2, 3, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    other = st.Tensor(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    base.add_scaled_(other, 0.5)
    _assert_matrix_close(base.tolist(), [[0.5, 1.0, 1.5], [3.0, 3.5, 4.0]])
    base.add_row_inplace([1.0, -1.0, 0.25])
    _assert_matrix_close(base.tolist(), [[1.5, 0.0, 1.75], [4.0, 2.5, 4.25]])
    with pytest.raises(ValueError):
        base.add_row_inplace([1.0, 2.0])


def test_squared_norm_and_hyperbolic_helpers(spiraltorch_stub) -> None:
    st = spiraltorch_stub
    tensor = st.Tensor(1, 3, [3.0, 4.0, 0.0])
    assert tensor.squared_l2_norm() == pytest.approx(25.0)
    projected = tensor.project_to_poincare(-1.0)
    assert projected.tolist()[0] == pytest.approx(
        [math.tanh(5.0) * 3.0 / 5.0, math.tanh(5.0) * 4.0 / 5.0, 0.0]
    )
    other = st.Tensor(1, 3, [2.0, 1.0, 0.0]).project_to_poincare(-1.0)
    distance = projected.hyperbolic_distance(other, -1.0)
    assert distance > 0.0
    with pytest.raises(ValueError):
        tensor.project_to_poincare(0.0)


def test_scaled_dot_attention_matches_reference(spiraltorch_stub) -> None:
    st = spiraltorch_stub
    contexts = 1
    sequence = 2
    head_dim = 3
    queries = st.Tensor(
        contexts * sequence,
        head_dim,
        [0.2, 0.4, -0.1, 0.5, -0.3, 0.1],
    )
    keys = st.Tensor(
        contexts * sequence,
        head_dim,
        [0.1, -0.2, 0.3, -0.4, 0.6, -0.1],
    )
    values = st.Tensor(
        contexts * sequence,
        head_dim,
        [1.0, 0.0, -1.0, -0.5, 0.75, 0.25],
    )
    z_bias = st.Tensor(contexts, sequence, [0.05, -0.1])
    attn_bias = st.Tensor(
        contexts * sequence,
        sequence,
        [0.0, 0.1, -0.2, 0.0],
    )
    result = queries.scaled_dot_attention(
        keys,
        values,
        contexts=contexts,
        sequence=sequence,
        scale=0.5,
        z_bias=z_bias,
        attn_bias=attn_bias,
    )
    reference = _reference_attention(
        list(queries._row_major_python()),
        list(keys._row_major_python()),
        list(values._row_major_python()),
        contexts=contexts,
        sequence=sequence,
        head_dim=head_dim,
        scale=0.5,
        z_bias=list(z_bias._row_major_python()),
        attn_bias=list(attn_bias._row_major_python()),
    )
    _assert_matrix_close(result.tolist(), reference, rel=1e-6, abs=1e-6)


def test_dlpack_methods_raise(spiraltorch_stub) -> None:
    st = spiraltorch_stub
    tensor = st.Tensor((1, 1))
    with pytest.raises(RuntimeError):
        st.Tensor.from_dlpack(object())
    with pytest.raises(RuntimeError):
        tensor.to_dlpack()
    with pytest.raises(RuntimeError):
        tensor.__dlpack__()
    with pytest.raises(RuntimeError):
        tensor.__dlpack_device__()
