from __future__ import annotations

import math

import spiraltorch as st


def test_row_softmax_rows_sum_to_one() -> None:
    tensor = st.Tensor(2, 3, [1.0, -1.5, 0.25, 0.5, 0.0, -2.0])
    softmax = tensor.row_softmax()
    for row in softmax.tolist():
        assert math.isclose(sum(row), 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_row_softmax_cpu_backend_matches_auto() -> None:
    tensor = st.Tensor(
        3,
        4,
        [0.5, -0.75, 1.25, 0.0, -1.0, 0.2, 0.4, -0.6, 1.2, -0.5, 0.3, -1.8],
    )
    auto = tensor.row_softmax()
    cpu = tensor.row_softmax(backend="cpu")

    auto_rows = auto.tolist()
    cpu_rows = cpu.tolist()
    for lhs, rhs in zip(auto_rows, cpu_rows):
        for a, b in zip(lhs, rhs):
            assert math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7)
