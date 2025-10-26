from __future__ import annotations

import spiraltorch as st


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
