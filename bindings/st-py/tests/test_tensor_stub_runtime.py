from __future__ import annotations


def test_tensor_constructor_and_shape_in_stub_environment(stub_spiraltorch) -> None:
    st = stub_spiraltorch
    assert hasattr(st, "available_stub_backends")

    tensor = st.Tensor((2, 3))
    assert tensor.shape() == (2, 3)
    assert tuple(tensor.shape) == (2, 3)
    assert tensor.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    eager_values = [float(i) for i in range(6)]
    constructed = [
        tensor,
        st.Tensor(2, 3),
        st.Tensor(2, 3, eager_values),
        st.Tensor(rows=2, cols=3),
        st.Tensor(rows=2, cols=3, data=eager_values),
        st.Tensor(shape=(2, 3)),
    ]
    for created in constructed:
        assert created.shape() == (2, 3)
