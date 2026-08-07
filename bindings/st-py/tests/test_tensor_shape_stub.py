from __future__ import annotations


def test_tensor_shape_method_available_in_stub(stub_spiraltorch):
    st = stub_spiraltorch
    assert hasattr(st, "available_stub_backends"), "stub bindings should expose helper APIs"

    tensor = st.Tensor((2, 3))

    assert tensor.shape() == (2, 3)
    assert tuple(tensor.shape) == (2, 3)
