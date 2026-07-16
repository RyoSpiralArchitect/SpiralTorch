import pytest

import spiraltorch as st


pytestmark = pytest.mark.skipif(
    getattr(st, "_rs", None) is None or not hasattr(st, "AutogradTensor"),
    reason="AutogradTensor requires the freshly built native extension",
)


def tensor(values: list[float], *, rows: int = 1) -> st.Tensor:
    assert len(values) % rows == 0
    return st.Tensor((rows, len(values) // rows), data=values)


def test_branching_graph_accumulates_every_rust_owned_contribution() -> None:
    x = st.AutogradTensor.variable(tensor([1.0, 2.0, -1.0]))
    loss = x.hadamard(x).add(x.scale(3.0)).sum()

    receipt = loss.backward()

    assert x.grad().tolist()[0] == pytest.approx([5.0, 7.0, 1.0])
    assert receipt["contract_version"] == "spiraltorch.autograd.v1"
    assert receipt["semantic_owner"] == "st-tensor"
    assert receipt["leaf_gradient_count"] == 1
    assert receipt["operation_count"] == 4


def test_non_scalar_backward_requires_an_explicit_vjp_seed() -> None:
    x = st.AutogradTensor.variable(tensor([2.0, 3.0]))
    y = x.scale(4.0)

    with pytest.raises(ValueError, match="explicit output gradient"):
        y.backward()

    y.backward(tensor([0.5, -1.0]))
    assert x.grad().tolist()[0] == pytest.approx([2.0, -4.0])


def test_vector_jacobian_product_is_side_effect_free_and_disconnect_safe() -> None:
    x = st.AutogradTensor.variable(tensor([2.0, -3.0]))
    output = x.hadamard(x)
    disconnected = st.AutogradTensor.variable(tensor([4.0, 5.0]))
    disconnected.scale(3.0).sum().backward()

    gradient = output.vector_jacobian_product(x, tensor([0.5, -2.0]))
    disconnected_gradient = output.vector_jacobian_product(
        disconnected, tensor([1.0, 1.0])
    )

    assert gradient.tolist()[0] == pytest.approx([2.0, 12.0])
    assert disconnected_gradient.tolist()[0] == [0.0, 0.0]
    assert x.grad() is None
    assert disconnected.grad().tolist()[0] == pytest.approx([3.0, 3.0])


def test_repeated_backward_and_graph_zeroing_are_explicit() -> None:
    x = st.AutogradTensor.variable(tensor([2.0]))
    loss = x.hadamard(x).sum()

    loss.backward()
    loss.backward()
    assert x.grad().tolist()[0] == pytest.approx([8.0])

    loss.zero_grad_graph()
    assert x.grad() is None
    assert loss.grad() is None


def test_mean_squared_error_and_graph_summary_share_contract_metadata() -> None:
    prediction = st.AutogradTensor.variable(tensor([1.0, 3.0]))
    target = st.AutogradTensor.constant(tensor([2.0, 1.0]))
    loss = prediction.mean_squared_error(target)

    assert loss.item() == pytest.approx(2.5)
    summary = loss.graph_summary()
    assert summary["contract_version"] == st.AUTOGRAD_CONTRACT_VERSION
    assert summary["semantic_owner"] == st.AUTOGRAD_SEMANTIC_OWNER
    assert summary["trainable_leaf_count"] == 1

    loss.backward()
    assert prediction.grad().tolist()[0] == pytest.approx([-1.0, 2.0])


def test_failed_backward_does_not_commit_partial_gradients() -> None:
    x = st.AutogradTensor.variable(tensor([1.0e-38]))
    output = x.scale(3.4028234663852886e38)

    with pytest.raises(ValueError, match="non-finite"):
        output.backward(tensor([2.0]))

    assert x.grad() is None
    assert output.grad() is None
