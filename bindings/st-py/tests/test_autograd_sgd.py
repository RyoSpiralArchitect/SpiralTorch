"""Native grouped SGD, including immutable graphs and failure atomicity."""
import importlib

import pytest

import spiraltorch as st


def variable(values):
    return st.AutogradTensor.variable(st.Tensor(1, len(values), values))


def seed(parameter, values):
    parameter.backward(st.Tensor(1, len(values), values))


def test_optimizer_is_native_and_replaces_all_leaves():
    native = importlib.import_module("spiraltorch.spiraltorch")
    assert st.AutogradSgd is native.AutogradSgd
    assert "AutogradSgd" in st.__all__
    x, bias = variable([1., -2.]), variable([0.])
    old_loss = x.hadamard(x).sum().add(bias)
    old_loss.backward()
    optimizer = st.AutogradSgd([x, bias], learning_rate=0.25)
    assert optimizer.step() is None
    assert optimizer.parameter(0).value().tolist() == [[0.5, -1.]]
    assert optimizer.parameter(1).value().tolist() == [[-0.25]]
    assert len(optimizer.parameters()) == 2
    assert all(p.grad() is None for p in optimizer.parameters())
    assert all(p.operation_name() == "leaf" for p in optimizer.parameters())
    assert optimizer.parameter(0).id() != x.id()
    old_loss.backward()
    assert old_loss.item() == 5.
    assert x.value().tolist() == [[1., -2.]]
    assert x.grad().tolist() == [[4., -8.]]
    with pytest.raises(RuntimeError, match="no gradient"):
        optimizer.step()


@pytest.mark.parametrize("missing", [True, False])
def test_late_failure_cannot_publish_a_partial_step(missing):
    maximum = float.fromhex("0x1.fffffep+127")
    first, last = variable([1.]), variable([maximum])
    optimizer = st.AutogradSgd([first, last], 1.)
    before = [p.id() for p in optimizer.parameters()]
    seed(first, [2.])
    if not missing:
        seed(last, [-maximum])
    with pytest.raises(RuntimeError if missing else ValueError):
        optimizer.step()
    assert [p.id() for p in optimizer.parameters()] == before
    assert first.value().tolist() == [[1.]]
    assert first.grad().tolist() == [[2.]]
    assert last.value().tolist() == [[maximum]]
    assert (last.grad() is None) == missing


def test_registration_zeroing_and_learning_rate_validation():
    x = variable([1.])
    optimizer = st.AutogradSgd([], 0.5)
    with pytest.raises(ValueError):
        optimizer.step()
    assert optimizer.add_parameter(x) == 0
    for invalid in (x, x.detach(), x.scale(2.)):
        with pytest.raises(RuntimeError):
            optimizer.add_parameter(invalid)
        assert len(optimizer.parameters()) == 1
    with pytest.raises(RuntimeError, match="unique"):
        st.AutogradSgd([x, x], 0.5)
    for invalid in (x.detach(), x.scale(2.)):
        with pytest.raises(RuntimeError, match="trainable leaves"):
            st.AutogradSgd([invalid], 0.5)
    with pytest.raises(RuntimeError, match="out of bounds"):
        optimizer.parameter(1)
    for rate in (0., -1., float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            st.AutogradSgd([x], rate)
        with pytest.raises(ValueError):
            optimizer.set_learning_rate(rate)
        assert optimizer.learning_rate() == 0.5
    seed(x, [2.])
    optimizer.zero_grad()
    assert x.grad() is None
    seed(x, [2.])
    seed(x, [2.])
    optimizer.set_learning_rate(0.25)
    assert optimizer.learning_rate() == 0.25
    optimizer.step()
    assert optimizer.parameter(0).value().tolist() == [[0.]]
    optimizer.zero_grad()
    assert x.grad().tolist() == [[4.]]


def test_native_classifier_learns_with_fresh_leaf_handles():
    inputs = st.AutogradTensor.constant(st.Tensor(3, 3, [1., 0., 0., 0., 1., 0., 0., 0., 1.]))
    weights = st.AutogradTensor.variable(st.Tensor.zeros(3, 3))
    bias = st.AutogradTensor.variable(st.Tensor.zeros(1, 3))
    optimizer = st.AutogradSgd([weights, bias], 0.5)
    for _ in range(180):
        weights, bias = optimizer.parameters()
        loss = inputs.matmul(weights).add_row(bias).cross_entropy_with_logits([0, 1, 2])
        assert loss.graph_summary()["node_count"] == 6
        loss.backward()
        optimizer.step()
    weights, bias = optimizer.parameters()
    final = inputs.matmul(weights).add_row(bias).cross_entropy_with_logits([0, 1, 2]).item()
    assert final < 0.03
    assert weights.grad() is None and bias.grad() is None


def test_update_matches_independent_torch_sgd():
    torch = pytest.importorskip("torch")
    values = ([1., -2., 0.5], [3., -1.])
    native = st.AutogradSgd([variable(row) for row in values], 0.125)
    reference = [torch.tensor([row], dtype=torch.float32, requires_grad=True) for row in values]
    reference_optimizer = torch.optim.SGD(reference, lr=0.125)
    for _ in range(12):
        reference_optimizer.zero_grad()
        # Two backwards deliberately accumulate, without implicit averaging.
        for _ in range(2):
            for actual, expected in zip(native.parameters(), reference):
                actual.hadamard(actual).sum().backward()
                (expected * expected).sum().backward()
        native.step()
        reference_optimizer.step()
        for actual, expected in zip(native.parameters(), reference):
            assert actual.value().tolist() == expected.detach().tolist()
