import pytest
import spiraltorch as st


pytestmark = pytest.mark.skipif(
    getattr(st, "_rs", None) is None or not hasattr(st, "AutogradTensor"),
    reason="requires the native Rust autograd implementation",
)


@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("readonly", [False, True])
def test_foreign_leaf_captures_forward_values(requires_grad, readonly):
    np = pytest.importorskip("numpy", minversion="2.0" if readonly else "1.26")
    source = np.array([[2.0, 3.0]], dtype=np.float32)
    source.flags.writeable = not readonly
    leaf = st.AutogradTensor(st.from_dlpack(source), requires_grad=requires_grad)
    loss = leaf.hadamard(leaf).sum()
    source.flags.writeable = True
    source[:] = 10.0
    assert leaf.value().tolist() == [[2.0, 3.0]]
    assert loss.item() == 13.0
    loss.backward()
    if requires_grad:
        assert leaf.grad().tolist() == [[4.0, 6.0]]
    else:
        assert leaf.grad() is None


def test_existing_writable_torch_alias_does_not_change_a_native_leaf():
    torch = pytest.importorskip("torch")
    source = st.Tensor(1, 2, [2.0, 3.0])
    alias = torch.from_dlpack(source)
    leaf = st.AutogradTensor.variable(source)
    loss = leaf.hadamard(leaf).sum()
    alias.fill_(10.0)
    assert source.tolist() == [[10.0, 10.0]]
    assert leaf.value().tolist() == [[2.0, 3.0]]
    loss.backward()
    assert leaf.grad().tolist() == [[4.0, 6.0]]


def test_snapshot_sharing_readonly_and_legacy_copy_are_distinct():
    np = pytest.importorskip("numpy", minversion="2.0")
    torch = pytest.importorskip("torch")
    source = st.Tensor(1, 2, [2.0, 3.0])
    snapshot = source.snapshot()
    assert not source.is_snapshot()
    assert snapshot.is_snapshot()
    source.add_row_inplace([10.0, 20.0])
    assert snapshot.tolist() == [[2.0, 3.0]]
    shared = np.from_dlpack(snapshot, copy=False)
    assert not shared.flags.writeable
    with pytest.raises(ValueError):
        shared[0, 0] = 999.0
    with pytest.raises(BufferError):
        snapshot.__dlpack__(max_version=(0, 8), copy=False)
    independent = torch.from_dlpack(snapshot.to_dlpack())
    independent.fill_(100.0)
    assert snapshot.tolist() == [[2.0, 3.0]]
    again = snapshot.snapshot()
    assert np.shares_memory(shared, np.from_dlpack(again, copy=False))
    snapshot.add_row_inplace([1.0, 2.0])
    assert snapshot.tolist() == [[3.0, 5.0]]
    assert again.tolist() == shared.tolist() == [[2.0, 3.0]]


def test_value_and_gradient_exports_cannot_mutate_the_graph():
    torch = pytest.importorskip("torch")
    leaf = st.AutogradTensor.variable(st.Tensor(1, 2, [2.0, 3.0]))
    loss = leaf.hadamard(leaf).sum()
    torch.from_dlpack(leaf.value().to_dlpack()).fill_(100.0)
    loss.backward()
    torch.from_dlpack(leaf.grad().to_dlpack()).fill_(200.0)
    assert leaf.value().tolist() == [[2.0, 3.0]]
    assert leaf.grad().tolist() == [[4.0, 6.0]]
    loss.backward()
    assert leaf.grad().tolist() == [[8.0, 12.0]]


def test_external_seed_does_not_alias_committed_gradients():
    np = pytest.importorskip("numpy")
    source = np.array([[4.0, 5.0]], dtype=np.float32)
    leaf = st.AutogradTensor.variable(st.Tensor(1, 2, [2.0, 3.0]))
    leaf.backward(st.from_dlpack(source))
    source[:] = 100.0
    assert leaf.grad().tolist() == [[4.0, 5.0]]


@pytest.mark.parametrize("operation", ["relu", "gelu", "row_softmax"])
def test_nonlinear_vjp_matches_pytorch(operation):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    values = np.array([[-2.0, 0.0, 1.3], [0.7, -0.8, 2.0]], dtype=np.float32)
    seed = np.array([[0.2, -1.0, 0.5], [0.7, -0.2, 1.1]], dtype=np.float32)
    x = st.AutogradTensor.variable(st.from_dlpack(values))
    y = getattr(x, operation)()
    native_grad = y.vector_jacobian_product(x, st.from_dlpack(seed))
    assert x.grad() is None
    reference = torch.tensor(values, requires_grad=True)
    if operation == "relu":
        output = torch.relu(reference)
    elif operation == "gelu":
        output = torch.nn.functional.gelu(reference, approximate="tanh")
    else:
        output = torch.softmax(reference, dim=-1)
    output.backward(torch.tensor(seed))
    np.testing.assert_allclose(y.value().tolist(), output.detach().numpy(), atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(native_grad.tolist(), reference.grad.numpy(), atol=2e-6, rtol=2e-5)
    y.backward(st.from_dlpack(seed))
    np.testing.assert_allclose(x.grad().tolist(), reference.grad.numpy(), atol=2e-6, rtol=2e-5)


def test_complete_mlp_gradient_matches_pytorch():
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    values = [[-1.0, 0.5], [0.7, -0.3]]
    weights = [[0.2, -0.5, 0.8], [0.6, 0.3, -0.2]]
    bias_values = [[0.1, -0.2, 0.4]]
    target = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    def make(values):
        return st.AutogradTensor.variable(
            st.from_dlpack(np.array(values, dtype=np.float32))
        )
    x, weight, bias = map(make, [values, weights, bias_values])
    y = x.matmul(weight).add_row(bias).gelu().row_softmax()
    loss = y.mean_squared_error(st.AutogradTensor.constant(
        st.from_dlpack(np.array(target, dtype=np.float32))
    ))
    receipt = loss.backward()
    tx, tw, tb = [torch.tensor(v, requires_grad=True) for v in [values, weights, bias_values]]
    output = torch.softmax(torch.nn.functional.gelu(tx @ tw + tb, approximate="tanh"), dim=-1)
    expected = torch.nn.functional.mse_loss(output, torch.tensor(target))
    expected.backward()
    assert loss.item() == pytest.approx(expected.item(), abs=2e-6)
    assert receipt["leaf_gradient_count"] == 3
    for actual, reference in [(x, tx), (weight, tw), (bias, tb)]:
        np.testing.assert_allclose(actual.grad().tolist(), reference.grad.numpy(), rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize("shape", [(0, 3), (2, 0), (0, 0)])
def test_empty_nonlinear_graph_keeps_shape(shape):
    rows, cols = shape
    x = st.AutogradTensor.variable(st.Tensor.zeros(rows, cols))
    bias = st.AutogradTensor.variable(st.Tensor.zeros(1, cols))
    output = x.add_row(bias).relu().gelu().row_softmax()
    assert output.shape() == shape
    output.sum().backward()
    assert x.grad().shape() == shape
    assert bias.grad().tolist() == [[0.0] * cols]
