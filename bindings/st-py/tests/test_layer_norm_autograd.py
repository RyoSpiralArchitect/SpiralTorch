import pytest
import spiraltorch as st


pytestmark = pytest.mark.skipif(
    getattr(st, "_rs", None) is None,
    reason="requires the native Rust implementation",
)


@pytest.mark.parametrize("rows,cols,offset,epsilon", [
    (2, 3, 0.0, 1e-5), (3, 7, 10000.0, 1e-5),
    (2, 1, 0.0, 1e-5), (2, 5, 0.0, 0.0),
])
def test_layer_norm_values_and_all_gradients_match_torch(rows, cols, offset, epsilon):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(17)
    values = (rng.normal(size=(rows, cols)) + offset).astype(np.float32)
    gamma_values = rng.normal(size=(1, cols)).astype(np.float32)
    beta_values = rng.normal(size=(1, cols)).astype(np.float32)
    seed = rng.normal(size=(rows, cols)).astype(np.float32)
    x, gamma, beta = [st.AutogradTensor.variable(st.from_dlpack(v))
                      for v in (values, gamma_values, beta_values)]
    y = x.layer_norm_affine(gamma, beta, epsilon=epsilon)
    assert y.operation_name() == "layer_norm_affine"
    tx = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    tg = torch.tensor(gamma_values[0], dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(beta_values[0], dtype=torch.float64, requires_grad=True)
    expected = torch.nn.functional.layer_norm(tx, (cols,), tg, tb, float(np.float32(epsilon)))
    expected.backward(torch.tensor(seed, dtype=torch.float64))
    np.testing.assert_allclose(y.value().tolist(), expected.detach().numpy(), rtol=2e-5, atol=2e-6)
    raw = x.value().layer_norm_affine_backward(gamma.value(), st.from_dlpack(seed), epsilon=epsilon)
    for index, (parent, reference) in enumerate(zip((x, gamma, beta), (tx, tg, tb))):
        vjp = y.vector_jacobian_product(parent, st.from_dlpack(seed))
        assert parent.grad() is None
        expected_grad = reference.grad.numpy().reshape(parent.shape())
        np.testing.assert_allclose(vjp.tolist(), expected_grad, rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(raw[index].tolist(), expected_grad, rtol=3e-5, atol=3e-6)
    y.backward(st.from_dlpack(seed))
    for parent, reference in zip((x, gamma, beta), (tx, tg, tb)):
        np.testing.assert_allclose(parent.grad().tolist(), reference.grad.numpy().reshape(parent.shape()), rtol=3e-5, atol=3e-6)


def test_layer_norm_frozen_affine_and_atomic_overflow():
    maximum = float.fromhex("0x1.fffffep127")
    x = st.AutogradTensor.variable(st.Tensor(2, 1, [1.0, 2.0]))
    gamma = st.AutogradTensor.constant(st.Tensor(1, 1, [maximum]))
    frozen_beta = st.AutogradTensor.constant(st.Tensor.zeros(1, 1))
    output = x.layer_norm_affine(gamma, frozen_beta)
    seed = st.Tensor(2, 1, [maximum, maximum])
    output.backward(seed)
    assert x.grad().tolist() == [[0.0], [0.0]]
    beta = st.AutogradTensor.variable(st.Tensor.zeros(1, 1))
    output = x.layer_norm_affine(gamma, beta)
    output.sum().backward()
    before = [parent.grad().tolist() for parent in (x, beta, output)]
    with pytest.raises(ValueError):
        output.backward(seed)
    assert [parent.grad().tolist() for parent in (x, beta, output)] == before


@pytest.mark.parametrize("values", [
    [[10000.0, 10000.0, 10000.0]],
    [[3e38, -3e38, 0.0]],
])
def test_layer_norm_extreme_rows_match_double_precision_reference(values):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    values = np.asarray(values, dtype=np.float32)
    gamma_values = [[1.5, -0.75, 0.35]]
    x = st.AutogradTensor.variable(st.from_dlpack(values))
    gamma = st.AutogradTensor.variable(st.Tensor(1, 3, gamma_values[0]))
    beta = st.AutogradTensor.variable(st.Tensor.zeros(1, 3))
    seed = np.asarray([[0.2, -0.15, 0.35]], dtype=np.float32)
    y = x.layer_norm_affine(gamma, beta)
    y.backward(st.from_dlpack(seed))
    tx = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    tg = torch.tensor(np.asarray(gamma_values[0], dtype=np.float32), dtype=torch.float64, requires_grad=True)
    tb = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    expected = torch.nn.functional.layer_norm(tx, (3,), tg, tb, float(np.float32(1e-5)))
    expected.backward(torch.tensor(seed, dtype=torch.float64))
    np.testing.assert_allclose(y.value().tolist(), expected.detach().numpy(), rtol=3e-5, atol=2e-6)
    for parent, reference in zip((x, gamma, beta), (tx, tg, tb)):
        # Tiny input gradients on huge rows are measured, not hidden by a unit-scale atol.
        atol = 1e-43 if np.max(np.abs(values)) > 1e30 and parent is x else 2e-6
        np.testing.assert_allclose(parent.grad().tolist(), reference.grad.numpy().reshape(parent.shape()), rtol=3e-5, atol=atol)


def test_normalized_residual_classifier_matches_torch_end_to_end():
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(29)
    arrays = [rng.normal(size=shape).astype(np.float32) for shape in
              [(2, 3), (3, 3), (1, 3), (1, 3), (1, 3), (3, 2)]]
    native = [st.AutogradTensor.variable(st.from_dlpack(value)) for value in arrays]
    x, weight, bias, gamma, beta, head = native
    residual = x.matmul(weight).add_row(bias).add(x)
    logits = residual.layer_norm_affine(gamma, beta).gelu().matmul(head)
    loss = logits.cross_entropy_with_logits([0, 1])
    loss.backward()
    reference = [torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in arrays]
    tx, tw, tb, tg, toffset, th = reference
    hidden = torch.nn.functional.layer_norm(tx @ tw + tb + tx, (3,), tg[0], toffset[0], float(np.float32(1e-5)))
    expected = torch.nn.functional.cross_entropy(torch.nn.functional.gelu(hidden, approximate="tanh") @ th, torch.tensor([0, 1]))
    expected.backward()
    assert loss.item() == pytest.approx(expected.item(), rel=3e-5, abs=2e-6)
    for parent, tensor in zip(native, reference):
        np.testing.assert_allclose(parent.grad().tolist(), tensor.grad.numpy(), rtol=2e-4, atol=3e-6)


def test_layer_norm_learns_via_native_sgd():
    input_tensor = st.Tensor(3, 3, [0.4, -0.8, 1.2, -0.3, 0.9, -1.1, 0.7, 0.1, -0.2])
    x = st.AutogradTensor.constant(input_tensor)
    target = st.AutogradTensor.constant(input_tensor.layer_norm_affine(
        st.Tensor(1, 3, [1.7, 0.5, -0.8]), st.Tensor(1, 3, [0.2, -0.3, 0.6]),
    ))
    optimizer = st.AutogradSgd([
        st.AutogradTensor.variable(st.Tensor(1, 3, [1.0] * 3)),
        st.AutogradTensor.variable(st.Tensor.zeros(1, 3)),
    ], 0.1)
    losses = []
    for _ in range(400):
        gamma, beta = optimizer.parameters()
        loss = x.layer_norm_affine(gamma, beta).mean_squared_error(target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0] * 1e-4


def test_layer_norm_rejects_invalid_arguments_and_preserves_empty_shape():
    x = st.AutogradTensor.variable(st.Tensor.zeros(0, 3))
    gamma = st.AutogradTensor.variable(st.Tensor(1, 3, [1.0] * 3))
    beta = st.AutogradTensor.variable(st.Tensor.zeros(1, 3))
    x.layer_norm_affine(gamma, beta).sum().backward()
    assert x.grad().shape() == (0, 3)
    assert beta.grad().tolist() == [[0.0] * 3]
    for epsilon in [-1.0, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            x.layer_norm_affine(gamma, beta, epsilon=epsilon)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        x.layer_norm_affine(st.AutogradTensor.variable(st.Tensor.zeros(2, 3)), beta)
