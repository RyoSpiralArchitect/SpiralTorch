import pytest
import spiraltorch as st

pytestmark = pytest.mark.skipif(getattr(st, "_rs", None) is None, reason="requires native Rust")


def test_integer_embedding_and_tied_gradients_match_torch():
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    values = np.random.default_rng(13).normal(size=(7, 3)).astype(np.float32)
    ids = [6, 0, 6, 2]
    table = st.AutogradTensor.variable(st.from_dlpack(values))
    gathered = table.gather_rows(ids)
    ids[0] = 1  # The graph owns an immutable ID snapshot.
    output = gathered.matmul(table.transpose())
    output.sum().backward()
    reference = torch.tensor(values, requires_grad=True)
    expected = torch.nn.functional.embedding(torch.tensor([6, 0, 6, 2]), reference) @ reference.T
    expected.sum().backward()
    np.testing.assert_allclose(output.value().tolist(), expected.detach().numpy(), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(table.grad().tolist(), reference.grad.numpy(), rtol=2e-6, atol=2e-6)


def test_scatter_duplicates_empty_and_atomic_failure():
    table = st.AutogradTensor.variable(st.Tensor(3, 2, [1, 2, 3, 4, 5, 6]))
    scatter = table.scatter_add_rows([2, 0, 2], 4)
    assert scatter.value().tolist() == [[3, 4], [0, 0], [6, 8], [0, 0]]
    scatter.backward(st.Tensor(4, 2, list(range(8))))
    assert table.grad().tolist() == [[4, 5], [0, 1], [4, 5]]
    gathered = table.gather_rows([2, 2])
    gathered.sum().backward()
    previous = table.grad().tolist()
    maximum = float.fromhex("0x1.fffffep127")
    with pytest.raises(ValueError):
        gathered.backward(st.Tensor(2, 2, [maximum] * 4))
    assert table.grad().tolist() == previous
    assert gathered.grad().tolist() == [[1, 1], [1, 1]]
    empty = table.gather_rows([])
    assert empty.shape() == (0, 2)
    empty.sum().backward()
    assert table.grad().tolist() == previous
    assert st.Tensor(0, 2, []).scatter_add_rows([], 3).tolist() == [[0, 0]] * 3


@pytest.mark.parametrize("bad", [[-1], [0.5], [True], [float("nan")], [float("inf")], ["1"], [2**100], [3]])
def test_bad_indices_are_not_coerced_or_clamped(bad):
    tensor = st.Tensor(3, 1, [1, 2, 3])
    graph = st.AutogradTensor.variable(tensor)
    for source in (tensor, graph):
        with pytest.raises((ValueError, TypeError, OverflowError)):
            source.gather_rows(bad)
    assert graph.grad() is None


def test_native_tensor_transpose_and_cpu_contract():
    np = pytest.importorskip("numpy")
    values = np.arange(6, dtype=np.float32).reshape(3, 2)
    # DLPack currently admits row-major storage only; Rust tests cover ColMajor.
    tensor = st.from_dlpack(np.ascontiguousarray(values.T)).transpose()
    assert tensor.gather_rows([2, 0, 2]).tolist() == [[4, 5], [0, 1], [4, 5]]
    assert tensor.scatter_add_rows([2, 0, 2], 3).tolist() == [[2, 3], [0, 0], [4, 6]]
    with pytest.raises(ValueError):
        tensor.gather_rows([0], backend="typo")
