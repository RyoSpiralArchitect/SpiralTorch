"""Optional format interoperability only; production loading never imports Torch."""

import pytest

torch = pytest.importorskip("torch")
safe_torch = pytest.importorskip("safetensors.torch")
import spiraltorch as st  # noqa: E402


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(), (6,), (2, 3), (1, 2, 3), (2, 0, 3)])
def test_official_writer_and_loader_agree(dtype, shape):
    count = 1
    for dimension in shape:
        count *= dimension
    source = ((torch.arange(count, dtype=torch.float32) - 3) * .125).reshape(shape).to(dtype)
    payload = safe_torch.save({"weight": source}, metadata={"format": "pt"})
    archive = st.SafeTensorArchive.from_bytes(payload)
    result = archive.tensor("weight")
    assert result.shape == list(shape)
    assert result.values() == safe_torch.load(payload)["weight"].float().reshape(-1).tolist()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_extreme_finite_values_and_subnormals(dtype):
    info = torch.finfo(dtype)
    source = torch.tensor([0., -0., info.max, info.min, info.tiny, info.tiny / 2, 1., -1.], dtype=dtype)
    payload = safe_torch.save({"weight": source})
    result = st.SafeTensorArchive.from_bytes(payload).tensor("weight")
    actual = torch.frombuffer(result.values_buffer(), dtype=torch.float32)
    assert torch.equal(actual.view(torch.int32), source.float().view(torch.int32))


def test_named_mixed_dtype_checkpoint_and_projection():
    source = {
        "projection.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3) / 8,
        "projection.bias": torch.tensor([.5, -.5, .25], dtype=torch.bfloat16),
        "counter": torch.tensor(17, dtype=torch.int64),
    }
    archive = st.SafeTensorArchive.from_bytes(safe_torch.save(source, metadata={"purpose": "interop"}))
    assert archive.names() == sorted(source)
    weight = archive.tensor("projection.weight").to_tensor()
    x = [[2., -1.], [.5, 3.]]
    actual = st.Tensor(x).matmul(weight, backend="cpu")
    actual.add_row_inplace(archive.tensor("projection.bias").values())
    expected = torch.tensor(x) @ source["projection.weight"] + source["projection.bias"].float()
    assert actual.tolist() == expected.tolist()
    assert archive.info("counter")["dtype"] == "I64"
    with pytest.raises(ValueError, match="supported: F32, F16, BF16"):
        archive.tensor("counter")
