"""Same ordinary residual control on native, Torch transport and portable state."""

from array import array
import copy
import json
from pathlib import Path

import pytest
import spiraltorch as st

FIXTURE = json.loads((Path(__file__).resolve().parents[3] /
    "crates/st-nn/tests/fixtures/channel_affine.fixture.json").read_text())


@pytest.mark.parametrize("buffered", [False, True])
def test_shared_forward_vjp_owned_snapshot_and_artifact(buffered):
    f = FIXTURE
    kernel = st.ChannelAffineKernel(max_values=f["max_values"])
    cast = lambda name: array("f", f[name]) if buffered else list(f[name])
    x, gate, bias = (cast(name) for name in ("input", "gate", "bias"))
    operation = kernel.forward_buffer if buffered else kernel.forward
    batch = operation(x, gate, bias, f["rows"], f["features"])
    x[0] = gate[0] = bias[0] = 99.
    assert batch.output == f["output"]
    assert batch.output_buffer() == array("f", f["output"]).tobytes()
    actual = batch.vjp_buffer(cast("upstream")) if buffered else batch.vjp(cast("upstream"))
    for name, value in zip(("grad_input", "grad_gate", "grad_bias"), actual):
        assert value == (array("f", f[name]).tobytes() if buffered else f[name])
    assert batch.vjp([1.] * 6)[1] == [5., 7., 9.]
    assert kernel.execution_backend == "rust_f32_cpu"
    artifact = st.GeometryAdapterArtifact.create_json(json.dumps(f["artifact"]))
    loaded = st.GeometryAdapterArtifact.from_json(
        artifact.to_json(), artifact.artifact_id, f["artifact"]["base_model_identity"])
    assert loaded.layout_id == artifact.layout_id


def test_boundary_checks_before_sequence_extraction_and_strict_types():
    class Oversized:
        def __len__(self):
            return 100
        def __getitem__(self, _):
            raise AssertionError("must reject before reading values")
    kernel = st.ChannelAffineKernel(max_values=6)
    with pytest.raises(ValueError, match="budget"):
        kernel.forward(Oversized(), [1.], [0.], 100, 1)
    for bad in [True, False, -1, .5, "6", 2**32]:
        with pytest.raises((TypeError, ValueError, OverflowError)):
            st.ChannelAffineKernel(max_values=bad)
        with pytest.raises((TypeError, ValueError, OverflowError)):
            kernel.forward([1.], [1.], [0.], bad, 1)
    for value in [float("nan"), float("inf"), True]:
        with pytest.raises((TypeError, ValueError)):
            kernel.forward([value], [1.], [0.], 1, 1)
    for buffer in [array("d", [1.]), array("B", [1, 2, 3, 4]),
                   memoryview(array("f", [1., 2., 3., 4.]))[::2]]:
        with pytest.raises((TypeError, ValueError, BufferError)):
            kernel.forward_buffer(buffer, array("f", [1.]), array("f", [0.]), 1, 1)
    empty = kernel.forward([], [1., 2.], [0., 0.], 0, 2)
    assert empty.output == [] and empty.vjp([]) == ([], [0., 0.], [0., 0.])
    with pytest.raises(ValueError):
        empty.vjp([1.])
    np = pytest.importorskip("numpy")
    for value in [np.bool_(True), np.bool_(False)]:
        with pytest.raises(TypeError):
            st.ChannelAffineKernel(max_values=value)
        with pytest.raises(TypeError):
            kernel.forward([value], [1.], [0.], 1, 1)


@pytest.mark.parametrize("buffered", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_torch_forward_vjp_matches_plain_control(buffered, empty, monkeypatch):
    torch = pytest.importorskip("torch")
    from spiraltorch import channel_affine
    monkeypatch.setattr(channel_affine, "_buffer_transport_available", lambda: buffered)
    # A non-contiguous activation and direction exercise transport, not just lists.
    values = torch.linspace(-1., 1., 24).reshape(2, 4, 3).transpose(0, 1)
    if empty:
        values = values[:0]
    x = values.detach().requires_grad_()
    gate = torch.tensor([.2, -.3, .4], requires_grad=True)
    bias = torch.tensor([.4, .1, -.2], requires_grad=True)
    reference = [p.detach().clone().requires_grad_() for p in (x, gate, bias)]
    actual = st.channel_affine_autograd(x, gate, bias)
    expected = reference[0] * reference[1] + reference[2]
    direction = torch.ones_like(values).transpose(0, 1).contiguous().transpose(0, 1)
    actual.backward(direction)
    expected.backward(direction)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    for a, b in zip((x, gate, bias), reference):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-6, atol=1e-6)


def test_identity_start_same_first_update_as_wave_gate_and_input_version_guard():
    torch = pytest.importorskip("torch")
    affine, wave = st.ChannelAffineAdapter(3), st.WaveGateAdapter(3, curvature=-1.)
    assert sum(p.numel() for p in affine.parameters()) == sum(p.numel() for p in wave.parameters()) == 6
    value = torch.linspace(-1., 1., 18).reshape(2, 3, 3)
    for adapter in (affine, wave):
        output = adapter(value)
        assert torch.equal(output, value)
        output.square().sum().backward()
    for a, b in zip(affine.parameters(), wave.parameters()):
        assert torch.equal(a.grad, b.grad)
    x = value.detach().clone().requires_grad_()
    result = affine(x)
    with torch.no_grad():
        affine.gate.add_(1.)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        result.sum().backward()
    before = copy.deepcopy(affine.get_extra_state())
    with pytest.raises(ValueError):
        affine.set_extra_state(dict(before, max_values=1))
    assert affine.get_extra_state() == before


def test_shared_artifact_restores_residual_and_rejects_invalid_recipes():
    torch = pytest.importorskip("torch")
    f = FIXTURE
    artifact = st.GeometryAdapterArtifact.create_json(json.dumps(f["artifact"]))
    stack = st.GeometryAdapterStack.from_artifact(
        artifact, expected_base_model_identity=f["artifact"]["base_model_identity"])
    assert type(stack.adapters[0]) is st.ChannelAffineAdapter
    x = torch.tensor(f["input"], dtype=torch.float32).reshape(2, 3)
    assert torch.equal(stack.adapters[0](x), x + .5 * torch.tensor(f["output"]).reshape(2, 3))
    assert stack.export_artifact(base_model_identity=artifact.base_model_identity).artifact_id == artifact.artifact_id
    for field, value in [("features", 2), ("max_values", 2), ("max_values", True),
                         ("strength", float("inf")), ("kernel", {})]:
        state = copy.deepcopy(f["artifact"])
        state["placements"][0]["recipe"][field] = value
        with pytest.raises(ValueError):
            st.GeometryAdapterArtifact.create_json(json.dumps(state))


def test_residual_overflow_rejects_even_if_inner_affine_is_finite():
    torch = pytest.importorskip("torch")
    adapter = st.ChannelAffineAdapter(1)
    with torch.no_grad():
        adapter.gate.fill_(1.)
    value = torch.tensor([[3.2e38]])
    assert torch.isfinite(st.channel_affine_autograd(value, adapter.gate, adapter.bias)).all()
    with pytest.raises(ValueError, match="residual must remain finite"):
        adapter(value)
