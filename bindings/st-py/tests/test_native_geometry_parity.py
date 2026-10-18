"""Optional Torch comparison; native learning itself has no Torch dependency."""

from pathlib import Path
import json
import runpy

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
artifact = runpy.run_path(str(Path(__file__).with_name("test_native_geometry_runtime.py")))["artifact"]


@pytest.mark.parametrize("kind", ["affine", "wave", "fixed", "learned", "elliptic", "anchored"])
def test_native_residual_pullback_and_sgd_match_optional_torch_transport(kind):
    saved = artifact(kind)
    native = st.NativeGeometryRuntime(saved)
    stack = st.GeometryAdapterStack.from_artifact(saved, expected_base_model_identity=saved.base_model_identity)
    adapter = stack.adapters[0]
    x = torch.tensor([[.2, -.3, .4], [.5, .1, -.2]], requires_grad=True)
    direction = torch.tensor([[.3, .5, -.4], [.2, -.1, .6]])
    batch = native.forward("site", x.detach().flatten().tolist(), 2)
    actual = adapter(x)
    torch.testing.assert_close(torch.tensor(batch.output).reshape_as(actual), actual, atol=1e-6, rtol=1e-6)
    actual.backward(direction)
    dx, gradients = batch.vjp(direction.flatten().tolist())
    assert set(gradients) == dict(adapter.named_parameters()).keys()
    torch.testing.assert_close(torch.tensor(dx).reshape_as(x), x.grad, atol=1e-6, rtol=1e-6)
    for name, parameter in adapter.named_parameters():
        torch.testing.assert_close(torch.tensor(gradients[name]).reshape_as(parameter), parameter.grad, atol=1e-6, rtol=1e-6)
    native.step_sgd(batch, direction.flatten().tolist(), .02)
    torch.optim.SGD(adapter.parameters(), lr=.02).step()
    restored = st.GeometryAdapterStack.from_artifact(native.artifact(), expected_base_model_identity=saved.base_model_identity)
    for (left_name, left), (right_name, right) in zip(adapter.named_parameters(), restored.adapters[0].named_parameters()):
        assert left_name == right_name
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("kind", ["elliptic", "anchored"])
@pytest.mark.parametrize("width", [1, 7, 64, 768])
def test_elliptic_projection_and_readout_cover_hidden_widths_and_leading_axes(kind, width):
    generator = torch.Generator().manual_seed(47)
    state = json.loads(artifact(kind).state_json())
    placement = state["placements"][0]
    placement["recipe"]["features"] = width
    shapes = {"orientation.weight": [2, width], "orientation.bias": [2],
              "readout.weight": [width, 9]}
    for parameter in placement["tensors"]:
        if parameter["name"] in shapes:
            shape = shapes[parameter["name"]]
            parameter["shape"] = shape
            parameter["values"] = (torch.randn(shape, generator=generator) * .02).flatten().tolist()
    saved = st.GeometryAdapterArtifact.create_json(json.dumps(state))
    native = st.NativeGeometryRuntime(saved)
    adapter = st.GeometryAdapterStack.from_artifact(saved, expected_base_model_identity=saved.base_model_identity).adapters[0]
    x = (torch.randn(2, 4, width, generator=generator) * .1).requires_grad_()
    direction = torch.randn(x.shape, generator=generator) * .1
    batch = native.forward("site", x.detach().flatten().tolist(), 8)
    actual = adapter(x)
    torch.testing.assert_close(torch.tensor(batch.output).reshape_as(x), actual, atol=1e-5, rtol=1e-5)
    actual.backward(direction)
    dx, parameters = batch.vjp(direction.flatten().tolist())
    torch.testing.assert_close(torch.tensor(dx).reshape_as(x), x.grad, atol=1e-5, rtol=1e-5)
    assert set(parameters) == dict(adapter.named_parameters()).keys()
    for name, parameter in adapter.named_parameters():
        torch.testing.assert_close(torch.tensor(parameters[name]).reshape_as(parameter), parameter.grad, atol=1e-5, rtol=1e-5)
