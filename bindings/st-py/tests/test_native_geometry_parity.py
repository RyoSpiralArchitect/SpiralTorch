"""Optional Torch comparison; native learning itself has no Torch dependency."""

from pathlib import Path
import runpy

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
artifact = runpy.run_path(str(Path(__file__).with_name("test_native_geometry_runtime.py")))["artifact"]


@pytest.mark.parametrize("kind", ["affine", "wave", "fixed", "learned"])
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
    torch.testing.assert_close(torch.tensor(dx).reshape_as(x), x.grad, atol=1e-6, rtol=1e-6)
    for name, parameter in adapter.named_parameters():
        torch.testing.assert_close(torch.tensor(gradients[name]).reshape_as(parameter), parameter.grad, atol=1e-6, rtol=1e-6)
    native.step_sgd(batch, direction.flatten().tolist(), .02)
    torch.optim.SGD(adapter.parameters(), lr=.02).step()
    restored = st.GeometryAdapterStack.from_artifact(native.artifact(), expected_base_model_identity=saved.base_model_identity)
    for (left_name, left), (right_name, right) in zip(adapter.named_parameters(), restored.adapters[0].named_parameters()):
        assert left_name == right_name
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)
