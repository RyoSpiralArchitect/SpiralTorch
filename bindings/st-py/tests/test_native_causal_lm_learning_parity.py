"""Optional reference for real language-loss gradients, not an execution dependency."""
import json
from pathlib import Path
import runpy

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
fixtures = runpy.run_path(str(Path(__file__).with_name("test_native_geometry_runtime.py")))
pair = runpy.run_path(str(Path(__file__).with_name("test_native_causal_lm_parity.py")))["pair"]
KINDS = ("affine", "wave", "fixed", "learned", "elliptic", "anchored", "causal", "gated", "topos", *fixtures["FRACTIONAL_KINDS"])


def attach(reference, state):
    stack = st.GeometryAdapterStack.from_artifact(state, expected_base_model_identity=state.base_model_identity)
    record = json.loads(state.state_json())
    handles = []
    for placement, adapter in zip(record["placements"], stack.adapters):
        module = reference.get_submodule(placement["path"])

        def hook(_module, _inputs, output, adapter=adapter):
            value = output[0] if isinstance(output, tuple) else output
            value = adapter(value)
            return (value,) + output[1:] if isinstance(output, tuple) else value

        handles.append(module.register_forward_hook(hook))
    return stack, handles


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype,tied,scaled,inverse", [
    (torch.float32, True, True, False), (torch.bfloat16, False, True, True), (torch.float16, True, False, False),
])
def test_language_loss_all_recipe_gradients_and_update_match_reference(kind, dtype, tied, scaled, inverse):
    reference, base = pair(dtype=dtype, tied=tied, scaled=scaled, inverse=inverse, width=3, heads=1)
    reference.requires_grad_(False)
    record = json.loads(fixtures["artifact"](kind).state_json())
    record["base_model_identity"] = base.base_model_identity
    record["placements"][0]["path"] = "transformer.h.0.attn"
    if kind == "affine":
        record["placements"][0]["recipe"]["max_values"] = 24  # Two full 4-token reference windows.
    state = st.GeometryAdapterArtifact.create_json(json.dumps(record))
    native = base.with_geometry(state)
    stack, handles = attach(reference, state)
    tokens = torch.tensor([[1, 5, 7, 3], [2, 8, 4, 6]])
    try:
        expected = reference(tokens, labels=tokens, use_cache=False).loss
        expected.backward()
    finally:
        for handle in handles:
            handle.remove()
    actual = native.geometry_loss_gradients(tokens.flatten().tolist(), list(tokens.shape))
    assert actual.loss == pytest.approx(expected.item(), abs=3e-6)
    gradients = actual.parameter_gradients()["transformer.h.0.attn"]
    adapter = stack.adapters[0]
    assert set(gradients) == dict(adapter.named_parameters()).keys()
    for name, parameter in adapter.named_parameters():
        torch.testing.assert_close(torch.tensor(gradients[name]).reshape_as(parameter), parameter.grad, atol=2e-6, rtol=3e-4)
    assert all(p.grad is None for p in reference.parameters())
    torch.optim.SGD(adapter.parameters(), lr=.03).step()
    updated = native.step_geometry_sgd(actual, .03)
    restored = st.GeometryAdapterStack.from_artifact(updated.geometry_artifact(), expected_base_model_identity=base.base_model_identity)
    for (name, left), (other_name, right) in zip(adapter.named_parameters(), restored.adapters[0].named_parameters()):
        assert name == other_name
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
    for (name, left), (other_name, right) in zip(adapter.named_buffers(), restored.adapters[0].named_buffers()):
        assert name == other_name
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_joint_attention_mlp_and_block_gradients_use_one_loss_snapshot():
    reference, base = pair(width=3, heads=1)
    reference.requires_grad_(False)
    record = json.loads(fixtures["artifact"]("wave").state_json())
    record["base_model_identity"] = base.base_model_identity
    record["placements"] = []
    for kind, path in [("wave", "transformer.h.0.attn"), ("history_angle", "transformer.h.0.mlp"), ("elliptic", "transformer.h.1")]:
        p = json.loads(fixtures["artifact"](kind).state_json())["placements"][0]
        p["path"] = path
        record["placements"].append(p)
    state = st.GeometryAdapterArtifact.create_json(json.dumps(record))
    native = base.with_geometry(state)
    stack, handles = attach(reference, state)
    tokens = torch.tensor([[1, 5, 7, 3], [2, 8, 4, 6]])
    try:
        loss = reference(tokens, labels=tokens, use_cache=False).loss
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    gradient = native.geometry_loss_gradients(tokens.flatten().tolist(), list(tokens.shape))
    assert gradient.loss == pytest.approx(loss.item(), abs=3e-6)
    paths = json.loads(state.state_json())["placements"]
    for placement, adapter in zip(paths, stack.adapters):
        for name, p in adapter.named_parameters():
            actual = gradient.parameter_gradients()[placement["path"]][name]
            torch.testing.assert_close(torch.tensor(actual).reshape_as(p), p.grad, atol=2e-6, rtol=3e-4)
    updated = native.step_geometry_sgd(gradient, .03)
    torch.optim.SGD(stack.parameters(), lr=.03).step()
    restored = st.GeometryAdapterStack.from_artifact(updated.geometry_artifact(), expected_base_model_identity=base.base_model_identity)
    for (name, left), (other, right) in zip(stack.named_parameters(), restored.named_parameters()):
        assert name == other
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
