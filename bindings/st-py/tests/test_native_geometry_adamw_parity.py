"""Optional optimizer comparison; Torch is never imported by the native path."""
import json
from pathlib import Path
import runpy

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
fixtures = runpy.run_path(str(Path(__file__).with_name("test_native_geometry_runtime.py")))
KINDS = ("affine", "wave", "fixed", "learned", "elliptic", "anchored", "causal", "gated", "topos", *fixtures["FRACTIONAL_KINDS"])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("clip", [None, .01])
@pytest.mark.parametrize("betas,epsilon", [((.9, .999), 1e-8), ((.5, .75), .25)])
def test_adamw_moments_updates_and_resume_match_optional_torch(kind, clip, betas, epsilon):
    artifact = fixtures["artifact"](kind)
    runtime = st.NativeGeometryRuntime(artifact)
    native_opt = st.NativeGeometryAdamW(artifact, beta1=betas[0], beta2=betas[1], epsilon=epsilon, max_gradient_norm=clip)
    stack = st.GeometryAdapterStack.from_artifact(artifact, expected_base_model_identity=artifact.base_model_identity)
    adapter = stack.adapters[0]
    parameters = dict(adapter.named_parameters())
    reference = torch.optim.AdamW(parameters.values(), lr=.01, betas=betas, eps=epsilon, weight_decay=.02, foreach=False)
    x = torch.tensor([[[.2, -.3, .4], [.5, .1, -.2]]])
    for step in range(8):
        reference.zero_grad(set_to_none=True)
        direction = torch.tensor([[[.3, .5, -.4], [.2, -.1, .6]]]) * (1. if step % 2 else -.5)
        batch = runtime.forward_sequence("site", x.flatten().tolist(), (1, 2))
        actual = adapter(x)
        actual.backward(direction)
        if clip is not None:
            # Torch's epsilon in its scale differs from the shared ST clip contract.
            # Normal-size norms and bounded tolerances below do not assert bit parity.
            torch.nn.utils.clip_grad_norm_(parameters.values(), clip)
        runtime, native_opt = runtime.step_adamw(batch, direction.flatten().tolist(), native_opt, .01, weight_decay=.02)
        reference.step()
        saved = runtime.artifact()
        state = json.loads(native_opt.to_json())
        values = {p["name"]: p for p in json.loads(saved.state_json())["placements"][0]["tensors"]}
        for name, parameter in parameters.items():
            torch.testing.assert_close(torch.tensor(values[name]["values"]).reshape_as(parameter), parameter, atol=3e-5, rtol=3e-4)
        for slot in state["slots"]:
            parameter = parameters[slot["parameter"]]
            expected = reference.state[parameter]
            assert slot["steps"] == int(expected["step"])
            torch.testing.assert_close(torch.tensor(slot["first"]).reshape_as(parameter), expected["exp_avg"], atol=3e-6, rtol=3e-4)
            torch.testing.assert_close(torch.tensor(slot["second"]).reshape_as(parameter), expected["exp_avg_sq"], atol=3e-6, rtol=3e-4)
        if step == 3:
            native_opt = st.NativeGeometryAdamW.from_json(native_opt.to_json(), native_opt.checkpoint_id, saved)
            runtime = st.NativeGeometryRuntime(st.GeometryAdapterArtifact.from_json(saved.to_json(), saved.artifact_id, saved.base_model_identity))
