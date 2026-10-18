"""Synthetic full-context adapter learning; no Torch, model download or KV cache."""

from array import array
import hashlib
import json

import spiraltorch as st


def main():
    base_id = "sha256:" + hashlib.sha256(b"spiraltorch.native_sequence.synthetic.v1").hexdigest()
    parameters = [
        ("orientation.weight", [2, 3], [.2, -.1, .15, .05, .3, -.2]),
        ("orientation.bias", [2], [.1, -.2]),
        ("readout.weight", [3, 9], [0.] * 27),
        ("raw_mix", [], [0.]),
    ]
    state = {
        "schema": "spiraltorch.geometry_adapter_artifact.v1",
        "base_model_identity": base_id, "policy": "full_prefix_f32_no_cache.v1", "training": True,
        "placements": [{
            "path": "activation", "training": True,
            "recipe": {"schema": "spiraltorch.elliptic_gated_causal_residual_adapter.v1",
                       "features": 3, "strength": .4,
                       "warp": {"curvature_radius": 1.3, "sheet_count": 3, "spin_harmonics": 2}},
            "tensors": [{"name": name, "shape": shape, "values": values,
                         "requires_grad": True, "role": "parameter"} for name, shape, values in parameters],
        }],
    }
    runtime = st.NativeGeometryRuntime(st.GeometryAdapterArtifact.create_json(json.dumps(state)))
    x = array("f", [(i * 7 % 19) * .05 - .3 for i in range(18)])
    target = array("f", [.1] * 18)
    losses = [runtime.train_sequence_mse_buffer("activation", x, target, (2, 3), .2) for _ in range(40)]
    saved = runtime.artifact()
    restored = st.NativeGeometryRuntime(st.GeometryAdapterArtifact.from_json(
        saved.to_json(), saved.artifact_id, base_id))
    next_loss = runtime.train_sequence_mse_buffer("activation", x, target, (2, 3), .2)
    assert next_loss == restored.train_sequence_mse_buffer("activation", x, target, (2, 3), .2)
    assert runtime.artifact().artifact_id == restored.artifact().artifact_id
    capture = runtime.forward_sequence_buffer("activation", x, (2, 3))
    changed = list(x[:6]) + [v + 3. for v in x[6:]]
    assert capture.output[:6] == runtime.forward_sequence("activation", changed, (2, 3)).output[:6]
    assert capture.vjp([.2, -.4, .1] + [0.] * 15)[0][3:] == [0.] * 15
    assert losses[-1] < losses[0]
    print(json.dumps({"backend": runtime.execution_backend, "shape": capture.sequence_shape,
                      "first_mse": losses[0], "last_mse": losses[-1], "steps": 40,
                      "continuation_steps": 1, "restored_continuation_exact": True,
                      "future_and_cross_batch_isolation": True,
                      "artifact_id": runtime.artifact().artifact_id,
                      "scope": "synthetic_activation_fit_not_language_quality"}, indent=2))


if __name__ == "__main__":
    main()
