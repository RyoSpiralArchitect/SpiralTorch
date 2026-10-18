"""Synthetic native adapter learning and exact continuation, without Torch."""

from array import array
import hashlib
import json

import spiraltorch as st


def main():
    # This identity names only this synthetic example, never a pretrained model.
    base_id = "sha256:" + hashlib.sha256(b"spiraltorch.native_geometry.synthetic.v1").hexdigest()
    state = {
        "schema": "spiraltorch.geometry_adapter_artifact.v1", "base_model_identity": base_id,
        "policy": "full_prefix_f32_no_cache.v1", "training": True,
        "placements": [{
            "path": "activation", "training": True,
            "recipe": {"schema": "spiraltorch.wave_gate_adapter.v1", "features": 3, "strength": .5,
                       "kernel": {"curvature": -1., "saturation": 1., "porosity": .05, "max_values": 64}},
            "tensors": [{"name": name, "shape": [3], "values": [0., 0., 0.],
                         "requires_grad": True, "role": "parameter"} for name in ("gate", "bias")],
        }],
    }
    runtime = st.NativeGeometryRuntime(st.GeometryAdapterArtifact.create_json(json.dumps(state)))
    x, target = array("f", [.2, -.3, .4, .5, .1, -.2]), array("f", [.1] * 6)
    losses = [runtime.train_mse_buffer("activation", x, target, 2, .2) for _ in range(40)]
    saved = runtime.artifact()
    restored = st.NativeGeometryRuntime(st.GeometryAdapterArtifact.from_json(
        saved.to_json(), saved.artifact_id, base_id))
    continued = runtime.train_mse_buffer("activation", x, target, 2, .2)
    assert continued == restored.train_mse_buffer("activation", x, target, 2, .2)
    assert runtime.artifact().artifact_id == restored.artifact().artifact_id
    assert losses[-1] < losses[0]
    print(json.dumps({"backend": runtime.execution_backend, "first_mse": losses[0],
                      "last_mse": losses[-1], "steps": 40, "continuation_steps": 1,
                      "restored_continuation_exact": True, "artifact_id": runtime.artifact().artifact_id,
                      "scope": "synthetic_activation_fit_not_language_quality"}, indent=2))


if __name__ == "__main__":
    main()
