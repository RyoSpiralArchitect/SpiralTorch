"""Synthetic safetensors -> native projection -> geometry learning, without Torch."""

from array import array
import hashlib
import json
import struct

import spiraltorch as st


def main():
    # A small synthetic checkpoint, not downloaded or pretrained model weights.
    values = [.5, -.2, .1, .3, .4, -.1]
    header = json.dumps({"projection.weight": {
        "dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
        "__metadata__": {"scope": "synthetic_projection"}}).encode()
    payload = struct.pack("<Q", len(header)) + header + struct.pack("<6f", *values)
    base_id = "sha256:" + hashlib.sha256(payload).hexdigest()
    archive = st.SafeTensorArchive.from_bytes(payload)
    weight = archive.tensor("projection.weight")
    x = st.Tensor([[.2, -.3], [.5, .1]])
    activation = st.NdTensor.from_tensor(x.matmul(weight.to_tensor(), backend="cpu"))
    state = {
        "schema": "spiraltorch.geometry_adapter_artifact.v1", "base_model_identity": base_id,
        "policy": "full_prefix_f32_no_cache.v1", "training": True,
        "placements": [{
            "path": "projection", "training": True,
            "recipe": {"schema": "spiraltorch.wave_gate_adapter.v1", "features": 3, "strength": .5,
                       "kernel": {"curvature": -1., "saturation": 1., "porosity": .05, "max_values": 64}},
            "tensors": [{"name": name, "shape": [3], "values": [0., 0., 0.],
                         "requires_grad": True, "role": "parameter"} for name in ("gate", "bias")],
        }],
    }
    runtime = st.NativeGeometryRuntime(st.GeometryAdapterArtifact.create_json(json.dumps(state)))
    inputs, targets = memoryview(activation.values_buffer()).cast("f"), array("f", [.1] * 6)
    losses = [runtime.train_mse_buffer("projection", inputs, targets, 2, .2) for _ in range(40)]
    saved = runtime.artifact()
    restored = st.NativeGeometryRuntime(st.GeometryAdapterArtifact.from_json(
        saved.to_json(), saved.artifact_id, base_id))
    continuation = runtime.train_mse_buffer("projection", inputs, targets, 2, .2)
    assert continuation == restored.train_mse_buffer("projection", inputs, targets, 2, .2)
    assert runtime.artifact().artifact_id == restored.artifact().artifact_id
    assert losses[-1] < losses[0]
    assert weight.values() == archive.tensor("projection.weight").values()
    print(json.dumps({"scope": "synthetic_frozen_projection_not_hf_model_execution",
                      "backend": runtime.execution_backend, "checkpoint_id": base_id,
                      "weight_shape": weight.shape, "first_mse": losses[0], "last_mse": losses[-1],
                      "steps": 40, "base_weights_unchanged": True, "restored_continuation_exact": True}, indent=2))


if __name__ == "__main__":
    main()
