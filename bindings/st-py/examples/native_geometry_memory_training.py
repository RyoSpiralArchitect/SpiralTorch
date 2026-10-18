"""Synthetic native memory/resonance fitting without Torch or model downloads."""

import argparse
from array import array
import hashlib
import json

import spiraltorch as st


KINDS = {
    "topos": "topos_resonator", "memory": "fractional_memory", "history": "fractional_history",
    "history_l2": "fractional_l2_history", "history_gain": "fractional_gain_history",
    "history_angle": "fractional_angle_gain_history",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=KINDS, default="history_angle")
    kind = parser.parse_args().recipe
    base_id = "sha256:" + hashlib.sha256(b"spiraltorch.native_memory.synthetic.v1").hexdigest()
    recipe = {"schema": "spiraltorch." + KINDS[kind] + "_adapter.v1", "features": 3, "strength": .4}
    parameters = [("gate", [3], [0.] * 3)]
    if kind == "topos":
        recipe["kernel"] = {"coupling": .2, "iterations": 4, "saturation": 1., "porosity": .05, "max_values": 96}
    else:
        recipe["kernel"] = {"kernel_len": 5, "step": .8, "max_values": 96, "max_products": 480}
        parameters.append(("history_angle" if kind == "history_angle" else "log_alpha", [], [-.2]))
        if kind != "memory":
            parameters.append(("local_gate", [3], [0.] * 3))
        if kind == "history_l2":
            recipe["gain"] = .8
        if kind in {"history_gain", "history_angle"}:
            recipe["lag_window"] = [1, 4]
            parameters.append(("log_gain", [], [.1]))
    state = {
        "schema": "spiraltorch.geometry_adapter_artifact.v1", "base_model_identity": base_id,
        "policy": "full_prefix_f32_no_cache.v1", "training": True,
        "placements": [{"path": "activation", "training": True, "recipe": recipe,
                        "tensors": [{"name": name, "shape": shape, "values": values,
                                     "requires_grad": True, "role": "parameter"}
                                    for name, shape, values in parameters]}],
    }
    runtime = st.NativeGeometryRuntime(st.GeometryAdapterArtifact.create_json(json.dumps(state)))
    x = array("f", [(i * 7 % 19) * .05 - .3 for i in range(18)])
    target = array("f", [.1] * 18)
    assert runtime.forward_sequence_buffer("activation", x, (2, 3)).output == list(x)
    losses = [runtime.train_sequence_mse_buffer("activation", x, target, (2, 3), .2) for _ in range(40)]
    saved = runtime.artifact()
    restored = st.NativeGeometryRuntime(st.GeometryAdapterArtifact.from_json(saved.to_json(), saved.artifact_id, base_id))
    loss = runtime.train_sequence_mse_buffer("activation", x, target, (2, 3), .2)
    assert loss == restored.train_sequence_mse_buffer("activation", x, target, (2, 3), .2)
    assert runtime.artifact().artifact_id == restored.artifact().artifact_id
    assert losses[-1] < losses[0]
    print(json.dumps({"recipe": kind, "backend": runtime.execution_backend,
                      "first_mse": losses[0], "last_mse": losses[-1], "steps": 40,
                      "continuation_steps": 1, "restored_continuation_exact": True,
                      "artifact_id": runtime.artifact().artifact_id,
                      "scope": "synthetic_activation_fit_not_language_quality"}, indent=2))


if __name__ == "__main__":
    main()
