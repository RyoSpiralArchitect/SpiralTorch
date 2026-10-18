"""Transport/reconstruction for the Rust-owned portable geometry contract."""

from __future__ import annotations

import json
from typing import Any

from .geometry_autograd import _require_torch, torch


def _types() -> dict[str, type]:
    from . import fractional_autograd as f
    from . import geometry_autograd as g

    return {
        "spiraltorch.wave_gate_adapter.v1": g.WaveGateAdapter,
        "spiraltorch.wave_gate_adapter.v2": g.WaveGateAdapter,
        "spiraltorch.topos_resonator_adapter.v1": g.ToposResonatorAdapter,
        "spiraltorch.elliptic_residual_adapter.v1": g.EllipticResidualAdapter,
        "spiraltorch.elliptic_causal_residual_adapter.v1": g.EllipticCausalResidualAdapter,
        "spiraltorch.elliptic_gated_causal_residual_adapter.v1": g.EllipticGatedCausalResidualAdapter,
        "spiraltorch.elliptic_anchored_residual_adapter.v1": g.EllipticAnchoredResidualAdapter,
        "spiraltorch.fractional_memory_adapter.v1": f.FractionalMemoryAdapter,
        "spiraltorch.fractional_history_adapter.v1": f.FractionalHistoryAdapter,
        "spiraltorch.fractional_l2_history_adapter.v1": f.FractionalL2HistoryAdapter,
        "spiraltorch.fractional_gain_history_adapter.v1": f.FractionalGainHistoryAdapter,
        "spiraltorch.fractional_angle_gain_history_adapter.v1": f.FractionalAngleGainHistoryAdapter,
    }


def export_stack(stack: Any, *, base_model_identity: str) -> Any:
    from . import GeometryAdapterArtifact
    from .geometry_adapters import GeometryAdapterStack

    _require_torch()
    if type(stack) is not GeometryAdapterStack:
        raise TypeError("portable artifacts require a built-in GeometryAdapterStack")
    if stack._active_token is not None:
        raise RuntimeError("detach geometry adapters before exporting an artifact")
    stack._validate_ownership()
    allowed = _types()
    placements = []
    for path, adapter in zip(stack.paths, stack.adapters):
        if type(adapter) not in allowed.values():
            raise TypeError("portable artifacts support only exact built-in adapter types")
        recipe = adapter.get_extra_state()
        if allowed.get(recipe["schema"]) is not type(adapter):
            raise ValueError("adapter recipe differs from its concrete type")
        parameters = dict(adapter.named_parameters())
        buffers = dict(adapter.named_buffers())
        state = adapter.state_dict()
        if set(state) != {*parameters, *buffers, "_extra_state"}:
            raise ValueError("adapter state contains unsupported extra modules or state")
        tensors = []
        for name, value in state.items():
            if name == "_extra_state":
                continue
            if value.dtype != torch.float32 or value.layout != torch.strided:
                raise TypeError("portable geometry tensors must be dense float32")
            owner = parameters[name] if name in parameters else buffers[name]
            tensors.append({"name": name, "shape": list(value.shape),
                            "values": value.detach().cpu().reshape(-1).tolist(),
                            "requires_grad": owner.requires_grad,
                            "role": "parameter" if name in parameters else "buffer"})
        placements.append({"path": path, "recipe": recipe, "training": adapter.training,
                           "tensors": tensors})
    return GeometryAdapterArtifact.create_json(json.dumps({
        "schema": "spiraltorch.geometry_adapter_artifact.v1",
        "base_model_identity": base_model_identity,
        "policy": "full_prefix_f32_no_cache.v1",
        "training": stack.training, "placements": placements,
    }, allow_nan=False))


def restore_stack(artifact: Any, *, expected_base_model_identity: str, device: Any) -> Any:
    from . import GeometryAdapterArtifact
    from .geometry_adapters import GeometryAdapterStack

    _require_torch()
    if not isinstance(artifact, GeometryAdapterArtifact):
        raise TypeError("artifact must be a Rust-validated GeometryAdapterArtifact")
    if artifact.base_model_identity != expected_base_model_identity:
        raise ValueError("geometry artifact belongs to a different base model identity")
    record = json.loads(artifact.state_json())
    allowed = _types()
    placements = {}
    # Constructors may initialize elliptic Linear layers. Loading an artifact
    # must not advance the caller's training RNG or touch a default GPU device.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        for placement in record["placements"]:
            recipe = placement["recipe"]
            options = {key: value for key, value in recipe.items()
                       if key not in {"schema", "kernel", "warp"}}
            options.update(recipe.get("kernel", recipe.get("warp", {})))
            if "learnable_radius" in options:
                options["log_radius"] = 0.0
            adapter = allowed[recipe["schema"]](**options)
            state = {tensor["name"]: torch.tensor(tensor["values"], dtype=torch.float32).reshape(tensor["shape"])
                     for tensor in placement["tensors"]}
            adapter.load_state_dict({**state, "_extra_state": recipe}, strict=True)
            parameters = dict(adapter.named_parameters())
            for tensor in placement["tensors"]:
                if tensor["name"] in parameters:
                    parameters[tensor["name"]].requires_grad_(tensor["requires_grad"])
            placements[placement["path"]] = adapter
        result = GeometryAdapterStack(placements).to(device=device)
        result.train(record["training"])
        for adapter, placement in zip(result.adapters, record["placements"]):
            adapter.train(placement["training"])
    return result
