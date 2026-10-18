#!/usr/bin/env python3
"""Compare only trusted, locally produced HF checkpoints from this probe.

RNG companions use pickle. Never run this on downloaded/untrusted checkpoints.
The comparison does not authenticate artifacts or certify arbitrary HF resume.
"""
import argparse
import hashlib
import json
from pathlib import Path

def equal(a, b):
    import numpy as np
    import torch
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return (a.dtype == b.dtype and a.shape == b.shape
                and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                                b.contiguous().reshape(-1).view(torch.uint8)))
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def companion_names(results):
    device = results[0].get("device")
    if device not in ("cpu", "mps") or any(r.get("device") != device for r in results):
        raise ValueError("continuation must use the same supported CPU/MPS device")
    names = ["spiraltorch-geometry.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"]
    if device == "mps":
        names.append("spiraltorch-mps-rng.pt")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("full", "partial", "resumed"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    paths = (args.full, args.partial, args.resumed)
    results = [json.loads((p / "result.json").read_text()) for p in paths]
    names = companion_names(results)
    import torch
    full, partial, resumed = results
    checks = {}
    for key in ("device", "runtime_identity", "tokenized_dataset_identity", "initial_artifact_id",
                "seed", "max_steps", "context_length", "batch_size", "learning_rate",
                "max_grad_norm", "run_config", "base_parameter_hash"):
        checks[key] = all(result[key] == full[key] for result in results)
    checks["steps"] = [r["step"] for r in results] == [32, 16, 32]
    checks["base_and_inference"] = all(r["base_unchanged"] and r["restored_logits_exact"] for r in results)
    for stage, left, right, step in (("stop", args.full, args.partial, 16),
                                    ("final", args.full, args.resumed, 32)):
        for name in names:
            a, b = (p / f"checkpoint-{step}" / name for p in (left, right))
            if name.endswith(".json"):
                checks[f"{stage}/{name}"] = a.read_bytes() == b.read_bytes()
            else:
                checks[f"{stage}/{name}"] = equal(*(
                    torch.load(p, map_location="cpu", weights_only=name != "rng_state.pth")
                    for p in (a, b)))
    checks["final_export"] = full["artifact_id"] == resumed["artifact_id"]
    checks["final_eval_loss"] = full["trained_geometry_eval_loss"] == resumed["trained_geometry_eval_loss"]
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    print(json.dumps({
        "schema": "spiraltorch.hf_affine_continuation_check.v1", "exact_checks": checks,
        "full_artifact_id": full["artifact_id"], "resumed_artifact_id": resumed["artifact_id"],
        "loss": full["trained_geometry_eval_loss"],
        "private_result_ids": ["sha256:" + hashlib.sha256((p / "result.json").read_bytes()).hexdigest()
                               for p in paths],
    }, indent=2))


if __name__ == "__main__":
    main()
