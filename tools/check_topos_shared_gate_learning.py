#!/usr/bin/env python3
"""Compare the bounded native shared-gate probe with independent Torch learning.

Torch owns a (1, F) leaf gate and differentiates its own broadcast and mean loss.
The two 100-update trajectories are not reset to Rust weights between updates.
This is correctness evidence, not a performance or model-quality benchmark.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def vector(value, count, label):
    require(isinstance(value, list) and len(value) == count, f"{label} length")
    require(all(finite_number(v) and abs(v) <= 3.4028234663852886e38 for v in value), f"{label} finite f32")


def validate_receipt(receipt):
    resident = receipt["schema"] == "spiraltorch.topos_resident_learning.v1"
    require(resident or receipt["schema"] == "spiraltorch.topos_shared_gate_learning.v1", "schema")
    require(receipt["status"] == "executed", "status")
    require(receipt["backend"] == ("wgpu" if resident else "cpu") and receipt["dtype"] == "float32", "backend/dtype")
    if resident:
        require(receipt.get("optimizer") == "resident_subtract_lr_times_gradient", "resident optimizer")
        require(receipt.get("host_readback_during_updates") is False, "resident observation policy")
        require(isinstance(receipt.get("adapter"), str) and bool(receipt["adapter"]), "adapter")
    require(receipt["gate_layout"] == "shared_rows", "gate layout")
    require(receipt["gate_gradient_reduction"] == "sum_without_additional_mean", "gate reduction")
    rate = receipt["learning_rate"]
    require(finite_number(rate) and 0 < rate <= 1, "learning rate")
    require(len(receipt["cases"]) == 2, "case count")
    for case, porosity in zip(receipt["cases"], (0.0, 0.3)):
        config = case["config"]
        require(type(config["iterations"]) is int and config["iterations"] == 5, "iterations")
        for name, expected in (("coupling", 0.2), ("saturation", 1.0), ("porosity", porosity)):
            require(finite_number(config[name]) and abs(config[name] - expected) < 1e-7, f"config {name}")
        vector(case["initial_gate"], 5, "initial gate")
        require(len(case["records"]) == 100, "update count")
        for index, record in enumerate(case["records"]):
            shape = [[1, 5], [3, 5], [8, 5], [2, 5]][index % 4]
            require(type(record["step"]) is int and record["step"] == index, "step sequence")
            require(isinstance(record["shape"], list) and all(type(n) is int for n in record["shape"])
                    and record["shape"] == shape, "shape sequence")
            require(record["input_layout"] == ("row_major" if index % 2 == 0 else "col_major"), "layout sequence")
            for name in ("input", "target", "output", "grad_input"):
                vector(record[name], math.prod(shape), name)
            for name in ("grad_gate", "gate_after"):
                vector(record[name], 5, name)
            require(finite_number(record["loss"]) and record["loss"] >= 0, "finite nonnegative loss")


def compare(receipt):
    validate_receipt(receipt)
    import torch
    from benchmark_topos_module_reference import reference

    torch.set_num_threads(2)
    errors = dict.fromkeys(("output", "grad_input", "grad_gate", "gate_after", "loss"), 0.0)
    for case in receipt["cases"]:
        gate = torch.tensor(case["initial_gate"], dtype=torch.float32, device="cpu").reshape(1, 5).requires_grad_()
        for record in case["records"]:
            shape = record["shape"]
            value = torch.tensor(record["input"], dtype=torch.float32, device="cpu").reshape(shape).requires_grad_()
            target = torch.tensor(record["target"], dtype=torch.float32, device="cpu").reshape(shape)
            output = reference(value, gate, case["config"])
            loss = ((output - target) ** 2).mean()
            dx, dg = torch.autograd.grad(loss, (value, gate))
            gate = (gate - receipt["learning_rate"] * dg).detach().requires_grad_()
            for name, actual in (("output", output), ("grad_input", dx), ("grad_gate", dg),
                                 ("gate_after", gate), ("loss", loss)):
                require(bool(torch.isfinite(actual).all()), f"nonfinite Torch {name}")
                dtype = torch.float64 if name == "loss" else torch.float32
                expected = torch.tensor(record[name], dtype=dtype, device="cpu").reshape(actual.shape)
                require(bool(torch.isfinite(expected).all()), f"nonfinite native {name}")
                observed = actual.detach().to(dtype)
                torch.testing.assert_close(observed, expected, rtol=5e-4, atol=3e-5)
                error = float((observed - expected).abs().max())
                require(math.isfinite(error), f"nonfinite comparison error {name}")
                errors[name] = max(errors[name], error)
    return {"schema": "spiraltorch.topos_shared_gate_torch_check.v1", "status": "passed",
            "source_backend": receipt["backend"],
            "torch": str(torch.__version__), "threads": 2, "updates": 200,
            "rtol": 5e-4, "atol": 3e-5, "max_abs_error": errors,
            "scope": "Native f32 closed-loop shared-gate trajectories compared with CPU Torch, independent broadcast/autograd/reduction. Numeric agreement is not proof of device residency, timing or model quality."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    raw = args.receipt.read_bytes()
    require(len(raw) <= 2_000_000, "receipt size budget")
    report = compare(json.loads(raw))
    report["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    report["client_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["torch_reference_sha256"] = hashlib.sha256(
        Path(__file__).with_name("benchmark_topos_module_reference.py").read_bytes()).hexdigest()
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
