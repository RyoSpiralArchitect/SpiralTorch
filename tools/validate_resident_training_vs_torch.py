#!/usr/bin/env python3
"""Independently replay source-bound resident training fixtures with PyTorch.

This validates derivatives and updates, not wall-clock speed or model quality.
Inputs are retained native/browser JSON reports; outputs are exclusive and keep
partial results on failure. Neither imported reports nor torch own Rust semantics.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform


def read_report(path):
    raw = path.read_bytes()
    report = json.loads(raw)
    if report.get("schema") != "spiraltorch.resident_training.fixture.v1" or report.get("status") != "passed":
        raise ValueError("expected a successful resident training fixture")
    if len(report["vjps"]["cases"]) != 18 or len(report["vjps"]["torch_cases"]) != 3 or len(report["learning"]["runs"]) != 3:
        raise ValueError("fixture cardinality differs")
    if len(report["guards"]["nonfinite_cases"]) != 6:
        raise ValueError("fixture lacks finite guards")
    return report, hashlib.sha256(raw).hexdigest()


def build(torch, layers, device):
    modules = []
    linear = []
    for layer in layers:
        module = torch.nn.Linear(layer["inner"], layer["cols"], device=device, dtype=torch.float32)
        with torch.no_grad():
            module.weight.copy_(torch.tensor(layer["weights"], device=device).reshape(layer["inner"], layer["cols"]).T)
            module.bias.copy_(torch.tensor(layer["bias"], device=device))
        linear.append(module)
        modules.append(module)
        if layer["gelu"]:
            modules.append(torch.nn.GELU(approximate="tanh"))
    return torch.nn.Sequential(*modules), linear


def snapshot(torch, model, linear, x, target, rate):
    model.zero_grad(set_to_none=True)
    x.grad = None
    prediction = model(x)
    loss = torch.nn.functional.mse_loss(prediction, target, reduction="mean")
    loss.backward()

    def array(tensor):
        value = tensor.detach().cpu().contiguous().flatten()
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite PyTorch reference")
        return value.tolist()

    result = dict(loss=loss.detach().cpu().item(), prediction=array(prediction),
                  input_gradient=array(x.grad), parameter_gradients=[
                      dict(weights=array(m.weight.grad.T), bias=array(m.bias.grad)) for m in linear])
    if rate:
        # Fresh stateless optimizer is exactly plain SGD, no momentum/weight decay.
        torch.optim.SGD(model.parameters(), lr=rate).step()
    result["parameters"] = [dict(weights=array(m.weight.T), bias=array(m.bias)) for m in linear]
    return result


def compare(actual, expected):
    errors = {}

    def check(label, a, b):
        if len(a) != len(b):
            raise ValueError(f"{label}: length differs")
        maximum = 0.0
        for index, (a, b) in enumerate(zip(a, b)):
            if not math.isfinite(a) or not math.isfinite(b) or abs(a - b) > 1e-5 + 1e-4 * abs(b):
                raise ValueError(f"{label}[{index}]: resident {a} != PyTorch {b}")
            maximum = max(maximum, abs(a - b))
        errors[label] = maximum

    check("loss", [actual["loss"]], [expected["loss"]])
    for key in ("prediction", "input_gradient"):
        check(key, actual[key], expected[key])
    for key in ("parameters", "parameter_gradients"):
        if len(actual[key]) != len(expected[key]):
            raise ValueError(f"{key}: stage count differs")
        for index, (a, b) in enumerate(zip(actual[key], expected[key])):
            for kind in ("weights", "bias"):
                check(f"{key}.{index}.{kind}", a[kind], b[kind])
    return errors


def inputs(torch, case, device):
    shape = case["shape"]
    if not shape or shape[-1] != 4 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("invalid fixture input shape")
    if len(case["input"]) != math.prod(shape) or len(case["target"]) != math.prod(shape[:-1]) * 3:
        raise ValueError("invalid fixture data length")
    x = torch.tensor(case["input"], dtype=torch.float32, device=device).reshape(shape).requires_grad_()
    target = torch.tensor(case["target"], dtype=torch.float32, device=device).reshape(shape[:-1] + [3])
    return x, target


def run(paths, devices, result):
    import torch
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    result.update(torch=torch.__version__, platform=platform.platform(), devices=devices)
    if "mps" in devices:
        if not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
            raise RuntimeError("MPS must be available with CPU fallback disabled")
    for path in paths:
        report, digest = read_report(path)
        source = dict(path=str(path), sha256=digest, adapter=report["adapter"],
                      build_manifest=report["build_manifest"], devices=[])
        result["sources"].append(source)
        for device in devices:
            checks = dict(device=device, vjps=[], learning=[])
            source["devices"].append(checks)
            for case in report["vjps"]["torch_cases"]:
                model, linear = build(torch, case["initial_parameters"], device)
                x, target = inputs(torch, case, device)
                row = dict(shape=case["shape"], states={})
                checks["vjps"].append(row)
                for name, rate in (("probe", 0.0), ("updated", case["learning_rate"])):
                    reference = snapshot(torch, model, linear, x, target, rate)
                    row["states"][name] = dict(reference=reference)
                    row["states"][name]["max_abs_errors"] = compare(case[name], reference)
            for case in report["learning"]["runs"]:
                if case["steps"] != 128 or case["learning_rate"] != 0.2:
                    raise ValueError("learning recipe differs")
                model, linear = build(torch, case["initial_parameters"], device)
                x, target = inputs(torch, case, device)
                row = dict(seed=case["seed"], states={})
                checks["learning"].append(row)
                initial = snapshot(torch, model, linear, x, target, 0.0)
                row["states"]["initial"] = dict(reference=initial)
                row["states"]["initial"]["max_abs_errors"] = compare(case["initial"], initial)
                optimizer = torch.optim.SGD(model.parameters(), lr=case["learning_rate"])
                for _ in range(case["steps"]):
                    optimizer.zero_grad(set_to_none=True)
                    x.grad = None
                    torch.nn.functional.mse_loss(model(x), target, reduction="mean").backward()
                    optimizer.step()
                final = snapshot(torch, model, linear, x, target, 0.0)
                row["states"]["final"] = dict(reference=final)
                row["states"]["final"]["max_abs_errors"] = compare(case["final"], final)
                row["loss_ratio"] = final["loss"] / initial["loss"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError("fixture changed during replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", required=True, nargs="+", type=Path)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "mps"), default=["cpu", "mps"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = dict(schema="spiraltorch.resident_training.torch_validation.v1", status="error", sources=[],
                  boundary="float32 mean-MSE, tanh GELU, exact VJP and plain SGD; correctness only, no timing/quality claim")
    # Reserve first: never spend GPU work on a path that would overwrite old evidence.
    with args.output.open("x") as output:
        try:
            run(args.reports, args.devices, result)
            result["status"] = "passed"
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(dict(status=result["status"], error=result.get("error"), output=str(args.output))))
    raise SystemExit(result["status"] != "passed")


if __name__ == "__main__":
    main()
