#!/usr/bin/env python3
"""Replay source-bound native/browser graph fixtures against eager PyTorch.

This is a correctness check, not a throughput benchmark or optimizer advantage.
ModuleCompatible applies the explicitly declared extra gain row average only.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch


def evaluate(plan, x, parameters):
    current = x
    for stage in plan["stages"]:
        if stage["kind"] == "linear":
            current = current @ parameters[stage["weight"]] + parameters[stage["bias"]]
            if stage["gelu"]:
                current = torch.nn.functional.gelu(current, approximate="tanh")
        elif stage["kind"] == "pointwise":
            inputs = [current, *(parameters[i] for i in stage["parameters"])]
            for step in stage["steps"]:
                op = step["op"]
                if op == "identity":
                    pass
                elif op == "add":
                    current = current + inputs[step["rhs"]]
                elif op == "multiply":
                    current = current * inputs[step["rhs"]]
                elif op == "relu":
                    current = torch.relu(current)
                elif op == "gelu":
                    current = torch.nn.functional.gelu(current, approximate="tanh")
                else:
                    raise ValueError(f"unknown operation {op}")
        else:
            raise ValueError("unknown stage")
    return current


def replay(case, device, *, expected_steps=9):
    plan = case["plan"]
    assert plan["schema"] == "spiraltorch.nn.inference_plan.v2"
    vjp = "replays" in case
    if not vjp:
        assert case["policy"] in ("Exact", "ModuleCompatible")
    x = torch.tensor(case["input"], dtype=torch.float32, device=device).reshape(
        plan["input_shape"]
    )
    x.requires_grad_()
    parameters = [
        torch.tensor(p["values"], dtype=torch.float32, device=device)
        .reshape(p["shape"])
        .requires_grad_()
        for p in plan["parameters"]
    ]
    rows = math.prod(plan["input_shape"][:-1])
    maximum = 0.0
    comparisons = 0

    def check(actual, expected):
        nonlocal maximum, comparisons
        actual = actual.detach().cpu().reshape(-1)
        expected = torch.tensor(expected, dtype=torch.float32).reshape(-1)
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise AssertionError("nonfinite comparison")
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        maximum = max(
            maximum, float((actual - expected).abs().max()) if actual.numel() else 0.0
        )
        comparisons += 1

    if vjp:
        assert len(case["replays"]) == 4
        steps = [(None, r) for r in case["replays"]]
    else:
        assert len(case["steps"]) == len(case["rates"]) == expected_steps
        steps = zip(case["rates"], case["steps"])
    for rate, expected in steps:
        current = evaluate(plan, x, parameters)
        if vjp:
            cotangent = torch.tensor(expected["cotangent"], dtype=torch.float32, device=device).reshape(current.shape)
            dx, *raw = torch.autograd.grad(current, [x, *parameters], grad_outputs=cotangent)
            check(current, case["prediction"])
            check(dx, expected["input_gradient"])
            assert len(raw) == len(expected["raw_gradients"])
            for actual, reference in zip(raw, expected["raw_gradients"]):
                check(actual, reference)
            continue
        target = torch.tensor(
            case["target"], dtype=torch.float32, device=device
        ).reshape(current.shape)
        objective = (current - target).square().mean()
        dx, *raw = torch.autograd.grad(objective, [x, *parameters])
        effective = [
            (
                g / rows
                if case["policy"] == "ModuleCompatible" and p["role"] == "gain"
                else g
            )
            for g, p in zip(raw, plan["parameters"])
        ]
        check(objective, [expected["loss"]])
        check(current, expected["prediction"])
        check(dx, expected["input_gradient"])
        with torch.no_grad():
            for i, p in enumerate(parameters):
                check(raw[i], expected["raw_gradients"][i])
                check(effective[i], expected["effective_gradients"][i])
                if rate != 0:
                    p.sub_(rate * effective[i])
                check(p, expected["parameters"][i])
    return {
        "device": device,
        "seed": case["seed"],
        "input_shape": case["input_shape"],
        "policy": "Exact" if vjp else case["policy"],
        "arbitrary_cotangent": vjp,
        "weighted_learning": False,
        "comparisons": comparisons,
        "max_abs_error": maximum,
    }


def replay_learning(case, device):
    plan = case["plan"]
    assert plan["schema"] == "spiraltorch.nn.inference_plan.v2"
    assert case["shape"] == plan["input_shape"]
    assert case["policy"] in ("exact", "module_compatible")
    assert case["steps"] == case["accepted_updates"] == 64
    assert case["coefficients"] == [0.75, 0.25] and abs(case["rate"] - 0.1) < 1e-7
    assert [c["step"] for c in case["captures"]] == [0, 1, 7, 31, 63]
    assert case["resume"] == "bit_identical"
    x = torch.tensor(case["input"], dtype=torch.float32, device=device).reshape(case["shape"]).requires_grad_()
    parameters = [torch.tensor(p["values"], dtype=torch.float32, device=device).reshape(p["shape"]).requires_grad_()
                  for p in plan["parameters"]]
    rows = math.prod(case["shape"][:-1])
    target = torch.tensor(case["target"], dtype=torch.float32, device=device)
    captures = {c["step"]: c for c in case["captures"]}
    maximum, comparisons = 0.0, 0

    def check(actual, expected):
        nonlocal maximum, comparisons
        actual = actual.detach().cpu().reshape(-1)
        expected = torch.tensor(expected, dtype=torch.float32).reshape(-1)
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        maximum = max(maximum, float((actual - expected).abs().max()))
        comparisons += 1

    def objective(prediction):
        e = prediction.reshape(-1) - target
        return (0.75 * 0.5 * e.square() + 0.25 * 0.25 * e.pow(4)).mean()

    for step in range(64):
        prediction = evaluate(plan, x, parameters)
        e = prediction - target.reshape(prediction.shape)
        first = torch.autograd.grad(prediction, [x, *parameters], grad_outputs=e / e.numel(), retain_graph=True)
        second = torch.autograd.grad(prediction, [x, *parameters], grad_outputs=e.pow(3) / e.numel())
        if step == 0:
            check(objective(prediction), case["initial_loss"])
        expected = captures.get(step)
        if expected:
            check(prediction, expected["prediction"])
            assert len(expected["input_gradients"]) == len(expected["raw_gradients"]) == 2
            for term, result in enumerate([first, second]):
                check(result[0], expected["input_gradients"][term])
                assert len(parameters) == len(expected["raw_gradients"][term])
                for actual, value in zip(result[1:], expected["raw_gradients"][term]):
                    check(actual, value)
        with torch.no_grad():
            for i, parameter in enumerate(parameters):
                gradient = first[i + 1] * 0.75 + second[i + 1] * 0.25
                if case["policy"] == "module_compatible" and plan["parameters"][i]["role"] == "gain":
                    gradient *= 1.0 / rows
                parameter.sub_(case["rate"] * gradient)
                if expected:
                    assert len(expected["parameters"]) == len(parameters)
                    check(parameter, expected["parameters"][i])
    final = evaluate(plan, x, parameters)
    check(final, case["final_prediction"])
    check(objective(final), case["final_loss"])
    assert 0 <= case["final_loss"] < case["initial_loss"]
    return dict(device=device, seed=case["seed"], input_shape=case["shape"], policy=case["policy"],
                arbitrary_cotangent=False, weighted_learning=True, comparisons=comparisons,
                updates=64, max_abs_error=maximum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--devices", nargs="+", choices=["cpu", "mps", "cuda"], default=["cpu", "mps"]
    )
    args = parser.parse_args()
    report = {
        "schema": "spiraltorch.graph_training_torch_replay.v1",
        "status": "error",
        "torch": torch.__version__,
        "cases": [],
        "inputs": [],
        "scope": "correctness only; eager Torch; no fallback",
    }
    with args.output.open("x", encoding="utf-8") as output:
        try:
            if "mps" in args.devices and (
                not torch.backends.mps.is_available()
                or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0"
            ):
                raise RuntimeError(
                    "MPS requires an actual device and explicit disabled fallback"
                )
            if "cuda" in args.devices and not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; no fallback")
            for path in args.inputs:
                data = path.read_bytes()
                fixture = json.loads(data)
                assert (
                    fixture["schema"]
                    == "spiraltorch.resident_graph_training_fixture.v1"
                )
                assert fixture["status"] == "passed" and len(fixture["cases"]) == 6
                report["inputs"].append(
                    {
                        "path": str(path.resolve()),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
                for device in args.devices:
                    fusion = fixture.get("pointwise_fusion")
                    if fusion is not None:
                        assert len(fusion["cases"]) == 6 and len(fusion["guards"]) == 16
                    for case in fixture["cases"] + (fusion["cases"] if fusion else []):
                        result = replay(case, device)
                        result["pointwise_fusion"] = "source_plan" in case
                        result["input"] = str(path.resolve())
                        report["cases"].append(result)
                    autograd = fixture.get("autograd")
                    if autograd is not None:
                        assert len(autograd["cases"]) == 6
                        assert len(autograd["guards"]) == 11
                        assert all(g["passed"] for g in autograd["guards"])
                        for case in autograd["cases"]:
                            result = replay(case, device)
                            result["input"] = str(path.resolve())
                            result["pointwise_fusion"] = case["fused"]
                            report["cases"].append(result)
                    learning = fixture.get("learning")
                    if learning is not None:
                        assert len(learning["cases"]) == 12 and len(learning["guards"]) == 8
                        assert all(g["passed"] for g in learning["guards"])
                        for case in learning["cases"]:
                            result = replay_learning(case, device)
                            result["input"] = str(path.resolve())
                            result["pointwise_fusion"] = case["fused"]
                            report["cases"].append(result)
            report["status"] = "passed"
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            json.dump(report, output, indent=2)
            output.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "cases": len(report["cases"]),
                "max_abs_error": max(c["max_abs_error"] for c in report["cases"]),
            }
        )
    )


if __name__ == "__main__":
    main()
