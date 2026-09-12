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
    classification = "label_smoothing" in case
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
        if classification:
            target = torch.tensor(case["target"], dtype=torch.long, device=device)
            objective = torch.nn.functional.cross_entropy(current.reshape(-1, current.shape[-1]), target,
                reduction=case["reduction"], ignore_index=case["ignore_index"], label_smoothing=case["label_smoothing"])
        else:
            target = torch.tensor(case["target"], dtype=torch.float32, device=device).reshape(current.shape)
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
    if classification:
        final = evaluate(plan, x, parameters)
        check(final, case["final_prediction"])
        check(torch.nn.functional.cross_entropy(final.reshape(-1, final.shape[-1]), target,
            reduction=case["reduction"], ignore_index=case["ignore_index"], label_smoothing=case["label_smoothing"]), [case["final_loss"]])
    return {
        "device": device,
        "seed": case["seed"],
        "input_shape": case["input_shape"],
        "policy": "Exact" if vjp else case["policy"],
        "arbitrary_cotangent": vjp,
        "weighted_learning": False,
        "classification": classification,
        "comparisons": comparisons,
        "max_abs_error": maximum,
    }


def admit_classification_probes(probes):
    expected = [(name, reduction, smoothing) for name in ("nd", "uniform", "strided", "broadcast")
                for reduction in ("none", "sum", "mean") for smoothing in (0., .2, 1.)]
    expected += [("wide_mean","mean",0.), ("tiny_smoothing","mean",1e-40),
                 ("tiny_tail","mean",0.), ("wide_vocab","mean",.2),
                 ("single_class","mean",.2), ("empty_none","none",0.), ("empty_sum","sum",0.)]
    assert [(p["name"],p["reduction"],p["label_smoothing"]) for p in probes] == expected
    for probe in probes:
        shape = probe["shape"]
        assert shape and shape[-1] > 0 and all(isinstance(d,int) and d >= 0 for d in shape)
        assert len(probe["prediction"]) == math.prod(shape)
        assert len(probe["target"]) == math.prod(shape[:-1])
    assert next(p for p in probes if p["name"] == "wide_vocab")["shape"] == [1,50257]


def replay_classification_probe(case, device):
    wide = case["name"] in ("wide_mean", "tiny_smoothing")
    assert not wide or device == "cpu", "wide reference uses explicit CPU float64, not an MPS fallback"
    dtype = torch.float64 if wide else torch.float32
    classes = case["shape"][-1]
    x = torch.tensor(case["prediction"], dtype=dtype, device=device).reshape(-1, classes).requires_grad_()
    labels = torch.tensor(case["target"], dtype=torch.long, device=device)
    objective = torch.nn.functional.cross_entropy(x, labels, reduction=case["reduction"],
        label_smoothing=case["label_smoothing"], ignore_index=case["ignore_index"])
    gradient, = torch.autograd.grad(objective.sum(), [x])
    maximum = 0.
    for actual, expected in [(objective, case["loss"]), (gradient, case["gradient"])]:
        actual = actual.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        expected = torch.tensor(expected, dtype=torch.float32)
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        maximum = max(maximum, float((actual-expected).abs().max()) if actual.numel() else 0.)
    return dict(device=device, probe=case["name"], reduction=case["reduction"],
        label_smoothing=case["label_smoothing"], classification=True, comparisons=2,
        max_abs_error=maximum, reference_dtype=str(dtype))


def classification_tail_gap(case, device):
    # This fixture deliberately exceeds ordinary f32 log_softmax tail precision.
    # Record the eager Torch result, but do not count a loose-atol zero as agreement.
    assert case["prediction"] == [80., 0.] and case["target"] == [0.]
    tail = math.exp(-80.)
    for actual, reference in zip(case["loss"]+case["gradient"], [tail, -tail, tail], strict=True):
        assert math.isfinite(actual) and abs(actual/reference-1) < 2e-5
    x = torch.tensor([[80., 0.]], dtype=torch.float32, device=device, requires_grad=True)
    value = torch.nn.functional.cross_entropy(x, torch.tensor([0], device=device))
    gradient, = torch.autograd.grad(value, [x])
    return dict(device=device, probe="tiny_tail", classification=True,
        scope="analytic two-class tail check, NOT a matched PyTorch-autograd comparison",
        spiraltorch_loss=case["loss"], spiraltorch_gradient=case["gradient"],
        torch_loss=float(value.detach().cpu()), torch_gradient=gradient.detach().cpu().reshape(-1).tolist())


def replay_loss_probe(case, device):
    shape = case["shape"]
    prediction = torch.tensor(case["prediction"], dtype=torch.float32, device=device).reshape(shape).requires_grad_()
    target = torch.tensor(case["target"], dtype=torch.float32, device=device).reshape(shape)
    # SpiralTorch explicitly defines empty MSE as zero; PyTorch mean(empty) is NaN.
    objective = (prediction - target).square().mean() if prediction.numel() else prediction.sum() * 0
    gradient, = torch.autograd.grad(objective, [prediction])
    maximum = 0.0
    for actual, expected in [(objective, [case["loss"]]), (gradient, case["gradient"])]:
        actual = actual.detach().cpu().reshape(-1)
        expected = torch.tensor(expected, dtype=torch.float32)
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        maximum = max(maximum, float((actual - expected).abs().max()) if actual.numel() else 0.)
    return dict(device=device, probe=case["name"], resident_loss=True, comparisons=2,
                max_abs_error=maximum, empty_contract_override=prediction.numel() == 0)


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


def admit_microbatch(case):
    assert case["policy"] in ("Exact", "ModuleCompatible")
    assert case["input_shape"] == case["plan"]["input_shape"] == [2,3,4]
    assert case["observations_after_updates"] == 32 and case["microbatches"] == 95
    assert case["module_parameters_applied"] == 7 and case["optimizer"] == "explicit_sgd_not_ModuleTrainer"
    assert len(case["windows"]) == 32 and len(case["dataset"]) == 7
    assert case["reduction"] in ("mean", "sum") and case["label_smoothing"] == .1 and case["ignore_index"] == -100
    counts = []
    for batch in case["dataset"]:
        assert len(batch["input"]) == 24 and all(math.isfinite(x) for x in batch["input"])
        assert len(batch["target"]) == 6 and all(y in (0,1,2,-100) for y in batch["target"])
        counts.append(sum(y != -100 for y in batch["target"]))
    assert counts == [6,4,2,5,3,1,6]
    for i, window in enumerate(case["windows"]):
        ids = [(i*3+j)%7 for j in range(2+i%3)]
        assert [m["batch"] for m in window["microbatches"]] == ids
        total = sum(counts[j] for j in ids)
        for m in window["microbatches"]:
            weight = (counts[m["batch"]] if case["reduction"] == "mean" else 1) / total
            assert math.isclose(m["weight"],weight,rel_tol=1e-7,abs_tol=0.)
        assert math.isclose(window["rate"],0. if i%11 == 0 else .1,rel_tol=1e-7,abs_tol=0.)
    assert [e["batch"] for e in case["evaluation"]] == list(range(7))


def replay_microbatch(case, device):
    admit_microbatch(case)
    plan = case["plan"]
    assert plan["schema"] == "spiraltorch.nn.inference_plan.v2"
    parameters = [torch.tensor(p["values"],dtype=torch.float32,device=device).reshape(p["shape"]).requires_grad_() for p in plan["parameters"]]
    original = [p.detach().clone() for p in parameters]
    comparisons = 0
    maximum = 0.
    def check(actual, expected):
        nonlocal comparisons, maximum
        actual = actual.detach().cpu().reshape(-1)
        expected = torch.tensor(expected,dtype=torch.float32).reshape(-1)
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-4)
        maximum = max(maximum,float((actual-expected).abs().max()) if actual.numel() else 0.)
        comparisons += 1
    def batch(index):
        data = case["dataset"][index]
        return (torch.tensor(data["input"],dtype=torch.float32,device=device).reshape([2,3,4]).requires_grad_(),
                torch.tensor(data["target"],dtype=torch.long,device=device))
    def loss(prediction, target, reduction):
        return torch.nn.functional.cross_entropy(prediction.reshape(-1,3),target,reduction=reduction,ignore_index=-100,label_smoothing=.1)
    for window in case["windows"]:
        aggregate = [torch.zeros_like(p) for p in parameters]
        for m in window["microbatches"]:
            x,y = batch(m["batch"])
            predicted = evaluate(plan,x,parameters)
            objective = loss(predicted,y,case["reduction"])
            dx,*raw = torch.autograd.grad(objective,[x,*parameters])
            check(predicted,m["prediction"]); check(objective,[m["loss"]]); check(dx,m["input_gradient"])
            for g, expected, accumulated in zip(raw,m["raw_gradients"],aggregate,strict=True):
                check(g,expected)
                accumulated.add_(g*m["weight"])
        for g, expected in zip(aggregate,window["accumulated_gradients"],strict=True): check(g,expected)
        with torch.no_grad():
            for p,g,spec,expected in zip(parameters,aggregate,plan["parameters"],window["parameters"],strict=True):
                if case["policy"] == "ModuleCompatible" and spec["role"] == "gain": g = g/6
                p.sub_(window["rate"]*g)
                check(p,expected)
    initial_total = final_total = 0.
    total = 0
    for e in case["evaluation"]:
        x,y = batch(e["batch"])
        with torch.no_grad():
            prediction = evaluate(plan,x,parameters)
            final = loss(prediction,y,"mean")
            initial = loss(evaluate(plan,x,original),y,"mean")
        check(prediction,e["prediction"]); check(final,[e["loss"]]); check(initial,[e["initial_loss"]])
        n = sum(v != -100 for v in case["dataset"][e["batch"]]["target"])
        initial_total += float(initial.cpu())*n; final_total += float(final.cpu())*n; total += n
    check(torch.tensor([initial_total/total]),[case["initial_loss"]])
    check(torch.tensor([final_total/total]),[case["final_loss"]])
    return dict(device=device,seed=case["seed"],policy=case["policy"],microbatch=True,
                updates=32,microbatches=95,comparisons=comparisons,max_abs_error=maximum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-resident-loss", action="store_true")
    parser.add_argument("--require-classification", action="store_true")
    parser.add_argument("--require-microbatch", action="store_true")
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
        "reference_gaps": [],
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
                if args.require_resident_loss:
                    assert fixture.get("resident_loss", {}).get("status") == "passed"
                if args.require_classification:
                    assert fixture.get("classification", {}).get("status") == "passed"
                if args.require_microbatch:
                    assert fixture.get("microbatch", {}).get("status") == "passed"
                report["inputs"].append(
                    {
                        "path": str(path.resolve()),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
                for device in args.devices:
                    microbatch = fixture.get("microbatch")
                    if microbatch is not None:
                        assert microbatch["status"] == "passed"
                        assert [(c["seed"],c["policy"],c["reduction"]) for c in microbatch["cases"]] == [
                            (seed,policy,reduction) for seed,reduction in [(17,"mean"),(29,"sum"),(43,"mean")]
                            for policy in ("Exact","ModuleCompatible")]
                        for case in microbatch["cases"]:
                            result = replay_microbatch(case,device)
                            result["input"] = str(path.resolve()); report["cases"].append(result)
                    classification = fixture.get("classification")
                    if classification is not None:
                        assert classification["status"] == "passed" and len(classification["cases"]) == 6
                        admit_classification_probes(classification["probes"])
                        assert [(c["seed"], c["policy"], c["label_smoothing"]) for c in classification["cases"]] == [
                            (seed, policy, smoothing) for seed, smoothing in [(17,0.),(29,.2),(43,.5)]
                            for policy in ("Exact","ModuleCompatible")]
                        for case in classification["cases"]:
                            assert case["observations_after_updates"] == 64 and case["module_parameters_applied"] == 7
                            assert case["optimizer"] == "explicit_sgd_not_ModuleTrainer" and case["reduction"] == "mean"
                            assert case["final_loss"] < case["steps"][0]["loss"]
                            result = replay(case, device, expected_steps=64)
                            result["input"] = str(path.resolve()); report["cases"].append(result)
                        for probe in classification["probes"]:
                            if probe["name"] == "tiny_tail":
                                gap = classification_tail_gap(probe, device)
                                gap["input"] = str(path.resolve()); report["reference_gaps"].append(gap)
                                continue
                            if device != "cpu" and probe["name"] in ("wide_mean", "tiny_smoothing"):
                                continue
                            result = replay_classification_probe(probe, device)
                            result["input"] = str(path.resolve()); report["cases"].append(result)
                    resident_loss = fixture.get("resident_loss")
                    if resident_loss is not None:
                        assert resident_loss["status"] == "passed"
                        assert len(resident_loss["cases"]) == 6
                        assert [(c["seed"], c["policy"]) for c in resident_loss["cases"]] == [
                            (seed, policy) for seed in (17,29,43) for policy in ("Exact","ModuleCompatible")]
                        assert [p["name"] for p in resident_loss["probes"]] == [
                            "scalar","empty","tail","many_partials","nd","strided","broadcast"]
                        for case in resident_loss["cases"]:
                            assert case["observations_after_updates"] == 64 and case["module_parameters_applied"] == 7
                            assert case["optimizer"] == "explicit_sgd_not_ModuleTrainer"
                            assert case["final_loss"] < case["steps"][0]["loss"]
                            result = replay(case, device, expected_steps=64)
                            result.update(resident_loss=True, input=str(path.resolve()))
                            report["cases"].append(result)
                        for probe in resident_loss["probes"]:
                            result = replay_loss_probe(probe, device)
                            result["input"] = str(path.resolve())
                            report["cases"].append(result)
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
                        assert len(autograd["guards"]) == 12
                        assert all(g["passed"] for g in autograd["guards"])
                        capture = [g for g in autograd["guards"] if g["case"] == "queued_capture_tail_shapes_detached_guard_recovery"]
                        assert len(capture) == 1 and capture[0]["captured_vjps"] == 4
                        assert capture[0]["input_shape"] == [2, 3, 257] and capture[0]["observation"] == "after_workspace_drop"
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
