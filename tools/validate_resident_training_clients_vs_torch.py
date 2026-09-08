#!/usr/bin/env python3
"""Replay public Python/WASM client captures with independent PyTorch CPU/MPS.

Shares only the independent torch oracle with the native Rust fixture validator.
No installed SpiralTorch package is required. Reports are read-only inputs;
exclusive output preserves partial results and failure evidence.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform

_spec = importlib.util.spec_from_file_location("_resident_torch_reference",
    Path(__file__).resolve().with_name("validate_resident_training_vs_torch.py"))
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)

RECIPES = {(17, (4,)), (29, (3, 4)), (43, (2, 47, 4))}
POLICIES = {(k, a) for k in ("scalar", "register_2x2")
            for a in ("sequential", "tiled", "compensated")}


def load_pair(python_path, browser_path):
    sources = []
    for path in (python_path, browser_path):
        raw = path.read_bytes()
        sources.append((json.loads(raw), dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())))
    python, browser = [s[0] for s in sources]
    if (python.get("status") != "passed" or python.get("schema") != "spiraltorch.nn.training_client_fixture.v1" or
            browser.get("status") != "passed" or browser.get("schema") != "spiraltorch.nn.training_browser_fixture.v1" or
            browser.get("cpu_only") is not False or len(browser.get("nonfinite_cases", [])) != 6):
        raise ValueError("expected successful GPU Python/browser client fixtures with finite guards")
    expected = {(seed, shape, k, a) for seed, shape in RECIPES for k, a in POLICIES}
    for report in (python, browser):
        actual = [(c["seed"], tuple(c["shape"]), c["kernel"], c["accumulation"]) for c in report["vjps"]]
        if len(actual) != 18 or set(actual) != expected:
            raise ValueError("VJP fixture recipes differ or contain duplicates")
        if len(report["learning"]) != 3 or {c["seed"] for c in report["learning"]} != {17, 29, 43}:
            raise ValueError("learning fixture seeds differ")
    for case in python["learning"]:
        if case["steps"] != 128 or case["learning_rate"] != .2 or case["shape"] != [2, 16, 4]:
            raise ValueError("learning recipe differs")
        if hashlib.sha256(case["half_plan_json"].encode()).hexdigest() != case["half_plan_sha256"]:
            raise ValueError("half-plan digest differs")
    for case in browser["learning"]:
        source = next(c for c in python["learning"] if c["seed"] == case["seed"])
        if case["steps"] != 64 or case["learning_rate"] != .2 or case["source_plan_sha256"] != source["half_plan_sha256"]:
            raise ValueError("browser handoff recipe/lineage differs")
        if hashlib.sha256(case["plan_json"].encode()).hexdigest() != case["plan_sha256"]:
            raise ValueError("return-plan digest differs")
    return python, browser, [s[1] for s in sources]


def run(args, result):
    python, browser, sources = load_pair(args.python_report, args.browser_report)
    result["sources"] = sources
    import torch
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    if "mps" in args.devices and (not torch.backends.mps.is_available() or
            os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0"):
        raise RuntimeError("MPS must be available with CPU fallback disabled")
    result.update(torch=torch.__version__, platform=platform.platform(),
                  python_adapter=python["adapter"], browser_adapter=browser["adapter"])
    for device in args.devices:
        checks = dict(device=device, vjps=[], learning=[])
        result["devices"].append(checks)
        for case in python["vjps"]:
            if case["learning_rate"] != .125:
                raise ValueError("VJP recipe differs")
            pair = next(c for c in browser["vjps"] if all(c[k] == case[k] for k in
                        ("seed", "shape", "kernel", "accumulation")))
            model, linear = reference.build(torch, case["initial_parameters"], device)
            x, target = reference.inputs(torch, case, device)
            row = dict(seed=case["seed"], shape=case["shape"], kernel=case["kernel"],
                       accumulation=case["accumulation"], states={})
            checks["vjps"].append(row)
            for name, rate in (("probe", 0.), ("updated", .125)):
                expected = reference.snapshot(torch, model, linear, x, target, rate)
                row["states"][name] = dict(reference=expected)
                for label, actual in (("python", case), ("browser", pair)):
                    row["states"][name][label] = reference.compare(actual[name], expected)
        for case in python["learning"]:
            pair = next(c for c in browser["learning"] if c["seed"] == case["seed"])
            model, linear = reference.build(torch, case["initial_parameters"], device)
            x, target = reference.inputs(torch, case, device)
            row = dict(seed=case["seed"], states={})
            checks["learning"].append(row)
            initial = reference.snapshot(torch, model, linear, x, target, 0.)
            row["states"]["initial"] = dict(reference=initial, python=reference.compare(case["initial"], initial))
            optimizer = torch.optim.SGD(model.parameters(), lr=.2)
            for _ in range(128):
                optimizer.zero_grad(set_to_none=True)
                x.grad = None
                torch.nn.functional.mse_loss(model(x), target, reduction="mean").backward()
                optimizer.step()
            final = reference.snapshot(torch, model, linear, x, target, 0.)
            row["states"]["final"] = dict(reference=final,
                python=reference.compare(case["final"], final), browser=reference.compare(pair["final"], final))
            row["loss_ratio"] = final["loss"] / initial["loss"]
    for source in sources:
        if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise RuntimeError("fixture changed during validation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-report", required=True, type=Path)
    parser.add_argument("--browser-report", required=True, type=Path)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "mps"), default=["cpu", "mps"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = dict(schema="spiraltorch.nn.training_clients.torch_validation.v1", status="error", devices=[],
                  boundary="float32 tanh GELU, mean-MSE VJP and plain SGD; correctness, not throughput or model quality")
    with args.output.open("x") as output:
        try:
            run(args, result)
            result["status"] = "passed"
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(dict(status=result["status"], error=result.get("error"), output=str(args.output))))
    raise SystemExit(result["status"] != "passed")


if __name__ == "__main__":
    main()
