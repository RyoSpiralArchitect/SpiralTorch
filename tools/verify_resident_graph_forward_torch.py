#!/usr/bin/env python3
"""Replay forward-only resident graphs and N-D composition with eager Torch.

No SpiralTorch import, no timing claim, no implicit device fallback.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def admit(report):
    client = report.get("schema") == "spiraltorch.resident_graph_forward_client.v1"
    if (not client and report.get("schema") != "spiraltorch.resident_graph_forward.v1") or report.get("status") != "passed":
        raise ValueError("not a passed forward fixture")
    expected = {(seed, shape, kernel, accumulation) for seed in (17, 29)
                for shape in ((4,), (3, 4), (2, 129, 4))
                for kernel, accumulation in (("scalar", "sequential"), ("register_2x2", "compensated"))}
    cases = report.get("cases", [])
    recipes = [(c["seed"], tuple(c["shape"]), c["kernel"], c["accumulation"]) for c in cases]
    if len(recipes) != 12 or set(recipes) != expected:
        raise ValueError("missing, duplicated or changed recipe")
    guards = {"pointwise_masked_overflow", "dense_masked_overflow", "whole_graph_guard_capture",
              "repeated_inherited_guard", "gpu_input_guard", "device_mismatch_atomic", "recovery",
              "broadcast_permute", "dense_v1_v2_specialized"}
    if client:
        guards = {"owned_output", "late_readback", "atomic_input", "single_consumption", "shared_runtime"}
        if report.get("client") not in ("python", "wasm") or len(report.get("source_fixture_sha256", "")) != 64:
            raise ValueError("missing client fixture lineage")
        if report["client"] == "wasm" and report.get("asset_sha256", {}).get("/fixture.json") != report["source_fixture_sha256"]:
            raise ValueError("browser fixture lineage mismatch")
    if set(report.get("guards", {})) != guards or not all(report["guards"][g] is True for g in guards):
        raise ValueError("missing guard checks")
    if report.get("page_errors"):
        raise ValueError("uncaptured browser failure")
    adapter = report.get("adapter", {})
    if adapter.get("device_type") in (None, "Cpu") or adapter.get("backend") not in ("Metal", "Vulkan", "Dx12", "BrowserWebGpu"):
        raise ValueError("missing device identity or CPU adapter")
    for c in cases:
        plan = c["plan"]
        if (plan["schema"] != "spiraltorch.nn.inference_plan.v2" or plan["input_shape"] != c["shape"]
                or c["dispatches_before_capture"] != 18 or c["pre_gains"] != [1., .5, -.75, 1.25]
                or c["post_shift"] != [-.125, .25, .5]):
            raise ValueError("graph recipe changed")
        if [p["role"] for p in plan["parameters"]] != ["gain", "weight", "bias", "gain", "weight", "bias"]:
            raise ValueError("parameter roles changed")
        if [p["shape"] for p in plan["parameters"]] != [[4], [4, 7], [7], [7], [7, 3], [3]]:
            raise ValueError("parameter shapes changed")
        if [s["kind"] for s in plan["stages"]] != ["pointwise", "linear", "pointwise", "pointwise", "linear"]:
            raise ValueError("graph topology changed")
        if len(c["input"]) != math.prod(c["shape"]):
            raise ValueError("input shape mismatch")
        for field in ("host", "strided", "prediction", "frozen", "post", "chained"):
            if len(c[field]) != math.prod(c["shape"][:-1]) * 3:
                raise ValueError("output shape mismatch")
    return cases


def forward(torch, plan, x, parameters):
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
            raise ValueError("unknown graph stage")
    return current


def check_lineage(documents, sources):
    baselines = {source["sha256"]: doc for doc, source in zip(documents, sources)
                 if doc["schema"] == "spiraltorch.resident_graph_forward.v1"}
    fields = ("seed", "shape", "kernel", "accumulation", "plan", "input", "pre_gains", "post_shift", "dispatches_before_capture")
    for doc in documents:
        if doc["schema"] != "spiraltorch.resident_graph_forward_client.v1":
            continue
        baseline = baselines.get(doc["source_fixture_sha256"])
        if baseline is None:
            raise ValueError("include the hash-matching core JSON fixture alongside client reports")
        for actual, expected in zip(doc["cases"], baseline["cases"]):
            if any(actual[name] != expected[name] for name in fields):
                raise ValueError("client changed the frozen input, plan or recipe")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "mps", "cuda"), default=("cpu", "mps"))
    args = parser.parse_args()
    report = dict(schema="spiraltorch.graph_forward_torch_replay.v1", status="error", sources=[], cases=[],
                  scope="correctness only; explicit eager Torch devices; browser physical adapter unverified")
    with args.output.open("x", encoding="utf-8") as output:
        try:
            inputs, documents = [], []
            for path in args.inputs:
                raw = path.read_bytes()
                doc = json.loads(raw)
                documents.append(doc)
                inputs.append(admit(doc))
                report["sources"].append(dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest()))
            check_lineage(documents, report["sources"])
            import torch
            torch.set_num_threads(1)
            torch.set_float32_matmul_precision("highest")
            report["torch"] = torch.__version__
            if "mps" in args.devices and (not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0"):
                raise RuntimeError("MPS requires a real device and explicitly disabled fallback")
            if "cuda" in args.devices and not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            for source, cases in enumerate(inputs):
                for c in cases:
                    for device in args.devices:
                        x = torch.tensor(c["input"], dtype=torch.float32, device=device).reshape(c["shape"])
                        params = [torch.tensor(p["values"], dtype=torch.float32, device=device).reshape(p["shape"])
                                  for p in c["plan"]["parameters"]]
                        raw = forward(torch, c["plan"], x, params)
                        pre = torch.relu(x * torch.tensor(c["pre_gains"], dtype=torch.float32, device=device))
                        predicted = forward(torch, c["plan"], pre, params)
                        post = torch.nn.functional.gelu(predicted + torch.tensor(c["post_shift"], dtype=torch.float32, device=device), approximate="tanh")
                        maximum = 0.
                        for name, expected in (("host", raw), ("strided", raw), ("prediction", predicted),
                                               ("frozen", predicted), ("post", post), ("chained", torch.relu(post))):
                            expected = expected.cpu().reshape(-1)
                            actual = torch.tensor(c[name], dtype=torch.float32)
                            if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
                                raise ValueError("nonfinite comparison")
                            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
                            maximum = max(maximum, float((actual-expected).abs().max()))
                        report["cases"].append(dict(source=source, seed=c["seed"], shape=c["shape"], device=device,
                                                    kernel=c["kernel"], comparisons=6, max_abs_error=maximum))
            report["status"] = "passed"
        except Exception as error:
            report["error"] = repr(error)
            raise
        finally:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")


if __name__ == "__main__":
    main()
