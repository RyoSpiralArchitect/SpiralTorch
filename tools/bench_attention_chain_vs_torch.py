#!/usr/bin/env python3
"""Matched frozen attention-chain timings; no model quality or training claim.

One driver rotates whole ST/Torch runs; each run also alternates observation
boundaries. Native binaries must be built in release mode by the caller.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def projection_options(values, binaries):
    pairs = [value.split("=", 1) for value in values]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("projection must be LABEL=MODE")
    options = dict(pairs)
    if (len(options) != len(pairs) or set(options) - set(binaries)
            or any(mode not in ("scalar", "register8", "register16") for mode in options.values())):
        raise ValueError("projection labels/modes must match unique native engines")
    return options


def require_projection(report, expected):
    if report.get("projection") != expected:
        raise ValueError("native report does not confirm the requested projection")


def validate_report(report, fixture, samples, warmup, burst):
    expected = {s["name"] + "/" + c["name"] for s in fixture["scenarios"] for c in s["cases"]}
    names = [c["name"] for c in report["cases"]]
    if report["status"] != "passed" or set(names) != expected or len(names) != len(expected):
        raise ValueError("incomplete or failed case coverage")
    if (report["samples_per_route"], report["warmup"], report["burst"]) != (samples, warmup, burst):
        raise ValueError("timing recipe differs")
    for case in report["cases"]:
        keys = [(s["block"], s["route"]) for s in case["samples"]]
        required = {(b, r) for b in range(samples) for r in ("resident", "host_to_host")}
        if set(keys) != required or len(keys) != len(required):
            raise ValueError("incomplete timing samples")
        for sample in case["samples"]:
            if sample["forwards"] != (burst if sample["route"] == "resident" else 1):
                raise ValueError("forward count differs")
            if not math.isfinite(sample["elapsed_ms"]) or sample["elapsed_ms"] <= 0:
                raise ValueError("invalid timing")
            if not math.isfinite(sample["max_abs_error"]) or sample["max_abs_error"] < 0:
                raise ValueError("invalid numerical result")


def torch_run(torch, fixture, device, samples, warmup, burst):
    from torch.nn import functional as F
    sync = torch.mps.synchronize if device == "mps" else lambda: None
    cases = []
    with torch.inference_mode():
        for scenario in fixture["scenarios"]:
            b, t, inner = scenario["input_shape"]
            h, w, out = scenario["heads"], scenario["width"], scenario["output_width"]
            tensor = lambda values: torch.tensor(values, dtype=torch.float32, device="cpu")
            host_input = tensor(scenario["input"]).reshape(b, t, inner)
            weights = [tensor(v).reshape(inner if i < 3 else w, w if i < 3 else out)
                       for i, v in enumerate(scenario["weights"])]
            biases = [tensor(v) for v in scenario["biases"]]
            qkv_weight = torch.cat(weights[:3], dim=1).to(device)
            qkv_bias = torch.cat(biases[:3]).to(device)
            out_weight, out_bias = weights[3].to(device), biases[3].to(device)
            resident_input = host_input.to(device, copy=True)
            kernel = tensor(scenario["expected_kernel"]).reshape(1, h, t, t)
            for case in scenario["cases"]:
                strength, causal = case["geometry_strength"], case["causal"]
                host_bias = None if strength is None else kernel * strength
                if host_bias is not None and causal:
                    host_bias.masked_fill_(torch.arange(t)[None, :] > torch.arange(t)[:, None], -math.inf)
                resident_bias = None if host_bias is None else host_bias.to(device, copy=True)
                expected = tensor(case["expected"]).reshape(b, t, out)

                def forward(x, bias):
                    packed = (x @ qkv_weight + qkv_bias).reshape(b, t, 3, h, w // h)
                    q, k, v = [packed[:, :, i].permute(0, 2, 1, 3) for i in range(3)]
                    attended = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                        dropout_p=0.0, is_causal=causal and bias is None, scale=1. / math.sqrt(w // h))
                    return attended.permute(0, 2, 1, 3).reshape(b, t, w) @ out_weight + out_bias

                def check(actual):
                    actual = actual.to("cpu", copy=True)
                    delta = (actual - expected).abs()
                    if actual.shape != expected.shape or not torch.isfinite(actual).all() or not (delta <= 3e-6 + 3e-5 * expected.abs()).all():
                        raise ValueError(f"{scenario['name']}/{case['name']}: Torch {device} parity failed; max={delta.max().item()}")
                    return delta.max().item()

                check(forward(resident_input, resident_bias))
                timings = []
                for block in range(warmup + samples):
                    for slot in range(2):
                        resident = (block + slot) % 2 == 0
                        forwards = burst if resident else 1
                        sync()
                        start = time.perf_counter()
                        if resident:
                            outputs = [forward(resident_input, resident_bias) for _ in range(forwards)]
                            sync()
                        else:
                            x = host_input.to(device, copy=True)
                            bias = None if host_bias is None else host_bias.to(device, copy=True)
                            outputs = [forward(x, bias).to("cpu", copy=True)]
                            sync()
                        elapsed_ms = (time.perf_counter() - start) * 1000.
                        max_error = max(check(o) for o in outputs)
                        if block >= warmup:
                            timings.append(dict(block=block-warmup, slot=slot,
                                route="resident" if resident else "host_to_host", forwards=forwards,
                                elapsed_ms=elapsed_ms, max_abs_error=max_error))
                cases.append(dict(name=scenario["name"] + "/" + case["name"], shape=scenario["input_shape"],
                                  heads=h, width=w, samples=timings))
    return dict(schema="spiraltorch.attention_chain_bench.v1", status="passed",
        engine="torch_"+device, torch_version=torch.__version__, warmup=warmup,
        samples_per_route=samples, burst=burst, cases=cases,
        boundary="Resident: fixed input/bias/weights, burst forwards and completion sync; reads/checks excluded. Host-to-host: fresh input/bias copies and owning host output, weights resident. CPU copies are included too. Setup and mask/geometry preparation excluded. Eager SDPA default dispatch; not a forced math kernel or kernel timestamp.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--native", action="append", default=[], metavar="LABEL=RELEASE_BINARY")
    parser.add_argument("--native-projection", action="append", default=[], metavar="LABEL=MODE",
                        help="scalar, register8 or register16; omitted preserves historical binary CLI")
    parser.add_argument("--devices", nargs="+", choices=("cpu", "mps"), default=["cpu", "mps"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=50,
                        help="warmup blocks per case/boundary; low counts are for scouts, not steady-state claims")
    parser.add_argument("--burst", type=int, default=4)
    parser.add_argument("--order-seed", type=int, default=23)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key in ("SPIRALTON_MAGIC", "SPIRALTON_TORCH", "SPIRALTON_MODEL_PATCHES", "SPIRALTON_NUMPY",
                "PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH"):
        if os.environ.get(key) != "0":
            parser.error(f"Set {key}=0 before Python startup")
    if args.output.exists() or args.rounds < 1 or args.samples < 3 or args.warmup < 1 or not 1 <= args.burst <= 64:
        parser.error("fresh output and valid nonzero timing recipe required")
    import torch
    torch.set_num_threads(1)
    if "mps" in args.devices and not torch.backends.mps.is_available():
        parser.error("MPS requested but unavailable; no CPU fallback")
    binaries = dict(value.split("=", 1) for value in args.native)
    if len(binaries) != len(args.native) or set(binaries) & {"torch_"+d for d in args.devices}:
        parser.error("engine labels must be unique")
    try:
        projections = projection_options(args.native_projection, binaries)
    except ValueError as error:
        parser.error(str(error))
    fixture = json.loads(args.fixture.read_text())
    if fixture["schema"] != "spiraltorch.attention_chain_torch.v1":
        parser.error("wrong fixture schema")
    engines = list(binaries) + ["torch_"+d for d in args.devices]
    random.Random(args.order_seed).shuffle(engines)
    orders = [engines[i % len(engines):] + engines[:i % len(engines)] for i in range(args.rounds)]
    report = dict(schema="spiraltorch.attention_chain_comparison.v1", status="running",
        fixture_sha256=digest(args.fixture), binary_sha256={k:digest(v) for k,v in binaries.items()},
        native_projections=projections,
        python_version=platform.python_version(), torch_version=torch.__version__, platform=platform.platform(),
        recipe=dict(rounds=args.rounds, samples=args.samples, warmup=args.warmup, burst=args.burst,
                    order_seed=args.order_seed, orders=orders), runs=[],
        scope="Host-observed full-chain forward latency; fixed parameters/geometry, no backward, learning or CUDA evidence. CPU/MPS/native WGPU are separate execution routes; resident timings are not kernel-only timings.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for round_index, order in enumerate(orders):
            for engine in order:
                print(f"round {round_index}: {engine}", file=sys.stderr, flush=True)
                if engine in binaries:
                    command = [binaries[engine], str(args.fixture), str(args.samples), str(args.warmup), str(args.burst)]
                    if engine in projections:
                        command.append(projections[engine])
                    completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=600)
                    result = json.loads(completed.stdout)
                    if engine in projections:
                        require_projection(result, projections[engine])
                else:
                    result = torch_run(torch, fixture, engine.removeprefix("torch_"), args.samples, args.warmup, args.burst)
                validate_report(result, fixture, args.samples, args.warmup, args.burst)
                report["runs"].append(dict(round=round_index, engine=engine, report=result))
        grouped = {}
        for run in report["runs"]:
            for case in run["report"]["cases"]:
                for sample in case["samples"]:
                    key = (run["engine"], case["name"], sample["route"])
                    grouped.setdefault(key, []).append(sample["elapsed_ms"]/sample["forwards"])
        report["summary"] = [dict(engine=k[0], case=k[1], route=k[2], count=len(v),
            median_ms=statistics.median(v), min_ms=min(v), max_ms=max(v)) for k,v in grouped.items()]
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")


if __name__ == "__main__":
    main()
