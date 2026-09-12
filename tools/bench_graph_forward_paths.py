#!/usr/bin/env python3
"""Frozen mixed-NN fixture: public Python resident handles and eager Torch.

Timing includes completed host observation, never just asynchronous submission.
This is a bounded diagnostic, not a fastest-PyTorch or browser hardware claim.
"""
import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import struct
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
WARMUP, SAMPLES, BURST = 3, 9, 8


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32_bytes(values):
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("non-float fixture data")
    return struct.pack("<" + "f"*len(values), *values)


def requests():
    return [(seed, shape, depth) for seed in (17, 29, 43)
            for shape, depth in (([2, 3, 7], 2), ([2, 8, 64], 8), ([4, 8, 128], 16))]


def close(actual, expected):
    if len(actual) != len(expected):
        raise ValueError("output length differs")
    maximum = 0.
    for a, b in zip(actual, expected):
        if not math.isfinite(a) or not math.isfinite(b) or abs(a-b) > 2e-5 + 2e-4*abs(b):
            raise ValueError(f"nonfinite or mismatched output: {a} != {b}")
        maximum = max(maximum, abs(a-b))
    return maximum


def admit_native(document):
    if (document.get("schema") != "spiraltorch.graph_forward_bench.v1" or document.get("status") != "passed"
            or document.get("client") != "native" or document.get("warmup") != WARMUP
            or document.get("samples_per_route") != SAMPLES or document.get("burst") != BURST
            or document.get("routes") != ["legacy_h2h", "scalar_h2h", "register_h2h", "scalar_burst", "register_burst"]):
        raise ValueError("native timing contract differs")
    adapter = document.get("adapter", {})
    if adapter.get("backend") not in ("Metal", "Vulkan", "Dx12") or adapter.get("device_type") in (None, "Cpu"):
        raise ValueError("native GPU identity missing")
    cases = document.get("cases", [])
    if len(cases) != len(requests()):
        raise ValueError("incomplete native matrix")
    for case, (seed, shape, depth) in zip(cases, requests()):
        if (case.get("seed"), case.get("shape"), case.get("depth")) != (seed, shape, depth):
            raise ValueError("native recipe changed")
        width = shape[-1]
        plan = case["plan"]
        if (plan.get("schema") != "spiraltorch.nn.inference_plan.v2" or plan.get("input_shape") != shape
                or case["source_operations"] != depth*4 or case["gpu_stages"] != depth*3
                or len(plan["parameters"]) != depth*3 or len(plan["stages"]) != depth*3):
            raise ValueError("mixed lowering changed")
        index = seed
        for i, parameter in enumerate(plan["parameters"]):
            role = ("gain", "weight", "bias")[i % 3]
            expected_shape = [width, width] if role == "weight" else [width]
            expected = []
            for j in range(math.prod(expected_shape)):
                jitter = ((index*17 % 23)-11)/1024
                expected.append(jitter + (1. if role == "gain" or role == "weight" and j//width == j%width else 0.))
                index += 1
            # Rust emits shortest round-tripping f32 decimals, not exact f64 rationals.
            if (set(parameter) != {"role", "shape", "values"} or parameter["role"] != role
                    or parameter["shape"] != expected_shape or f32_bytes(parameter["values"]) != f32_bytes(expected)):
                raise ValueError("frozen parameter recipe changed")
        for i in range(depth):
            expected = [dict(kind="pointwise", parameters=[i*3], steps=[dict(op="multiply", rhs=1)]),
                        dict(kind="linear", weight=i*3+1, bias=i*3+2, gelu=True),
                        dict(kind="pointwise", parameters=[], steps=[dict(op="relu", rhs=None)])]
            if plan["stages"][i*3:i*3+3] != expected:
                raise ValueError("frozen topology changed")
        expected_input = [((j+seed)%29)/16-.5 for j in range(math.prod(shape))]
        if case["input"] != expected_input or len(case["reference"]) != len(expected_input):
            raise ValueError("frozen input changed")
        close(case["reference"], case["reference"])
        if len(case["last_outputs"]) != 5:
            raise ValueError("missing native captures")
        for output in case["last_outputs"]:
            close(output, case["reference"])
        validate_samples(case, list(range(5)), native=True)
    return cases


def validate_samples(case, routes, native=False):
    samples = case["samples"]
    if len(samples) != SAMPLES * len(routes):
        raise ValueError("sample matrix incomplete")
    expected = []
    for block in range(SAMPLES):
        offset = (block + WARMUP + case["seed"]) % len(routes)
        order = routes[offset:] + routes[:offset]
        expected.extend((block, route, order) for route in order)
    for sample, (block, route, order) in zip(samples, expected):
        if sample["block"] != block or sample["route"] != route or sample["order"] != order:
            raise ValueError("sample order changed")
        burst = route >= 3 if native else str(route).endswith("_burst")
        if type(sample["forwards"]) is not int or sample["forwards"] != (BURST if burst else 1):
            raise ValueError("forward count changed")
        if (not isinstance(sample["elapsed_ms"], (int, float)) or isinstance(sample["elapsed_ms"], bool)
                or not math.isfinite(sample["elapsed_ms"]) or sample["elapsed_ms"] <= 0
                or not math.isfinite(sample["max_abs_error"]) or sample["max_abs_error"] < 0):
            raise ValueError("invalid timing or comparison")


def summarize(samples):
    result = {}
    for route in dict.fromkeys(sample["route"] for sample in samples):
        rows = [row for row in samples if row["route"] == route]
        values = [row["elapsed_ms"]/row["forwards"] for row in rows]
        result[route] = dict(samples=len(rows), forwards_per_sample=rows[0]["forwards"],
                             median_ms_per_forward=statistics.median(values),
                             min_ms_per_forward=min(values), max_ms_per_forward=max(values))
    return result


def load_native(library):
    name = "spiraltorch.spiraltorch"
    loader = importlib.machinery.ExtensionFileLoader(name, str(library))
    spec = importlib.util.spec_from_file_location(name, library, loader=loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    sys.path.insert(0, str(ROOT / "bindings/st-py"))
    import spiraltorch as st
    if (Path(st._rs.__file__).resolve() != library or st.build_info()["profile"] != "release"
            or not st.build_info()["features"]["wgpu"]):
        raise ValueError("a frozen release WGPU binding is required")
    return st


def eager(torch, plan, x, parameters):
    current = x
    for stage in plan["stages"]:
        if stage["kind"] == "linear":
            shape = current.shape
            current = torch.addmm(parameters[stage["bias"]], current.reshape(-1, shape[-1]),
                                  parameters[stage["weight"]]).reshape(
                                      (*shape[:-1], parameters[stage["weight"]].shape[1]))
            if stage["gelu"]:
                current = torch.nn.functional.gelu(current, approximate="tanh")
        else:
            for step in stage["steps"]:
                if step["op"] == "relu":
                    current = torch.relu(current)
                elif step["op"] == "multiply":
                    current = current * parameters[stage["parameters"][step["rhs"]-1]]
                else:
                    raise ValueError("unadmitted benchmark operation")
    return current


def original_module(case, st):
    """Rebuild the admitted fixture using real high-level Rust layers."""
    net = st.nn.Sequential()
    width = case["shape"][-1]
    for i in range(case["depth"]):
        gain, weight, bias = case["plan"]["parameters"][3*i:3*i+3]
        net.add(st.nn.Scaler.from_gain(f"scale{i}", st.Tensor(1,width,gain["values"])))
        linear = st.nn.Linear(f"linear{i}", width, width)
        linear.load_state_dict([(f"linear{i}::weight", st.Tensor(width,width,weight["values"])),
                                (f"linear{i}::bias", st.Tensor(1,width,bias["values"]))])
        net.add(linear)
        net.add(st.nn.Gelu())
        net.add(st.nn.Relu())
    frozen = json.loads(net.inference_plan(case["shape"]).to_json())
    if frozen != case["plan"]:
        raise ValueError("original Module differs from the admitted plan")
    return net


def run_case(case, st, torch, devices, include_module=False, terminal_module=False):
    plan = st.nn.InferencePlan.from_json(json.dumps(case["plan"]))
    graphs = dict(scalar=plan.compile_graph_wgpu(),
                  register=plan.compile_graph_wgpu(kernel="register_2x2", accumulation="compensated"))
    host = torch.tensor(case["input"], dtype=torch.float32).reshape(case["shape"])
    contexts = {device: (host.to(device).clone(),
                        [torch.tensor(p["values"], dtype=torch.float32, device=device).reshape(p["shape"])
                         for p in case["plan"]["parameters"]]) for device in devices}
    routes = [f"python_{kernel}_{cadence}" for kernel in graphs for cadence in ("h2h", "burst")]
    routes += [f"torch_{device}_{cadence}" for device in devices for cadence in ("h2h", "burst")]
    module, module_input, cold_ms = None, None, None
    if include_module:
        module = original_module(case, st)
        module_input = st.WgpuTensorDevice.create().upload(case["shape"], case["input"])
        start = perf_counter()
        capture = module.forward_snapshot(module_input) if terminal_module else module(module_input).snapshot()
        close(capture.read_values(), case["reference"])
        cold_ms = (perf_counter()-start)*1000
        routes += ["module_resident_d2h", "module_resident_burst"]
    samples, last_outputs = [], {}
    for block in range(WARMUP + SAMPLES):
        offset = (block + case["seed"]) % len(routes)
        order = routes[offset:] + routes[:offset]
        for route in order:
            family, variant, cadence = route.split("_")
            forwards = BURST if cadence == "burst" else 1
            if family == "python":
                graph = graphs[variant]
                if cadence == "burst":
                    graph.upload_values(case["input"])
                    graph.dispatch()
                    graph.snapshot().read_values()
                before = graph.submitted_dispatches
                start = perf_counter()
                if cadence == "h2h": graph.upload_values(case["input"])
                for _ in range(forwards): graph.dispatch()
                output = graph.snapshot().read_values()
                elapsed = (perf_counter()-start)*1000
                if graph.submitted_dispatches-before != forwards: raise ValueError("dispatch count differs")
            elif family == "module":
                start = perf_counter()
                if terminal_module:
                    for _ in range(forwards-1): value = module(module_input)
                    output = module.forward_snapshot(module_input).read_values()
                else:
                    for _ in range(forwards): value = module(module_input)
                    output = value.snapshot().read_values()
                elapsed = (perf_counter()-start)*1000
            else:
                x, parameters = contexts[variant]
                if variant == "mps": torch.mps.synchronize()
                start = perf_counter()
                if cadence == "h2h": x.copy_(host, non_blocking=True)
                for _ in range(forwards): value = eager(torch, case["plan"], x, parameters)
                output = value.cpu().reshape(-1).tolist()
                elapsed = (perf_counter()-start)*1000
            maximum = close(output, case["reference"])
            last_outputs[route] = output
            if block >= WARMUP:
                samples.append(dict(block=block-WARMUP, route=route, order=order, forwards=forwards,
                                    elapsed_ms=elapsed, max_abs_error=maximum))
    result = dict(seed=case["seed"], shape=case["shape"], depth=case["depth"], routes=routes,
                  samples=samples, last_outputs=last_outputs, adapter=graphs["scalar"].adapter_info())
    if include_module:
        expected = 1 + (WARMUP+SAMPLES)*(1+BURST)
        stats = module.resident_cache_info()
        if stats != dict(compilations=1, cache_hits=expected-1, submitted_forwards=expected):
            raise ValueError("original Module did not reuse exactly one graph")
        result.update(module_cold_ms=cold_ms, module_cache=stats)
    validate_samples(result, routes)
    result["summary"] = summarize(samples)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-module", action="store_true",
                        help="Add ordinary model(WgpuTensor) routes; keep the legacy v1 benchmark unchanged by default")
    parser.add_argument("--terminal-module", action="store_true",
                        help="Explicit forward_snapshot for the final forward only; requires --include-module")
    args = parser.parse_args()
    if args.terminal_module and not args.include_module:
        parser.error("--terminal-module requires --include-module")
    report = dict(schema="spiraltorch.graph_forward_paths.v1", status="error", cases=[],
                  boundary="Three warmups, nine retained rotated blocks. H2H includes input transfer and host-list output. Burst keeps fixed input device-resident for eight independent forwards and includes one final host-list read. Setup excluded; numerical checks outside timing. Torch eager addmm/bias/tanh-GELU, no compile; native controls measured separately. macOS GPU contention UNKNOWN; no fastest-Torch claim.")
    if args.include_module:
        report["schema"] = "spiraltorch.module_forward_paths.v1"
        report["boundary"] += " Module routes use the same original Rust NN model; d2h keeps input resident and reads one output, burst performs eight independent forwards and reads the last output. Per-call weight bit checks and the resident I/O implementation of the hashed native artifact are included. No host transfer occurs between resident forwards. Cold compilation is recorded separately. d2h is not interchangeable with h2h."
    if args.terminal_module:
        report.update(schema="spiraltorch.module_terminal_forward_paths.v1", module_api="forward_snapshot")
        report["boundary"] += " Terminal forward and snapshot copy share one submission; only the last forward in each burst uses this API. Host wrapper differences are included; not isolated GPU submission timing."
    paths = [args.fixture.resolve(strict=True), args.native_library.resolve(strict=True)]
    identities = {str(path): digest(path) for path in paths}
    with args.output.open("x") as output:
        try:
            document = json.loads(args.fixture.read_bytes())
            cases = admit_native(document)
            st = load_native(paths[1])
            import torch
            sys.path.insert(0, str(ROOT / "tools"))
            from bench_resident_nn_vs_torch import admit_device, match_adapter
            admission = admit_device("mps")
            if not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
                raise RuntimeError("real MPS with explicit no-fallback is required")
            torch.set_num_threads(1)
            torch.set_float32_matmul_precision("highest")
            report.update(sources=identities, torch=torch.__version__, build_info=st.build_info(),
                          device_admission=admission, devices=["cpu", "mps"])
            match_adapter(document["adapter"], "mps", admission["name"])
            with torch.inference_mode():
                for case in cases:
                    result = run_case(case, st, torch, ["cpu", "mps"], args.include_module, args.terminal_module)
                    match_adapter(result["adapter"], "mps", admission["name"])
                    report["cases"].append(result)
                    print(json.dumps({k:result[k] for k in ("shape", "seed", "depth", "summary")}), flush=True)
            if identities != {str(path): digest(path) for path in paths} or admit_device("mps") != admission:
                raise RuntimeError("artifact or device identity changed")
            report["status"] = "passed"
        except Exception as error:
            report["error"] = repr(error)
            raise
        finally:
            json.dump(report, output, allow_nan=False)
            output.write("\n")


if __name__ == "__main__":
    main()
