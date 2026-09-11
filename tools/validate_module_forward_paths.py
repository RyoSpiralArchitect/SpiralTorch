#!/usr/bin/env python3
"""Validate ordinary-Module benchmark captures and replay browser outputs in Torch."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_graph_forward_paths import (admit_native, close, digest, eager,
                                       summarize, validate_samples)


def validate_python(document, native, native_path):
    routes = [f"python_{k}_{c}" for k in ("scalar", "register") for c in ("h2h", "burst")]
    routes += [f"torch_{d}_{c}" for d in ("cpu", "mps") for c in ("h2h", "burst")]
    routes += ["module_resident_d2h", "module_resident_burst"]
    if (document.get("schema") != "spiraltorch.module_forward_paths.v1" or document.get("status") != "passed"
            or document.get("sources", {}).get(str(native_path.resolve())) != digest(native_path)
            or document.get("devices") != ["cpu", "mps"] or len(document.get("cases", [])) != 9
            or document.get("build_info", {}).get("profile") != "release"
            or document.get("build_info", {}).get("features", {}).get("wgpu") is not True):
        raise ValueError("module benchmark lineage/build/matrix differs")
    rows = []
    for case, frozen in zip(document["cases"], native["cases"]):
        if (any(case.get(k) != frozen[k] for k in ("seed", "shape", "depth")) or case["routes"] != routes
                or case["module_cache"] != dict(compilations=1,cache_hits=108,submitted_forwards=109)
                or not math.isfinite(case["module_cold_ms"]) or case["module_cold_ms"] <= 0):
            raise ValueError("original module/counter fixture differs")
        validate_samples(case, routes)
        if set(case["last_outputs"]) != set(routes):
            raise ValueError("missing owning captures")
        for values in case["last_outputs"].values(): close(values, frozen["reference"])
        summary = summarize(case["samples"])
        if summary != case["summary"]: raise ValueError("summary differs from raw samples")
        median = lambda r: summary[r]["median_ms_per_forward"]
        rows.append(dict(shape=case["shape"],depth=case["depth"],seed=case["seed"],
            module_cold_ms=case["module_cold_ms"],summary=summary,
            module_burst_over_torch_mps=median("module_resident_burst")/median("torch_mps_burst"),
            module_burst_over_precompiled_scalar=median("module_resident_burst")/median("python_scalar_burst")))
    return rows


def replay_browser(document, torch):
    if (document.get("schema") != "spiraltorch.module_resident_forward.v1"
            or document.get("status") != "passed" or document.get("page_errors") != []
            or document.get("adapter", {}).get("backend") != "BrowserWebGpu"
            or document.get("adapter", {}).get("device_type") in (None, "Cpu")
            or len(document.get("cases", [])) != 3):
        raise ValueError("browser execution incomplete")
    rows = []
    for case, width in zip(document["cases"], [3,7,16]):
        shape = [3,2,width]
        if (case["shape"] != shape or case["repetitions"] != 20
                or case["cache"] != dict(compilations="1",cache_hits="19",submitted_forwards="20")):
            raise ValueError("browser fixture/cache changed")
        plan = case["plan"]
        if (plan["input_shape"] != shape or len(plan["parameters"]) != 5
                or plan["parameters"] == case["updated_plan"]["parameters"]):
            raise ValueError("browser plan/handoff changed")
        for device in ("cpu", "mps"):
            x = torch.tensor(case["input"], dtype=torch.float32, device=device).reshape(shape)
            params = lambda p: [torch.tensor(v["values"],dtype=torch.float32,device=device).reshape(v["shape"])
                                for v in p["parameters"]]
            weights = params(plan)
            current = x
            errors = []
            for step in range(20):
                current = eager(torch, plan, current, weights)
                if step == 0: errors.append(close(case["first"],current.cpu().reshape(-1).tolist()))
            errors.append(close(case["final"],current.cpu().reshape(-1).tolist()))
            learned = eager(torch,case["updated_plan"],x,params(case["updated_plan"]))
            errors.append(close(case["trained"],learned.cpu().reshape(-1).tolist()))
            rows.append(dict(width=width,device=device,comparisons=3*math.prod(shape),max_abs_error=max(errors)))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("native","python","browser","output"): p.add_argument("--"+name,type=Path,required=True)
    args = p.parse_args()
    paths = [args.native,args.python,args.browser]
    report = dict(schema="spiraltorch.module_forward_validation.v1",status="error",
        boundary="Correctness and bounded completed eager timings. Lower ratios are faster. GPU input/output copies and host parameter comparison included. No autograd migration, zero-copy or fastest-Torch claim. Browser physical GPU identity UNKNOWN.")
    with args.output.open("x") as out:
        try:
            hashes = {str(path.resolve()):digest(path) for path in paths}
            native,python,browser = [json.loads(path.read_bytes()) for path in paths]
            admit_native(native)
            report["timing"] = validate_python(python,native,args.native)
            import torch
            if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0" or not torch.backends.mps.is_available():
                raise ValueError("explicit MPS no-fallback required")
            torch.set_num_threads(1)
            with torch.inference_mode(): report["browser_replay"] = replay_browser(browser,torch)
            if hashes != {str(path.resolve()):digest(path) for path in paths}: raise ValueError("sources changed")
            report.update(sources=hashes,torch=torch.__version__,status="passed")
        except BaseException as error:
            report["error"]=repr(error)
            raise
        finally:
            json.dump(report,out,indent=2,allow_nan=False)
            out.write("\n")


if __name__ == "__main__": main()
