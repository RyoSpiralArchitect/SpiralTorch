#!/usr/bin/env python3
"""Validate paired original-Module runs against exact frozen product identities."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_graph_forward_paths import admit_native, close, digest, summarize, validate_samples
from validate_module_forward_paths import validate_python

ROUTES = ["baseline_module_d2h", "candidate_module_d2h", "baseline_module_burst",
          "candidate_module_burst", "baseline_scalar_burst", "candidate_scalar_burst"]


def validate_browser(document, native):
    if (document.get("schema") != "spiraltorch.module_direct_io_matched.v1"
            or document.get("status") != "passed" or document.get("page_errors") != []
            or document.get("fixture_request") != "nn-module-matched"
            or document.get("routes") != ROUTES or document.get("warmup") != 3
            or document.get("samples_per_route") != 9 or document.get("burst") != 8
            or len(document.get("cases", [])) != 9):
        raise ValueError("browser matched contract incomplete")
    rows = []
    for case, frozen in zip(document["cases"], native["cases"]):
        if (any(case.get(k) != frozen[k] for k in ("seed", "shape", "depth"))
                or case.get("routes") != ROUTES or set(case["last_outputs"]) != set(ROUTES)
                or set(case["adapters"]) != {"baseline", "candidate"}
                or set(case["cold_ms"]) != {"baseline", "candidate"}
                or set(case["cache"]) != {"baseline", "candidate"}):
            raise ValueError("browser matched case differs")
        for name in ("baseline", "candidate"):
            if (case["adapters"][name].get("backend") != "BrowserWebGpu"
                    or case["adapters"][name].get("device_type") in (None, "Cpu")
                    or not math.isfinite(case["cold_ms"][name]) or case["cold_ms"][name] <= 0
                    or case["cache"][name] != dict(compilations="1",cache_hits="108",submitted_forwards="109")):
                raise ValueError("browser device/cache/cold timing differs")
        validate_samples(case, ROUTES)
        maximum = max(close(v, frozen["reference"]) for v in case["last_outputs"].values())
        summary = summarize(case["samples"])
        median = lambda route: summary[route]["median_ms_per_forward"]
        rows.append(dict(shape=case["shape"],depth=case["depth"],seed=case["seed"],
            comparisons=len(frozen["reference"])*len(ROUTES),max_abs_error=maximum,summary=summary,
            candidate_over_baseline={route:median("candidate_"+route)/median("baseline_"+route)
                for route in ("module_d2h", "module_burst", "scalar_burst")}))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("fixture", "baseline-python", "candidate-python", "browser", "baseline-receipt",
             "candidate-library", "candidate-wasm", "output")
    for name in names: parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    paths = [getattr(args, n.replace("-", "_")) for n in names if n != "output"]
    report = dict(schema="spiraltorch.module_direct_io_validation.v1",status="error",
        boundary="Lower ratios are faster. Same-page rotated browser A/B; Python versions run serially in separate processes, each with eager Torch controls. Physical browser GPU and host exclusivity UNKNOWN. No peak-performance or automatic pure Tensor/autograd migration claim.")
    with args.output.open("x") as out:
        try:
            hashes = {str(p.resolve()):digest(p) for p in paths}
            native, old, new, browser, receipt = [json.loads(p.read_bytes()) for p in paths[:5]]
            admit_native(native)
            if receipt.get("status") != "passed": raise ValueError("baseline verification did not pass")
            base = args.baseline_receipt.parent
            for name, sha in receipt["products"].items():
                if digest(base/name) != sha: raise ValueError("baseline product changed: "+name)
            expected = {"/module/spiraltorch_wasm_bg.wasm":digest(args.candidate_wasm),
                        "/baseline/spiraltorch_wasm_bg.wasm":receipt["products"]["client/spiraltorch_wasm_bg.wasm"],
                        "/fixture.json":digest(args.fixture)}
            if any(browser.get("asset_sha256", {}).get(name) != sha for name, sha in expected.items()):
                raise ValueError("browser artifact identities differ")
            for doc, path, sha in [(old,base/"python-release.dylib",receipt["products"]["python-release.dylib"]),
                                   (new,args.candidate_library,digest(args.candidate_library))]:
                if doc["sources"].get(str(path.resolve())) != sha: raise ValueError("Python product identity differs")
            a = validate_python(old,native,args.fixture)
            b = validate_python(new,native,args.fixture)
            report["python"] = [dict(shape=n["shape"],depth=n["depth"],seed=n["seed"],
                candidate=n,baseline=o,candidate_over_baseline={
                    route:n["summary"][route]["median_ms_per_forward"]/o["summary"][route]["median_ms_per_forward"]
                    for route in ("module_resident_d2h", "module_resident_burst", "python_scalar_burst", "torch_mps_burst")})
                for o,n in zip(a,b)]
            report["browser"] = validate_browser(browser,native)
            if hashes != {str(p.resolve()):digest(p) for p in paths}: raise ValueError("source changed")
            report.update(sources=hashes,baseline_source=receipt["source"],status="passed")
        except BaseException as error:
            report["error"]=repr(error)
            raise
        finally:
            json.dump(report,out,indent=2,allow_nan=False);out.write("\n")


if __name__ == "__main__": main()
