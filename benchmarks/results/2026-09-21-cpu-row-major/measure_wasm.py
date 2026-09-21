"""Fixed WASM/Node client comparison, not browser or GPU performance."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

SHAPES = [(8,768,3072), (32,768,3072), (64,256,1024),
          (128,256,256), (17,137,195), (65,1025,97)]
NAMES = ["warm-baseline", "warm-candidate", "baseline-a", "candidate-a", "candidate-b", "baseline-b"]
def require(value, message):
    if not value:
        raise ValueError(message)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def report(directory):
    receipt = json.loads((directory / "receipt.json").read_text())
    require(receipt["modules"]["baseline"]["wasm_sha256"] != receipt["modules"]["candidate"]["wasm_sha256"], "Identical WASM modules")
    require([s["name"] for s in receipt["steps"]] == NAMES and all(s["exit_code"] == 0 for s in receipt["steps"]), "Incomplete WASM run")
    maps = {}
    runtime = None
    for name in NAMES:
        doc = json.loads((directory / (name+".json")).read_text())
        require(doc["schema"] == "spiraltorch.wasm_cpu_dense.v1", "Wrong WASM schema")
        require(doc["requires_grad"] is False and doc["fixture"] == "dyadic_13_17_over_8", "Wrong WASM fixture")
        this_runtime = [doc[k] for k in ["node", "arch", "platform"]]
        require(runtime is None or runtime == this_runtime, "Runtime changed")
        runtime = this_runtime
        cases = doc["cases"]
        mapping = {tuple(c[k] for k in ["rows","inner","cols","packed"]): c for c in cases}
        require(len(cases) == 12 and set(mapping) == {(*s,p) for s in SHAPES for p in [False,True]}, "Wrong/duplicate WASM conditions")
        for case in cases:
            require(case["bitwise_equal"] is True and len(case["output_sha256"]) == 64, "Invalid WASM numerics")
            require(case["repetitions"] == 2 and len(case["elapsed_ns"]) == 9 and all(math.isfinite(t) and t>0 for t in case["elapsed_ns"]), "Invalid WASM timing")
        maps[name] = mapping
    comparisons = []
    for key in maps["baseline-a"]:
        require(len({m[key]["output_sha256"] for m in maps.values()}) == 1, "WASM output hash mismatch")
        require(maps["baseline-a"][key]["output_sha256"] == maps["baseline-a"][(*key[:3],not key[3])]["output_sha256"], "WASM layout outputs differ")
        ratios = [statistics.median(maps["baseline-"+r][key]["elapsed_ns"])/statistics.median(maps["candidate-"+r][key]["elapsed_ns"]) for r in ["a","b"]]
        comparisons.append(dict(shape=list(key[:3]), packed=key[3], baseline_over_candidate=ratios))
    summaries = []
    for packed in [False,True]:
        ratios = [r for c in comparisons if c["packed"]==packed for r in c["baseline_over_candidate"]]
        summaries.append(dict(packed=packed, geometric_mean=statistics.geometric_mean(ratios),
            minimum=min(ratios),maximum=max(ratios),favorable=sum(r>1 for r in ratios),count=len(ratios)))
    return dict(measured_condition_runs=48, preconditioning_condition_runs=24, runtime=runtime,
        comparisons=comparisons,summaries=summaries,numerical_replay=False)

def measure(directory, harness, baseline, candidate):
    directory.mkdir()
    receipt = dict(harness=str(harness),harness_sha256=digest(harness),modules={},steps=[],
        policy="Recorded full-grid preconditioning for both modules, then AB/BA; nine untrimmed intervals; one Node process at a time")
    for label,path in [("baseline",baseline),("candidate",candidate)]:
        receipt["modules"][label] = dict(js=str(path),js_sha256=digest(path),wasm_sha256=digest(path.with_name(path.stem+"_bg.wasm")))
    require(receipt["modules"]["baseline"]["wasm_sha256"] != receipt["modules"]["candidate"]["wasm_sha256"], "Identical WASM modules")
    env = dict(os.environ)
    env.pop("NODE_OPTIONS", None)
    for name in NAMES:
        command = ["node",str(harness),str(baseline if "baseline" in name else candidate)]
        with (directory/(name+".json")).open("x") as out, (directory/(name+".stderr")).open("x") as err:
            result = subprocess.run(command,env=env,stdout=out,stderr=err)
        receipt["steps"].append(dict(name=name,command=command,exit_code=result.returncode))
        (directory/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
        result.check_returncode()
        print(name,"complete",flush=True)
    (directory/"comparison.json").write_text(json.dumps(report(directory),indent=2)+"\n")
    print("All WASM numerical gates passed",flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["measure","report"])
    parser.add_argument("--directory",type=Path,required=True)
    for name in ["harness","baseline","candidate"]:
        parser.add_argument("--"+name,type=Path)
    args = parser.parse_args()
    if args.mode == "report":
        print(json.dumps(report(args.directory),indent=2))
    else:
        measure(args.directory,args.harness.resolve(),args.baseline.resolve(),args.candidate.resolve())
