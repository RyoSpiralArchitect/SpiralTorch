"""Wider matched CPU products, full-grid preconditioning, and explicit PyTorch out buffers."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

SHAPES = [(8,768,3072), (8,3072,768), (32,768,3072), (32,3072,768),
          (64,256,1024), (128,256,256), (256,128,256), (17,137,195),
          (65,1025,97), (128,1024,1024)]

def require(value, message):
    if not value:
        raise ValueError(message)

def repetitions(shape):
    return 2 if math.prod(shape) >= 64*1024*1024 else 4

def key(case):
    return tuple(case[k] for k in ["rows", "inner", "cols", "packed"])

def torch_run(threads):
    import torch
    require("sitecustomize" not in sys.modules, "Startup customization is forbidden")
    torch.set_num_interop_threads(1)
    torch.set_num_threads(threads)
    cases = []
    with torch.inference_mode():
        for rows, inner, cols in SHAPES:
            a = (((torch.arange(rows*inner, device="cpu")*17)%127).float()-63.0).div(31.0).reshape(rows, inner)
            b = (((torch.arange(inner*cols, device="cpu")*29)%131).float()-65.0).div(37.0).reshape(inner, cols)
            expected = a.double() @ b.double()
            for packed in [False, True]:
                rhs = b.T.contiguous().T if packed else b
                out = torch.empty((rows, cols), device="cpu", dtype=torch.float32)
                pointer = out.data_ptr()
                reps = repetitions((rows, inner, cols))
                for _ in range(3):
                    torch.mm(a, rhs, out=out)
                intervals = []
                for _ in range(9):
                    start = time.perf_counter_ns()
                    for _ in range(reps):
                        torch.mm(a, rhs, out=out)
                    intervals.append((time.perf_counter_ns()-start)/reps)
                require(out.data_ptr() == pointer and str(out.device) == "cpu", "Output residency changed")
                difference = (out.double()-expected).abs()
                valid = bool(torch.isfinite(out).all() and (difference <= 1e-3+1e-4*expected.abs()).all())
                cases.append(dict(rows=rows, inner=inner, cols=cols, packed=packed,
                    float64_valid=valid, max_abs=float(difference.max()), elapsed_ns=intervals,
                    repetitions=reps, rhs_stride=list(rhs.stride()), dtype=str(out.dtype)))
    print(json.dumps(dict(schema="spiraltorch.cpu_dense_extended.torch.v1", cases=cases,
        version=torch.__version__, module=torch.__file__, build_config=torch.__config__.show(),
        device="cpu", threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        sitecustomize_loaded=False, tolerance=dict(atol=1e-3,rtol=1e-4))))

def check_cases(document, rust):
    require(len(document["cases"]) == 20, "Incomplete case list")
    mapping = {key(c):c for c in document["cases"]}
    require(set(mapping) == {(*s,p) for s in SHAPES for p in [False,True]}, "Wrong/duplicate conditions")
    require(document["tolerance"] == dict(atol=1e-3,rtol=1e-4), "Wrong tolerance")
    for identity, case in mapping.items():
        require(case["float64_valid"] is True and math.isfinite(case["max_abs"]), "Float64 reference failure")
        require(len(case["elapsed_ns"]) == 9 and all(math.isfinite(v) and v>0 for v in case["elapsed_ns"]), "Invalid timing")
        require(case["repetitions"] == repetitions(identity[:3]), "Wrong repetitions")
        if rust:
            require(case["bitwise_equal"] is True, "Sequential float32 mismatch")
            require(all(isinstance(case[k],int) and case[k]>=0 for k in ["allocation_calls","allocated_bytes"]), "Invalid allocation count")
        else:
            require(case["rhs_stride"] == ([1,case["inner"]] if case["packed"] else [case["cols"],1]), "Wrong RHS layout")
            require(case["dtype"] == "torch.float32", "Wrong dtype")
    return mapping

def report(directory):
    comparisons = []
    for mode,threads,deterministic in [("serial",1,"1"),("parallel",4,"0")]:
        maps = []
        for name in ["warm-baseline","warm-candidate","baseline-a","candidate-a","baseline-b","candidate-b"]:
            doc = json.loads((directory/f"{mode}-{name}.json").read_text())
            require(doc["schema"] == "spiraltorch.cpu_dense_extended.v1", "Wrong Rust schema")
            require(doc["rayon_threads"] == str(threads) and doc["deterministic"] == deterministic, "Wrong Rust thread mode")
            mapping = check_cases(doc, True)
            if not name.startswith("warm"):
                maps.append(mapping)
        torch_doc = json.loads((directory/f"{mode}-torch.json").read_text())
        require(torch_doc["schema"] == "spiraltorch.cpu_dense_extended.torch.v1", "Wrong Torch schema")
        require(torch_doc["threads"] == threads and torch_doc["interop_threads"] == 1 and torch_doc["device"] == "cpu", "Wrong Torch mode")
        require(torch_doc["sitecustomize_loaded"] is False, "Torch startup not isolated")
        torch_cases = check_cases(torch_doc, False)
        for identity in maps[0]:
            b1,c1,b2,c2 = [m[identity] for m in maps]
            comparisons.append(dict(mode=mode,shape=list(identity[:3]),packed=identity[3],
                baseline_over_candidate=[statistics.median(b["elapsed_ns"])/statistics.median(c["elapsed_ns"]) for b,c in [(b1,c1),(b2,c2)]],
                torch_over_candidate=[statistics.median(torch_cases[identity]["elapsed_ns"])/statistics.median(c["elapsed_ns"]) for c in [c1,c2]],
                baseline_allocations=[[c["allocation_calls"],c["allocated_bytes"]] for c in [b1,b2]],
                candidate_allocations=[[c["allocation_calls"],c["allocated_bytes"]] for c in [c1,c2]]))
    summaries = []
    for mode in ["serial","parallel"]:
        for packed in [False,True]:
            group = [c for c in comparisons if c["mode"]==mode and c["packed"]==packed]
            values = [x for c in group for x in c["baseline_over_candidate"]]
            summaries.append(dict(mode=mode,packed=packed,geometric_mean=statistics.geometric_mean(values),
                minimum=min(values),maximum=max(values),favorable=sum(x>1 for x in values),count=len(values),
                torch_over_candidate=statistics.geometric_mean(x for c in group for x in c["torch_over_candidate"])))
    return dict(measured_rust_condition_runs=160,preconditioning_condition_runs=80,torch_condition_runs=40,
                comparisons=comparisons,summaries=summaries,numerical_replay=False)

def measure(directory, baseline, candidate, torch_python, torch_site):
    directory.mkdir()
    receipt = dict(steps=[],binaries={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [baseline,candidate]},
                   policy="One complete recorded preconditioning grid per binary/mode, then fixed AB/BA; no interval trimming")
    def run(name,command,env):
        with (directory/(name+".json")).open("x") as out, (directory/(name+".stderr")).open("x") as err:
            result = subprocess.run(command,env=env,stdout=out,stderr=err)
        receipt["steps"].append(dict(name=name,command=command,exit_code=result.returncode))
        (directory/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
        result.check_returncode()
        print(name,"complete",flush=True)
    for mode,threads,deterministic in [("serial",1,"1"),("parallel",4,"0")]:
        env=dict(os.environ,RAYON_NUM_THREADS=str(threads),SPIRAL_DETERMINISTIC=deterministic,
                 SPIRAL_DETERMINISTIC_REDUCTION=deterministic,SPIRAL_DETERMINISTIC_SEED="17")
        env.pop("HOME",None)
        env.pop("SPIRALTORCH_AUTOTUNE_STORE",None)
        for name in ["warm-baseline","warm-candidate","baseline-a","candidate-a","candidate-b","baseline-b"]:
            run(mode+"-"+name,[str(baseline if "baseline" in name else candidate)],env)
        bootstrap="import sys,runpy;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name='__main__')"
        run(mode+"-torch",[torch_python,"-I","-S","-c",bootstrap,torch_site,str(Path(__file__).resolve()),"torch","--threads",str(threads)],env)
    (directory/"comparison.json").write_text(json.dumps(report(directory),indent=2)+"\n")
    print("All extended numerical gates passed",flush=True)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["measure","report","torch"])
    parser.add_argument("--directory",type=Path)
    parser.add_argument("--baseline",type=Path)
    parser.add_argument("--candidate",type=Path)
    parser.add_argument("--torch-python",default=sys.executable)
    parser.add_argument("--torch-site")
    parser.add_argument("--threads",type=int,choices=[1,4],default=1)
    args=parser.parse_args()
    if args.mode=="torch":
        torch_run(args.threads)
    elif args.mode=="report":
        print(json.dumps(report(args.directory),indent=2))
    else:
        measure(args.directory,args.baseline.resolve(),args.candidate.resolve(),args.torch_python,args.torch_site)
