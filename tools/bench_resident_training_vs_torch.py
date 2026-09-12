#!/usr/bin/env python3
"""Rotating source-bound resident training A/B and eager PyTorch GPU comparison.

Every interval starts from the same weights and uploaded batch, then performs
eight nonzero-rate SGD updates. Every loss is retained and read. Initial/final
zero-rate VJP probes, resets, compilation and validation are outside timing.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import select
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_rank_vs_torch as audit
from bench_resident_nn_vs_torch import admit_device, match_adapter
import validate_resident_training_vs_torch as reference
import resident_graph_bench_reference as graph_reference
import resident_learner_bench_reference as learner_reference


def recipes(graph=False, matrix="standard", learner_optimizer=None):
    optimizer_settings(learner_optimizer)
    if learner_optimizer is not None and not graph:
        raise ValueError("learner optimizer requires a graph")
    if matrix not in ("standard", "wide") or (matrix == "wide" and not graph):
        raise ValueError("wide training matrix requires a mixed graph")
    if matrix == "wide":
        shapes = (([4, 64, 64], 8), ([2, 128, 128], 8), ([2, 64, 256], 4))
    else:
        shapes = ((([2, 16, 32], 2), ([2, 129, 32], 4), ([4, 32, 64], 8)) if graph else
                  (([2, 16, 32], 2), ([4, 16, 64], 8), ([4, 32, 128], 16)))
    return [dict(shape=shape, depth=depth, seed=seed, steps=8, **({"graph": True} if graph else {}),
                 **({"learner_optimizer": learner_optimizer} if learner_optimizer is not None else {}))
            for seed in (17, 29, 43) for shape, depth in shapes]


def optimizer_settings(name):
    if name not in (None, "topos_ema", "clipped_topos_ema"):
        raise ValueError("unknown learner optimizer")
    return (None, None) if name is None else (0.5, 1. / 1024 if name == "clipped_topos_ema" else None)


def source_for(ref):
    return dict(commit=audit.git_bytes("rev-parse", ref).decode().strip(),
                tree=audit.git_bytes("rev-parse", ref+"^{tree}").decode().strip(), tracked_dirty=False)


class Native:
    def __init__(self, executable, source, stderr):
        self.executable = executable.resolve(strict=True)
        self.file = audit.file_identity(self.executable)
        self.identity = audit.read_native_build_identity(self.executable)
        self.binding = audit.validate_source_binding(self.identity, source_for(source))
        if not self.binding["valid"]:
            raise ValueError("native binary is not bound to its supplied immutable source")
        self.process = subprocess.Popen([str(self.executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=stderr, bufsize=0)
        self.pending = b""

    def request(self, value):
        self.process.stdin.write((json.dumps(value)+"\n").encode())
        deadline = time.monotonic()+120
        while b"\n" not in self.pending:
            timeout = deadline-time.monotonic()
            if timeout <= 0 or not select.select([self.process.stdout], [], [], timeout)[0]:
                raise TimeoutError("native response timeout")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("native worker exited before a response")
            self.pending += chunk
            if len(self.pending) > 64*1024*1024:
                raise ValueError("native response exceeds fixture budget")
        line, self.pending = self.pending.split(b"\n", 1)
        return json.loads(line)

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdout.close()
        if self.process.returncode != 0:
            raise RuntimeError(f"native worker ended with status {self.process.returncode}")
        if audit.file_identity(self.executable) != self.file:
            raise RuntimeError("native executable changed during measurement")


def close_values(actual, expected):
    if len(actual) != len(expected) or any(not math.isfinite(a) or not math.isfinite(b) or
            abs(a-b) > 1e-5+1e-4*abs(b) for a,b in zip(actual,expected)):
        raise ValueError("loss trajectory differs")


def torch_sample(torch, fixture, device, cadence, synchronize):
    parameters = [dict(inner=p["inner"], cols=p["cols"], weights=p["weight"], bias=p["bias"], gelu=p["gelu"])
                  for p in json.loads(fixture["plan_json"])["stages"]]
    shape = fixture["config"]["shape"]
    model, linear = reference.build(torch, parameters, device)
    x = torch.tensor(fixture["input"], dtype=torch.float32, device=device).reshape(shape).requires_grad_()
    target = torch.tensor(fixture["target"], dtype=torch.float32, device=device).reshape(shape)
    initial = reference.snapshot(torch, model, linear, x, target, 0.)
    optimizer = torch.optim.SGD(model.parameters(), lr=fixture["learning_rate"], foreach=False)
    losses = []
    synchronize()
    start = time.perf_counter()
    for _ in range(fixture["config"]["steps"]):
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        prediction = model(x)
        loss = torch.nn.functional.mse_loss(prediction, target, reduction="mean")
        loss.backward()
        optimizer.step()
        if cadence == "immediate":
            synchronize()
            losses.append(loss.detach().cpu().item())
        else:
            losses.append(loss.detach())
    if cadence == "deferred":
        losses = torch.stack(losses).cpu().tolist()
    synchronize()
    elapsed = (time.perf_counter()-start)*1000
    state = reference.snapshot(torch, model, linear, x, target, 0.)
    return dict(status="passed",cadence=cadence,steps=fixture["config"]["steps"],elapsed_ms=elapsed,
                losses=losses,initial_loss=initial["loss"],state=state)


def validate_sample(value, cadence, steps, *, learner=False, lane=None, seed_fusion=False, learner_optimizer=None):
    damping, clip = optimizer_settings(learner_optimizer)
    if learner_optimizer is not None and not learner:
        raise ValueError("optimizer requires a learner interval")
    if (value.get("learner_optimizer") != learner_optimizer or value.get("momentum_damping") != damping
            or value.get("grad_clip_max_norm") != clip):
        raise ValueError("optimizer execution differs")
    state = value.get("state")
    if isinstance(state, dict):
        if ("momentum" in state) != (learner_optimizer is not None):
            raise ValueError("optimizer history missing or unexpectedly present")
        if learner_optimizer is not None:
            history, parameters = state["momentum"], state["parameters"]
            if (len(history) != len(parameters) or not history or
                    any(len(h) != len(p) or any(type(v) not in (int, float) or not math.isfinite(v) for v in h)
                        for h, p in zip(history, parameters))):
                raise ValueError("invalid optimizer history")
    if seed_fusion and not learner:
        raise ValueError("seed fusion requires the learner workload")
    if value.get("fused_learner_seeds", False) is not seed_fusion:
        raise ValueError("incorrect seed fusion execution")
    if (value.get("status") != "passed" or value.get("cadence") != cadence or value.get("steps") != steps or
            len(value.get("losses", [])) != (0 if learner else steps) or type(value.get("elapsed_ms")) not in (int, float) or
            not math.isfinite(value["elapsed_ms"]) or value["elapsed_ms"] <= 0):
        raise ValueError("sample contract differs")
    if value.get("learner", False) is not learner:
        raise ValueError("wrong learning workload")
    if learner:
        if value.get("losses") != [] or type(value.get("completed_updates")) is not int or value["completed_updates"] != steps:
            raise ValueError("missing completed learner updates")
        for key in ("initial_loss", "final_loss"):
            if type(value.get(key)) not in (int,float) or not math.isfinite(value[key]) or value[key] < 0:
                raise ValueError("invalid learner objective")
        if lane == "torch":
            if value.get("acceptance") != "synchronized_only" or "accepted_updates" in value:
                raise ValueError("Torch cannot claim Rust guard receipts")
        elif (value.get("acceptance") != "guarded_receipts" or
              value.get("accepted_updates") != list(range(2,steps+2)) or
              any(type(v) is not int for v in value["accepted_updates"])):
            raise ValueError("missing ordered update receipts")


def match_seed_fixture(baseline, candidate):
    config = dict(candidate["config"])
    if config.pop("fuse_learner_seeds", None) is not True or config != baseline["config"]:
        raise ValueError("seed fusion recipe differs")
    for key in ("plan_json", "input", "target", "learning_rate", "kernel", "accumulation", "adapter"):
        if candidate[key] != baseline[key]: raise ValueError("seed fusion fixture differs: " + key)


def run(args, result, baseline_stderr, candidate_stderr):
    graph = getattr(args, "graph", False)
    learner = getattr(args, "learner", False)
    seed_fusion = getattr(args, "fuse_learner_seeds", False)
    learner_optimizer = getattr(args, "learner_optimizer", None)
    compare = learner_reference.compare if learner else (graph_reference.compare if graph else reference.compare)
    sample_torch = learner_reference.torch_sample if learner else (graph_reference.torch_sample if graph else torch_sample)
    result["device_admission"] = admit_device(args.device)
    before = audit.source_identity()
    if before["tracked_dirty"] or audit.git_bytes("ls-files", "--others", "--exclude-standard"):
        raise ValueError("freeze the benchmark source before measuring")
    result["harness_source"] = before
    import torch
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    if args.device == "mps":
        if not torch.backends.mps.is_available(): raise RuntimeError("MPS unavailable")
        name, synchronize = result["device_admission"]["name"], torch.mps.synchronize
    else:
        if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
        name, synchronize = torch.cuda.get_device_name(), torch.cuda.synchronize
    result.update(torch=torch.__version__, torch_device=args.device)
    workers = {}
    try:
        for label, path, ref, stderr in (("baseline", args.baseline, args.baseline_source, baseline_stderr),
                                        ("candidate", args.candidate, args.candidate_source, candidate_stderr)):
            workers[label] = Native(path, ref, stderr)
        result["native_products"] = {k:dict(file=w.file,identity=w.identity,binding=w.binding) for k,w in workers.items()}
        for config in recipes(graph, getattr(args, "matrix", "standard"), learner_optimizer):
            row = dict(config=config,samples=[],captures={},fingerprints={})
            result["cases"].append(row)
            fixture = workers["baseline"].request(dict(op="init",config=config))
            fusion = getattr(args, "fuse_pointwise", False)
            candidate = workers["candidate"].request(dict(op="init",config=dict(config, **({"fuse_pointwise": True} if fusion else {}), **({"fuse_learner_seeds": True} if seed_fusion else {}))))
            if fusion:
                graph_reference.match_fused_fixture(fixture, candidate)
                row["candidate_fixture"] = candidate
            elif seed_fusion:
                match_seed_fixture(fixture, candidate)
                row["candidate_fixture"] = candidate
            else:
                for key in ("config","plan_json","input","target","learning_rate","kernel","accumulation","adapter"):
                    if fixture[key] != candidate[key]: raise ValueError("native paired fixture differs: "+key)
            match_adapter(fixture["adapter"],args.device,name)
            row["fixture"] = fixture
            for cadence in ("immediate","deferred"):
                captured = {}
                for block in range(10):
                    lanes=["baseline","candidate","torch"]
                    rotation=(block+config["seed"])%3
                    order=lanes[rotation:]+lanes[:rotation]
                    sample=dict(cadence=cadence,block=block,warmup=block<2,order=order,times_ms={},fused_learner_seeds={},learner_optimizers={})
                    row["samples"].append(sample)
                    for lane in order:
                        if lane == "torch":
                            value=sample_torch(torch,fixture,args.device,cadence,synchronize)
                        else:
                            value=workers[lane].request(dict(op="learn" if learner else "sample",cadence=cadence,capture=lane not in captured))
                            fingerprint=row["fingerprints"].setdefault(lane,value["state_sha256"])
                            if value["state_sha256"] != fingerprint: raise ValueError("native state changed across identical reset trajectories")
                        validate_sample(value,cadence,config["steps"],learner=learner,lane=lane,seed_fusion=seed_fusion and lane=="candidate",learner_optimizer=learner_optimizer)
                        sample["learner_optimizers"][lane] = value.get("learner_optimizer")
                        sample["fused_learner_seeds"][lane] = value.get("fused_learner_seeds", False)
                        if lane not in captured: captured[lane]=value
                        if lane == "torch": compare(value["state"],captured[lane]["state"])
                        close_values(value["losses"],captured[lane]["losses"])
                        sample["times_ms"][lane]=value["elapsed_ms"]
                    for lane in ("baseline","candidate"):
                        row["captures"][cadence+"_"+lane] = captured[lane]
                        compare(captured[lane]["state"],captured["torch"]["state"])
                        close_values(captured[lane]["losses"],captured["torch"]["losses"])
                    row["captures"][cadence+"_torch"] = captured["torch"]
            row["summary"]={cadence:{lane:statistics.median(s["times_ms"][lane] for s in row["samples"]
                if not s["warmup"] and s["cadence"]==cadence) for lane in ("baseline","candidate","torch")}
                for cadence in ("immediate","deferred")}
            print(json.dumps(dict(config=config,summary=row["summary"])),flush=True)
    finally:
        failures=[]
        for worker in workers.values():
            try: worker.close()
            except Exception as error: failures.append(str(error))
        if failures: raise RuntimeError("native cleanup failed: "+"; ".join(failures))
    if audit.source_identity() != before or admit_device(args.device) != result["device_admission"]:
        raise RuntimeError("source/device admission changed")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for label in ("baseline","candidate"):
        parser.add_argument("--"+label,type=Path,required=True)
        parser.add_argument("--"+label+"-source",required=True)
    parser.add_argument("--device",choices=("mps","cuda"),required=True)
    parser.add_argument("--graph",action="store_true",help="v2 Linear/GELU/gain/ReLU graph, exact gradients")
    parser.add_argument("--learner",action="store_true",help="custom quadratic/quartic VJPs and weighted SGD; update receipts, not per-step loss reads")
    parser.add_argument("--learner-optimizer",choices=("topos_ema","clipped_topos_ema"),
                        help="same zero-initialized EMA in every lane: damping 0.5, optional norm limit 1/1024")
    parser.add_argument("--fuse-learner-seeds",action="store_true",help="candidate-only reusable cubic-cotangent fusion; unchanged graph and optimizer")
    parser.add_argument("--fuse-pointwise",action="store_true",help="opt in to candidate graph fusion; preserve and structurally compare both plans")
    parser.add_argument("--matrix",choices=("standard","wide"),default="standard",
                        help="wide: graph-only 64/128/256 feature widths, same eight SGD updates")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.matrix == "wide" and not args.graph:
        parser.error("--matrix wide requires --graph")
    if args.fuse_pointwise and not args.graph:
        parser.error("--fuse-pointwise requires --graph")
    if args.learner and (not args.graph or args.fuse_pointwise):
        parser.error("--learner requires --graph and disallows changing fusion between lanes")
    if args.fuse_learner_seeds and not args.learner:
        parser.error("--fuse-learner-seeds requires --learner")
    if args.learner_optimizer and not args.learner:
        parser.error("--learner-optimizer requires --learner")
    result=dict(schema="spiraltorch.resident_training_comparison.v1",status="error",cases=[],
        workload="learner" if args.learner else ("graph" if args.graph else "dense"),
        matrix=args.matrix,
        pointwise_fusion=args.fuse_pointwise,
        learner_seed_fusion=args.fuse_learner_seeds,
        learner_optimizer=args.learner_optimizer,
        boundary="rotating native A/B and eager torch; f32 tanh-GELU mean-MSE plain SGD; device-persistent batch/weights; 8 updates, all losses read; 2 warmups+8 retained samples per cadence; reset/probes excluded; no fastest-PyTorch/quality claim",
        safety_difference="Rust validates every intermediate and commits parameters transactionally; eager torch has no matching per-stage finite/rollback checks for these fixed finite fixtures",
        readback_difference="Immediate reads each step; deferred Rust reads owning snapshots after enqueue, while torch stacks retained losses for one final host copy")
    if args.learner:
        result.update(boundary="rotating native A/B and eager Torch; two custom GPU cotangents, exact VJPs and weighted SGD; eight nonzero-rate updates; all Rust acceptance receipts read; reset/probes and terminal state excluded; two warmups+eight retained blocks",
            readback_difference="Rust reads every owning update receipt; Torch only synchronizes completion, per step or at the end. No timed loss observations or equivalent Torch guard/rollback claim.")
    if args.learner_optimizer:
        result["boundary"] += "; matched zero-initialized Topos EMA, damping 0.5, optional global norm limit 1/1024; not torch.optim.SGD momentum; history observed outside timing"
    with args.output.open("x") as output, args.output.with_suffix(".baseline.stderr").open("xb") as baseline_stderr, args.output.with_suffix(".candidate.stderr").open("xb") as candidate_stderr:
        try:
            run(args,result,baseline_stderr,candidate_stderr)
            result["status"]="passed"
        except Exception as error:
            result["error"]=f"{type(error).__name__}: {error}"
        json.dump(result,output,indent=2,allow_nan=False)
        output.write("\n")
    print(json.dumps(dict(status=result["status"],error=result.get("error"))))
    raise SystemExit(result["status"] != "passed")


if __name__ == "__main__": main()
