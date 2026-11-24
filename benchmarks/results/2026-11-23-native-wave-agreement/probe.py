"""Prepare, run once, or re-score saved rows from the matched geometry pilot."""
import argparse
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
P = runpy.run_path(str(HERE/"protocol.py"))
N = runpy.run_path(str(HERE/"native.py"))
digest, encoded, need = (P[k] for k in ("digest", "encoded", "need"))
GIB = 1 << 30
ENV = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                     "RAYON_NUM_THREADS", "MKL_NUM_THREADS"), "1")
ENV.update(SPIRALTORCH_BACKEND="cpu", SPIRALTORCH_AUTOTUNE="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")


def write(path, raw):
    with path.open("xb") as f:
        f.write(raw)


def save(path, data):
    write(path, encoded(data))


def load(path):
    return json.loads(path.read_bytes())


def sources():
    return {name: P["file_hash"](HERE/name) for name in ("protocol.py","native.py","probe.py")}


def plan_from(directory, pin, kind):
    raw = (directory/"plan.json").read_bytes()
    need(digest(raw) == pin, "plan pin mismatch")
    plan = json.loads(raw)
    need(plan["kind"] == kind and plan["source_sha256"] == sources(), "plan scope or code differs")
    return plan


def stages(plan):
    return ["initial-wave","initial-affine"] + [
        "%s-%s" % (seed, arm) for seed in plan["seeds"] for arm in P["ARMS"]]


def capture(directory, name, model, base):
    need(model.base_model_identity == base.base_model_identity and model.lora_artifact_id is None,
         "base or LoRA changed")
    artifact = model.geometry_artifact()
    raw = artifact.to_json().encode()
    need("sha256:"+digest(raw) == artifact.artifact_id == model.geometry_artifact_id, "artifact pin mismatch")
    write(directory/(name+".geometry.json"), raw)
    return artifact.artifact_id


def train_and_seal(st, plan, start, base):
    initial = {k: N["geometry"](st, base, k, plan["shape"], plan["strength"]) for k in ("wave","affine")}
    ids = {"initial-"+k: capture(start, "initial-"+k, m, base) for k,m in initial.items()}
    for seed, order in zip(plan["seeds"], plan["order"]):
        for arm in P["ARMS"]:
            kind = "affine" if arm == "affine_sgd" else "wave"
            name = "%s-%s" % (seed, arm)
            m, hist = N["train_one"](initial[kind], plan["train"], order, plan["lr"], arm == "wave_fixed")
            ids[name] = capture(start, name, m, base)
            need(initial[kind].geometry_artifact_id == ids["initial-"+kind], "initial state mutated")
            need(arm != "wave_fixed" or ids[name] == ids["initial-wave"], "frozen geometry changed")
            save(start/(name+".steps.json"), hist)
    save(start/"weights.json", ids)


def assess(st, base, plan, dest):
    ids = load(dest/"weights.json")
    need(set(ids) == set(stages(plan)), "missing sealed weights")
    for name in stages(plan):
        raw = (dest/(name+".geometry.json")).read_bytes()
        need("sha256:"+digest(raw) == ids[name], "saved geometry differs")
        artifact = st.GeometryAdapterArtifact.from_json(raw.decode(), ids[name], base.base_model_identity)
        model = base.with_geometry(artifact)
        with (dest/(name+".f32le")).open("xb") as f:
            for row in plan["evaluation"]:
                inputs = row["prefix"]
                out = model.forward(inputs, [1,len(inputs)], last_only=True)
                need(list(out.shape) == [1,1,plan["shape"][-1]], "incomplete held-out logits")
                f.write(out.values_buffer())


def check_geometry(state, plan, base_id, kind):
    site, = state["placements"]
    width = plan["shape"][1]
    tensors = {t["name"]:t for t in site["tensors"]}
    wave = kind == "wave"
    need(state["base_model_identity"] == base_id and site["path"] == "transformer.h.%d.attn" % (plan["shape"][0]-1)
         and site["recipe"]["schema"] == ("spiraltorch.wave_gate_adapter.v2" if wave else "spiraltorch.channel_affine_adapter.v1")
         and site["recipe"]["strength"] == plan["strength"], "wrong base or geometry site")
    need(len(tensors) == len(site["tensors"]) and set(tensors) ==
         ({"gate","bias","log_radius"} if wave else {"gate","bias"}), "unexpected trainable state")
    for key in ("gate","bias"):
        t = tensors[key]
        need(t["shape"] == [width] and t["role"] == "parameter" and t["requires_grad"] and
             len(t["values"]) == width and all(map(math.isfinite,t["values"])), "wrong gate or bias")
    if wave:
        t = tensors["log_radius"]
        need(t["values"] == [0.] and not t["requires_grad"] and t["role"] == "buffer" and
             site["recipe"]["learnable_radius"] is False, "fixed radius changed")
    return [tensors[k]["values"] for k in ("gate","bias")]


def score(plan, dest):
    need(sys.byteorder == "little", "readback needs little-endian f32")
    ids, identity = load(dest/"weights.json"), load(dest/"identity.json")
    need(set(ids) == set(stages(plan)) and plan["extension_pin"] in (None,identity["extension"]), "wrong run identity")
    scored, pins, logs, ngrad, napplied = {}, {}, {}, 0, 0
    starts = {}
    width = plan["shape"][-1]
    for name in stages(plan):
        raw = (dest/(name+".geometry.json")).read_bytes()
        need("sha256:"+digest(raw) == ids[name], "stored geometry mismatch")
        kind = "affine" if "affine" in name else "wave"
        params = check_geometry(json.loads(raw),plan,identity["base"],kind)
        if name.startswith("initial-"):
            starts[kind] = params
        data = (dest/(name+".f32le")).read_bytes()
        need(len(data) == 4*width*len(plan["evaluation"]), "incomplete saved logits")
        pins[name] = digest(data)
        vals = memoryview(data).cast("f")
        scored[name] = {row["key"]: P["metrics"](vals[i*width:(i+1)*width], row["gold"], row["rival"])
                        for i,row in enumerate(plan["evaluation"])}
    for seed, order in zip(plan["seeds"], plan["order"]):
        for arm in P["ARMS"]:
            name = "%s-%s" % (seed, arm)
            hist = load(dest/(name+".steps.json"))
            logs[name] = P["file_hash"](dest/(name+".steps.json"))
            prev = ids["initial-"+("affine" if arm=="affine_sgd" else "wave")]
            need(len(hist) == len(order), "wrong gradient count")
            for step,i in zip(hist, order):
                need(step["key"] == plan["train"][i]["key"] and step["before"] == prev and
                     math.isfinite(step["ce"]), "wrong training chain")
                prev = step["after"]
                ngrad += 1
                if arm == "wave_fixed":
                    need(prev == ids["initial-wave"], "fixed control updated")
                else:
                    napplied += 1
            need(ids[name] == prev, "final state differs from last step")
            if arm == "wave_fixed":
                need(pins[name] == pins["initial-wave"], "fixed forward bytes changed")
    need((ngrad,napplied,sum(map(len,scored.values()))) ==
         (plan["gradient_calls"],plan["applied_updates"],plan["scored_rows"]), "execution count mismatch")
    need(starts["wave"] == starts["affine"], "initial channel parameters differ")
    return dict(plan_sha256=digest(encoded(plan)), kind=plan["kind"], gradients=ngrad, updates=napplied,
                scored_rows=plan["scored_rows"], identity=identity, weights=ids, training_log_sha256=logs,
                row_sha256=pins, by_item=scored, **P["summarize"](plan,scored))


def worker(plan, dest, model_dir):
    for module in ("torch","numpy","transformers","safetensors"):
        need(module not in sys.modules, "unexpected preloaded runtime: "+module)
        sys.modules[module] = None
    import spiraltorch as st
    ext = P["file_hash"](Path(st._rs.__file__))
    need(plan["extension_pin"] in (None,ext), "wrong native extension")
    if plan["kind"] == "rehearsal":
        base = N["tiny"](st)
    else:
        archive = st.SafeTensorArchive(model_dir/"model.safetensors")
        config = (model_dir/"config.json").read_bytes()
        need(archive.content_id == "sha256:"+P["PINS"]["model.safetensors"] and
             digest(config) == P["PINS"]["config.json"], "owned base differs")
        base = st.NativeCausalLm.from_gpt2(config.decode(),archive)
        del archive
    base = base.with_forward_readout_storage("original_packed").with_readout_accumulator("f32_wide_backward")
    need(sys.byteorder == "little" and base.execution_backend == "rust_f32_cpu_prepacked" and
         base.lora_artifact_id is None and base.geometry_artifact_id is None, "wrong head or model")
    save(dest/"identity.json",dict(base=base.base_model_identity, extension=ext))
    train_and_seal(st,plan,dest,base)
    assess(st,base,plan,dest)
    save(dest/"worker-result.json",score(plan,dest))


def supervise(plan, directory, pin, model_dir):
    dest = directory/"attempt"
    dest.mkdir(mode=0o700, exist_ok=False)
    start, floor = time.monotonic(), 4*GIB
    minimum = shutil.disk_usage(dest).free
    try:
        need(minimum >= floor+(192<<20), "space admission failed")
        cmd = [sys.executable,"-I","-B",str(HERE/"probe.py"),"_worker",
               "--directory",str(directory),"--kind",plan["kind"],"--pin",pin]
        if model_dir is not None:
            cmd.extend(["--model-dir",str(model_dir)])
        env = dict(os.environ, **ENV, SPIRALTORCH_AUTOTUNE_STORE=str(dest/"tuning.json"))
        with (dest/"worker.log").open("xb") as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                while proc.poll() is None:
                    minimum = min(minimum,shutil.disk_usage(dest).free)
                    need(minimum >= floor and time.monotonic()-start < plan["wall_seconds"], "resource limit")
                    time.sleep(.1)
                need(proc.returncode == 0, "worker stopped; see worker.log")
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
        report = load(dest/"worker-result.json")
        need(report == score(plan,dest), "independent readback mismatch")
        minimum = min(minimum,shutil.disk_usage(dest).free)
        need(minimum >= floor and time.monotonic()-start <= plan["wall_seconds"], "end limit")
        save(dest/"receipt.json",dict(plan=pin, seconds=time.monotonic()-start, min_free=minimum,
                                     result_sha256=P["file_hash"](dest/"worker-result.json")))
    except BaseException as exc:
        save(dest/"failure.json",dict(error=str(exc), min_free=minimum, retry=False))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare","run","verify","_worker"))
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--kind", choices=("gpt2","rehearsal"), required=True)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--pin")
    a = p.parse_args()
    directory = a.directory.absolute()
    if a.action == "prepare":
        plan = P["build_plan"](a.kind,sources(),a.model_dir)
        directory.mkdir(mode=0o700, exist_ok=False)
        save(directory/"plan.json",plan)
        print(digest(encoded(plan)))
    else:
        plan = plan_from(directory,a.pin,a.kind)
        dest = directory/"attempt"
        if a.action == "verify":
            res = score(plan,dest)
            need(not (dest/"failure.json").exists() and res == load(dest/"worker-result.json") and
                 load(dest/"receipt.json")["result_sha256"] == digest(encoded(res)), "unverified attempt")
            print(json.dumps({k:res[k] for k in ("kind","gradients","updates","scored_rows","pilot_positive")}))
        elif a.action == "run":
            need(plan == P["build_plan"](a.kind,sources(),a.model_dir), "plan differs from fixed recipe")
            supervise(plan,directory,a.pin,a.model_dir)
        else:
            need(all(os.environ.get(k) == v for k,v in ENV.items()), "worker environment differs")
            worker(plan,dest,a.model_dir)
