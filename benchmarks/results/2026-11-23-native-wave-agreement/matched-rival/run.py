"""Offline four-arm experiment; Python orchestrates, Rust differentiates and updates."""
import argparse
from functools import lru_cache
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
D = runpy.run_path(str(HERE/"design.py"))
P, N, OLD = D["P"], D["N"], D["OLD"]
need, digest, encoded, f32 = (D[k] for k in ("need","digest","encoded","f32"))
save, load, write = (OLD[k] for k in ("save","load","write"))
FLOOR = 4 << 30


def blob(root, pin):
    need(isinstance(pin,str) and len(pin)==71 and pin.startswith("sha256:") and
         all(c in "0123456789abcdef" for c in pin[7:]), "invalid state identity")
    return root/"states"/(pin[7:]+".json")


def keep(root, model, base):
    a = model.geometry_artifact()
    raw = a.to_json().encode()
    need(a.artifact_id==model.geometry_artifact_id=="sha256:"+digest(raw) and
         model.base_model_identity==base.base_model_identity and model.lora_artifact_id is None,
         "wrong stored state")
    path = blob(root,a.artifact_id)
    if path.exists():
        need(path.read_bytes()==raw,"state identity collision")
    else:
        write(path,raw)
    return a.artifact_id


def gradient(model,row,objective):
    xs = row["prefix"]+[row["gold"]]
    shape, labels = [1,len(xs)], [-100]*(len(xs)-1)+[row["gold"]]
    if objective=="full":
        g = model.adaptation_loss_gradients(xs,shape,mode="geometry",labels=labels)
    else:
        need(objective=="rival","unknown objective")
        g = model.geometry_declared_rival_gradients(xs,shape,labels=labels,mix=1.,
                contrasts=[(len(xs)-1,row["gold"],[row["rival"]])])
    need(g.target_tokens==1 and list(g.window_shape)==shape and g.geometry_artifact_id==model.geometry_artifact_id
         and g.base_model_identity==model.base_model_identity and g.lora_artifact_id is None
         and g.readout_accumulator=="f32_wide_backward","wrong gradient owner or supervision")
    report = g.declared_rival_report()
    D["check_objective"](objective,g.loss,report)
    return g,report


def train(st,base,spec,root):
    (root/"states").mkdir()
    initial = {k:N["geometry"](st,base,k,spec["shape"],spec["strength"]) for k in ("wave","affine")}
    sealed = {"initial-"+k:keep(root,m,base) for k,m in initial.items()}
    site = "transformer.h.%d.attn" % (spec["shape"][0]-1)
    for seed,order in zip(spec["seeds"],spec["order"]):
        for arm in D["ARMS"]:
            kind,obj = arm.split("_")
            stage = "%s-%s"%(seed,arm)
            model = initial[kind]
            with (root/(stage+".jsonl")).open("xb") as log, (root/(stage+".g32")).open("xb") as tape:
                for index in order:
                    row = spec["train"][index]
                    before = model.geometry_artifact()
                    g,report = gradient(model,row,obj)
                    params = g.parameter_gradients()
                    need(set(params)=={site} and set(params[site])=={"gate","bias"} and
                         all(len(params[site][k])==spec["shape"][1] for k in ("gate","bias")), "wrong gradient layout")
                    raw = f32(params[site]["gate"]+params[site]["bias"]).tobytes()
                    model,trace = model.step_geometry_gradient_at_distance(g,spec["target_l2"])
                    after = keep(root,model,base)
                    need(trace["origin_id"]==before.artifact_id and trace["result_id"]==after and
                         trace["realized_l2"]==before.trainable_delta_l2(model.geometry_artifact()),
                         "native state or displacement mismatch")
                    tape.write(raw)
                    log.write(encoded(dict(key=row["key"], position=len(row["prefix"]),gold=row["gold"],
                              rival=row["rival"] if obj=="rival" else None, objective=obj,
                              grad_sha256=digest(raw),loss=g.loss,declared=report,step=trace)))
            need(initial[kind].geometry_artifact_id==sealed["initial-"+kind],"initial model mutated")
            sealed[stage] = model.geometry_artifact_id
    save(root/"sealed.json",sealed)
    return sealed


def measure(st,base,spec,root,sealed):
    for stage,pin in sealed.items():
        raw = blob(root,pin).read_bytes()
        adapter = st.GeometryAdapterArtifact.from_json(raw.decode(),pin,base.base_model_identity)
        model = base.with_geometry(adapter)
        with (root/(stage+".f32le")).open("xb") as sink:
            for row in spec["evaluation"]:
                xs = row["prefix"]
                out = model.forward(xs,[1,len(xs)],last_only=True)
                need(list(out.shape)==[1,1,spec["shape"][-1]],"wrong evaluation shape")
                sink.write(out.values_buffer())


def audit_training(spec,root,sealed,base_id):
    width=spec["shape"][1]
    @lru_cache(maxsize=60)
    def state(pin,kind):
        raw = blob(root,pin).read_bytes()
        need("sha256:"+digest(raw)==pin,"changed state bytes")
        rec=json.loads(raw)
        gate,bias=OLD["check_geometry"](rec,spec,base_id,kind)
        weights=f32(gate+bias)
        for t in rec["placements"][0]["tensors"]:
            if t["requires_grad"]:
                del t["values"]
        return weights,digest(encoded(rec))
    need(state(sealed["initial-wave"],"wave")[0]==state(sealed["initial-affine"],"affine")[0],
         "different initial gate/bias")
    routes, count={},0
    for seed,order in zip(spec["seeds"],spec["order"]):
        for arm in D["ARMS"]:
            stage="%s-%s"%(seed,arm)
            kind,obj=arm.split("_")
            pin=sealed["initial-"+kind]
            initial,layout=state(pin,kind)
            journal=(root/(stage+".jsonl")).read_bytes()
            rows=[json.loads(line) for line in journal.splitlines()]
            tape=(root/(stage+".g32")).read_bytes()
            stride=width*2*4
            need(len(rows)==len(order) and len(tape)==len(order)*stride,"partial gradient tape")
            actual,errors=[],[]
            for j,(r,index) in enumerate(zip(rows,order)):
                case=spec["train"][index]
                need((r["key"],r["position"],r["gold"],r["rival"],r["objective"])==
                     (case["key"],len(case["prefix"]),case["gold"],case["rival"] if obj=="rival" else None,obj),
                     "training row or target differs")
                D["check_objective"](obj,r["loss"],r["declared"])
                raw=tape[j*stride:(j+1)*stride]
                need(digest(raw)==r["grad_sha256"],"changed sealed gradient")
                trace=r["step"]
                need(trace["origin_id"]==pin,"broken step predecessor")
                old,_=state(pin,kind)
                pin=trace["result_id"]
                new,new_layout=state(pin,kind)
                need(new_layout==layout,"changed recipe or frozen state")
                dist,err=D["check_step"](old,new,memoryview(raw).cast("f"),trace,spec)
                actual.append(dist)
                errors.append(err)
                count+=1
            need(pin==sealed[stage],"wrong final state")
            end,_=state(pin,kind)
            routes[stage]=dict(path_l2=math.fsum(actual),min_step=min(actual),max_step=max(actual),
                               max_direction_error=max(errors),endpoint_l2=D["length"](D["delta"](end,initial)),
                               journal_sha256=digest(journal),tape_sha256=digest(tape))
    need(count==spec["updates"],"wrong update count")
    return routes


def replay(spec,root):
    need(sys.byteorder=="little","require little endian")
    sealed,identity=load(root/"sealed.json"),load(root/"identity.json")
    required={"initial-wave","initial-affine"} | {"%s-%s"%(s,a) for s in spec["seeds"] for a in D["ARMS"]}
    need(set(sealed)==required and spec["extension_pin"] in (None,identity["extension"]) and
         identity["backend"]=="rust_f32_cpu_prepacked" and identity["head"]=="original_packed" and
         identity["accumulator"]=="f32_wide_backward","wrong weights or runtime")
    routes=audit_training(spec,root,sealed,identity["base"])
    scored,pins={},{}
    vocab=spec["shape"][-1]
    for stage in sorted(required):
        raw=(root/(stage+".f32le")).read_bytes()
        need(len(raw)==4*vocab*len(spec["evaluation"]),"partial prediction row")
        logits=memoryview(raw).cast("f")
        scored[stage]={r["key"]:P["metrics"](logits[i*vocab:(i+1)*vocab],r["gold"],r["rival"])
                       for i,r in enumerate(spec["evaluation"])}
        pins[stage]=digest(raw)
    need(sum(map(len,scored.values()))==spec["scored_rows"],"wrong prediction count")
    return dict(plan=digest(encoded(spec)),kind=spec["kind"],identity=identity,updates=spec["updates"],
                scored_rows=spec["scored_rows"],paths=routes,weights=sealed,row_sha256=pins,
                by_item=scored,**D["contrast"](spec,scored))


def execute(spec,root,model_dir):
    for key in ("torch","numpy","transformers","safetensors"):
        need(key not in sys.modules,"unexpected preloaded package")
        sys.modules[key]=None
    import spiraltorch as st
    ext=P["file_hash"](Path(st._rs.__file__))
    need(spec["extension_pin"] in (None,ext),"wrong extension")
    if spec["kind"]=="rehearsal":
        base=N["tiny"](st)
    else:
        config=(model_dir/"config.json").read_bytes()
        weights=st.SafeTensorArchive(model_dir/"model.safetensors")
        need(weights.content_id=="sha256:"+P["PINS"]["model.safetensors"] and
             digest(config)==P["PINS"]["config.json"],"owned model bytes differ")
        base=st.NativeCausalLm.from_gpt2(config.decode(),weights)
        del weights
    base=base.with_forward_readout_storage("original_packed").with_readout_accumulator("f32_wide_backward")
    need(base.geometry_artifact_id is None and base.lora_artifact_id is None and
         (base.execution_backend,base.forward_readout_storage,base.readout_accumulator)==
         ("rust_f32_cpu_prepacked","original_packed","f32_wide_backward"),"wrong initial model")
    save(root/"identity.json",dict(base=base.base_model_identity,extension=ext,backend=base.execution_backend,
                                   head=base.forward_readout_storage,accumulator=base.readout_accumulator))
    sealed=train(st,base,spec,root)
    # No held-out model calls until every arm is sealed and every actual step verified.
    audit_training(spec,root,sealed,base.base_model_identity)
    measure(st,base,spec,root,sealed)
    save(root/"worker-result.json",replay(spec,root))


def verify(spec,directory,pin):
    root=directory/"attempt"
    receipt=load(root/"receipt.json")
    need(not (root/"failure.json").exists() and receipt["plan"]==pin and
         receipt["min_free"]>=FLOOR and 0<receipt["seconds"]<=spec["seconds"],"invalid completion receipt")
    result=replay(spec,root)
    need(result==load(root/"worker-result.json") and receipt["result_sha256"]==digest(encoded(result)),
         "stored result does not replay")
    return result


def launch(spec,directory,model_dir,pin):
    root=directory/"attempt"
    root.mkdir(mode=0o700,exist_ok=False)
    start=time.monotonic()
    minimum=shutil.disk_usage(root).free
    try:
        need(minimum>=FLOOR+(192<<20),"disk admission")
        args=[sys.executable,"-I","-B",str(HERE/"run.py"),"_worker","--kind",spec["kind"],
              "--directory",str(directory),"--pin",pin]
        if model_dir is not None:
            args.extend(["--model-dir",str(model_dir)])
        env=dict(os.environ,**OLD["ENV"],SPIRALTORCH_AUTOTUNE_STORE=str(root/"tuning.json"))
        with (root/"worker.log").open("xb") as log:
            proc=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT)
            try:
                while proc.poll() is None:
                    minimum=min(minimum,shutil.disk_usage(root).free)
                    need(minimum>=FLOOR and time.monotonic()-start<spec["seconds"],"resource bound")
                    time.sleep(.1)
                need(proc.returncode==0,"worker stopped: inspect first error in worker.log")
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
        minimum=min(minimum,shutil.disk_usage(root).free)
        elapsed=time.monotonic()-start
        need(minimum>=FLOOR and elapsed<=spec["seconds"],"end bound")
        result=load(root/"worker-result.json")
        need(result["plan"]==pin and result["updates"]==spec["updates"] and result["scored_rows"]==spec["scored_rows"],
             "incomplete worker")
        save(root/"receipt.json",dict(plan=pin,seconds=elapsed,min_free=minimum,result_sha256=digest(encoded(result))))
    except BaseException as exc:
        save(root/"failure.json",dict(error=str(exc),min_free=minimum,retry=False))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("prepare","run","verify","_worker"))
    p.add_argument("--kind",choices=("rehearsal","gpt2"),required=True)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--model-dir",type=Path)
    p.add_argument("--pin")
    a=p.parse_args()
    directory=a.directory.absolute()
    if a.action=="prepare":
        spec=D["plan"](a.kind,a.model_dir)
        directory.mkdir(mode=0o700,exist_ok=False)
        save(directory/"plan.json",spec)
        print(digest(encoded(spec)))
        return
    raw=(directory/"plan.json").read_bytes()
    need(digest(raw)==a.pin,"plan pin mismatch")
    spec=json.loads(raw)
    need(encoded(spec)==raw and spec["kind"]==a.kind and spec["sources"]==D["sources"](),"wrong source or scope")
    if a.action=="run":
        need(D["plan"](a.kind,a.model_dir)==spec,"changed fixed inputs")
        launch(spec,directory,a.model_dir,a.pin)
    elif a.action=="verify":
        result=verify(spec,directory,a.pin)
        print(json.dumps({k:result[k] for k in ("kind","updates","scored_rows","objective_advantage","wave_specific")}))
    else:
        need(all(os.environ.get(k)==v for k,v in OLD["ENV"].items()),"wrong worker environment")
        execute(spec,directory/"attempt",a.model_dir)


if __name__=="__main__":
    main()
