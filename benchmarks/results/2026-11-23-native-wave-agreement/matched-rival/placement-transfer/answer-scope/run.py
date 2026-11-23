"""Prepare a frozen synthetic study or execute it once. No external model or ML package."""
import argparse
import importlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import types

s = types.SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("study.py"))))
d, need = s.d, s.need


def raw_new(path, data):
    with path.open("xb") as f:
        f.write(data)


def prepare(root, formal, extension):
    root.mkdir(mode=0o700)
    sp = s.spec(formal,extension)
    for seed in sp["bases"]:
        raw_new(root/("base-%d.safetensors"%seed),s.generate(s.fixtures_cached()[2],seed))
    print(s.save(root/"spec.json",sp))


def origins(st,root,sp,seed):
    data = (root/("base-%d.safetensors"%seed)).read_bytes()
    need(s.digest(data)==sp["weights"][str(seed)],"base archive hash")
    case,lora,_ = s.fixtures_cached()
    base = st.NativeCausalLm.from_gpt2(case["config"],st.SafeTensorArchive.from_bytes(data)).with_readout_accumulator("f32")
    need(base.base_model_identity != d.BASE,"not a new base")
    lora = dict(lora,base_model_identity=base.base_model_identity)
    geo = dict(json.loads(case["geometry"]),base_model_identity=base.base_model_identity)
    active = base.with_lora_json(json.dumps(lora)).with_geometry(st.GeometryAdapterArtifact.create_json(json.dumps(geo)))
    wave = active.freeze_lora().with_single_geometry_at(s.SITE)
    art = json.loads(wave.geometry_artifact().to_json())
    p, = art["placements"]
    w = p["recipe"]
    p["recipe"] = dict(schema="spiraltorch.channel_affine_adapter.v1",features=w["features"],
                       strength=w["strength"],max_values=w["kernel"]["max_values"])
    affine = wave.with_geometry(st.GeometryAdapterArtifact.create_json(json.dumps(art)))
    raw = active.lora_to_json()
    fixed = d.owner_id(raw.encode(),d.REF)
    need(fixed == active.lora_artifact_id and fixed == wave.frozen_lora_artifact_id and
         json.loads(raw)==lora and wave.frozen_lora_to_json()==raw, "LoRA values changed")
    return (wave,affine),dict(original_base=base.base_model_identity,base=wave.base_model_identity,
                              frozen_lora=fixed,lora_weights=raw)


def weights(m,geo,host):
    raw = m.geometry_artifact().to_json()
    pin = d.identity(raw.encode())
    need(m.geometry_artifact_id==pin and m.base_model_identity==host["base"] and m.readout_accumulator=="f32" and
         m.lora_artifact_id is None and m.frozen_lora_artifact_id==host["frozen_lora"] and
         m.frozen_lora_to_json()==host["lora_weights"], "owner or precision drift")
    return raw,pin,s.coordinates(raw,geo,host["base"])


def capture(m,path,pin,sp):
    rows = sp["train"]+sp["test"]
    out = {}
    for r in rows:
        costs = m.score_lines(r["prompt"],d.CANDIDATES,stop_id=0)
        s.P.verify_costs(costs)
        out[r["key"]] = costs
    p = rows[0]["prompt"]
    logits = [m.forward(ids,[1,len(ids)],last_only=True).values() for ids in (p,p+[7],p+[8])]
    s.P.verify_sentinel(logits,out[rows[0]["key"]])
    s.save(path,dict(owner=pin,rows=out,sentinel=logits))
    return out


def gradient(m,row,gold,scope):
    prompt = row["prompt"]
    if scope == "answer":
        g = m.geometry_token_menu_gradients(prompt,s.TOKENS,gold_index=gold,mix=.5)
        report = g.declared_rival_report()
        need((report["mix"],report["contrasted_tokens"],report["rival_ids"],g.target_tokens)==(.5,1,1,1)
             and g.line_menu_report() is None and list(g.window_shape)==[1,len(prompt)+1], "wrong answer supervision")
        expected = .5*(report["full_ce"]+report["declared_ce"])
    else:
        need(scope=="through_stop","unknown supervision")
        g = m.geometry_line_menu_gradients(prompt,d.CANDIDATES,gold_index=gold,stop_id=0,mix=.5,max_tile_values=11)
        report = g.line_menu_report()
        need(report["gold_index"]==gold and report["mix"]==.5 and report["target_counts"]==[2,2] and
             report["stop_id"]==0 and g.target_tokens==4 and g.declared_rival_report() is None, "wrong line supervision")
        expected = .5*(report["gold_mean_nll"]+report["menu_nll"])
    need(g.base_model_identity==m.base_model_identity and g.geometry_artifact_id==m.geometry_artifact_id and
         g.lora_artifact_id is None and g.readout_accumulator=="f32" and abs(float(g.loss)-expected)<3e-6,
         "gradient owner, precision or mixed objective")
    params = g.parameter_gradients()
    need(set(params)=={s.SITE} and set(params[s.SITE])=={"bias","gate"},"unexpected trainable site")
    vector = [d.f32(x) for name in ("bias","gate") for x in params[s.SITE][name]]
    return g,report,vector


def finite_difference(st,m,scope,row):
    _,_,analytical = gradient(m,row,row["head"],scope)
    artifact = m.geometry_artifact().to_json()
    measured = []
    for i in range(6):
        outputs,positions = [],[]
        for sign in (-1,1):
            record=json.loads(artifact)
            tensor=record["placements"][0]["tensors"][i//3]["values"]
            tensor[i%3] = d.f32(tensor[i%3]+sign/512.)
            positions.append(tensor[i%3])
            perturbed=m.with_geometry(st.GeometryAdapterArtifact.create_json(json.dumps(record)))
            c=perturbed.score_lines(row["prompt"],d.CANDIDATES,stop_id=0)
            outputs.append(s.loss(c,row["head"],scope))
        measured.append((outputs[1]-outputs[0])/(positions[1]-positions[0]))
    need(max(map(abs,analytical))>1e-6 and
         all(abs(x-y)<=6e-5+.015*abs(x) for x,y in zip(analytical,measured)), "disjoint finite difference")
    return dict(analytical=analytical,central_difference=measured)


def execute(root,pin):
    sp = json.loads((root/"spec.json").read_bytes())
    need(pin==s.digest((root/"spec.json").read_bytes()) and s.encoded(sp)==s.encoded(s.spec(sp["formal"],sp["extension"])),
         "plan or source changed")
    out=root/"attempt"
    out.mkdir(mode=0o700)
    for name in ("weights","scores"):
        (out/name).mkdir(mode=0o700)
    need(os.environ.get("SPIRALTORCH_BACKEND")=="cpu" and
         all(os.environ.get(k)=="1" for k in ("RAYON_NUM_THREADS","VECLIB_MAXIMUM_THREADS","OMP_NUM_THREADS")), "CPU controls")
    def guard(event,args):
        if event in ("socket.connect","socket.getaddrinfo") or (event=="import" and args[0].split(".")[0] in
                {"torch","numpy","transformers","safetensors","accelerate","huggingface_hub"}):
            raise RuntimeError("offline native study")
    sys.addaudithook(guard)
    import spiraltorch as st
    native=importlib.import_module("spiraltorch.spiraltorch")
    need(st.NativeCausalLm is native.NativeCausalLm and s.digest(Path(native.__file__).read_bytes())==sp["extension"],
         "installed native extension identity")
    by_key, origin_costs, hosts = {},{},{}
    for seed in sp["bases"]:
        models,host=origins(st,root,sp,seed)
        hosts[str(seed)]=host
        vectors=[]
        for geo,m in zip(s.GEOS,models):
            k=s.key(seed,geo)
            raw,old,vec=weights(m,geo,host)
            raw_new(out/"weights"/(k+".json"),raw.encode())
            by_key[k]=m,old,vec
            vectors.append(vec)
            origin_costs[k]=capture(m,out/"scores"/(k+".json"),old,sp)
            if not sp["formal"]:
                s.save(out/("finite-"+k+".json"),{t:finite_difference(st,m,t,sp["train"][0]) for t in s.SCOPES})
        need(vectors[0]==vectors[1],"unmatched six initial scalars")
    need(len({h["base"] for h in hosts.values()})==len(hosts),"duplicate bases")
    s.save(out/"hosts.json",hosts)
    with (out/"steps.jsonl").open("xb") as journal:
        for seed,geo,scope,teacher,order in s.cells(sp):
            start=s.key(seed,geo)
            cell=s.key(seed,geo,scope,teacher,order)
            m,previous,vec=by_key[start]
            for i,r in enumerate(d.schedule(sp["train"],order)*sp["epochs"]):
                gold=r[teacher]
                g,report,grad=gradient(m,r,gold,scope)
                if i==0:
                    need(abs(s.loss(origin_costs[start][r["key"]],gold,scope)-g.loss)<3e-6,"first loss vs label-free costs")
                m,ray=m.step_adaptation_gradient_at_distance(g,mode="geometry",target_l2=d.D)
                raw,now,after=weights(m,geo,hosts[str(seed)])
                rec=dict(cell=cell,i=i,row=r["key"],gold=gold,id=now,values=after,grad=grad,
                         loss=g.loss,report=report,ray=ray)
                journal.write(s.encoded(rec))
                journal.flush()
                s.verify_step(rec,vec,previous,now)
                previous,vec=now,after
            raw_new(out/"weights"/(cell+".json"),raw.encode())
            capture(m,out/"scores"/(cell+".json"),now,sp)
            need(by_key[start][0].geometry_artifact_id==by_key[start][1],"origin mutated")
    files={p.relative_to(out).as_posix():s.digest(p.read_bytes()) for p in sorted(out.rglob("*")) if p.is_file()}
    s.save(root/"complete.json",dict(spec=pin,files=files))
    print("finished",len(list(s.cells(sp))),"trajectories",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="cmd",required=True)
    start=sub.add_parser("prepare")
    start.add_argument("kind",choices=("formal","rehearsal"))
    start.add_argument("directory",type=Path)
    start.add_argument("extension_sha256")
    run=sub.add_parser("run")
    run.add_argument("directory",type=Path)
    run.add_argument("spec_sha256")
    a=p.parse_args()
    if a.cmd=="prepare":
        prepare(a.directory,a.kind=="formal",a.extension_sha256)
    else:
        execute(a.directory,a.spec_sha256)
