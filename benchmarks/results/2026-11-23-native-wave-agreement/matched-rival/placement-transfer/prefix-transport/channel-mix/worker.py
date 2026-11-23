"""Train using only the training manifest, seal the endpoints, then separately admit new evaluation."""
import argparse
import json
import math
from itertools import product
from pathlib import Path
import runpy
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
c=SimpleNamespace(**runpy.run_path(str(HERE/"design.py")))
v,s,k,d,p,need=c.v,c.s,c.k,c.d,c.p,c.need


def hosts(parent,sp):
    manifest=json.loads(c.pinned(parent/"complete.json",sp["parent_manifest"]))["files"]
    return json.loads(c.pinned(parent/"attempt/hosts.json",manifest["hosts.json"]))


def decoder(st,parent,sp,b,host):
    raw=c.pinned(parent/("base-%d.safetensors"%b),sp["weights"][str(b)])
    base=st.NativeCausalLm.from_gpt2(s.fixtures_cached()[0]["config"],st.SafeTensorArchive.from_bytes(raw)).with_readout_accumulator("f32")
    need(base.base_model_identity==host["original_base"],"foreign decoder")
    frozen=base.with_lora_json(host["lora_weights"]).freeze_lora()
    need(frozen.base_model_identity==host["base"] and frozen.frozen_lora_artifact_id==host["frozen_lora"]
         and frozen.frozen_lora_to_json()==host["lora_weights"] and frozen.lora_artifact_id is None,"foreign LoRA")
    return frozen


def current(m,host,mode):
    raw=m.geometry_artifact().to_json().encode()
    ident,vec=c.state(raw,host,mode)
    need(ident==m.geometry_artifact_id and m.base_model_identity==host["base"] and m.readout_accumulator=="f32"
         and m.lora_artifact_id is None and m.frozen_lora_artifact_id==host["frozen_lora"]
         and m.frozen_lora_to_json()==host["lora_weights"],"owner drift")
    return ident,vec,raw


def archive_step(journal,record):
    journal.write(d.encoded(record))
    journal.flush()


def train(root,parent,prev,signal,inputs,extension,st):
    _,sp=c.data(root,inputs)
    owners=hosts(parent,sp)
    bank=c.raw_signals(signal,sp)
    manifest,past=c.historical(prev,sp["kind"])
    out=root/"training"
    out.mkdir(mode=0o700)
    (out/"weights").mkdir()
    with (out/"steps.jsonl").open("xb") as journal,(out/"physical.jsonl").open("xb") as physical:
        for b in sp["bases"]:
            host=owners[str(b)]; frozen=decoder(st,parent,sp,b,host)
            examples=[bank[b,r["key"]]["trace"] for r in sp["train"]]
            moments=v.moments(examples)
            for mode in c.MAPS:
                start=s.key(b,mode)
                art=st.GeometryAdapterArtifact.create_json(json.dumps(c.record(host,mode,[0.]*6)))
                zero=frozen.with_geometry(art)
                orig,initial,raw=current(zero,host,mode)
                k.native.raw_new(out/"weights"/(start+".json"),raw)
                metric=zero.calibrate_prefix_residual_metric([r["prompt"] for r in sp["train"]])
                report=metric.report()
                d.save(out/("metric-"+start+".json"),report)
                need(report["anchor"]==orig and report["samples"]==len(examples) and
                     report["base_model_identity"]==host["base"] and report["frozen_lora_artifact_id"]==host["frozen_lora"]
                     and len(report["second_moments"])==3 and all(math.isclose(a,b,rel_tol=1e-14) for a,b in zip(moments,report["second_moments"])),"wrong calibration")
                if mode=="identity":
                    ref=s.key(b,c.TURN,"gate")
                    refm="metric-"+ref+".json"
                    need(report==json.loads(c.pinned(prev/"attempt"/refm,manifest[refm])) and
                         all(r["geometry_artifact_id"]==orig for r in examples),"unmixed anchor")
                for teacher,order in product(d.TEACHERS,sp["orders"]):
                    name=s.key(b,mode,teacher,order)
                    m,ident,vec=zero,orig,initial
                    schedule=k.schedule(sp["train"],order,"pair")*sp["epochs"]
                    for i,pair in enumerate(schedule):
                        gs,ms=zip(*(v.u.old.handle(m,r,r[teacher],1.) for r in pair))
                        r=dict(cell=name,i=i,keys=[row["key"] for row in pair],members=ms)
                        if i==0:
                            for sample,mem in zip(pair,ms):
                                c.check_origin(mem,bank[b,sample["key"]],mode)
                        nxt,step=m.step_prefix_residual_rms(metric,gs,"diagonal",target_rms=c.TARGET,max_parameter_l2=16.)
                        after,vals,raw=current(nxt,host,mode)
                        r.update(id=after,values=vals,report=step)
                        archive_step(journal,r)
                        v.validate_step(r,vec,ident,"diagonal",moments,report["id"])
                        if mode=="identity":
                            ref=past[b,teacher,order,i]
                            need(d.encoded(dict(r,cell=ref["cell"]))==d.encoded(ref),"unmixed update differs from raw predecessor")
                        if i in (0,len(schedule)-1):
                            x,y=v.native_rows(st,m,examples),v.native_rows(st,nxt,examples)
                            archive_step(physical,dict(cell=name,i=i,before=x,after=y))
                            need(c.same_rows(x,c.outputs(examples,vec[3:],mode)) and c.same_rows(y,c.outputs(examples,vals[3:],mode))
                                 and abs(v.rms_rows(x,y)-step["realized_rms"])<=2e-6,"native delivered residual")
                        m,ident,vec=nxt,after,vals
                    k.native.raw_new(out/"weights"/(name+".json"),raw)
                    if mode=="identity":
                        f="weights/"+s.key(b,c.TURN,"diagonal",teacher,order)+".json"
                        need(raw==c.pinned(prev/"attempt"/f,manifest[f]),"historical endpoint bytes")
    files=k.identities(out)
    return d.save(root/"trained.json",dict(inputs=inputs,extension=extension,files=files,
                                            run_card=d.digest((root/"run-card.json").read_bytes())))


def evaluate(root,parent,inputs,trained,checked,extension,st):
    info,sp=c.data(root,inputs)
    m=json.loads(c.pinned(root/"trained.json",trained))
    need(json.loads(c.pinned(root/"training-checked.json",checked))["trained"]==trained,"training audit missing")
    need(m["inputs"]==inputs and m["extension"]==extension and m["files"]==k.identities(root/"training"),"unsealed training")
    cases=json.loads(c.pinned(root/"evaluation.json",info["evaluation"]))
    need(cases==c.fresh(sp["kind"]),"new template drift")
    out=root/"evaluation"
    out.mkdir(mode=0o700)
    # Durable before the first call: a stopped or partly scored split is never called fresh again.
    d.save(out/"consumed.json",dict(trained=trained,checked=checked,evaluation=info["evaluation"],retry=False))
    owners=hosts(parent,sp)
    with (out/"scores.jsonl").open("xb") as log:
        for b in sp["bases"]:
            host=owners[str(b)]; base=decoder(st,parent,sp,b,host)
            origin_z={}
            for mode in c.MAPS:
                start=s.key(b,mode)
                names=[start]+[s.key(b,mode,t,o) for t,o in product(d.TEACHERS,sp["orders"])]
                for name in names:
                    path="weights/"+name+".json"
                    raw=c.pinned(root/"training"/path,m["files"][path])
                    ident,vec=c.state(raw,host,mode)
                    restored=base.with_geometry(st.GeometryAdapterArtifact.from_json(raw.decode(),ident,host["base"]))
                    need(current(restored,host,mode)==(ident,vec,raw),"endpoint reload")
                    for r in cases:
                        z=restored.forward(r["prompt"],[1,len(r["prompt"])],last_only=True).values()
                        archive_step(log,dict(name=name,key=r["key"],owner=ident,logits=z))
                        bits=v.u.bits(z)
                        if name==start:
                            need(origin_z.setdefault(r["key"],bits)==bits,"nonidentical zero-gate reference")
    return d.save(root/"scored.json",dict(trained=trained,inputs=inputs,files=k.identities(out)))


def run(root,parent,previous,signal,kind,inputs,extension):
    root,parent,previous,signal=(x/kind for x in (root,parent,previous,signal))
    need(c.data(root,inputs)[1]["kind"]==kind,"split mismatch")
    d.save(root/"run-card.json",dict(inputs=inputs,extension=extension,sources=c.sources()))
    st=v.sig.native(extension)
    trained=train(root,parent,previous,signal,inputs,extension,st)
    print("training sealed",trained,flush=True)
    reviewer=runpy.run_path(str(HERE/"verify.py"))
    evidence=reviewer["verify_training"](root,parent,previous,signal,inputs,trained)
    checked=d.save(root/"training-checked.json",evidence)
    print("training audited",checked,flush=True)
    print("scoring sealed",evaluate(root,parent,inputs,trained,checked,extension,st),flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for n in ("root","parent","previous","signal"): cli.add_argument(n,type=Path)
    cli.add_argument("kind",choices=list(c.PREVIOUS));cli.add_argument("inputs");cli.add_argument("extension")
    run(**vars(cli.parse_args()))
