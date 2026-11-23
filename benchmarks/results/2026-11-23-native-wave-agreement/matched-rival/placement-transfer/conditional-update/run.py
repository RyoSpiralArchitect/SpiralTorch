"""One-pass native CPU study. Reusing a directory never resumes or overwrites a run."""
import argparse
import importlib
import math
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
k=SimpleNamespace(**runpy.run_path(str(HERE/"design.py")))
probe=SimpleNamespace(**runpy.run_path(str(HERE/"probe.py")))
s,d,need=k.s,k.d,k.need


def prepare(root,formal):
    root.mkdir(mode=0o700)
    sp=k.plan(formal)
    for b in sp["bases"]:
        raw=s.generate(s.fixtures_cached()[2],b)
        k.native.raw_new(root/("base-%d.safetensors"%b),raw)
    print(d.save(root/"spec.json",sp))


def handle(model,r,gold,mix):
    g=model.geometry_token_menu_gradients(r["prompt"],k.TOK,gold_index=gold,mix=mix)
    report=g.declared_rival_report()
    need((g.geometry_artifact_id,g.base_model_identity,g.lora_artifact_id,g.readout_accumulator)==
         (model.geometry_artifact_id,model.base_model_identity,None,"f32") and g.line_menu_report() is None and
         list(g.window_shape)==[1,len(r["prompt"])+1] and g.target_tokens==1 and
         (report["mix"],report["rival_ids"],report["contrasted_tokens"],report["target_tokens"],report["loss"])==
         (mix,1,1,1,g.loss),"wrong supervision or gradient owner")
    params=g.parameter_gradients()
    need(set(params)=={s.SITE} and set(params[s.SITE])=={"bias","gate"},"wrong gradient coordinates")
    v=[d.f32(x) for name in ("bias","gate") for x in params[s.SITE][name]]
    out=dict(v=v,gold=gold,mix=mix,ce=report["full_ce"],binary=report["declared_ce"],loss=g.loss)
    k.check_grad(out,mix,gold)
    return g,out


def bank(model,rows,logits,train):
    result={}
    for r in rows:
        rec={scope:[handle(model,r,gold,k.MIX[scope])[1] for gold in (0,1)]
             for scope in (k.MIX if train else ("pure",))}
        k.check_bank(rec,logits[r["key"]],train)
        result[r["key"]]=rec
    return result


def snapshot(model,geo,host,where,name,rows):
    raw,pin,vec=k.native.weights(model,geo,host)
    k.native.raw_new(where/"weights"/(name+".json"),raw.encode())
    logits={r["key"]:model.forward(r["prompt"],[1,len(r["prompt"])],last_only=True).values() for r in rows}
    need(all(len(v)==11 and all(math.isfinite(x) and x==d.f32(x) for x in v) for v in logits.values()),"bad logits")
    d.save(where/"logits"/(name+".json"),dict(owner=pin,rows=logits))
    return pin,vec,logits


def run(root,pin):
    raw=(root/"spec.json").read_bytes()
    sp=k.read(root/"spec.json")
    need(d.digest(raw)==pin and raw==d.encoded(k.plan(sp["formal"])),"specification drift")
    attempt=root/"attempt"
    attempt.mkdir(mode=0o700)
    for which in ("weights","logits"):
        (attempt/which).mkdir()
    need(os.environ.get("SPIRALTORCH_BACKEND")=="cpu" and
         all(os.environ.get(x)=="1" for x in ("RAYON_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS")),
         "CPU controls")
    def offline(event,args):
        if event in ("socket.connect","socket.getaddrinfo") or (event=="import" and args[0].split(".")[0] in
                {"torch","numpy","safetensors","transformers","accelerate","huggingface_hub"}):
            raise RuntimeError("offline native study only")
    sys.addaudithook(offline)
    import spiraltorch as st
    extension=importlib.import_module("spiraltorch.spiraltorch")
    need(st.NativeCausalLm is extension.NativeCausalLm and d.digest(Path(extension.__file__).read_bytes())==k.EXT,
         "extension pin")
    hosts,origins,train_bank={},{},{}
    initial=None
    all_rows=sp["train"]+sp["test"]
    for b in sp["bases"]:
        models,hosts[str(b)]=k.native.origins(st,root,sp,b)
        for geo,m in zip(s.GEOS,models):
            start=s.key(b,geo)
            old,vec,z=snapshot(m,geo,hosts[str(b)],attempt,start,all_rows)
            if initial is None:
                initial=vec
            need(vec==initial,"unmatched geometry initialization")
            origins[start]=m,old,vec,z
            train_bank[start]=bank(m,sp["train"],z,True)
    d.save(attempt/"hosts.json",hosts)
    d.save(attempt/"train-bank.json",train_bank)
    with (attempt/"steps.jsonl").open("xb") as log:
        for b,geo,scope,pack,teacher,order in k.specs(sp):
            start=s.key(b,geo)
            cell=s.key(b,geo,scope,pack,teacher,order)
            m,old,before,z=origins[start]
            batches=k.schedule(sp["train"],order,pack)*sp["epochs"]
            for i,rows in enumerate(batches):
                pairs=[handle(m,r,r[teacher],k.MIX[scope]) for r in rows]
                handles,grads=zip(*pairs)
                if i==0:
                    for r,g in zip(rows,grads):
                        need(g==train_bank[start][r["key"]][scope][r[teacher]],"wrong start state")
                m,report=m.step_geometry_cohort_gradient_at_distance(handles,target_l2=len(rows)*d.D)
                _,now,after=k.native.weights(m,geo,hosts[str(b)])
                rec=dict(cell=cell,i=i,keys=[r["key"] for r in rows],members=grads,values=after,id=now,report=report)
                log.write(d.encoded(rec))
                log.flush()
                k.check_step(rec,before,old,hosts[str(b)])
                old,before=now,after
            end,_,_=snapshot(m,geo,hosts[str(b)],attempt,cell,all_rows)
            need(end==old and origins[start][0].geometry_artifact_id==origins[start][1],"endpoint drift")
    # No held-out VJP or linear probe is available to the optimization loop above.
    seal=d.save(root/"endpoints.json",dict(spec=pin,files=k.seal_snapshot(sp,attempt)))
    held={start:bank(m,sp["test"],z,False) for start,(m,_,_,z) in origins.items()}
    held_pin=d.save(root/"held-origin-bank.json",held)
    probes={}
    for name in ["position","bag"]+list(origins):
        orig=origins[name][3] if name in origins else None
        kind="logits" if orig is not None else name
        probes[name]={role:probe.fit(st,kind,sp,orig,role) for role in d.TEACHERS}
        for role,result in probes[name].items():
            probe.verify(kind,sp,orig,role,result)
    probes_pin=d.save(root/"probes.json",probes)
    d.save(root/"completed.json",dict(spec=pin,sealed=seal,held_bank=held_pin,probes=probes_pin))
    print("finished",len(list(k.specs(sp))),"trajectories and",2*len(probes),"linear probes",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    prep=sub.add_parser("prepare")
    prep.add_argument("kind",choices=("rehearsal","formal"))
    prep.add_argument("root",type=Path)
    worker=sub.add_parser("run")
    worker.add_argument("root",type=Path)
    worker.add_argument("sha")
    a=p.parse_args()
    if a.command=="prepare":
        prepare(a.root,a.kind=="formal")
    else:
        run(a.root,a.sha)
