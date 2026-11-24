"""Run one fixed native CPU study per partition, with fresh journals and no retries."""
import argparse
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

audit=SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("audit.py"))))
s=audit.s
x,d,k,v,old,need=s.x,s.d,s.k,s.v,s.old,s.need

def train(st,sp,parent,out):
    hosts=old.hosts(parent,sp)
    with (out/"steps.jsonl").open("xb") as journal,(out/"physical.jsonl").open("xb") as physical:
        for b in sp["bases"]:
            h=hosts[str(b)];base=old.decoder(st,parent,sp,b,h)
            for mode in s.MODES:
                origin=base.with_geometry(st.GeometryAdapterArtifact.create_json(json.dumps(s.record(h,mode,[0.]*6))))
                ident,p0,raw=s.owner(origin,h,mode);n0=s.arm(b,mode)
                save=lambda n,raw:k.native.raw_new(out/"weights"/(n+".json"),raw)
                need(p0==[0.]*6,"nonzero origin");save(n0,raw)
                sig={r["key"]:origin.last_mlp_value_read_signal(r["prompt"],7,8) for r in sp["train"]}
                examples=list(sig.values())
                means=[s.unit(tr,r,h,ident,mode) for tr,r in zip(examples,sp["train"])]
                d.save(out/(n0+"-unit.json"),sig)
                metric=origin.calibrate_prefix_residual_metric([r["prompt"] for r in sp["train"]])
                mr=metric.report();q=s.metric(mr,means,h,ident);d.save(out/(n0+"-metric.json"),mr)
                for n,t,o in s.paths(sp,b,mode):
                    model,pre,params=origin,ident,p0
                    schedule=x.pairs(sp["train"],t,o)*sp["epochs"]
                    for i,pair in enumerate(schedule):
                        g=model.geometry_paired_token_gap_gradients(pair[0]["prompt"],pair[1]["prompt"],[7,8],"weakest_lane")
                        params_by_site=g.parameter_gradients()
                        need(set(params_by_site)=={x.s.SITE},"foreign gradient site")
                        row=dict(name=n,i=i,keys=[r["key"] for r in pair],objective=g.paired_token_gap_report(),
                                 g=dict(base=g.base_model_identity,geometry=g.geometry_artifact_id,params=params_by_site[x.s.SITE],
                                        shape=g.window_shape,tokens=g.target_tokens,loss=g.loss))
                        s.s.validate_gradient(row,pair,"weakest_lane",[sig[z] for z in row["keys"]],h,pre,i==0)
                        nxt,step=model.step_prefix_residual_rms(metric,[g],"diagonal",target_rms=v.TARGET,max_parameter_l2=16.)
                        end,vals,raw=s.owner(nxt,h,mode)
                        row.update(id=end,values=vals,step=step,state=raw.decode())
                        old.archive_step(journal,row);s.ck.step(row,params,pre,q,mr["id"])
                        if i in (0,len(schedule)-1):
                            wr=dict(name=n,i=i,before=s.native_rows(st,model,examples),after=s.native_rows(st,nxt,examples))
                            old.archive_step(physical,wr);s.witness(wr,params,vals,examples,means,step)
                        model,pre,params=nxt,end,vals
                    need(origin.geometry_artifact_id==ident,"mutated origin");save(n,raw)

def score(st,sp,parent,rows,out,files,dest):
    hosts=old.hosts(parent,sp)
    with dest.open("xb") as f:
        for b in sp["bases"]:
            h=hosts[str(b)];base=old.decoder(st,parent,sp,b,h)
            for mode in s.MODES:
                for n in [s.arm(b,mode)]+[n for n,_,_ in s.paths(sp,b,mode)]:
                    rel="weights/"+n+".json"
                    raw=s.s.c.pinned(out/rel,files[rel]);ident,params=s.state(raw,h,mode)
                    model=base.with_geometry(st.GeometryAdapterArtifact.from_json(raw.decode(),ident,h["base"]))
                    need(s.owner(model,h,mode)==(ident,params,raw),"endpoint reload")
                    for r in rows:
                        z=model.forward(r["prompt"],[1,len(r["prompt"])],last_only=True).values()
                        v.u.bits(z);old.archive_step(f,dict(name=n,owner=ident,key=r["key"],logits=z))

def run(root,parent,kind,pin):
    if kind=="formal":
        receipt=(root/"rehearsal-cpu/receipt.json").read_bytes()
        need(json.loads(receipt)["success"] and k.read(root/"admitted.json")==dict(receipt=d.digest(receipt),
             result=d.digest((root/"rehearsal/result.json").read_bytes())),"no admitted rehearsal")
    folder=root/kind;parent=parent/kind
    inp,sp=s.data(folder,pin);need(sp["kind"]==kind,"split")
    installed=k.read(root/"installed.json")
    card=d.save(folder/"run-card.json",dict(inputs=pin,installed=installed,sources=inp["sources"],retry=False,
                                           seconds=300 if kind=="formal" else 120))
    st=v.sig.native(installed["extension"])
    out=folder/"training";out.mkdir(mode=0o700);(out/"weights").mkdir()
    train(st,sp,parent,out)
    score(st,sp,parent,sp["train"],out,k.identities(out),out/"scores.jsonl")
    trained=d.save(folder/"trained.json",dict(inputs=pin,card=card,files=k.identities(out)))
    checked=d.save(folder/"training-checked.json",audit.training(folder,parent,sp,pin,trained))
    print("sealed train",trained,checked,flush=True)
    rows=x.read(folder/"transfer.json",inp["transfer"]);need(rows==s.cases(kind,True),"wrong transfer")
    ev=folder/"evaluation";ev.mkdir(mode=0o700)
    d.save(ev/"consumed.json",dict(inputs=pin,transfer=inp["transfer"],trained=trained,checked=checked,retry=False))
    score(st,sp,parent,rows,out,x.read(folder/"trained.json",trained)["files"],ev/"scores.jsonl")
    scored=d.save(folder/"scored.json",dict(inputs=pin,trained=trained,checked=checked,files=k.identities(ev)))
    print("result",d.save(folder/"result.json",audit.result(folder,parent,pin,trained,checked,scored)),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("root","parent"):p.add_argument(n,type=Path)
    p.add_argument("kind",choices=s.PARENT);p.add_argument("pin")
    run(**vars(p.parse_args()))
