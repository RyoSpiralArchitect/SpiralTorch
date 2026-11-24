"""Frozen train, seal, and transfer sequence; one attempt per synthetic split."""
import argparse
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

a=SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("audit.py"))))
s=a.s
x,old,ck,k,v,d,need=s.x,s.old,s.ck,s.k,s.v,s.d,s.need


def learn(st,sp,parent,out):
    owners=old.hosts(parent,sp)
    with (out/"steps.jsonl").open("xb") as journal,(out/"physical.jsonl").open("xb") as measured:
        for b in sp["bases"]:
            host=owners[str(b)]
            base=old.decoder(st,parent,sp,b,host)
            rec=s.c.record(host,"identity",[0.]*6)
            origin=base.with_geometry(st.GeometryAdapterArtifact.create_json(json.dumps(rec)))
            oid,p0,raw=old.current(origin,host,"identity")
            need(p0==[0.]*6,"nonzero initial parameters")
            save=lambda n,raw:k.native.raw_new(out/"weights"/(n+".json"),raw)
            save(x.name(b),raw)
            sig={}
            for r in sp["train"]:
                tr=origin.last_mlp_prefix_margin_signal(r["prompt"],7,8)
                ck.unit(tr,r,host,oid)
                sig[r["key"]]=tr
            d.save(out/("unit-%d.json"%b),sig)
            examples=list(sig.values())
            metric=origin.calibrate_prefix_residual_metric([r["prompt"] for r in sp["train"]])
            mr=metric.report();d.save(out/("metric-%d.json"%b),mr)
            q=ck.metric(mr,examples,host,oid)
            for n,arm,t,o in s.trajectories(sp,b):
                model,ident,params=origin,oid,p0
                schedule=x.pairs(sp["train"],t,o)*sp["epochs"]
                for i,pair in enumerate(schedule):
                    keys=[p["key"] for p in pair]
                    g=model.geometry_paired_token_gap_gradients(pair[0]["prompt"],pair[1]["prompt"],[7,8],arm)
                    all_params=g.parameter_gradients()
                    need(set(all_params)=={x.s.SITE},"wrong trainable site")
                    row=dict(name=n,i=i,keys=keys,objective=g.paired_token_gap_report(),
                             g=dict(base=g.base_model_identity,geometry=g.geometry_artifact_id,params=all_params[x.s.SITE],
                                    shape=g.window_shape,tokens=g.target_tokens,loss=g.loss))
                    s.validate_gradient(row,pair,arm,[sig[z] for z in keys],host,ident,i==0)
                    nxt,step=model.step_prefix_residual_rms(metric,[g],"diagonal",target_rms=v.TARGET,max_parameter_l2=16.)
                    end,vals,raw=old.current(nxt,host,"identity")
                    row.update(id=end,values=vals,step=step,state=raw.decode())
                    old.archive_step(journal,row)
                    ck.step(row,params,ident,q,mr["id"])
                    if i in (0,len(schedule)-1):
                        w=dict(name=n,i=i,before=v.native_rows(st,model,examples),after=v.native_rows(st,nxt,examples))
                        old.archive_step(measured,w)
                        ck.witness(w,params,vals,examples,step)
                    model,ident,params=nxt,end,vals
                need(origin.geometry_artifact_id==oid,"source changed during updates")
                save(n,raw)


def score(st,parent,sp,rows,weights,sealed,dest):
    owners=old.hosts(parent,sp)
    with dest.open("xb") as log:
        for b in sp["bases"]:
            h=owners[str(b)]
            base=old.decoder(st,parent,sp,b,h)
            for n in [x.name(b)]+[k[0] for k in s.trajectories(sp,b)]:
                rel="weights/"+n+".json"
                raw=s.c.pinned(weights/rel,sealed[rel])
                ident,vec=s.c.state(raw,h,"identity")
                model=base.with_geometry(st.GeometryAdapterArtifact.from_json(raw.decode(),ident,h["base"]))
                need(old.current(model,h,"identity")== (ident,vec,raw),"endpoint reload failed")
                for r in rows:
                    p=r["prompt"];z=model.forward(p,[1,len(p)],last_only=True).values()
                    v.u.bits(z)
                    old.archive_step(log,dict(name=n,owner=ident,key=r["key"],logits=z))


def run(root,parent,kind,pin):
    if kind=="formal":
        rr=(root/"rehearsal-cpu/receipt.json").read_bytes()
        need(json.loads(rr)["success"] and k.read(root/"admitted.json")==dict(receipt=d.digest(rr),
             result=d.digest((root/"rehearsal/result.json").read_bytes())), "missing independently replayed rehearsal")
    installed=k.read(root/"installed.json")
    folder=root/kind;parent=parent/kind
    inp,sp=s.data(folder,pin)
    need(sp["kind"]==kind,"split mismatch")
    card=d.save(folder/"run-card.json",dict(inputs=pin,sources=inp["sources"],installed=installed,retry=False,
                                          limit=300 if kind=="formal" else 120))
    st=v.sig.native(installed["extension"])
    out=folder/"training";out.mkdir(mode=0o700);(out/"weights").mkdir()
    learn(st,sp,parent,out)
    score(st,parent,sp,sp["train"],out,k.identities(out),out/"scores.jsonl")
    trained=d.save(folder/"trained.json",dict(inputs=pin,card=card,files=k.identities(out)))
    checked=d.save(folder/"training-checked.json",a.training(folder,parent,inp,sp,pin,trained))
    print("training sealed",trained,checked,flush=True)
    rows=x.read(folder/"transfer.json",inp["transfer"])
    need(rows==s.cases(kind,True),"transfer prompts changed")
    ev=folder/"evaluation";ev.mkdir(mode=0o700)
    d.save(ev/"consumed.json",dict(inputs=pin,transfer=inp["transfer"],trained=trained,checked=checked,retry=False))
    score(st,parent,sp,rows,out,x.read(folder/"trained.json",trained)["files"],ev/"scores.jsonl")
    scored=d.save(folder/"scored.json",dict(inputs=pin,trained=trained,checked=checked,files=k.identities(ev)))
    result=a.result(folder,parent,pin,trained,checked,scored)
    print("result",d.save(folder/"result.json",result),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("root","parent"):p.add_argument(n,type=Path)
    p.add_argument("kind",choices=s.PREV);p.add_argument("pin")
    run(**vars(p.parse_args()))
