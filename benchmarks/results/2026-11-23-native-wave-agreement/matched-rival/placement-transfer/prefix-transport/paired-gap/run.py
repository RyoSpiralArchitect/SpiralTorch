"""One bounded attempt. Load only train rows until their endpoints and logits pass the audit."""
import argparse
from itertools import product
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
check = SimpleNamespace(**runpy.run_path(str(HERE/"checks.py")))
x, need = check.x, check.need
old, c, d, v, k = x.old,x.c,x.d,x.v,x.k
audit = SimpleNamespace(**runpy.run_path(str(HERE/"audit.py")))


def weight_path(folder,name):
    return folder/"weights"/(name+".json")


def capture(st, parent, sp, out):
    owners = old.hosts(parent,sp)
    with (out/"steps.jsonl").open("xb") as journal, (out/"physical.jsonl").open("xb") as phys:
        for b in sp["bases"]:
            h = owners[str(b)]
            base = old.decoder(st,parent,sp,b,h)
            zero = base.with_geometry(st.GeometryAdapterArtifact.create_json(json.dumps(c.record(h,"identity",[0.]*6))))
            orig, start, raw = old.current(zero,h,"identity")
            need(start==[0.]*6,"incorrect trainable origin")
            k.native.raw_new(weight_path(out,x.name(b)),raw)
            signals = {}
            for row in sp["train"]:
                trace = zero.last_mlp_prefix_margin_signal(row["prompt"],7,8)
                signals[row["key"]] = trace
                check.unit(trace,row,h,orig)
            d.save(out/("signals-%d.json"%b),signals)
            examples = list(signals.values())
            metric = zero.calibrate_prefix_residual_metric([r["prompt"] for r in sp["train"]])
            mr = metric.report()
            d.save(out/("metric-%d.json"%b),mr)
            moments = check.metric(mr,examples,h,orig)
            for _,arm,teacher,order in x.cells(sp,b):
                name = x.name(b,arm,teacher,order)
                m,pre,params = zero,orig,start
                schedule = x.pairs(sp["train"],teacher,order)*sp["epochs"]
                for i, pair in enumerate(schedule):
                    keys = [r["key"] for r in pair]
                    traces = [signals[p] for p in keys]
                    opts = dict(reference_center=check.center(traces),stiffness=1.) if arm=="anchored_rank" else {}
                    g = m.geometry_paired_token_gap_gradients(pair[0]["prompt"],pair[1]["prompt"],[7,8],arm,**opts)
                    pg = g.parameter_gradients()
                    need(set(pg)=={x.s.SITE},"unexpected parameter site")
                    r = dict(name=name,i=i,keys=keys,objective=g.paired_token_gap_report(),
                             g=dict(geometry=g.geometry_artifact_id,base=g.base_model_identity,params=pg[x.s.SITE],
                                    loss=g.loss,shape=g.window_shape,tokens=g.target_tokens))
                    check.gradient(r,pair,arm,traces,h,pre,i==0)
                    nxt, stp = m.step_prefix_residual_rms(metric,[g],"diagonal",target_rms=v.TARGET,max_parameter_l2=16.)
                    post, vals, raw = old.current(nxt,h,"identity")
                    r.update(id=post,values=vals,step=stp,state=raw.decode())
                    old.archive_step(journal,r)
                    check.step(r,params,pre,moments,mr["id"])
                    if i in (0,len(schedule)-1):
                        w = dict(name=name,i=i,before=v.native_rows(st,m,examples),after=v.native_rows(st,nxt,examples))
                        old.archive_step(phys,w)
                        check.witness(w,params,vals,examples,stp)
                    m,pre,params = nxt,post,vals
                need(zero.geometry_artifact_id==orig,"alias changed origin")
                k.native.raw_new(weight_path(out,name),raw)


def score(st, parent, sp, cases, weights, sealed, file):
    owners = old.hosts(parent,sp)
    with file.open("xb") as sink:
        for b in sp["bases"]:
            h = owners[str(b)]
            base = old.decoder(st,parent,sp,b,h)
            for name in [x.name(b)]+[x.name(*cell) for cell in x.cells(sp,b)]:
                path = weight_path(weights,name)
                raw = c.pinned(path,sealed[path.relative_to(weights).as_posix()])
                ident, vec = c.state(raw,h,"identity")
                model = base.with_geometry(st.GeometryAdapterArtifact.from_json(raw.decode(),ident,h["base"]))
                need(old.current(model,h,"identity")==(ident,vec,raw),"reloaded state mismatch")
                for row in cases:
                    ids = row["prompt"]
                    z = model.forward(ids,[1,len(ids)],last_only=True).values()
                    v.u.bits(z)
                    old.archive_step(sink,dict(name=name,owner=ident,key=row["key"],logits=z))


def campaign(root,parent,kind,pin,extension):
    if kind=="formal":
        prior = k.read(root/"rehearsal-succeeded.json")
        need(prior["receipt"]==d.digest((root/"rehearsal-cpu/receipt.json").read_bytes()) and
             k.read(root/"rehearsal-cpu/receipt.json")["success"] and
             prior["result"]==d.digest((root/"rehearsal/result.json").read_bytes()),"rehearsal did not pass")
    whole = root
    root,parent = root/kind,parent/kind
    inp,sp = x.data(root,pin)
    installed = k.read(whole/"installed.json")
    need(sp["kind"]==kind and installed["extension"]==extension,"split/native identity")
    # This exclusive card is both the input provenance and the one-shot latch.
    card = d.save(root/"run-card.json",dict(inputs=pin,installed=installed,kind=kind,retry=False,
                    sources=inp["sources"],seconds=300 if kind=="formal" else 120))
    st = v.sig.native(extension)
    out = root/"training"
    out.mkdir(mode=0o700)
    (out/"weights").mkdir()
    capture(st,parent,sp,out)
    wfiles = k.identities(out)
    score(st,parent,sp,sp["train"],out,wfiles,out/"scores.jsonl")
    trained = d.save(root/"trained.json",dict(inputs=pin,card=card,files=k.identities(out)))
    checked = d.save(root/"training-checked.json",audit.training(root,parent,inp,sp,pin,trained))
    print("training sealed and audited",trained,checked,flush=True)
    cases = x.read(root/"transfer.json",inp["transfer"])
    need(cases==x.rows(kind,True),"unfrozen transfer cases")
    ev = root/"evaluation"
    ev.mkdir(mode=0o700)
    d.save(ev/"consumed.json",dict(trained=trained,checked=checked,transfer=inp["transfer"],retry=False))
    weights = x.read(root/"trained.json",trained)["files"]
    score(st,parent,sp,cases,out,weights,ev/"scores.jsonl")
    scored = d.save(root/"scored.json",dict(trained=trained,checked=checked,inputs=pin,files=k.identities(ev)))
    result = audit.result(root,parent,pin,trained,checked,scored)
    print("result",d.save(root/"result.json",result),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for field in ("root","parent"): ap.add_argument(field,type=Path)
    ap.add_argument("kind",choices=x.BANK);ap.add_argument("pin");ap.add_argument("extension")
    campaign(**vars(ap.parse_args()))
