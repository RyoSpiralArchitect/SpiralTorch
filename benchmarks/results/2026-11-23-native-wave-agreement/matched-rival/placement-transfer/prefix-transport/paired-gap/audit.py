"""Replay every saved update, separate absolute decisions from within-pair ordering, no model calls."""
import argparse
import json
import math
from pathlib import Path
import runpy
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
check = SimpleNamespace(**runpy.run_path(str(HERE/"checks.py")))
x, need = check.x, check.need
c,d,v,k = x.c,x.d,x.v,x.k


def records(path):
    with path.open("rb") as raw:
        for line in raw:
            yield json.loads(line)


def load_scores(sp,cases,out,owners,origins=None):
    z = {}
    journal = records(out)
    for b in sp["bases"]:
        for n in [x.name(b)]+[x.name(*key) for key in x.cells(sp,b)]:
            z[n] = {}
            for row in cases:
                r = next(journal)
                key = row["key"]
                need((r["key"],r["name"],r["owner"])==(key,n,owners[n]) and key not in z[n], "logit owner, order or coverage")
                vals = r["logits"]
                v.u.bits(vals)
                z[n][key] = vals
                if origins is not None and n==x.name(b):
                    u=origins[b][key]["contrast"]
                    need((vals[7],vals[8],max(range(11),key=vals.__getitem__))==
                         (u["target_logit"],u["alternative_logit"],u["greedy_id"]), "origin unit vs full-logit score")
    need(next(journal,None) is None, "excess score rows")
    return z


def training(root,parent,inp,sp,pin,trained):
    out = root/"training"
    sealed = x.read(root/"trained.json",trained)
    expected = {"scores.jsonl","steps.jsonl","physical.jsonl"}
    names, units, checkpoints = {},{},{}
    for b in sp["bases"]:
        expected.update(("signals-%d.json"%b,"metric-%d.json"%b))
        expected.update("weights/"+n+".json" for n in [x.name(b)]+[x.name(*p) for p in x.cells(sp,b)])
    files = k.identities(out)
    need(set(files)==expected and files==sealed["files"] and sealed["inputs"]==pin, "training manifest")
    card = x.read(root/"run-card.json",sealed["card"])
    need(card["inputs"]==pin and card["sources"]==inp["sources"] and not card["retry"]
         and card["installed"]==k.read(root.parent/"installed.json"),"training run card")
    owners = x.old.hosts(parent,sp)
    steps,wits = records(out/"steps.jsonl"), records(out/"physical.jsonl")
    nsteps = nwit = 0
    max_rms_rel = 0.
    for b in sp["bases"]:
        h = owners[str(b)]
        def state(n):
            raw=(out/"weights"/(n+".json")).read_bytes()
            ident, vec=c.state(raw,h,"identity")
            return ident,vec,raw
        start = x.name(b)
        orig, initial, _ = state(start)
        need(initial==[0.]*6,"nonzero origin")
        checkpoints[start]=orig
        unit = k.read(out/("signals-%d.json"%b))
        need(set(unit)=={r["key"] for r in sp["train"]},"origin signal coverage")
        examples=[unit[r["key"]] for r in sp["train"]]
        for r,tr in zip(sp["train"], examples):
            check.unit(tr,r,h,orig)
        units[b]=unit
        metric=k.read(out/("metric-%d.json"%b))
        q=check.metric(metric,examples,h,orig)
        for _,arm,t,o in x.cells(sp,b):
            n=x.name(b,arm,t,o)
            pairs=x.pairs(sp["train"],t,o)*sp["epochs"]
            ident,vec=orig,initial
            for i,pair in enumerate(pairs):
                r=next(steps)
                need((r["name"],r["i"])==(n,i),"trajectory order")
                check.gradient(r,pair,arm,[unit[p["key"]] for p in pair],h,ident,i==0)
                check.step(r,vec,ident,q,metric["id"])
                need(c.state(r["state"].encode(),h,"identity")==(r["id"],r["values"]),"unbound intermediate weights")
                max_rms_rel=max(max_rms_rel,abs(r["step"]["realized_rms"]/v.TARGET-1))
                if i in (0,len(pairs)-1):
                    w=next(wits)
                    need((w["name"],w["i"])==(n,i),"witness order")
                    check.witness(w,vec,r["values"],examples,r["step"])
                    nwit+=1
                ident,vec=r["id"],r["values"]
                nsteps+=1
            need(state(n)==(ident,vec,r["state"].encode()),"serialized endpoint mismatch")
            checkpoints[n]=ident
    need(next(steps,None) is None and next(wits,None) is None,"extra updates or witnesses")
    z=load_scores(sp,sp["train"],out/"scores.jsonl",checkpoints,units)
    calls=sum(map(len,z.values()))
    nunit=sum(map(len,units.values()))
    need((nsteps,nwit,nunit,calls)==((3456,108,192,3648) if sp["kind"]=="formal" else (24,12,8,56)),
         "wrong training work")
    return dict(trained=trained,steps=nsteps,differentiated_lanes=2*nsteps,unit_captures=nunit,
                calibration_prefixes=nunit,physical_checks=nwit,train_forwards=calls,
                max_relative_rms_error=max_rms_rel,owners=checkpoints)


def metrics(cases, origin, endpoint, teacher):
    def margin(z,key):
        return z[key][7]-z[key][8]
    def pairs(z):
        return [((margin(z,a["key"])-margin(z,b["key"]))/2,
                 (margin(z,a["key"])+margin(z,b["key"]))/2) for a,b in x.pairs(cases,teacher)]
    ps,qs=pairs(origin),pairs(endpoint)
    raw_a=[(1-2*r[teacher])*margin(origin,r["key"]) for r in cases]
    raw_b=[(1-2*r[teacher])*margin(endpoint,r["key"]) for r in cases]
    def flips(a,b):
        return dict(before=sum(q>0 for q in a),correct=sum(q>0 for q in b),
                    up=sum(q<=0<t for q,t in zip(a,b)),down=sum(t<=0<q for q,t in zip(a,b)),
                    ties_before=a.count(0),ties=b.count(0))
    rank=flips([q[0] for q in ps],[q[0] for q in qs])
    raw=flips(raw_a,raw_b)
    return dict(n=len(cases),pairs=len(ps),rank=rank,raw=raw,
                mean_half_gap=math.fsum(q[0] for q in qs)/len(qs),
                origin_half_gap=math.fsum(q[0] for q in ps)/len(ps),
                signed_margin=math.fsum(raw_b)/len(cases),
                abs_center_drift=math.fsum(abs(q[1]-p[1]) for p,q in zip(ps,qs))/len(qs),
                up_keys=[r["key"] for r,a,b in zip(cases,raw_a,raw_b) if a<=0<b],
                down_keys=[r["key"] for r,a,b in zip(cases,raw_a,raw_b) if b<=0<a])


def analyze(sp,cases,z):
    table,by_base=[],{}
    for b in sp["bases"]:
        primary=raw_gain=drift=True
        start=z[x.name(b)]
        for order in sp["orders"]:
            for t in x.TEACHERS:
                for template in sorted({r["template"] for r in cases}):
                    rows=[r for r in cases if r["template"]==template]
                    stats={}
                    for arm in x.ARMS:
                        st=metrics(rows,start,z[x.name(b,arm,t,order)],t)
                        stats[arm]=st
                        table.append(dict(base=b,order=order,teacher=t,template=template,arm=arm,**st))
                    a,ce,rank=(stats[p] for p in ("anchored_rank","binary","rank"))
                    if t=="head":
                        primary &= (a["rank"]["correct"]>max(a["rank"]["before"],ce["rank"]["correct"]) and
                            a["mean_half_gap"]>max(a["origin_half_gap"],ce["mean_half_gap"])+.001)
                        raw_gain &= a["raw"]["correct"]>max(a["raw"]["before"],ce["raw"]["correct"])
                    drift &= (rank["abs_center_drift"]>0 and a["abs_center_drift"]<=.9*rank["abs_center_drift"]
                              and a["rank"]["correct"]>=rank["rank"]["correct"])
        by_base[str(b)]=dict(ordering=primary,ordering_and_raw=primary and raw_gain,center_control=drift)
    counts={gate:sum(v[gate] for v in by_base.values()) for gate in ("ordering","ordering_and_raw","center_control")}
    return dict(cells=table,base_gates=by_base,passing_base_counts=counts)


def result(root,parent,pin,trained,checked,scored):
    inp,sp=x.data(root,pin)
    evidence=x.read(root/"training-checked.json",checked)
    need(evidence==training(root,parent,inp,sp,pin,trained),"training replay mismatch")
    sm=x.read(root/"scored.json",scored)
    ev=root/"evaluation"
    need((sm["trained"],sm["checked"],sm["inputs"])==(trained,checked,pin) and
         sm["files"]==k.identities(ev) and set(sm["files"])=={"scores.jsonl","consumed.json"} and
         k.read(ev/"consumed.json")==dict(trained=trained,checked=checked,transfer=inp["transfer"],retry=False),
         "unsealed or foreign transfer")
    cases=x.read(root/"transfer.json",inp["transfer"])
    need(cases==x.rows(sp["kind"],True),"transfer identity")
    z=load_scores(sp,cases,ev/"scores.jsonl",evidence["owners"])
    tr=load_scores(sp,sp["train"],root/"training/scores.jsonl",evidence["owners"])
    count=sum(map(len,z.values()))
    need(count==(1824 if sp["kind"]=="formal" else 56),"wrong transfer budget")
    return dict(kind=sp["kind"],inputs=pin,trained=trained,checked=checked,scored=scored,
                execution={k:v for k,v in evidence.items() if k!="owners"},transfer_forwards=count,
                train=analyze(sp,sp["train"],tr),transfer=analyze(sp,cases,z))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ("root","parent"): ap.add_argument(n,type=Path)
    ap.add_argument("pin")
    a=ap.parse_args()
    pin=lambda n:d.digest((a.root/n).read_bytes())
    out=result(a.root,a.parent,a.pin,pin("trained.json"),pin("training-checked.json"),pin("scored.json"))
    raw=d.encoded(out)
    need(raw==(a.root/"result.json").read_bytes(),"result did not reproduce byte for byte")
    print(json.dumps(dict(sha256=d.digest(raw),train=out["train"]["passing_base_counts"],
                          transfer=out["transfer"]["passing_base_counts"]),sort_keys=True))
