"""Replay signed-lane gradients, sealed native updates and all decisions without importing an ML runtime."""
import argparse
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

m=SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("metrics.py"))))
s,x=m.s,m.x
ck,c,k,v,d,need=s.ck,s.c,s.k,s.v,s.d,s.need


def bank(sp,rows,path,owners,units=None):
    journal=s.legacy.records(path)
    out={}
    for b in sp["bases"]:
        for name in [x.name(b)]+[r[0] for r in s.trajectories(sp,b)]:
            by_key=out[name]={}
            for case in rows:
                r=next(journal); key=case["key"]
                need((r["name"],r["key"],r["owner"])==(name,key,owners[name]) and key not in by_key,"logit order or owner")
                z=r["logits"];v.u.bits(z);by_key[key]=z
                if name==x.name(b) and units is not None:
                    trace=units[b][key]["contrast"]
                    need((z[7],z[8],max(range(11),key=z.__getitem__))==
                         (trace["target_logit"],trace["alternative_logit"],trace["greedy_id"]),"origin full logits vs trace")
    need(next(journal,None) is None,"extra full-logit rows")
    return out


def training(folder,parent,inp,sp,pin,trained):
    rec=x.read(folder/"trained.json",trained)
    out=folder/"training"
    files=k.identities(out)
    expected={"steps.jsonl","scores.jsonl","physical.jsonl"}
    for b in sp["bases"]:
        expected.update(("unit-%d.json"%b,"metric-%d.json"%b))
        expected.update("weights/"+n+".json" for n in [x.name(b)]+[z[0] for z in s.trajectories(sp,b)])
    need(set(files)==expected and rec["inputs"]==pin and rec["files"]==files,"unsealed training files")
    card=x.read(folder/"run-card.json",rec["card"])
    need(card==dict(inputs=pin,sources=inp["sources"],installed=k.read(folder.parent/"installed.json"),
             retry=False,limit=300 if sp["kind"]=="formal" else 120),"training owner or worker identity")
    hosts=s.old.hosts(parent,sp)
    steps=s.legacy.records(out/"steps.jsonl");witnesses=s.legacy.records(out/"physical.jsonl")
    owners={};units={};selection={};nstep=nwit=0;max_rms=0.
    for b in sp["bases"]:
        h=hosts[str(b)]
        origin=x.name(b)
        init=(out/"weights"/(origin+".json")).read_bytes()
        oid,init_vec=c.state(init,h,"identity")
        need(init_vec==[0.]*6,"wrong initial coordinates")
        owners[origin]=oid
        signals=units[b]=k.read(out/("unit-%d.json"%b))
        need(set(signals)=={r["key"] for r in sp["train"]},"missing origin observations")
        examples=[signals[r["key"]] for r in sp["train"]]
        for case,tr in zip(sp["train"],examples):ck.unit(tr,case,h,oid)
        metric=k.read(out/("metric-%d.json"%b))
        q=ck.metric(metric,examples,h,oid)
        for n,arm,t,o in s.trajectories(sp,b):
            pre,before=oid,init_vec
            pairs=x.pairs(sp["train"],t,o)*sp["epochs"]
            counts={"first":0,"second":0,"tie":0}
            for i,pair in enumerate(pairs):
                r=next(steps)
                need((r["name"],r["i"])==(n,i),"wrong step order")
                s.validate_gradient(r,pair,arm,[signals[p["key"]] for p in pair],h,pre,i==0)
                if arm=="weakest_lane":
                    act=r["objective"]["weakest_lane"]["active"]
                    counts["tie" if all(act) else "first" if act[0] else "second"]+=1
                ck.step(r,before,pre,q,metric["id"])
                max_rms=max(max_rms,abs(r["step"]["realized_rms"]/v.TARGET-1))
                need(c.state(r["state"].encode(),h,"identity")==(r["id"],r["values"]),"actual state digest or parameters")
                if i in (0,len(pairs)-1):
                    w=next(witnesses)
                    need((w["name"],w["i"])==(n,i),"misplaced physical observation")
                    ck.witness(w,before,r["values"],examples,r["step"])
                    nwit+=1
                pre,before=r["id"],r["values"];nstep+=1
            need((out/"weights"/(n+".json")).read_bytes()==r["state"].encode(),"endpoint drift")
            owners[n]=pre
            if arm=="weakest_lane":selection[n]=counts
    need(next(steps,None) is None and next(witnesses,None) is None,"extra update evidence")
    z=bank(sp,sp["train"],out/"scores.jsonl",owners,units)
    calls=sum(map(len,z.values()));captures=sum(map(len,units.values()))
    need((nstep,nwit,captures,calls)==((3456,108,192,3648) if sp["kind"]=="formal" else (24,12,8,56)),
         "wrong update, witness or training-forward budget")
    need(3*sum(sum(r.values()) for r in selection.values())==nstep,"missing weakest lane selection")
    return dict(trained=trained,owners=owners,selection=selection,updates=nstep,differentiated_lanes=2*nstep,
                physical_checks=nwit,max_rms_relative_error=max_rms,train_forwards=calls,
                unit_captures=captures,calibration_prefixes=captures)


def result(folder,parent,pin,trained,checked,scored):
    inp,sp=s.data(folder,pin)
    proof=x.read(folder/"training-checked.json",checked)
    need(proof==training(folder,parent,inp,sp,pin,trained),"training archive failed replay")
    ev=folder/"evaluation";sm=x.read(folder/"scored.json",scored)
    need(sm==dict(inputs=pin,trained=trained,checked=checked,files=k.identities(ev)) and
         set(sm["files"])=={"consumed.json","scores.jsonl"} and k.read(ev/"consumed.json")==
         dict(inputs=pin,trained=trained,checked=checked,transfer=inp["transfer"],retry=False),"transfer seal")
    cases=x.read(folder/"transfer.json",inp["transfer"])
    need(cases==s.cases(sp["kind"],True),"wrong transfer split")
    test=bank(sp,cases,ev/"scores.jsonl",proof["owners"])
    train=bank(sp,sp["train"],folder/"training/scores.jsonl",proof["owners"])
    n=sum(map(len,test.values()))
    need(n==(1824 if sp["kind"]=="formal" else 56),"wrong transfer budget")
    return dict(kind=sp["kind"],inputs=pin,trained=trained,checked=checked,scored=scored,
                execution={k:v for k,v in proof.items() if k not in ("trained","owners","selection")},
                selection=proof["selection"],transfer_forwards=n,
                train=m.analyze(sp,sp["train"],train),transfer=m.analyze(sp,cases,test))


def verify(folder,parent,pin):
    sha=lambda f:d.digest((folder/f).read_bytes())
    report=result(folder,parent,pin,sha("trained.json"),sha("training-checked.json"),sha("scored.json"))
    need(d.encoded(report)==(folder/"result.json").read_bytes(),"result bytes changed")
    return report


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder",type=Path);ap.add_argument("parent",type=Path);ap.add_argument("pin")
    ap.add_argument("--admit-rehearsal",action="store_true")
    ap.add_argument("--table",type=Path)
    args=ap.parse_args()
    r=verify(args.folder,args.parent,args.pin)
    sha=d.digest(d.encoded(r))
    if args.admit_rehearsal:
        base=args.folder.parent
        raw=(base/"rehearsal-cpu/receipt.json").read_bytes();rc=json.loads(raw)
        need(r["kind"]=="rehearsal" and rc["success"] and not rc["retry"] and rc["exit_code"]==0
             and 0<rc["elapsed"]<=rc["seconds_limit"]==120,"cannot admit formal worker")
        d.save(base/"admitted.json",dict(receipt=d.digest(raw),result=sha))
    if args.table:
        fields=("base","order","teacher","template","arm","n","pairs","both_before","both",
                "mean_min_before","mean_min","weak_ties_before","weak_ties","signed_margin","mean_half_gap","abs_center_drift")
        with args.table.open("x",encoding="ascii",newline="") as table:
            print("split",*fields,"both_up","both_down","raw_before","raw_after","raw_up","raw_down","raw_ties",
                  "rank_before","rank_after","rank_up","rank_down","rank_ties",sep="\t",file=table)
            for split in ("train","transfer"):
                for row in r[split]["cells"]:
                    print(split,*(row[f] for f in fields),len(row["gained_both"]),len(row["lost_both"]),
                          *(row[k][f] for k in ("raw","rank") for f in ("before","correct","up","down","ties")),
                          sep="\t",file=table)
    print(json.dumps(dict(result=sha,counts=r["execution"],transfer=r["transfer"]["base_gates"]),sort_keys=True))
