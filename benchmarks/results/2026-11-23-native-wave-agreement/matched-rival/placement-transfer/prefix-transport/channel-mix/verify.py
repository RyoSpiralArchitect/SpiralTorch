"""Read-only CPU audit of training and scoring; never load, update or rescore a model."""
import argparse
import csv
import json
import math
from itertools import product
from pathlib import Path
import runpy
from statistics import median
from types import SimpleNamespace

c=SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("design.py"))))
v,k,d,s,p,need=c.v,c.k,c.d,c.s,c.p,c.need


def names(sp):
    return [(s.key(b,mode),[s.key(b,mode,t,o) for t,o in product(d.TEACHERS,sp["orders"])])
            for b,mode in product(sp["bases"],c.MAPS)]


def verify_training(root,parent,prev,signal,inputs,trained):
    info,sp=c.data(root,inputs)
    card=k.read(root/"run-card.json")
    m=json.loads(c.pinned(root/"trained.json",trained))
    need(m["inputs"]==inputs==card["inputs"] and card["extension"]==m["extension"] and card["sources"]==c.sources() and
         m["run_card"]==d.digest((root/"run-card.json").read_bytes()),"training source not pinned")
    starts={a for a,_ in names(sp)};ends={n for _,ls in names(sp) for n in ls}
    out=root/"training";ids=k.identities(out)
    expected={"steps.jsonl","physical.jsonl"} | {"metric-"+a+".json" for a in starts} | {"weights/"+n+".json" for n in starts|ends}
    need(m["files"]==ids and set(ids)==expected,"missing or extra training evidence")
    hm=json.loads(c.pinned(parent/"complete.json",sp["parent_manifest"]))["files"]["hosts.json"]
    hosts=json.loads(c.pinned(parent/"attempt/hosts.json",hm))
    bank=c.raw_signals(signal,sp)
    historical,past=c.historical(prev,sp["kind"])
    counts=dict(origins=len(starts),endpoints=len(ends),updates=0,vjps=0,origin_member_checks=0,
                bitwise_legacy_steps=0,bitwise_legacy_endpoints=0,native_residual_rows=0,calibration_prefixes=0)
    high=dict(relative_rms_error=0.,physical_rms_error=0.,max_parameter_l2=0.)
    with (out/"steps.jsonl").open("rb") as log,(out/"physical.jsonl").open("rb") as physical:
        for b in sp["bases"]:
            host=hosts[str(b)]
            xs=[bank[b,r["key"]]["trace"] for r in sp["train"]]
            q=v.moments(xs);seen_metrics=set()
            for mode in c.MAPS:
                start=s.key(b,mode)
                raw=(out/"weights"/(start+".json")).read_bytes()
                orig,initial=c.state(raw,host,mode)
                metric=k.read(out/("metric-"+start+".json"))
                need(initial==[0.]*6 and metric["id"] not in seen_metrics and metric["anchor"]==orig and
                     len(metric["second_moments"])==3 and metric["samples"]==len(xs) and
                     metric["base_model_identity"]==host["base"] and metric["frozen_lora_artifact_id"]==host["frozen_lora"] and
                     all(math.isclose(x,y,rel_tol=1e-14) for x,y in zip(q,metric["second_moments"])),"fit/owner/Q invariance")
                seen_metrics.add(metric["id"]);counts["calibration_prefixes"]+=len(xs)
                for teacher,order in product(d.TEACHERS,sp["orders"]):
                    name=s.key(b,mode,teacher,order);ident,vec=orig,initial
                    pairs=k.schedule(sp["train"],order,"pair")*sp["epochs"]
                    for i,pair in enumerate(pairs):
                        r=json.loads(next(log))
                        need((r["i"],r["cell"],r["keys"])==(i,name,[x["key"] for x in pair]) and len(r["members"])==2,"row/order drift")
                        for member,sample in zip(r["members"],pair):
                            k.check_grad(member,1.,sample[teacher])
                            if i==0:
                                c.check_origin(member,bank[b,sample["key"]],mode)
                                counts["origin_member_checks"]+=1
                        v.validate_step(r,vec,ident,"diagonal",q,metric["id"])
                        step=r["report"];counts["updates"]+=1;counts["vjps"]+=2
                        high["relative_rms_error"]=max(high["relative_rms_error"],abs(step["realized_rms"]/c.TARGET-1.))
                        high["max_parameter_l2"]=max(high["max_parameter_l2"],step["realized_parameter_l2"])
                        if mode=="identity":
                            old=past[b,teacher,order,i]
                            need(d.encoded(dict(r,cell=old["cell"]))==d.encoded(old),"historical step differs")
                            counts["bitwise_legacy_steps"]+=1
                        if i in (0,len(pairs)-1):
                            f=json.loads(next(physical))
                            need((f["cell"],f["i"])==(name,i) and c.same_rows(f["before"],c.outputs(xs,vec[3:],mode))
                                 and c.same_rows(f["after"],c.outputs(xs,r["values"][3:],mode)),"independent residual coordinates")
                            err=abs(v.rms_rows(f["before"],f["after"])-step["realized_rms"])
                            need(err<=2e-6,"delivered RMS exceeded tolerance")
                            high["physical_rms_error"]=max(high["physical_rms_error"],err)
                            counts["native_residual_rows"]+=2*len(xs)
                        ident,vec=r["id"],r["values"]
                    raw=(out/"weights"/(name+".json")).read_bytes()
                    need(c.state(raw,host,mode)==(ident,vec),"trajectory/endpoint mismatch")
                    if mode=="identity":
                        path="weights/"+s.key(b,c.TURN,"diagonal",teacher,order)+".json"
                        need(raw==c.pinned(prev/"attempt"/path,historical[path]),"legacy endpoint differs")
                        counts["bitwise_legacy_endpoints"]+=1
        need(not log.read(1) and not physical.read(1),"extra steps or witnesses")
    need(counts["updates"]==len(ends)*len(sp["train"])*sp["epochs"]//2 and counts["bitwise_legacy_steps"]*3==counts["updates"],
         "incomplete fits")
    return dict(trained=trained,counts=counts,worst=high)


def changes(rows,z0,z,teacher):
    wrong=[]; to_right=to_wrong=0
    for r in rows:
        old,new=[(1-2*r[teacher])*(x[r["key"]][7]-x[r["key"]][8]) for x in (z0,z)]
        to_right+=old<=0<new;to_wrong+=old>0>=new
        if old<0: wrong.append((new-old)/(-old))
    return dict(to_right=to_right,to_wrong=to_wrong,wrong_initial=len(wrong),
                median_gap_closed=median(wrong) if wrong else None,max_gap_closed=max(wrong) if wrong else None)


def qualifies(sub,frames):
    ok=all(min(r[k] for k in ("h","S","vs_identity_h","vs_identity_S","vs_inverse_h","vs_inverse_S"))>.001
           and r["positive_pairs"]>=frames for r in sub)
    return dict(margin=ok,decision=ok and all(r["wins"]>r["wins_before"] and min(r["vs_identity_wins"],r["vs_inverse_wins"])>0 for r in sub))


def analyze(sp,cases,zs):
    table=[];gates={}
    for b in sp["bases"]:
        for form in ("hc","ac"):
            selected=[r for r in cases if r["template"]==form]
            for order in sp["orders"]:
                arms={}
                for mode in c.MAPS:
                    before=zs[s.key(b,mode)]
                    zh,za=[zs[s.key(b,mode,t,order)] for t in d.TEACHERS]
                    orig,hd,at=[p.a.metrics(selected,x) for x in (before,zh,za)]
                    rel=[dict(r,head=r["attractor"]) for r in selected]
                    r=dict(base=b,template=form,order=order,mode=mode,**p.a.effect(orig,hd,at))
                    r.update(attractor_wins=p.a.metrics(rel,za)["wins"],attractor_before=p.a.metrics(rel,before)["wins"],
                             attractor_switches=p.a.effect(orig,at,at)["switches"])
                    for teacher,z in (("head",zh),("attractor",za)):
                        r.update({teacher+"_"+key:v for key,v in changes(selected,before,z,teacher).items()})
                    arms[mode]=r
                for r in arms.values():
                    for label,mode in (("identity",c.MAPS[0]),("inverse",c.MAPS[2])):
                        for key in ("h","S","wins"):
                            r["vs_"+label+"_"+key]=r[key]-arms[mode][key]
                    table.append(r)
        sub=[r for r in table if r["base"]==b and r["mode"]==c.MAPS[1]]
        need(len(sub)==2*len(sp["orders"]),"missing primary template or order")
        gates[str(b)]=qualifies(sub,3 if sp["kind"]=="formal" else 1)
    return table,gates


def audit(root,parent,previous,signal,inputs,output,table):
    info,sp=c.data(root,inputs); scored=k.read(root/"scored.json")
    training=verify_training(root,parent,previous,signal,inputs,scored["trained"])
    need(training==k.read(root/"training-checked.json"),"training not audited before scoring")
    out=root/"evaluation";files=k.identities(out)
    seal=k.read(out/"consumed.json")
    need(set(files)=={"scores.jsonl","consumed.json"} and scored["inputs"]==inputs and scored["files"]==files
         and seal==dict(checked=d.digest((root/"training-checked.json").read_bytes()),trained=training["trained"],
                        evaluation=info["evaluation"],retry=False),"scoring receipt")
    data=json.loads(c.pinned(root/"evaluation.json",info["evaluation"]))
    need(data==c.fresh(sp["kind"]),"unexpected evaluation split")
    z={};count=0
    with (out/"scores.jsonl").open("rb") as log:
        for start,ends in names(sp):
            for name in [start]+ends:
                owner=d.identity((root/"training/weights"/(name+".json")).read_bytes())
                rows={}
                for sample in data:
                    r=json.loads(next(log))
                    need((r["name"],r["key"],r["owner"])==(name,sample["key"],owner),"logits/owner mismatch")
                    v.u.bits(r["logits"]);rows[r["key"]]=r["logits"];count+=1
                z[name]=rows
        need(not log.read(1),"extra evaluations")
    for b in sp["bases"]:
        for mode in c.MAPS[1:]:
            v.old.same_logits(z[s.key(b,mode)],z[s.key(b,c.MAPS[0])])
    rows,gates=analyze(sp,data,z)
    with table.open("x",newline="",encoding="ascii") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter="\t",lineterminator="\n")
        w.writeheader();w.writerows(rows)
    return d.save(output,dict(schema="spiraltorch.mixed_prefix_transfer.v1",inputs=inputs,training=training,
                              scored=d.digest((root/"scored.json").read_bytes()),scores=count,by_base=gates,rows=rows))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for n in ("root","parent","previous","signal"): cli.add_argument(n,type=Path)
    cli.add_argument("inputs")
    for n in ("output","table"): cli.add_argument(n,type=Path)
    print(audit(**vars(cli.parse_args())))
