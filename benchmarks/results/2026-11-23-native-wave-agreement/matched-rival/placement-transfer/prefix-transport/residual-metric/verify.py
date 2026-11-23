"""Rebuild counts, direction checks and teacher effects from immutable records; no model loads."""
import argparse
import csv
import json
import math
from itertools import product
from pathlib import Path
import runpy
from types import SimpleNamespace

v = SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("study.py"))))
s,k,d,a,p,need = v.s,v.k,v.d,v.p.a,v.p,v.need


def expected_outputs(examples, gate):
    out = []
    for rec in examples:
        n,w = rec["shape"]
        mean = [d.f32(math.fsum(rec["source_features"][j::w])) for j in range(w)]
        out.append([d.f32(x+d.f32(rec["strength"]*d.f32(g*m)))
                    for x,g,m in zip(rec["site_input"][-w:],gate,mean)])
    return out


def read(root,parent,signal):
    raw=(root/"spec.json").read_bytes()
    sp=json.loads(raw)
    need(raw==d.encoded(v.plan(sp["kind"],parent,signal,sp["extension"])), "plan changed")
    complete=(root/"complete.json").read_bytes()
    out=root/"attempt"
    names=set()
    starts=set()
    for b,t in product(sp["bases"],v.TURNS):
        starts.add(s.key(b,t,"gate"))
        for rule,teacher,order in product(v.RULES,d.TEACHERS,sp["orders"]):
            names.add(s.key(b,t,rule,teacher,order))
    files={"steps.jsonl","physical.jsonl"} | {"metric-"+n+".json" for n in starts}
    files |= {part+"/"+n+".json" for part in ("weights","logits") for n in names|starts}
    ids=k.identities(out)
    need(set(ids)==files and json.loads(complete)==dict(spec=d.digest(raw),files=ids), "manifest or coverage")
    hosts=k.read(parent/"attempt/hosts.json")
    anchors=p.anchor_rows(parent)
    bank=v.traces(signal,sp)
    counts=dict(calibration_prefixes=0,updates=0,training_vjps=0,snapshot_forwards=0,native_residual_rows=0,
                matched_origin_member_vjps=0,origins=len(starts),endpoints=len(names))
    high=dict(relative_rms_error=0.,native_rms_abs_error=0.,parameter_l2=0.)
    zs,metrics={},{}
    keys={r["key"] for r in sp["train"]+sp["test"]}
    def snapshot(name,b,t):
        pin,vec=p.state((out/"weights"/(name+".json")).read_bytes(),hosts[str(b)],t,"gate")
        f=k.read(out/"logits"/(name+".json"))
        need(f["owner"]==pin and f["rows"].keys()==keys and all(len(z)==11 and v.u.bits(z) for z in f["rows"].values()),"snapshot")
        zs[name]=f["rows"]
        counts["snapshot_forwards"]+=len(keys)
        return pin,vec
    with (out/"steps.jsonl").open("rb") as journal,(out/"physical.jsonl").open("rb") as physical:
        for b,t in product(sp["bases"],v.TURNS):
            start=s.key(b,t,"gate")
            orig,initial=snapshot(start,b,t)
            need(initial==[0.]*6,"nonzero origin")
            v.old.same_logits(zs[start],k.read(parent/"attempt"/("frozen-%d.json"%b)))
            examples=[bank[b,t,r["key"]] for r in sp["train"]]
            q=v.moments(examples)
            metric=k.read(out/("metric-"+start+".json"))
            need(metric["samples"]==len(examples) and metric["anchor"]==orig and
                 len(metric["second_moments"])==3 and all(math.isclose(x,y,rel_tol=1e-14) for x,y in zip(q,metric["second_moments"])),"independent calibration")
            counts["calibration_prefixes"]+=len(examples)
            metrics[start]=q
            for rule,teacher,order in product(v.RULES,d.TEACHERS,sp["orders"]):
                name=s.key(b,t,rule,teacher,order)
                ident,vec=orig,initial
                pairs=k.schedule(sp["train"],order,"pair")*sp["epochs"]
                for i,pair in enumerate(pairs):
                    r=json.loads(next(journal))
                    need((r["cell"],r["i"])==(name,i),"trajectory order")
                    p.check_members(r,pair,teacher,zs[start],anchors[s.key(b,t,teacher,order)])
                    v.validate_step(r,vec,ident,rule,q,metric["id"])
                    rr=r["report"]
                    counts["updates"]+=1
                    counts["training_vjps"]+=len(pair)
                    if i==0: counts["matched_origin_member_vjps"]+=len(pair)
                    high["relative_rms_error"]=max(high["relative_rms_error"],abs(rr["realized_rms"]/v.TARGET-1))
                    high["parameter_l2"]=max(high["parameter_l2"],rr["realized_parameter_l2"])
                    if i in (0,len(pairs)-1):
                        f=json.loads(next(physical))
                        need((f["cell"],f["i"])==(name,i) and f["before"]==expected_outputs(examples,vec[3:])
                             and f["after"]==expected_outputs(examples,r["values"][3:]),"native residual rows vs independent f32 calculation")
                        err=abs(v.rms_rows(f["before"],f["after"])-rr["realized_rms"])
                        need(err<=2e-6,"actual residual RMS")
                        high["native_rms_abs_error"]=max(high["native_rms_abs_error"],err)
                        counts["native_residual_rows"]+=2*len(examples)
                    ident,vec=r["id"],r["values"]
                need(snapshot(name,b,t)==(ident,vec),"trajectory does not reach saved weights")
        need(not journal.read(1) and not physical.read(1),"extra observations")
    return sp,zs,counts,high,metrics,d.digest(complete)


def analyze(sp,zs):
    table=[]
    decisions={}
    for b in sp["bases"]:
        for split in ("train","test"):
            for order in sp["orders"]:
                effects={}
                for t,mode in product(v.TURNS,v.RULES):
                    rows=sp[split]
                    origin=a.metrics(rows,zs[s.key(b,t,"gate")])
                    head,other=[a.metrics(rows,zs[s.key(b,t,mode,teacher,order)]) for teacher in d.TEACHERS]
                    e=a.effect(origin,head,other)
                    # Wins for the counterteacher are measured against its own labels.
                    rel=[dict(r,head=r["attractor"]) for r in rows]
                    counter=a.metrics(rel,zs[s.key(b,t,mode,"attractor",order)])
                    counter0=a.metrics(rel,zs[s.key(b,t,"gate")])
                    effects[t,mode]=dict(base=b,split=split,order=order,turn=t,rule=mode,
                                         attractor_wins=counter["wins"],attractor_before=counter0["wins"],
                                         attractor_switches=a.effect(counter0,counter,counter)["switches"],**e)
                for (t,mode),r in effects.items():
                    for key,ctrl in (("uniform",effects[t,"uniform"]),("cyclic",effects[t,"cyclic"]),
                                     ("identity",effects["identity",mode])):
                        for metric in ("h","S","wins"):
                            r[key+"_"+metric]=r[metric]-ctrl[metric]
                    table.append(r)
        for t in v.TURNS:
            sub=[r for r in table if (r["base"],r["split"],r["turn"],r["rule"])==(b,"test",t,"diagonal")]
            need(len(sub)==len(sp["orders"]),"missing evaluation order")
            margins=all(min(r[c] for c in ("h","S","uniform_h","uniform_S","cyclic_h","cyclic_S"))>.001
                        and r["positive_pairs"]>=3 for r in sub)
            choices=margins and all(r["wins"]>r["wins_before"] and min(r["uniform_wins"],r["cyclic_wins"])>0 for r in sub)
            decisions[s.key(b,t)]=dict(margin=margins,decision=choices)
    return table,decisions


def main(root,parent,signal,result,table):
    sp,z,c,h,metrics,manifest=read(root,parent,signal)
    rows,gates=analyze(sp,z)
    with table.open("x",newline="",encoding="ascii") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter="\t",lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print(d.save(result,dict(schema="spiraltorch.residual_metric_results.v1",spec=d.digest((root/"spec.json").read_bytes()),
                 manifest=manifest,counts=c,worst=h,moments=metrics,by_base=gates,rows=rows)))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("root","parent","signal","result","table"): cli.add_argument(name,type=Path)
    args=cli.parse_args()
    main(**vars(args))
