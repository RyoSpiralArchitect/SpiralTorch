"""Recompute the selector, decisions and transfer gates from immutable logits; never load weights into a model."""
import argparse
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import runpy
import struct
from types import SimpleNamespace

u = SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("design.py"))))
s,d,need = u.s,u.d,u.need
POLICIES = ("selected","blind") + tuple(u.MAPS)
LABELS = ("head","attractor")

def journal(path,bank,manifest,sp,rows,signals=None):
    z = {}
    with path.open("rb") as f:
        for b,m,name in u.cells(sp):
            owner=d.identity(u.parent_artifact(bank,manifest,name))
            values={}
            for sample in rows:
                r=json.loads(next(f))
                key=sample["key"]
                need((r["key"],r["owner"],r["name"])==(key,owner,name),"row, model or artifact mismatch")
                u.c.v.u.bits(r["logits"])
                values[key]=r["logits"]
            z[name]=values
            if name==s.key(b,m):
                origin=z[s.key(b,u.MAPS[0])]
                for key,v in values.items():
                    need(u.c.v.u.bits(v)==u.c.v.u.bits(origin[key]),"zero-route logits differ")
                    if signals is not None:
                        ref=signals[b,key]["trace"]["contrast"]
                        need((ref["target_id"],ref["alternative_id"])==(7,8) and
                             struct.pack("<2f",*v[7:9])==struct.pack("<2f",ref["target_logit"],ref["alternative_logit"]),
                             "archived training origin differs")
        need(not f.read(1),"extra scored row")
    return z

def packets(sp,z,b,t,o):
    names=[s.key(b,u.MAPS[0])]+[s.key(b,m,t,o) for m in u.MAPS]
    return [[tuple(z[n][k][7:9] for k in keys) for n in names] for keys in u.quartets(sp["train"])]

def reference(groups,t):
    # Independent exact arithmetic on the saved f32 values, not the Rust accumulation order.
    sign = 1 if t=="head" else -1
    total = Fraction(0)
    gains = [[],[],[]]
    for frame in groups:
        margins=[[Fraction(a)-Fraction(b) for a,b in q] for q in frame]
        total+=sum(margins[0])
        for i,vs in enumerate(margins[1:]):
            gains[i].append(sign*(vs[1]-vs[2]-margins[0][1]+margins[0][2])/2)
    n=len(groups)
    mins=[min(g) for g in gains]; means=[sum(g)/n for g in gains]
    best=max(range(3),key=lambda i:(mins[i],means[i],-i))
    return dict(frames=n,selected=best,threshold=float(total/(4*n)),
                worst=list(map(float,mins)),mean=list(map(float,means)))

def choices(sp,z,select):
    result={}
    for b in sp["bases"]:
        for i,t in enumerate(LABELS):
            for o in sp["orders"]:
                rep=select(packets(sp,z,b,t,o),t)
                result[s.key(b,t,o)]=dict(rep,blind=(b+o+i)%3)
    return result

def review_training(root,bank,signal,inputs):
    cfg,sp,m=u.load_inputs(root,bank,inputs)
    need(json.loads((root/"started.json").read_bytes())==dict(inputs=inputs,selector=cfg["selector"],
         extension=cfg["extension"],retry=False),"attempt or runtime identity differs")
    rec=u.read(root/"calibration.json",u.sha(root/"calibration.json"))
    need(rec["inputs"]==inputs and rec["scores"]==u.sha(root/"training-logits.jsonl"),"unsealed training logits")
    z=journal(root/"training-logits.jsonl",bank,m,sp,sp["train"],u.c.raw_signals(signal,sp))
    exact=choices(sp,z,reference)
    need(exact==rec["choices"],"Rust calibration disagrees with independent exact readback")
    count=sum(map(len,z.values()))
    need(count==rec["forwards"],"training score count")
    for b in sp["bases"]:
        need(len({exact[s.key(b,t,o)]["threshold"] for t in LABELS for o in sp["orders"]})==1,"inconsistent prior")
    return dict(calibration=u.sha(root/"calibration.json"),forwards=count,choices=len(exact),
                checked_verb_pairs=len(sp["bases"])*3*len(sp["train"]))

def delta(z,k):
    v=z[k]
    return v[7]-v[8]

def role_stats(rows,origin,endpoint,teacher,threshold):
    other=LABELS[1-LABELS.index(teacher)]
    keys=u.quartets(rows)
    margin=[];select=[]
    stats={}
    for label,tau in (("raw",0.),("centered",threshold)):
        before=after=up=down=ties0=ties=0
        for r in rows:
            k=r["key"];sign=1-2*r[teacher]
            p,q=sign*(delta(origin,k)-tau),sign*(delta(endpoint,k)-tau)
            before+=p>0; after+=q>0; up+=p<=0<q; down+=p>0>=q;ties0+=p==0;ties+=q==0
            if label=="raw":
                step=delta(endpoint,k)-delta(origin,k)
                margin.append(sign*step)
                select.append((sign-(1-2*r[other]))*step)
        need(after-before==up-down,"decision accounting")
        stats.update({label+"_"+k:v for k,v in dict(before=before,after=after,up=up,down=down,
                                                      ties_before=ties0,ties=ties).items()})
    pairs=((0,2),(1,3)) if teacher=="head" else ((0,1),(2,3))
    old=[];new=[]
    for f in keys:
        for i,j in pairs:
            old.append(delta(origin,f[i])-delta(origin,f[j]))
            new.append(delta(endpoint,f[i])-delta(endpoint,f[j]))
    stats.update(rank_before=sum(n>0 for n in old),rank_after=sum(n>0 for n in new),
                 rank_up=sum(a<=0<b for a,b in zip(old,new)),rank_down=sum(a>0>=b for a,b in zip(old,new)),
                 rank_ties_before=sum(n==0 for n in old),rank_ties=sum(n==0 for n in new))
    return math.fsum(margin)/len(rows),math.fsum(select)/len(rows),stats

def analyze(sp,rows,z,chosen):
    table=[];gates={}
    for b in sp["bases"]:
        origin=z[s.key(b,u.MAPS[0])]
        cells=[]
        for o in sp["orders"]:
            for form in ("hh","aa"):
                test=[r for r in rows if r["template"]==form]
                by_policy={}
                for policy in POLICIES:
                    result=dict(base=b,order=o,template=form,policy=policy,questions=len(test))
                    cs=[]
                    for t in LABELS:
                        choice=chosen[s.key(b,t,o)]
                        idx=(choice[policy] if policy in ("selected","blind") else u.MAPS.index(policy))
                        model=z[s.key(b,u.MAPS[idx],t,o)]
                        h,c,stats=role_stats(test,origin,model,t,choice["threshold"])
                        result[t+"_route"]=u.MAPS[idx]
                        result[t+"_gain"]=h;result[t+"_selectivity"]=c
                        result.update({t+"_"+k:v for k,v in stats.items()})
                        cs.append(c)
                    result.update(h=result["head_gain"], S=sum(cs)/2)
                    table.append(result);by_policy[policy]=result
                sel=by_policy["selected"]
                good=min(sel["h"],sel["S"])>0.001 and all(sel[k]-by_policy[p][k]>0.001
                       for k in ("h","S") for p in ("identity","blind"))
                raw=sel["head_raw_after"]>max(sel["head_raw_before"],by_policy["identity"]["head_raw_after"],
                                               by_policy["blind"]["head_raw_after"])
                cells.append((good,raw))
        gates[str(b)]=dict(margin=all(g for g,_ in cells),decision=all(g and r for g,r in cells))
    macro={p:{k:math.fsum(x[k] for x in table if x["policy"]==p)/(len(table)//len(POLICIES))
              for k in ("h","S")} for p in POLICIES}
    dominates=all(macro["selected"][k]-macro[p][k]>.001 for k in ("h","S") for p in POLICIES[1:])
    return dict(rows=table,by_base=gates,macro=macro,dominates=dominates,
                margin_transfer=dominates and sum(x["margin"] for x in gates.values())>=2,
                decision_transfer=dominates and sum(x["decision"] for x in gates.values())>=2)

def audit(root,bank,signal,inputs,output,table):
    cfg,sp,m=u.load_inputs(root,bank,inputs)
    training=review_training(root,bank,signal,inputs)
    checked=u.sha(root/"checked.json")
    need(u.read(root/"checked.json",checked)==training,"training seal differs")
    consumption={"inputs":inputs,"checked":checked,"transfer":cfg["transfer"],"retry":False}
    need(u.read(root/"consumed.json",u.sha(root/"consumed.json"))==consumption,"no verified transfer seal")
    complete=u.read(root/"complete.json",u.sha(root/"complete.json"))
    need(complete["checked"]==checked and complete["consumed"]==u.sha(root/"consumed.json") and
         complete["scores"]==u.sha(root/"transfer-logits.jsonl"),"transfer logits not sealed")
    rows=u.read(root/"transfer.json",cfg["transfer"])
    need(rows==u.fresh(cfg["kind"]),"unexpected transfer prompts")
    z=journal(root/"transfer-logits.jsonl",bank,m,sp,rows)
    count=sum(map(len,z.values()))
    need(count==complete["forwards"],"transfer score count")
    chosen=u.read(root/"calibration.json",training["calibration"])["choices"]
    result=analyze(sp,rows,z,chosen)
    with table.open("x",encoding="ascii",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(result["rows"][0]),delimiter="\t",lineterminator="\n")
        w.writeheader();w.writerows(result["rows"])
    return d.save(output,dict(result,training=training,transfer_forwards=count,inputs=inputs,complete=u.sha(root/"complete.json"),
                              chosen=chosen,schema="spiraltorch.train_only_route_transfer.v1"))

if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for n in ("root","bank","signal"): cli.add_argument(n,type=Path)
    cli.add_argument("inputs")
    for n in ("output","table"): cli.add_argument(n,type=Path)
    print(audit(**vars(cli.parse_args())))
