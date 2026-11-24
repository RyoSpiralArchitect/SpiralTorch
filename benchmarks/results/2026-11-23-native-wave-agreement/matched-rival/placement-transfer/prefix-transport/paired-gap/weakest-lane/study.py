"""Frozen weakest-lane experiment inputs. Import numerical checks, never a model."""
from itertools import product
import json
import math
from pathlib import Path
import runpy
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
legacy=SimpleNamespace(**runpy.run_path(str(HERE.parent/"audit.py")))
ck,x=legacy.check,legacy.x
old,c,v,k,d,need=x.old,x.c,x.v,x.k,x.d,x.need
ARMS=("binary","rank","weakest_lane")
PREV={"formal":"d97ad574b956fce4dddad67c1d535f4e9409b83592862501e15e5538615896bb","rehearsal":"7d66c915c6a203d8e5ed34da019839bdb2392197d5b71cc652b3c94ddc5db44d"}


def cases(kind,test):
    out=[]
    for r in x.rows(kind,test):
        ids=r["prompt"]
        need(ids[0]==0 and ids[-1]==10 and 0 not in ids[1:],"unexpected parent template")
        template={"head_first_modifier":"prehead","nearby_first_modifier":"prenear"}[r["template"]]
        out.append(dict(r,key=template+"."+r["key"].split(".",1)[1],template=template,
                        prompt=ids[1:-1]+[0,10]))
    return out


def sources():
    repo=x.REPO
    paths=[HERE/p for p in ("PROTOCOL.md","study.py","run.py","audit.py","metrics.py")]
    paths += [HERE.parent/p for p in ("design.py","checks.py","audit.py")]
    paths += [repo/p for p in x.sources()]
    paths += [repo/"bindings/st-py/spiraltorch/__init__.pyi",repo/"bindings/st-py/tests/test_native_weakest_lane.py"]
    return {p.relative_to(repo).as_posix():d.digest(p.read_bytes()) for p in paths}


def freeze(root,prev,bank,selector,residual):
    historical=set(); specs={}
    for kind in PREV:
        inp=x.read(prev/kind/"inputs.json",PREV[kind])
        sp=x.read(prev/kind/"training.json",inp["training"])
        a,banksp=c.data(bank/kind,x.BANK[kind])
        sel=x.read(selector/kind/"inputs.json",x.SELECT[kind])
        six=x.read(residual/kind/"spec.json",c.PREVIOUS[kind][0])
        need(sp["train"]==x.rows(kind) and banksp["train"]==six["train"] and
             (inp["bank"],inp["selector"],sel["bank"])==(x.BANK[kind],x.SELECT[kind],x.BANK[kind]),"foreign parents")
        inputs=(sp["train"],x.read(prev/kind/"transfer.json",inp["transfer"]),six["train"],six["test"],
                x.read(bank/kind/"evaluation.json",a["evaluation"]),x.read(selector/kind/"transfer.json",sel["transfer"]))
        for batch in inputs:
            historical.update(tuple(r["prompt"]) for r in batch)
        specs[kind]={f:sp[f] for f in ("bases","orders","epochs","weights","parent_manifest")}
    seen=set(); pins={}
    for kind in PREV:
        sp=dict(specs[kind],kind=kind,train=cases(kind,False))
        tr=cases(kind,True)
        formal=kind=="formal"
        need((sp["bases"],sp["orders"],sp["epochs"])==(([557,563,569],[17,31,73],2) if formal else ([571],[5],1)),
             "unfrozen workload")
        for group,n in ((sp["train"],64 if formal else 8),(tr,32 if formal else 8)):
            ids={tuple(r["prompt"]) for r in group}
            need(len(ids)==len(group)==n and not ids.intersection(historical|seen),"duplicate new prompts")
            for t in x.TEACHERS:x.pairs(group,t)
            seen.update(ids)
        if formal:
            bare=lambda rows:{(r["frame"][0],r["frame"][2]) for r in rows}
            need(not bare(sp["train"])&bare(tr),"lexical frame leak")
        out=root/kind
        out.mkdir(mode=0o700)
        pins[kind]=d.save(out/"inputs.json",dict(kind=kind,previous=PREV[kind],
             training=d.save(out/"training.json",sp),transfer=d.save(out/"transfer.json",tr),sources=sources()))
    return pins


def data(folder,pin):
    inp=x.read(folder/"inputs.json",pin)
    sp=x.read(folder/"training.json",inp["training"])
    need(inp["kind"]==sp["kind"] and inp["previous"]==PREV[sp["kind"]] and inp["sources"]==sources()
         and sp["train"]==cases(sp["kind"],False),"input drift")
    return inp,sp


def trajectories(sp,b):
    for arm,t,o in product(ARMS,x.TEACHERS,sp["orders"]):
        yield x.name(b,arm,t,o),arm,t,o


def validate_gradient(r,pair,arm,traces,host,ident,first):
    report,g=r["objective"],r["g"]
    if arm!="weakest_lane":
        need("weakest_lane" not in report,"changed old objective report")
        ck.gradient(r,pair,arm,traces,host,ident,first)
        return
    need(report["schema"]=="spiraltorch.paired_token_gap.v1" and report["objective"]=={"kind":arm}
         and report["tokens"]==[7,8] and g["geometry"]==ident and g["base"]==host["base"]
         and g["tokens"]==2 and g["shape"]==[2,9] and r["keys"]==[p["key"] for p in pair],
         "wrong paired objective identity")
    z=report["logits"]
    need(len(z)==2 and all(len(row)==2 and all(math.isfinite(y) and d.f32(y)==y for y in row) for row in z),
         "invalid raw decision logits")
    (a,b),(u,v0)=z
    signed=(a-b,v0-u)
    diff=math.fsum((a,-b,u,-v0))
    active=[diff<=0,diff>=0]
    weak=signed[0 if active[0] else 1]
    tail=math.exp(-abs(weak))
    loss=max(-weak,0.)+math.log1p(tail)
    slope=(tail if weak>=0 else 1.)/(1+tail)
    der=[-slope*active[0]/sum(active),slope*active[1]/sum(active)]
    ctl=ck.reference(z,"binary",0.)
    need(report["weakest_lane"]==dict(active=active,signed_margin=weak) and ck.numeric(report["loss"],loss)
         and g["loss"]==d.f32(loss) and len(report["row_derivatives"])==2
         and all(ck.numeric(w,y) for w,y in zip(der,report["row_derivatives"])),
         "wrong active lane, loss or subgradient")
    need(all(report[f]==ctl[f] if f in ("center_drift","gaps") else ck.numeric(report[f],ctl[f])
             for f in ("half_difference","center","gaps","binary_loss","rank_loss","center_drift","center_penalty")),
         "common score fields")
    need(set(g["params"])=={"bias","gate"} and all(len(q)==3 and all(math.isfinite(a) and d.f32(a)==a for a in q)
         for q in g["params"].values()),"missing or nonfinite gate derivatives")
    if first:
        for row,tr in zip(z,traces):
            need(row==[tr["contrast"][q] for q in ("target_logit","alternative_logit")],"unit-vs-packed logits")
        for block,vec in g["params"].items():
            for j,actual in enumerate(vec):
                expected=math.fsum(w*tr[block+"_gradient"][j] for w,tr in zip(der,traces))
                need(abs(actual-expected)<=2e-7+3e-4*abs(expected),"weakest lane vs independent unit VJP")


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("root","prev","bank","selector","residual"):p.add_argument(n,type=Path)
    print(json.dumps(freeze(**vars(p.parse_args())),sort_keys=True))
