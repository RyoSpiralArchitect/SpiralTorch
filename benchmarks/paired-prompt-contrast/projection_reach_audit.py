"""Reconstruct directions, physical signed displacements and finite interaction changes without a model."""
from collections import Counter
import io
import json
import math
import os
import struct

from projection_reach import (ATOL,MAX,RADII,REL,RESOLVE,ROLES,VALUE,W,WALL,admit,base,coordinates,decompose,f32,fold_rows,task)
from alias_interference_audit import coordinates as artifact_coords
from byte_release_artifacts import PACKAGE_ROOTS
from shared_prefix_study import obj,put,read,require,sha


def mean(v):return math.fsum(v)/len(v)
def dot(a,b):return math.fsum(x*y for x,y in zip(a,b,strict=True))
def sg(row):return 1 if row["gold"]==task.IDLE else -1
def route(label):return label.replace("/gauge","/diagonal")
def central(pred_plus,pred_minus,plus,minus):
    predicted=(pred_plus-pred_minus)/2;observed=(plus-minus)/2
    return dict(predicted=predicted,central=observed,plus=plus,minus=minus,bend=(plus+minus)/2,
                agrees=abs(observed-predicted)<=ATOL+REL*abs(predicted),resolved_ascent=min(predicted,observed)>RESOLVE and plus>0.)


def decision(rows,label):
    def at(r,k,fold):return r["radii"][str(k)][fold]
    if rows[label] is None:return False
    layer,_,basis=label.split("/");references=[f"{layer}/output/{basis}"]+(["geometry"] if basis=="diagonal" else [])
    if any(rows[k] is None for k in references):return False
    q=rows[label];big=RADII[1]
    return (q["fit_norm"]>2*max(rows[k]["fit_norm"] for k in references) and
        all(at(q,big,f)["resolved_ascent"] for f in ("fit","check")) and
        all(at(q,h,f)["agrees"] for h in RADII for f in ("fit","check")) and
        at(q,big,"check")["central"]>2*max(RESOLVE,*(at(rows[k],big,"check")["central"] for k in references)) and
        sum(x>RESOLVE for x in at(q,big,"check")["by_edge"].values())>=3 and min(at(q,big,"check")["by_order"])>0.)


def audit(proto,pp,out,mp,dst):
    p=obj(proto/"plan.json",pp);m=obj(out/"complete.json",mp);inv=admit(proto,p)
    claim=dict(plan=pp,attempt=1,output=os.path.relpath(out,proto));ex=json.loads((out/"exit.json").read_bytes())
    require(all(json.loads(f.read_bytes())==claim for f in (proto/"attempt.json",out/"attempt.json")) and m["plan"]==pp and
            m["installed"]==dict(version=inv["version"],wheel_sha256=p["wheel"],native_sha256=p["native"],
              package_files=sum(k.split("/",1)[0] in PACKAGE_ROOTS for k in inv["files"]),all_original_files_match=True,python_bytecode_absent=True) and
            ex["exit"]==0 and 0<ex["elapsed"]<=WALL and read(out/"process.log",ex["log"]).splitlines()[-1:]==[mp.encode()],"unverified installation or attempt")
    folds={k:obj(proto/(k+".json"),p["folds"][k]) for k in ("fit","check")}
    for k,parity in (("fit",0),("check",1)):fold_rows(folds[k],parity)
    require(folds["fit"]+folds["check"]==[r for i in (0,1) for r in task.deck("train") if r["edge"]%2==i],"not the old training deck")
    dirs=obj(out/"directions.json",m["directions"],4<<20);require(dirs["plan"]==pp and dirs["fit"]==p["folds"]["fit"] and set(dirs["roles"])==set(ROLES),"wrong construction")
    dense=8*(W*W+W)*8;size=dense+2*W*2*4
    events=iter(io.BytesIO(read(out/"linear.jsonl",m["files"]["linear.jsonl"],4<<20)))
    binary=io.BytesIO(read(out/"pullbacks.bin",m["files"]["pullbacks.bin"],size*256))
    values={};primal={};means={};typical={};counts=Counter();replay=0
    for fold,rows in folds.items():
        for role in ROLES:
            owner=p["roles"][role];gs=[];cs=[]
            for i,row in enumerate(rows):
                t=json.loads(next(events));b=binary.read(size)
                require((t["role"],t["fold"],t["id"],t["pullback"])==(role,fold,row["id"],sha(b)) and len(b)==size,"unbound analytic row")
                c=t["primal"];require((c["target_id"],c["alternative_id"],c["prompt_tokens"])==(task.IDLE,task.ON,9) and
                  0<=c["greedy_id"]<64 and c["target_minus_alternative"]==c["target_logit"]-c["alternative_logit"] and
                  abs(t["ce"][0]-t["ce"][1]-c["target_minus_alternative"])<=base.CROSS_TOL,"false primal")
                ref=p["refs"][role.split("/")[0]].get(row["id"]) if role.endswith("/origin") else None
                if ref:require(sha(b[dense:])==ref,"old training CE mismatch");replay+=1
                gs.append(decompose(b));cs.append(c);counts.update(factual=1,dense_vjps=8,geometry_ce=2,diagonal_checks=8*(i==0 and fold=="fit"))
            values[role,fold]=gs;primal[role,fold]=cs
            for k in gs[0]:
                v=[math.fsum(sg(row)*x for row,x in zip(rows,col,strict=True))/len(rows) for col in zip(*(g[k] for g in gs),strict=True)]
                means[role,fold,k]=v;typical[role,fold,k]=mean([math.hypot(*g[k]) for g in gs])
    require(next(events,None) is None and not binary.read(1) and replay==32,"missing original rows")
    events=iter(io.BytesIO(read(out/"finite.jsonl",m["files"]["finite.jsonl"],4<<20)))
    binary=io.BytesIO(read(out/"contrasts.bin",m["files"]["contrasts.bin"],MAX["finite"]*VALUE.size))
    results={};gauges={};gates={};total_finite=0
    for role in ROLES:
        owner=p["roles"][role];a,layout=artifact_coords(proto/"model"/(owner[7:]+".json"),owner[7:])
        dmap=dirs["roles"][role];require(set(dmap)==set(coordinates()),"missing route");results[role]={};gauges[role]={}
        for label in coordinates():
            key=route(label);g=means[role,"fit",key];n=math.hypot(*g);v=dmap[label]
            gauge=label.endswith("/gauge")
            expected=([(-1.)**j/math.sqrt(W) for j in range(W)]+[0.]*W if gauge else [x/n for x in g] if n else None)
            require(v==expected,"direction not fitted exclusively to even edges")
            if v is None:results[role][label]=None;continue
            other=means[role,"check",key];on=math.hypot(*other)
            def ratio(f):
                denom=typical[role,f,key]
                return math.hypot(*means[role,f,key])/denom if denom else None
            info=dict(fit_norm=n,check_norm=on,fit_cancellation=ratio("fit"),check_cancellation=ratio("check"),
                      check_cosine=dot(other,v)/on if on else None,radii={})
            max_gauge=0.;max_gc=0.;flips=0
            for radius in RADII:
                cuts={};physical={}
                for side in (-1,1):
                    t=json.loads(next(events));h=radius*side
                    require((t["role"],t["owner"],t["label"],t["radius"],t["sign"])==(role,owner,label,radius,side),"wrong physical comparison")
                    if label=="geometry":
                        pin=t["temporary_geometry"];require(isinstance(pin,str) and pin.startswith("sha256:") and len(pin)==71 and all(c in "0123456789abcdef" for c in pin[7:]),"bad temporary owner")
                        end,le=artifact_coords(out/"geometry"/(pin[7:]+".json"),pin[7:])
                        require(le==layout and end==[f32(x+h*y) for x,y in zip(a,v,strict=True)],"not the recorded rounded geometry")
                        delta=[y-x for x,y in zip(a,end,strict=True)]
                    else:
                        require(t["temporary_geometry"] is None,"unexpected model edit");delta=[f32(h*y) for y in v]
                    require(t["actual_delta"]==sha(struct.pack(f"<{len(delta)}d",*delta)) and
                            abs(math.hypot(*delta)/radius-1)<=1e-4,"wrong displacement")
                    block=binary.read(64*VALUE.size);require(len(block)==64*VALUE.size and sha(block)==t["values"],"unbound observations")
                    trial=list(VALUE.iter_unpack(block));require(all(0<=k<64 and math.isfinite(x-y) for x,y,k in trial),"nonfinite observations")
                    physical[side]=delta;total_finite+=64
                    for j,(fold,rows) in enumerate(folds.items()):
                        obs=trial[j*32:(j+1)*32];ds=[]
                        for row,(x,y,k),origin in zip(rows,obs,primal[role,fold],strict=True):
                            gap=(x-y)-origin["target_minus_alternative"];ds.append(sg(row)*gap)
                            if gauge:
                                max_gauge=max(max_gauge,abs(gap),abs(x-origin["target_logit"]),abs(y-origin["alternative_logit"]))
                                flips+=k!=origin["greedy_id"]
                        cuts[side,fold]=ds
                detail={}
                for fold,rows in folds.items():
                    grad=means[role,fold,key];yp,ym=cuts[1,fold],cuts[-1,fold]
                    ref=central(dot(grad,physical[1]),dot(grad,physical[-1]),mean(yp),mean(ym))
                    obs=[(x-y)/2 for x,y in zip(yp,ym,strict=True)]
                    ref["by_edge"]={str(e):mean([x for x,r in zip(obs,rows,strict=True) if r["edge"]==e]) for e in sorted({r["edge"] for r in rows})}
                    ref["by_order"]=[mean([x for x,r in zip(obs,rows,strict=True) if r["order"]==o]) for o in (0,1)]
                    half=[(x-y)/2 for x,y in zip(physical[1],physical[-1],strict=True)]
                    ref["max_prompt_error"]=max(abs(z-sg(row)*dot(gg[key],half)) for z,row,gg in zip(obs,rows,values[role,fold],strict=True))
                    detail[fold]=ref
                    if gauge:max_gc=max(max_gc,abs(ref["central"]))
                info["radii"][str(radius)]=detail
            results[role][label]=info
            if gauge:gauges[role][label]=dict(max_drift=max_gauge,max_central=max_gc,greedy_flips=flips,passed=max_gauge<=2e-5 and max_gc<=1e-6)
        gates[role]={label:decision(results[role],label) and all(x["passed"] for x in gauges[role].values())
                     for label in coordinates() if label.split("/")[1:2] in (["query"],["key"]) and not label.endswith("/gauge")}
    counts["finite"]=total_finite
    require(next(events,None) is None and not binary.read(1) and dict(counts)==m["counts"] and all(v<=MAX[k] for k,v in counts.items()) and
            all(counts[k]==v for k,v in MAX.items() if k!="finite"),"unexpected number of model calls")
    dst.mkdir(mode=0o700)
    return put(dst/"results.json",dict(plan=pp,measurement=mp,counts=dict(counts),replayed_training_word_pairs=replay,roles=results,gauges=gauges,gates=gates,
        origins={k:all(gates[s+"/origin"][k] for s in task.STATES) for k in gates[ROLES[0]]},
        endpoints={k:all(gates[s+"/direct_matched"][k] for s in task.STATES) for k in gates[ROLES[0]]},
        scope="Virtual local coordinates at fixed models on two disjoint sets of training edges; not a training update, unseen evaluation, or causal storage attribution."))
