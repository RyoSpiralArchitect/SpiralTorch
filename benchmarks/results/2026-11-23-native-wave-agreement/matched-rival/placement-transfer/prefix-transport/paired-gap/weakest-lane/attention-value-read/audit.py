"""No model imports. Reconstruct all saved steps, owners, calibration and pair outcomes."""
from pathlib import Path
import json
import math
import runpy
from statistics import median
from types import SimpleNamespace

s=SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("study.py"))))
x,d,k,v,need=s.x,s.d,s.k,s.v,s.need

def bank(sp,rows,path,owners,units=None):
    log=s.s.legacy.records(path);out={}
    for b in sp["bases"]:
        for mode in s.MODES:
            origin=s.arm(b,mode)
            for n in [origin]+[r[0] for r in s.paths(sp,b,mode)]:
                rec=out[n]={}
                for case in rows:
                    r=next(log);key=case["key"]
                    need((r["name"],r["key"],r["owner"])==(n,key,owners[n]) and key not in rec,"scoring ownership")
                    z=rec[key]=r["logits"];v.u.bits(z)
                    if n==origin and units is not None:
                        q=units[origin][key]["contrast"]
                        need((z[7],z[8],max(range(11),key=z.__getitem__))==
                             (q["target_logit"],q["alternative_logit"],q["greedy_id"]),"trace vs origin")
        origins=[out[s.arm(b,m)] for m in s.MODES]
        need(all(len({v.u.bits(z[key]) for z in origins})==1 for key in origins[0]),"origins changed logits")
    need(next(log,None) is None,"extra scored rows")
    return out

def capacity(rows,sp,units,metrics):
    cells=[];gates={};all_pairs=[]
    for b in sp["bases"]:
        gates[str(b)]={}
        for t in x.TEACHERS:
            ok=True
            for f in sorted({r["template"] for r in rows}):
                by={}
                for mode in s.MODES:
                    n=s.arm(b,mode);q=metrics[n];sig=units[n]
                    vals=[]
                    for pos,neg in x.pairs([r for r in rows if r["template"]==f],t):
                        l,r=[sig[z["key"]]["gate_gradient"] for z in (pos,neg)]
                        val=s.cap(l,[-y for y in r],q,v.TARGET);vals.append(val)
                        all_pairs.append(dict(base=b,teacher=t,template=f,mode=mode,keys=[pos["key"],neg["key"]],**val))
                    ratios=[z["ratio"] for z in vals if z["ratio"] is not None]
                    st=by[mode]=dict(pairs=len(vals),defined=len(ratios),ratio=median(ratios) if ratios else None,
                        gain=median(z["common_gain"] for z in vals),over_cap=sum(z["physical_l2"]>16 for z in vals),
                        opposing=sum(z["cosine"] is not None and z["cosine"]<0 for z in vals))
                    cells.append(dict(base=b,teacher=t,template=f,mode=mode,**st))
                att=by["attention"];controls=[by[m] for m in ("uniform","fixed2")]
                ok &= (all(c["defined"]==c["pairs"] and not c["over_cap"] for c in by.values()) and
                       att["ratio"]>max(c["ratio"] for c in controls)+.05 and att["gain"]>=max(c["gain"] for c in controls))
            gates[str(b)][t]=ok
    return dict(cells=cells,base_gates=gates,pairs=all_pairs)

def training(folder,parent,sp,pin,trained):
    info=x.read(folder/"trained.json",trained);inp,_=s.data(folder,pin)
    out=folder/"training";files=k.identities(out);owners={};units={};metrics={}
    need(info["inputs"]==pin and info["files"]==files,"broken training file seal")
    card=x.read(folder/"run-card.json",info["card"])
    need(card==dict(inputs=pin,installed=k.read(folder.parent/"installed.json"),sources=inp["sources"],
                    retry=False,seconds=300 if sp["kind"]=="formal" else 120),"run card")
    hosts=s.old.hosts(parent,sp)
    steps=s.s.legacy.records(out/"steps.jsonl");actual=s.s.legacy.records(out/"physical.jsonl")
    count=phys=0;err=0.;expected={"steps.jsonl","physical.jsonl","scores.jsonl"}
    for b in sp["bases"]:
        h=hosts[str(b)]
        for mode in s.MODES:
            origin=s.arm(b,mode)
            def raw(n):
                f="weights/"+n+".json";expected.add(f);return s.s.c.pinned(out/f,files[f])
            def evidence(suffix):
                f=origin+"-"+suffix+".json";expected.add(f);return x.read(out/f,files[f])
            sig=units[origin]=evidence("unit");mr=evidence("metric")
            oid,p0=s.state(raw(origin),h,mode);owners[origin]=oid
            need(p0==[0.]*6 and set(sig)=={r["key"] for r in sp["train"]},"wrong origin evidence")
            traces=[sig[r["key"]] for r in sp["train"]]
            means=[s.unit(tr,r,h,oid,mode) for tr,r in zip(traces,sp["train"])]
            q=metrics[origin]=s.metric(mr,means,h,oid)
            for n,t,o in s.paths(sp,b,mode):
                pre,params=oid,p0
                schedule=x.pairs(sp["train"],t,o)*sp["epochs"]
                for i,pair in enumerate(schedule):
                    r=next(steps);count+=1
                    need((r["name"],r["i"])==(n,i),"update order")
                    end,vals=s.state(r["state"].encode(),h,mode)
                    need((r["id"],r["values"])==(end,vals),"wrong archived step bytes")
                    s.s.validate_gradient(r,pair,"weakest_lane",[sig[z["key"]] for z in pair],h,pre,i==0)
                    s.ck.step(r,params,pre,q,mr["id"])
                    err=max(err,abs(r["step"]["realized_rms"]/v.TARGET-1))
                    if i in (0,len(schedule)-1):
                        w=next(actual);phys+=1;need((w["name"],w["i"])==(n,i),"witness order")
                        s.witness(w,params,vals,traces,means,r["step"])
                    pre,params=end,vals
                need(raw(n)==r["state"].encode(),"wrong endpoint");owners[n]=pre
        factual=[units[s.arm(b,m)] for m in s.MODES]
        need(len({owners[s.arm(b,m)] for m in s.MODES})==3 and
             all(factual[0][key][f]==other[key][f] for key in factual[0] for other in factual[1:]
                 for f in ("query","keys","values","site_input","upstream","contrast","scale","key_lag")),
             "not a matched factual host")
    need(set(files)==expected and next(steps,None) is None and next(actual,None) is None,"extra training evidence")
    logits=bank(sp,sp["train"],out/"scores.jsonl",owners,units)
    calls=sum(map(len,logits.values()));captures=sum(map(len,units.values()))
    need((count,phys,calls,captures)==((3456,108,4032,576) if sp["kind"]=="formal" else (24,12,72,24)),
         "wrong training budget")
    return dict(trained=trained,owners=owners,updates=count,physical_checks=phys,unit_captures=captures,
                calibration_prefixes=captures,training_forwards=calls,max_relative_rms_error=err,
                mechanism=capacity(sp["train"],sp,units,metrics),routing=routing(sp,units))

def routing(sp,units):
    out=[]
    for b in sp["bases"]:
        sig=units[s.arm(b,"attention")]
        for f in sorted({r["template"] for r in sp["train"]}):
            values=[]
            for r in sp["train"]:
                if r["template"]!=f:continue
                head,near=[i for i,t in enumerate(r["prompt"]) if t in (1,2)]
                ps,_=s.mix(sig[r["key"]])
                for p in ps:
                    values.append([p[head],p[near],.5*math.fsum(abs(y-1/len(p)) for y in p)])
            means=[math.fsum(c)/len(c) for c in zip(*values)]
            out.append(dict(base=b,template=f,head_queries=len(values),head_marker_mass=means[0],
                            nearby_marker_mass=means[1],tv_from_uniform=means[2]))
    return out

def outcomes(rows,sp,z):
    cells=[];gates={}
    for b in sp["bases"]:
        start=z[s.arm(b,"attention")];gates[str(b)]={}
        for t in x.TEACHERS:
            passed=[True,True,True]
            for o in sp["orders"]:
                for f in sorted({r["template"] for r in rows}):
                    part=[r for r in rows if r["template"]==f]
                    stats=[]
                    for mode in s.MODES:
                        result=s.m.outcomes(part,start,z[s.arm(b,mode,t,o)],t);stats.append(result)
                        cells.append(dict(base=b,teacher=t,order=o,template=f,mode=mode,**result))
                    r,u,fixed=stats
                    margin=r["mean_min"]>max(fixed["mean_min"],u["mean_min"],r["mean_min_before"])+.001
                    both=r["both"]>max(fixed["both"],u["both"],r["both_before"])
                    raw=r["raw"]["correct"]>=max(fixed["raw"]["correct"],u["raw"]["correct"],r["raw"]["before"])
                    passed=[a and k for a,k in zip(passed,(margin,margin and both,margin and both and raw))]
            gates[str(b)][t]=dict(zip(("margin","both_and_margin","quality"),passed))
    return dict(cells=cells,base_gates=gates)

def result(folder,parent,pin,trained,checked,scored):
    inp,sp=s.data(folder,pin)
    proof=x.read(folder/"training-checked.json",checked)
    need(proof==training(folder,parent,sp,pin,trained),"training replay")
    ev=folder/"evaluation";sm=x.read(folder/"scored.json",scored)
    need(sm==dict(inputs=pin,trained=trained,checked=checked,files=k.identities(ev)) and
         set(sm["files"])=={"scores.jsonl","consumed.json"} and k.read(ev/"consumed.json")==
         dict(inputs=pin,transfer=inp["transfer"],trained=trained,checked=checked,retry=False),"transfer seal")
    rows=x.read(folder/"transfer.json",inp["transfer"])
    need(rows==s.cases(sp["kind"],True),"transfer inputs")
    z=bank(sp,rows,ev/"scores.jsonl",proof["owners"])
    train=bank(sp,sp["train"],folder/"training/scores.jsonl",proof["owners"])
    n=sum(map(len,z.values()));need(n==(2016 if sp["kind"]=="formal" else 72),"transfer budget")
    return dict(inputs=pin,trained=trained,checked=checked,scored=scored,transfer_forwards=n,
        work={k:v for k,v in proof.items() if k not in ("trained","owners","mechanism","routing")},routing=proof["routing"],
        mechanism={k:v for k,v in proof["mechanism"].items() if k!="pairs"},
        train=outcomes(sp["train"],sp,train),transfer=outcomes(rows,sp,z))

if __name__=="__main__":
    import sys
    folder,parent=map(Path,sys.argv[1:3])
    pins=[d.digest((folder/n).read_bytes()) for n in ("inputs.json","trained.json","training-checked.json","scored.json")]
    raw=d.encoded(result(folder,parent,*pins));need(raw==(folder/"result.json").read_bytes(),"result bytes")
    print(d.digest(raw))
