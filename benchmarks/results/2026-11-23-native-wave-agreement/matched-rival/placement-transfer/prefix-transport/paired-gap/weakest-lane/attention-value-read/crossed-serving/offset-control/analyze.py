"""Apply frozen offsets using Rust and replay every stop, strict sign and set intersection."""
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import contract as k


def both(z):
    return z[0]>0 and z[1]<0


def exact_decisions(source,target):
    low,high=-math.inf,math.inf
    equal=None
    for p,q in zip(source,target,strict=True):
        for d,y in zip(p,q,strict=True):
            if y>0: low=max(low,-d)
            elif y<0: high=min(high,-d)
            elif equal is not None and equal!=-d: return False
            else: equal=-d
    if equal is not None:
        return low<equal<high
    return math.nextafter(low,math.inf)<high


def stats(rows):
    n=len(rows); k.need(n>0,"empty metric stratum")
    out=dict(pairs=n,both=0,individual=0,ties=0,target_both=0,shared=0,missed=0,other=0,identical_sign=0)
    gaps=[];err=[]
    sign=lambda v:(v>0)-(v<0)
    for _,_,u,t in rows:
        hit,ref=both(u),both(t)
        out["both"]+=hit;out["target_both"]+=ref
        out["individual"]+=(u[0]>0)+(u[1]<0)
        out["ties"]+=sum(v==0 for v in u)
        out["shared"]+=hit and ref;out["missed"]+=ref and not hit;out["other"]+=hit and not ref
        out["identical_sign"]+=sum(sign(v)==sign(w) for v,w in zip(u,t))
        gaps.append(min(u[0],-u[1]));err.extend((v-w)**2 for v,w in zip(u,t))
    k.need(out["both"]==out["shared"]+out["other"] and out["target_both"]==out["shared"]+out["missed"],"bad overlap accounting")
    return dict(**out,mean_weaker=math.fsum(gaps)/n,contrast_rmse=math.sqrt(math.fsum(err)/(2*n)))


def analysis(groups,fits,program):
    queries=[];pool={};bounds=[];cells=[]
    for g in groups:
        for src in k.SOURCES:
            key=g["group"]+"."+src
            selected=[p for p in g["pairs"] if both(k.contrasts(p["scores"]["UU"]))]
            if selected:
                queries.append(("B",key+".ceiling",[w for p in selected for w in k.words(p["scores"][src])]))
            for how in k.METHODS:
                offset=0. if how=="unshifted" else fits[key][how]
                args=[k.F64.pack(offset).hex()]+[w for p in g["pairs"] for w in k.words(p["scores"][src])]
                queries.append(("S",key+"."+how,args))
    ans=k.engine(program,queries)
    for g in groups:
        b,t,o,scale=(g[n] for n in ("base","teacher","order","scale"))
        for src in k.SOURCES:
            key=g["group"]+"."+src
            target=[k.contrasts(p["scores"]["UU"]) for p in g["pairs"]]
            source=[k.contrasts(p["scores"][src]) for p in g["pairs"]]
            winners=[v for v,y in zip(source,target) if both(y)]
            sep,maximum=(0,0) if not winners else map(int,ans[key+".ceiling"])
            if winners:
                k.need(k.exhaustive(winners)[:2]==(sep,maximum),"incorrect strict interval upper bound")
            bounds.append(dict(base=b,teacher=t,order=o,scale=scale,source=src,target_wins=len(winners),
                               separable=sep,simultaneous=maximum,exact_signs_possible=exact_decisions(source,target)))
            for how in k.METHODS:
                offset=0. if how=="unshifted" else fits[key][how]
                gaps=list(map(k.wide,ans[key+"."+how]));k.need(len(gaps)==2*len(target),"wrong Rust score count")
                pairs=[]
                for i,(p,z,y) in enumerate(zip(g["pairs"],source,target,strict=True)):
                    u=gaps[2*i:2*i+2]
                    k.need(u==[v+offset for v in z],"wrong serving offset or sign")
                    pairs.append((p["template"],p["keys"],u,y))
                for f in sorted({r[0] for r in pairs}):
                    cells.append(dict(base=b,teacher=t,order=o,scale=scale,source=src,method=how,template=f,
                                      offset=offset,**stats([r for r in pairs if r[0]==f])))
                for base in (b,None):pool.setdefault((base,t,scale,src,how),[]).extend(pairs)
    totals=[dict(base=b,teacher=t,scale=sc,source=s,method=m,**stats(p)) for (b,t,sc,s,m),p in pool.items()]
    ceilings=[]
    for b in [None]+sorted({g["base"] for g in groups}):
        for t in k.c.x.TEACHERS:
            for sc in k.c.SCALES:
                for src in k.SOURCES:
                    part=[z for z in bounds if (b is None or z["base"]==b) and (z["teacher"],z["scale"],z["source"])==(t,sc,src)]
                    ceilings.append(dict(base=b,teacher=t,scale=sc,source=src,groups=len(part),
                                **{n:sum(z[n] for z in part) for n in ("target_wins","separable","simultaneous","exact_signs_possible")}))
    return dict(totals=totals,cells=cells,group_bounds=bounds,ceilings=ceilings)
