"""Task outcomes from the complete frozen 11-logit rows; implementations do not import a model."""
from pathlib import Path
import math
import runpy
from types import SimpleNamespace

s=SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("study.py"))))
x,d,need=s.x,s.d,s.need


def lanes(z,pair):
    rp,rn=[z[r["key"]] for r in pair]
    plus,minus=rp[7]-rp[8],rn[8]-rn[7]
    return min(plus,minus), (plus+minus)/2, (plus-minus)/2


def outcomes(rows,origin,end,teacher):
    qa,qb=[],[]
    ups,downs=[],[]
    pairs=x.pairs(rows,teacher)
    for pair in pairs:
        a,b=lanes(origin,pair),lanes(end,pair)
        need(math.isclose(b[0],b[1]-abs(b[2]),abs_tol=2e-13,rel_tol=2e-13),"invalid pair decomposition")
        qa.append(a[0]);qb.append(b[0])
        tag="/".join(r["key"] for r in pair)
        if a[0]<=0<b[0]:ups.append(tag)
        if b[0]<=0<a[0]:downs.append(tag)
    st=s.legacy.metrics(rows,origin,end,teacher)
    n=len(qa)
    st.update(both_before=sum(m>0 for m in qa),both=sum(m>0 for m in qb),
              weak_ties_before=qa.count(0.),weak_ties=qb.count(0.),
              mean_min_before=math.fsum(qa)/n,mean_min=math.fsum(qb)/n,
              gained_both=ups,lost_both=downs)
    need(st["both"]-st["both_before"]==len(ups)-len(downs),"bad raw pair accounting")
    return st


def analyze(sp,rows,z):
    table=[];by_base={};disagreements={}
    for b in sp["bases"]:
        start=z[x.name(b)]
        by_base[str(b)]={};disagreements[str(b)]={}
        for t in x.TEACHERS:
            passed=[True]*3
            changes={ref:[] for ref in ("binary","rank")}
            for o in sp["orders"]:
                ends={arm:z[x.name(b,arm,t,o)] for arm in s.ARMS}
                for f in sorted({r["template"] for r in rows}):
                    part=[r for r in rows if r["template"]==f]
                    stats={arm:outcomes(part,start,end,t) for arm,end in ends.items()}
                    for arm,st in stats.items():
                        table.append(dict(base=b,order=o,teacher=t,template=f,arm=arm,**st))
                    w,ce,rk=[stats[a] for a in ("weakest_lane","binary","rank")]
                    cont=w["mean_min"]>max(w["mean_min_before"],ce["mean_min"],rk["mean_min"])+.001
                    both=w["both"]>max(w["both_before"],ce["both"],rk["both"])
                    raw=w["raw"]["correct"]>=max(w["raw"]["before"],ce["raw"]["correct"],rk["raw"]["correct"])
                    passed=[a and b0 for a,b0 in zip(passed,(cont,cont and both,cont and both and raw))]
                    for pair in x.pairs(part,t):
                        weak=lanes(ends["weakest_lane"],pair)[0]
                        for ref in changes:
                            control=lanes(ends[ref],pair)[0]
                            if (weak>0)!=(control>0):
                                changes[ref].append(dict(order=o,template=f,key="/".join(r["key"] for r in pair),
                                                         gain=weak>0,weak=weak,control=control))
            by_base[str(b)][t]=dict(zip(("continuous","both_and_margin","quality"),passed))
            disagreements[str(b)][t]=changes
    return dict(cells=table,base_gates=by_base,differences=disagreements)
