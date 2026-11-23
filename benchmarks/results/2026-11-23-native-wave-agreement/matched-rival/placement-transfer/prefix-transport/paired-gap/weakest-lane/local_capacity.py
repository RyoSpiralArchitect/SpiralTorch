"""Per-pair optimistic first-order capacity at the three frozen origins; training traces only."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import runpy
from statistics import median
from types import SimpleNamespace

s=SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("study.py"))))
d,x,need=s.d,s.x,s.need


def common_direction(a,b,q,rms):
    """Maximize the smaller linearized margin gain on a metric ball. No damped optimizer."""
    need(len(a)==len(b)==len(q)>0 and math.isfinite(rms) and rms>0 and
         all(math.isfinite(v) and v>0 for v in q) and all(math.isfinite(v) for row in (a,b) for v in row),
         "degenerate metric")
    dot=lambda p,r:math.fsum(i*j for i,j in zip(p,r))
    # The metric dual inner product is rational in the stored data. Work in these
    # coordinates before taking a square root so exact opposition yields an exact zero.
    dual=[len(q)/Fraction(v) for v in q]
    u,v=([Fraction(z) for z in vec] for vec in (a,b))
    inner=lambda p,r:sum(w*i*j for w,i,j in zip(dual,p,r))
    diff=[x-y for x,y in zip(u,v)]
    dd=inner(diff,diff)
    # The point closest to zero on [v,u] gives a dual upper bound. Its direction attains it.
    t=max(0,min(1,-inner(v,diff)/dd)) if dd else Fraction(1,2)
    p=[t*x+(1-t)*y for x,y in zip(u,v)]
    norm=lambda z:math.sqrt(inner(z,z))
    n=norm(p)
    step=[rms*float(w*y)/n for w,y in zip(dual,p)] if n else [0.]*len(q)
    upper=rms*n
    min_gain=min(dot(a,step),dot(b,step))
    smaller=rms*min(norm(u),norm(v))
    need(abs(min_gain-upper)<=2e-11*max(1e-12,smaller,rms*norm(u),rms*norm(v)),"primal/dual gap")
    delivered=math.sqrt(math.fsum(z*y*y for z,y in zip(q,step))/len(q))
    need(delivered<=rms*(1+1e-12) and (not n or math.isclose(delivered,rms,rel_tol=1e-12)),"wrong lifted metric")
    return dict(common_gain=upper,smaller_individual_gain=smaller,ratio=upper/smaller if smaller else None,
                physical_l2=math.sqrt(dot(step,step)),lambda_first=float(t),
                cosine=float(inner(u,v))/(norm(u)*norm(v)) if smaller else None)


def measure(folder,pin):
    inp,sp=s.data(folder,pin)
    raw=(folder/"training-checked.json").read_bytes()
    checked=json.loads(raw)
    trained=x.read(folder/"trained.json",checked["trained"])
    need(trained["inputs"]==pin and inp["kind"]=="formal","wrong training seal")
    files=trained["files"]; out=folder/"training"
    rows=[];summary=[]
    for b in sp["bases"]:
        unit=x.read(out/("unit-%d.json"%b),files["unit-%d.json"%b])
        metric=x.read(out/("metric-%d.json"%b),files["metric-%d.json"%b])
        need(metric["anchor"]==checked["owners"][x.name(b)] and metric["samples"]==len(sp["train"]),"foreign calibration")
        for teacher in x.TEACHERS:
            group=[]
            for p,n in x.pairs(sp["train"],teacher):
                pair=[unit[p["key"]],unit[n["key"]]]
                need(all(v["geometry_artifact_id"]==metric["anchor"] for v in pair),"foreign gradient")
                cap=common_direction(pair[0]["gate_gradient"],[-y for y in pair[1]["gate_gradient"]],
                                     metric["second_moments"],s.v.TARGET)
                group.append(cap)
                rows.append(dict(base=b,teacher=teacher,keys=[p["key"],n["key"]],**cap))
            usable=[r for r in group if r["ratio"] is not None]
            summary.append(dict(base=b,teacher=teacher,pairs=len(group),defined=len(usable),
                 over_cap=sum(r["physical_l2"]>16. for r in group),negative_cosines=sum(r["cosine"]<0 for r in usable),
                 median_ratio=median([r["ratio"] for r in usable]) if usable else None,
                 median_cosine=median([r["cosine"] for r in usable]) if usable else None,
                 median_common_gain=median(r["common_gain"] for r in group)))
    need(len(rows)==192,"not the expected distinct origin pairs")
    return dict(inputs=pin,training_check=d.digest(raw),model_calls=0,summary=summary,pairs=rows)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder",type=Path);ap.add_argument("pin");ap.add_argument("output",type=Path)
    args=ap.parse_args()
    print(d.save(args.output,measure(args.folder,args.pin)))
