"""All control-vs-attention decisions and actual source separation; use saved rows only."""
from pathlib import Path
import argparse
import json
import math
from statistics import median
import runpy
from types import SimpleNamespace

a=SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("audit.py"))))
s,x,d,k,need=a.s,a.x,a.d,a.k,a.need

def readback(folder,pin):
    result=x.read(folder/"result.json",pin);inp,sp=s.data(folder,result["inputs"])
    proof=x.read(folder/"training-checked.json",result["checked"]);train=folder/"training"
    files=x.read(folder/"trained.json",result["trained"])["files"]
    ev=x.read(folder/"scored.json",result["scored"]);path=folder/"evaluation/scores.jsonl"
    need(ev["files"]["scores.jsonl"]==d.digest(path.read_bytes()),"wrong transfer journal")
    cases=x.read(folder/"transfer.json",inp["transfer"]);z=a.bank(sp,cases,path,proof["owners"])
    records=[];switches=[];sources=[]
    for b in sp["bases"]:
        bank={m:x.read(train/(s.arm(b,m)+"-unit.json"),files[s.arm(b,m)+"-unit.json"]) for m in s.MODES}
        for f in sorted({r["template"] for r in cases}):
            rows=[r for r in sp["train"] if r["template"]==f]
            for ctl in s.MODES[1:]:
                sep=[];rel=[]
                for r in rows:
                    p=bank["attention"][r["key"]]["source"];q=bank[ctl][r["key"]]["source"]
                    delta=math.sqrt(math.fsum((i-j)**2 for i,j in zip(p,q)))
                    norm=math.sqrt(math.fsum(i*i for i in p));sep.append(delta)
                    if norm:rel.append(delta/norm)
                sources.append(dict(base=b,template=f,control=ctl,rows=len(sep),nonidentical=sum(t>0 for t in sep),
                    median_l2=median(sep),relative_defined=len(rel),median_relative=median(rel) if rel else None))
        for t in x.TEACHERS:
            for o in sp["orders"]:
                att=z[s.arm(b,"attention",t,o)]
                for ctl in s.MODES[1:]:
                    other=z[s.arm(b,ctl,t,o)];delta=[];n=0
                    for r in cases:
                        key=r["key"];p,q=att[key],other[key]
                        g,w=p[7]-p[8],q[7]-q[8];delta.append(g-w)
                        sign=lambda a:(a>0)-(a<0)
                        if sign(g)!=sign(w):
                            switches.append(dict(base=b,teacher=t,order=o,key=key,control=ctl,
                                                 attention_gap=g,control_gap=w,expected_sign=1-2*r[t]))
                            n+=1
                    records.append(dict(base=b,teacher=t,order=o,control=ctl,rows=len(delta),decision_switches=n,
                        median_abs_gap_change=median(abs(y) for y in delta),max_abs_gap_change=max(map(abs,delta))))
    return dict(result=pin,model_calls=0,sources=sources,decisions=records,switches=switches)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("folder",type=Path);p.add_argument("pin");p.add_argument("output",type=Path)
    args=p.parse_args();print(d.save(args.output,readback(args.folder,args.pin)))
