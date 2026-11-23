"""Descriptive only: decompose the already scored verb decisions, without refitting or a forward."""
import argparse
import json
import math
from pathlib import Path
import runpy
from types import SimpleNamespace

a=SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("audit.py"))))
u,d,s,need=a.u,a.d,a.s,a.need
VIEWS=("full","train_shift_only","train_shift_removed","frame_shift_only","frame_shift_removed")

def avg(xs):
    return math.fsum(xs)/len(xs)

def sign(v):
    return int(v>0)-int(v<0)

def one(train,rows,z0,z,origin,end,t):
    drift=avg([a.delta(z,r["key"])-a.delta(z0,r["key"]) for r in train])
    steps={r["key"]:a.delta(end,r["key"])-a.delta(origin,r["key"]) for r in rows}
    shifts={key:avg([steps[k] for k in group]) for group in u.quartets(rows) for key in group}
    ss=math.fsum(x*x for x in steps.values())
    out={v:dict(right=0,up=0,down=0,correctness_flips=0,reproduced=0,extra=0) for v in VIEWS}
    keys={"up":[],"down":[]}
    before=0;full_changed=0
    for r in rows:
        k=r["key"];base=a.delta(origin,k);last=a.delta(end,k);label=1-2*r[t]
        old=sign(label*base);actual=sign(label*last)
        before+=old>0;changed=(old>0)!=(actual>0);full_changed+=changed
        if changed:keys["up" if actual>0 else "down"].append(k)
        vs=(last,base+drift,last-drift,base+shifts[k],last-shifts[k])
        for view,x in zip(VIEWS,vs):
            v=out[view];pred=sign(label*x);on=(old>0)!=(pred>0)
            v["right"]+=pred>0;v["up"]+=old<=0<pred;v["down"]+=old>0>=pred
            v["correctness_flips"]+=on;v["reproduced"]+=changed and pred==actual;v["extra"]+=on and not changed
    for v in out.values():
        need(v["right"]-before==v["up"]-v["down"] and v["correctness_flips"]==v["up"]+v["down"],"flip counts")
    return dict(before=before,full_changed=full_changed,changed_keys=keys,n=len(rows),views=out,train_offset=drift,
                transfer_offset=avg(list(steps.values())),frame_mean_energy_fraction=math.fsum(v*v for v in shifts.values())/ss if ss else None)

def readback(root,bank,signal,primary,pin,out):
    report=u.read(primary,pin)
    cfg,sp,m=u.load_inputs(root,bank,report["inputs"])
    checked=a.review_training(root,bank,signal,report["inputs"])
    need(report["training"]==checked and json.loads((root/"checked.json").read_bytes())==checked,"not audited train split")
    end=u.read(root/"complete.json",report["complete"])
    need(end["checked"]==u.sha(root/"checked.json") and end["consumed"]==u.sha(root/"consumed.json") and
         end["scores"]==u.sha(root/"transfer-logits.jsonl"),"not audited transfer logits")
    train=sp["train"];cases=u.read(root/"transfer.json",cfg["transfer"])
    z=a.journal(root/"training-logits.jsonl",bank,m,sp,train)
    e=a.journal(root/"transfer-logits.jsonl",bank,m,sp,cases)
    table=[]
    for cell in report["rows"]:
        b,o,form=(cell[k] for k in ("base","order","template"))
        for t in a.LABELS:
            mode=cell[t+"_route"];owner=s.key(b,mode,t,o);origin=s.key(b,u.MAPS[0])
            result=one(train,[r for r in cases if r["template"]==form],z[origin],z[owner],e[origin],e[owner],t)
            actual=result["views"]["full"]
            need((actual["right"],actual["up"],actual["down"])==tuple(cell[t+"_raw_"+k] for k in ("after","up","down")),
                 "primary decision does not reproduce")
            table.append(dict(result,base=b,order=o,template=form,policy=cell["policy"],teacher=t,route=mode))
    return d.save(out,dict(schema="spiraltorch.matched_route.shift_readback.v1",source_report=pin,
                           new_model_calls=0,post_hoc=True,rows=table))

if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for n in ("root","bank","signal","primary"):cli.add_argument(n,type=Path)
    cli.add_argument("pin");cli.add_argument("out",type=Path)
    print(readback(**vars(cli.parse_args())))
