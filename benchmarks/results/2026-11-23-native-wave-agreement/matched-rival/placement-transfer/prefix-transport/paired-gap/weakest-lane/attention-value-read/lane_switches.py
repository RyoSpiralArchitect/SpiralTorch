"""Post hoc: when did matched training trajectories choose different weakest lanes?
Saved training steps only. A different subgradient is not proof of a cause of transfer.
"""
from pathlib import Path
import json
import sys
import runpy
from types import SimpleNamespace

s=SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("study.py"))))
x,d,need=s.x,s.d,s.need

def describe(folder,result_id):
    result=x.read(folder/"result.json",result_id);_,sp=s.data(folder,result["inputs"])
    seal=x.read(folder/"trained.json",result["trained"])
    path=folder/"training/steps.jsonl";raw=s.s.c.pinned(path,seal["files"]["steps.jsonl"])
    arms={}
    for r in map(json.loads,raw.splitlines()):
        seq=arms.setdefault(r["name"],[])
        need(r["i"]==len(seq),"step order")
        active=r["objective"]["weakest_lane"]["active"]
        need(active in ([True,False],[False,True],[True,True]),"invalid active lanes")
        seq.append((r["keys"],active))
    cells=[]
    for b in sp["bases"]:
        for teacher in x.TEACHERS:
            for order in sp["orders"]:
                schedule=[[p["key"] for p in pair] for pair in x.pairs(sp["train"],teacher,order)]*sp["epochs"]
                actual=arms[s.arm(b,"attention",teacher,order)]
                for mode in s.MODES[1:]:
                    control=arms[s.arm(b,mode,teacher,order)]
                    need(len(schedule)==len(actual)==len(control) and
                         all(a[0]==c[0]==p for p,a,c in zip(schedule,actual,control)),"unmatched schedule")
                    diffs=[i+1 for i,(a,c) in enumerate(zip(actual,control)) if a[1]!=c[1]]
                    need(1 not in diffs,"zero-gate origin chose different lane")
                    cells.append(dict(base=b,teacher=teacher,order=order,control=mode,
                                      steps=len(actual),different=len(diffs),first=diffs[0] if diffs else None,
                                      different_steps=diffs))
    return dict(result=result_id,trained=result["trained"],step_sha256=seal["files"]["steps.jsonl"],
                post_hoc=True,model_calls=0,cells=cells)

if __name__=="__main__":
    folder,pin,out=sys.argv[1:];print(d.save(Path(out),describe(Path(folder),pin)))
