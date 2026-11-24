"""Descriptive block-size readback only: no model calls, new gradients or tuning."""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import median
import sys

def report(root):
    from hashlib import sha256
    raw=(root/"attempt/steps.jsonl").read_bytes()
    if sha256(raw).hexdigest()!=json.loads((root/"complete.json").read_bytes())["files"]["steps.jsonl"]:
        raise ValueError("step journal changed")
    prev={}
    groups=defaultdict(list)
    def fraction(vec):
        den=math.fsum(x*x for x in vec)
        if not den or len(vec)!=6: raise ValueError("undefined block fraction")
        return math.fsum(x*x for x in vec[3:])/den  # canonical [bias, gate]
    for line in raw.splitlines():
        rec=json.loads(line);cell=rec["cell"];b,t,role,_=cell.split("-")
        last=prev.get(cell,[0.]*6)
        delta=[x-y for x,y in zip(rec["values"],last)]
        prev[cell]=rec["values"]
        mean=[math.fsum(pair)/2 for pair in zip(*(g["v"] for g in rec["members"]))]
        groups[(b,t,role)].append((fraction(mean),fraction(delta)))
    out=[]
    for (b,t,role),row in sorted(groups.items()):
        predicted,actual=zip(*row)
        out.append(dict(base=b,rotation=t,teacher=role,steps=len(row),
                        median_gate_squared_step_fraction=median(actual),min_gate_fraction=min(actual),
                        max_gate_fraction=max(actual),max_abs_fraction_error=max(abs(x-y) for x,y in zip(predicted,actual))))
    return out

if __name__=="__main__":
    if len(sys.argv)!=2: sys.exit("usage: blocks.py formal_directory")
    rows=report(Path(sys.argv[1]))
    w=csv.DictWriter(sys.stdout,fieldnames=list(rows[0]),delimiter="\t",lineterminator="\n")
    w.writeheader();w.writerows(rows)
