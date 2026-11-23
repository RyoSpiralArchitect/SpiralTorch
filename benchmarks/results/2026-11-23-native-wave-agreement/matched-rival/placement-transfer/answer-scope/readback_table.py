"""Independent test-split contrasts and exploratory first-step geometry; no model calls."""
import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path


def check(ok,message):
    if not ok:
        raise ValueError(message)


def read(path,pin):
    raw=path.read_bytes()
    check(sha256(raw).hexdigest()==pin,"digest: "+path.name)
    return json.loads(raw)


def export(root,report,table,gradients):
    r=json.loads(report.read_bytes())
    sp=read(root/"spec.json",r["spec"])
    raw=read(root/"complete.json",r["evidence"])
    check(sp["formal"] and r["formal"],"not formal")
    held={q["key"]:q for q in sp["test"]}
    def records(name):
        f="scores/"+name+".json"
        payload=read(root/"attempt"/f,raw["files"][f])["rows"]
        return {k:payload[k]["token_nlls"] for k in held}
    def contrast(base,trained,role,idx):
        def avg(bit):
            ks=[k for k in held if held[k][role]==bit]
            return math.fsum((trained[k][1][idx]-base[k][1][idx])-(trained[k][0][idx]-base[k][0][idx]) for k in ks)/len(ks)
        return (avg(0)-avg(1))/2
    results=[]
    for seed in sp["bases"]:
        for geo in ("affine","wave"):
            origin=records("%d-%s"%(seed,geo))
            for order in sp["orders"]:
                order_key="%d-%s-%d"%(seed,geo,order)
                archived=r["rows"][order_key]["test"]
                out=dict(base=seed,geometry=geo,order=order)
                for scope in ("answer","through_stop"):
                    head,alt=(records("%d-%s-%s-%s-%d"%(seed,geo,scope,t,order)) for t in ("head","attractor"))
                    h,a,hc,ac=(contrast(origin,trained,role,0) for trained in (head,alt) for role in ("head","attractor"))
                    c=archived[scope]
                    for observed,k in ((h,"h"),(((h-a)-(hc-ac))/2,"S"),(contrast(origin,head,"head",1),"delta_stop")):
                        check(abs(observed-c[k])<1e-12,"split contrast "+order_key)
                    def win(rows,k):
                        n=held[k]["head"]
                        return rows[k][n][0]<rows[k][1-n][0]
                    check(sum(win(origin,k) for k in held)==c["before_wins"] and
                          sum(win(head,k) for k in held)==c["wins"],"decision readback")
                    for k in ("h","S","delta_stop","gain_gold","gain_choice","positive_pairs","before_wins","wins",
                              "before_ties","ties","switches","min_initial_abs_margin","max_margin_movement"):
                        out[scope+"_"+k]=c[k]
                out.update(gap_h=archived["gap_h"],gap_S=archived["gap_S"])
                results.append(out)
    with table.open("x",encoding="ascii",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]),delimiter="\t",lineterminator="\n")
        w.writeheader()
        w.writerows(results)

    # Only the initial update guarantees the *same* parameters for both teachers.
    path=root/"attempt"/"steps.jsonl"
    check(sha256(path.read_bytes()).hexdigest()==raw["files"]["steps.jsonl"],"raw steps")
    starts={}
    with path.open("rb") as f:
        for line in f:
            x=json.loads(line)
            if x["i"]==0:
                base,geo,scope,teacher,order=x["cell"].split("-")
                starts.setdefault((base,geo,scope,order),{})[teacher]=x
    def norm(values):
        return math.sqrt(math.fsum(z*z for z in values))
    pairs=[]
    for (base,geo,scope,order),group in sorted(starts.items()):
        x,y=group["head"],group["attractor"]
        check(x["row"]==y["row"] and x["ray"]["geometry"]["origin_id"]==y["ray"]["geometry"]["origin_id"],"not paired")
        if x["gold"]==y["gold"]:
            check(x["grad"]==y["grad"],"identical objectives produced different gradients")
            continue
        u,v=x["grad"],y["grad"]
        nu,nv=norm(u),norm(v)
        common=norm([(a+b)/2 for a,b in zip(u,v)])
        differential=norm([(a-b)/2 for a,b in zip(u,v)])
        cosine=math.fsum(a*b for a,b in zip(u,v))/(nu*nv) if nu and nv else None
        pairs.append(dict(base=base,geometry=geo,scope=scope,order=order,row=x["row"],cosine=cosine,
                          label_common_l2=common,label_difference_l2=differential))
    with gradients.open("x",encoding="ascii",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]),delimiter="\t",lineterminator="\n")
        w.writeheader()
        w.writerows(pairs)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("root","report","table","gradients"):
        p.add_argument(k,type=Path)
    a=p.parse_args()
    export(a.root,a.report,a.table,a.gradients)
