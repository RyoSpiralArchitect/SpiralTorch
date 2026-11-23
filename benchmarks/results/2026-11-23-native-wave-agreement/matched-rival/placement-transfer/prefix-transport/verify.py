"""Recompute updates and held-out gates without importing an ML library or reading a model into a decoder."""
import argparse
import csv
import json
import math
from pathlib import Path
import runpy
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
u = SimpleNamespace(**runpy.run_path(str(HERE/"study.py")))
a = SimpleNamespace(**runpy.run_path(str(u.PREV/"analyze.py")))
d, k, s, need = u.d, u.k, u.s, u.need


def gate(records):
    need(bool(records), "missing order")
    prob=all(min(v["h"],v["S"],v["gap_h"],v["gap_S"])>.001 and v["positive_pairs"]>=3 for v in records)
    return dict(probability=prob,decision=prob and all(v["wins"]>max(v["wins_before"],v["identity_wins"]) for v in records))


def verify(root):
    raw=(root/"spec.json").read_bytes();sp=json.loads(raw)
    need(d.encoded(u.plan(sp["formal"],sp["extension"]))==raw,"source or protocol drift")
    complete=k.read(root/"complete.json");out=root/"attempt"; ids=k.identities(out)
    starts=[s.key(b,t) for b in sp["bases"] for t in u.TURNS]
    cells=[s.key(*c) for c in u.cells(sp)]
    expected={sub+"/"+name+".json" for name in starts+cells for sub in ("weights","logits")}
    expected.update(["steps.jsonl","hosts.json"]+["frozen-%d.json"%b for b in sp["bases"]])
    need(complete==dict(spec=d.digest(raw),files=ids) and set(ids)==expected,"archive membership or digest")
    hosts=k.read(out/"hosts.json")
    need(set(hosts)==set(map(str,sp["bases"])) and len({h["base"] for h in hosts.values()})==len(hosts),"base identities")
    allrows=sp["train"]+sp["test"]; keys={r["key"] for r in allrows}
    snapshots,weights,scored={}, {}, {}
    for b in sp["bases"]:
        host=hosts[str(b)]
        need(d.digest((root/("base-%d.safetensors"%b)).read_bytes())==sp["weights"][str(b)] and
             json.loads(host["lora_weights"])==dict(s.fixtures_cached()[1],base_model_identity=host["original_base"]) and
             d.owner_id(host["lora_weights"].encode(),d.REF)==host["frozen_lora"], "changed base or companion")
        ref=k.read(out/("frozen-%d.json"%b))
        need(set(ref)==keys,"reference coverage")
        for name in starts+cells:
            parts=name.split("-")
            if parts[0]!=str(b): continue
            raww=(out/"weights"/(name+".json")).read_bytes()
            vec=u.coordinates(raww,parts[1],host);pin=d.identity(raww)
            weights[name]=(pin,vec)
            v=k.read(out/"logits"/(name+".json"))
            z=snapshots[name]=v["rows"]
            need(v["owner"]==pin and set(z)==keys,"snapshot coverage/owner")
            for key in keys:
                data=u.bits(z[key])
                if name in starts: need(vec==[0.]*6 and data==u.bits(ref[key]),"origin is not identical to frozen LoRA")
            scored[name]={}
            for split in ("train","test"):
                rows=sp[split]
                scored[name][split]={teacher:a.metrics([dict(r,head=r[teacher]) for r in rows],z) for teacher in d.TEACHERS}
    n=0;worst=0.
    with (out/"steps.jsonl").open("rb") as f:
        for b,t,teacher,order in u.cells(sp):
            start=s.key(b,t);cell=s.key(b,t,teacher,order)
            prev,vec=weights[start]
            for i,pair in enumerate(k.schedule(sp["train"],order,"pair")*sp["epochs"]):
                rec=json.loads(next(f))
                need((rec["cell"],rec["i"],rec["keys"])==(cell,i,[r["key"] for r in pair]) and len(rec["members"])==2,"step order")
                for r,g in zip(pair,rec["members"]):
                    k.check_grad(g,1.,r[teacher],snapshots[start][r["key"]] if i==0 else None)
                _,err=k.check_step(rec,vec,prev,hosts[str(b)])
                n+=1;worst=max(worst,err);prev,vec=rec["id"],rec["values"]
            need((prev,vec)==weights[cell],"unsealed endpoint")
        need(f.read(1)==b"","extra optimizer steps")
    counts=("wins","ties","global_wins","gold_nll","menu_nll","menu_mass")
    records=[];by_base={}
    for b in sp["bases"]:
        for o in sp["orders"]:
            for split in ("train","test"):
                baseline=None
                for t in u.TURNS:
                    origin=scored[s.key(b,t)][split]["head"]
                    heads=scored[s.key(b,t,"head",o)][split]
                    distract=scored[s.key(b,t,"attractor",o)][split]
                    # For S, evaluate both endpoints with the same two fixed grammatical signs.
                    e=a.effect(origin,heads["head"],distract["head"])
                    if t=="identity": baseline=e
                    row=dict(base=b,order=o,split=split,rotation=t,gap_h=e["h"]-baseline["h"],gap_S=e["S"]-baseline["S"],
                             identity_wins=baseline["wins"],**e)
                    for teacher,trained in (("head",heads),("attractor",distract)):
                        row.update({teacher+"_"+key:trained[teacher][key] for key in counts})
                    records.append(row)
        for t in u.TURNS[1:]:
            by_base[s.key(b,t)]=gate([r for r in records if (r["base"],r["rotation"],r["split"])==(b,t,"test")])
    passes={t:{g:[b for b in sp["bases"] if by_base[s.key(b,t)][g]] for g in ("probability","decision")} for t in u.TURNS[1:]}
    need(n==len(cells)*len(sp["train"])*sp["epochs"]//2 and len(records)==len(sp["bases"])*len(sp["orders"])*6,"study coverage")
    return dict(schema="spiraltorch.prefix_transport_results.v1",formal=sp["formal"],spec=d.digest(raw),
                manifest=d.digest((root/"complete.json").read_bytes()), endpoints=len(cells), updates=n, vjps=2*n,
                snapshot_forwards=(len(starts)+len(cells))*len(allrows), reference_forwards=len(hosts)*len(allrows),
                worst_relative_motion_error=worst, by_base=by_base, qualifying_bases=passes, rows=records)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path);p.add_argument("result",type=Path);p.add_argument("table",type=Path)
    q=p.parse_args()
    res=verify(q.root)
    print(d.save(q.result,res))
    with q.table.open("x",encoding="ascii",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(res["rows"][0]),delimiter="\t",lineterminator="\n")
        w.writeheader();w.writerows(res["rows"])
