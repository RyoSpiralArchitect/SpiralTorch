"""Rebuild the study from sealed weights, step journals and full logits; never load a model."""
import argparse
import csv
import json
import math
from pathlib import Path
import runpy
from statistics import median
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
k=SimpleNamespace(**runpy.run_path(str(HERE/"design.py")))
probe=SimpleNamespace(**runpy.run_path(str(HERE/"probe.py")))
s,d,need=k.s,k.d,k.need


def span(values):
    good=[v for v in values if v is not None]
    return dict(n=len(good),undefined=len(values)-len(good),min=min(good) if good else None,
                median=median(good) if good else None,max=max(good) if good else None)


def metrics(rows,z):
    hs,ats,ce,bce,mass=[],[],[],[],[]
    wins=ties=global_wins=0
    frames={}
    margins={}
    for r in rows:
        a=z[r["key"]]
        g=r["head"]
        margin=a[7]-a[8]
        m=(1-2*g)*margin
        hs.append(m)
        ats.append((1-2*r["attractor"])*margin)
        frames.setdefault(r["frame"],[]).append(m)
        margins[r["key"]]=margin
        wins+=m>0
        ties+=m==0
        global_wins+=all(a[k.TOK[g]]>val for i,val in enumerate(a) if i!=k.TOK[g])
        c,b,_=k.costs(a,g,1.)
        ce.append(c); bce.append(b); mass.append(math.exp(b-c))
    H,A,C,Q,M=k.avg(list(zip(hs,ats,ce,bce,mass)))
    return dict(H=H,A=A,gold_nll=C,menu_nll=Q,menu_mass=M,wins=wins,ties=ties,global_wins=global_wins,
                frames={f:math.fsum(v)/len(v) for f,v in frames.items()},margins=margins)


def effect(origin,h,a):
    return dict(h=h["H"]-origin["H"],S=((h["H"]-h["A"])-(a["H"]-a["A"]))/2,B=h["H"]-a["H"],
                gain_nll=origin["gold_nll"]-h["gold_nll"],gain_menu=origin["menu_nll"]-h["menu_nll"],
                mass_before=origin["menu_mass"],mass_after=h["menu_mass"],
                wins_before=origin["wins"],wins=h["wins"],ties_before=origin["ties"],ties=h["ties"],
                global_before=origin["global_wins"],global_wins=h["global_wins"],
                positive_pairs=sum(h["frames"][f]>x for f,x in origin["frames"].items()),
                switches=sum(((m>0)-(m<0))!=((h["margins"][f]>0)-(h["margins"][f]<0))
                             for f,m in origin["margins"].items()))


def qualify(rows):
    need(bool(rows),"no matched orders")
    passed=all(min(r["pure"]["h"],r["pure"]["S"],r["gap_h"],r["gap_S"])>.001
               and r["pure"]["positive_pairs"]>=3 for r in rows)
    return dict(probability=passed,decision=passed and all(r["pure"]["wins"]>max(r["mixed"]["wins"],r["pure"]["wins_before"])
                                                           for r in rows))


def prediction(work,H,A):
    head,other=work
    if head is None or other is None:
        return None
    return dict(h=k.dot(H,head),S=k.dot(k.sub(H,A),k.sub(head,other))/2)


def frozen_ray(steps,bank,scope,role,epochs):
    pieces=[]
    for rows in steps:
        unit=k.unit(k.avg([bank[r["key"]][scope][r[role]]["v"] for r in rows]))
        if unit is None:
            return None
        pieces.append(k.scaled(unit,-len(rows)*d.D*epochs))
    return [math.fsum(x) for x in zip(*pieces)]


def inspect(root):
    sp=k.read(root/"spec.json")
    spec_pin=d.digest((root/"spec.json").read_bytes())
    need(d.encoded(k.plan(sp["formal"]))==(root/"spec.json").read_bytes(),"sources/spec mismatch")
    def get(path,pin):
        raw=path.read_bytes()
        need(d.digest(raw)==pin,"digest: "+path.name)
        return json.loads(raw)
    done=k.read(root/"completed.json")
    need(done["spec"]==spec_pin,"completion plan")
    seal=get(root/"endpoints.json",done["sealed"])
    out=root/"attempt"
    need(seal==dict(spec=spec_pin,files=k.seal_snapshot(sp,out)),"changed sealed endpoint")
    te_bank=get(root/"held-origin-bank.json",done["held_bank"])
    probes=get(root/"probes.json",done["probes"])
    tr_bank=k.read(out/"train-bank.json")
    hosts=k.read(out/"hosts.json")
    for b in sp["bases"]:
        h=hosts[str(b)]
        raw=h["lora_weights"]
        need(d.digest((root/("base-%d.safetensors"%b)).read_bytes())==sp["weights"][str(b)]
             and json.loads(raw)==dict(s.fixtures_cached()[1],base_model_identity=h["original_base"]) and
             h["frozen_lora"]==d.owner_id(raw.encode(),d.REF),"base archive/LoRA")
    need(len({h["base"] for h in hosts.values()})==len(sp["bases"]),"nonunique frozen base")
    starts=[s.key(b,g) for b in sp["bases"] for g in s.GEOS]
    ends=[s.key(*a) for a in k.specs(sp)]
    need(set(tr_bank)==set(te_bank)==set(starts) and set(probes)==set(starts)|{"position","bag"},"incomplete controls")
    z,states,measured={},{},{}
    nscore=0
    for name in starts+ends:
        b,geo=name.split("-")[:2]
        raw=(out/"weights"/(name+".json")).read_bytes()
        owner=d.identity(raw)
        states[name]=owner,s.coordinates(raw,geo,hosts[b]["base"])
        src=k.read(out/"logits"/(name+".json"))
        need(src["owner"]==owner and set(src["rows"])=={r["key"] for r in sp["train"]+sp["test"]} and
             all(len(v)==11 and all(math.isfinite(x) and x==d.f32(x) for x in v) for v in src["rows"].values()),
             "logit snapshot")
        z[name]=src["rows"]
        measured[name]={split:metrics(sp[split],z[name]) for split in ("train","test")}
        nscore+=len(z[name])
    need(all(states[n][1]==states[starts[0]][1] for n in starts),"unmatched initial coordinates")
    bankstats,held_slopes={},{}
    nbank_train=nbank_test=0
    for name in starts:
        stats=[]
        for split,bank in (("train",tr_bank),("test",te_bank)):
            slopes=[]
            need(set(bank[name])=={r["key"] for r in sp[split]},"missing local VJP")
            for r in sp[split]:
                entry=bank[name][r["key"]]
                need(set(entry)==(set(k.MIX) if split=="train" else {"pure"}),"bank scopes")
                slope,checks=k.check_bank(entry,z[name][r["key"]],split=="train")
                slopes.append(slope)
                if split=="train":
                    stats.append(checks); nbank_train+=4
                else:
                    nbank_test+=2
            if split=="test":
                held_slopes[name]=[k.avg([k.scaled(v,1-2*r[t]) for v,r in zip(slopes,sp["test"])]) for t in d.TEACHERS]
        bankstats[name]={f:span([row[f] for row in stats]) for f in stats[0]}
    nsteps=nvjps=0
    worst=0.
    displacements,paths={},{}
    with (out/"steps.jsonl").open("rb") as log:
        for b,geo,scope,pack,teacher,order in k.specs(sp):
            start=s.key(b,geo); cell=s.key(b,geo,scope,pack,teacher,order)
            old,before=states[start]
            lengths=[]
            for i,rows in enumerate(k.schedule(sp["train"],order,pack)*sp["epochs"]):
                r=json.loads(next(log))
                need(r["cell"]==cell and r["i"]==i and r["keys"]==[q["key"] for q in rows] and
                     len(r["members"])==len(rows),"training order")
                for q,g in zip(rows,r["members"]):
                    k.check_grad(g,k.MIX[scope],q[teacher],None if i else z[start][q["key"]])
                    if not i:
                        need(g==tr_bank[start][q["key"]][scope][q[teacher]],"different origin probe/training VJP")
                    nvjps+=1
                length,error=k.check_step(r,before,old,hosts[str(b)])
                lengths.append(length); worst=max(worst,error); nsteps+=1
                before,old=r["values"],r["id"]
            need(states[cell]==(old,before),"broken endpoint chain")
            displacements[cell]=k.sub(before,states[start][1])
            paths[cell]=dict(endpoint=k.norm(displacements[cell]),length=math.fsum(lengths))
        need(not log.readline(),"extra training updates")
    rows,by_base,summary={}, {}, {}
    for b,geo in product2(sp["bases"],s.GEOS):
        start=s.key(b,geo)
        H,A=held_slopes[start]
        for pack in k.PACK:
            per_order=[]
            for order in sp["orders"]:
                row={}
                for split in ("train","test"):
                    scope_effect={}
                    for scope in k.MIX:
                        keys=[s.key(b,geo,scope,pack,t,order) for t in d.TEACHERS]
                        e=effect(measured[start][split],*[measured[name][split] for name in keys])
                        if split=="test":
                            schedule=k.schedule(sp["train"],order,pack)
                            v=[frozen_ray(schedule,tr_bank[start],scope,t,sp["epochs"]) for t in d.TEACHERS]
                            e["frozen"]=prediction(v,H,A)
                            e["endpoint_linear"]=prediction([displacements[name] for name in keys],H,A)
                            e["head_endpoint_l2"]=paths[keys[0]]["endpoint"]
                            e["head_path_l2"]=paths[keys[0]]["length"]
                        scope_effect[scope]=e
                    mix,pure=scope_effect["mixed"],scope_effect["pure"]
                    row[split]=dict(scope_effect,gap_h=pure["h"]-mix["h"],gap_S=pure["S"]-mix["S"])
                rows[s.key(b,geo,pack,order)]=row
                per_order.append(row["test"])
            by_base[s.key(b,geo,pack)]=qualify(per_order)
    for pack in k.PACK:
        summary[pack]={geo:{gate:[b for b in sp["bases"] if by_base[s.key(b,geo,pack)][gate]]
                            for gate in ("probability","decision")} for geo in s.GEOS}
        summary[pack]["wave_specific"]=[b for b in summary[pack]["wave"]["probability"]
            if all(rows[s.key(b,"wave",pack,o)]["test"][metric]-rows[s.key(b,"affine",pack,o)]["test"][metric]>.001
                   for o in sp["orders"] for metric in ("gap_h","gap_S"))]
    interaction={}
    for b,geo in product2(sp["bases"],s.GEOS):
        for o in sp["orders"]:
            pair,single=[rows[s.key(b,geo,p,o)]["test"] for p in ("pair","single")]
            interaction[s.key(b,geo,o)]={col:pair[col]-single[col] for col in ("gap_h","gap_S")}
    probe_summary={}
    for name,ref in probes.items():
        need(set(ref)==set(d.TEACHERS),"probe teachers")
        probe_summary[name]={}
        for role,v in ref.items():
            probe.verify("logits" if name in starts else name,sp,z.get(name),role,v)
            probe_summary[name][role]={q:v[q+"_counts"] for q in ("test","train")}
    return dict(schema="spiraltorch.conditional_update_results.v1",formal=sp["formal"],spec=spec_pin,
                completed=d.digest((root/"completed.json").read_bytes()),endpoints_digest=done["sealed"],
                updates=nsteps,training_vjps=nvjps,train_origin_vjps=nbank_train,held_origin_vjps=nbank_test,
                forwards=nscore,checkpoints=len(ends),probe_updates=64*len(probes)*len(d.TEACHERS),
                worst_relative_motion_error=worst,origin_checks=bankstats,rows=rows,by_base=by_base,
                qualifying_bases=summary,schedule_interaction=interaction,probes=probe_summary)


def product2(xs,ys):
    return ((x,y) for x in xs for y in ys)


def tables(r,out):
    out.mkdir(mode=0o700)
    def write(name,rows):
        with (out/(name+".tsv")).open("x",encoding="ascii",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter="\t",lineterminator="\n")
            w.writeheader(); w.writerows(rows)
    pairs=[]
    for label,record in r["rows"].items():
        b,g,p,o=label.split("-")
        e=record["test"]
        item=dict(base=b,geometry=g,update=p,order=o,gap_h=e["gap_h"],gap_S=e["gap_S"])
        for scope in k.MIX:
            for metric,value in e[scope].items():
                if metric in ("frozen","endpoint_linear"):
                    for col in ("h","S"):
                        item[scope+"_"+metric+"_"+col]=value[col] if value is not None else None
                else:
                    item[scope+"_"+metric]=value
        pairs.append(item)
    write("per-order",pairs)
    write("probes",[dict(features=name,teacher=t,**{split+"_"+k:v for split,q in counts.items() for k,v in q.items()})
                    for name,ls in r["probes"].items() for t,counts in ls.items()])
    write("origin-gradients",[dict(origin=n,**{k+"_"+f:x for k,v in agg.items() for f,x in v.items()}) for n,agg in r["origin_checks"].items()])
    write("schedule-interaction",[dict(comparison=key,**value) for key,value in r["schedule_interaction"].items()])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path); p.add_argument("output",type=Path)
    p.add_argument("--tables",type=Path)
    a=p.parse_args()
    result=inspect(a.root)
    print(d.save(a.output,result))
    if a.tables:
        tables(result,a.tables)
