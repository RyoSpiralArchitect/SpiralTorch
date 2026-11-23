"""Check all paths and compare the frozen conditions from captured bytes, without a model."""
import argparse
import json
import math
from pathlib import Path
import runpy
import types

s=types.SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("study.py"))))
d,need=s.d,s.need


def mean(values):
    return math.fsum(values)/len(values)


def components(rows, costs):
    head,attr,stop,gold,choice=[],[],[],[],[]
    margins={}
    by_pair={}
    wins=ties=0
    for r in rows:
        c=costs[r["key"]]
        content=[v[0] for v in c["token_nlls"]]
        end=[v[1] for v in c["token_nlls"]]
        m=content[1]-content[0]
        hs,ats=1-2*r["head"],1-2*r["attractor"]
        margins[r["key"]]=m
        head.append(hs*m)
        attr.append(ats*m)
        stop.append(hs*(end[1]-end[0]))
        gold.append(content[r["head"]])
        # Full-vocabulary normalisation cancels in the conditional odds.
        x=-hs*m
        choice.append(max(0.,x)+math.log1p(math.exp(-abs(x))))
        by_pair.setdefault(r["frame"],[]).append(hs*m)
        wins+=int(hs*m>0)
        ties+=int(m==0)
    return dict(H=mean(head), A=mean(attr), stop=mean(stop), complete=mean(head)+mean(stop),
                gold=mean(gold),choice=mean(choice),wins=wins,ties=ties,margins=margins,
                pairs={k:mean(v) for k,v in by_pair.items()})


def effects(before, head, ctrl):
    h=head["H"]-before["H"]
    S=((head["H"]-head["A"])-(ctrl["H"]-ctrl["A"]))/2
    old=before["margins"]
    movements=[abs(head["margins"][k]-v) for k,v in old.items()]
    min_margin=min(map(abs,old.values()))
    changed=sum(((v>0)-(v<0))!=((head["margins"][k]>0)-(head["margins"][k]<0)) for k,v in old.items())
    need(max(movements)>=min_margin or not changed,"impossible switch")
    return dict(h=h,S=S,B=head["H"]-ctrl["H"],positive_pairs=sum(head["pairs"][k]>v for k,v in before["pairs"].items()),
                delta_stop=head["stop"]-before["stop"],delta_line=head["complete"]-before["complete"],
                gain_gold=before["gold"]-head["gold"],gain_choice=before["choice"]-head["choice"],
                before_wins=before["wins"],wins=head["wins"],before_ties=before["ties"],ties=head["ties"],
                switches=changed,min_initial_abs_margin=min_margin,max_margin_movement=max(movements))


def compare(answer, through):
    return dict(answer=answer,through_stop=through,gap_h=answer["h"]-through["h"],gap_S=answer["S"]-through["S"])


def gates(orders):
    need(bool(orders),"no matched orders")
    probability=all(min(r["answer"]["h"],r["answer"]["S"],r["gap_h"],r["gap_S"])>.001 and
                    r["answer"]["positive_pairs"]>=3 for r in orders)
    decision=probability and all(r["answer"]["wins"]>max(r["answer"]["before_wins"],r["through_stop"]["wins"]) for r in orders)
    return dict(probability=probability,decision=decision)


def wave_gate(w,a):
    need(len(w)==len(a),"orders not paired")
    return gates(w)["probability"] and all(min(x[k]-y[k] for k in ("gap_h","gap_S"))>.001 for x,y in zip(w,a))


def inspect(root):
    def read(path):
        return json.loads(path.read_bytes())
    raw_plan=(root/"spec.json").read_bytes()
    sp=json.loads(raw_plan)
    need(s.encoded(s.spec(sp["formal"],sp["extension"]))==raw_plan,"spec/source drift")
    for seed in sp["bases"]:
        need(s.digest((root/("base-%d.safetensors"%seed)).read_bytes())==sp["weights"][str(seed)],"new base digest")
    manifest=read(root/"complete.json")
    out=root/"attempt"
    need(manifest["spec"]==s.digest(raw_plan),"wrong completion")
    starts=[s.key(b,g) for b in sp["bases"] for g in s.GEOS]
    ends=[s.key(*c) for c in s.cells(sp)]
    names=starts+ends
    files={"steps.jsonl","hosts.json"} | {k+"/"+n+".json" for k in ("weights","scores") for n in names}
    if not sp["formal"]:
        files.update("finite-"+k+".json" for k in starts)
    need(set(manifest["files"])==files=={p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()},
         "incomplete or extra raw artifact")
    for name,pin in manifest["files"].items():
        need(s.digest((out/name).read_bytes())==pin,"raw digest: "+name)
    hosts=read(out/"hosts.json")
    for b,host in hosts.items():
        lora=host["lora_weights"]
        need(b in map(str,sp["bases"]) and d.owner_id(lora.encode(),d.REF)==host["frozen_lora"] and
             json.loads(lora)==dict(s.fixtures_cached()[1],base_model_identity=host["original_base"]),"LoRA setup")
    need(len(set(h["base"] for h in hosts.values()))==len(sp["bases"]),"duplicate frozen host")
    states,costs,metrics,travel={},{},{},{}
    worst_direct=worst_direction=worst_length=0.
    for name in names:
        b,geo=name.split("-")[:2]
        raw=(out/"weights"/(name+".json")).read_bytes()
        pin=d.identity(raw)
        states[name]=pin,s.coordinates(raw,geo,hosts[b]["base"])
        rec=read(out/"scores"/(name+".json"))
        need(pin==rec["owner"] and set(rec["rows"])=={r["key"] for r in sp["train"]+sp["test"]},"snapshot rows or owner")
        costs[name]=rec["rows"]
        for c in rec["rows"].values():
            s.P.verify_costs(c)
        worst_direct=max(worst_direct,s.P.verify_sentinel(rec["sentinel"],rec["rows"][sp["train"][0]["key"]]))
        metrics[name]={k:components(sp[k],rec["rows"]) for k in ("train","test")}
    reference=states[starts[0]][1]
    need(all(states[n][1]==reference for n in starts),"initial geometry coordinates differed")
    nsteps=0
    with (out/"steps.jsonl").open("rb") as journal:
        for seed,geo,scope,teacher,order in s.cells(sp):
            origin=s.key(seed,geo)
            cell=s.key(seed,geo,scope,teacher,order)
            prev,coords=states[origin]
            initial=coords
            lengths=[]
            for i,row in enumerate(d.schedule(sp["train"],order)*sp["epochs"]):
                r=json.loads(next(journal))
                need((r["cell"],r["row"],r["i"],r["gold"])==(cell,row["key"],i,row[teacher]),"order or target drift")
                rep=r["report"]
                if scope=="answer":
                    need((rep["mix"],rep["target_tokens"],rep["contrasted_tokens"],rep["rival_ids"])==(.5,1,1,1),"mask")
                    value=.5*(rep["full_ce"]+rep["declared_ce"])
                else:
                    need(rep["target_counts"]==[2,2] and (rep["mix"],rep["stop_id"],rep["gold_index"])==(.5,0,r["gold"]),
                         "wrong terminated menu")
                    value=.5*(rep["menu_nll"]+rep["gold_mean_nll"])
                need(abs(r["loss"]-value)<3e-6 and r["loss"]==rep["loss"],"invalid mixture")
                if not i:
                    need(abs(s.loss(costs[origin][row["key"]],r["gold"],scope)-value)<3e-6,"origin loss mismatch")
                length,direction=s.verify_step(r,coords,prev,r["id"])
                worst_length=max(worst_length,abs(length/d.D-1))
                worst_direction=max(worst_direction,direction)
                lengths.append(length)
                prev,coords=r["id"],r["values"]
                nsteps+=1
            need(states[cell]==(prev,coords),"final artifact not the trained endpoint")
            travel[cell]=dict(path=math.fsum(lengths),endpoint=d.distance(initial,coords))
        need(journal.readline()==b"","excess updates")
    rows={}
    decision={}
    for seed in sp["bases"]:
        for geo in s.GEOS:
            seq=[]
            for order in sp["orders"]:
                contrasts={}
                for split in ("test","train"):
                    baseline=metrics[s.key(seed,geo)][split]
                    results=[]
                    for scope in s.SCOPES:
                        head,ctrl=(metrics[s.key(seed,geo,scope,t,order)][split] for t in d.TEACHERS)
                        results.append(effects(baseline,head,ctrl))
                    contrasts[split]=compare(*results)
                rows[s.key(seed,geo,order)]=contrasts
                seq.append(contrasts["test"])
            decision[s.key(seed,geo)]=gates(seq)
    qualifying={g:{k:[b for b in sp["bases"] if decision[s.key(b,g)][k]] for k in ("probability","decision")} for g in s.GEOS}
    wave=[b for b in sp["bases"] if wave_gate(*[[rows[s.key(b,g,o)]["test"] for o in sp["orders"]] for g in s.GEOS])]
    return dict(schema="spiraltorch.answer_scope_results.v1",formal=sp["formal"],spec=manifest["spec"],
                evidence=s.digest((root/"complete.json").read_bytes()),base_hashes=sp["weights"],native_sha256=sp["extension"],
                updates=nsteps,full_menu_scores=len(names)*(len(sp["train"])+len(sp["test"])),direct_forwards=3*len(names),
                checkpoints=len(ends),worst_direct_cost_error=worst_direct,worst_length_relative_error=worst_length,
                worst_direction_relative_error=worst_direction,rows=rows,qualifying_bases=qualifying,wave_specific_bases=wave,
                probability_transfer={g:len(v["probability"])>=2 for g,v in qualifying.items()},
                decision_transfer={g:len(v["decision"])>=2 for g,v in qualifying.items()},wave_specific=len(wave)>=2,paths=travel)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("new_output",type=Path)
    a=p.parse_args()
    print(s.save(a.new_output,inspect(a.root)))
