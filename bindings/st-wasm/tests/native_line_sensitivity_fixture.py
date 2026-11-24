"""Bounded, generated complete-line training. Rust decides and applies every update."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import runpy
import sys

prior = runpy.run_path(str(Path(__file__).with_name("native_line_prompt_fixture.py")))
mid, common, base, ray, prev, cases, dump, T = (prior[n] for n in ("mid","common","base","ray","prev","cases","dump","T"))
H, C, CEILING = 1/256, 2*T, 1/8
ORIGINAL = "20d430ff3734fb1440bcab0b38163e82112850e9c997b50fc00f52048e9d18bb"
BASIS = ("parameter_block","coordinate")
SELECT = ("gradient_norm","envelope")


def plans():
    for offset in range(3):
        modes = base["MODES"] if not offset else ("geometry",)
        for mode,basis,select in itertools.product(modes,BASIS,SELECT):
            p=json.loads(mid["prior"]["plan"](mode,"raw"))
            p["examples"]=cases[offset:]+cases[:offset]
            p["update"]["anchor_examples"]=[(i-offset)%3 for i in range(3)]
            p["update"]["direction"]=dict(kind="raw_then_sensitivity_ray",probe_l2=H,basis=basis,select=select)
            yield offset,mode,basis,select,dump(p)


def raw_result(attempt):
    k=mid["entry"](attempt)
    miss=k.get("raw_miss")
    if miss:
        return (False,miss["reason"],miss["trials"],miss["anchor_readouts"])
    assert attempt["status"]=="applied"
    return (True,k,attempt["step"]["parameter_ray"])


def read_prior(path, initial):
    data=path.read_bytes()
    assert hashlib.sha256(data).hexdigest()==ORIGINAL
    owners,records={},{}
    for r in map(json.loads,data.splitlines()):
        offset,mode,_,bank=r["case"].split("/")
        if bank!="012":
            continue
        a=r["attempt"]
        key=(mode,(r["cursor"]+int(offset))%3,owners.get(r["case"],initial))
        original=raw_result(a)
        assert records.setdefault(key,original)==original
        if a["status"]=="applied":
            s=a["step"]
            owners[r["case"]]=(s["post_lora_id"],s["post_geometry_id"])
    return records


def independent_mask(m,axis,source,distance):
    path,name,index=(axis[n] for n in ("path","tensor","coordinate"))
    grad=list(map(ray["f32"],source[path][name]))
    norm=abs(grad[index]) if index is not None else math.sqrt(math.fsum(x*x for x in grad))
    assert prev["near"](norm,axis["gradient_l2"])
    geom=not path.endswith(".attn.c_proj")
    record=json.loads(m.geometry_artifact().to_json() if geom else m.lora_to_json())
    if geom:
        place,=[p for p in record["placements"] if p["path"]==path]
        tensor,=[t for t in place["tensors"] if t["name"]==name and t["requires_grad"]]
        values=tensor["values"]
    else:
        values=record[name]
    if not norm:
        return None
    for i in range(len(grad)) if index is None else (index,):
        values[i]=ray["f32"](ray["f32"](values[i])+distance*(-grad[i]/norm))
    return (m.with_geometry(common["st"].GeometryAdapterArtifact.create_json(dump(record))) if geom
            else m.with_lora_json(dump(record)))


def predict(kl,norm):
    mean,peak=math.fsum(kl)/len(kl),max(kl)
    if mean==0:
        return None
    d=min(CEILING,H*math.sqrt(T/mean),H*math.sqrt(C/peak))
    return dict(radius=d,target_fraction=min(1.,(mean/T)*(d/H)**2),linear_loss_decrease=d*norm)


def verify_map(m,g,logits,sensitivity,mode,basis,selector,first):
    rows=sensitivity["axes"]
    blocks=ray["blocks"](m)
    active=sorted(k for k in blocks if mode=="joint" or ((mode=="lora")==k[0].endswith(".attn.c_proj")))
    expected=[(p,n,i) for p,n in active for i in ((None,) if basis=="parameter_block" else range(len(blocks[p,n])))]
    assert [(a["path"],a["tensor"],a["coordinate"]) for a in rows]==expected
    scores=[]
    for i,a in enumerate(rows):
        path,name,coord=expected[i]
        v=list(map(ray["f32"],g[path][name]))
        norm=abs(v[coord]) if coord is not None else math.sqrt(math.fsum(x*x for x in v))
        assert prev["near"](norm,a["gradient_l2"])
        k=a["probe_per_anchor_kl"]
        if not norm:
            assert k is None and a["prediction"] is None
            continue
        assert len(k)==3
        if first:
            actual=independent_mask(m,a,g,H)
            measured=prev["readout"](logits,ray["logits"](actual))
            assert all(prev["near"](x,y) for x,y in zip(measured,k)), (a,measured)
        pred=predict(k,norm)
        if pred is None:
            assert a["prediction"] is None
        else:
            assert all(prev["near"](x,a["prediction"][n]) for n,x in pred.items())
        if selector=="gradient_norm":
            scores.append(((norm,),i))
        elif pred is not None:
            f=pred["target_fraction"]
            scores.append(((f>=.98, 0 if f>=.98 else f, pred["linear_loss_decrease"]),i))
    chosen=max(scores,key=lambda s:s[0])[1] if scores else None
    assert chosen==sensitivity["selected_axis"]
    return None if chosen is None else rows[chosen]


def run(old,out):
    origin,_=common["host"](common["HOST"])
    initial=common["owners"](origin)
    originals=read_prior(old,initial)
    configs=list(plans())
    assert len(configs)==20
    results={}
    with out.with_suffix(".jsonl").open("x") as journal:
        for offset,mode,basis,select,plan in configs:
            key="/".join([str(offset),mode,basis,select])
            s,z=mid["Session"](origin,plan),ray["logits"](origin)
            horizon=1 if offset else 6
            events=[]
            for i in range(horizon):
                loss=(offset+i)%3
                m,ck=s.model,s.checkpoint_with_id()
                pre=common["owners"](m)
                a=s.attempt_step()
                journal.write(dump(dict(case=key,cursor=i,attempt=a))+"\n")
                journal.flush()
                ok=a["status"]=="applied"
                k=mid["entry"](a)
                raw=raw_result(a)
                orig=(mode,loss,pre)
                if orig in originals:
                    assert raw==originals[orig], "raw stage changed from the saved same-state reference"
                e=dict(applied=ok,loss_case=loss,raw="raw_miss" not in k,prior_checked=orig in originals,
                       readouts=k["anchor_readouts"],trials=len(k["trials"]))
                if not e["raw"]:
                    miss=k["raw_miss"]
                    base["check_trials"](dict(k,**{n:miss[n] for n in ("reason","trials","anchor_readouts")}),"capped",False)
                    info=k["sensitivity"]
                    assert info["basis"]==basis and info["select"]==select and info["probe_l2"]==H and "direction" not in k
                    g=common["grad"](m,cases[loss],mode).parameter_gradients()
                    chosen=verify_map(m,g,z,info,mode,basis,select,i==0)
                    count=sum(a["probe_per_anchor_kl"] is not None for a in info["axes"])
                    assert info["probe_readouts"]==3*count
                    assert k["anchor_readouts"]==miss["anchor_readouts"]+info["probe_readouts"]+3*len(k["trials"])
                    e.update(selected=info["selected_axis"],probes=count,initial_probe_checks=count if i==0 else 0,
                             raw_trials=len(miss["trials"]))
                    if chosen is None:
                        assert not ok and not k["trials"] and k["reason"]=="no measurable sensitivity direction"
                    else:
                        base["check_trials"](dict(k,anchor_readouts=3*(1+len(k["trials"]))),"capped",ok)
                else:
                    assert ok and "sensitivity" not in k and "direction" not in k
                    base["check_trials"](k,"capped",True)
                if not ok:
                    assert a["next_update"]==i+1 and a["example_index"]==i%3 and s.checkpoint_with_id()==ck
                    e["reason"]=k["reason"]
                else:
                    step=a["step"]
                    new=s.model
                    assert (step["example_index"],step["update"])==(i%3,i+1)
                    after=common["owners"](new)
                    assert pre==(step["pre_lora_id"],step["pre_geometry_id"]) and after==(step["post_lora_id"],step["post_geometry_id"])
                    z2=ray["logits"](new)
                    kl=prev["readout"](z,z2)
                    mean=base["check_candidate"](kl,k["trials"][-1])
                    assert .98*T<=mean<=T and max(kl)<=C and all(prev["near"](x,y) for x,y in zip(kl,k["per_anchor_kl"]))
                    if not e["raw"]:
                        rebuilt=independent_mask(m,chosen,g,k["trials"][-1]["requested_l2"])
                        assert common["owners"](rebuilt)==after and ray["blocks"](rebuilt)==ray["blocks"](new)
                        x,y=ray["blocks"](m),ray["blocks"](new)
                        dot=math.fsum((b-a)*ray["f32"](v) for (path,n),values in x.items() if path in g
                                      for a,b,v in zip(values,y[path,n],g[path][n]))
                        assert dot<0 and prev["near"](dot,info["source_loss_dot_step"])
                        assert step["parameter_ray"]["method"]=="selected_parameter_sensitivity_ray_f32.v1"
                    e.update(loss=step["loss"],kl=kl,owners=after)
                    z=z2
                events.append(e)
                if not ok:
                    break
            assert s.completed_updates==sum(e["applied"] for e in events) and s.exhausted==(s.completed_updates==6)
            results[key]=dict(plan=plan,events=events,completed=s.completed_updates,endpoint=z,scores=common["losses"](s.model))
    common["save"](out,dict(schema="spiraltorch.synthetic_line_sensitivity.v1",previous=ORIGINAL,cases=results,pretrained_calls=0))
    print(dump({n:c["completed"] for n,c in results.items()}))


if __name__=="__main__":
    if len(sys.argv)!=3 or not __debug__:
        sys.exit("PINNED_PROMPT_JOURNAL NEW_RESULT (assertions required)")
    run(*map(Path,sys.argv[1:]))
