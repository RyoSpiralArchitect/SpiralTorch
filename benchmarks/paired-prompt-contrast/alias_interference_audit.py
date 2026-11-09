"""Model-free validation of class-mean gradients, realized displacements and finite training losses."""
import json
import math
import struct
from pathlib import Path
from shared_prefix_study import SITE, obj, put, read, require


NAMES=("on","off","idle")
def f32(x): return struct.unpack("<f",struct.pack("<f",x))[0]
def dot(a,b): return math.fsum(x*y for x,y in zip(a,b,strict=True))
def norm(v): return math.sqrt(dot(v,v))
def avg(xs): return math.fsum(xs)/len(xs)
def mean_vec(vs): return [avg(col) for col in zip(*vs,strict=True)]
def cosine(a,b):
    n=norm(a)*norm(b)
    return max(-1.,min(1.,dot(a,b)/n)) if n else None
def close(a,b): return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-12+2e-10*abs(b)


def coordinates(path,pin):
    a=obj(path,pin,100000);site,=a["placements"];require(site["path"]==SITE,"unexpected placement")
    train={}
    for t in site["tensors"]:
        if t["requires_grad"]:
            require(t["role"]=="parameter" and t["shape"]==[32],"unexpected coordinates")
            train[t["name"]]=[f32(v) for v in t["values"]];t["values"]=None
    require(train.keys()=={"bias","gate"},"trainable schema changed")
    return train["bias"]+train["gate"],a


def directional_gate(point,sign,plan):
    p,actual=point["predicted"],point["measured"]
    return (sign*p>plan["margin"] and sign*actual>plan["margin"] and
            abs(actual-p)<=max(plan["margin"],plan["linear_rel"]*abs(p)))


def audit(planned,ppin,measured,mpin,out):
    pl=obj(planned/"plan.json",ppin); m=obj(measured/"measured.json",mpin)
    require(m["plan"]==ppin and m["origins"]==pl["origins"] and
            m["counts"]==dict(sealed_gradients=36,independent_proposals=20,extra_forwards=360),"measurement domain")
    raw=[json.loads(x) for x in read(measured/"observations.jsonl",m["observations"],4_000_000).splitlines()]
    require(len(raw)==436,"incomplete observation journal")
    at=iter(raw)
    def take(kind,**keys):
        v=next(at)
        require(v["event"]==kind and all(v[k]==x for k,x in keys.items()),"wrong observation order or membership")
        return v
    summaries={}
    for state in pl["states"]:
        source=pl["origins"][state]
        before,structure=coordinates(measured/(state+".json"),source.removeprefix("sha256:"))
        base=[take("gradient",source=state,slot=i,owner=source) for i in range(18)]
        for i,v in enumerate(base):
            require(len(v["v"])==64 and all(math.isfinite(x) for x in v["v"]) and f32(v["cost"])==v["loss"] and
                    i in pl["members"][NAMES[v["meaning"]]],"invalid per-example gradient")
        means={k:mean_vec([base[i]["v"] for i in slots]) for k,slots in pl["members"].items()}
        groups={}
        for k in NAMES:
            idx=pl["members"][k]; require(len(idx)==6,"class missing members")
            norms=[norm(base[i]["v"]) for i in idx]; gm=norm(means[k])
            groups[k]=dict(mean_loss=avg([base[i]["cost"] for i in idx]),mean_gradient_l2=gm,
                mean_example_l2=avg(norms),within_class_retention=gm/avg(norms) if any(norms) else None)
        contrasts={}
        for a,b in (("on","off"),("on","idle"),("off","idle"),("non_idle","idle"),("balanced","idle")):
            x,y=means[a],means[b]
            contrasts[a+":"+b]=dict(dot=dot(x,y),cosine=cosine(x,y),bias_dot=dot(x[:32],y[:32]),gate_dot=dot(x[32:],y[32:]))
        probes={}
        for direction in ("on","off","idle","non_idle","balanced"):
            slots=pl["members"][direction];results=[]
            for j,radius in enumerate(pl["radii"]):
                key=dict(source=state,direction=direction,radius=j);take("start",**key)
                v=take("proposal",**key);r=v["receipt"]
                after,check=coordinates(measured/f"{state}-{direction}-{j}.json",v["pin"])
                require(check==structure and r["pre_geometry_id"]==source and r["post_geometry_id"]=="sha256:"+v["pin"] and
                    r["base_model_identity"]==pl["base"] and r["lora_artifact_id"]==pl["lora"] and
                    r["readout_accumulator"]=="f32_wide_backward" and r["trainable_values"]==64 and
                    r["members"]==len(slots) and r["total_target_tokens"]==18*len(slots) and f32(r["learning_rate"])==f32(pl["learning_rate"]),
                    "changed owners, frozen coordinates, or vote domain")
                d=[y-x for x,y in zip(before,after,strict=True)]
                require(close(r["applied_l2"],norm(d)) and abs(norm(d)/radius-1)<1e-4 and
                        r["distance_match"]["requested_l2"]==radius and
                        close(r["mean_input_loss"],avg([base[i]["loss"] for i in slots])) and
                        close(r["mean_gradient_l2"],norm(list(map(f32,means[direction])))) and
                        all(close(x,dot(base[i]["v"],d)) for i,x in zip(slots,r["member_loss_dots"],strict=True)),
                        "realized native receipt differs from parameter readback")
                along=cosine(d,means[direction]);require(along is not None and along < -1+1e-6,"not a negative mean step")
                scored=[take("score",slot=i,**key) for i in range(18)]
                outcomes={}
                for k in (*NAMES,"non_idle","balanced"):
                    ids=pl["members"][k];p=dot(means[k],d)
                    changes={fld:avg([scored[i][fld]-base[i][fld] for i in ids]) for fld in ("cost","weak","absolute")}
                    require(all(map(math.isfinite,changes.values())),"nonfinite finite step")
                    outcomes[k]=dict(predicted=p, measured=changes["cost"], absolute_delta=changes["absolute"],weak_delta=changes["weak"])
                results.append(dict(radius=radius,realized=norm(d),mean_direction_cosine=along,groups=outcomes))
            probes[direction]=results
        def both(direction,target,sign):
            return all(directional_gate(row["groups"][target],sign,pl) for row in probes[direction])
        co=contrasts["non_idle:idle"]["cosine"]
        conditions=dict(opposed=co is not None and co<0,other_mean_descends=both("non_idle","non_idle",-1),
                        idle_harmed=both("non_idle","idle",1),idle_self_control=both("idle","idle",-1))
        summaries[state]=dict(classes=groups,contrasts=contrasts,probes=probes,conditions=conditions,
            primary_confirmed=all(conditions.values()),balanced_harms_idle=both("balanced","idle",1),
            balanced_helps_idle=both("balanced","idle",-1))
    require(next(at,None) is None,"unexpected extra record")
    out.mkdir(mode=0o700)
    return put(out/"results.json",dict(plan=ppin,measured=mpin,states=summaries,
        persistent_local_conflict=all(s["primary_confirmed"] for s in summaries.values()),
        inference="Local fixed-weight directions on training inputs only; neither past trajectory causation nor held-out quality."))
