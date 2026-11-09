"""Reconstruct measured group changes and first derivatives from immutable bytes, without a model."""
import json
import math
from shared_prefix_study import obj,put,read,require,sha
from alias_interference_audit import coordinates,cosine,dot,mean_vec,norm


def qualifies(pred,measured,pl):
    return (pred < -pl["absolute_margin"] and measured < -pl["absolute_margin"] and
            abs(pred-measured)<=max(pl["absolute_margin"],pl["linear_rel"]*abs(pred)))


def audit(planned,pp,prior,observed,mp,out):
    from alias_common_step import archive,triangle
    pl=obj(planned/"plan.json",pp);m=obj(observed/"measured.json",mp);previous,old=archive(prior,pl["prior_measured"])
    require(m["plan"]==pp and m["counts"]==dict(gradients=36,equal_proposals=4,common_proposals=4,forwards=144) and
            previous["observations"]==pl["prior_journal"],"measurement domain")
    records=[json.loads(l) for l in read(observed/"observations.jsonl",m["observations"],2_000_000).splitlines()]
    require(len(records)==196,"incomplete measurement")
    it=iter(records)
    def take(event,**expected):
        r=next(it);require(r["event"]==event and all(r[k]==v for k,v in expected.items()),"record order/domain");return r
    def close(a,b):return math.isclose(a,b,abs_tol=1e-12,rel_tol=2e-10)
    states={}; groups=pl["group_slots"]
    for s in pl["states"]:
        owner=pl["origins"][s];before,struct=coordinates(observed/(s+".json"),owner[7:])
        base={r["slot"]:r for r in old if r["source"]==s and r["event"]=="gradient"}
        for i in range(18):
            r=take("gradient",source=s,slot=i,owner=owner)
            require(r["archive_sha"]==sha(json.dumps(base[i],sort_keys=True).encode()),"wrong source vector")
        means=[mean_vec([base[i]["v"] for i in ids]) for ids in groups];ref=triangle(means)
        require(ref==pl["reference"][s],"three-way reference replay changed")
        probes={}
        for policy in ("equal","common"):
            runs=[]
            for j,radius in enumerate(pl["radii"]):
                keys=dict(source=s,policy=policy,radius=j);take("start",**keys);row=take("proposal",**keys);r=row["receipt"]
                after,layout=coordinates(observed/f"{s}-{policy}-{j}.json",row["pin"])
                require(layout==struct and r["pre_geometry_id"]==owner and r["post_geometry_id"]=="sha256:"+row["pin"] and
                        r["base_model_identity"]==pl["base"] and r["lora_artifact_id"]==pl["lora"] and
                        r["readout_accumulator"]=="f32_wide_backward" and r["trainable_values"]==64,"changed owners or coordinates")
                delta=[y-x for x,y in zip(before,after,strict=True)];length=norm(delta)
                require(abs(length/radius-1)<1e-4 and close(length,r["applied_l2"]) and r["distance_match"]["requested_l2"]==radius,"wrong step")
                actual=[take("score",slot=i,**keys) for i in range(18)]
                if policy=="equal":
                    eq=next(v for v in old if v["event"]=="proposal" and (v["source"],v["direction"],v["radius"])==(s,"balanced",j))
                    require(eq["pin"]==row["pin"] and eq["receipt"]==r,"control differs")
                    for i,target in enumerate(actual):
                        oldrow=next(v for v in old if v["event"]=="score" and (v["source"],v["direction"],v["radius"],v["slot"])==(s,"balanced",j,i))
                        require(all(target[k]==oldrow[k] for k in ("cost","absolute","weak")),"control score differs")
                else:
                    require(r["group_sizes"]==list(map(len,groups)) and r["group_target_tokens"]==[18*len(g) for g in groups] and
                        all(close(x,norm(g)) for x,g in zip(r["group_gradient_l2"],means,strict=True)) and
                        all(close(x,math.fsum(base[i]["loss"] for i in ids)/len(ids)) for x,ids in zip(r["group_mean_losses"],groups,strict=True)),
                        "wrong member means")
                    for field,key in (("unit_group_weights","unit_weights"),("raw_group_weights","raw_weights")):
                        require(max(abs(x-y) for x,y in zip(r[field],ref[key],strict=True))<1e-8,"wrong direction weights")
                    require(abs(r["hull_norm"]-ref["margin"])<1e-8 and abs(r["predicted_min_unit_margin"]-ref["margin"])<1e-8
                        and cosine(delta,ref["direction"])>1.-1e-6,"distance-normalized direction disagrees with oracle")
                    require(close(r["applied_min_unit_margin"],min(-dot(g,delta)/(norm(g)*length) for g in means)),"wrong realized margin")
                classes={}
                for k,(label,ids,g) in enumerate(zip(pl["classes"],groups,means,strict=True)):
                    pred=dot(g,delta)
                    changes={f:math.fsum(actual[i][f]-base[i][f] for i in ids)/len(ids) for f in ("cost","absolute","weak")}
                    require(all(map(math.isfinite,changes.values())),"nonfinite score")
                    if policy=="common":require(close(r["group_loss_dots"][k],pred),"native realized dot differs")
                    classes[label]=dict(predicted=pred,measured=changes["cost"],absolute_delta=changes["absolute"],
                        weak_alias_delta=changes["weak"],passes=qualifies(pred,changes["cost"],pl))
                runs.append(dict(radius=radius,applied_l2=length,classes=classes))
            probes[policy]=runs
        control=all(v["classes"]["idle"]["measured"]>pl["absolute_margin"] for v in probes["equal"])
        finite=all(c["passes"] for v in probes["common"] for c in v["classes"].values())
        states[s]=dict(control_reproduced=control,all_three_descend=finite,probes=probes,reference_margin=ref["margin"])
    require(next(it,None) is None,"extra observations");out.mkdir(mode=0o700)
    return put(out/"results.json",dict(plan=pp,measured=mp,states=states,
        primary_pass=all(v["control_reproduced"] and v["all_three_descend"] for v in states.values()),
        inference="Finite class-mean conditional NLL at the two designated training snapshots only; not held-out, not a training trajectory."))
