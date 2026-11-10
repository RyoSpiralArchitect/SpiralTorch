"""Confirm the predicted common direction on two frozen synthetic training snapshots."""
import itertools
import json
import math
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
from alias_interference import CLASSES, RADII, STATES, fit_inputs, forbid_nontraining_files, signature
from alias_interference_audit import coordinates, cosine, dot, f32, mean_vec, norm
from alias_subset_study import MENU, RATE, STOP, native, open_model
from shared_prefix_study import SITE, obj, put, read, record, require, sha, write


def triangle(means):
    """Independent three-group reference: project onto three edges and the triangle plane."""
    lengths=[norm(g) for g in means]
    require(len(means)==3 and all(v>0 for v in lengths),"need three nonzero class means")
    u=[[x/n for x in v] for v,n in zip(means,lengths,strict=True)]
    def diff(a,b):return [x-y for x,y in zip(a,b,strict=True)]
    def blend(w):return [math.fsum(x*y for x,y in zip(col,w,strict=True)) for col in zip(*u,strict=True)]
    candidates=[[float(i==j) for i in range(3)] for j in range(3)]
    for i,j in itertools.combinations(range(3),2):
        v=diff(u[j],u[i]);v2=dot(v,v)
        if v2:
            t=max(0.,min(1.,-dot(u[i],v)/v2));w=[0.,0.,0.];w[i]=1-t;w[j]=t;candidates.append(w)
    v,w=diff(u[0],u[2]),diff(u[1],u[2]);a,b,c=dot(v,v),dot(v,w),dot(w,w)
    den=a*c-b*b
    if den>1e-14*a*c:
        y,z=-dot(u[2],v),-dot(u[2],w);x=(y*c-b*z)/den;t=(a*z-b*y)/den
        if min(x,t,1-x-t)>=0:candidates.append([x,t,1-x-t])
    alpha=min(candidates,key=lambda w:norm(blend(w)));v=blend(alpha);margin=norm(v)
    require(margin>1e-10 and all(dot(g,v)>=margin**2-1e-12 for g in u),"no resolved reference direction")
    direction=[-x/margin for x in v];beta=[a/n for a,n in zip(alpha,lengths,strict=True)]
    return dict(unit_weights=alpha,raw_weights=[a/math.fsum(beta) for a in beta],margin=margin,direction=direction,
                group_l2=lengths,slopes=[dot(g,direction) for g in means])


def archive(measured,pin):
    m=obj(measured/"measured.json",pin)
    records=[json.loads(v) for v in read(measured/"observations.jsonl",m["observations"],4_000_000).splitlines()]
    require(len(records)==436 and m["counts"]==dict(sealed_gradients=36,independent_proposals=20,extra_forwards=360),"incomplete prior study")
    return m,records


def plan(prior,pp,measured,mp,out):
    source=obj(prior/"plan.json",pp);m,raw=archive(measured,mp)
    require(m["plan"]==pp and source["origins"]==m["origins"],"different prior reports")
    groups=[source["members"][k] for k in CLASSES]
    require(sorted(i for ids in groups for i in ids)==list(range(18)) and [len(ids) for ids in groups]==[6,6,6],"prior classes differ")
    predictions={}
    for s in STATES:
        gs={v["slot"]:v["v"] for v in raw if v["source"]==s and v["event"]=="gradient"}
        predictions[s]=triangle([mean_vec([gs[i] for i in ids]) for ids in groups])
    out.mkdir(mode=0o700)
    return put(out/"plan.json",dict(kind="spiraltorch.alias_group_common_direction.v1",script=sha(Path(__file__).read_bytes()),
        prior_plan=pp,prior_measured=mp,prior_journal=m["observations"],fixture=source["fixture"],training=source["training"],
        base=source["base"],lora=source["lora"],origins=m["origins"],group_slots=groups,reference=predictions,
        radii=RADII,learning_rate=RATE,classes=CLASSES,states=STATES,absolute_margin=5e-6,linear_rel=.15,
        policy="unweighted per-class means, normalize each, maximum worst normalized slope; no Python-supplied step coefficients",
        gate="At both radii in each source: equal control replays prior endpoint and training scores exactly and harms idle. For all three class means the common step must predict and measure conditional NLL decreasing beyond 5e-6, agree within max(5e-6,15% of predicted change), and direction must match the frozen reference. No requirement for absolute or worse-alias NLL; report them separately.",
        budget=dict(gradients=36,equal_proposals=4,common_proposals=4,forwards=144,seconds=120,retries=0,trajectory_updates=0)))


def measure(folder,trained,prior,planned,pp,native_pin,out):
    forbid_nontraining_files(folder,trained)
    pl=obj(planned/"plan.json",pp);m0,old=archive(prior,pl["prior_measured"])
    require(pl["script"]==sha(Path(__file__).read_bytes()) and m0["observations"]==pl["prior_journal"],"changed protocol")
    fx,tr,data=fit_inputs(folder,pl["fixture"],trained,pl["training"])
    out.mkdir(mode=0o700)
    st=native();require(signature(st._rs.__file__)==native_pin,"unapproved installed extension")
    origin=open_model(st,folder,fx["files"]);h=tr["endpoints"]["subset_rank"]
    art=st.GeometryAdapterArtifact.from_json(read(trained/"subset_rank.json",h,100000).decode(),"sha256:"+h,tr["base"])
    states=dict(initial=origin,subset_rank=origin.with_geometry(art));frozen=origin.lora_to_json()
    counts=dict(gradients=0,equal_proposals=0,common_proposals=0,forwards=0)
    with (out/"observations.jsonl").open("xb") as f:
        for s in STATES:
            model=states[s];owner=model.geometry_artifact_id
            require(owner==pl["origins"][s] and model.base_model_identity==pl["base"] and model.lora_artifact_id==pl["lora"],"wrong source")
            require("sha256:"+write(out/(s+".json"),model.geometry_artifact().to_json().encode())==owner,"source digest")
            raw={v["slot"]:v for v in old if v["source"]==s and v["event"]=="gradient"}
            handles=[]
            for i,row in enumerate(data):
                c=row["meaning"];g=model.geometry_line_subset_gradients(row["prompt"],MENU,accepted_indices=[c,c+3],stop_id=STOP);counts["gradients"]+=1
                report=g.line_subset_report();params=g.parameter_gradients()
                v=params[SITE]["bias"]+params[SITE]["gate"]
                require(set(params)=={SITE} and set(params[SITE])=={"bias","gate"} and g.geometry_artifact_id==owner and
                        g.target_tokens==18 and (v,g.loss,report["conditional_nll"],report["accepted_set_nll"],
                        max(report["line_nlls"][c],report["line_nlls"][c+3]))==
                        (raw[i]["v"],raw[i]["loss"],raw[i]["cost"],raw[i]["absolute"],raw[i]["weak"]),"retained baseline mismatch")
                record(f,dict(event="gradient",source=s,slot=i,owner=owner,archive_sha=sha(json.dumps(raw[i],sort_keys=True).encode())))
                handles.append(g)
            grouped=[[handles[i] for i in ids] for ids in pl["group_slots"]]
            for policy in ("equal","common"):
                for j,radius in enumerate(RADII):
                    record(f,dict(event="start",source=s,policy=policy,radius=j))
                    if policy=="equal":probe,receipt=model.step_geometry_cohort_sgd(handles,RATE,target_l2=radius)
                    else:probe,receipt=model.step_common_geometry_sgd(grouped,RATE,target_l2=radius)
                    counts[policy+"_proposals"]+=1
                    require(receipt["pre_geometry_id"]==owner==model.geometry_artifact_id and receipt["post_geometry_id"]==probe.geometry_artifact_id and
                        probe.lora_to_json()==frozen and probe.base_model_identity==pl["base"] and abs(receipt["applied_l2"]/radius-1)<1e-4,"proposal domain")
                    pin=write(out/f"{s}-{policy}-{j}.json",probe.geometry_artifact().to_json().encode())
                    require(receipt["post_geometry_id"]=="sha256:"+pin,"wrong endpoint hash")
                    if policy=="equal":
                        eq=next(v for v in old if v["event"]=="proposal" and (v["source"],v["direction"],v["radius"])==(s,"balanced",j))
                        require(pin==eq["pin"] and receipt==eq["receipt"],"equal control no longer byte-identical")
                    else:
                        ref=pl["reference"][s]
                        require(max(abs(a-b) for a,b in zip(ref["unit_weights"],receipt["unit_group_weights"],strict=True))<1e-8 and
                                abs(receipt["predicted_min_unit_margin"]-ref["margin"])<1e-8,"solver differs from frozen oracle")
                    record(f,dict(event="proposal",source=s,policy=policy,radius=j,pin=pin,receipt=receipt))
                    for i,row in enumerate(data):
                        c=row["meaning"];r=probe.line_subset_loss(row["prompt"],MENU,accepted_indices=[c,c+3],stop_id=STOP)
                        counts["forwards"]+=1
                        scored=dict(cost=r["conditional_nll"],absolute=r["accepted_set_nll"],weak=max(r["line_nlls"][c],r["line_nlls"][c+3]))
                        if policy=="equal":
                            eq=next(v for v in old if v["event"]=="score" and (v["source"],v["direction"],v["radius"],v["slot"])==(s,"balanced",j,i))
                            require(all(x==eq[k] for k,x in scored.items()),"equal score differs from archive")
                        record(f,dict(event="score",source=s,policy=policy,radius=j,slot=i,**scored))
    require(all(counts[k]==pl["budget"][k] for k in counts),"model call budget")
    return put(out/"measured.json",dict(plan=pp,native=native_pin,counts=counts,observations=sha((out/"observations.jsonl").read_bytes())))


if __name__=="__main__":
    from alias_common_step_audit import audit
    cmd,*a=sys.argv[1:]
    if cmd=="plan" and len(a)==5: print(plan(Path(a[0]),a[1],Path(a[2]),a[3],Path(a[4])))
    elif cmd=="measure" and len(a)==7:print(measure(Path(a[0]),Path(a[1]),Path(a[2]),Path(a[3]),a[4],a[5],Path(a[6])))
    elif cmd=="audit" and len(a)==6:print(audit(Path(a[0]),a[1],Path(a[2]),Path(a[3]),a[4],Path(a[5])))
    else:raise SystemExit("plan OLD_PLAN OP OLD_MEASURED OM OUT | measure FIXTURE TRAIN OLD_MEASURED PLAN PIN LIB_SHA OUT | audit PLAN PIN OLD_MEASURED MEASURED MP OUT")
