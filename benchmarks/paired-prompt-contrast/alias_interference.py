"""Fixed-snapshot class-gradient interference. Only the earlier synthetic *training* inputs are opened."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
from alias_subset_study import MENU, RATE, STOP, native, open_model
from shared_prefix_study import SITE, obj, pack, put, read, record, require, sha, write

CLASSES = ("on","off","idle")
DIRECTIONS = {"on":(0,), "off":(1,), "idle":(2,), "non_idle":(0,1), "balanced":(0,1,2)}
RADII = (.001,.002)
STATES = ("initial","subset_rank")


def fit_inputs(folder,fp,trained,tp):
    fx=obj(folder/"fixture.json",fp)
    tr=obj(trained/"training.json",tp)
    data=obj(folder/"train.json",fx["files"]["train.json"])
    require(tr["fixture"]==fp and fx["menu"]==MENU and fx["stop"]==STOP and len(data)==18 and
            len({tuple(t["prompt"]) for t in data})==18 and
            [sum(r["meaning"]==i for r in data) for i in range(3)]==[6]*3,"wrong frozen training cohort")
    return fx,tr,data


def plan(folder,fp,trained,tp,out):
    fx,tr,data=fit_inputs(folder,fp,trained,tp)
    out.mkdir(mode=0o700)
    return put(out/"plan.json",dict(schema="spiraltorch.alias_interference.v1",script=sha(Path(__file__).read_bytes()),
        fixture=fp,training=tp,inputs=fx["files"]["train.json"],base=tr["base"],lora=tr["lora"],
        origins=dict(initial=tr["origin"],subset_rank="sha256:"+tr["endpoints"]["subset_rank"]),
        members={k:[i for i,r in enumerate(data) if r["meaning"] in cs] for k,cs in DIRECTIONS.items()},
        radii=RADII,learning_rate=RATE,states=STATES,margin=5e-6,linear_rel=.15,
        primary="At each snapshot separately, the on/off mean must predict and measure idle loss rising and mean on/off loss falling at BOTH radii. The idle-only control must decrease idle at both. Signs exceed 5e-6; predicted vs measured error <= max(5e-6, 15% of |prediction|). Both snapshots required for persistent local conflict.",
        secondary="Balanced-three damage to idle tested separately at both radii. Cosines alone do not qualify. No historical causal attribution or held-out quality inference.",
        budgets=dict(frozen_snapshots=2,sealed_gradients=36,independent_proposals=20,extra_forwards=360,
                     max_cpu_and_wall_seconds=120,retries=0,trajectory_updates=0), no_evaluation_files=True))


def signature(path):
    import hashlib
    with Path(path).open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()


def forbid_nontraining_files(folder,trained):
    f,t=folder.resolve(),trained.resolve()
    allowed={f/k for k in ("fixture.json","config.json","model.safetensors","train.json")}|{t/"training.json",t/"subset_rank.json"}
    root=f.parent
    def guard(event,args):
        if event=="open" and isinstance(args[0],(str,bytes)):
            import os
            p=Path(os.fsdecode(args[0])).resolve()
            if (p.is_relative_to(f) or p.is_relative_to(t) or
                    any(p.is_relative_to(root/k) for k in ("scoring","audit"))) and p not in allowed:
                raise PermissionError("study cannot open this prior file: "+p.name)
    sys.addaudithook(guard)


def measure(folder,trained,planned,ppin,native_pin,out):
    forbid_nontraining_files(folder,trained)
    pl=obj(planned/"plan.json",ppin)
    require(pl["script"]==sha(Path(__file__).read_bytes()),"study changed after plan")
    fx,tr,data=fit_inputs(folder,pl["fixture"],trained,pl["training"])
    out.mkdir(mode=0o700); st=native()
    require(signature(st._rs.__file__)==native_pin,"wrong installed native extension")
    initial=open_model(st,folder,fx["files"]); h=tr["endpoints"]["subset_rank"]
    art=st.GeometryAdapterArtifact.from_json(read(trained/"subset_rank.json",h,100000).decode(),"sha256:"+h,tr["base"])
    sources={"initial":initial, "subset_rank":initial.with_geometry(art)}
    counts=dict(sealed_gradients=0,independent_proposals=0,extra_forwards=0)
    def state(m,path):
        pin=write(path,m.geometry_artifact().to_json().encode())
        require("sha256:"+pin==m.geometry_artifact_id,"artifact digest differs")
        return pin
    frozen=initial.lora_to_json()
    with (out/"observations.jsonl").open("xb") as f:
        for s in STATES:
            m=sources[s]; source=m.geometry_artifact_id
            require(source==pl["origins"][s] and (m.base_model_identity,m.lora_artifact_id)==(tr["base"],tr["lora"]),
                    "wrong source snapshot")
            state(m,out/(s+".json")); handles=[]
            for i,row in enumerate(data):
                c=row["meaning"]
                g=m.geometry_line_subset_gradients(row["prompt"],MENU,accepted_indices=[c,c+3],stop_id=STOP)
                counts["sealed_gradients"]+=1
                r=g.line_subset_report(); vm=g.parameter_gradients()
                require(set(vm)=={SITE} and set(vm[SITE])=={"gate","bias"} and
                    (g.geometry_artifact_id,g.base_model_identity,g.lora_artifact_id,g.readout_accumulator)==
                    (source,tr["base"],tr["lora"],"f32_wide_backward") and g.target_tokens==18,"wrong gradient scope")
                v=vm[SITE]["bias"]+vm[SITE]["gate"]
                require(len(v)==64 and all(map(math.isfinite,v)),"invalid gradient array")
                record(f,dict(event="gradient",source=s,slot=i,meaning=c,owner=source,cost=r["conditional_nll"],
                              loss=g.loss,absolute=r["accepted_set_nll"],weak=max(r["line_nlls"][c],r["line_nlls"][c+3]),v=v))
                handles.append(g)
            for direction in DIRECTIONS:
                slots=pl["members"][direction]; members=[handles[i] for i in slots]
                for j,radius in enumerate(RADII):
                    record(f,dict(event="start",source=s,direction=direction,radius=j))
                    probe,receipt=m.step_geometry_cohort_sgd(members,RATE,target_l2=radius)
                    counts["independent_proposals"]+=1
                    require(receipt["pre_geometry_id"]==source==m.geometry_artifact_id and
                            probe.geometry_artifact_id==receipt["post_geometry_id"] and
                            probe.base_model_identity==tr["base"] and probe.lora_to_json()==frozen and
                            receipt["members"]==len(slots) and receipt["total_target_tokens"]==18*len(slots) and
                            receipt["trainable_values"]==64 and abs(receipt["applied_l2"]/radius-1)<1e-4 and
                            receipt["distance_match"]["relative_vector_error"]<=1e-4,"invalid finite proposal")
                    pin=state(probe,out/f"{s}-{direction}-{j}.json")
                    record(f,dict(event="proposal",source=s,direction=direction,radius=j,pin=pin,receipt=receipt))
                    for i,row in enumerate(data):
                        c=row["meaning"]
                        r=probe.line_subset_loss(row["prompt"],MENU,accepted_indices=[c,c+3],stop_id=STOP)
                        counts["extra_forwards"]+=1
                        record(f,dict(event="score",source=s,direction=direction,radius=j,slot=i,cost=r["conditional_nll"],
                                      absolute=r["accepted_set_nll"],weak=max(r["line_nlls"][c],r["line_nlls"][c+3])))
    require(all(pl["budgets"][k]==v for k,v in counts.items()),"call budgets differ")
    return put(out/"measured.json",dict(plan=ppin,installed_extension=native_pin,origins=pl["origins"],counts=counts,
        observations=sha((out/"observations.jsonl").read_bytes())))


if __name__=="__main__":
    from alias_interference_audit import audit
    command,*args=sys.argv[1:]
    if command=="plan" and len(args)==5:
        print(plan(Path(args[0]),args[1],Path(args[2]),args[3],Path(args[4])))
    elif command=="measure" and len(args)==6:
        print(measure(Path(args[0]),Path(args[1]),Path(args[2]),args[3],args[4],Path(args[5])))
    elif command=="audit" and len(args)==5:
        print(audit(Path(args[0]),args[1],Path(args[2]),args[3],Path(args[4])))
    else:
        raise SystemExit("plan FIXTURE FP TRAIN TP OUT | measure FIXTURE TRAIN PLAN PP NATIVE_SHA OUT | audit PLAN PP MEASURED MP OUT")
