"""Prospective, training-only comparison of primary steering against a byte-replayed guard."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
import context_token_training as prior
from alias_guarded_training import fingerprint, retain
from alias_interference import signature
from shared_prefix_study import obj, put, read, record, require, sha, write

ARMS, STATES = prior.ARMS, prior.STATES
KIND = "spiraltorch.train_primary_context.v1"
RESOLVED = {
    "geometry steering requires nonzero finite group means", "geometry steering anchor misses a protected cosine floor",
    "unresolved primary geometry steering direction", "geometry guard displacement overflowed f32",
    "geometry guard direction or length is not representable in f32", "rounded geometry steering step misses a protected cosine floor",
    "rounded geometry steering step does not improve primary descent",
}
LIMITS = prior.BUDGET | {"max_steers": 3*len(STATES)*prior.STEPS}


def sources():
    return prior.sources() | {"primary_context_training":signature(Path(__file__)),
                              "primary_context_audit":signature(Path(__file__).with_name("primary_context_audit.py"))}


def freeze(previous, readback_pin, extension, wheel_pin, wheel, out):
    rb = obj(previous/"readback.json", readback_pin)
    old = obj(previous/"protocol/plan.json",rb["plan"])
    measured = obj(previous/"measured/measured.json",rb["measurement"])
    audited = obj(previous/"report/results.json",rb["official_report"],2<<20)
    require(old["code"] == prior.sources() and old["kind"] == prior.KIND and measured["plan"] == audited["plan"] == rb["plan"] and
            audited["measurement"] == rb["measurement"] and rb["counts"] == measured["counts"] and
            not any(rb["gates"].values()) and rb["counts"]["adopted"] == 512 and old["steps"] == prior.STEPS and
            old["floor"] == prior.FLOOR and old["radius"] == prior.RADIUS, "wrong historical training control")
    inv = prior.candidate_inventory(wheel,wheel_pin,extension)
    out.mkdir(mode=0o700); inputs=out/"inputs"; inputs.mkdir(mode=0o700)
    for f,k in (("train.json","train.json"),("config.json","config.json"),("model.safetensors","weights")):
        write(inputs/f,read(previous/"protocol"/"inputs"/f,old["files"][k],100000))
    for state,owner in old["origins"].items():
        write(inputs/(state+".json"),read(previous/"protocol/inputs"/(state+".json"),owner[7:],100000))
    keep = ("groups","pairs","triples","origins","base","lora","files","refs","primary","secondary","margin","effect","eps","floor","rate","radius")
    plan = {k:old[k] for k in keep}
    return put(out/"plan.json",plan | dict(kind=KIND,code=sources(),previous_readback=readback_pin,previous_plan=rb["plan"],
        previous_wheel=old["wheel"],wheel=wheel_pin,native=extension,inventory=put(out/"wheel-inventory.json",inv),
        arms=ARMS,states=STATES,steps=prior.STEPS,max_budget=LIMITS,wall=prior.WALL,retries=0,
        treatment="At each current snapshot: same four-group proposal, same nine-group guard, same 66 gradients. "
                  "Control adopts the anchor; the other arms steer the anchor toward positive word CE, matched, or rotated preference. "
                  "No extra forward for a steer. Reject a rounded primary with no strict actual dot and cosine improvement.",
        negative="First recognized numerical refusal retires only that path; no substitution or retry. All other errors stop. "
                 "An incomplete path cannot pass its endpoint gate. Primary/secondary efficacy gates match the previous trial. "
                 "Separately count measured finite nine-objective descent, per-step primary finite loss and absolute spillover."))


def admitted(protocol,p):
    require(p["kind"]==KIND and p["code"]==sources() and (p["steps"],p["max_budget"],p["wall"],p["retries"])==
            (prior.STEPS,LIMITS,prior.WALL,0) and p["arms"]==list(ARMS) and p["states"]==list(STATES), "not the frozen trial")
    inv=obj(protocol/"wheel-inventory.json",p["inventory"],2<<20)
    require((inv["sha256"],inv["native"]["sha256"])==(p["wheel"],p["native"]),"foreign wheel")
    return inv


def worker(protocol,pp,out):
    import resource

    resource.setrlimit(resource.RLIMIT_CPU,(prior.WALL,prior.WALL))
    def training_only(event,args):
        if event=="open" and isinstance(args[0],(str,bytes)):
            f=Path(os.fsdecode(args[0]))
            if f.name in {"truth.json","queries.json","answer-key.json"} or "2026-11-04-native-alias-fresh-context-transfer-v1" in f.parts:
                raise PermissionError("no held-out inputs in primary training")
    sys.addaudithook(training_only)
    p=obj(protocol/"plan.json",pp); inv=admitted(protocol,p); installed=prior.installed_identity(inv)
    st=prior.native()
    require(Path(st._rs.__file__).resolve()==Path(st.__file__).resolve().parent.parent/inv["native"]["path"] and
            st.NativeCausalLm is st._rs.NativeCausalLm,"foreign decoder")
    inputs=protocol/"inputs"; data=obj(inputs/"train.json",p["files"]["train.json"])
    require(prior.pairs(data)==(p["triples"],p["pairs"]) and
            p["groups"]==[[i for i,v in enumerate(data) if v["meaning"]==k] for k in range(3)],"not the training roster")
    base=prior.open_model(st,inputs,p["files"]); frozen=base.lora_to_json()
    require((base.base_model_identity,base.lora_artifact_id)==(p["base"],p["lora"]),"foreign frozen model")
    counts=dict.fromkeys(("score_menus","score_singletons","adopted","backwards","proposals","guards","steers"),0)
    ends={}; (out/"geometry").mkdir(mode=0o700)
    with (out/"journal.jsonl").open("xb") as f,(out/"vectors.f32le").open("xb") as binary:
        for state in STATES:
            src=p["origins"][state]
            start=base.with_geometry(st.GeometryAdapterArtifact.from_json(read(inputs/(state+".json"),src[7:],100000).decode(),src,p["base"]))
            require(retain(out,start)==src,"wrong beginning")
            for arm in ARMS:
                key=state+"/"+arm; model=start; rows=prior.score(model,data,counts); chosen=0; refusal=None
                record(f,dict(event="origin",key=key,owner=src,rows=rows))
                for t in range(p["steps"]):
                    owner=model.geometry_artifact_id; tags=dict(key=key,t=t,owner=owner)
                    record(f,dict(event="start",**tags))
                    classes,tail,branches,loss,reports,raw,og,old_scores=prior.gradients(model,p,data,rows,counts)
                    protected=[*classes,*([g] for g in tail)]; binary.write(raw)
                    record(f,dict(event="gradients",**tags,sha=sha(raw),old_gradient=og,old_scores=old_scores,loss=loss,reports=reports))
                    ref=p["refs"][state][t]
                    if arm=="guard_only" or owner==ref["pre"]:
                        require((owner,og,old_scores)==(ref["pre"],ref["gradient"],ref["before"]),"control inputs drifted")
                    phase="proposal"
                    try:
                        counts["proposals"]+=1
                        proposal,r=model.step_common_geometry_sgd([*classes,tail],p["rate"],target_l2=p["radius"])
                        require(retain(out,proposal)==r["post_geometry_id"] and owner==r["pre_geometry_id"],"proposal ownership")
                        record(f,dict(event="proposal",**tags,receipt=r))
                        phase="guard"; counts["guards"]+=1
                        anchor,g=model.guard_geometry_step(proposal,protected,min_cosine=p["floor"])
                        aid=retain(out,anchor)
                        require((g["pre_geometry_id"],g["candidate_geometry_id"],g["post_geometry_id"])==(owner,proposal.geometry_artifact_id,aid),
                                "anchor ownership")
                        record(f,dict(event="anchor",**tags,receipt=g))
                        if arm=="guard_only" or owner==ref["pre"]:
                            require((aid,proposal.geometry_artifact_id,fingerprint(r),fingerprint(g))==
                                    (ref["chosen"],ref["candidate"],ref["proposal"],ref["guard"]),"control update drifted")
                        selected=anchor; proof=None
                        if arm!="guard_only":
                            phase="steer";counts["steers"]+=1
                            selected,proof=model.steer_geometry_step(anchor,branches[arm],protected,min_cosine=p["floor"])
                            require((proof["pre_geometry_id"],proof["anchor_geometry_id"],proof["post_geometry_id"])==
                                    (owner,aid,selected.geometry_artifact_id),"steering ownership")
                    except ValueError as ex:
                        allowed=RESOLVED if phase=="steer" else prior.REFUSALS[phase]
                        if str(ex) not in allowed: raise
                        refusal=dict(phase=phase,message=str(ex))
                        record(f,dict(event="refusal",**tags,reason=refusal)); break
                    end=retain(out,selected)
                    require(model.geometry_artifact_id==owner and selected.lora_to_json()==frozen and selected.base_model_identity==p["base"],
                            "changed frozen owner")
                    rows=prior.score(selected,data,counts)
                    record(f,dict(event="adopt",**tags,post=end,steering=proof,rows=rows))
                    model=selected;chosen+=1;counts["adopted"]+=1
                ends[key]=dict(adopted=chosen,owner=model.geometry_artifact_id,refusal=refusal)
    return put(out/"measured.json",dict(plan=pp,installed=installed,ends=ends,counts=counts,
                journal=signature(out/"journal.jsonl"),binary=signature(out/"vectors.f32le")))


def once(protocol,pp,out):
    p=obj(protocol/"plan.json",pp); prior.installed_identity(admitted(protocol,p))
    out.mkdir(mode=0o700);put(out/"attempt.json",dict(plan=pp,attempt=1,native=p["native"]))
    env=dict(os.environ,SPIRALTORCH_BACKEND="cpu",RAYON_NUM_THREADS="1",OMP_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1",
             OPENBLAS_NUM_THREADS="1",HF_HUB_OFFLINE="1",SPIRALTORCH_AUTOTUNE_STORE=str(out/"tuning.json"))
    start=time.monotonic()
    with (out/"process.log").open("xb") as f:
        try: rc=subprocess.run([sys.executable,"-I","-B",str(Path(__file__).resolve()),"_worker",str(protocol),pp,str(out)],
                               cwd=out,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=p["wall"]).returncode
        except subprocess.TimeoutExpired: rc="timeout"
    elapsed=time.monotonic()-start
    put(out/"exit.json",dict(exit=rc,elapsed=elapsed,log=signature(out/"process.log")))
    require(rc==0 and elapsed<=p["wall"],"stopped; first attempt retained")
    return signature(out/"measured.json")


if __name__=="__main__":
    from primary_context_audit import audit
    cmd,*a=sys.argv[1:]
    if cmd=="freeze" and len(a)==6: print(freeze(Path(a[0]),*a[1:4],Path(a[4]),Path(a[5])))
    elif cmd in ("run","_worker") and len(a)==3: print((once if cmd=="run" else worker)(Path(a[0]).resolve(),a[1],Path(a[2]).resolve()))
    elif cmd=="audit" and len(a)==5: print(audit(Path(a[0]),a[1],Path(a[2]),a[3],Path(a[4])))
    else: raise SystemExit("freeze PREVIOUS READBACK EXT_SHA WHEEL_SHA WHEEL OUT | run PLAN PIN OUT | audit PLAN PIN MEASURED MP OUT")
