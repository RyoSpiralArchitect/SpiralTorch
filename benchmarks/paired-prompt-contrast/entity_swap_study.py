"""Geometry training and separate key-blind endpoint scoring for entity-state swaps."""
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import context_token_training as prev
import primary_context_training as primary
import entity_swap_task as task
from alias_guarded_training import VECTOR, fingerprint, geometry_path, retain, forward
from alias_interference import signature
from shared_prefix_study import obj, put, read, record, require, sha, write

KIND="spiraltorch.entity_state_swaps.v1"
WALL={"train":180,"score":60}
LOGITS=struct.Struct("<64f")
NGRAD=88
PATHS=len(task.STATES)*len(task.ARMS)
LIMITS=dict(updates=PATHS*task.STEPS,backwards=NGRAD*PATHS*task.STEPS,anchor_proposals=512,guards=512,direct=384,steers=384,
            retention_forwards=24*PATHS*task.STEPS,score_roles=16,score_batches=256,score_rows=2048)
DIRECT_REFUSALS={"cannot assign a positive distance to a zero geometry direction","requested geometry direction or distance cannot be represented accurately in f32"}


def sources():
    return primary.sources() | {k:signature(Path(__file__).with_name(k+".py")) for k in
                                ("entity_swap_task","entity_swap_study","entity_swap_readback")}


def freeze(old,previous_pin,wheel,out):
    rb=obj(old/"readback.json",previous_pin); p=obj(old/"protocol/plan.json",rb["plan"])
    m=obj(old/"measured/measured.json",rb["measurement"]);r=obj(old/"report/results.json",rb["report"],2<<20)
    require(p["kind"]==primary.KIND and p["code"]==primary.sources() and m["plan"]==r["plan"]==rb["plan"] and
            r["measurement"]==rb["measurement"] and rb["counts"]==m["counts"] and rb["gates"]["all_paths_nine_finite"] and
            not rb["gates"]["entity_binding"] and rb["counts"]["adopted"]==512,"wrong prior steering trial")
    inv=prev.candidate_inventory(wheel,p["wheel"],p["native"])
    require(inv==obj(old/"protocol/wheel-inventory.json",p["inventory"],2<<20),"wrong original wheel")
    train,test=task.deck("train"),task.deck("test");task.design(train,test)
    out.mkdir(mode=0o700)
    for name in ("model","fit","blind","keys"): (out/name).mkdir(mode=0o700)
    keep=("origins","base","lora","refs","groups","files","rate","radius","floor")
    plan={k:p[k] for k in keep}
    for f,k in (("config.json","config.json"),("model.safetensors","weights")):
        write(out/"model"/f,read(old/"protocol/inputs"/f,p["files"][k],100000))
    for name,h in p["origins"].items():
        write(out/"model"/(name+".json"),read(old/"protocol/inputs"/(name+".json"),h[7:],100000))
    write(out/"fit/retention.json",read(old/"protocol/inputs/train.json",p["files"]["train.json"],100000))
    decks={}
    for name,rows in (("train",train),("test",test)):
        decks[name]=dict(queries=put(out/"blind"/(name+".json"),task.queries(rows)),key=put(out/"keys"/(name+".json"),task.key(rows)))
    return put(out/"plan.json",plan|dict(kind=KIND,code=sources(),prior_readback=previous_pin,prior_plan=rb["plan"],
        native=p["native"],wheel=p["wheel"],inventory=put(out/"wheel-inventory.json",inv),fit=put(out/"fit/task.json",train),decks=decks,
        states=task.STATES,arms=task.ARMS,steps=task.STEPS,limits=LIMITS,wall=WALL,retries=0,effect=task.EFFECT,eps=task.EPS,
        split="Eight identities in each split; train and held-out contain disjoint degree-two sets of eight unordered pairs. Every pair crosses "
              "both orders of facts, both choices of queried identity and both assignments of on/idle. Exactly 64 rows per split.",
        intervention="Cycle four 16-example batches (two query identities, both partners and all swaps). Each training row occurs 16 times. "
                     "A matched negative swaps both states with the query unchanged. A rotated negative changes just the distractor identity "
                     "relative to that negative; query identity, query slot, state positions, target, positive/negative marginals are fixed. "
                     "CE uses gold word only. Direct arms use distance-matched native cohort SGD without retention protection. Safe arms use "
                     "the same archived-task four-group proposal, nine-group guard and primary steering. All extract the same 88 handles at each attempted step.",
        endpoint="Only complete step-64 endpoints and both origins are eligible; deduplicate by weight hash. Read all 64 vocabulary logits on both "
                 "decks in fixed batches of eight in a fresh process. No answer keys until the sealed scores and exit receipt exist.",
        primary="Held-out mean assignment-by-query interaction = (L_AA-L_AB-L_BA+L_BB)/4 per order, L=logit(standby)-logit(on). "
                "In BOTH starts, safe matched must exceed origin, retention, safe CE and safe rotated by >0.01 nats each, improve by >5e-6 "
                "on at least 6/8 held-out entity pairs and in both presentation-order means for each contrast.",
        secondary="Apply the same contrasts to direct matched vs origin/direct CE/direct rotated. Separately require at least 6/8 full edges "
                  "(all eight choices correct by >5e-6) and at least one more full edge than origin for competence, both starts. Report swaps, "
                  "relative ranks, gold NLL, retention and training scores separately. Ties do not win. No generation or real-model claim.",
        stop="First recognized geometry refusal ends only that path, without fallback. Any other error stops the phase; no retry, threshold "
             "change, endpoint selection or re-use as fresh held-out data. No held-out inference before training and training audit are closed."))


def inventory(proto,p):
    require(p["kind"]==KIND and p["code"]==sources() and p["limits"]==LIMITS and p["wall"]==WALL and p["steps"]==task.STEPS and
            p["states"]==list(task.STATES) and p["arms"]==list(task.ARMS) and (p["effect"],p["eps"],p["retries"])==(task.EFFECT,task.EPS,0),"wrong frozen experiment")
    inv=obj(proto/"wheel-inventory.json",p["inventory"],2<<20)
    require((inv["sha256"],inv["native"]["sha256"])==(p["wheel"],p["native"]),"different wheel")
    return inv


def fence(proto,stage):
    forbidden=[proto/"keys",proto/("blind" if stage=="train" else "fit")]
    def guard(event,args):
        if event=="open" and isinstance(args[0],(str,bytes)):
            path=Path(os.path.abspath(os.fsdecode(args[0])))
            for source in (path,Path(os.path.realpath(path))):
                if any(source==bad or bad in source.parents for bad in forbidden) or source.name in {"truth.json","queries.json","answer-key.json"} or \
                        "2026-11-04-native-alias-fresh-context-transfer-v1" in source.parts:
                    raise PermissionError("data not admitted in "+stage+" phase")
    sys.addaudithook(guard)


def backend(proto,p):
    inv=inventory(proto,p); receipt=prev.installed_identity(inv); st=prev.native()
    require(st.NativeCausalLm is st._rs.NativeCausalLm and Path(st._rs.__file__).resolve()==Path(st.__file__).resolve().parent.parent/inv["native"]["path"],
            "foreign native model")
    model=prev.open_model(st,proto/"model",p["files"])
    require((model.base_model_identity,model.lora_artifact_id)==(p["base"],p["lora"]),"wrong base/LoRA")
    return st,model,receipt


def attach(st,model,path,owner):
    require(owner.startswith("sha256:"),"invalid owner")
    return model.with_geometry(st.GeometryAdapterArtifact.from_json(read(path,owner[7:],100000).decode(),owner,model.base_model_identity))


def handles(model,p,retention,batch,ns):
    owner=model.geometry_artifact_id
    hs=[];ts=[];vecs=[];tail=[];rows=[];losses=[];tl=[]
    for i,e in enumerate(retention):
        ids=[e["meaning"],e["meaning"]+3]
        g=model.geometry_line_subset_gradients(e["prompt"],prev.MENU,accepted_indices=ids,stop_id=prev.STOP)
        v=prev.sealed(g,owner,p,18);r=dict(slot=i,**prev.scores(g.line_subset_report(),ids))
        require(v["loss"]==prev.f32(r["cost"]),"wrong retention gradient")
        hs.append(g);vecs.append(v["v"]);losses.append(v["loss"])
        if e["meaning"]==2:
            h=model.geometry_line_set_gradients(e["prompt"],[prev.MENU[5]],stop_id=prev.STOP)
            x=prev.sealed(h,owner,p,3);r["single"]=h.line_set_report()["set_nll"]
            require(x["loss"]==prev.f32(r["single"]) and abs(r["single"]-r["lines"][5])<=prev.CROSS_TOL,"wrong retention singleton")
            ts.append(h);tail.append(x["v"]);tl.append(x["loss"])
        rows.append(r)
    vecs.extend(tail);losses.extend(tl);words=[]; data=[];branches={"ce":[]};reports={}
    for row in batch:
        pair=[];hs_word=[]
        for token in task.ANSWERS:
            ids=row["prompt"]+[token]
            g=model.adaptation_loss_gradients(ids,[1,len(ids)],mode="geometry",labels=[-100]*(len(ids)-1)+[token])
            v=prev.sealed(g,owner,p,1);pair.append(v);hs_word.append(g);vecs.append(v["v"]);losses.append(v["loss"])
        data.append(pair);words.append(hs_word)
    branches["ce"]=[hs_word[task.ANSWERS.index(row["gold"])] for hs_word,row in zip(words,batch,strict=True)]
    for name in ("matched","rotated"):
        branches[name]=[];reports[name]=[]
        for i,j in enumerate(ns[name]):
            row=batch[i];k=task.ANSWERS.index(row["gold"])
            g=model.geometry_context_token_preference_gradients(row["prompt"],batch[j]["prompt"],row["gold"])
            v=prev.sealed(g,owner,p,2);r=g.context_token_preference_report()
            require((r["target_id"],r["vocabulary"],r["margin"])==(row["gold"],64,0.) and prev.f32(r["loss"])==v["loss"] and
                    abs(r["preferred_nll"]-data[i][k]["loss"])<=prev.CROSS_TOL and abs(r["other_nll"]-data[j][k]["loss"])<=prev.CROSS_TOL,
                    "preference scored different contexts")
            prev.check_seed(v["v"],data[i][k]["v"],data[j][k]["v"],r["gradient_weight"])
            branches[name].append(g);reports[name].append(r);vecs.append(v["v"]);losses.append(v["loss"])
    require(len(vecs)==len(losses)==NGRAD,"wrong backward budget")
    return [[hs[i] for i in ids] for ids in p["groups"]],ts,branches,rows,losses,reports,b"".join(VECTOR.pack(*v) for v in vecs)


def train_worker(proto,pp,out):
    p=obj(proto/"plan.json",pp);st,base,admitted=backend(proto,p); frozen=base.lora_to_json()
    retain_data=obj(proto/"fit/retention.json",p["files"]["train.json"])
    data=obj(proto/"fit/task.json",p["fit"]);require(data==task.deck("train"),"wrong training cases")
    counts=dict.fromkeys(("updates","backwards","anchor_proposals","guards","direct","steers","retention_forwards"),0);ends={}
    (out/"geometry").mkdir(mode=0o700)
    with (out/"journal.jsonl").open("xb") as log,(out/"vectors.f32le").open("xb") as binary:
        for state in task.STATES:
            src=p["origins"][state];start=attach(st,base,proto/"model"/(state+".json"),src);require(retain(out,start)==src,"foreign origin")
            for arm in task.ARMS:
                model=start;key=state+"/"+arm;adopted=0;refusal=None;last=None
                for t in range(task.STEPS):
                    owner=model.geometry_artifact_id;tag=dict(key=key,t=t,owner=owner)
                    batch=task.batch(data,t);ns=task.negatives(batch)
                    record(log,dict(event="start",**tag,ids=[r["id"] for r in batch]))
                    cs,ts,br,rows,losses,reports,block=handles(model,p,retain_data,batch,ns)
                    counts["backwards"]+=NGRAD;binary.write(block)
                    record(log,dict(event="gradients",**tag,sha=sha(block),losses=losses,reports=reports,before=rows))
                    require(last is None or rows==last,"retention forward did not replay at adopted state")
                    protected=[*cs,*([v] for v in ts)];ref=p["refs"][state][t];same=owner==ref["pre"]
                    if same or arm=="retention":
                        require(same and (sha(block[:24*VECTOR.size]),fingerprint(rows))==(ref["gradient"],ref["before"]),"historical retention drift")
                    phase="direct" if arm.startswith("direct_") else "proposal";extra=None
                    try:
                        if phase=="direct":
                            counts["direct"]+=1
                            end,extra=model.step_geometry_cohort_sgd(br[arm.split("_")[1]],p["rate"],target_l2=p["radius"])
                        else:
                            counts["anchor_proposals"]+=1
                            cand,r=model.step_common_geometry_sgd([*cs,ts],p["rate"],target_l2=p["radius"])
                            require(retain(out,cand)==r["post_geometry_id"] and r["pre_geometry_id"]==owner,"foreign proposal")
                            record(log,dict(event="proposal",**tag,receipt=r))
                            phase="guard";counts["guards"]+=1
                            end,g=model.guard_geometry_step(cand,protected,min_cosine=p["floor"])
                            aid=retain(out,end)
                            require((g["pre_geometry_id"],g["candidate_geometry_id"],g["post_geometry_id"])==(owner,cand.geometry_artifact_id,aid),"foreign guard")
                            record(log,dict(event="guard",**tag,receipt=g))
                            if same: require((aid,cand.geometry_artifact_id,fingerprint(r),fingerprint(g))==
                                             (ref["chosen"],ref["candidate"],ref["proposal"],ref["guard"]),"old update drifted")
                            if arm!="retention":
                                counts["steers"]+=1;phase="steer"
                                end,extra=model.steer_geometry_step(end,br[arm.split("_")[1]],protected,min_cosine=p["floor"])
                                require(extra["anchor_geometry_id"]==aid,"foreign steering anchor")
                    except ValueError as ex:
                        allowed=DIRECT_REFUSALS if phase=="direct" else primary.RESOLVED if phase=="steer" else prev.REFUSALS[phase]
                        if str(ex) not in allowed:raise
                        refusal=dict(phase=phase,message=str(ex));record(log,dict(event="refusal",**tag,reason=refusal));break
                    post=retain(out,end)
                    require((extra is None or (extra["pre_geometry_id"],extra["post_geometry_id"])==(owner,post)) and
                            model.geometry_artifact_id==owner and end.base_model_identity==p["base"] and end.lora_to_json()==frozen,"foreign selected model")
                    last=[dict(slot=i,**forward(end,e)) for i,e in enumerate(retain_data)];counts["retention_forwards"]+=24
                    record(log,dict(event="adopt",**tag,post=post,receipt=extra,after=last))
                    model=end;adopted+=1;counts["updates"]+=1
                ends[key]=dict(adopted=adopted,owner=model.geometry_artifact_id,refusal=refusal)
    return put(out/"complete.json",dict(plan=pp,installed=admitted,ends=ends,counts=counts,journal=signature(out/"journal.jsonl"),binary=signature(out/"vectors.f32le")))


def complete(folder,pin,phase,pp,parents):
    result=obj(folder/"complete.json",pin)
    attempt=json.loads((folder/"attempt.json").read_bytes());ex=json.loads((folder/"exit.json").read_bytes())
    require(result["plan"]==pp and attempt==dict(phase=phase,plan=pp,parents=parents,attempt=1) and ex["exit"]==0 and
            0<ex["elapsed"]<=WALL[phase] and read(folder/"process.log",ex["log"]).splitlines()[-1:]==[pin.encode()],"phase did not close successfully")
    return result


def score_worker(proto,pp,out,trained,tp,audited,ap):
    p=obj(proto/"plan.json",pp);m=complete(trained,tp,"train",pp,[]);a=obj(audited/"verified.json",ap)
    require((a["plan"],a["training"],a["ends"])==(pp,tp,m["ends"]),"unverified endpoints")
    st,base,admitted=backend(proto,p)
    put(out/"test-consumed.json",dict(plan=pp,training=tp,training_audit=ap,policy="one first look, never fresh again"))
    sets={k:obj(proto/"blind"/(k+".json"),p["decks"][k]["queries"]) for k in ("train","test")}
    for qs in sets.values():
        require(len(qs)==64 and all(set(q)=={"id","prompt"} and q["id"]==task.identifier(q["prompt"]) and len(q["prompt"])==9 for q in qs) and
                [q["id"] for q in qs]==sorted({q["id"] for q in qs}),"score inputs contain labels or duplicate/mismatched rows")
    roles=a["roles"]; owners=sorted({v for v in roles.values() if v is not None});calls=0
    with (out/"logits.f32le").open("xb") as binary,(out/"batches.jsonl").open("xb") as log:
        for owner in owners:
            model=attach(st,base,geometry_path(trained,owner),owner)
            for split,qs in sets.items():
                for i in range(0,64,8):
                    part=qs[i:i+8];x=model.forward([token for q in part for token in q["prompt"]],[8,9],last_only=True)
                    require(x.shape==[8,1,64],"did not obtain full last-token logits")
                    block=bytes(x.values_buffer());require(len(block)==8*LOGITS.size and all(math.isfinite(y) for v in LOGITS.iter_unpack(block) for y in v),"invalid logits")
                    binary.write(block);record(log,dict(owner=owner,split=split,first=i,ids=[q["id"] for q in part],inputs=fingerprint(part),logits=sha(block)))
                    calls+=1
    return put(out/"complete.json",dict(plan=pp,training=tp,training_audit=ap,installed=admitted,roles=roles,owners=owners,forwards=calls,rows=8*calls,
                log=signature(out/"batches.jsonl"),binary=signature(out/"logits.f32le"),consumed=signature(out/"test-consumed.json")))


def claim(proto,phase,pp,out,parents):
    return dict(phase=phase,plan=pp,output_relative=os.path.relpath(out.resolve(),proto.resolve()),parents=parents,attempt=1)


def check_claim(proto,phase,pp,out,parents):
    require(json.loads((proto/(phase+"-claim.json")).read_bytes())==claim(proto,phase,pp,out,parents),"foreign or absent phase claim")


def run(stage,proto,pp,out,*args):
    p=obj(proto/"plan.json",pp);prev.installed_identity(inventory(proto,p))
    parents=list(args[1::2])
    if stage=="score":
        m=complete(Path(args[0]),args[1],"train",pp,[]);a=obj(Path(args[2])/"verified.json",args[3])
        require(a["training"]==args[1] and a["plan"]==pp and a["ends"]==m["ends"],"score admission failed")
    require(not out.exists(),"phase output already exists")
    # An alternate output directory must not silently grant a second look at this frozen deck.
    put(proto/(stage+"-claim.json"),claim(proto,stage,pp,out,parents))
    out.mkdir(mode=0o700);put(out/"attempt.json",dict(phase=stage,plan=pp,parents=parents,attempt=1))
    env=dict(os.environ,SPIRALTORCH_BACKEND="cpu",RAYON_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
             HF_HUB_OFFLINE="1",SPIRALTORCH_AUTOTUNE_STORE=str(out/"tuning.json"))
    cmd=[sys.executable,"-I","-B",str(Path(__file__).resolve()),"_"+stage,str(proto),pp,str(out),*map(str,args)]
    start=time.monotonic()
    with (out/"process.log").open("xb") as f:
        try:rc=subprocess.run(cmd,env=env,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=WALL[stage]).returncode
        except subprocess.TimeoutExpired:rc="timeout"
    elapsed=time.monotonic()-start;put(out/"exit.json",dict(exit=rc,elapsed=elapsed,log=signature(out/"process.log")))
    require(rc==0 and elapsed<=WALL[stage],"first failure retained; no retry")
    return signature(out/"complete.json")


if __name__=="__main__":
    from entity_swap_readback import audit_train, report
    cmd,*a=sys.argv[1:]
    if cmd=="freeze" and len(a)==4: print(freeze(Path(a[0]),a[1],Path(a[2]),Path(a[3])))
    elif cmd in ("train","_train","score","_score") and len(a)==(3 if "train" in cmd else 7):
        proto,pp,out=Path(a[0]).resolve(),a[1],Path(a[2]).resolve()
        extra=[Path(v).resolve() if i%2==0 else v for i,v in enumerate(a[3:])]
        if cmd.startswith("_"):
            import resource
            stage=cmd[1:];check_claim(proto,stage,pp,out,list(extra[1::2]))
            resource.setrlimit(resource.RLIMIT_CPU,(WALL[stage],WALL[stage]));fence(proto,stage)
            print((train_worker if stage=="train" else score_worker)(proto,pp,out,*extra))
        else: print(run(cmd,proto,pp,out,*extra))
    elif cmd=="audit-train" and len(a)==5:print(audit_train(Path(a[0]),a[1],Path(a[2]),a[3],Path(a[4])))
    elif cmd=="report" and len(a)==9:print(report(Path(a[0]),a[1],Path(a[2]),a[3],Path(a[4]),a[5],Path(a[6]),a[7],Path(a[8])))
    else:raise SystemExit("freeze PREV RESULT_SHA WHEEL OUT | train PROTO PP OUT | audit-train PROTO PP TRAIN TP OUT | score PROTO PP OUT TRAIN TP AUDIT AP | report PROTO PP TRAIN TP AUDIT AP SCORES SP OUT")
