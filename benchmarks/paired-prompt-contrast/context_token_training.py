"""Train on matched *training* contexts. The consumed transfer deck is never an input."""
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(1,str(REPO/"tools"))
from byte_release_artifacts import installed_identity, wheel_archive_inventory
from alias_guarded_training import VECTOR, fingerprint, retain, sources as prior_sources
from alias_individual_step import FLOOR
from alias_interference import STATES, signature
from alias_interference_audit import f32, norm
from alias_protection import CROSS_TOL, MARGIN, MENU, RATE, RADIUS, STOP, native, open_model, scores, sealed
from shared_prefix_study import obj, put, read, record, require, sha, write

ARMS=("guard_only","positive_ce","matched_pair","rotated_pair")
KIND="spiraltorch.train_context_token_preference.v2"
STEPS,WALL,TARGET=64,180,16
GROUPS=66
EFFECT=.01
BUDGET=dict(path_count=8,max_steps=512,max_backwards=33792,max_score_menus=9360,max_score_singletons=3120,
            max_proposals=512,max_guards=512)
REFUSALS={"proposal":{
    "common descent requires nonzero finite group means", "no numerically resolved strict common geometry direction",
    "common geometry direction failed its normalized margin certificate", "common geometry step is invisible at model precision",
    "rounded geometry proposal does not strictly descend for every group", "cannot assign a positive distance to a zero geometry direction",
    "requested geometry direction or distance cannot be represented accurately in f32",
}, "guard":{
    "geometry guard requires nonzero finite group means", "no feasible interval on this geometry guard segment",
    "no open feasible interval on this geometry guard segment", "unresolved geometry guard direction",
    "geometry guard direction or length is not representable in f32", "rounded geometry guard step misses a group's required cosine",
}}


def sources():
    return {**prior_sources(),"context_token_training":signature(Path(__file__)),
            "context_token_training_audit":signature(Path(__file__).with_name("context_token_training_audit.py")),
            "byte_release_artifacts":signature(REPO/"tools/byte_release_artifacts.py")}


def candidate_inventory(wheel, wheel_pin, extension):
    inventory=wheel_archive_inventory(wheel)
    require((inventory["sha256"],inventory["native"]["sha256"])==(wheel_pin,extension),
            "wheel archive and tested extension are not the pinned pair")
    for name in ("spiraltorch/__init__.py","spiraltorch/__init__.pyi"):
        require(inventory["files"].get(name,{}).get("sha256")==signature(REPO/"bindings/st-py"/name),
                "wheel Python facade or type stub differs from the study source")
    return inventory


def frozen_inventory(protocol,p):
    require(p["kind"]==KIND,"expected a plan bound to actual wheel bytes")
    inventory=obj(protocol/"wheel-inventory.json",p["inventory"],2<<20)
    require((inventory["sha256"],inventory["native"]["sha256"])==(p["wheel"],p["native"]),
            "wheel inventory differs from plan")
    return inventory


def pairs(data):
    triples=[{} for _ in range(6)]
    for i,e in enumerate(data):
        prompt=e["prompt"]; c=e["meaning"]
        require(isinstance(prompt,list) and len(prompt)==5 and all(type(x) is int for x in prompt) and type(c) is int,
                "malformed training prompt or class")
        b=prompt[1]-32
        require(0<=b<6 and c in (0,1,2) and prompt==[49,32+b,24+c,27+b%2,29+c] and c not in triples[b],
                "not the fixed training roster")
        triples[b][c]=i
    require(len(data)==18 and all(set(t)=={0,1,2} for t in triples),"incomplete training triplets")
    slots=[[t[c] for c in range(3)] for t in triples]
    return slots, {name:[[slots[b][2],slots[(b+offset)%6][c]] for b in range(6) for c in (0,1)]
                  for name,offset in (("matched_pair",0),("rotated_pair",2))}


def training_reference(trained,fixture,readback_pin):
    rb=obj(trained/"readback.json",readback_pin)
    p=obj(trained/"protocol/plan.json",rb["pins"]["plan"])
    m=obj(trained/"measured/complete.json",rb["pins"]["measurement"])
    meta=obj(trained/"measured/part0/complete.json",m["parts"][0])
    require(p["code"]==prior_sources() and m["plan"]==meta["plan"]==rb["pins"]["plan"] and p["split"]==STEPS and
            p["radius"]==RADIUS and p["floor"]==FLOOR, "wrong archived training trajectory")
    refs={s:[] for s in STATES}
    for raw in io.BytesIO(read(trained/"measured/part0/steps.jsonl",meta["files"]["steps.jsonl"],32<<20)):
        r=json.loads(raw)
        if r["event"]=="adopt" and r["arm"]=="individual_guard":
            require(r["t"]==len(refs[r["state"]]),"control not sequential")
            refs[r["state"]].append(r["proof"])
    require(all(len(v)==STEPS for v in refs.values()),"missing reference steps")
    fx=obj(fixture/"fixture.json",p["fixture"])
    data=obj(fixture/"train.json",fx["files"]["train.json"])
    triples,roster=pairs(data)
    require(fx["menu"]==MENU and MENU[5]==[23,TARGET] and fx["stop"]==STOP and
            p["groups"]==[[i for i,e in enumerate(data) if e["meaning"]==c] for c in range(3)],"wrong training roster or word")
    return rb["pins"]["plan"],p,refs,fx["files"],triples,roster


def freeze(trained,fixture,readback_pin,extension,wheel_pin,wheel,out):
    inventory=candidate_inventory(wheel,wheel_pin,extension)
    prior,p,refs,files,triples,roster=training_reference(trained,fixture,readback_pin)
    out.mkdir(mode=0o700); inputs=out/"inputs"; inputs.mkdir(mode=0o700)
    for f,k in (("config.json","config.json"),("model.safetensors","weights"),("train.json","train.json")):
        write(inputs/f,read(fixture/f,files[k],100000))
    for s,id_ in p["origins"].items():
        write(inputs/(s+".json"),read(trained/"protocol"/(s+".json"),id_[7:],100000))
    return put(out/"plan.json",dict(kind=KIND,code=sources(),native=extension,wheel=wheel_pin,
        inventory=put(out/"wheel-inventory.json",inventory),
        prior_readback=readback_pin,prior=prior,refs=refs,base=p["base"],lora=p["lora"],origins=p["origins"],
        files={k:files[k] for k in ("config.json","weights","train.json")},groups=p["groups"],triples=triples,pairs=roster,
        states=STATES,arms=ARMS,steps=STEPS,rate=RATE,radius=RADIUS,floor=FLOOR,margin=0.,effect=EFFECT,eps=MARGIN,
        max_budget=BUDGET,wall=WALL,retries=0,
        change="All arms compute the same 18 class, 6 absolute line, 18 word CE and 24 context-pair handles per step. "
               "Only the fifth proposal group differs. Positive CE uses the same six positives twice; rotated pairs "
               "use a parity-preserving two-entity offset and the exact same positive and negative marginals. "
               "All use the SAME nine-objective post-rounding guard, not an extra treatment-only guard.",
        primary="Both starts, final step 64 only: matched must increase mean lexical selectivity by >0.01 nats vs "
                "its origin AND versus each of guard-only and positive CE, with positive differences >5e-6 in at "
                "least 4/6 same-entity blocks for each contrast. Separately count finite nine-objective steps.",
        secondary="Separate gates: lexical matching benefit vs rotated; selectivity of the stopped line (same criterion); "
                  "absolute containment from origin (correct log gain >5e-6 and both wrong class means <=5e-6). "
                  "A better relative gap is not proof of suppression.",
        negative="First native numerical proposal/guard refusal terminates that path with its evidence. Other fixed arms "
                 "still run; no fallback, resize or retry. Any other error stops the entire run. A required incomplete "
                 "path cannot pass its gate. Every score is on the 18 training prompts only; no held-out access."))


def nll(v):
    a=min(v); return a-math.log(math.fsum(math.exp(a-x) for x in v))


def score(model,data,counts):
    rows=[]
    for i,e in enumerate(data):
        c=e["meaning"];r=model.score_lines(e["prompt"],MENU,stop_id=STOP)
        counts["score_menus"]+=1
        row=dict(slot=i,lines=r["line_nlls"],terms=r["token_nlls"][5],conditional=r["conditional_nlls"],
                 cost=nll([r["conditional_nlls"][j] for j in (c,c+3)]))
        require(len(row["terms"])==3 and abs(math.fsum(row["terms"])-row["lines"][5])<1e-11,"wrong stopped line")
        if c==2:
            row["single"]=model.line_set_loss(e["prompt"],[MENU[5]],stop_id=STOP)["set_nll"]
            counts["score_singletons"]+=1
            require(abs(row["single"]-row["lines"][5])<=CROSS_TOL,"singleton layout drift")
        rows.append(row)
    return rows


def check_seed(actual,pos,neg,scale):
    expected=[scale*(x-y) for x,y in zip(pos,neg,strict=True)]
    err=[x-y for x,y in zip(actual,expected,strict=True)]
    high=max(map(abs,err)); l2=norm(err)
    require(high<=2e-6+1e-4*max(map(abs,expected)) and l2<=2e-5+3e-4*norm(expected),"signed VJP differs from independent single CEs")
    return high,l2


def gradients(model,p,data,before,counts):
    owner=model.geometry_artifact_id
    hs=[]; ts=[]; ws=[]; allv=[]; tail=[]; words=[]; loss=[]; tl=[]; wl=[]; original=[]
    for i,e in enumerate(data):
        prompt=e["prompt"]; c=e["meaning"]; ids=[c,c+3]
        g=model.geometry_line_subset_gradients(prompt,MENU,accepted_indices=ids,stop_id=STOP); counts["backwards"]+=1
        v=sealed(g,owner,p,18); r=dict(slot=i,**scores(g.line_subset_report(),ids))
        require(r["lines"]==before[i]["lines"] and abs(r["cost"]-before[i]["cost"])<1e-11,"class readout")
        hs.append(g); allv.append(v["v"]); loss.append(v["loss"])
        if c==2:
            g=model.geometry_line_set_gradients(prompt,[MENU[5]],stop_id=STOP); counts["backwards"]+=1
            v=sealed(g,owner,p,3); r["single"]=g.line_set_report()["set_nll"]
            require(r["single"]==before[i]["single"],"line readout")
            ts.append(g);tail.append(v["v"]);tl.append(v["loss"])
        original.append(r)
        ids=prompt+MENU[5]
        g=model.adaptation_loss_gradients(ids,[1,len(ids)],mode="geometry",labels=[-100]*(len(ids)-1)+[TARGET])
        counts["backwards"]+=1; v=sealed(g,owner,p,1)
        require(abs(v["loss"]-before[i]["terms"][1])<=CROSS_TOL,"word CE does not score the lexical token")
        ws.append(g); words.append(v["v"]); wl.append(v["loss"])
    allv.extend(tail); allv.extend(words); loss.extend(tl); loss.extend(wl)
    original_block=b"".join(VECTOR.pack(*v) for v in allv[:24])
    branches={"positive_ce":[ws[pos] for pos,_ in p["pairs"]["matched_pair"]]}
    reports={}
    for name in ("matched_pair","rotated_pair"):
        branches[name]=[]; reports[name]=[]
        for pos,neg in p["pairs"][name]:
            a,b=(data[j]["prompt"]+[23] for j in (pos,neg))
            g=model.geometry_context_token_preference_gradients(a,b,TARGET,margin=0.); counts["backwards"]+=1
            v=sealed(g,owner,p,2); r=g.context_token_preference_report()
            require(r["target_id"]==TARGET and r["vocabulary"]==64 and r["margin"]==0. and f32(r["loss"])==g.loss and
                    abs(r["preferred_nll"]-before[pos]["terms"][1])<=CROSS_TOL and
                    abs(r["other_nll"]-before[neg]["terms"][1])<=CROSS_TOL,"foreign pair readout")
            check_seed(v["v"],words[pos],words[neg],r["gradient_weight"])
            branches[name].append(g);reports[name].append(r);allv.append(v["v"]);loss.append(v["loss"])
    require(len(allv)==GROUPS and len(loss)==GROUPS,"wrong gradient allocation")
    classes=[[hs[i] for i in ids] for ids in p["groups"]]
    return classes,ts,branches,loss,reports,b"".join(VECTOR.pack(*v) for v in allv),sha(original_block),fingerprint(original)


def worker(protocol,pp,out):
    import resource
    resource.setrlimit(resource.RLIMIT_CPU,(WALL,WALL))
    def no_eval(event,args):
        if event=="open" and isinstance(args[0],(str,bytes)):
            path=Path(os.fsdecode(args[0]))
            if path.name in ("truth.json","queries.json","answer-key.json") or "2026-11-04-native-alias-fresh-context-transfer-v1" in path.parts:
                raise PermissionError("no evaluation files in training")
    sys.addaudithook(no_eval)
    p=obj(protocol/"plan.json",pp); inputs=protocol/"inputs"
    require(sources()==p["code"],"source changed")
    data=obj(inputs/"train.json",p["files"]["train.json"]); triples,roster=pairs(data)
    require(triples==p["triples"] and roster==p["pairs"] and p["groups"]==[[i for i,e in enumerate(data) if e["meaning"]==c] for c in range(3)],"wrong pairs")
    inventory=frozen_inventory(protocol,p)
    admitted=installed_identity(inventory)
    st=native()
    package_root=Path(st.__file__).resolve(strict=True).parent.parent
    require(Path(st._rs.__file__).resolve(strict=True)==package_root/inventory["native"]["path"] and
            st.NativeCausalLm is st._rs.NativeCausalLm,"not the admitted native decoder")
    base=open_model(st,inputs,p["files"]); frozen=base.lora_to_json()
    require((base.base_model_identity,base.lora_artifact_id)==(p["base"],p["lora"]),"wrong model")
    counts=dict.fromkeys(("backwards","score_menus","score_singletons","proposals","guards","adopted"),0)
    ends={}; (out/"geometry").mkdir(mode=0o700)
    with (out/"journal.jsonl").open("xb") as f,(out/"vectors.f32le").open("xb") as binary:
        for s in STATES:
            source=p["origins"][s]
            origin=base.with_geometry(st.GeometryAdapterArtifact.from_json(read(inputs/(s+".json"),source[7:],100000).decode(),source,p["base"]))
            require(retain(out,origin)==source,"source identity")
            for arm in ARMS:
                model=origin; rows=score(model,data,counts); key=s+"/"+arm; adopted=0; reason=None
                record(f,dict(event="origin",key=key,owner=source,rows=rows))
                for t in range(STEPS):
                    owner=model.geometry_artifact_id; tags=dict(key=key,t=t,owner=owner)
                    record(f,dict(event="start",**tags))
                    classes,tail,branches,loss,reports,raw,og,oscore=gradients(model,p,data,rows,counts)
                    binary.write(raw)
                    record(f,dict(event="gradients",**tags,sha=sha(raw),loss=loss,reports=reports,old_gradient=og,old_scores=oscore))
                    ref=p["refs"][s][t]
                    if arm=="guard_only" or owner==ref["pre"]:
                        require((owner,og,oscore)==(ref["pre"],ref["gradient"],ref["before"]),"archived input drift")
                    phase="proposal"
                    try:
                        groups=[*classes,tail]+([] if arm=="guard_only" else [branches[arm]])
                        counts["proposals"]+=1
                        cand,r=model.step_common_geometry_sgd(groups,RATE,target_l2=RADIUS)
                        require(retain(out,cand)==r["post_geometry_id"] and r["pre_geometry_id"]==owner,"candidate identity")
                        record(f,dict(event="candidate",**tags,receipt=r))
                        phase="guard"
                        counts["guards"]+=1
                        selected,g=model.guard_geometry_step(cand,[*classes,*([h] for h in tail)],min_cosine=FLOOR)
                    except ValueError as ex:
                        if str(ex) not in REFUSALS[phase]: raise
                        reason=dict(phase=phase,message=str(ex))
                        record(f,dict(event="refusal",**tags,reason=reason))
                        break
                    end=retain(out,selected)
                    require((g["pre_geometry_id"],g["candidate_geometry_id"],g["post_geometry_id"])==(owner,cand.geometry_artifact_id,end)
                            and model.geometry_artifact_id==owner and selected.lora_to_json()==frozen and selected.base_model_identity==p["base"],
                            "wrong protected owner")
                    if arm=="guard_only":
                        require((end,fingerprint(r),fingerprint(g))==(ref["chosen"],ref["proposal"],ref["guard"]),"baseline drift")
                    rows=score(selected,data,counts)
                    record(f,dict(event="adopt",**tags,receipt=g,rows=rows))
                    adopted+=1;counts["adopted"]+=1;model=selected
                ends[key]=dict(adopted=adopted,owner=model.geometry_artifact_id,refusal=reason)
    return put(out/"measured.json",dict(plan=pp,native=p["native"],installed=admitted,counts=counts,ends=ends,journal=signature(out/"journal.jsonl"),
                                       binary=signature(out/"vectors.f32le")))


def once(protocol,pp,out):
    p=obj(protocol/"plan.json",pp)
    require(p["code"]==sources() and p["max_budget"]==BUDGET and p["wall"]==WALL and p["retries"]==0,"wrong frozen run")
    # Admission failures are not physical trials. The fresh worker rechecks its own environment.
    installed_identity(frozen_inventory(protocol,p))
    out.mkdir(mode=0o700);put(out/"attempt.json",dict(plan=pp,attempt=1,native=p["native"]))
    env=dict(os.environ,SPIRALTORCH_BACKEND="cpu",HF_HUB_OFFLINE="1",RAYON_NUM_THREADS="1",OMP_NUM_THREADS="1",
             VECLIB_MAXIMUM_THREADS="1",OPENBLAS_NUM_THREADS="1",SPIRALTORCH_AUTOTUNE_STORE=str(out/"tuning.json"))
    cmd=[sys.executable,"-I","-B",str(Path(__file__).resolve()),"_worker",str(protocol),pp,str(out)]
    start=time.monotonic()
    with (out/"process.log").open("xb") as f:
        try: rc=subprocess.run(cmd,cwd=out,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=WALL).returncode
        except subprocess.TimeoutExpired: rc="timeout"
    elapsed=time.monotonic()-start
    put(out/"exit.json",dict(exit=rc,elapsed=elapsed,log=signature(out/"process.log")))
    require(rc==0 and elapsed<=WALL,"first unexpected failure retained; no retry")
    return signature(out/"measured.json")


if __name__=="__main__":
    from context_token_training_audit import audit
    cmd,*a=sys.argv[1:]
    if cmd=="freeze" and len(a)==7:print(freeze(Path(a[0]),Path(a[1]),*a[2:5],Path(a[5]),Path(a[6])))
    elif cmd in ("run","_worker") and len(a)==3:print((once if cmd=="run" else worker)(Path(a[0]).resolve(),a[1],Path(a[2]).resolve()))
    elif cmd=="audit" and len(a)==5:print(audit(Path(a[0]),a[1],Path(a[2]),a[3],Path(a[4])))
    else:raise SystemExit("freeze PRIOR_TRAIN FIXTURE TRAIN_READBACK EXTENSION_SHA WHEEL_SHA WHEEL_FILE OUT | run PLAN PIN OUT | audit PLAN PIN MEASURED MP OUT")
