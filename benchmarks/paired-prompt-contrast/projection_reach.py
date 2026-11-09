"""Fit local parameter directions on four training edges; measure both signs on distinct training edges."""
import io
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import context_token_training as base
import entity_swap_study as swap
import entity_swap_task as task
from alias_guarded_training import VECTOR, geometry_path
from alias_interference import signature
from alias_interference_audit import f32
from shared_prefix_study import obj,put,read,record,require,sha,write

KIND="spiraltorch.entity_projection_reach.v1"
W=32
SITES=("query","key","value","output")
ROLES=tuple(s+"/"+r for s in task.STATES for r in ("origin","direct_matched"))
RADII=(.002,.02)
RESOLVE,ATOL,REL=5e-6,1e-6,.10
WALL=180
VALUE=struct.Struct("<ffI")
MAX=dict(factual=256,dense_vjps=2048,diagonal_checks=32,geometry_ce=512,finite=19456)


def sites():
    return [(layer,site) for layer in range(2) for site in SITES]


def coordinates():
    return [f"{i}/{s}/{b}" for i,s in sites() for b in ("dense","diagonal")]+["geometry"]+[f"{i}/key/gauge" for i in range(2)]


def sources():
    return swap.sources()|{k:signature(Path(__file__).with_name(k+".py")) for k in ("projection_reach","projection_reach_audit")}


def freeze(previous,old_pp,tp,ap,ext,wheel_pin,wheel,out):
    p=obj(previous/"protocol/plan.json",old_pp)
    m=swap.complete(previous/"trained",tp,"train",old_pp,[]);a=obj(previous/"audit/verified.json",ap)
    require(p["kind"]==swap.KIND and p["code"]==swap.sources() and (a["plan"],a["training"],a["ends"])==(old_pp,tp,m["ends"]),"wrong training archive")
    data=obj(previous/"protocol/fit/task.json",p["fit"]);require(data==task.deck("train"),"not the 64 training prompts")
    inv=base.candidate_inventory(wheel,wheel_pin,ext)
    roles={k:a["roles"][k] for k in ROLES}
    require(None not in roles.values() and len(set(roles.values()))==4 and p["native"]!=ext,"need four specified checkpoints and the probe extension")
    out.mkdir(mode=0o700);model=out/"model";model.mkdir(mode=0o700)
    for name,key in (("config.json","config.json"),("model.safetensors","weights")):
        write(model/name,read(previous/"protocol/model"/name,p["files"][key],100000))
    for owner in roles.values():
        write(model/(owner[7:]+".json"),read(geometry_path(previous/"trained",owner),owner[7:],100000))
    # Extract only the training-origin CE witnesses. Never open prior keys, blind decks, logits or held-out reports.
    refs={s:{} for s in task.STATES}
    buf=io.BytesIO(read(previous/"trained/vectors.f32le",m["binary"],swap.LIMITS["backwards"]*VECTOR.size))
    ids=None
    for row in map(json.loads,io.BytesIO(read(previous/"trained/journal.jsonl",m["journal"],96<<20))):
        if row["event"]=="start":ids=row["ids"]
        if row["event"]!="gradients":continue
        block=buf.read(swap.NGRAD*VECTOR.size);require(sha(block)==row["sha"],"old training bytes differ")
        state,arm=row["key"].split("/")
        if arm=="retention" and row["t"]==0:
            require(row["owner"]==roles[state+"/origin"] and len(ids)==16,"wrong old gradient start")
            refs[state]={h:sha(block[(24+2*i)*VECTOR.size:(26+2*i)*VECTOR.size]) for i,h in enumerate(ids)}
    require(not buf.read(1) and all(len(v)==16 for v in refs.values()),"incomplete CE references")
    folds={k:put(out/(k+".json"),[r for r in data if r["edge"]%2==parity]) for k,parity in (("fit",0),("check",1))}
    return put(out/"plan.json",dict(kind=KIND,code=sources(),old_plan=old_pp,old_training=tp,old_audit=ap,native=ext,wheel=wheel_pin,
        inventory=put(out/"wheel-inventory.json",inv),base=p["base"],lora=p["lora"],files=p["files"],roles=roles,refs=refs,folds=folds,
        coordinates=coordinates(),radii=RADII,resolve=RESOLVE,atol=ATOL,rel=REL,wall=WALL,budget=MAX,retries=0,
        experiment="Both prior origins and their completed direct-matched geometries. At each: gradient of mean sign*(standby-on logit) "
            "on 4 even training edges defines each unit direction before reading the four odd training edges. Every edge has all 8 views. "
            "Each fold covers all 8 entities. Actual geometry: original bias/gate coordinates; virtual Q/K/V/output: matched input-major "
            "dense or diagonal projection weights plus bias. Store both f32 primal logits and f64 coordinate reductions. Fixed shared key-bias null.",
        gate="Each Q or K layer/basis independently in BOTH origins: on fit norm >2x the same-layer same-basis output (also actual geometry "
            "for diagonal). At h=.02, on both folds the central signed change AND predicted change exceed 5e-6 and the positive step improves I. "
            "On the check fold central change >2x max(5e-6, corresponding output change, also geometry for diagonal), positive check order means "
            "and >5e-6 on >=3/4 check edges. Both radii, both folds: abs(central-predicted)<=1e-6+0.10*abs(predicted). Gauge per-prompt "
            "drift <=2e-5 and central <=1e-6. Report h=.002 resolution separately, all sites even if no gate passes. Endpoint replication secondary.",
        negative="Zero fit gradient means no direction, no replacement. First error or resource limit stops, no retries or further scales. "
            "This is fixed-state local sensitivity on training inputs; no adapter/base updates, new held-out scores, quality or training claim."))


def admit(proto,p):
    require(p["kind"]==KIND and p["code"]==sources() and p["coordinates"]==coordinates() and set(p["roles"])==set(ROLES) and
            (p["radii"],p["budget"],p["resolve"],p["atol"],p["rel"],p["wall"],p["retries"])==(list(RADII),MAX,RESOLVE,ATOL,REL,WALL,0),"unfrozen probe")
    inv=obj(proto/"wheel-inventory.json",p["inventory"],2<<20)
    require((inv["sha256"],inv["native"]["sha256"])==(p["wheel"],p["native"]),"foreign wheel")
    return inv


def sign(row):
    a,b,qa,qb,q=(row["prompt"][i] for i in (1,4,2,5,7))
    require(q in (a,b) and {qa,qb}=={24,26} and a!=b,"invalid entity training row")
    sg=1 if (qa if q==a else qb)==26 else -1
    require(row["gold"]==(task.IDLE if sg==1 else task.ON),"wrong target")
    return sg


def fold_rows(rows,parity):
    require(len(rows)==32 and [r["id"] for r in rows]==sorted({r["id"] for r in rows}) and
            all(r["edge"]%2==parity and r["id"]==task.identifier(r["prompt"]) and r["prompt"]==task.render(r["entities"],r["order"],r["idle"],r["query"]) for r in rows) and
            len({(r["edge"],r["order"],r["idle"],r["query"]) for r in rows})==32 and
            {r["edge"] for r in rows}==set(range(parity,8,2)) and sum(map(sign,rows))==0,"wrong fold")


def nums(code,raw,n):
    require(len(raw)==struct.calcsize(code)*n,"unexpected coordinate storage")
    values=struct.unpack("<"+str(n)+code,raw);require(all(map(math.isfinite,values)),"nonfinite coordinates")
    return values


def decompose(block):
    dense=8*(W+W*W)*8
    require(len(block)==dense+2*VECTOR.size,"wrong VJP block")
    arr=nums("d",block[:dense],dense//8);result={};i=0
    for layer,site in sites():
        n=W+W*W;vs=arr[i:i+n];i+=n
        result[f"{layer}/{site}/dense"]=vs
        result[f"{layer}/{site}/diagonal"]=vs[:W]+tuple(vs[W+j*(W+1)] for j in range(W))
    on,idle=VECTOR.iter_unpack(block[dense:])
    result["geometry"]=tuple(x-y for x,y in zip(on,idle,strict=True))
    return result


def means(rows,blocks):
    ds=[decompose(b) for b in blocks];ss=[sign(r) for r in rows]
    require(len(ds)==len(ss)==32,"wrong number of VJPs")
    return {k:[math.fsum(s*x for s,x in zip(ss,col,strict=True))/len(ss) for col in zip(*(r[k] for r in ds),strict=True)] for k in ds[0]}


def directions(mu):
    out={}
    for k in coordinates():
        if k.endswith("/gauge"):out[k]=[((-1.)**j)/math.sqrt(W) for j in range(W)]+[0.]*W;continue
        length=math.hypot(*mu[k]);require(math.isfinite(length),"invalid construction gradient")
        out[k]=[x/length for x in mu[k]] if length else None
    return out


def one(model,p,role,row,first):
    prompt=row["prompt"];pair=(task.IDLE,task.ON);owner=p["roles"][role]
    primal=model.next_token_contrast(prompt,*pair);chunks=[]
    for layer,site in sites():
        kw=dict(layer=layer,site=site,basis="dense");r=model.projection_contrast_pullback(prompt,*pair,**kw)
        require((r["base_model_identity"],r["lora_artifact_id"],r["geometry_artifact_id"],r["readout_accumulator"],r["width"])==
                (p["base"],p["lora"],owner,"f32_wide_backward",W) and all(r[k]==v for k,v in kw.items()) and r["contrast"]==primal,"foreign site, logits or owner")
        b,w=r["bias_f64le"],r["weight_f64le"];nums("d",b,W);ws=nums("d",w,W*W)
        if first:
            diag=model.projection_contrast_pullback(prompt,*pair,layer=layer,site=site)
            require(diag["contrast"]==primal and diag["bias_f64le"]==b and nums("d",diag["weight_f64le"],W)==tuple(ws[j*(W+1)] for j in range(W)),"diagonal is not a slice of dense")
        chunks.extend((b,w))
    ce=[];raws=[]
    for k in (task.ON,task.IDLE):
        ids=prompt+[k];g=model.adaptation_loss_gradients(ids,[1,len(ids)],mode="geometry",labels=[-100]*(len(ids)-1)+[k])
        v=base.sealed(g,owner,p,1);raws.append(VECTOR.pack(*v["v"]));ce.append(v["loss"])
    require(abs(ce[0]-ce[1]-primal["target_minus_alternative"])<=base.CROSS_TOL,"geometry seed/readout mismatch")
    ref=p["refs"][role.split("/")[0]].get(row["id"]) if role.endswith("/origin") else None
    if ref is not None:require(sha(b"".join(raws))==ref,"archived word gradients no longer replay")
    return primal,ce,b"".join([*chunks,*raws])


def shifted(st,model,unit,h):
    rec=json.loads(model.geometry_artifact().to_json());site,=rec["placements"]
    by={v["name"]:v for v in site["tensors"] if v["requires_grad"]}
    require(set(by)=={"bias","gate"} and all(len(v["values"])==W for v in by.values()),"different trainable parameters")
    delta=[]
    for i,name in enumerate(("bias","gate")):
        xs=by[name]["values"]
        for j,x in enumerate(xs):
            x=f32(x);xs[j]=f32(x+h*unit[i*W+j]);delta.append(xs[j]-x)
    artifact=st.GeometryAdapterArtifact.create_json(json.dumps(rec,allow_nan=False))
    end=model.with_geometry(artifact)
    require(end.base_model_identity==model.base_model_identity and end.lora_artifact_id==model.lora_artifact_id,"perturbation changed owner")
    return end,delta


def worker(proto,pp,out):
    import resource
    claim=dict(plan=pp,attempt=1,output=os.path.relpath(out,proto))
    require(all(json.loads(f.read_bytes())==claim for f in (proto/"attempt.json",out/"attempt.json")),"unclaimed probe attempt")
    resource.setrlimit(resource.RLIMIT_CPU,(WALL,WALL))
    unlocked=False
    def fence(event,args):
        if event=="open" and isinstance(args[0],(str,bytes)):
            path=Path(os.path.realpath(os.fsdecode(args[0])))
            if (not unlocked and path==proto/"check.json") or path.name in {"truth.json","queries.json","answer-key.json"} or any(k in path.parts for k in ("keys","blind","scored")):
                raise PermissionError("input not admitted in this phase")
    sys.addaudithook(fence)
    p=obj(proto/"plan.json",pp);inv=admit(proto,p);installed=base.installed_identity(inv);st=base.native()
    require(st.NativeCausalLm is st._rs.NativeCausalLm and Path(st._rs.__file__).resolve()==Path(st.__file__).resolve().parent.parent/inv["native"]["path"],"foreign decoder")
    origin=base.open_model(st,proto/"model",p["files"])
    require((origin.base_model_identity,origin.lora_artifact_id)==(p["base"],p["lora"]),"wrong model")
    models={k:swap.attach(st,origin,proto/"model"/(v[7:]+".json"),v) for k,v in p["roles"].items()}
    counts=dict.fromkeys(MAX,0);folds={};dirs={}
    with (out/"linear.jsonl").open("xb") as log,(out/"pullbacks.bin").open("xb") as raw:
        for fold,parity in (("fit",0),("check",1)):
            rows=obj(proto/(fold+".json"),p["folds"][fold]);fold_rows(rows,parity);folds[fold]=rows
            for role in ROLES:
                blocks=[]
                for i,row in enumerate(rows):
                    verify_diag=fold=="fit" and i==0
                    primal,losses,block=one(models[role],p,role,row,verify_diag)
                    counts["factual"]+=1;counts["dense_vjps"]+=8;counts["geometry_ce"]+=2;counts["diagonal_checks"]+=8*verify_diag
                    raw.write(block);record(log,dict(role=role,fold=fold,id=row["id"],primal=primal,ce=losses,pullback=sha(block)))
                    if fold=="fit":blocks.append(block)
                if fold=="fit":dirs[role]=directions(means(rows,blocks))
            if fold=="fit":
                direction_pin=put(out/"directions.json",dict(plan=pp,fit=p["folds"]["fit"],roles=dirs));unlocked=True
    rows=folds["fit"]+folds["check"]
    (out/"geometry").mkdir(mode=0o700)
    with (out/"finite.jsonl").open("xb") as log,(out/"contrasts.bin").open("xb") as raw:
        for role in ROLES:
            model=models[role];owner=model.geometry_artifact_id
            for label in coordinates():
                unit=dirs[role][label]
                if unit is None:continue
                for radius in RADII:
                    for s in (-1,1):
                        h=s*radius
                        if label=="geometry":
                            end,delta=shifted(st,model,unit,h);perturbed=swap.retain(out,end)
                            def score(ids):return end.next_token_contrast(ids,task.IDLE,task.ON)
                        else:
                            layer,site,basis=label.split("/");delta=[f32(h*x) for x in unit];perturbed=None
                            kw=dict(layer=int(layer),site=site,basis="diagonal" if basis=="gauge" else basis,
                                    bias_f32le=struct.pack(f"<{W}f",*delta[:W]),weight_f32le=struct.pack(f"<{len(delta)-W}f",*delta[W:]))
                            def score(ids):return model.projection_perturbed_contrast(ids,task.IDLE,task.ON,**kw)
                        block=bytearray()
                        for row in rows:
                            v=score(row["prompt"]);a,b=v["target_logit"],v["alternative_logit"]
                            require((v["target_id"],v["alternative_id"],v["prompt_tokens"])==(task.IDLE,task.ON,9) and math.isfinite(a-b) and
                                    v["target_minus_alternative"]==a-b and 0<=v["greedy_id"]<64,"invalid finite contrast")
                            block.extend(VALUE.pack(a,b,v["greedy_id"]));counts["finite"]+=1
                        raw.write(block);record(log,dict(role=role,owner=owner,label=label,radius=radius,sign=s,temporary_geometry=perturbed,
                            actual_delta=sha(struct.pack(f"<{len(delta)}d",*delta)),values=sha(block)))
            require(model.geometry_artifact_id==owner and model.lora_artifact_id==p["lora"],"source changed")
    files={k:signature(out/k) for k in ("linear.jsonl","pullbacks.bin","finite.jsonl","contrasts.bin")}
    return put(out/"complete.json",dict(plan=pp,directions=direction_pin,installed=installed,counts=counts,files=files))


def once(proto,pp,out):
    p=obj(proto/"plan.json",pp);base.installed_identity(admit(proto,p))
    require(not out.exists(),"output already exists")
    claim=dict(plan=pp,attempt=1,output=os.path.relpath(out,proto))
    put(proto/"attempt.json",claim);out.mkdir(mode=0o700);put(out/"attempt.json",claim)
    env=dict(os.environ,SPIRALTORCH_BACKEND="cpu",HF_HUB_OFFLINE="1",RAYON_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1",
             OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",SPIRALTORCH_AUTOTUNE_STORE=str(out/"tuning.json"))
    start=time.monotonic()
    with (out/"process.log").open("xb") as f:
        try:rc=subprocess.run([sys.executable,"-I","-B",str(Path(__file__).resolve()),"_worker",str(proto),pp,str(out)],cwd=out,
                                env=env,stdout=f,stderr=subprocess.STDOUT,timeout=WALL).returncode
        except subprocess.TimeoutExpired:rc="timeout"
    elapsed=time.monotonic()-start;put(out/"exit.json",dict(exit=rc,elapsed=elapsed,log=signature(out/"process.log")))
    require(rc==0 and elapsed<=WALL,"first physical failure retained; no retry")
    return signature(out/"complete.json")


if __name__=="__main__":
    from projection_reach_audit import audit
    cmd,*a=sys.argv[1:]
    if cmd=="freeze" and len(a)==8:print(freeze(Path(a[0]),*a[1:6],Path(a[6]),Path(a[7])))
    elif cmd in ("run","_worker") and len(a)==3:print((once if cmd=="run" else worker)(Path(a[0]).resolve(),a[1],Path(a[2]).resolve()))
    elif cmd=="audit" and len(a)==5:print(audit(Path(a[0]),a[1],Path(a[2]),a[3],Path(a[4])))
    else:raise SystemExit("freeze OLD OLD_PLAN TRAIN_PIN TRAIN_AUDIT EXTENSION WHEEL_PIN WHEEL OUT | run PROTOCOL PIN OUT | audit PROTOCOL PIN MEASURED PIN OUT")
