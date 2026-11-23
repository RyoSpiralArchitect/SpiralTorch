"""Fresh matched prompts and independent checks on the actual final-query value source."""
import json
import math
from pathlib import Path
import runpy
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
prev=SimpleNamespace(**runpy.run_path(str(HERE.parent/"prefix-innovation/study.py")))
s,m,k,v,x,d,old,ck,cap,need=prev.s,prev.m,prev.k,prev.v,prev.x,prev.d,prev.old,prev.ck,prev.cap,prev.need
MODES=("attention","uniform","fixed2")
ROUTES={"attention":{"kind":"attention"},"uniform":{"kind":"uniform"},"fixed2":{"kind":"fixed","position":2}}
PARENT={"formal":"77a6b58a60dfa37ae49abde433a32bd3519a51e55193e8116c6d5ba978b1e309",
        "rehearsal":"2a794e64f68e250849d2260a36ba6f8b7f0d33cf0a12a43caaa0a533f354d17d"}
arm=prev.arm
paths=prev.paths
metric=prev.metric
witness=prev.witness

def cases(kind,test):
    out=[]
    for r in prev.cases(kind,test):
        z=r["prompt"];i=z.index(9);need(z[i:i+2]==[9,0],"wrong parent")
        out.append(dict(r,prompt=z[:i]+[0,9]+z[i+2:]))
    return out

def sources():
    paths=set(prev.sources())
    paths.update(str((HERE/n).relative_to(x.REPO)) for n in ("PROTOCOL.md","study.py","run.py","audit.py"))
    paths.update(("crates/st-nn/src/causal_lm/value_read.rs","crates/st-nn/src/geometry_runtime/value_read.rs",
        "crates/st-nn/src/causal_lm/decoder.rs","crates/st-nn/src/layers/channel_affine.rs",
        "crates/st-kernel-contracts/src/attention.rs","crates/st-kernel-contracts/src/attention/backward.rs",
        "bindings/st-py/src/geometry_runtime.rs","bindings/st-py/tests/test_native_attention_value_gate.py"))
    return {p:d.digest((x.REPO/p).read_bytes()) for p in sorted(paths)}

def freeze(root,parent):
    used=set();previous=set();plans={}
    for kind,pin in PARENT.items():
        info=x.read(parent/kind/"inputs.json",pin);sp=x.read(parent/kind/"training.json",info["train"])
        before=x.read(parent/kind/"transfer.json",info["transfer"])
        need(sp["train"]==prev.cases(kind,False) and before==prev.cases(kind,True),"wrong parent rows")
        for group in (sp["train"],before,s.cases(kind,False),s.cases(kind,True),x.rows(kind),x.rows(kind,True)):
            previous.update(tuple(r["prompt"]) for r in group)
        tr,te=cases(kind,False),cases(kind,True)
        need((sp["bases"],sp["orders"],sp["epochs"],len(tr),len(te))==
            (([557,563,569],[17,31,73],2,64,32) if kind=="formal" else ([571],[5],1,8,8)),"wrong schedule")
        for rows in (tr,te):
            ids={tuple(r["prompt"]) for r in rows};need(len(ids)==len(rows) and not used&ids,"duplicates");used.update(ids)
            for teacher in x.TEACHERS:x.pairs(rows,teacher)
        if kind=="formal":
            bare=lambda rows:{(r["frame"][0],r["frame"][2]) for r in rows}
            need(not bare(tr)&bare(te),"lexical leak")
        plans[kind]=(dict(sp,train=tr),te)
    need(not previous&used,"previously scored prompts")
    pins={}
    for kind,(sp,tr) in plans.items():
        out=root/kind;out.mkdir(mode=0o700)
        pins[kind]=d.save(out/"inputs.json",dict(parent=PARENT[kind],sources=sources(),
                                  train=d.save(out/"training.json",sp),transfer=d.save(out/"transfer.json",tr)))
    return pins

def data(folder,pin):
    inp=x.read(folder/"inputs.json",pin);sp=x.read(folder/"training.json",inp["train"])
    need(inp["parent"]==PARENT[sp["kind"]] and inp["sources"]==sources() and sp["train"]==cases(sp["kind"],False),"input drift")
    return inp,sp

def record(h,mode,vec):
    r=s.c.record(h,"identity",vec);recipe=r["placements"][0]["recipe"]
    del recipe["rotation"];recipe.update(schema="spiraltorch.attention_value_gate.v1",route=ROUTES[mode])
    return r

def state(raw,h,mode):
    r=json.loads(raw);ts=r["placements"][0]["tensors"]
    for t in ts:t["values"]=list(map(d.f32,t["values"]))
    vec=[y for t in ts for y in t["values"]]
    need(len(vec)==6 and vec[:3]==[0.]*3 and all(map(math.isfinite,vec)) and r==record(h,mode,vec),"state/recipe drift")
    return d.identity(raw),vec

def owner(model,h,mode):
    raw=model.geometry_artifact().to_json().encode();ident,vec=state(raw,h,mode)
    need((model.geometry_artifact_id,model.base_model_identity,model.frozen_lora_artifact_id,model.lora_artifact_id,
          model.frozen_lora_to_json(),model.readout_accumulator)==(ident,h["base"],h["frozen_lora"],None,h["lora_weights"],"f32"),
         "changed model or LoRA")
    return ident,vec,raw

def mix(tr):
    n,w=tr["shape"];heads=tr["heads"];dim=w//heads;lag=tr["key_lag"];vs=tr["values"]
    need(heads>=1 and w%heads==0 and len(vs)==len(tr["keys"])==n*w and len(tr["query"])==w and 0<=lag<n,"read shape")
    prob=[];att=[]
    for h in range(heads):
        channels=range(h*dim,(h+1)*dim);scores=[]
        for j in range(n):
            dot=0.
            for i in channels:dot=d.f32(dot+d.f32(tr["query"][i]*tr["keys"][max(0,j-lag)*w+i]))
            scores.append(d.f32(dot*tr["scale"]))
        mx=max(scores);ps=[math.exp(a-mx) for a in scores];z=math.fsum(ps);ps=[a/z for a in ps]
        prob.append(ps)
        att.extend(math.fsum(p*vs[j*w+i] for j,p in enumerate(ps)) for i in channels)
    return prob,att

def unit(tr,r,h,oid,mode):
    n=len(r["prompt"]);w=3
    need((tr["schema"],tr["site"],tr["base_model_identity"],tr["frozen_lora_artifact_id"],tr["geometry_artifact_id"],
          tr["readout_accumulator"],tr["route"],tr["strength"],tr["shape"],tr["key_lag"])==
          ("spiraltorch.last_mlp_value_read_signal.v1",x.s.SITE,h["base"],h["frozen_lora"],oid,"f32",ROUTES[mode],.25,[n,w],0),
          "wrong source or attribution owner")
    for f in ("query","keys","values","source","site_input","upstream","gate_gradient","bias_gradient"):
        need(all(math.isfinite(y) and y==d.f32(y) for y in tr[f]) and (f in ("keys","values") or len(tr[f])==w),"nonfinite/unrounded trace "+f)
    q=tr["contrast"]
    need((q["target_id"],q["alternative_id"],q["prompt_tokens"],q["target_minus_alternative"])==
         (7,8,n,q["target_logit"]-q["alternative_logit"]),"wrong readout")
    vs=tr["values"];ps,att=mix(tr)
    if mode=="attention":
        bound=4e-7+4e-6*max(map(abs,vs))
        need(all(abs(y-z)<=bound for y,z in zip(att,tr["source"])),"attention does not read Q/K-weighted V")
    elif mode=="uniform":
        need(tr["source"]==[d.f32(math.fsum(vs[i::w])/n) for i in range(w)],"uniform route")
    else:
        j=min(n-1,2)*w;need(tr["source"]==vs[j:j+w],"fixed route")
    for g,b,mu,u in zip(tr["gate_gradient"],tr["bias_gradient"],tr["source"],tr["upstream"]):
        need(b==d.f32(.25*u) and math.isclose(g,b*mu,rel_tol=2e-6,abs_tol=2e-8),"unit gate chain")
    return tr["source"]

def native_rows(st,model,examples):
    rt=st.NativeGeometryRuntime(model.geometry_artifact())
    return [rt.forward_value_read_sequence(x.s.SITE,tr["site_input"],tr["source"],[1,1]).output for tr in examples]

if __name__=="__main__":
    import sys
    print(json.dumps(freeze(*map(Path,sys.argv[1:])),sort_keys=True))
