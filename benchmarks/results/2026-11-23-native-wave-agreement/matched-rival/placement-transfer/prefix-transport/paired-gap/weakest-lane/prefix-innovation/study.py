"""Frozen synthetic data, artifact ownership and independent level/change equations."""
from itertools import product
import json
import math
from pathlib import Path
import runpy
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
def module(path):
    return SimpleNamespace(**runpy.run_path(str(path)))
m = module(HERE.parent / "metrics.py")
cap = module(HERE.parent / "local_capacity.py").common_direction
s,x,d,need = m.s,m.x,m.d,m.need
k,v,old,ck = s.k,s.v,s.old,s.ck
MODES = ("level","change")
PARENT = {"formal":"bdf037246e0836b07f993a04679cbb43d4eb3abb6657ccec27fc23c7a630f6a4",
          "rehearsal":"0a10562f01729ef4c4fd31514cafead11641581bd5e30babaf20f0584a094d47"}

def cases(kind,test):
    out=[]
    for r in s.cases(kind,test):
        z=r["prompt"]
        need(z[-2:]==[0,10] and z.count(9)==1,"unexpected prior template")
        z=z[:-2]+[10]; i=z.index(9)+1
        out.append(dict(r,prompt=z[:i]+[0]+z[i:]))
    return out

def sources():
    paths=set(s.sources())
    paths.update(str((HERE/f).relative_to(x.REPO)) for f in ("study.py","audit.py","run.py","PROTOCOL.md"))
    paths.update(str((HERE.parent/f).relative_to(x.REPO)) for f in ("metrics.py","local_capacity.py"))
    paths.update(("crates/st-nn/src/geometry_artifact.rs","crates/st-nn/src/geometry_runtime.rs",
                  "crates/st-nn/src/causal_lm/context.rs","crates/st-nn/src/causal_lm/prefix_signal.rs",
                  "bindings/st-py/tests/test_native_prefix_innovation.py"))
    return {p:d.digest((x.REPO/p).read_bytes()) for p in sorted(paths)}

def freeze(root,previous):
    seen=set(); origins=set(); plan={}
    for kind,parent in PARENT.items():
        rec=x.read(previous/kind/"inputs.json",parent)
        sp=x.read(previous/kind/"training.json",rec["training"])
        tr=x.read(previous/kind/"transfer.json",rec["transfer"])
        need(sp["train"]==s.cases(kind,False) and tr==s.cases(kind,True),"wrong predecessor")
        origins.update(tuple(r["prompt"]) for r in sp["train"]+tr)
        train,transfer=cases(kind,False),cases(kind,True)
        n=len(train)
        need((sp["bases"],sp["orders"],sp["epochs"],n,len(transfer))==
             (([557,563,569],[17,31,73],2,64,32) if kind=="formal" else ([571],[5],1,8,8)),"wrong workload")
        for rows in (train,transfer):
            ids={tuple(r["prompt"]) for r in rows}
            need(len(ids)==len(rows) and not seen.intersection(ids),"repeated prompt")
            seen.update(ids)
            for t in x.TEACHERS:x.pairs(rows,t)
        if kind=="formal":
            bare=lambda rows:{(r["frame"][0],r["frame"][2]) for r in rows}
            need(not bare(train)&bare(transfer),"leaking lexical pairs")
        plan[kind]=(dict(sp,train=train),transfer)
    need(not seen&origins,"reused a preceding prompt")
    pins={}
    for kind,(sp,tr) in plan.items():
        folder=root/kind;folder.mkdir(mode=0o700)
        pins[kind]=d.save(folder/"inputs.json",dict(parent=PARENT[kind],sources=sources(),
            train=d.save(folder/"training.json",sp),transfer=d.save(folder/"transfer.json",tr)))
    return pins

def data(folder,pin):
    inp=x.read(folder/"inputs.json",pin)
    sp=x.read(folder/"training.json",inp["train"])
    need(inp["parent"]==PARENT[sp["kind"]] and inp["sources"]==sources() and
         sp["train"]==cases(sp["kind"],False),"changed frozen training input")
    return inp,sp

def arm(b,mode,t=None,o=None):
    return "%d-%s" % (b,mode) + ("-origin" if t is None else "-%s-%d" % (t,o))

def paths(sp,b,mode):
    for t,o in product(x.TEACHERS,sp["orders"]):
        yield arm(b,mode,t,o),t,o

def record(host,mode,vec):
    need(mode in MODES,"unknown source")
    r=s.c.record(host,"identity",vec)
    if mode=="change":r["placements"][0]["recipe"]["schema"]="spiraltorch.prefix_transport_adapter.v3"
    return r

def state(raw,h,mode):
    r=json.loads(raw)
    tensors=r["placements"][0]["tensors"]
    for t in tensors:t["values"]=[d.f32(y) for y in t["values"]]
    vec=[y for t in tensors for y in t["values"]]
    need(len(vec)==6 and vec[:3]==[0.]*3 and all(map(math.isfinite,vec)) and r==record(h,mode,vec),
         "wrong recipe, parameter owner or frozen bias")
    return d.identity(raw),vec

def owner(model,h,mode):
    raw=model.geometry_artifact().to_json().encode()
    ident,params=state(raw,h,mode)
    need((model.geometry_artifact_id,model.base_model_identity,model.frozen_lora_artifact_id,
          model.lora_artifact_id,model.readout_accumulator,model.frozen_lora_to_json())==
         (ident,h["base"],h["frozen_lora"],None,"f32",h["lora_weights"]),"model provenance")
    return ident,params,raw

def feature(site,mode):
    """Quadratic lag oracle: no streaming state, no native attribution or metric."""
    n=len(site)//3
    need(n>0 and len(site)==3*n and mode in MODES,"feature shape")
    rows=[]
    for i in range(n):
        row=site[3*i:3*i+3]
        if mode=="change":
            row=[v-site[3*(i-1)+j] for j,v in enumerate(row)] if i else [0.]*3
        a,b,c=row
        for _ in range(n-1-i):a,b=-b,a
        rows.append([a/n,b/n,c/n])
    return [v for row in rows for v in row],[d.f32(math.fsum(col)) for col in zip(*rows)]

def unit(tr,r,h,ident,mode):
    n=len(r["prompt"]); site=tr["site_input"];q=tr["contrast"]
    need((tr["base_model_identity"],tr["geometry_artifact_id"],tr["frozen_lora_artifact_id"],
          tr["lora_artifact_id"],tr["site"],tr["readout_accumulator"],tr["strength"],tr["shape"],tr["rotation"])==
         (h["base"],ident,h["frozen_lora"],None,x.s.SITE,"f32",.25,[n,3],"quarter_turn"),"unit owner")
    need((tr["schema"],tr.get("source_encoding"))==
         (("spiraltorch.last_mlp_prefix_margin_signal.v1",None) if mode=="level" else
          ("spiraltorch.last_mlp_prefix_margin_signal.v2","adjacent_difference")) and "output_mix" not in tr,"signal schema")
    need((q["target_id"],q["alternative_id"],q["prompt_tokens"])==(7,8,n) and
         q["target_minus_alternative"]==q["target_logit"]-q["alternative_logit"],"contrast")
    need(len(site)==n*3 and all(len(tr[f])==3 for f in ("gate_gradient","bias_gradient","upstream"))
         and all(math.isfinite(z) for f in ("site_input","gate_gradient","bias_gradient","upstream") for z in tr[f]),"invalid trace")
    src,mean=feature(site,mode)
    need(tr["source_features"]==src,"incorrect per-position decomposition")
    for g,b,u,mu in zip(tr["gate_gradient"],tr["bias_gradient"],tr["upstream"],mean):
        need(b==d.f32(.25*u) and math.isclose(g,b*mu,rel_tol=2e-6,abs_tol=2e-8),"unit derivative")
    return mean

def metric(report,means,h,ident):
    q=[math.fsum((.25*y)**2 for y in col)/len(col) for col in zip(*means)]
    need((report["samples"],report["anchor"],report["base_model_identity"],report["frozen_lora_artifact_id"])==
         (len(means),ident,h["base"],h["frozen_lora"]) and len(report["second_moments"])==3 and
         all(a>0 and math.isclose(a,b,rel_tol=1e-14) for a,b in zip(q,report["second_moments"])),"wrong metric")
    return q

def witness(w,p0,p1,traces,means,st):
    def outputs(p):
        return [[d.f32(raw+d.f32(.25*d.f32(g*y))) for raw,g,y in zip(tr["site_input"][-3:],p[3:],mu)]
                for tr,mu in zip(traces,means)]
    need(s.c.same_rows(w["before"],outputs(p0)) and s.c.same_rows(w["after"],outputs(p1)) and
         abs(v.rms_rows(w["before"],w["after"])-st["realized_rms"])<=2e-6,"native physical witness")

if __name__=="__main__":
    import sys
    print(json.dumps(freeze(*map(Path,sys.argv[1:])),sort_keys=True))
