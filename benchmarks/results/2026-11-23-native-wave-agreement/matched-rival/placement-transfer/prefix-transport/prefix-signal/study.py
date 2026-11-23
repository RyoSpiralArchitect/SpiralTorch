"""Frozen, synthetic training inputs and a write-once journal for the native prefix trace."""
import argparse
import importlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
p=SimpleNamespace(**runpy.run_path(str(HERE.parent/"owner-control/design.py")))
d,k,s,u,need=p.d,p.k,p.s,p.u,p.need
MASK={"formal":"29cac1ee5b5c832d69da9e1e8ddf1b81429fbf4523cb200a45b7df4cc94d8aaa",
      "rehearsal":"79971287c074eefd68f9092df124999470996c4dd0ddd593ecb5382a1f2d4c79"}
DELTA=1/128
RADIUS=.25


def pinned(path,pin):
    data=path.read_bytes()
    need(d.digest(data)==pin,"changed input: "+str(path))
    return data


def preparation(kind,parent,masked,extension):
    need(len(extension)==64 and all(c in "0123456789abcdef" for c in extension),"extension pin")
    spec=json.loads(pinned(parent/"spec.json",p.PARENT[kind][0]))
    prev=json.loads(pinned(parent/"complete.json",p.PARENT[kind][1]))
    masks=json.loads(pinned(masked/"complete.json",MASK[kind]))
    hosts=json.loads(pinned(parent/"attempt/hosts.json",prev["files"]["hosts.json"]))
    origins,refs={},{}
    train=spec["train"]
    for b in spec["bases"]:
        ref="frozen-%d.json"%b
        z=json.loads(pinned(parent/"attempt"/ref,prev["files"][ref]))
        refs[str(b)]={r["key"]:z[r["key"]] for r in train}
        pinned(parent/("base-%d.safetensors"%b),spec["weights"][str(b)])
        for turn in spec["turns"]:
            key=s.key(b,turn,"gate")
            f="weights/"+key+".json"
            dat=pinned(masked/"attempt"/f,masks["files"][f])
            ident,coords=p.state(dat,hosts[str(b)],turn,"gate")
            need(coords==[0.]*6,"not the original gate")
            origins[key]=ident
    repo=HERE.parents[6]
    files=[HERE/f for f in ("study.py","verify.py","PROTOCOL.md")]
    files += [repo/x for x in ("crates/st-nn/src/causal_lm/prefix_signal.rs","crates/st-nn/src/causal_lm/decoder.rs",
                               "crates/st-nn/src/geometry_runtime/prefix_transport.rs","bindings/st-py/src/causal_lm.rs")]
    return dict(schema="spiraltorch.prefix_signal_plan.v1",kind=kind,train=train,bases=spec["bases"],
                turns=spec["turns"],weights=spec["weights"],hosts=hosts,origins=origins,refs=refs,extension=extension,
                parent_spec=p.PARENT[kind][0],masked_manifest=MASK[kind],
                sources={f.relative_to(repo).as_posix():d.digest(f.read_bytes()) for f in files})


def perturb(model,host,turn,values,st):
    obj=p.artifact(host["base"],turn,"gate",[0.]*3+list(map(d.f32,values)))
    return model.with_geometry(st.GeometryAdapterArtifact.create_json(json.dumps(obj)))


def vectors(trace):
    h,w=trace["shape"]
    names={"site_input":h*w,"source_features":h*w,"upstream":w,"gate_gradient":w,"bias_gradient":w}
    for key,n in names.items():
        v=trace[key]
        need(len(v)==n and all(math.isfinite(x) and (key=="source_features" or d.f32(x)==x) for x in v),
             "non-finite or incomplete "+key)


def oracle_values(trace,head):
    margin=trace["contrast"]["target_minus_alternative"]
    g=trace["gate_gradient"]
    norm=k.norm(g)
    sign=1-2*head
    if sign*margin>=0 or norm==0:
        return None
    return [d.f32(sign*RADIUS*x/norm) for x in g]


def close(x,y,err,label):
    need(math.isfinite(x) and math.isfinite(y) and abs(x-y)<=err,label)


def validate(row,sp,case,b,turn):
    name=s.key(b,turn,"gate")
    host=sp["hosts"][str(b)]
    trace=row["trace"]
    need((row["base"],row["turn"],row["key"])==(b,turn,case["key"]),"journal order")
    need(trace["schema"]=="spiraltorch.last_mlp_prefix_margin_signal.v1" and trace["site"]==s.SITE
         and trace["rotation"]==turn and trace["shape"]==[len(case["prompt"]),3] and trace["strength"]==.25
         and trace["readout_accumulator"]=="f32" and trace["geometry_artifact_id"]==sp["origins"][name]
         and trace["base_model_identity"]==host["base"] and trace["lora_artifact_id"] is None
         and trace["frozen_lora_artifact_id"]==host["frozen_lora"],"factual owner/site")
    vectors(trace)
    c=trace["contrast"]
    z=sp["refs"][str(b)][case["key"]]
    need((c["target_id"],c["alternative_id"],c["target_logit"],c["alternative_logit"],c["target_minus_alternative"],
          c["prompt_tokens"],c["greedy_id"])==(7,8,z[7],z[8],z[7]-z[8],len(case["prompt"]),max(range(len(z)),key=z.__getitem__)),
         "different frozen logits or tokens")
    need(len(row["ce"])==2 and len(row["fd"])==3,"wrong numerical controls")
    for gold,m in enumerate(row["ce"]):
        k.check_grad(m,1.,gold,z)
    err={"add":0.,"ce":0.,"fd":0.}
    n=trace["shape"][0]
    for i in range(n):
        x=trace["site_input"][3*i:3*i+3]
        for _ in range((n-1-i)%4):
            if turn=="quarter_turn": x=[-x[1],x[0],x[2]]
            if turn=="inverse_quarter_turn": x=[x[1],-x[0],x[2]]
        for j in range(3):
            close(trace["source_features"][3*i+j],x[j]/n,1e-14,"lag decomposition")
    for j,(g,bias,up) in enumerate(zip(trace["gate_gradient"],trace["bias_gradient"],trace["upstream"])):
        need(bias==d.f32(.25*up),"chain rule at residual")
        rec=math.fsum(trace["source_features"][j::3])*bias
        ce=k.sub(row["ce"][1]["v"],row["ce"][0]["v"])
        fd=row["fd"][j]
        need(len(fd)==2,"missing central difference")
        central=(fd[1]-fd[0])/(2*DELTA)
        for label,obs,ref in (("add",rec,g),("ce",ce[j+3],g),("ce",ce[j],bias),("fd",central,g)):
            tol=2e-5+.02*abs(g) if label=="fd" else 2e-6*max(1.,abs(ref))
            close(obs,ref,tol,"inconsistent "+label+" derivative")
            err[label]=max(err[label],abs(obs-ref))
    vals=oracle_values(trace,case["head"])
    need((row["oracle"] is None) == (vals is None),"oracle census")
    if vals is not None:
        need(len(vals)==3 and abs(k.norm(vals)-RADIUS)<3e-8 and row["oracle"]["gate"]==vals
             and math.isfinite(row["oracle"]["gap"]),"wrong oracle perturbation")
    return err


def native(pin):
    need(os.environ.get("SPIRALTORCH_BACKEND")=="cpu" and all(os.environ.get(k)=="1" for k in
         ("OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","RAYON_NUM_THREADS")),"wrong CPU controls")
    def offline(event,args):
        if event in ("socket.connect","socket.getaddrinfo") or (event=="import" and args[0].split(".")[0] in
                {"torch","numpy","transformers","accelerate","safetensors","huggingface_hub"}):
            raise ValueError("offline native CPU only")
    sys.addaudithook(offline)
    import spiraltorch as st
    ext=importlib.import_module("spiraltorch.spiraltorch")
    need(st.NativeCausalLm is ext.NativeCausalLm and d.digest(Path(ext.__file__).read_bytes())==pin,"extension drift")
    return st


def run(root,parent,masked,pin):
    raw=pinned(root/"spec.json",pin);sp=json.loads(raw)
    need(raw==d.encoded(preparation(sp["kind"],parent,masked,sp["extension"])),"plan drift")
    out=root/"attempt";out.mkdir(mode=0o700)
    st=native(sp["extension"])
    case=s.fixtures_cached()[0]
    with (out/"rows.jsonl").open("xb") as journal:
        for b in sp["bases"]:
            host=sp["hosts"][str(b)]
            data=pinned(parent/("base-%d.safetensors"%b),sp["weights"][str(b)])
            base=st.NativeCausalLm.from_gpt2(case["config"],st.SafeTensorArchive.from_bytes(data)).with_readout_accumulator("f32")
            need(base.base_model_identity==host["original_base"],"wrong decoder")
            frozen=base.with_lora_json(host["lora_weights"]).freeze_lora()
            for turn in sp["turns"]:
                m=perturb(frozen,host,turn,[0.]*3,st)
                axes=[[perturb(frozen,host,turn,[side*DELTA*(i==j) for i in range(3)],st)
                       for side in (-1,1)] for j in range(3)]
                for sample in sp["train"]:
                    ids=sample["prompt"]
                    trace=m.last_mlp_prefix_margin_signal(ids,7,8)
                    need(m.geometry_artifact_id==trace["geometry_artifact_id"],"origin mutated")
                    def gap(model):
                        return model.next_token_contrast(ids,7,8)["target_minus_alternative"]
                    members=[u.old.handle(m,sample,j,1.)[1] for j in range(2)]
                    fd=[[gap(model) for model in pair] for pair in axes]
                    vals=oracle_values(trace,sample["head"])
                    oracle=None if vals is None else dict(gate=vals,gap=gap(perturb(frozen,host,turn,vals,st)))
                    row=dict(base=b,turn=turn,key=sample["key"],trace=trace,ce=members,fd=fd,oracle=oracle)
                    journal.write(d.encoded(row));journal.flush()
                    validate(row,sp,sample,b,turn)
    d.save(root/"complete.json",dict(spec=pin,files=k.identities(out)))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode",choices=("prepare","run"))
    cli.add_argument("kind",choices=list(MASK))
    for name in ("root","parent","masked"):
        cli.add_argument(name,type=Path)
    cli.add_argument("pin")
    q=cli.parse_args()
    if q.mode=="prepare":
        sp=preparation(q.kind,q.parent,q.masked,q.pin)
        q.root.mkdir(mode=0o700)
        print(d.save(q.root/"spec.json",sp))
    else:
        need(k.read(q.root/"spec.json")["kind"]==q.kind,"wrong worker kind")
        run(q.root,q.parent,q.masked,q.pin)
