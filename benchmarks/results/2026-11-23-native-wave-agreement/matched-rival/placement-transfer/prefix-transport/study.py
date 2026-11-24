"""Frozen CPU prefix-transport pilot and original-byte journal. No model at import."""
import argparse
import importlib
from itertools import product
import json
import math
import os
from pathlib import Path
import runpy
import struct
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PREV = HERE.parent / "conditional-update"
old = SimpleNamespace(**runpy.run_path(str(PREV / "run.py")))
k, s, d, need = old.k, old.s, old.d, old.need
TURNS = ("identity", "quarter_turn", "inverse_quarter_turn")


def plan(formal, extension):
    need(len(extension)==64 and all(c in "0123456789abcdef" for c in extension), "native extension id")
    ds = d.design(formal)
    bases = [557,563,569] if formal else [571]
    repo = HERE.parents[5]
    files = ([HERE / f for f in ("PROTOCOL.md","study.py","verify.py")] +
             [PREV / f for f in ("design.py","run.py","analyze.py")] +
             [HERE.parent/"answer-scope"/f for f in ("study.py","run.py")] +
             [HERE.parent/f for f in ("design.py","analyze.py")] +
             [repo/"crates/st-nn/src"/f for f in ("geometry_artifact.rs","geometry_runtime.rs",
                "geometry_runtime/prefix_transport.rs","causal_lm/decoder.rs","causal_lm/context.rs",
                "causal_lm/token_menu.rs","causal_lm/geometry_cohort.rs")])
    return dict(schema="spiraltorch.prefix_transport_study.v1", formal=formal, bases=bases, turns=TURNS,
                orders=ds["orders"], epochs=ds["epochs"], train=ds["train"], test=ds["test"],
                extension=extension, fixture_hashes=d.DATA,
                weights={str(b):d.digest(s.generate(s.fixtures_cached()[2],b)) for b in bases},
                sources={f.relative_to(repo).as_posix():d.digest(f.read_bytes()) for f in files})


def prepare(root, formal, extension):
    root.mkdir(mode=0o700)
    sp=plan(formal,extension)
    for b in sp["bases"]:
        k.native.raw_new(root/("base-%d.safetensors"%b),s.generate(s.fixtures_cached()[2],b))
    print(d.save(root/"spec.json",sp))


def recipe(turn):
    need(turn in TURNS, "unknown transport")
    return dict(schema="spiraltorch.prefix_transport_adapter.v1",features=3,strength=.25,max_values=64,rotation=turn)


def record(base,turn,coords):
    need(len(coords)==6,"six coordinates")
    return dict(schema="spiraltorch.geometry_adapter_artifact.v1",base_model_identity=base,
                policy="full_prefix_f32_no_cache.v1",training=True,
                placements=[dict(path=s.SITE,training=True,recipe=recipe(turn),
                  tensors=[dict(name=name,shape=[3],values=coords[i:i+3],role="parameter",requires_grad=True)
                           for name,i in (("bias",0),("gate",3))])])


def coordinates(raw,turn,host):
    data=json.loads(raw)
    p,=data["placements"]
    vec=[d.f32(v) for t in p["tensors"] for v in t["values"]]
    need(len(vec)==6 and all(math.isfinite(v) for v in vec),"nonfinite or missing coordinate")
    # A shortest decimal in Rust's canonical JSON denotes f32, not necessarily that exact Python f64.
    for t in p["tensors"]:
        t["values"]=[d.f32(v) for v in t["values"]]
    need(data==record(host["base"],turn,vec),"changed weights/placement contract")
    return vec


def weights(m,turn,host):
    raw=m.geometry_artifact().to_json().encode()
    need(m.geometry_artifact_id==d.identity(raw) and m.base_model_identity==host["base"] and
         m.readout_accumulator=="f32" and m.lora_artifact_id is None and
         m.frozen_lora_artifact_id==host["frozen_lora"] and m.frozen_lora_to_json()==host["lora_weights"],
         "frozen base or arithmetic drift")
    return raw,m.geometry_artifact_id,coordinates(raw,turn,host)


def cells(sp):
    return product(sp["bases"],sp["turns"],d.TEACHERS,sp["orders"])


def logits(m,rows):
    return {r["key"]:m.forward(r["prompt"],[1,len(r["prompt"])],last_only=True).values() for r in rows}


def bits(values):
    need(len(values)==11 and all(math.isfinite(v) and v==d.f32(v) for v in values),"not 11 f32 logits")
    return struct.pack("<11f",*values)


def capture(m,turn,host,folder,key,rows):
    raw,pin,vec=weights(m,turn,host)
    k.native.raw_new(folder/"weights"/(key+".json"),raw)
    z=logits(m,rows)
    for v in z.values(): bits(v)
    d.save(folder/"logits"/(key+".json"),dict(owner=pin,rows=z))
    return pin,vec,z


def run(root,pin):
    raw=(root/"spec.json").read_bytes();sp=json.loads(raw)
    need(d.digest(raw)==pin and raw==d.encoded(plan(sp["formal"],sp["extension"])),"plan/source drift")
    out=root/"attempt";out.mkdir(mode=0o700)
    for name in ("weights","logits"): (out/name).mkdir()
    need(os.environ.get("SPIRALTORCH_BACKEND")=="cpu" and
         all(os.environ.get(k)=="1" for k in ("RAYON_NUM_THREADS","VECLIB_MAXIMUM_THREADS","OMP_NUM_THREADS")),"CPU controls")
    def offline(event,args):
        if event in ("socket.connect","socket.getaddrinfo") or (event=="import" and args[0].split(".")[0] in
            {"torch","numpy","safetensors","transformers","accelerate","huggingface_hub"}):
            raise RuntimeError("offline CPU native study only")
    sys.addaudithook(offline)
    import spiraltorch as st
    ext=importlib.import_module("spiraltorch.spiraltorch")
    need(st.NativeCausalLm is ext.NativeCausalLm and d.digest(Path(ext.__file__).read_bytes())==sp["extension"],
         "native extension mismatch")
    rows=sp["train"]+sp["test"]
    starts,hosts={},{}
    for b in sp["bases"]:
        case,lora,_=s.fixtures_cached()
        data=(root/("base-%d.safetensors"%b)).read_bytes()
        need(d.digest(data)==sp["weights"][str(b)],"wrong generated base")
        base=st.NativeCausalLm.from_gpt2(case["config"],st.SafeTensorArchive.from_bytes(data)).with_readout_accumulator("f32")
        active=base.with_lora_json(json.dumps(dict(lora,base_model_identity=base.base_model_identity)))
        frozen=active.freeze_lora()
        h=hosts[str(b)]=dict(base=frozen.base_model_identity,original_base=base.base_model_identity,
                            frozen_lora=active.lora_artifact_id,lora_weights=active.lora_to_json())
        reference=logits(frozen,rows)
        d.save(out/("frozen-%d.json"%b),reference)
        for turn in sp["turns"]:
            m=frozen.with_geometry(st.GeometryAdapterArtifact.create_json(json.dumps(record(h["base"],turn,[0.]*6))))
            name=s.key(b,turn)
            ident,vec,z=capture(m,turn,h,out,name,rows)
            need(all(bits(v)==bits(reference[key]) for key,v in z.items()) and vec==[0.]*6,"nonidentity origin")
            starts[name]=m,ident,vec,z
    d.save(out/"hosts.json",hosts)
    with (out/"steps.jsonl").open("xb") as log:
        for b,turn,teacher,order in cells(sp):
            start=s.key(b,turn);name=s.key(b,turn,teacher,order)
            m,origin,vec,z=starts[start];prev=origin;host=hosts[str(b)]
            for i,pair in enumerate(k.schedule(sp["train"],order,"pair")*sp["epochs"]):
                handles,grads=zip(*(old.handle(m,r,r[teacher],1.) for r in pair))
                if i==0:
                    for r,g in zip(pair,grads): k.check_grad(g,1.,r[teacher],z[r["key"]])
                m,report=m.step_geometry_cohort_gradient_at_distance(handles,target_l2=2*d.D)
                _,new,after=weights(m,turn,host)
                rec=dict(cell=name,i=i,keys=[r["key"] for r in pair],members=grads,values=after,id=new,report=report)
                log.write(d.encoded(rec));log.flush()
                k.check_step(rec,vec,prev,host)
                vec,prev=after,new
            end,_,_=capture(m,turn,host,out,name,rows)
            need(end==prev and starts[start][0].geometry_artifact_id==origin,"endpoint identity")
    d.save(root/"complete.json",dict(spec=pin,files=k.identities(out)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    p1=sub.add_parser("prepare");p1.add_argument("kind",choices=("rehearsal","formal"));p1.add_argument("root",type=Path);p1.add_argument("extension")
    p2=sub.add_parser("run");p2.add_argument("root",type=Path);p2.add_argument("pin")
    a=p.parse_args()
    if a.command=="prepare": prepare(a.root,a.kind=="formal",a.extension)
    else: run(a.root,a.pin)
