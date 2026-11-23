"""Compare two interventions on every trained endpoint; verify mode is model-free."""
import argparse
import csv
import importlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
v=SimpleNamespace(**runpy.run_path(str(HERE/"verify.py")))
u,k,d,s,need=v.u,v.k,v.d,v.s,v.need
PIN="74437a17603c8f19ee3fae3e1802adfaa4f459126c68a8edf3a731b378ea7515"


def components(O,G,B,F):
    gate=((G-O)+(F-B))/2
    bias=((B-O)+(F-G))/2
    need(math.isclose(gate+bias,F-O,abs_tol=1e-12,rel_tol=1e-12),"component closure")
    return gate,bias,(F-B)-(G-O)


def source(root):
    need(d.digest(d.encoded(v.verify(root)))==PIN,"wrong formal evidence")
    return k.read(root/"spec.json"), k.read(root/"attempt/hosts.json")


def specification():
    return dict(schema="spiraltorch.prefix_transport_components.v1",parent=PIN,calls=2736,
                code=d.digest(Path(__file__).read_bytes()),protocol=d.digest((HERE/"COMPONENTS.md").read_bytes()))


def capture(root,out):
    out.mkdir(mode=0o700)
    sp,hosts=source(root);rr=sp["test"];old=root/"attempt"
    plan=d.save(out/"plan.json",specification())
    need(os.environ.get("SPIRALTORCH_BACKEND")=="cpu" and all(os.environ.get(k)=="1" for k in
        ("RAYON_NUM_THREADS","VECLIB_MAXIMUM_THREADS","OMP_NUM_THREADS")),"CPU only")
    def guard(event,args):
        if event in ("socket.connect","socket.getaddrinfo") or (event=="import" and args[0].split(".")[0] in
                {"torch","numpy","safetensors","transformers","accelerate","huggingface_hub"}):
            raise RuntimeError("offline native only")
    sys.addaudithook(guard)
    import spiraltorch as st
    native=importlib.import_module("spiraltorch.spiraltorch")
    need(st.NativeCausalLm is native.NativeCausalLm and d.digest(Path(native.__file__).read_bytes())==sp["extension"],"build pin")
    with (out/"journal.jsonl").open("xb") as log:
        for b in sp["bases"]:
            h=hosts[str(b)]
            data=(root/("base-%d.safetensors"%b)).read_bytes()
            base=st.NativeCausalLm.from_gpt2(s.fixtures_cached()[0]["config"],st.SafeTensorArchive.from_bytes(data)).with_readout_accumulator("f32")
            need(base.base_model_identity==h["original_base"],"base changed")
            frozen=base.with_lora_json(h["lora_weights"]).freeze_lora()
            for turn in u.TURNS:
                names=[s.key(b,turn)]+[s.key(b,turn,role,o) for role in d.TEACHERS for o in sp["orders"]]
                for index,name in enumerate(names):
                    raw=(old/"weights"/(name+".json")).read_bytes()
                    want=k.read(old/"logits"/(name+".json"))
                    full=st.GeometryAdapterArtifact.from_json(raw.decode(),want["owner"],h["base"])
                    vec=u.coordinates(raw,turn,h)
                    for kind in (("O",) if index==0 else ("F","G","B")):
                        values={"O":vec,"F":vec,"G":[0.]*3+vec[3:],"B":vec[:3]+[0.]*3}[kind]
                        art=(full if kind in ("O","F") else st.GeometryAdapterArtifact.create_json(
                            json.dumps(u.record(h["base"],turn,values))))
                        m=frozen.with_geometry(art)
                        payload,pin,actual=u.weights(m,turn,h)
                        need(values==actual,"wrong intervention")
                        z=u.logits(m,rr)
                        log.write(d.encoded(dict(name=name,kind=kind,owner=pin,weights=payload.decode(),z=z)));log.flush()
                        if kind in ("O","F"):
                            need(all(u.bits(vals)==u.bits(want["rows"][kk]) for kk,vals in z.items()),"origin/full did not replay")
    d.save(out/"complete.json",dict(plan=plan,journal=d.digest((out/"journal.jsonl").read_bytes())))


def readback(root,out):
    sp,hosts=source(root);rs=sp["test"];old=root/"attempt";data=(out/"journal.jsonl").read_bytes()
    plan=(out/"plan.json").read_bytes()
    need(plan==d.encoded(specification()) and k.read(out/"complete.json")==dict(plan=d.digest(plan),journal=d.digest(data)),"diagnostic seal")
    group={};calls=0
    for line in data.splitlines():
        j=json.loads(line); name,kind=j["name"],j["kind"]; host=hosts[name.split("-")[0]];turn=name.split("-")[1]
        need(kind not in group.setdefault(name,{}) and j["owner"]==d.identity(j["weights"].encode()),"duplicate or changed artifact")
        orig=(old/"weights"/(name+".json")).read_bytes()
        vec=u.coordinates(orig,turn,host)
        expected={"O":vec,"F":vec,"G":[0.]*3+vec[3:],"B":vec[:3]+[0.]*3}[kind]
        need(u.coordinates(j["weights"].encode(),turn,host)==expected and set(j["z"])=={r["key"] for r in rs},"bad intervention")
        for vv in j["z"].values(): u.bits(vv)
        if kind in ("O","F"):
            z=k.read(old/"logits"/(name+".json"))["rows"]
            need(j["weights"].encode()==orig and all(u.bits(vv)==u.bits(z[key]) for key,vv in j["z"].items()),"reference changed")
        group[name][kind]=[j["z"][r["key"]][7]-j["z"][r["key"]][8] for r in rs]
        calls+=len(rs)
    expect={s.key(b,t) for b in sp["bases"] for t in u.TURNS}
    cells=list(u.cells(sp))
    need(set(group)==expect|{s.key(*c) for c in cells} and all(set(group[k])=={"O"} for k in expect) and calls==2736,"coverage")
    rows=[]
    for b,t,role,order in cells:
        name=s.key(b,t,role,order);vals=group[name]
        need(set(vals)=={"G","B","F"},"missing endpoint arm")
        O=group[s.key(b,t)]["O"]
        terms=list(zip(*(components(*a) for a in zip(O,vals["G"],vals["B"],vals["F"]))))
        row=dict(base=b,rotation=t,teacher=role,order=order)
        def sgn(x): return (x>0)-(x<0)
        for key,arm in vals.items():
            row[key+"_switches"]=sum(sgn(y)!=sgn(x) for y,x in zip(arm,O))
        for key,y in zip(("gate","bias","interaction","full"),(*terms,k.sub(vals["F"],O))):
            row[key+"_rms"]=math.sqrt(math.fsum(x*x for x in y)/len(rs))
            row[key+"_teacher_delta"]=math.fsum((1-2*r[role])*x for x,r in zip(y,rs))/len(rs)
        rows.append(row)
    return dict(calls=calls,manifest=d.digest((out/"complete.json").read_bytes()),rows=rows)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("measure","verify"));p.add_argument("root",type=Path);p.add_argument("directory",type=Path)
    p.add_argument("--json",type=Path);p.add_argument("--table",type=Path)
    q=p.parse_args()
    if q.action=="measure": capture(q.root,q.directory)
    else:
        need(q.json is not None and q.table is not None,"two new output paths required")
        r=readback(q.root,q.directory);print(d.save(q.json,r))
        with q.table.open("x",encoding="ascii",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(r["rows"][0]),delimiter="\t",lineterminator="\n")
            w.writeheader();w.writerows(r["rows"])
