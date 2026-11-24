"""Forward-only interpolation of sealed artificial endpoints, plus model-free readback."""
import argparse
import importlib
import json
import math
import os
from pathlib import Path
import runpy
import struct
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
k=SimpleNamespace(**runpy.run_path(str(HERE/"design.py")))
s,d,need=k.s,k.d,k.need
PIN="ec081e8987bb56af70716de5227a9355f0326e419e0b06cb935567685bd917ff"
STEPS=(-16,-4,-1,1,4,16,64,128,192,256)


def pinned(path,pin):
    raw=path.read_bytes()
    need(d.digest(raw)==pin,"pin: "+path.name)
    return json.loads(raw)


def evidence(root,result):
    r=pinned(result,PIN)
    sp=pinned(root/"spec.json",r["spec"])
    done=pinned(root/"completed.json",r["completed"])
    seal=pinned(root/"endpoints.json",done["sealed"])
    need(sp==k.plan(True) and done["spec"]==r["spec"] and r["endpoints_digest"]==done["sealed"] and
         seal["files"]==k.seal_snapshot(sp,root/"attempt"),"not the sealed formal study")
    return sp,r,pinned(root/"held-origin-bank.json",done["held_bank"])


def typed_logits(m,rows):
    return {r["key"]:m.forward(r["prompt"],[1,len(r["prompt"])],last_only=True).values() for r in rows}


def same_logits(a,b):
    return set(a)==set(b) and all(struct.pack("<11f",*a[k])==struct.pack("<11f",*b[k]) for k in a)


def interpolate(origin,end,t):
    return [d.f32(a+(t/256)*(b-a)) for a,b in zip(origin,end)]


def bracket(pred,observed,sign):
    n=len(sign)
    need(n==len(observed)==len(pred) and n>0,"incompatible bracket")
    h=k.dot(pred,sign)/n
    actual=k.dot(observed,sign)/n
    rms=k.norm(pred)/math.sqrt(n)
    error=k.norm(k.sub(pred,observed))/math.sqrt(n)
    h_resolved=min(abs(actual),abs(h))>1e-6
    row_resolved=rms>2e-6
    return dict(predicted_H=h,observed_H=actual,rms_prediction=rms,rms_error=error,
                H_consistent=(actual*h>0 and abs(actual-h)<=max(1e-6,.15*abs(h))) if h_resolved else None,
                rows_consistent=error<=max(2e-6,.10*rms) if row_resolved else None)


def execute(root,report,out):
    sp,_,_=evidence(root,report)
    out.mkdir(mode=0o700)
    need(os.environ.get("SPIRALTORCH_BACKEND")=="cpu" and all(os.environ.get(x)=="1" for x in
         ("OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","RAYON_NUM_THREADS")),"CPU controls")
    sys.addaudithook(runpy.run_path(str(HERE.parent/"run.py"))["guard"])
    import spiraltorch as st
    ext=importlib.import_module("spiraltorch.spiraltorch")
    need(st.NativeCausalLm is ext.NativeCausalLm and d.digest(Path(ext.__file__).read_bytes())==k.EXT,"changed binary")
    original=root/"attempt"
    hosts=k.read(original/"hosts.json")
    calls=0
    with (out/"points.jsonl").open("xb") as tape:
        for b in sp["bases"]:
            models,host=k.native.origins(st,root,sp,b)
            need(host==hosts[str(b)],"changed base/LoRA")
            for geo,m in zip(s.GEOS,models):
                name=s.key(b,geo)
                raw,pin,origin=k.native.weights(m,geo,host)
                need(raw.encode()==(original/"weights"/(name+".json")).read_bytes(),"changed origin")
                z=typed_logits(m,sp["test"]); calls+=len(z)
                prev=k.read(original/"logits"/(name+".json"))["rows"]
                need(same_logits(z,{q:prev[q] for q in z}),"origin logits not bit-identical")
                tape.write(d.encoded(dict(arm=name,t=0,owner=pin,values=origin,z=z))); tape.flush()
                for pack in k.PACK:
                    cell=s.key(b,geo,"pure",pack,"head",17)
                    end_raw=(original/"weights"/(cell+".json")).read_bytes()
                    end=s.coordinates(end_raw,geo,host["base"])
                    for t in STEPS:
                        vec=interpolate(origin,end,t)
                        state=json.loads(raw)
                        for i,block in enumerate(state["placements"][0]["tensors"]):
                            block["values"]=vec[i*3:i*3+3]
                        model=m.with_geometry(st.GeometryAdapterArtifact.create_json(json.dumps(state)))
                        owner_raw,owner,actual=k.native.weights(model,geo,host)
                        need(actual==vec,"stored interpolation coordinate changed")
                        z=typed_logits(model,sp["test"]); calls+=len(z)
                        tape.write(d.encoded(dict(arm=cell,t=t,owner=owner,values=actual,z=z))); tape.flush()
                        if t==256:
                            original_z=k.read(original/"logits"/(cell+".json"))["rows"]
                            need(owner_raw.encode()==end_raw and same_logits(z,{q:original_z[q] for q in z}),
                                 "endpoint owner or logits not bit-identical")
    need(calls==2016,"unexpected forward count")
    d.save(out/"completed.json",dict(formal_readback=PIN,points=d.digest((out/"points.jsonl").read_bytes()),
                                     script=d.digest(Path(__file__).read_bytes()),calls=calls))


def analyze(root,report,out):
    sp,primary,bank=evidence(root,report)
    done=k.read(out/"completed.json")
    data=(out/"points.jsonl").read_bytes()
    need(done==dict(formal_readback=PIN,points=d.digest(data),script=d.digest(Path(__file__).read_bytes()),calls=2016),
         "changed diagnostic")
    points={}
    for line in data.splitlines():
        q=json.loads(line)
        group=points.setdefault(q["arm"],{})
        need(q["t"] not in group,"duplicate point")
        group[q["t"]]=q
    old=root/"attempt"
    counts=0
    cells={}
    rs=sp["test"]
    sign=[1-2*r["head"] for r in rs]
    def validate(q):
        need(set(q["z"])=={r["key"] for r in rs} and all(len(v)==11 and all(x==d.f32(x) and math.isfinite(x) for x in v)
             for v in q["z"].values()),"invalid point logits")
    def read_origin(n,b,g):
        q=points[n][0]
        raw=(old/"weights"/(n+".json")).read_bytes()
        base=k.read(old/"hosts.json")[str(b)]["base"]
        z=k.read(old/"logits"/(n+".json"))["rows"]
        need(q["owner"]==d.identity(raw) and q["values"]==s.coordinates(raw,g,base) and
             same_logits(q["z"],{r["key"]:z[r["key"]] for r in rs}),"endpoint/origin replay mismatch")
        return q
    def margins(q):
        return [q["z"][r["key"]][7]-q["z"][r["key"]][8] for r in rs]
    for b in sp["bases"]:
        for geo in s.GEOS:
            start=s.key(b,geo)
            need(set(points[start])=={0},"extra origin")
            origin=read_origin(start,b,geo)
            validate(origin); counts+=len(rs)
            v0=origin["values"]; m0=margins(origin)
            slopes=[k.sub(*[bank[start][r["key"]]["pure"][i]["v"] for i in (1,0)]) for r in rs]
            for pack in k.PACK:
                cell=s.key(b,geo,"pure",pack,"head",17)
                pts=points[cell]
                need(set(pts)==set(STEPS),"missing interpolation points")
                ref=pts[256]
                raw=(old/"weights"/(cell+".json")).read_bytes()
                base=k.read(old/"hosts.json")[str(b)]["base"]
                z=k.read(old/"logits"/(cell+".json"))["rows"]
                need(ref["owner"]==d.identity(raw) and ref["values"]==s.coordinates(raw,geo,base) and
                     same_logits(ref["z"],{r["key"]:z[r["key"]] for r in rs}),"changed endpoint")
                curve={}
                for t,q in pts.items():
                    validate(q); counts+=len(rs)
                    need(q["values"]==interpolate(v0,ref["values"],t),"wrong path")
                    dx=k.sub(q["values"],v0)
                    curve[str(t)]=dict(predicted=k.dot([k.dot(j,dx) for j in slopes],sign)/len(rs),
                                      observed=k.dot(k.sub(margins(q),m0),sign)/len(rs))
                reference=primary["rows"][s.key(b,geo,pack,17)]["test"]["pure"]
                need(abs(curve["256"]["predicted"]-reference["endpoint_linear"]["h"])<1e-12 and
                     abs(curve["256"]["observed"]-reference["h"])<1e-12,"different primary effect")
                central={}
                selected={}
                for t in (1,4,16):
                    dx=k.scaled(k.sub(pts[t]["values"],pts[-t]["values"]),.5)
                    delta=k.scaled(k.sub(margins(pts[t]),margins(pts[-t])),.5)
                    c=central[str(t)]=bracket([k.dot(j,dx) for j in slopes],delta,sign)
                    for criterion in ("H_consistent","rows_consistent"):
                        if criterion not in selected and c[criterion] is not None:
                            selected[criterion]=dict(t=t,consistent=c[criterion])
                cells[cell]=dict(curve=curve,central=central,selected=selected)
    need(counts==2016 and len(points)==18,"wrong diagnostic coverage")
    return dict(calls=counts,diagnostic=d.digest((out/"completed.json").read_bytes()),cells=cells)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("run","verify"))
    p.add_argument("root",type=Path); p.add_argument("formal_result",type=Path); p.add_argument("directory",type=Path)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    if args.command=="run":
        execute(args.root,args.formal_result,args.directory)
    else:
        need(args.output is not None,"new output path required")
        print(d.save(args.output,analyze(args.root,args.formal_result,args.directory)))
