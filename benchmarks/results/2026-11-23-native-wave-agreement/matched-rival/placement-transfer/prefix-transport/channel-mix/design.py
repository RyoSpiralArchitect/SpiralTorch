"""Frozen synthetic compound templates, owner layouts and immutable input identities."""
import argparse
from fractions import Fraction
from itertools import product
import json
import math
import struct
from pathlib import Path
import runpy
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
v=SimpleNamespace(**runpy.run_path(str(HERE.parent/"residual-metric/study.py")))
exact=SimpleNamespace(**runpy.run_path(str(HERE.parent/"residual-metric/verify_exact.py")))
p,k,s,d,u,need=v.p,v.k,v.s,v.d,v.u,v.need
TARGET=v.TARGET
TURN="quarter_turn"
MAPS=("identity","cayley_triples","inverse_cayley_triples")
MATRIX=((15,-6,10),(10,15,-6),(-6,10,15))
PREVIOUS={
  "formal":("9e97a9bb137f606bc1a151d4f1868d9a95a3ee5db18b4391233b8c11bcf82976","3268ed35efa6744bda3b7a9c9cd53af8d1b0d3286a3427e56ee640a88ee0a27f"),
  "rehearsal":("28393430ca79b2ca55e6bdc4d191198606622d546ea986d82b73dc080f86c434","5884fd1be33935090c3dcd43f8f4ef1a72982d4d2c4751ea1ca0b7c3cdb95931")}
pinned=v.sig.pinned


def fresh(kind):
    out=[]
    for h in (range(3,7) if kind=="formal" else [3]):
        a,c=(3+(h-3+2)%4,3+(h-3+1)%4) if kind=="formal" else (3,6)
        for i,j in product((1,2),repeat=2):
            for name,ids in (("hc",[h,c,i,9,a,j,10]),("ac",[h,i,9,a,c,j,10])):
                frame="%d%d%d"%(h,c,a)
                out.append(dict(key="%s.%s.%d%d"%(name,frame,i,j),frame=frame,template=name,
                                prompt=ids,head=i-1,attractor=j-1))
    return out


def prepare(kind,root,previous):
    prev=json.loads(pinned(previous/kind/"spec.json",PREVIOUS[kind][0]))
    pinned(previous/kind/"complete.json",PREVIOUS[kind][1])
    seen=set()
    for s0 in PREVIOUS:
        spec=json.loads(pinned(previous/s0/"spec.json",PREVIOUS[s0][0]))
        seen.update(tuple(r["prompt"]) for r in spec["train"]+spec["test"])
    sets=[{tuple(r["prompt"]) for r in fresh(t)} for t in ("formal","rehearsal")]
    need(len(sets[0])==32 and len(sets[1])==8 and not(sets[0]&sets[1]) and not(seen&(sets[0]|sets[1])),"prompt overlap")
    train=dict(kind=kind,maps=list(MAPS),turn=TURN,rule="diagonal",target_rms=TARGET,
               **{key:prev[key] for key in ("train","bases","orders","epochs","weights","train_signals","parent_spec","parent_manifest")})
    dest=root/kind
    dest.mkdir(mode=0o700)
    a=d.save(dest/"training.json",train)
    b=d.save(dest/"evaluation.json",fresh(kind))
    return d.save(dest/"inputs.json",dict(kind=kind,training=a,evaluation=b,previous=PREVIOUS[kind],
                                         protocol=d.digest((HERE/"PROTOCOL.md").read_bytes())))


def data(root,pin):
    obj=json.loads(pinned(root/"inputs.json",pin))
    need(obj["protocol"]==d.digest((HERE/"PROTOCOL.md").read_bytes()) and tuple(obj["previous"])==PREVIOUS[obj["kind"]],"input provenance")
    return obj,json.loads(pinned(root/"training.json",obj["training"]))


def sources():
    repo=HERE.parents[6]
    files=[HERE/n for n in ("PROTOCOL.md","design.py","worker.py","verify.py")]
    files += [repo/n for n in ("crates/st-nn/src/geometry_artifact.rs","crates/st-nn/src/geometry_runtime.rs",
              "crates/st-nn/src/geometry_runtime/prefix_transport.rs","crates/st-nn/src/causal_lm/prefix_signal.rs",
              "crates/st-nn/src/causal_lm/geometry_cohort/prefix_metric.rs","bindings/st-py/src/causal_lm.rs")]
    return {f.relative_to(repo).as_posix():d.digest(f.read_bytes()) for f in files}


def record(host,mode,vec):
    obj=p.artifact(host["base"],TURN,"gate",vec)
    if mode!="identity":
        rec=obj["placements"][0]["recipe"]
        rec.update(schema="spiraltorch.prefix_transport_adapter.v2",output_mix=mode)
    return obj


def state(raw,host,mode):
    rec=json.loads(raw)
    for t in rec["placements"][0]["tensors"]:
        t["values"]=list(map(d.f32,t["values"]))
    vec=[x for t in rec["placements"][0]["tensors"] for x in t["values"]]
    need(mode in MAPS and rec==record(host,mode,vec),"wrong recipe, owner mask or parameter layout")
    return d.identity(raw),vec


def historical(previous,kind):
    manifest=json.loads(pinned(previous/"complete.json",PREVIOUS[kind][1]))["files"]
    rows=pinned(previous/"attempt/steps.jsonl",manifest["steps.jsonl"])
    past={}
    for r in map(json.loads,rows.splitlines()):
        parts=r["cell"].split("-")
        if parts[1:3]==[TURN,"diagonal"]:
            key=(int(parts[0]),parts[3],int(parts[4]),r["i"])
            need(key not in past,"duplicate historical update");past[key]=r
    return manifest,past


def raw_signals(signal,sp):
    rows={}
    for r in map(json.loads,pinned(signal/"attempt/rows.jsonl",sp["train_signals"]).splitlines()):
        if r["turn"]==TURN:
            key=(r["base"],r["key"])
            need(key not in rows,"duplicate training signal");rows[key]=r
    need(set(rows)==set(product(sp["bases"],[r["key"] for r in sp["train"]])),"training signal coverage")
    return rows


def mix3(vs, mode):
    if mode=="identity": return list(vs)
    return [math.fsum((MATRIX[i][j] if mode==MAPS[1] else MATRIX[j][i])*vs[j] for j in range(3))/19. for i in range(3)]


def check_origin(member,raw,mode):
    baseline=raw["ce"][member["gold"]]
    need(all(member[k]==baseline[k] for k in ("loss","ce","binary","gold","mix")),"zero gate changed objective")
    adj=mix3(baseline["v"][:3],{"identity":"identity",MAPS[1]:MAPS[2],MAPS[2]:MAPS[1]}[mode])
    trace=raw["trace"]
    mean=exact.means(tuple(trace["site_input"]),trace["shape"][0],TURN)
    ideal=adj+[m*g for m,g in zip(mean,adj)]
    need(all(math.isclose(g,e,rel_tol=2e-5,abs_tol=3e-8) for g,e in zip(member["v"],ideal)),"wrong routed origin VJP")


def outputs(examples,gate,mode):
    out=[]
    for tr in examples:
        mean=exact.means(tuple(tr["site_input"]),tr["shape"][0],TURN)
        vals=[d.f32(m*g) for m,g in zip(mean,gate)]
        if mode!="identity":
            coeff=MATRIX if mode==MAPS[1] else list(zip(*MATRIX))
            vals=[exact.nearest32(sum(k*Fraction(x) for k,x in zip(row,vals))/19) for row in coeff]
        out.append([d.f32(x+d.f32(tr["strength"]*v)) for x,v in zip(tr["site_input"][-3:],vals)])
    return out


def same_rows(a,b):
    return len(a)==len(b) and all(struct.pack("<3f",*x)==struct.pack("<3f",*y) for x,y in zip(a,b))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("kind",choices=list(PREVIOUS))
    for name in ("root","previous"): cli.add_argument(name,type=Path)
    print(prepare(**vars(cli.parse_args())))
