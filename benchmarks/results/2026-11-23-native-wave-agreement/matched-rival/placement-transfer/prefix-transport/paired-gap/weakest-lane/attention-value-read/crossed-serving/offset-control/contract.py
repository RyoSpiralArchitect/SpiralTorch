"""Frozen cohort/score identities and lossless interop with st_nn::language::paired_offset."""
import json
import math
from pathlib import Path
import runpy
import struct
import subprocess
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
c = SimpleNamespace(**runpy.run_path(str(HERE.parent/"cross_design.py")))
d, need = c.d, c.need
CROSS = "fdeda563994b622f94a54338369d237bbd68a425131a5fb7d4143ec7c1bf64b0"
RESULT = {"formal":"26dc374dced142e1c69644347c1e43d0d010b4d69033d4f40805657198c32eb0",
          "rehearsal":"ca4cad8b11b605c9a40217146b0458c6ebef9de1bcdfe457c35ebd64fc5ccb48"}
PARENT = {"formal":("9bdd6e64513e4eb019ecb2006d64314a295944a387ad3ab7d32e2255a3d27693",
                    "b5699ec606cb9013f25222f44135882771f071873e2b82300f40be2451e7c43c",
                    "7bd95b75397df3a34824e3f637b4a4613aa5e1e36bb89383f6963df0f2dd4aec"),
          "rehearsal":("59e61bd9b27f7d869425fbbcb13e6fda44a3476dce62f9d0fa70fdcc11d50040",
                       "916c84d2f91939f84e48af32834c397e4982123a1ec01b580a47359143acea4a",
                       "5d99cc691310cad879805d459d47827ebeff74b5ca1de0bea0c4edaad2b1980b")}
REPO = c.x.REPO
RUST = "crates/st-nn/src/language/paired_offset.rs"
GOV = HERE.parents[6]/"parameter-vjp/bounded.py"
SOURCES = ("UA","origin")
METHODS = ("unshifted","task","mimic")
F32 = struct.Struct(">f")
F64 = struct.Struct(">d")


def sources():
    fs = [HERE/n for n in ("PROTOCOL.md","contract.py","measure.py","analyze.py","workflow.py","test_offset.py","offset_stream.rs")]
    fs += [REPO/RUST, REPO/"crates/st-nn/src/language/mod.rs", GOV]
    return {p.relative_to(REPO).as_posix():d.digest(p.read_bytes()) for p in fs}


def group(b,t,o,scale):
    return f"{b}-{t}-{o}-{scale}"


def training_context(old, cross, kind):
    folder = old/kind
    inp, sp = c.s.data(folder,PARENT[kind][0])
    trained = c.x.read(folder/"trained.json",PARENT[kind][1])
    checked = c.x.read(folder/"training-checked.json",PARENT[kind][2])
    need(trained["inputs"] == PARENT[kind][0] and checked["trained"]==PARENT[kind][1] and sp["kind"]==kind,"train lineage")
    original = folder/"training/scores.jsonl"
    need(d.digest(original.read_bytes()) == trained["files"]["scores.jsonl"], "original training logits")
    bank = c.a.bank(sp,sp["train"],original,checked["owners"])
    cf = c.x.read(cross/"frozen.json",CROSS)
    plan = c.x.read(cross/(kind+"-plan.json"),cf["plans"][kind])
    need(plan["parent"]==c.PINS[kind] and plan["training_manifest"]==inp["train"],"wrong serving treatments")
    return sp,checked["owners"],bank,{v["name"]:v for v in plan["cells"]}


def pairs_record(b,t,o,scale,rows,banks):
    out=[]
    for l,r in c.x.pairs(rows,t):
        need(l[t]==0 and r[t]==1 and l["template"]==r["template"],"pair polarity")
        keys=[l["key"],r["key"]]
        scores={m:[banks[m][k] for k in keys] for m in (*SOURCES,"UU")}
        for pair in scores.values():
            for z in pair: c.s.v.u.bits(z)
        out.append(dict(keys=keys,template=l["template"],scores=scores))
    return dict(base=b,teacher=t,order=o,scale=scale,group=group(b,t,o,scale),pairs=out)


def contrasts(pair):
    return [z[7]-z[8] for z in pair]


def words(pair):
    return [F32.pack(z[i]).hex() for z in pair for i in (7,8)]


def engine(program, queries):
    need(len({k for _,k,_ in queries})==len(queries),"duplicate Rust query")
    text="".join(" ".join((op,k,*data))+"\n" for op,k,data in queries)
    raw=subprocess.check_output([str(program)],input=text,text=True)
    lines=raw.splitlines()
    need(len(lines)==len(queries),"incomplete Rust response")
    result={}
    for line,(op,k,_) in zip(lines,queries):
        ident,task,*data=line.split()
        need((ident,task)==(k,op),"response ownership")
        result[k]=data
    return result


def wide(bits):
    need(len(bits)==16,"malformed f64")
    v,=F64.unpack(bytes.fromhex(bits))
    need(math.isfinite(v),"nonfinite offset or score")
    return v


def exhaustive(gaps):
    """Small independent oracle: enumerate strict hits at every boundary/midpoint."""
    need(bool(gaps),"empty oracle")
    intervals=[(-p,-n) for p,n in gaps if p>n]
    bounds=sorted({x for lr in intervals for x in lr})
    points=[(0.,0.)]+[(b,0.) for b in bounds]
    points += [(m,r-l) for l,r in zip(bounds,bounds[1:]) if l<(m:=l+(r-l)/2)<r]
    hits=lambda z:sum(l<z<r for l,r in intervals)
    hit,z=max(((hits(m),m) for m,_ in points),key=lambda v:v[0])
    if hits(0.)==hit:
        z=0.
    else:
        z=max(((m,w) for m,w in points if hits(m)==hit),key=lambda v:(v[1],-abs(v[0]),-v[0]))[0]
    return len(intervals),hit,z


def fit(groups,program):
    q=[]
    for g in groups:
        for src in SOURCES:
            args=[w for p in g["pairs"] for mode in (src,"UU") for w in words(p["scores"][mode])]
            q.append(("F",g["group"]+"."+src,args))
    ans=engine(program,q)
    out={}
    for g in groups:
        for src in SOURCES:
            key=g["group"]+"."+src
            a,m,n,possible,hits=ans[key]
            train=[contrasts(p["scores"][src]) for p in g["pairs"]]
            target=[contrasts(p["scores"]["UU"]) for p in g["pairs"]]
            sep,max_hits,chosen=exhaustive(train)
            mean=math.fsum(term for y,z in zip(train,target) for pair in zip(y,z) for term in (pair[1],-pair[0]))/(2*len(train))
            need((int(n),int(possible),int(hits),wide(a),wide(m))==(len(train),sep,max_hits,chosen,mean),"fit vs independent oracle: "+key)
            out[key]=dict(task=wide(a),mimic=wide(m),pairs=int(n),separable=sep,covered=max_hits)
    for b in {g["base"] for g in groups}:
        for t in c.x.TEACHERS:
            vals={out[g["group"]+".origin"]["task"] for g in groups if g["base"]==b and g["teacher"]==t}
            need(len(vals)==1,"origin task offset must not change across orders/scales")
    return out
