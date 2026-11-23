"""Fixed design and arithmetic checks. Standard library only, no model."""
from itertools import product
import json
import math
from pathlib import Path
import runpy
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PREV = HERE.parent/"answer-scope"
native = SimpleNamespace(**runpy.run_path(str(PREV/"run.py")))
s, d, need = native.s, native.d, native.need
EXT = "67fe523ba0b15205312231bd13d446992c46c6cf2a586241c55211e213007d02"
MIX = {"mixed": .5, "pure": 1.}
PACK = {"single": 1, "pair": 2}
TOK = [7, 8]


def plan(formal):
    bases = [503,509,521] if formal else [541]
    task = d.design(formal)
    repo = HERE.parents[5]
    files = list(HERE/n for n in ("PROTOCOL.md","design.py","run.py","probe.py","analyze.py"))
    files += [PREV/"run.py", PREV/"study.py", HERE.parent/"design.py", HERE.parent/"analyze.py",
              HERE.parent.parent/"parameter-vjp/bounded.py",
              repo/"crates/st-nn/src/causal_lm/geometry_cohort.rs",repo/"crates/st-nn/src/causal_lm/token_menu.rs"]
    return dict(schema="spiraltorch.conditional_update.v1",formal=formal,bases=bases,extension=EXT,
                train=task["train"], test=task["test"], epochs=task["epochs"],orders=task["orders"],
                weights={str(b):d.digest(s.generate(s.fixtures_cached()[2],b)) for b in bases},
                sources={f.relative_to(repo).as_posix():d.digest(f.read_bytes()) for f in files})


def specs(sp):
    return product(sp["bases"], s.GEOS, MIX, PACK, d.TEACHERS, sp["orders"])


def schedule(rows, order, pack):
    need(pack in PACK,"unknown update schedule")
    def sort(seq, tag):
        return sorted(seq,key=lambda r:(d.digest(("%d:%s:%s"%(order,tag,r["key"])).encode("ascii")),r["key"]))
    by_frame = {(r["frame"],r["head"],r["attractor"]):r for r in rows}
    first = sort([r for r in rows if r["head"]==0],"pair")
    pairs = [sort([r,by_frame[r["frame"],1,1-r["attractor"]]],"row") for r in first]
    need(len({r["key"] for p in pairs for r in p})==len(rows)==2*len(pairs),"unbalanced pairs")
    return pairs if pack=="pair" else [[r] for p in pairs for r in p]


def norm(v):
    return math.sqrt(math.fsum(x*x for x in v))


def dot(a,b):
    need(len(a)==len(b),"dot dimensions")
    return math.fsum(x*y for x,y in zip(a,b))


def sub(a,b):
    need(len(a)==len(b),"vector dimensions")
    return [x-y for x,y in zip(a,b)]


def avg(vs):
    need(bool(vs) and all(len(v)==len(vs[0]) for v in vs),"mean dimensions")
    return [math.fsum(col)/len(vs) for col in zip(*vs)]


def scaled(v,k):
    return [k*x for x in v]


def unit(v):
    n=norm(v)
    return scaled(v,1/n) if n else None


def cosine(a,b):
    u,v=unit(a),unit(b)
    return max(-1.,min(1.,dot(u,v))) if u is not None and v is not None else None


def nll(v,i):
    peak=max(v)
    return (peak-v[i])+math.log(math.fsum(math.exp(x-peak) for x in v))


def costs(z,gold,mix):
    pair=[z[i] for i in TOK]
    full,cond=nll(z,TOK[gold]),nll(pair,gold)
    return full,cond,(1-mix)*full+mix*cond


def check_grad(g, mix, gold, z=None):
    v=g["v"]
    need(len(v)==6 and all(math.isfinite(x) and d.f32(x)==x for x in v) and
         g["mix"]==mix and g["gold"]==gold and math.isfinite(g["loss"]) and
         min(g["ce"],g["binary"])>=0 and abs(g["loss"]-((1-mix)*g["ce"]+mix*g["binary"]))<3e-6,
         "invalid single-target mixture or gradient")
    if z is not None:
        need(all(abs(x-y)<3e-6 for x,y in zip(costs(z,gold,mix),[g["ce"],g["binary"],g["loss"]])),
             "token loss vs full logits")


def check_bank(row, z, train):
    for scope in (MIX if train else ("pure",)):
        for i,g in enumerate(row[scope]):
            check_grad(g,MIX[scope],i,z)
        need(len(row[scope])==2,"opposite labels")
    a,b=[g["v"] for g in row["pure"]]
    slope=sub(b,a)
    # Score the two softmax probabilities independently, including the smaller one.
    p,q=[math.exp(-nll([z[7],z[8]],i)) for i in (0,1)]
    ref=sub(row["mixed"][1]["v"],row["mixed"][0]["v"]) if train else slope
    errors=[norm(sub(slope,ref)),norm(sub(a,scaled(ref,-q))),norm(sub(b,scaled(ref,p)))]
    lens=[norm(g["v"]) for vals in row.values() for g in vals]
    worst=max(errors)
    need(worst<=2e-6*max([1.]+lens),"label-difference or binary-gradient identity")
    return slope,dict(abs_error=worst,rel_error=worst/norm(ref) if norm(ref) else None,
                      pure_cos=cosine(a,b),mixed_cos=cosine(*[g["v"] for g in row["mixed"]]) if train else None)


def check_step(rec, before, old, host):
    report=rec["report"]
    ray=report["step"]
    grads=[g["v"] for g in rec["members"]]
    n=len(grads)
    v=avg(grads)
    dx=sub(rec["values"],before)
    length=norm(dx)
    target=n*d.D
    need(n in (1,2) and len(dx)==6 and all(math.isfinite(x) and d.f32(x)==x for x in rec["values"]) and
         report["members"]==report["total_target_tokens"]==n and report["readout_accumulator"]=="f32" and
         report.get("lora_artifact_id") is None and report["base_model_identity"]==host["base"] and
         ray["method"]=="cohort_mean_gradient_ray_f64.v1" and ray["trainable_values"]==6 and
         ray["origin_id"]==old and ray["result_id"]==rec["id"]!=old and ray["requested_l2"]==target,
         "ray ownership, target or method")
    nv=norm(v)
    need(nv>0 and dot(v,dx)<0,"not average descent")
    error=norm([x/target+y/nv for x,y in zip(dx,v)])
    need(error<=1e-4 and abs(length/target-1)<=1e-4 and ray["relative_vector_error"]<=1e-4,"wrong parameter motion")
    native_vals=[report["mean_input_loss"],ray["gradient_l2"],ray["realized_l2"],ray["gradient_dot_step"]]+report["member_loss_dots"]
    measured=[math.fsum(g["loss"] for g in rec["members"])/n,nv,length,dot(v,dx)]+[dot(g,dx) for g in grads]
    need(len(native_vals)==len(measured) and all(math.isclose(x,y,rel_tol=3e-7,abs_tol=1e-12)
         for x,y in zip(native_vals,measured)),"native vs independent gradient projection")
    return length,error


def read(path):
    return json.loads(path.read_bytes())


def identities(folder):
    return {p.relative_to(folder).as_posix():d.digest(p.read_bytes()) for p in sorted(folder.rglob("*")) if p.is_file()}


def seal_snapshot(sp,folder):
    names=[s.key(b,g) for b in sp["bases"] for g in s.GEOS]+[s.key(*args) for args in specs(sp)]
    expected={k+"/"+name+".json" for name in names for k in ("weights","logits")} | {"hosts.json","train-bank.json","steps.jsonl"}
    actual=identities(folder)
    need(set(actual)==expected,"missing or extra trajectory artifact")
    return actual
