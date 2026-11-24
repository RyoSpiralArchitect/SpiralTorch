"""Frozen symbolic agreement task and byte identities; no ML runtime."""
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parents[4]/"bindings/st-wasm/tests"
DATA = {"native_causal_lm_cache_fixture.json": "1c76e3a53b92ec9d7496dbd84f2d348abe550656b5afd61822dc27d53d1e21e7",
        "native_line_owners_bridge.json": "37ce558df19569717486813b38becc3cdd912b80e747d83474e25985d0860ea9"}
EXTENSION = "171bfceac37ac4e8c26d8b6da6ecbc0da8733124c73df059f6e440ab8eaf7c18"
BASE = "sha256:9f9c4fa145a8df64e56d864d65484617dacdc653ae8f0383d9271e0207146df5"
FROZEN = "sha256:eff2df74fa081fcbff84d048a8594e96821a31ec5bdc3a80969c5573b7c5bce4"
LORA = "sha256:2d5e02f430376e9476bc4bddbaaaec7db5ac72b1d568f373132e867136976337"
SITES = ["transformer.h.%d%s"%(n,s) for n in range(2) for s in (".attn",".mlp","")]
RECIPES = ("wave","affine")
TEACHERS = ("head","attractor")
REF, D, FLOOR = "lora-only", 1/256, .001
CANDIDATES = [[7],[8]]


def need(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(x):
    return (json.dumps(x, sort_keys=True, separators=(",",":"), allow_nan=False)+"\n").encode("ascii")


def save(path, x):
    raw = encoded(x)
    with path.open("xb") as f:
        f.write(raw)
    return digest(raw)


def digest(raw):
    return sha256(raw).hexdigest()


def identity(raw):
    return "sha256:"+digest(raw)


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def distance(a,b):
    need(len(a)==len(b) and bool(a), "mismatched parameter blocks")
    return math.sqrt(math.fsum((x-y)**2 for x,y in zip(a,b)))


def row(h,a,i,j):
    return dict(key="%d%d.%d%d"%(h,a,i,j), frame="%d%d"%(h,a),
                prompt=[h,i,9,a,j,10], head=i-1, attractor=j-1)


def schedule(rows, seed):
    return sorted(rows, key=lambda r:(digest(("%d:%s"%(seed,r["key"])).encode("ascii")),r["key"]))


def design(formal):
    pairs = list(product(range(3,7), repeat=2))
    def rows(frames):
        return [row(h,a,i,j) for h,a in frames for i,j in product((1,2),repeat=2)]
    train = rows([(h,a) for h,a in pairs if (h-a)%4 in (1,3)] if formal else [(3,3)])
    test = rows([(h,a) for h,a in pairs if (h-a)%4 == 2] if formal else [(4,4)])
    arms = [r+"@"+s for r in RECIPES for s in SITES]+[REF] if formal else ["wave@"+SITES[0],"affine@"+SITES[-1],REF]
    return dict(schema="spiraltorch.role_placement.v1", formal=formal, train=train, test=test,
                arms=arms, orders=[17,31,73] if formal else [5], epochs=2 if formal else 1,
                distance=D, floor=FLOOR, teachers=list(TEACHERS), extension=EXTENSION, inputs=DATA)


def sources():
    files = [HERE/n for n in ("PROTOCOL.md","design.py","run.py","analyze.py")]
    files.append(HERE.parent/"parameter-vjp/bounded.py")
    return {p.name:digest(p.read_bytes()) for p in files}


def cells(spec):
    return [(arm,role,seed) for arm in spec["arms"] for role in TEACHERS for seed in spec["orders"]]


def cell_key(arm,role,seed):
    return "%s-%s-%d"%(arm,role,seed)


def session_plan(spec,arm,role,seed):
    rows = schedule(spec["train"],seed)
    mode = "lora" if arm == REF else "geometry"
    plan = dict(schema="spiraltorch.native_causal_lm_line_plan.v3",mode=mode,epochs=spec["epochs"],
                max_tile_values=11, update=dict(kind="parameter_gradient_l2",target_l2=D),
                examples=[dict(prompt_ids=r["prompt"],candidates=CANDIDATES,stop_id=0,
                               objective=dict(kind="menu",mix=.5,gold_index=r[role])) for r in rows])
    return encoded(plan), rows


def coordinates(raw,arm):
    record=json.loads(raw)
    if arm == REF:
        need(record["base_model_identity"]==BASE and record["layer"]==0 and record["rank"]==2 and
             record["width"]==3 and len(record["a"])==len(record["b"])==6, "wrong LoRA layout")
        data=record["a"]+record["b"]
    else:
        p,=record["placements"]
        recipe,site=arm.split("@")
        expected="wave_gate" if recipe=="wave" else "channel_affine"
        need(record["base_model_identity"]==FROZEN and p["path"]==site and p["training"] and
             p["recipe"]["schema"]=="spiraltorch."+expected+"_adapter.v1" and p["recipe"]["strength"]==.5
             and p["recipe"]["features"]==3, "wrong geometry recipe or site")
        blocks=p["tensors"]
        need([b["name"] for b in blocks]==["bias","gate"] and
             all(b["requires_grad"] and b["role"]=="parameter" and b["shape"]==[3] and
                 len(b["values"])==3 for b in blocks), "wrong geometry owner layout")
        data=[x for b in blocks for x in b["values"]]
    need(all(map(math.isfinite,data)),"nonfinite coordinate")
    return list(map(f32,data))


def owner_id(raw,arm):
    return identity((b"spiraltorch.native_lora.v1\0" if arm==REF else b"")+raw)


def read_plan(root):
    raw=(root/"spec.json").read_bytes()
    spec=json.loads(raw)
    need(encoded(dict(design(spec["formal"]),sources=sources()))==raw,"plan/source drift")
    return spec
