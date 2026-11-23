"""Base generation, native parameter identities and scope readback; no ML imports."""
import base64
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
import runpy
import struct
import types

HERE = Path(__file__).resolve().parent
P = types.SimpleNamespace(**runpy.run_path(str(HERE.parent/"analyze.py")))
d = P.d
need, save, encoded, digest = d.need, d.save, d.encoded, d.digest
SITE = "transformer.h.1.mlp"
SCOPES = ("answer","through_stop")
GEOS = ("wave","affine")
TOKENS = [7,8]


def fixtures():
    def read(name):
        raw = (d.FIXTURES/name).read_bytes()
        need(digest(raw) == d.DATA[name], "source fixture drift")
        return json.loads(raw)
    fixture = read("native_causal_lm_cache_fixture.json")
    case, = [c for c in fixture["cases"] if c["name"] == "ordinary/wave"]
    return case, json.loads(read("native_line_owners_bridge.json")["lora"]), base64.b64decode(fixture["weights"],validate=True)


def generate(template, seed):
    size, = struct.unpack_from("<Q",template)
    data = bytearray(template)
    header = json.loads(template[8:8+size])
    spans = []
    for name,t in header.items():
        a,b = t["data_offsets"]
        shape = t["shape"]
        need(t["dtype"] == "F32" and b-a == 4*math.prod(shape), "fixture layout")
        spans.append((a,b))
        if name in ("wte.weight","wpe.weight"):
            offset,scale = 0.,.25
        elif name.endswith(".bias"):
            offset,scale = 0.,.04
        elif name.endswith(".weight") and len(shape) == 1 and "ln_" in name:
            offset,scale = 1.,.1
        else:
            need(name.endswith(".weight") and len(shape) == 2,"unknown projection")
            offset,scale = 0.,.6/math.sqrt(shape[0])
        for i in range((b-a)//4):
            word = sha256(("scope-v1:%d:%s:%d"%(seed,name,i)).encode("ascii")).digest()[:4]
            u = (int.from_bytes(word,"big")+.5)/2**31 - 1
            struct.pack_into("<f",data,8+size+a+4*i,offset+scale*u)
    spans.sort()
    need(spans[0][0] == 0 and spans[-1][1] == len(data)-8-size and
         all(a[1] == b[0] for a,b in zip(spans,spans[1:])), "unreplaced parameter bytes")
    return bytes(data)


def spec(formal, extension):
    need(len(extension)==64 and all(x in "0123456789abcdef" for x in extension), "extension pin")
    parent = d.design(formal)
    bases = [101,211,307] if formal else [409]
    cd = HERE.parents[5]
    files = [HERE/n for n in ("PROTOCOL.md","study.py","run.py","verify.py")]
    files += [HERE.parent/n for n in ("design.py","analyze.py")]
    files += [HERE.parent.parent/"parameter-vjp/bounded.py",cd/"crates/st-nn/src/causal_lm/token_menu.rs",
              cd/"bindings/st-py/src/causal_lm.rs",cd/"bindings/st-wasm/src/causal_lm_lines.rs"]
    src = {p.relative_to(cd).as_posix():digest(p.read_bytes()) for p in files}
    template = fixtures()[2]
    return dict(schema="spiraltorch.answer_scope_study.v1",formal=formal,bases=bases,
                weights={str(k):digest(generate(template,k)) for k in bases},train=parent["train"],test=parent["test"],
                orders=parent["orders"],epochs=parent["epochs"],extension=extension,sources=src)


def cells(sp):
    return product(sp["bases"], GEOS, SCOPES, d.TEACHERS, sp["orders"])


def key(*args):
    return "-".join(map(str,args))


def loss(costs, gold, scope):
    if scope == "answer":
        values = [r[0] for r in costs["token_nlls"]]
        full = values[gold]
    else:
        need(scope == "through_stop","unknown scope")
        values = costs["line_nlls"]
        full = values[gold]/2
    low = min(values)
    conditional = (values[gold]-low)+math.log(math.fsum(math.exp(low-x) for x in values))
    return .5*(full+conditional)


def coordinates(raw,geo,base):
    state = json.loads(raw)
    p, = state["placements"]
    rec = p["recipe"]
    need(state["base_model_identity"] == base and p["path"] == SITE and p["training"] and rec["strength"] == .5
         and rec["schema"] == "spiraltorch."+{"wave":"wave_gate","affine":"channel_affine"}[geo]+"_adapter.v1"
         and rec["features"] == 3 and (rec["kernel"] if geo=="wave" else rec)["max_values"] == 64, "geometry placement")
    init = json.loads(fixtures_cached()[0]["geometry"])["placements"][0]["recipe"]
    if geo == "wave":
        need(rec == init,"fixed Wave kernel changed")
    blocks = p["tensors"]
    need([t["name"] for t in blocks]==["bias","gate"] and
         all(t["shape"]==[3] and len(t["values"])==3 and t["requires_grad"] and t["role"]=="parameter" for t in blocks),
         "geometry parameter layout")
    return [d.f32(v) for t in blocks for v in t["values"]]


fixtures_cached = lru_cache(maxsize=1)(fixtures)


def verify_step(r, prev, expected_pin, target_pin):
    ray = r["ray"]
    group = ray["geometry"]
    pos,grad = r["values"],r["grad"]
    need(len(pos)==len(grad)==len(prev)==6 and all(math.isfinite(z) and z==d.f32(z) for z in pos+grad),
         "six f32 parameters and gradients")
    dx = [p-q for p,q in zip(pos,prev)]
    length = d.distance(pos,prev)
    norm = math.sqrt(math.fsum(g*g for g in grad))
    dot = math.fsum(x*g for x,g in zip(dx,grad))
    need(norm>0 and dot<0 and abs(length/d.D-1)<=1e-4, "not matched descent")
    residual = math.sqrt(math.fsum((x/d.D+g/norm)**2 for x,g in zip(dx,grad)))
    need(residual <= 1e-4 and ray["method"]=="selected_parameter_gradient_ray_f32.v1" and ray["mode"]=="geometry" and
         ray["requested_l2"]==d.D and ray.get("lora") is None and group["trainable_values"]==6 and
         group["origin_id"]==expected_pin and group["result_id"]==target_pin!=expected_pin,
         "direction or owner chain")
    for block in (group,ray):
        for k,v in (("realized_l2",length),("gradient_l2",norm),("gradient_dot_step",dot)):
            need(math.isclose(block[k],v,rel_tol=2e-7,abs_tol=1e-12),"native vs recomputed "+k)
    return length,residual
