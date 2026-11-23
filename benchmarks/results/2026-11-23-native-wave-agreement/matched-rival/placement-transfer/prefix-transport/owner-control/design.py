"""Contracts shared by the runner and its stdlib-only journal reader."""
import json
import math
from itertools import product
from pathlib import Path
import runpy
import struct
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
parent = SimpleNamespace(**runpy.run_path(str(HERE.parent / "verify.py")))
u, a = parent.u, parent.a
d, k, s, need = u.d, u.k, u.s, u.need
OWNERS = ("gate", "bias")
EXTENSION = "95c496e1c970133d0e816a096fcc94f816c5cd9e329cb71e82eb9d0e4f6275c8"
PARENT = {
    "formal": ("c97829b0567f4cbe69d548b559b108f8b88021b4893a14932bf3e27b2622fa1c",
               "0acc236078ff54aae0519d10f9768d8ac13b29a481c4d2290c5bcc097bbdfba2"),
    "rehearsal": ("9312cee2b7f5764d9d5195c5b38dfa5587947e5db1564e912fa9fa8a3d7cf9d3",
                  "0a041deff6a3115308bca28749ccd2b8eb5266f0cfba9d8b1b1ce10ed2480835"),
}


def plan(previous, kind):
    spec, manifest = [(previous / f).read_bytes() for f in ("spec.json", "complete.json")]
    need((d.digest(spec), d.digest(manifest)) == PARENT[kind], "different parent plan or journal")
    sp = json.loads(spec)
    need(spec == d.encoded(u.plan(kind == "formal", EXTENSION)), "parent source or design changed")
    m = json.loads(manifest)
    need(m["spec"] == d.digest(spec) and m["files"] == k.identities(previous / "attempt"),
         "parent journal files changed")
    for b in sp["bases"]:
        need(d.digest((previous / ("base-%d.safetensors" % b)).read_bytes()) == sp["weights"][str(b)],
             "base weights changed")
    return dict(schema="spiraltorch.prefix_owner_plan.v1", kind=kind,
                parent_spec=d.digest(spec), parent_manifest=d.digest(manifest), **{
                    name: sp[name] for name in ("bases","turns","orders","epochs","train","test","weights","extension")},
                owners=list(OWNERS), sources={f:d.digest((HERE/f).read_bytes()) for f in
                                              ("PROTOCOL.md","design.py","run.py","verify.py")})


def starts(sp):
    return product(sp["bases"], sp["turns"], sp["owners"])


def cells(sp):
    return product(sp["bases"], sp["turns"], sp["owners"], d.TEACHERS, sp["orders"])


def section(v, owner):
    need(len(v) == 6 and owner in OWNERS, "wrong coordinate block")
    return v[3:] if owner == "gate" else v[:3]


def coords(v, owner):
    need(all(math.isfinite(x) and d.f32(x) == x for x in v), "invalid f32 coordinate")
    inactive = section(v, "bias" if owner == "gate" else "gate")
    need(struct.pack("<3f", *inactive) == bytes(12), "fixed weights are not positive zero")
    return section(v, owner)


def artifact(base, turn, owner, v):
    coords(v, owner)
    record = u.record(base, turn, v)
    for t in record["placements"][0]["tensors"]:
        t["requires_grad"] = t["name"] == owner
    return record


def state(raw, host, turn, owner):
    data = json.loads(raw)
    tensors = data["placements"][0]["tensors"]
    for t in tensors:
        t["values"] = [d.f32(x) for x in t["values"]]
    v = [x for t in tensors for x in t["values"]]
    need(data == artifact(host["base"], turn, owner, v), "wrong placement, order or trainable flags")
    return d.identity(raw), v


def anchor_rows(previous):
    with (previous / "attempt/steps.jsonl").open("rb") as f:
        rows = (json.loads(line) for line in f)
        return {r["cell"]: (r["keys"], d.encoded(r["members"])) for r in rows if r["i"] == 0}


def check_members(record, pair, teacher, z, anchor):
    need(len(record["members"]) == len(pair) == 2 and record["keys"] == [r["key"] for r in pair],
         "missing members or pair order changed")
    for member, row in zip(record["members"], pair):
        k.check_grad(member, 1., row[teacher], z[row["key"]] if record["i"] == 0 else None)
    if record["i"] == 0:
        need((record["keys"], d.encoded(record["members"])) == anchor, "origin VJP vs joint parent")


def check_motion(rec, before, origin, host, owner):
    report = rec["report"]
    ray = report["step"]
    target = 2*d.D
    need(report["members"] == report["total_target_tokens"] == len(rec["members"]) == 2
         and report["readout_accumulator"] == "f32" and report["lora_artifact_id"] is None
         and report["base_model_identity"] == host["base"] and ray["origin_id"] == origin
         and ray["result_id"] == rec["id"] != origin and ray["requested_l2"] == target
         and ray["method"] == "cohort_mean_gradient_ray_f64.v1" and ray["trainable_values"] == 3,
         "native ray identity, distance or mask")
    grads = [section(g["v"], owner) for g in rec["members"]]
    mean = k.avg(grads)
    dx = k.sub(coords(rec["values"],owner), coords(before,owner))
    length, norm = k.norm(dx), k.norm(mean)
    need(norm > 0 and k.dot(dx,mean) < 0, "zero or non-descending active ray")
    err = k.norm([x/target + y/norm for x,y in zip(dx,mean)])
    need(abs(length/target - 1) <= 1e-4 and err <= 1e-4 and 0 <= ray["relative_vector_error"] <= 1e-4,
         "wrong active-vector displacement")
    reference = [sum(g["loss"] for g in rec["members"])/2, norm, length, k.dot(mean,dx)] + [
        k.dot(g,dx) for g in grads]
    observed = [report["mean_input_loss"], ray["gradient_l2"], ray["realized_l2"],
                ray["gradient_dot_step"]] + report["member_loss_dots"]
    need(len(observed) == len(reference) and all(math.isclose(x,y,rel_tol=3e-7,abs_tol=1e-12)
                                                for x,y in zip(observed, reference)), "ray projected wrong gradients")
    return err


def bias_signature(rec):
    # Rotation is allowed to affect frozen gate derivatives, not the applied bias, its gradient or loss.
    return d.encoded([rec["keys"], rec["values"], [dict(g,v=section(g["v"],"bias")) for g in rec["members"]]])


def qualify(records, control):
    need(bool(records) and control in ("position", "owner"), "missing matched controls")
    p = all(min(r["h"], r["S"], r[control+"_gap_h"], r[control+"_gap_S"]) > .001
            and r["positive_pairs"] >= 3 for r in records)
    return dict(probability=p, decision=p and all(r["wins"] > max(r["wins_before"],r[control+"_wins"])
                                                   for r in records))
