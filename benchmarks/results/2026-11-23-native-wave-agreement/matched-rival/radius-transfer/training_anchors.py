"""Join just the first training step of each order to the same-snapshot factual margin."""
import argparse
import base64
import json
import math
from pathlib import Path
import re
import runpy
import struct
import sys

HERE = Path(__file__).resolve().parent
E = runpy.run_path(str(HERE / "experiment.py"))
S, D, FIT = E["R"], E["D"], E["FIT"]
need, sha, enc = E["need"], E["sha"], E["enc"]
HOSTS = (("affine", "affine"), ("both", "wave"))


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def coverage(full_ce, choice_ce):
    """Probability consistent with the full CE's binary32 rounding cell (not model error bounds)."""
    need(math.isfinite(full_ce) and math.isfinite(choice_ce) and min(full_ce, choice_ce) >= 0, "invalid NLL")
    q = f32(max(0., full_ce))
    bits, = struct.unpack("<I", struct.pack("<f", q))
    at = lambda b: struct.unpack("<f", struct.pack("<I", b))[0]
    prev = at(bits-1) if bits else 0.
    nxt = at(bits+1) if bits < 0x7f7fffff else 2.**128
    lo, hi = (prev+q)*.5-choice_ce, (q+nxt)*.5-choice_ce
    need(hi >= 0., "conditional loss exceeds full CE rounding cell")
    lo, mid = max(0., lo), max(0., q-choice_ce)
    return dict(full_ce_f32=q, choice_nll=choice_ce, coverage_nll=mid, coverage_nll_rounding=[lo, hi],
                gold_given_pair=math.exp(-choice_ce), gold_mass=math.exp(-q),
                pair_mass=math.exp(-mid), pair_mass_rounding=[math.exp(-hi), math.exp(-lo)])


def slopes(full, choice, move):
    need(len(full) == len(choice) == len(move) and len(full) > 0
         and all(all(map(math.isfinite, xs)) for xs in (full, choice, move)), "bad gradient or move")
    dot = lambda a, b: math.fsum(x*y for x,y in zip(a,b))
    full_dot, choice_dot = dot(full, move), dot(choice, move)
    # Subtract the two saved f32 gradients *after* widening, coordinate by coordinate.
    mass_dot = dot((x-y for x,y in zip(full,choice)), move)
    scale = math.fsum((abs(x)+abs(y))*abs(d) for x,y,d in zip(full,choice,move))
    need(abs(math.fsum((full_dot, -choice_dot, -mass_dot))) <= 1e-14*scale, "loss slopes do not close")
    return dict(full_nll=full_dot, choice_nll=choice_dot, coverage_nll_estimate=mass_dot)


def independent_choice(margin_grad, q, stored, tolerance):
    need(0 < q <= 1 and len(margin_grad)==len(stored), "invalid binary margin anchor")
    ref = [-v/q for v in stored]
    ref_l2 = D["length"](ref)
    need(ref_l2 > 0, "zero factual reference")
    err = D["length"](a-b for a,b in zip(margin_grad,ref))/ref_l2
    need(err <= tolerance, "independent margin pullback differs")
    return [-q*v for v in margin_grad], err


def training_reader(root, kind, source_plan):
    manifest = (root/"SHA256SUMS").read_bytes()
    need(sha(manifest) == E["ROOTS"][2], "wrong matched training archive")
    pins = dict((name,h) for h,name in (line.split("  ",1) for line in manifest.decode().splitlines()))
    need(pins[kind+"/plan.json"] == source_plan, "wrong parent training plan")
    prefix = kind+"/attempt/"
    allowed = re.compile(r"(?:identity\.json|sealed\.json|receipt\.json|states/[0-9a-f]{64}\.json|"
                         r"(?:11|17|31|73)-(?:affine|wave)_(?:full|rival)\.(?:jsonl|g32))")
    opened = {}
    def read(name):
        need(allowed.fullmatch(name) is not None, "not a training journal, gradient or state")
        raw = (root/(prefix+name)).read_bytes()
        need(sha(raw) == pins[prefix+name], "changed training input: "+name)
        opened[name] = pins[prefix+name]
        return raw
    receipt = json.loads(read("receipt.json"))
    need(receipt["plan"] == source_plan and receipt["result_sha256"] == pins[prefix+"worker-result.json"],
         "unsealed training archive")
    return read, opened


def state(raw, p, family, pin):
    need("sha256:"+sha(raw) == pin, "wrong state bytes")
    rec = json.loads(raw)
    gate, bias = FIT["OLD"]["check_geometry"](rec, p, p["base"], family)
    for t in rec["placements"][0]["tensors"]:
        if t["requires_grad"]:
            del t["values"]
    return D["f32"](gate+bias), sha(enc(rec))


def check_join(case, first, arm, owner):
    need((first["key"],first["gold"],first["rival"],first["position"],first["objective"],first["step"]["origin_id"])
         == (case["key"],case["gold"],case["rival"] if arm=="rival" else None,len(case["prefix"]),arm,owner),
         "different prefix, supervision, objective or starting snapshot")
    D["check_objective"](arm, first["loss"], first["declared"])


def first_update(read, stage, arm, p, family, case, owner, initial, layout):
    journal = read(stage+".jsonl").splitlines()
    tape = read(stage+".g32")
    stride = 8*p["shape"][1]
    need(len(journal)==len(p["train"]) and len(tape)==len(journal)*stride, "truncated first-step source")
    first, gradient = json.loads(journal[0]), tape[:stride]
    check_join(case, first, arm, owner)
    need(sha(gradient)==first["grad_sha256"] and sys.byteorder=="little", "wrong gradient or byte order")
    grad = memoryview(gradient).cast("f")
    step = first["step"]
    end = step["result_id"]
    need(re.fullmatch(r"sha256:[0-9a-f]{64}", end), "invalid endpoint id")
    nxt, after_layout = state(read("states/"+end[7:]+".json"), p, family, end)
    need(layout==after_layout, "frozen geometry changed")
    D["check_step"](initial, nxt, grad, step, dict(shape=p["shape"],target_l2=D["DISTANCE"],tolerance=1e-4))
    return first, gradient, grad, D["delta"](nxt, initial)


def run(capture, matched, kind):
    p, native, source = S["inputs"](capture, kind)
    need("evaluation" not in p and p["archive_manifest"]==E["ROOTS"][2], "not a matched train-only capture")
    read, opened = training_reader(matched, kind, p["source_plan"])
    sealed = json.loads(read("sealed.json"))
    ident = json.loads(read("identity.json"))
    need(ident == dict(base=p["base"],extension=E["P"]["EXTENSION"],backend="rust_f32_cpu_prepacked",
                       head="original_packed",accumulator="f32_wide_backward"), "wrong training decoder")
    forward = {(r["key"],r["host"]):r["receipt"] for r in native}
    seeds = (17,31,73) if kind=="gpt2" else (11,)
    starts = [min(p["train"], key=lambda c:sha(("relative/%s/%s"%(s,c["key"])).encode())) for s in seeds]
    keys = {c["key"] for c in starts}
    need(len(keys)==len(seeds), "repeated first case")
    results = []
    for host, family in HOSTS:
        anchors = p["anchors"][host]
        owner = sealed["initial-"+family]
        need(set(anchors)==keys and owner==p["geometry"][host]["pin"], "unmatched initial cases or geometry")
        original = read("states/"+owner[7:]+".json")
        need(original==base64.b64decode(p["geometry"][host]["bytes"],validate=True), "wrong initial artifact")
        initial, layout = state(original,p,family,owner)
        for seed, case in zip(seeds,starts):
            arms = {a:first_update(read,"%s-%s_%s"%(seed,family,a),a,p,family,case,owner,initial,layout)
                    for a in ("full","rival")}
            fr, _, fg, fd = arms["full"]
            cr, cg_raw, cg, cd = arms["rival"]
            decl = cr["declared"]
            receipt, anchor = forward[case["key"],host], anchors[case["key"]]
            need(receipt["contrast"]["target_minus_alternative"]==anchor["margin"]==decl["mean_target_minus_declared_rival"]
                 and base64.b64decode(anchor["gradient"],validate=True)==cg_raw
                 and f32(fr["loss"])==f32(decl["full_ce"]), "independent same-state losses or gradient differ")
            q = f32(-math.expm1(-decl["declared_ce"]))
            need(q==anchor["q"], "rival probability differs")
            unit = [v for k in S["BLOCKS"] for v in S["vector"](S["unpack"](receipt["arms"][host][k]),p["shape"][1])]
            independent, err = independent_choice(unit, q, cg, p["anchor_tolerance"])
            measure = coverage(decl["full_ce"],decl["declared_ce"])
            changes = {a:slopes(fg,cg,d) for a,d in (("full",fd),("rival",cd))}
            for a, own in (("full","full_nll"),("rival","choice_nll")):
                need(math.isclose(changes[a][own],arms[a][0]["step"]["gradient_dot_step"],
                                  rel_tol=1e-8,abs_tol=5e-15), "diagonal does not match native update")
                other = slopes(fg,independent,arms[a][3])
                changes[a]["independent_coverage_slope"] = other["coverage_nll_estimate"]
            results.append(dict(seed=seed,key=case["key"],plural=case["plural"],conflict=case["conflict"],
                                host=host,origin=owner,margin=anchor["margin"],
                                independent_margin_relative_l2_error=err,probabilities=measure,slopes=changes,
                                endpoints={a:arm[0]["step"]["result_id"] for a,arm in arms.items()},
                                gradient_hashes={a:arm[0]["grad_sha256"] for a,arm in arms.items()}))
    return dict(schema="spiraltorch.same_snapshot_choice_coverage.v1",kind=kind,rows=results,
                scope="first step of each archived order only",capture_inputs=source,training_inputs=opened,
                training_identity=ident,training_manifest=E["ROOTS"][2],capture_manifest=S["ARCHIVE"],
                source_training_plan=p["source_plan"],analysis_sha256=sha(Path(__file__).read_bytes()),
                new_model_calls=0,new_updates=0,new_scored_rows=0)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture",type=Path)
    parser.add_argument("matched",type=Path)
    parser.add_argument("kind",choices=("rehearsal","gpt2"))
    parser.add_argument("new_output",type=Path)
    args=parser.parse_args()
    result=run(args.capture,args.matched,args.kind)
    with args.new_output.open("xb") as f:
        f.write(enc(result))
