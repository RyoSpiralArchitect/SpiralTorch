"""Independent binary32-integer dot products and a second probability identity."""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import re
import struct

MANIFEST = "2e60ffdf268a9a9e78be20f7816f2b94a2e32ddf3b41e49d7ee24696b3641be3"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ticks(bits):
    """An exact signed integer in units of the least positive binary32 value."""
    sign, e, fraction = bits >> 31, (bits >> 23) & 255, bits & 0x7fffff
    require(e < 255, "nonfinite value")
    mag = fraction if e == 0 else ((1 << 23) + fraction) << (e-1)
    return -mag if sign else mag


def verify(matched, report, report_sha):
    digest = lambda raw: hashlib.sha256(raw).hexdigest()
    raw = report.read_bytes()
    require(digest(raw) == report_sha, "wrong analysis report")
    rep = json.loads(raw)
    sums = (matched/"SHA256SUMS").read_bytes()
    require(digest(sums) == rep["training_manifest"] == MANIFEST, "wrong training manifest")
    pins = {n:h for h,n in (s.split("  ",1) for s in sums.decode().splitlines())}
    kind = rep["kind"]
    seeds = {"rehearsal": [11], "gpt2": [17,31,73]}[kind]
    require([(x["host"],x["seed"]) for x in rep["rows"]] == [(h,s) for h in ("affine","both") for s in seeds],
            "expected all first-step anchors")
    def read(rel):
        path = kind+"/attempt/"+rel
        raw = (matched/path).read_bytes()
        require(digest(raw) == pins[path] == rep["training_inputs"][rel], "changed "+rel)
        return raw
    def coords(pin):
        require(re.fullmatch("sha256:[0-9a-f]{64}",pin), "invalid state ID")
        data = read("states/"+pin[7:]+".json")
        require(pin == "sha256:"+digest(data), "wrong state")
        site, = json.loads(data)["placements"]
        ts = {t["name"]:t["values"] for t in site["tensors"] if t["requires_grad"]}
        require(set(ts)=={"gate","bias"}, "not gate and bias")
        vals = ts["gate"]+ts["bias"]
        return [ticks(b) for b, in struct.iter_unpack("<I",struct.pack("<%df"%len(vals),*vals))]
    largest, decisions = 0., 0
    for x in rep["rows"]:
        family = {"affine":"affine","both":"wave"}[x["host"]]
        start = coords(x["origin"])
        gradients, ends, rows = {}, {}, {}
        for arm in ("full","rival"):
            stem = "%s-%s_%s" % (x["seed"],family,arm)
            row = json.loads(read(stem+".jsonl").splitlines()[0])
            data = read(stem+".g32")[:len(start)*4]
            require(row["key"]==x["key"] and row["step"]["origin_id"]==x["origin"] and
                    row["step"]["result_id"]==x["endpoints"][arm] and
                    digest(data)==row["grad_sha256"]==x["gradient_hashes"][arm], "wrong row or gradient")
            gradients[arm] = [ticks(b) for b, in struct.iter_unpack("<I",data)]
            ends[arm], rows[arm] = coords(row["step"]["result_id"]), row
            require(len(start)==len(ends[arm])==len(gradients[arm]), "incomplete coordinates")
        for arm, end in ends.items():
            # Evaluate each linear form at the endpoint and origin, *then* subtract.
            dot = lambda g, p: sum(v*w for v,w in zip(g,p))
            full, choice = [dot(gradients[a],end)-dot(gradients[a],start) for a in ("full","rival")]
            for k, expected in [("full_nll",full),("choice_nll",choice),("coverage_nll_estimate",full-choice)]:
                actual = x["slopes"][arm][k]
                ref = math.ldexp(expected,-298)
                require((actual>0)-(actual<0) == (expected>0)-(expected<0)
                        and math.isclose(actual,ref,rel_tol=1e-12,abs_tol=1e-20), "wrong slope: "+k)
                largest = max(largest, abs(actual-ref))
                decisions += 1
        with localcontext() as ctx:
            ctx.prec = 70
            loss = Decimal.from_float(rows["full"]["loss"])
            margin = Decimal.from_float(rows["rival"]["declared"]["mean_target_minus_declared_rival"])
            independent_mass = (-loss).exp() * (1+(-margin).exp())
            lo, hi = map(Decimal.from_float, x["probabilities"]["pair_mass_rounding"])
            require(lo <= independent_mass <= hi, "probability outside f32-rounding envelope")
    return dict(rows=len(rep["rows"]),exact_slope_comparisons=decisions,max_abs_dot_error=largest,
                training_report=report_sha)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("matched",type=Path)
    p.add_argument("report",type=Path)
    p.add_argument("report_sha256")
    args=p.parse_args()
    print(json.dumps(verify(args.matched,args.report,args.report_sha256),sort_keys=True))
