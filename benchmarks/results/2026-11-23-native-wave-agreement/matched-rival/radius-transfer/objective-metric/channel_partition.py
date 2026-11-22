"""Post-hoc accounting of the affine-to-saturation dot change, not a model ablation."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import runpy
import struct


M = runpy.run_path(str(Path(__file__).with_name("metric.py")))
PINS = {"rehearsal": ("bb7b61a9ce8c43b56d022a249491f23e076a53ceb12db873a65932c4ec916684", 1, 16),
        "gpt2": ("cb1b245bf9435a3b36fbe31a80b8468bc6a65d973d163f37482089dd0d88f9f8", 3, 768)}


def parts(pair):
    full, choice = ([M["ticks"](n) for n, in struct.iter_unpack("<I", raw)] for raw in pair)
    M["need"](len(full) == len(choice) and len(full) > 0 and len(full) % 2 == 0, "invalid gate/bias blocks")
    dot = [x*y for x,y in zip(full,choice)]
    h = len(dot)//2
    return dot[:h], dot[h:]


def partition(affine, saturation):
    f, s = parts(affine), parts(saturation)
    n = len(f[0])
    M["need"](n == len(s[0]), "different channel widths")
    sign = lambda x: (x>0)-(x<0)
    value = lambda x: math.ldexp(x, -298)
    fa, sa = [list(map(sum, zip(*pair))) for pair in (f,s)]
    F, S = sum(fa), sum(sa)
    without = [(sign(F-x), sign(S-y)) for x,y in zip(fa,sa)]
    change = [y-x for x,y in zip(fa,sa)]
    # Take the largest absolute contribution to the change; no favorable-sign selection.
    top = max(range(n), key=lambda j: abs(change[j]))
    changed_signs = [j for j,t in enumerate(without) if t != (sign(F),sign(S))]
    return dict(channels=n, affine_dot=value(F), saturation_dot=value(S),
                delta_gate=value(sum(s[0])-sum(f[0])),delta_bias=value(sum(s[1])-sum(f[1])),
                delta_positive=sum(v>0 for v in change),delta_negative=sum(v<0 for v in change),
                highest_abs_delta_channel=top,highest_abs_delta=value(change[top]),
                max_share_of_absolute_change=abs(change[top])/sum(map(abs,change)) if any(change) else None,
                original_signs=[sign(F),sign(S)],same_sign_pair_without_one=n-len(changed_signs),
                exceptions=[dict(channel=j, signs=list(without[j])) for j in changed_signs],
                only_gate_replaced_dot=value(sum(s[0])+sum(f[1])),
                only_bias_replaced_dot=value(sum(f[0])+sum(s[1])))


def support(affine, saturation, gate, bias, x, limit, top):
    n = len(x)
    M["need"](n == len(gate) == len(bias) and all(abs(v)<limit for v in gate), "raw gates must be strictly inside")
    f32 = lambda v: struct.unpack("<f",struct.pack("<f",v))[0]
    z = [f32(f32(v*g)+b) for v,g,b in zip(x,gate,bias)]
    outside = {i for i,v in enumerate(z) if abs(v)>limit}
    M["need"](outside == {i for i,(v,g,b) in enumerate(zip(x,gate,bias)) if abs(f32(v*g+b))>limit},
              "boundary classification depends on a multiply-add rounding")
    changed = set()
    for a,s in zip(affine,saturation):
        M["need"](len(a)==len(s)==8*n, "wrong two-block buffer size")
        for i in range(2*n):
            if a[4*i:4*i+4] != s[4*i:4*i+4]:
                changed.add(i%n)
    M["need"](not changed-outside, "an unsaturated coordinate changed")
    return dict(raw_gate_abs_max=max(map(abs,gate)), threshold=limit, post_affine_outside=sorted(outside),
                changed_parameter_channels=sorted(changed), outside_unchanged=sorted(outside-changed),
                top_delta_pre_saturation=z[top], nearest_threshold_distance=min(abs(abs(v)-limit) for v in z))


def run(kind, path, capture=None):
    raw = path.read_bytes()
    pin, count, width = PINS[kind]
    M["need"](hashlib.sha256(raw).hexdigest()==pin, "wrong local-map tape")
    if capture is not None:
        S = runpy.run_path(str(Path(__file__).resolve().parents[2] / "parameter-vjp/radius-sweep/sweep.py"))
        p, captures, _ = S["inputs"](capture,kind)
        S["check_parent"](p)
        artifact = json.loads(S["raw_record"](p,"affine"))
        gate,bias = [S["tensor_at"](artifact,k)["values"] for k in ("gate","bias")]
        xs = {r["key"]:S["unpack"](r["receipt"]["site_input"]) for r in captures if r["host"]=="affine"}
    rows = []
    for r in map(json.loads,raw.splitlines()):
        pair = lambda k: [base64.b64decode(s,validate=True) for s in r["maps"][k]]
        result = partition(pair("affine"),pair("saturation"))
        M["need"](result["channels"]==width, "wrong channel count")
        if capture is not None:
            x = xs[r["key"]]
            M["need"](hashlib.sha256(x).hexdigest()==r["site_input_sha"], "wrong site input")
            result["support"] = support(pair("affine"),pair("saturation"),gate,bias,M["values"](x),
                                        p["config"]["saturation"],result["highest_abs_delta_channel"])
        rows.append(dict(key=r["key"], **result))
    M["need"](len(rows)==len({r["key"] for r in rows})==count, "missing or duplicated anchor")
    return dict(post_hoc=True, source_sha256=pin, rows=rows, model_calls=0)


if __name__=="__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kind",choices=tuple(PINS))
    ap.add_argument("tape",type=Path)
    ap.add_argument("new_output",type=Path)
    ap.add_argument("--capture",type=Path,help="check whether gradient changes coincide with the post-affine saturation boundary")
    args=ap.parse_args()
    result=run(args.kind,args.tape,args.capture)
    with args.new_output.open("x") as f:
        json.dump(result,f,sort_keys=True,allow_nan=False,indent=2)
        f.write("\n")
