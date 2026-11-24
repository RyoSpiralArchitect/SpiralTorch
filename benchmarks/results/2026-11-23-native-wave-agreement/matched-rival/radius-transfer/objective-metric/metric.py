"""Binary32 adjoint recovery and exact parameter-space alignment; no model runtime."""
import math
from pathlib import Path
import runpy
import struct


ticks = runpy.run_path(str(Path(__file__).resolve().parents[1] / "verify_training_anchors.py"))["ticks"]


def need(ok, message):
    if not ok:
        raise ValueError(message)


def values(raw):
    return tuple(x for x, in struct.iter_unpack("<f", raw))


def packed(xs):
    return struct.pack("<%df" % len(xs), *xs)


def recover(site, gradient, strength):
    """Only one supervised final row; its native affine bias is strength times dy."""
    need(strength == .25 and len(site) and len(gradient) == 2 * len(site), "wrong affine layout or strength")
    x, b = values(site), values(gradient[len(site):])
    need(all(math.isfinite(v) and abs(v) > 2.**-126 for v in b), "bias does not identify a normal incoming value")
    dy = packed([4*v for v in b])
    need(packed([v*.25 for v in values(dy)]) == gradient[len(site):], "inexact affine inverse")
    need(packed([u*v for u,v in zip(x,b)]) == gradient[:len(site)], "independent affine gate does not close")
    return dy


def relative(actual, reference):
    need(len(actual) == len(reference), "different direction lengths")
    norm = math.fsum(v*v for v in reference)
    need(norm > 0, "undefined relative error against zero direction")
    return math.sqrt(math.fsum((a-b)**2 for a,b in zip(actual, reference))/norm)


def alignment(full, choice):
    need(len(full) == len(choice) and len(full) > 0 and len(full) % 8 == 0, "expected gate followed by bias")
    ints = lambda b: [ticks(i) for i, in struct.iter_unpack("<I", b)]
    f, c = ints(full), ints(choice)
    split = len(f)//2
    parts = [sum(x*y for x,y in zip(a,b)) for a,b in [(f[:split],c[:split]),(f[split:],c[split:])]]
    dot = sum(parts)
    ff, cc = sum(x*x for x in f), sum(x*x for x in c)
    wide = math.ldexp(dot, -298)
    need(wide == math.fsum(a*b for a,b in zip(values(full), values(choice))), "wide and exact dots disagree")
    length = math.ldexp(math.sqrt(cc), -149)
    return dict(full_l2=math.ldexp(math.sqrt(ff), -149), choice_l2=length,
                gate_dot=math.ldexp(parts[0], -298), bias_dot=math.ldexp(parts[1], -298), full_choice_dot=wide,
                cosine=dot/(math.sqrt(ff)*math.sqrt(cc)) if ff and cc else None,
                full_slope_at_1_256=(-wide/length)/256 if cc else None,
                full_along_negative_choice=("undefined" if not cc else "down" if dot>0 else "up" if dot<0 else "flat"))
