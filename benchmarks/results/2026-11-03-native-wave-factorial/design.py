"""Frozen admission and pooled estimands. No model import or tokenization."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PREV = HERE.parent / "2026-11-03-native-extraction-pullback"
for name, pin in (("metrics.py","b206435b777ce94950ebfbc92e71d611426fc71e9fb66613db911360890e0c9f"),
                  ("observe.py","97fcf24b30f295787b53c6f4e0a03e61e18b501541745a976a1ef9e3d8ccead6")):
    if hashlib.sha256((PREV / name).read_bytes()).hexdigest() != pin:
        raise ValueError("changed predecessor source: " + name)
sys.path.insert(0, str(PREV))
import observe as old  # noqa: E402
m = old.m
KINDS = ("affine","saturation","projection","both")
SOURCES = ("affine","wave_fixed")
MEASURES = ("residual_sq","correction_sq")
CFG = dict(curvature=-1., saturation=1., porosity=.05, log_radius=0.)
ANCHOR = "aa9c76c81602c693c358269b973b5679ce7c31a1a26905f8b8c87658e220d65a"
CORE = "33ab011b242cf14947a5b073984201c9ab9444e7"


def anchors():
    data = PREV / "receipts/observations.jsonl.gz"
    m.check(m.sha(data) == ANCHOR, "changed frozen factual receipts")
    expected = {}
    with gzip.open(data, "rb") as f:
        for line in f:
            row = json.loads(line)
            if row["arm"] not in SOURCES:
                continue
            idx = (row["order"],row["arm"],row["index"])
            m.check(idx not in expected, "duplicate factual reference")
            expected[idx] = row
    m.check(set(expected) == {(o,a,i) for o in m.ORDERS for a in SOURCES for i in range(m.N)}, "missing factual endpoint")
    return expected


def validate(row, prior):
    m.check(set(row) == {"order","arm","index","key","observation"}, "unexpected measurement fields")
    obs = dict(row["observation"])
    m.check(obs.pop("schema") == "spiraltorch.native_causal_lm_adaptation_observation.v2" and set(obs["sites"]) == {m.SITE},
            "missing factorial schema or wrong site")
    site = dict(obs["sites"][m.SITE])
    f = site.pop("wave_factorial")
    obs.update(schema="spiraltorch.native_causal_lm_adaptation_observation.v1", sites={m.SITE:site})
    m.check(dict(row,observation=obs) == prior, "factual CE, primal, adjoint, gradient, conditioning or owner changed")
    k = "affine" if row["arm"] == "affine" else "both"
    m.check(set(f) == {"schema","config","factual","strength","elements","input_sq","upstream_sq",
            "affine_on_outside","gate_outside","affine_outside","factual_output_max_abs_diff","factual_adjoint_max_abs_diff",*KINDS}
            and f["schema"] == "spiraltorch.wave_factorial.v1" and f["config"] == CFG and f["strength"] == .25
            and f["factual"] == k and f["elements"] == site["elements"]
            and f["factual_output_max_abs_diff"] == f["factual_adjoint_max_abs_diff"] == 0., "local anchor/recipe")
    for name in (*KINDS,"affine_on_outside"):
        m.check(set(f[name]) == set(MEASURES) and all(type(x) in (float,int) and math.isfinite(x) and x >= 0 for x in f[name].values()),
                "invalid operator energies")
    m.check(0 <= f["affine_outside"] <= f["elements"] and 0 <= f["gate_outside"] <= m.WIDTH
            and all(f["affine_on_outside"][v] <= f["affine"][v] for v in MEASURES), "outside-coordinate denominator")
    if k == "both":
        m.check(f["affine_outside"] == site["wave"]["affine_outside_values"]
                and f["gate_outside"] == site["wave"]["gate_outside_values"], "crossings differ from factual run")
    for square, rms in ((f[k]["residual_sq"],site["residual_rms"]),(f[k]["correction_sq"],site["pullback"]["correction_rms"]),
                        (f["input_sq"],site["input_rms"]),(f["upstream_sq"],site["pullback"]["upstream_rms"])):
        m.check(math.isclose(square,rms**2*f["elements"],rel_tol=2e-12,abs_tol=1e-25),"factual energy reproduction")
    return f


def fraction(a,b):
    return a / b if b else None


def decompose(e):
    a,s,p,b = (e[k] for k in KINDS)
    if a == 0:
        return None
    # Equal average over the two possible orders of enabling the stages.
    return dict(rms_relative_to_affine={k:math.sqrt(v/a) for k,v in e.items()},
                saturation_first=(a-s)/a, saturation_after_projection=(p-b)/a,
                projection_first=(a-p)/a, projection_after_saturation=(s-b)/a,
                saturation_average=((a-s)+(p-b))/(2*a), projection_average=((a-p)+(s-b))/(2*a),
                interaction=(b-s-p+a)/a, total_reduction=(a-b)/a)


def summarize(lines, expected):
    results = {}
    for order in m.ORDERS:
        arms = {}
        for arm in SOURCES:
            fs = []
            for i in range(m.N):
                line = next(lines, None)
                m.check(line is not None and 0 < len(line) <= 8192, "incomplete or oversized observation")
                fs.append(validate(json.loads(line), expected[order,arm,i]))
            elems = sum(f["elements"] for f in fs)
            m.check(elems == m.ROWS*m.WIDTH, "changed hidden rows")
            stats = dict(elements=elems, outside=sum(f["affine_outside"] for f in fs),
                         gate_outside_per_snapshot=fs[0]["gate_outside"])
            m.check(all(f["gate_outside"] == stats["gate_outside_per_snapshot"] for f in fs),"gate snapshot changed")
            for measure, denom in zip(MEASURES, ("input_sq","upstream_sq"), strict=True):
                energy = {k:math.fsum(f[k][measure] for f in fs) for k in KINDS}
                d = math.fsum(f[denom] for f in fs)
                stats[measure] = dict(energies=energy,denominator=d,
                     pooled_relative_rms={k:m.ratio(v,d) for k,v in energy.items()}, decomposition=decompose(energy),
                     affine_energy_fraction_on_crossings=fraction(math.fsum(f["affine_on_outside"][measure] for f in fs),energy["affine"]))
            arms[arm]=stats
        results[order]=arms
    m.check(next(lines,None) is None,"extra model calls")
    return results
