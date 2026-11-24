"""Prepare train-only, delivered-residual-matched counterfactual geometry artifacts."""
import json
import math
from pathlib import Path
import runpy
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
a = SimpleNamespace(**runpy.run_path(str(HERE.parent / "audit.py")))
s, x, d, need = a.s, a.x, a.d, a.need
ROUTES = ("attention", "uniform")
SCALES = ("raw", "matched")
PINS = {"formal": "161b6a3b980d9f0a916f3f99968d397e0b004eb58dc8e12b47208b0f9001c87a",
        "rehearsal": "60927d5428f4b7626c486b5007270e5e5814e2d9fe5a427e9f05ae7c9c178a0a"}
ARCHIVE = "8a0d747f394342a5af7a525aca1076a5b29f6cf377fb535f636ec1e5076a9e19"
WHEEL = "c702b7158c085c6907214c87e0dc052349f1b54ebc1c40f9d9c754ad6829c857"
EXTENSION = "f89885fc2f365824bcfcf1cdb0326344cc5b0332a48b263c7b218ecb02973958"
COUNTS = {"formal": (144, 4608, 576), "rehearsal": (16, 128, 16)}


def code():
    names = ("PROTOCOL.md", "cross_design.py", "prepare.py", "run.py", "audit.py", "workflow.py", "test_cross.py")
    return {n: d.digest((HERE / n).read_bytes()) for n in names}


def delivery(vec, traces):
    need(len(vec) == 6 and vec[:3] == [0.] * 3 and all(map(math.isfinite, vec)), "invalid bias/gate")
    rows = []
    for tr in traces:
        z, mu = tr["site_input"], tr["source"]
        need(len(z) == len(mu) == 3 and all(map(math.isfinite, (*z, *mu))), "invalid site")
        rows.append([d.f32(base + d.f32(.25 * d.f32(g * v))) for base, g, v in zip(z, vec[3:], mu)])
    return rows


def rms(rows, traces):
    need(len(rows) == len(traces) > 0 and all(len(r) == 3 for r in rows), "wrong physical rows")
    r = math.sqrt(math.fsum((out - z)**2 for row, tr in zip(rows, traces)
                             for out, z in zip(row, tr["site_input"])) / (3 * len(rows)))
    need(math.isfinite(r) and r > 0, "singular or nonfinite residual; no matching possible")
    return r


def groups(sp):
    for b in sp["bases"]:
        for teacher in x.TEACHERS:
            for order in sp["orders"]:
                yield b, teacher, order


def name(b, teacher, order, scale, donor, route):
    return f"{b}-{teacher}-{order}-{scale}-{donor}-{route}"


def saved(parent, kind):
    folder = parent / kind
    result = x.read(folder / "result.json", PINS[kind])
    inp, sp = s.data(folder, result["inputs"])
    trained = x.read(folder / "trained.json", result["trained"])
    proof = x.read(folder / "training-checked.json", result["checked"])
    scored = x.read(folder / "scored.json", result["scored"])
    need(trained["inputs"] == result["inputs"] and proof["trained"] == result["trained"]
         and scored["checked"] == result["checked"] and scored["trained"] == result["trained"]
         and scored["inputs"] == result["inputs"] and sp["kind"] == kind, "broken parent ancestry")
    return SimpleNamespace(folder=folder, inp=inp, sp=sp, owners=proof["owners"],
                           files=trained["files"], result=result, old_scores=scored["files"]["scores.jsonl"])


def banks(ctx):
    out = {}
    for b in ctx.sp["bases"]:
        for route in ROUTES:
            n = s.arm(b, route) + "-unit.json"
            sig = x.read(ctx.folder / "training" / n, ctx.files[n])
            keys = [r["key"] for r in ctx.sp["train"]]
            need(set(keys) == set(sig) and len(set(keys)) == len(keys), "wrong calibration rows")
            out[b, route] = [sig[k] for k in keys]
        for u, v in zip(out[b, ROUTES[0]], out[b, ROUTES[1]]):
            need(all(u[k] == v[k] for k in ("site_input", "base_model_identity", "frozen_lora_artifact_id",
                                             "query", "keys", "values")), "not the same upstream model")
    return out


def plan(parent, hosts_path, kind):
    ctx = saved(parent, kind)
    hosts = s.old.hosts(hosts_path / kind, ctx.sp)
    traces = banks(ctx)
    cells = []
    for b, t, o in groups(ctx.sp):
        h = hosts[str(b)]
        factual = {}
        for donor in ROUTES:
            n = s.arm(b, donor, t, o); rel = "weights/" + n + ".json"
            raw = s.s.c.pinned(ctx.folder / "training" / rel, ctx.files[rel])
            owner, vec = s.state(raw, h, donor)
            need(owner == ctx.owners[n], "foreign endpoint")
            factual[donor] = (owner, ctx.files[rel], vec)
        norms = {(donor, route): rms(delivery(v[2], traces[b, route]), traces[b, route])
                 for donor, v in factual.items() for route in ROUTES}
        target = math.sqrt(norms["attention", "attention"] * norms["uniform", "uniform"])
        for scale in SCALES:
            for donor, (oid, sha, vec) in factual.items():
                for route in ROUTES:
                    raw_rms = norms[donor, route]
                    factor = 1. if scale == "raw" else target / raw_rms
                    coords = vec[:3] + [d.f32(z * factor) for z in vec[3:]]
                    outs = delivery(coords, traces[b, route])
                    actual = rms(outs, traces[b, route])
                    need(scale == "raw" or abs(actual/target - 1) <= 2e-5, "RMS matching failed")
                    cells.append(dict(name=name(b, t, o, scale, donor, route), base=b, teacher=t, order=o,
                                      scale=scale, donor=donor, route=route, donor_id=oid, donor_file=sha,
                                      raw_rms=raw_rms, target=target, factor=factor, rms=actual, params=coords,
                                      artifact=s.record(h, route, coords), site_rows_sha256=d.digest(d.encoded(outs))))
    need(len(cells) == COUNTS[kind][0] == len({z["name"] for z in cells}), "cell coverage")
    return dict(parent=PINS[kind], code=code(), old_scores=ctx.old_scores,
                transfer_manifest=ctx.inp["transfer"], training_manifest=ctx.inp["train"], cells=cells)


def transfer(ctx):
    rows = x.read(ctx.folder / "transfer.json", ctx.inp["transfer"])
    need(rows == s.cases(ctx.sp["kind"], True), "wrong transfer split")
    journal = ctx.folder / "evaluation/scores.jsonl"
    need(d.digest(journal.read_bytes()) == ctx.old_scores, "wrong original predictions")
    return rows, a.bank(ctx.sp, rows, journal, ctx.owners)


def check_plan(parent, out, kind, pins):
    frozen = x.read(out / (kind + "-plan.json"), pins["plans"][kind])
    need(frozen == plan(parent, out / "base", kind), "frozen treatment changed")
    return frozen
