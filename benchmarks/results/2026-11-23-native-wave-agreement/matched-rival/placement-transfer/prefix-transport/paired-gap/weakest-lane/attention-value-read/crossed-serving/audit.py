"""Replay a serving-route cross from all logits, gate artifacts and delivered site rows; no ML import."""
import itertools
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_design as c

BITS = tuple("".join(v) for v in itertools.product("01", repeat=4))
CELLS = ("AA", "AU", "UA", "UU")


def pattern(a):
    c.need(len(a) == 4 and all(v in (0, 1) for v in a), "bad pair outcomes")
    key = "".join(str(int(v)) for v in a)
    return key, {"0001": "both_needed", "0011": "weights_only_suffices",
                 "0101": "read_only_suffices", "0111": "either_suffices"}.get(key)


def paired(row0, row1):
    p, n = row0[7] - row0[8], row1[8] - row1[7]
    return dict(both=int(min(p, n) > 0), correct=int(p > 0) + int(n > 0),
                ties=int(p == 0) + int(n == 0), weaker=min(p, n))


def summarize(pairs):
    c.need(bool(pairs), "empty stratum")
    by = {}
    for n in CELLS:
        u = [p["outcomes"][n] for p in pairs]
        by[n] = dict(both=sum(e["both"] for e in u), correct=sum(e["correct"] for e in u),
                     ties=sum(e["ties"] for e in u), mean_weaker=math.fsum(e["weaker"] for e in u)/len(u))
    hist = {k: 0 for k in BITS}
    transitions = dict(both_needed=0, weights_only_suffices=0, read_only_suffices=0, either_suffices=0)
    for p in pairs:
        key, why = pattern([p["outcomes"][n]["both"] for n in CELLS])
        hist[key] += 1
        if why:
            transitions[why] += 1
    aa, au, ua, uu = [by[n]["mean_weaker"] for n in CELLS]
    return dict(pairs=len(pairs), outcomes=by, histogram=hist, transitions=transitions,
                margin_effects=dict(read_at_A=au-aa, read_at_U=uu-ua, weights_at_A=ua-aa,
                                    weights_at_U=uu-au, interaction=(uu-ua)-(au-aa)))


def audit(parent, out, kind, freeze):
    frozen = c.x.read(out / "frozen.json", freeze)
    ctx = c.saved(parent, kind)
    plan = c.check_plan(parent, out, kind, frozen)
    dest = out / kind
    receipt = c.s.k.read(dest / "scored.json")
    c.need(receipt["scores"] == c.d.digest((dest / "scores.jsonl").read_bytes())
           and receipt["witnesses"] == c.d.digest((dest / "witnesses.jsonl").read_bytes())
           and (receipt["frozen"], receipt["plans"], receipt["updates"]) ==
               (freeze, frozen["plans"][kind], 0), "bad scored journal")
    traces = c.banks(ctx)
    rows, old = c.transfer(ctx)
    hosts = c.s.old.hosts(out / "base" / kind, ctx.sp)
    journal = c.s.s.legacy.records(dest / "scores.jsonl")
    witnesses = c.s.s.legacy.records(dest / "witnesses.jsonl")
    raw_bank = {}
    calls = diag = batches = 0
    errors = []
    for cell in plan["cells"]:
        b, route, donor = cell["base"], cell["route"], cell["donor"]
        r = next(witnesses)
        raw = r["state"].encode()
        oid, coords = c.s.state(raw, hosts[str(b)], route)
        c.need((r["name"], r["owner"], coords) == (cell["name"], oid, cell["params"]), "wrong witness owner")
        tr = traces[b, route]
        c.need(c.d.digest(c.d.encoded(r["rows"])) == cell["site_rows_sha256"]
               and r["rows"] == c.delivery(coords, tr), "saved physical site rows")
        measured = c.rms(r["rows"], tr)
        c.need(measured == cell["rms"], "changed delivered energy")
        batches += (len(tr) + 15)//16
        if cell["scale"] == "matched":
            errors.append(abs(measured/cell["target"] - 1))
        factual = cell["scale"] == "raw" and route == donor
        archived = old[c.s.arm(b, donor, cell["teacher"], cell["order"])]
        if factual:
            c.need(oid == cell["donor_id"] and c.d.digest(raw) == cell["donor_file"], "not the factual endpoint")
        bucket = raw_bank[cell["name"]] = {}
        for row in rows:
            rr = next(journal); key = row["key"]
            c.need((rr["name"], rr["key"], rr["owner"]) == (cell["name"], key, oid)
                   and key not in bucket, "score owner/key/order")
            z = bucket[key] = rr["logits"]
            packed = c.s.v.u.bits(z)
            if factual:
                c.need(packed == c.s.v.u.bits(archived[key]), "not the saved 11-logit row")
                diag += 1
            calls += 1
    c.need(next(journal, None) is None and next(witnesses, None) is None, "unclaimed scores/witnesses")
    c.need((len(raw_bank), calls, batches) == c.COUNTS[kind] and
           (calls, diag, batches) == (receipt["forwards"], receipt["diagonal"], receipt["site_batches"])
           and diag*4 == calls and len(errors) == len(raw_bank)//2 and max(errors) <= 2e-5, "wrong budget/energy")

    parent_cells = {(g["base"], g["teacher"], g["order"], g["template"], g["mode"]): g
                    for g in ctx.result["transfer"]["cells"]}
    cases = []
    strata = []
    for b, t, o in c.groups(ctx.sp):
        for template in sorted({r["template"] for r in rows}):
            pairs = c.x.pairs([r for r in rows if r["template"] == template], t)
            for scale in c.SCALES:
                label = dict(base=b, teacher=t, order=o, template=template, scale=scale)
                subset = []
                for l, r in pairs:
                    c.need((l[t], r[t]) == (0, 1), "signed pair order")
                    outs = {}
                    for code, (donor, serve) in zip(CELLS, itertools.product(c.ROUTES, repeat=2)):
                        z = raw_bank[c.name(b, t, o, scale, donor, serve)]
                        outs[code] = paired(z[l["key"]], z[r["key"]])
                    subset.append(dict(**label, keys=[l["key"], r["key"]], outcomes=outs))
                st = summarize(subset)
                if scale == "raw":
                    for code, mode in (("AA", "attention"), ("UU", "uniform")):
                        src = parent_cells[b, t, o, template, mode]
                        hit = st["outcomes"][code]
                        c.need((hit["both"], hit["correct"], hit["ties"], hit["mean_weaker"]) ==
                               (src["both"], src["raw"]["correct"], src["raw"]["ties"], src["mean_min"]),
                               "original task accounting changed")
                cases.extend(subset); strata.append(dict(**label, **st))
    totals = []
    for scale in c.SCALES:
        for t in c.x.TEACHERS:
            for b in [None] + ctx.sp["bases"]:
                group = [p for p in cases if p["scale"] == scale and p["teacher"] == t
                         and (b is None or p["base"] == b)]
                st = summarize(group)
                needed = [p for p in group if not p["outcomes"]["AA"]["both"] and p["outcomes"]["UU"]["both"]]
                distinct = len({(p["base"], *p["keys"]) for p in needed})
                totals.append(dict(scale=scale, teacher=t, base=b, factual_gap_pairs=sum(st["transitions"].values()),
                                   distinct_gap_pairs=distinct, **st))
    return dict(frozen=freeze, parent=c.PINS[kind], scored=c.d.digest((dest / "scored.json").read_bytes()),
                cells=len(raw_bank), forwards=calls, site_batches=batches, diagonal_rows=diag, updates=0,
                max_matched_rms_relative_error=max(errors), totals=totals, strata=strata, pairs=cases)


if __name__ == "__main__":
    parent, out = map(Path, sys.argv[1:3])
    kind, freeze = sys.argv[3:5]
    result = audit(parent, out, kind, freeze)
    path = out / kind / "result.json"
    if sys.argv[5:] == ["--verify"]:
        c.need(path.read_bytes() == c.d.encoded(result), "result digest does not replay")
        print(c.d.digest(path.read_bytes()))
    else:
        c.need(not sys.argv[5:], "unexpected mode")
        print(c.d.save(path, result))
