"""Predictions and summaries for pre-update training observations; stdlib only."""
from collections import defaultdict
import math
from pathlib import Path
import runpy
import struct

ticks = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                           "radius-transfer/verify_training_anchors.py"))["ticks"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def coords(raw):
    require(raw and len(raw) % 4 == 0, "invalid float32 coordinate vector")
    return (tuple(x for x, in struct.iter_unpack("<f", raw)),
            tuple(ticks(x) for x, in struct.iter_unpack("<I", raw)))


def displacement(current, start):
    require(len(current[0]) == len(start[0]), "displacement layout mismatch")
    return tuple(tuple(x-y for x, y in zip(a, b)) for a, b in zip(current, start))


def tangent(gradient, move):
    g, gi = gradient
    dx, di = move
    require(len(g) == len(gi) == len(dx) == len(di) > 0, "tangent layout mismatch")
    exact = math.ldexp(sum(a*b for a, b in zip(gi, di)), -298)
    terms = [a*b for a, b in zip(g, dx)]
    require(abs(exact - math.fsum(terms)) <= 2e-15 * math.fsum(map(abs, terms)),
            "float and integer gradient-dot-displacement disagree")
    return exact


def mean(xs):
    return math.fsum(xs) / len(xs)


def controls(cases):
    require(len({r["key"] for r in cases}) == len({tuple(r["prefix"]) for r in cases}) == len(cases),
            "repeated prefix or key")
    cells = defaultdict(list)
    for c in cases:
        cells[c["plural"], c["conflict"]].append(c)
    families = {c["pair"] for c in cases}
    require(len(cells) == 4 and all(len(group) == len(families) and
            {c["pair"] for c in group} == families for group in cells.values()),
            "incomplete lexical quartets")
    return {c["key"]: sorted(other["key"] for other in cells[c["plural"], c["conflict"]]
                             if other["pair"] != c["pair"]) for c in cases}


def errors(rows, field):
    if all(r[field] is None for r in rows):
        return None
    require(all(r[field] is not None for r in rows), "partial control coverage")
    obs, pred = [r["change"] for r in rows], [r[field] for r in rows]
    err = [p-y for p, y in zip(pred, obs)]
    i = max(range(len(err)), key=lambda j: abs(err[j]))
    return dict(n=len(rows), predicted_mean=mean(pred), bias=mean(err),
                mae=mean(list(map(abs, err))), rmse=math.sqrt(mean([e*e for e in err])),
                sign_same=sum(p*y > 0 for p, y in zip(pred, obs)),
                sign_opposite=sum(p*y < 0 for p, y in zip(pred, obs)),
                sign_tied=sum(p*y == 0 for p, y in zip(pred, obs)),
                observed_zero=sum(y == 0 for y in obs), predicted_zero=sum(p == 0 for p in pred),
                worst=dict(key=rows[i]["key"], seed=rows[i]["seed"], prior=rows[i]["prior"], error=err[i]))


def describe(rows):
    require(rows and all(r["prior"] > 0 for r in rows), "no noninitial observations")
    return dict(n=len(rows), observed_mean=mean([r["change"] for r in rows]),
                observed_rising=sum(r["change"] > 0 for r in rows),
                observed_falling=sum(r["change"] < 0 for r in rows),
                preference_flips=sum(r["margin"]*r["margin0"] < 0 for r in rows),
                preference_ties=sum(r["margin"]*r["margin0"] == 0 for r in rows),
                mean_rival_minus_full_nll=mean([r["rival_minus_full_nll"] for r in rows]),
                forecasts={k: errors(rows, k) for k in ("zero", "tangent", "lexical")})


def summarize(rows, seeds):
    result = {}
    for host in ("affine", "both"):
        nonzero = [r for r in rows if r["host"] == host and r["prior"] > 0]
        orders = {str(s): describe([r for r in nonzero if r["seed"] == s]) for s in seeds}
        forecasts = [o["forecasts"] for o in orders.values()]
        by_key = defaultdict(list)
        for r in nonzero:
            by_key[r["key"]].append(r)
        result[host] = dict(orders=orders, all_visits=describe(nonzero),
                            one_prior=describe([r for r in nonzero if r["prior"] == 1]),
                            equal_prefix=dict(n=len(by_key), mean_change=mean([
                                mean([r["change"] for r in rs]) for rs in by_key.values()]),
                                rival_minus_full_nll=mean([mean([r["rival_minus_full_nll"] for r in rs])
                                                           for rs in by_key.values()])),
                            beats_zero_each_order=all(f["tangent"]["rmse"] < f["zero"]["rmse"] for f in forecasts),
                            beats_lexical_each_order=None if forecasts[0]["lexical"] is None else
                            all(f["tangent"]["rmse"] < f["lexical"]["rmse"] for f in forecasts))
    return result
