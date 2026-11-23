"""Post hoc: split signed weaker-margin changes into separation and centering, with no model."""
import itertools
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_design as c


def geometry(positive, negative):
    # Both are singular-minus-plural gaps; only the desired sign changes across the pair.
    sep, offset = (positive-negative)/2, (positive+negative)/2
    weak = min(positive, -negative)
    c.need(math.isclose(weak, sep-abs(offset), abs_tol=2e-13, rel_tol=2e-13), "bad pair geometry")
    return (sep, abs(offset), weak)


def change(before, after):
    sep, center, weak = after[0]-before[0], before[1]-after[1], after[2]-before[2]
    c.need(math.isclose(weak, sep+center, rel_tol=2e-13, abs_tol=2e-13), "bad gain decomposition")
    return [sep, center, weak]


def stats(values):
    if not values:
        return dict(n=0, separation=None, recentering=None, weaker=None,
                    separation_larger=0, recentering_larger=0, equal=0)
    s, r, w = [math.fsum(v)/len(values) for v in zip(*values)]
    return dict(n=len(values), separation=s, recentering=r, weaker=w,
                separation_larger=sum(v[0]>v[1] for v in values), recentering_larger=sum(v[1]>v[0] for v in values),
                equal=sum(v[0]==v[1] for v in values))


def read(folder, pin):
    result = c.x.read(folder/"result.json", pin)
    provenance = c.x.read(folder/"scored.json", result["scored"])
    f = folder/"scores.jsonl"
    c.need(provenance["scores"] == c.d.digest(f.read_bytes()), "wrong logits")
    by = {}
    for r in c.s.s.legacy.records(f):
        key = (r["name"], r["key"])
        c.need(key not in by, "duplicate row")
        z = r["logits"]; c.s.v.u.bits(z)
        by[key] = z[7]-z[8]
    c.need(len(by) == result["forwards"], "incomplete")
    pairs = {}
    visited = set()
    for p in result["pairs"]:
        b,t,o = p["base"], p["teacher"], p["order"]
        scale, keys = p["scale"], tuple(p["keys"])
        pair = (b,t,o,p["template"],keys)
        output = pairs.setdefault(pair,{})
        c.need(scale not in output, "duplicate pair in result")
        geos = output[scale] = {}
        for donor, route in itertools.product(c.ROUTES, repeat=2):
            code = donor[0].upper()+route[0].upper()
            n = c.name(b,t,o,scale,donor,route)
            ids = [(n,k) for k in keys]
            z = geos[code] = geometry(*(by[i] for i in ids))
            c.need(z[2] == p["outcomes"][code]["weaker"] and int(z[2]>0) == p["outcomes"][code]["both"],
                   "sign orientation mismatches scored pair")
            visited.update(ids)
    c.need(visited == by.keys() and all(set(v) == set(c.SCALES) for v in pairs.values()), "unused rows or pairs")
    out = []
    for b in [None]+sorted({key[0] for key in pairs}):
        for t in c.x.TEACHERS:
            entries = [(key, v) for key, v in pairs.items() if key[1]==t and (b is None or key[0]==b)]
            overlap = dict(neither=0, both=0, raw_only=0, matched_only=0)
            marginal = {k: [] for k in ("all", *overlap)}
            route = {}
            for scale in c.SCALES:
                for ref in ("AA","AU","UA"):
                    for cohort in ("all", "UU_only"):
                        route[scale, ref, cohort] = []
            for key, v in entries:
                raw, match = v["raw"]["UU"], v["matched"]["UU"]
                category = ("neither","matched_only","raw_only","both")[2*int(raw[2]>0)+int(match[2]>0)]
                overlap[category] += 1
                diff = change(match, raw)
                marginal["all"].append(diff); marginal[category].append(diff)
                for scale, cells in v.items():
                    only = cells["UU"][2]>0 and all(cells[i][2]<=0 for i in ("AA","AU","UA"))
                    for ref in ("AA","AU","UA"):
                        diff = change(cells[ref],cells["UU"])
                        route[scale,ref,"all"].append(diff)
                        if only:
                            route[scale,ref,"UU_only"].append(diff)
            out.append(dict(base=b,teacher=t,overlap=overlap,matched_to_raw_UU={k:stats(d) for k,d in marginal.items()},
                            into_UU=[dict(scale=k[0],from_cell=k[1],cohort=k[2],**stats(d)) for k,d in route.items()]))
    return dict(primary_result=pin, additional_model_calls=0, post_hoc=True, comparisons=out)


def table(result):
    columns = "base teacher order template scale coords serving pairs both individual_correct ties mean_weaker"
    lines = [columns.replace(" ", "\t")]
    for s in result["strata"]:
        for code, (coords, serving) in zip(("AA","AU","UA","UU"), itertools.product(c.ROUTES, repeat=2)):
            v = s["outcomes"][code]
            row = [s["base"],s["teacher"],s["order"],s["template"],s["scale"],coords,serving,s["pairs"],
                   v["both"],v["correct"],v["ties"],format(v["mean_weaker"],".17g")]
            lines.append("\t".join(map(str,row)))
    return ("\n".join(lines)+"\n").encode("ascii")


if __name__ == "__main__":
    folder, pin, dest = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    result = read(folder,pin)
    print(c.d.save(dest/"gap-readback.json", result))
    raw = table(c.x.read(folder/"result.json",pin))
    with (dest/"strata.tsv").open("xb") as f:
        f.write(raw)
    print(c.d.digest(raw))
