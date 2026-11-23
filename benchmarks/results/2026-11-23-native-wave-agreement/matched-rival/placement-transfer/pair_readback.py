"""Export per-order results; independently derive role contrasts from raw token costs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def check(ok, message):
    if not ok:
        raise ValueError(message)


def load(path, sha=None):
    raw = path.read_bytes()
    if sha:
        check(hashlib.sha256(raw).hexdigest() == sha, "digest: "+path.name)
    return json.loads(raw)


def main(root, report_path, output):
    report = load(report_path)
    spec = load(root/"spec.json", report["spec_sha256"])
    manifest = load(root/"completed.json", report["evidence_sha256"])
    check(spec["formal"] is True and report["formal"] is True and manifest["spec"] == report["spec_sha256"],
          "not the bound formal study")
    held = {r["key"]: r for r in spec["test"]}
    frames = {r["frame"] for r in held.values()}
    check(len(held) == 16 and len(frames) == 4, "split size")
    def observe(snapshot):
        name = "scores/"+snapshot+".json"
        rows = load(root/"attempt"/name, manifest["files"][name])["rows"]
        data = {}
        for r in rows:
            if r["key"] in held:
                pairs = r["costs"]["token_nlls"]
                totals = list(map(math.fsum, pairs))
                lo = min(totals)
                coverage = lo-math.log(sum(math.exp(lo-x) for x in totals))
                data[r["key"]] = dict(d=totals[1]-totals[0], stop=pairs[1][1]-pairs[0][1],
                                      gold=totals[held[r["key"]]["head"]]/2, coverage=coverage/2,
                                      winner=(totals[1]>totals[0])-(totals[1]<totals[0]))
        check(set(data) == set(held),"missing held-out readout")
        return data

    def effect(before, after, role, column, subset):
        # A contrast of means within number groups; no signed-score accumulator from analyze.py.
        def change(bit):
            vals = [after[k][column]-before[k][column] for k in subset if held[k][role] == bit]
            return math.fsum(vals)/len(vals)
        return (change(0)-change(1))/2

    all_rows = []
    for arm in spec["arms"]:
        origin = observe(arm)
        for order in spec["orders"]:
            head_key = "%s-head-%d" % (arm,order)
            head = observe(head_key)
            other = observe("%s-attractor-%d" % (arm,order))
            h1,a1,h2,a2 = [effect(origin,end,role,"d",held) for end in (head,other) for role in ("head","attractor")]
            stop = effect(origin,head,"head","stop",held)
            nll,coverage = (math.fsum(origin[k][col]-head[k][col] for k in held)/len(held)
                            for col in ("gold","coverage"))
            positive = sum(effect(origin,head,"head","d",[k for k in held if held[k]["frame"] == frame]) > 0
                           for frame in frames)
            values = dict(delta_H=h1,B=h1-h2,S=((h1-a1)-(h2-a2))/2,gold_gain=nll,positive_pairs=positive,
                          delta_H_stop=stop,delta_H_content=h1-stop,coverage_gain=coverage,choice_gain=nll-coverage)
            first = report["contrasts"][arm][str(order)]["test"]
            check(all(abs(value-first[key]) < 1e-12 for key,value in values.items()), "contrast mismatch: "+head_key)
            row = dict(arm=arm,order=order,**values)
            row.update(wins_before=first["wins_before"],wins_after=first["wins_after"],
                       changed_choices=sum(head[k]["winner"] != origin[k]["winner"] for k in held),
                       endpoint_l2=report["paths"][head_key]["endpoint_l2"],
                       path_l2=report["paths"][head_key]["path_l2"])
            all_rows.append(row)
    check(len(all_rows)==39,"missing arm/order")
    def candidate(rows):
        return all(min(r["delta_H"],r["B"],r["S"],r["gold_gain"]) > .001 and r["positive_pairs"] >= 3 for r in rows)
    groups = {arm:[r for r in all_rows if r["arm"]==arm] for arm in spec["arms"]}
    candidates = [a for a in spec["arms"] if a != "lora-only" and candidate(groups[a])]
    wave_specific = []
    for arm in candidates:
        if arm.startswith("wave@"):
            site = arm.partition("@")[2]
            paired = zip(groups[arm],groups["affine@"+site])
            if all(min(w[k]-a[k] for k in ("delta_H","S")) > .001 and w["order"] == a["order"] for w,a in paired):
                wave_specific.append(site)
    check((candidates,wave_specific) == (report["candidates"],report["wave_specific"]), "gate mismatch")
    with output.open("x", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root","report","output"):
        parser.add_argument(key,type=Path)
    args = parser.parse_args()
    main(args.root,args.report,args.output)
