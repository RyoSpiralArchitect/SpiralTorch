"""Verify a complete readback without loading any model; report all declared arms."""
import argparse
import json
import math
from pathlib import Path
import runpy
import statistics
import types

d = types.SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("design.py"))))
need = d.need


def check_close(x, y, tol, context):
    need(abs(x-y) <= tol, "%s: %.12g vs %.12g" % (context,x,y))


def verify_costs(c):
    need(c["stop_id"] == 0 and c["target_counts"] == [2,2] and
         len(c["line_nlls"]) == len(c["conditional_nlls"]) == len(c["token_nlls"]) == 2 and
         all(len(s) == 2 for s in c["token_nlls"]), "wrong line layout")
    values = c["line_nlls"]+c["conditional_nlls"]+[c["menu_set_nll"]]+sum(c["token_nlls"], [])
    need(all(math.isfinite(x) and x >= 0 for x in values), "invalid cost")
    for i in (0,1):
        check_close(sum(c["token_nlls"][i]), c["line_nlls"][i], 1e-10, "content plus stop")
        check_close(c["conditional_nlls"][i]+c["menu_set_nll"], c["line_nlls"][i], 1e-10, "menu plus coverage")
    check_close(sum(math.exp(-x) for x in c["conditional_nlls"]), 1., 1e-10, "binary menu")
    need(c["winner_index"] == int(c["line_nlls"][1] < c["line_nlls"][0]), "winner")


def verify_sentinel(logits, costs):
    need(len(logits) == 3 and all(len(v) == 11 and all(map(math.isfinite,v)) for v in logits),
         "wrong full-vocabulary sentinel")
    def nll(v, token):
        top = max(v)
        return (top-v[token])+math.log(math.fsum(math.exp(x-top) for x in v))
    errors = [abs(nll(v,tok) - costs["token_nlls"][i][j]) for i in (0,1)
              for j, (v,tok) in enumerate(((logits[0],7+i), (logits[1+i],0)))]
    need(max(errors) <= 3e-6, "uncached content/termination mismatch")
    return max(errors)


def verify_step(record, arm, role, plan, i, ordering, previous, before):
    s = record["step"]
    mode = "lora" if arm == d.REF else "geometry"
    other = "geometry" if arm == d.REF else "lora"
    epoch, ix = divmod(i,len(ordering))
    row = ordering[ix]
    need(record["key"] == row["key"] and s["epoch"] == epoch and s["example_index"] == ix and
         s["update"] == i+1 and s["plan_id"] == plan and s["mode"] == mode, "step ordering or mode")
    obj = s["objective"]["scores"]
    need(s["objective"]["kind"] == "menu" and obj["gold_index"] == row[role] and obj["mix"] == .5 and
         math.isfinite(s["loss"]) and obj["loss"] == s["loss"], "wrong training objective")
    r = s["parameter_ray"]
    group = r[mode]
    need(r["method"] == "selected_parameter_gradient_ray_f32.v1" and r["mode"] == mode and
         r["requested_l2"] == d.D and 0 <= r["relative_vector_error"] <= 1e-4 and
         r.get(other) is None and s.get("pre_"+other+"_id") is None and s.get("post_"+other+"_id") is None,
         "wrong descent method or owner")
    after = record["coordinates"]
    need(all(x == d.f32(x) and math.isfinite(x) for x in after) and
         group["trainable_values"] == len(after) and group["origin_id"] == s["pre_"+mode+"_id"] == previous and
         group["result_id"] == s["post_"+mode+"_id"] == record["id"] != previous, "broken owner chain")
    measured = d.distance(before, after)
    check_close(measured, d.D, d.D*1e-4, "independent step distance")
    for block in (r,group):
        check_close(measured, block["realized_l2"], 1e-10, "native step distance")
        need(math.isfinite(block["gradient_l2"]) and block["gradient_l2"] > 0 and
             math.isfinite(block["gradient_dot_step"]) and block["gradient_dot_step"] < 0, "not descent")
    return measured


def verify_checkpoint(data, pin, plan, arm, origin, current, count, vector):
    cp = json.loads(data)
    mode,other = ("lora","geometry") if arm == d.REF else ("geometry","lora")
    need(d.identity(data) == pin and cp["plan_id"] == plan and cp["updates"] == count and cp["mode"] == mode
         and cp["base_id"] == (d.BASE if arm == d.REF else d.FROZEN), "endpoint checkpoint")
    owner = cp[mode]
    need(cp[other] == {"origin_id":None,"id":None,"weights":None} and owner["origin_id"] == origin and
         owner["id"] == current == d.owner_id(owner["weights"].encode(),arm) and
         d.coordinates(owner["weights"],arm) == vector, "endpoint owner bytes")


def metrics(rows, costs):
    vals = {k:[] for k in ("H","A","H_content","H_stop","gold_nll","choice_nll","coverage_nll")}
    wins = ties = 0
    for row in rows:
        c = costs[row["key"]]
        head, attractor = row["head"], row["attractor"]
        hs, ass = 1-2*head, 1-2*attractor
        margin = c["line_nlls"][1]-c["line_nlls"][0]
        vals["H"].append(hs*margin)
        vals["A"].append(ass*margin)
        for j,k in enumerate(("H_content","H_stop")):
            vals[k].append(hs*(c["token_nlls"][1][j]-c["token_nlls"][0][j]))
        vals["gold_nll"].append(c["line_nlls"][head]/2)
        vals["choice_nll"].append(c["conditional_nlls"][head]/2)
        vals["coverage_nll"].append(c["menu_set_nll"]/2)
        wins += hs*margin > 0
        ties += margin == 0
    out = {k:math.fsum(v)/len(rows) for k,v in vals.items()}
    check_close(out["H_content"]+out["H_stop"],out["H"],1e-10,"head cue decomposition")
    check_close(out["choice_nll"]+out["coverage_nll"],out["gold_nll"],1e-10,"NLL decomposition")
    return dict(out, wins=wins, ties=ties, n=len(rows))


def reports(rows, costs):
    frames = sorted({r["frame"] for r in rows})
    return dict(metrics(rows,costs), H_by_pair={f:metrics([r for r in rows if r["frame"]==f],costs)["H"]
                                              for f in frames})


def contrast(before, head, attractor):
    dh, ah = head["H"]-before["H"], attractor["H"]-before["H"]
    da, aa = head["A"]-before["A"], attractor["A"]-before["A"]
    return dict(delta_H=dh, delta_A=da, control_delta_H=ah, control_delta_A=aa, B=dh-ah,
                S=((dh-da)-(ah-aa))/2, delta_H_content=head["H_content"]-before["H_content"],
                delta_H_stop=head["H_stop"]-before["H_stop"], gold_gain=before["gold_nll"]-head["gold_nll"],
                choice_gain=before["choice_nll"]-head["choice_nll"],
                coverage_gain=before["coverage_nll"]-head["coverage_nll"],
                positive_pairs=sum(v > before["H_by_pair"][k] for k,v in head["H_by_pair"].items()),
                wins_before=before["wins"], wins_after=head["wins"], ties_before=before["ties"], ties_after=head["ties"])


GATES = ("delta_H","B","S","gold_gain")


def passes(orders):
    return bool(orders) and all(all(x[k] > d.FLOOR for k in GATES) and x["positive_pairs"] >= 3 for x in orders)


def wave_advantage(wave, affine):
    need(len(wave)==len(affine), "unmatched controls")
    return passes(wave) and all(all(w[k]-a[k] > d.FLOOR for k in ("delta_H","S")) for w,a in zip(wave,affine))


def read_json(path):
    return json.loads(path.read_bytes())


def inspect(root):
    spec, out = d.read_plan(root), root/"attempt"
    manifest = read_json(root/"completed.json")
    plan_sha = d.digest((root/"spec.json").read_bytes())
    need(plan_sha == manifest["spec"], "complete manifest plan")
    ids = [d.cell_key(*t) for t in d.cells(spec)]
    def paths(folder, names):
        return {folder+"/"+n+".json" for n in names}
    expected = (paths("starts",spec["arms"]) | paths("scores",spec["arms"]+ids) |
                paths("checkpoints",ids) | {"steps.jsonl"})
    need(set(manifest["files"]) == expected == {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
         and set(manifest["checkpoints"]) == set(ids), "missing or extra artifacts")
    for f,sha in manifest["files"].items():
        need(d.digest((out/f).read_bytes()) == sha, "changed artifact: "+f)
    snapshots, paths_by_cell, origins = {}, {}, {}
    worst_sentinel = worst_ray = 0.
    steps = scores = 0
    def read_scores(name,pin):
        nonlocal worst_sentinel, scores
        payload = read_json(out/"scores"/(name+".json"))
        need(payload["owner"] == pin and [r["key"] for r in payload["rows"]] ==
             [r["key"] for r in spec["train"]+spec["test"]], "snapshot rows or owner")
        costs = {}
        for record in payload["rows"]:
            verify_costs(record["costs"])
            costs[record["key"]] = record["costs"]
        worst_sentinel = max(worst_sentinel,verify_sentinel(payload["sentinel"],payload["rows"][0]["costs"]))
        snapshots[name] = {split:reports(spec[split],costs) for split in ("train","test")}
        scores += len(costs)
    for arm in spec["arms"]:
        start = read_json(out/"starts"/(arm+".json"))
        pin,raw = start["id"], start["weights"]
        need(pin == d.owner_id(raw.encode(),arm) and start["base"] == (d.BASE if arm == d.REF else d.FROZEN)
             and start["frozen_lora"] == (None if arm == d.REF else d.LORA) and
             (arm != d.REF or pin == d.LORA), "starting owner")
        origins[arm] = pin, d.coordinates(raw,arm)
        read_scores(arm,pin)
    need(len({tuple(coords) for arm,(_,coords) in origins.items() if arm != d.REF}) == 1, "unmatched origins")
    with (out/"steps.jsonl").open("rb") as journal:
        for arm,role,seed in d.cells(spec):
            key = d.cell_key(arm,role,seed)
            start,vec = origins[arm]
            previous,initial = start, vec
            raw,order = d.session_plan(spec,arm,role,seed)
            pin = d.identity(raw)
            count = len(order)*spec["epochs"]
            travelled = []
            for i in range(count):
                r = json.loads(next(journal))
                need(r["cell"] == key, "interleaved path")
                length = verify_step(r,arm,role,pin,i,order,previous,vec)
                travelled.append(length)
                worst_ray = max(worst_ray, abs(length/d.D-1))
                previous,vec = r["id"], r["coordinates"]
            steps += count
            verify_checkpoint((out/"checkpoints"/(key+".json")).read_bytes(),manifest["checkpoints"][key],
                              pin,arm,start,previous,count,vec)
            paths_by_cell[key] = dict(endpoint_l2=d.distance(initial,vec), path_l2=math.fsum(travelled))
            read_scores(key,previous)
        need(journal.readline() == b"", "extra updates")
    contrasts, candidates, wins, summary = {}, [], [], {}
    for arm in spec["arms"]:
        contrasts[arm] = {}
        for seed in spec["orders"]:
            h,a = (snapshots[d.cell_key(arm,r,seed)] for r in d.TEACHERS)
            contrasts[arm][str(seed)] = {s:contrast(snapshots[arm][s],h[s],a[s]) for s in ("test","train")}
        orders = [contrasts[arm][str(seed)]["test"] for seed in spec["orders"]]
        med = {k:statistics.median(x[k] for x in orders) for k in orders[0]}
        misses = [k for k in GATES if any(x[k] <= d.FLOOR for x in orders)]
        if any(x["positive_pairs"] < 3 for x in orders):
            misses.append("positive_pairs")
        summary[arm] = dict(median=med, failed=misses)
        if spec["formal"] and arm != d.REF and passes(orders):
            candidates.append(arm)
    if spec["formal"]:
        for site in d.SITES:
            wave,affine = ([contrasts[r+"@"+site][str(seed)]["test"] for seed in spec["orders"]] for r in d.RECIPES)
            if wave_advantage(wave,affine):
                wins.append(site)
    return dict(schema="spiraltorch.placement_transfer_readback.v1", formal=spec["formal"], spec_sha256=plan_sha,
                evidence_sha256=d.digest((root/"completed.json").read_bytes()), updates=steps, score_readouts=scores,
                uncached_forwards=3*len(snapshots), checkpoints=len(ids), snapshots=snapshots, paths=paths_by_cell,
                contrasts=contrasts, summary=summary, candidates=candidates, wave_specific=wins,
                max_realized_length_relative_error=worst_ray, max_direct_token_cost_error=worst_sentinel)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("output",type=Path)
    args = p.parse_args()
    result = inspect(args.root)
    print(d.save(args.output,result))
