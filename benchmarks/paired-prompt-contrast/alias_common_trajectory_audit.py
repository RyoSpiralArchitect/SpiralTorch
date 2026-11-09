"""Model-free readback of every realized path, gradient and measured group outcome."""
import io
import json
import math

from alias_common_step import triangle
from alias_interference_audit import avg, close, coordinates, cosine, dot, f32, mean_vec, norm
from alias_subset_study import metrics
from shared_prefix_study import obj, pack, put, read, require, sha


def subtract(x, y):
    return [a - b for a, b in zip(x, y, strict=True)]


def read_scores(row, c, weak):
    lines = row["lines"]
    require(len(lines) == 6 and all(math.isfinite(x) and x >= 0 for x in lines), "invalid line costs")
    check = metrics(lines, c)
    require(all(abs(row[field] - check[key]) <= 2e-6 for field, key in (("cost", "conditional"), ("absolute", "absolute"))),
            "group loss does not follow from the six full lines")
    return dict(cost=row["cost"], absolute=row["absolute"], baseline_weak=lines[weak],
                worst_alias=check["worst_alias"], first_alias=lines[c], second_alias=lines[c + 3])


def validate_step(arm, t, origin, first, key, step, before, after, start, first_vector, samples, pl):
    radius = pl["radius"]
    delta = subtract(after, before)
    length = norm(delta)
    require(abs(length / radius - 1) <= 1e-4 and close(length, step["length"]), "unmatched actual step")
    prec = step["receipt"]
    vecs = [r["v"] for r in samples]
    means = [mean_vec([vecs[i] for i in ids]) for ids in pl["groups"]]
    predictions = [dot(g, delta) for g in means]
    margins = [-d / (norm(g) * length) for g, d in zip(means, predictions, strict=True) if norm(g)]
    min_margin = min(margins) if len(margins) == len(means) else None
    if arm == "initial_ray" and t:
        require((prec["origin_id"], prec["direction_id"], prec["result_id"]) ==
                (origin, first, "sha256:" + step["post"]) and prec["trainable_values"] == 64 and
                prec["requested_l2"] == (t + 1) * radius and close(prec["realized_l2"], norm(subtract(after, start))) and
                close(prec["direction_l2"], norm(first_vector)) and prec["relative_vector_error"] <= 1e-4 and
                cosine(delta, first_vector) > 1 - 1e-7, "ray drifted away from its original direction")
    else:
        dm = prec["distance_match"]
        require((prec["pre_geometry_id"], prec["post_geometry_id"], prec["base_model_identity"], prec["lora_artifact_id"]) ==
                (key["owner"], "sha256:" + step["post"], pl["base"], pl["lora"]) and
                prec["readout_accumulator"] == "f32_wide_backward" and prec["trainable_values"] == 64 and
                prec["learning_rate"] == f32(pl["rate"]) and close(length, prec["applied_l2"]) and
                (dm["origin_id"], dm["result_id"]) == (prec["pre_geometry_id"], prec["post_geometry_id"]) and
                dm["requested_l2"] == radius and close(length, dm["realized_l2"]) and dm["relative_vector_error"] <= 1e-4,
                "optimizer receipt does not identify this native step")
        if arm == "equal":
            mean = mean_vec(vecs)
            require(prec["members"] == 18 and prec["total_target_tokens"] == 324 and
                    close(prec["mean_input_loss"], avg([r["loss"] for r in samples])) and
                    close(prec["mean_gradient_l2"], norm(list(map(f32, mean)))) and
                    cosine(mean, delta) < -1 + 1e-6 and
                    all(close(dot(v, delta), d) for v, d in zip(vecs, prec["member_loss_dots"], strict=True)),
                    "not a fresh equal-vote step")
        else:
            ref = triangle(means)
            require(prec["group_sizes"] == [6] * 3 and prec["group_target_tokens"] == [108] * 3 and
                    all(close(x, avg([samples[i]["loss"] for i in ids]))
                        for x, ids in zip(prec["group_mean_losses"], pl["groups"], strict=True)) and
                    all(close(x, norm(g)) for x, g in zip(prec["group_gradient_l2"], means, strict=True)) and
                    all(close(x, y) for x, y in zip(prec["group_loss_dots"], predictions, strict=True)) and
                    min_margin is not None and close(prec["applied_min_unit_margin"], min_margin) and min_margin > 0 and
                    abs(prec["predicted_min_unit_margin"] - ref["margin"]) < 1e-8 and
                    abs(prec["hull_norm"] - ref["margin"]) < 1e-8 and cosine(ref["direction"], delta) > 1 - 1e-6 and
                    all(abs(x - y) < 1e-8 for field, k in (("unit_group_weights", "unit_weights"),
                                                            ("raw_group_weights", "raw_weights"))
                        for x, y in zip(prec[field], ref[k], strict=True)), "not the fresh common direction")
    return dict(length=length, min_normalized_margin=min_margin, predicted=predictions,
                cosine_to_first=cosine(delta, first_vector))


def summarize(points, steps, margin):
    paths = [[g["cost"] for g in p] for p in points]
    changes = [subtract(b, a) for a, b in zip(paths[:-1], paths[1:], strict=True)]
    require(len(changes) == len(steps), "step/score count differs")
    endpoint = [{k: b[k] - a[k] for k in a} for a, b in zip(points[0], points[-1], strict=True)]
    margins = [s["min_normalized_margin"] for s in steps]
    return dict(endpoint=endpoint, mean_endpoint_cost=avg([v["cost"] for v in endpoint]),
                worst_group_gain=min(-g["cost"] for g in endpoint),
                class_descents=[sum(d < -margin for d in seq) for seq in zip(*changes, strict=True)],
                all_descents=sum(all(d < -margin for d in ds) for ds in changes),
                first_nonstrict=next((i for i, ds in enumerate(changes, 1) if max(ds) >= -margin), None),
                first_later_harm=next((i for i, ds in enumerate(changes, 1) if i > 1 and max(ds) > margin), None),
                weakest_local_margin=min(margins) if None not in margins else None,
                final_step_cosine_to_first=steps[-1]["cosine_to_first"], path_length=math.fsum(s["length"] for s in steps),
                max_first_order_residual=max(abs(d - p) for ds, s in zip(changes, steps, strict=True)
                                             for d, p in zip(ds, s["predicted"], strict=True)))


def decision(summaries, steps, margin):
    common, equal, ray = (summaries[k] for k in ("common", "equal", "initial_ray"))
    gain = common["worst_group_gain"] - ray["worst_group_gain"]
    return dict(common_all_groups_every_step=common["all_descents"] == steps,
                equal_harms_idle_at_end=equal["endpoint"][2]["cost"] > margin,
                common_avoids_later_ray_harm=common["all_descents"] == steps and ray["first_later_harm"] is not None,
                common_minus_ray_worst_gain=gain, common_better_worst_gain=gain > margin)


def audit(planned, pp, measured, mp, out):
    from alias_common_trajectory import ARMS, BUDGET, CLASSES, MARGIN, RADIUS, STATES, STEPS, code_pins
    pl = obj(planned / "plan.json", pp)
    m = obj(measured / "measured.json", mp)
    attempt = json.loads((measured / "attempt.json").read_bytes())
    process = json.loads((measured / "process.json").read_bytes())
    require(attempt == dict(attempt=1, plan=pp, native=pl["native"]) and process["exit"] == 0 and
            0 < process["seconds"] <= pl["seconds"] + 1 and
            read(measured / "process.log", process["log"]).decode().splitlines()[-1:] == [mp], "incomplete worker receipt")
    require(pl["code"] == code_pins() and pl["arms"] == list(ARMS) and pl["states"] == list(STATES) and
            pl["classes"] == list(CLASSES) and (pl["steps"], pl["radius"], pl["margin"]) == (STEPS, RADIUS, MARGIN) and
            pl["budget"] == m["counts"] == BUDGET and m["plan"] == pp and m["native"] == pl["native"], "wrong trial")
    groups = pl["groups"]
    require(list(map(len, groups)) == [6] * 3 and sorted(i for ids in groups for i in ids) == list(range(18)),
            "not a disjoint balanced cohort")
    meaning = {i: c for c, slots in enumerate(groups) for i in slots}
    records = iter(io.BytesIO(read(measured / "journal.jsonl", m["journal"], 16 << 20)))

    def take(event, **tags):
        row = json.loads(next(records))
        require(row["event"] == event and all(row[k] == x for k, x in tags.items()), "broken journal sequence")
        return row

    results = {}
    for state in STATES:
        origin = pl["origins"][state]
        start, layout = coordinates(measured / (state + ".json"), origin[7:])
        summaries, full = {}, {}
        baseline = None
        for arm in ARMS:
            owner, before = origin, start
            points, steps, first, first_vector = [], [], None, None
            first_samples = None
            replay = pl["replay"][state]["equal" if arm == "equal" else "common"]
            for t in range(STEPS + 1):
                key = dict(state=state, arm=arm, time=t, owner=owner)
                take("point", **key)
                samples = [take("example", slot=i, **key) for i in range(18)]
                if t < STEPS:
                    require(all(len(r["v"]) == 64 and all(map(math.isfinite, r["v"])) and r["loss"] == f32(r["cost"])
                                for r in samples), "invalid gradient row")
                else:
                    require(all("v" not in r and "loss" not in r for r in samples), "terminal point not score-only")
                if t == 0:
                    # Keep the identity of the original weaker alias, even if the ranking later reverses.
                    weak = [max((meaning[i], meaning[i] + 3), key=r["lines"].__getitem__) for i, r in enumerate(samples)]
                    base_digest = sha(pack([{k: r[k] for k in ("v", "lines", "cost", "absolute")} for r in samples]))
                    if baseline is None:
                        baseline = base_digest
                    require(baseline == base_digest, "arms did not start from the same gradients and scores")
                per_slot = [read_scores(r, meaning[i], weak[i]) for i, r in enumerate(samples)]
                if t == 0:
                    first_samples = per_slot
                points.append([{k: avg([per_slot[i][k] for i in ids]) for k in per_slot[0]} for ids in groups])
                if t == 1:
                    prior_scores = [[r["cost"], r["absolute"], r["worst_alias"]] for r in per_slot]
                    require(sha(pack(prior_scores)) == replay["scores"], "first scores no longer replay")
                if t == STEPS:
                    break
                row = take("step", **key)
                after, other_layout = coordinates(measured / f"{state}-{arm}-{t + 1:02d}.json", row["post"])
                require(other_layout == layout, "frozen geometry changed")
                if t == 0:
                    require(row["post"] == replay["artifact"] and sha(pack(row["receipt"])) == replay["receipt"],
                            "first step no longer replays")
                    first, first_vector = "sha256:" + row["post"], subtract(after, start)
                steps.append(validate_step(arm, t, origin, first, key, row, before, after, start, first_vector, samples, pl))
                before, owner = after, "sha256:" + row["post"]
            summary = summarize(points, steps, pl["margin"])
            summary["endpoint_chord"] = norm(subtract(before, start))
            summary["end"] = owner
            summary["harmed_examples"] = [{k: sum(per_slot[i][k] - first_samples[i][k] > MARGIN for i in ids)
                                            for k in ("cost", "absolute", "baseline_weak")} for ids in groups]
            summaries[arm] = summary
            full[arm] = dict(points=points, steps=steps)
        results[state] = dict(arms=summaries, path=full, gates=decision(summaries, STEPS, MARGIN))
    require(next(records, None) is None, "extra model calls or rows")
    gates = [r["gates"] for r in results.values()]
    out.mkdir(mode=0o700)
    return put(out / "results.json", dict(plan=pp, measurement=mp, classes=CLASSES, states=results,
        primary_pass=all(g["common_all_groups_every_step"] and g["equal_harms_idle_at_end"] for g in gates),
        turning_prevents_ray_harm_at_both_starts=all(g["common_avoids_later_ray_harm"] for g in gates),
        turning_improves_worst_endpoint_at_both_starts=all(g["common_better_worst_gain"] for g in gates),
        scope="Sequential, matched, deterministic trajectories on the same 18 synthetic training prompts. "
              "Not held-out, not pretrained, not general quality. CPU, no external ML framework."))
