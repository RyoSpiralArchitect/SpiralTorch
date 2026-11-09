"""Reconstruct each finite mixed-objective step and distinguish full-line, word and share improvements."""
import io
import json
import math

from alias_common_trajectory_audit import read_scores, subtract
from alias_interference_audit import avg, close, coordinates, cosine, dot, f32, mean_vec, norm
from shared_prefix_study import obj, pack, put, read, require, sha


def certify(groups, delta, receipt):
    """A feasible simplex point upper-bounds the best margin; its worst cosine lower-bounds it."""
    n = len(groups)
    means = [mean_vec([r["v"] for r in group]) for group in groups]
    norms = [norm(v) for v in means]
    require(len(delta) == 64 and norm(delta) > 0 and all(x > 0 for x in norms), "zero/empty geometry direction")
    units = [[v / size for v in vec] for vec, size in zip(means, norms, strict=True)]
    weights = receipt["unit_group_weights"]
    require(len(weights) == n and all(w >= 0 and math.isfinite(w) for w in weights) and
            close(math.fsum(weights), 1.), "invalid convex weights")
    z = [dot(weights, col) for col in zip(*units, strict=True)]
    upper = norm(z)
    require(upper > 1e-10, "no resolved common direction")
    lower = min(dot(u, z) / upper for u in units)
    beta = [a / size for a, size in zip(weights, norms, strict=True)]
    predicted = [dot(g, delta) for g in means]
    realized = min(-d / (l * norm(delta)) for d, l in zip(predicted, norms, strict=True))
    require(lower > 0 and abs(upper - lower) <= 1e-9 and close(upper, receipt["hull_norm"]) and
            close(lower, receipt["predicted_min_unit_margin"]) and cosine(z, delta) < -1 + 1e-6 and
            realized > 0 and close(realized, receipt["applied_min_unit_margin"]), "false common-descent certificate")
    expected = dict(raw_group_weights=[w / math.fsum(beta) for w in beta], group_gradient_l2=norms,
                    group_mean_losses=[avg([r["loss"] for r in g]) for g in groups], group_loss_dots=predicted)
    for k, ys in expected.items():
        require(all(close(a, b) for a, b in zip(receipt[k], ys, strict=True)), "wrong native " + k)
    return dict(unit_weights=weights, min_margin=realized, predicted=predicted)


def point_metrics(rows, ids):
    return {**{name: avg([rows[i]["cost"] for i in group]) for name, group in zip(("on", "off", "idle"), ids, strict=True)},
            **{name: avg([rows[i]["singletons"][k]["cost"] for i in ids[2]])
               for k, name in enumerate(("idle_line", "standby_line"))},
            "standby_menu_line": avg([rows[i]["lines"][5] for i in ids[2]]),
            "idle_accepted": avg([rows[i]["absolute"] for i in ids[2]]),
            "standby_within_accepted": avg([rows[i]["lines"][5] - rows[i]["absolute"] for i in ids[2]])}


def token_costs(row, tol):
    v = row["terms"]
    require(len(v) == 5 and all(math.isfinite(x) and x >= 0 for x in v) and
            abs(math.fsum(v[:3]) - row["lines"][2]) <= tol and
            abs(math.fsum((v[0], v[3], v[4])) - row["lines"][5]) <= tol, "full-line token decomposition differs")
    return v


def summarize(points, steps, individual, tokens, margin):
    changes = [{k: b[k] - a[k] for k in a} for a, b in zip(points[:-1], points[1:], strict=True)]
    require(len(changes) == len(steps) and len(tokens) == 5 and len(individual) == 6, "incomplete trajectory")
    endpoints = {k: v - points[0][k] for k, v in points[-1].items()}
    protected = ("on", "off", "idle", "standby_line", "standby_menu_line")
    joint = [all(row[k] < -margin for k in protected) for row in changes]
    return dict(changes=endpoints, jointly_descending=sum(joint),
                first_missed_step=next((t for t, ok in enumerate(joint, 1) if not ok), None),
                improvements_per_metric={k: sum(row[k] < -margin for row in changes) for k in endpoints},
                standby_prompt_changes=individual, standby_prompts_better=sum(d < -margin for d in individual),
                component_changes=dict(zip(("status", "idle_word", "idle_stop", "standby_word", "standby_stop"), tokens, strict=True)),
                min_margin=min(r["min_margin"] for r in steps), fourth_active_steps=sum(
                    len(r["unit_weights"]) == 4 and r["unit_weights"][3] > 1e-8 for r in steps))


def decision(arms, n, margin):
    target = arms["guard_standby"]
    delta = target["changes"]["standby_line"]
    better = {a: arms[a]["changes"]["standby_line"] - delta for a in ("three", "guard_idle")}
    return dict(primary=target["jointly_descending"] == n, target_advantage=better,
                beats_both_controls=min(better.values()) > margin, all_six_benefit=target["standby_prompts_better"] == 6,
                standby_word_gains=target["component_changes"]["standby_word"] < -margin,
                standby_share_gains=target["changes"]["standby_within_accepted"] < -margin)


def audit(protocol, pp, folder, mp, out):
    from alias_protection import ALIASES, ARMS, BUDGET, CLASSES, CROSS_TOL, MARGIN, SECONDS, STATES, STEPS, control_point, sources
    p, m = obj(protocol / "plan.json", pp), obj(folder / "measured.json", mp)
    proc = json.loads((folder / "process.json").read_bytes())
    require(p["code"] == sources() and m["plan"] == pp and p["native"] == m["native"] and
            json.loads((folder / "attempt.json").read_bytes()) == dict(attempt=1, plan=pp, native=p["native"]) and
            proc["exit"] == 0 and 0 < proc["seconds"] <= SECONDS + 1 and
            read(folder / "process.log", proc["log"]).decode().splitlines()[-1:] == [mp] and
            m["counts"] == p["budget"] == BUDGET and p["arms"] == ARMS and p["arm_order"] == list(ARMS) and
            p["states"] == list(STATES) and p["classes"] == list(CLASSES) and
            p["margin"] == MARGIN and p["steps"] == STEPS, "invalid completed attempt")
    ids = p["groups"]
    require(list(map(len, ids)) == [6] * 3 and sorted(i for g in ids for i in g) == list(range(18)), "wrong groups")
    labels = {i: c for c, group in enumerate(ids) for i in group}
    iterator = iter(io.BytesIO(read(folder / "journal.jsonl", m["journal"], 24 << 20)))
    def take(event, tags):
        row = json.loads(next(iterator))
        require(row["event"] == event and all(row[k] == v for k, v in tags.items()), "journal order changed")
        return row
    result = {}
    for state in STATES:
        origin = p["origins"][state]
        start, layout = coordinates(folder / f"{state}.json", origin[7:])
        baseline, tokens0 = None, None
        summaries, paths = {}, {}
        for arm, guard in ARMS.items():
            points, steps = [], []
            owner, before = origin, start
            for t in range(STEPS + 1):
                tags = dict(state=state, arm=arm, t=t, owner=owner)
                take("start", tags)
                rows = [take("example", dict(**tags, slot=i)) for i in range(18)]
                for i, row in enumerate(rows):
                    c = labels[i]
                    read_scores(row, c, c + 3)
                    require(("singletons" in row) == (c == 2) and
                            ("terms" in row) == (c == 2 and ((t == 0 and arm == "three") or t == STEPS)), "unexpected scoring scope")
                    singles = row.get("singletons", [])
                    if c == 2:
                        require([v["index"] for v in singles] == list(ALIASES) and
                                all(abs(r["cost"] - row["lines"][k]) <= CROSS_TOL for k, r in zip(ALIASES, singles, strict=True)),
                                "wrong protected absolute line")
                    for r in (row, *singles):
                        require(("loss" in r) == ("v" in r) == (t < STEPS), "wrong score/gradient domain")
                        if t < STEPS:
                            require(len(r["v"]) == 64 and all(math.isfinite(x) for x in r["v"]) and
                                    r["loss"] == f32(r["cost"]), "invalid gradient")
                prior = p["replay"][state][t]
                if arm == "three" or t == 0:
                    require(owner == prior["owner"] and control_point(rows) == prior["scores"], "lost control scores")
                if t == 0:
                    first = [rows[i]["singletons"][1]["cost"] for i in ids[2]]
                    singles_pin = sha(pack([rows[i]["singletons"] for i in ids[2]]))
                    if arm == "three":
                        baseline = singles_pin
                        tokens0 = [token_costs(rows[i], CROSS_TOL) for i in ids[2]]
                    require(singles_pin == baseline, "initial singleton gradients differ across arms")
                points.append(point_metrics(rows, ids))
                if t == STEPS:
                    break
                step = take("step", tags)
                r = step["receipt"]
                after, end_layout = coordinates(folder / f"{state}-{arm}-{t+1:02}.json", step["post"])
                length = norm(subtract(after, before))
                nxt = "sha256:" + step["post"]
                dm = r["distance_match"]
                require(end_layout == layout and
                        (r["pre_geometry_id"], r["post_geometry_id"], r["base_model_identity"], r["lora_artifact_id"]) ==
                        (owner, nxt, p["base"], p["lora"]) and r["readout_accumulator"] == "f32_wide_backward" and
                        r["trainable_values"] == 64 and r["learning_rate"] == f32(p["rate"]) and
                        r["group_target_tokens"] == ([108] * 3 + ([] if guard is None else [18])) and
                        r["group_sizes"] == [6] * (3 if guard is None else 4) and close(length, r["applied_l2"]) and
                        abs(length / p["radius"] - 1) <= 1e-4 and dm["requested_l2"] == p["radius"] and
                        (dm["origin_id"], dm["result_id"]) == (owner, nxt) and close(dm["realized_l2"], length) and
                        dm["relative_vector_error"] <= 1e-4, "bad native update or unequal distance")
                if arm == "three":
                    require(step["post"] == prior["post"] and sha(pack(r)) == prior["receipt"], "lost control step")
                groups = [[rows[i] for i in group] for group in ids]
                if guard is not None:
                    groups.append([rows[i]["singletons"][ALIASES.index(guard)] for i in ids[2]])
                steps.append(certify(groups, subtract(after, before), r))
                before, owner = after, nxt
            singles = subtract([rows[i]["singletons"][1]["cost"] for i in ids[2]], first)
            term_changes = mean_vec([subtract(token_costs(rows[i], CROSS_TOL), base)
                                     for i, base in zip(ids[2], tokens0, strict=True)])
            summaries[arm] = summarize(points, steps, singles, term_changes, MARGIN)
            paths[arm] = dict(points=points, steps=steps, endpoint=owner)
        result[state] = dict(arms=summaries, paths=paths, gates=decision(summaries, STEPS, MARGIN))
    require(next(iterator, None) is None, "trailing calls")
    out.mkdir(mode=0o700)
    return put(out / "results.json", dict(plan=pp, measurement=mp, states=result,
        primary_pass=all(v["gates"]["primary"] for v in result.values()),
        specificity_pass=all(v["gates"]["beats_both_controls"] for v in result.values()),
        scope="Training-only, post-discovery synthetic comparison. Group means, individual prompts, exact-word likelihood, "
              "within-accepted share and stop likelihood are distinct outcomes."))
