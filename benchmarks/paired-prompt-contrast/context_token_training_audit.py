"""Check training selectivity against absolute probabilities, controls and native update receipts."""
import io
import json
import math

from alias_common_trajectory_audit import subtract
from alias_guarded_training import VECTOR, fingerprint, geometry_path
from alias_guarded_training_audit import evaluate
from alias_individual_step import witness
from alias_interference_audit import avg, close, coordinates, dot, f32, mean_vec, norm
from alias_protection_audit import certify
from context_token_training import (
    ARMS, BUDGET, CROSS_TOL, EFFECT, FLOOR, GROUPS, MARGIN, RADIUS, RATE, REFUSALS, STATES,
    STEPS, TARGET, WALL, check_seed, frozen_inventory, nll, pairs, sources,
)
from byte_release_artifacts import PACKAGE_ROOTS
from shared_prefix_study import obj, put, read, require, sha


def validate_rows(rows, groups):
    require([r["slot"] for r in rows] == list(range(18)), "wrong score roster")
    for c, ids in enumerate(groups):
        for i in ids:
            r = rows[i]
            lines, terms, cond = r["lines"], r["terms"], r["conditional"]
            require(len(lines) == len(cond) == 6 and len(terms) == 3 and
                    all(math.isfinite(x) and x >= 0 for x in (*lines, *terms, *cond)), "invalid line scores")
            z = nll(lines)
            require(close(math.fsum(terms), lines[5]) and all(close(x-z, y) for x,y in zip(lines,cond,strict=True)) and
                    close(r["cost"], nll([cond[c], cond[c+3]])), "inconsistent scoring layouts")
            require(("single" in r) == (c == 2) and (c != 2 or abs(r["single"]-lines[5]) <= CROSS_TOL),
                    "invalid singleton observation")


def components(row):
    prefix, word, stop = row["terms"]
    return dict(prefix=prefix, word=word, stop=stop, line=row["lines"][5], menu=row["conditional"][5])


def contrast(reference, tested, triples):
    """Log-probability gain (ref NLL minus tested NLL); do not confuse a relative gain with suppression."""
    before, after = list(map(components, reference)), list(map(components, tested))
    out = {}
    for k in before[0]:
        changes = [a[k]-b[k] for a,b in zip(before, after, strict=True)]
        blocks = [[changes[i] for i in slots] for slots in triples]
        effects = [idle-(on+off)/2 for on,off,idle in blocks]
        on, off, idle = mean_vec(blocks)
        out[k] = dict(on=on, off=off, idle=idle, entity_gaps=effects, gap=avg(effects),
                      selectivity=avg(effects) > EFFECT and sum(x > MARGIN for x in effects) >= 4,
                      contained=idle > MARGIN and on <= MARGIN and off <= MARGIN)
    return out


def decisions(origin, paths, triples):
    candidate = paths["matched_pair"]
    contrasts = {}
    for key in ("origin", "guard_only", "positive_ce", "rotated_pair"):
        eligible = candidate["complete"] and (key == "origin" or paths[key]["complete"])
        contrasts[key] = contrast(origin if key == "origin" else paths[key]["rows"], candidate["rows"], triples) if eligible else None
    def gate(metric, controls):
        return all(contrasts[k] is not None and contrasts[k][metric]["selectivity"] for k in controls)
    main = ("origin", "guard_only", "positive_ce")
    return dict(contrasts=contrasts, primary=gate("word",main), entity_binding=gate("word",["rotated_pair"]),
                stopped_line=gate("line",main), contained=bool(contrasts["origin"] and contrasts["origin"]["word"]["contained"]))


def audit_vectors(raw, event, rows, p):
    require(len(raw) == GROUPS * VECTOR.size and event["sha"] == sha(raw) and
            event["old_gradient"] == sha(raw[:24*VECTOR.size]), "gradient bytes changed")
    losses = event["loss"]
    require(len(losses) == GROUPS and all(math.isfinite(x) for x in losses), "invalid loss block")
    gs = [dict(v=v, loss=x) for v,x in zip(VECTOR.iter_unpack(raw), losses, strict=True)]
    require(all(math.isfinite(x) for g in gs for x in g["v"]), "nonfinite gradient")
    require(all(close(f32(r["cost"]),g["loss"]) for r,g in zip(rows,gs[:18],strict=True)) and
            all(f32(rows[i]["single"]) == g["loss"] for i,g in zip(p["groups"][2],gs[18:24],strict=True)) and
            all(abs(r["terms"][1]-g["loss"]) <= CROSS_TOL for r,g in zip(rows,gs[24:42],strict=True)), "wrong target losses")
    largest = 0.
    for name, start in (("matched_pair",42), ("rotated_pair",54)):
        for (i,j), r, g in zip(p["pairs"][name],event["reports"][name],gs[start:start+12],strict=True):
            gap = r["other_nll"]-r["preferred_nll"]
            z = math.exp(-abs(gap))
            weight = z/(1+z) if gap >= 0 else 1/(1+z)
            require((r["target_id"],r["margin"],r["vocabulary"]) == (TARGET,0.,64) and
                    abs(r["preferred_nll"]-rows[i]["terms"][1]) <= CROSS_TOL and
                    abs(r["other_nll"]-rows[j]["terms"][1]) <= CROSS_TOL and
                    close(r["log_probability_gap"],gap) and close(weight,r["gradient_weight"]) and
                    close(max(-gap,0.)+math.log1p(z),r["loss"]) and f32(r["loss"]) == g["loss"],
                    "context preference does not match its two full-vocabulary target scores")
            high,_ = check_seed(g["v"],gs[24+i]["v"],gs[24+j]["v"],weight)
            largest = max(largest,high)
    return gs,largest


def proposal_groups(gs, p, arm):
    classes = [[gs[i] for i in ids] for ids in p["groups"]]
    tail = gs[18:24]
    groups = [*classes,tail]
    if arm == "positive_ce":
        groups.append([gs[24+i] for i,_ in p["pairs"]["matched_pair"]])
    elif arm in ("matched_pair","rotated_pair"):
        groups.append(gs[42:54] if arm == "matched_pair" else gs[54:66])
    else:
        require(arm == "guard_only","unknown proposal arm")
    return groups,[*classes,*([g] for g in tail)]


def verify_receipt(r, p, owner, groups, tokens):
    require((r["pre_geometry_id"],r["base_model_identity"],r["lora_artifact_id"],r["readout_accumulator"]) ==
            (owner,p["base"],p["lora"],"f32_wide_backward") and r["trainable_values"] == 64 and
            r["group_sizes"] == list(map(len,groups)) and r["group_target_tokens"] == tokens, "wrong gradient cohort")


def verify_guard(r, groups, delta, applied):
    means = [mean_vec([g["v"] for g in group]) for group in groups]
    norms = [norm(g) for g in means]
    ref = witness(means,delta)
    l, k = norm(delta), norm(applied)
    require(k > 0 and abs(k/l-1) <= 1e-4 and close(l,r["candidate_l2"]) and close(k,r["applied_l2"]) and
            r["min_cosine"] == FLOOR and ref["worst"] == r["worst_candidate_group"] and close(ref["mix"],r["mix"]),
            "wrong guard or actual step length")
    err = norm([x/l-y for x,y in zip(applied,ref["direction"],strict=True)])
    require(err <= 1e-4 and close(err,r["relative_vector_error"]) and
            (ref["interval"] is None) == (r["feasible_mix"] is None), "incorrect guard interval or direction")
    if ref["interval"] is not None:
        require(all(close(x,y) for x,y in zip(ref["interval"],r["feasible_mix"],strict=True)), "incorrect guard endpoints")
    else:
        require(r["candidate_geometry_id"] == r["post_geometry_id"], "pass-through changed parameters")
    def verify_values(field, expected):
        require(all(close(x,y) for x,y in zip(r[field],expected,strict=True)), "wrong guard: "+field)
    verify_values("group_gradient_l2",norms)
    verify_values("group_mean_losses",[avg([g["loss"] for g in group]) for group in groups])
    for label, d in (("candidate",delta),("applied",applied)):
        dots = [dot(g,d) for g in means]
        verify_values(label+"_loss_dots",dots)
        minimum = min(-x/(norm(d)*n) for x,n in zip(dots,norms,strict=True))
        require(close(minimum,r[label+"_min_cosine"]) and (label != "applied" or minimum > FLOOR), "wrong guard slopes")
    return [dot(g,applied) for g in means]


def audit(protocol, pp, root, mp, out):
    p,m = obj(protocol/"plan.json",pp),obj(root/"measured.json",mp)
    inv = frozen_inventory(protocol,p)
    expected_files = sum(n.split("/",1)[0] in PACKAGE_ROOTS for n in inv["files"])
    require(m["installed"] == dict(version=inv["version"],wheel_sha256=p["wheel"],native_sha256=p["native"],
                                    package_files=expected_files,all_original_files_match=True,python_bytecode_absent=True),
            "measurement is not from the admitted installation")
    attempt = json.loads((root/"attempt.json").read_bytes())
    ex = json.loads((root/"exit.json").read_bytes())
    require(m["plan"] == pp and m["native"] == p["native"] and p["code"] == sources() and
            p["arms"] == list(ARMS) and p["states"] == list(STATES) and p["max_budget"] == BUDGET and
            (p["steps"],p["wall"],p["rate"],p["radius"],p["floor"],p["margin"],p["effect"],p["eps"],p["retries"]) ==
            (STEPS,WALL,RATE,RADIUS,FLOOR,0.,EFFECT,MARGIN,0) and attempt == dict(attempt=1,plan=pp,native=p["native"]) and
            ex["exit"] == 0 and 0 < ex["elapsed"] <= WALL and read(root/"process.log",ex["log"]).splitlines()[-1:] == [mp.encode()],
            "not a complete bounded trial")
    data = obj(protocol/"inputs/train.json", p["files"]["train.json"])
    require(pairs(data) == (p["triples"],p["pairs"]) and
            p["groups"] == [[i for i,e in enumerate(data) if e["meaning"] == c] for c in range(3)], "wrong training pairs")
    events = iter(io.BytesIO(read(root/"journal.jsonl",m["journal"],64<<20)))
    binary = io.BytesIO(read(root/"vectors.f32le",m["binary"],GROUPS*VECTOR.size*BUDGET["max_proposals"]))
    def take(tags, *kinds):
        r = json.loads(next(events,b"null"))
        require(isinstance(r,dict) and r["event"] in kinds and all(r[k] == v for k,v in tags.items()), "broken journal chain")
        return r
    def coords(owner):
        return coordinates(geometry_path(root,owner),owner[7:])
    counts = dict.fromkeys(("adopted","score_menus","score_singletons","backwards","proposals","guards"),0)
    summary, ends = {}, {}
    for state in STATES:
        snapshots, initial = {}, None
        for arm in ARMS:
            key = state+"/"+arm
            owner = p["origins"][state]
            a,layout = coords(owner)
            before = take(dict(key=key,owner=owner),"origin")["rows"]
            validate_rows(before,p["groups"])
            if initial is None:
                initial = before
            require(before == initial,"different starting scores")
            counts["score_menus"] += 18
            counts["score_singletons"] += 6
            steps, reason, max_err = [], None, 0.
            for t in range(STEPS):
                tags = dict(key=key,t=t,owner=owner)
                take(tags,"start")
                event = take(tags,"gradients")
                gs,err = audit_vectors(binary.read(GROUPS*VECTOR.size),event,before,p)
                max_err = max(max_err,err)
                ref = p["refs"][state][t]
                if arm == "guard_only" or owner == ref["pre"]:
                    require((owner,event["old_scores"],event["old_gradient"]) == (ref["pre"],ref["before"],ref["gradient"]),
                            "archive mismatch")
                groups, guarded = proposal_groups(gs,p,arm)
                counts["backwards"] += GROUPS
                counts["proposals"] += 1
                event = take(tags,"refusal","candidate")
                if event["event"] == "candidate":
                    r = event["receipt"]
                    tokens = [108]*3+[18]+([] if arm == "guard_only" else [12 if arm == "positive_ce" else 24])
                    verify_receipt(r,p,owner,groups,tokens)
                    candidate = r["post_geometry_id"]
                    b,lb = coords(candidate)
                    delta = subtract(b,a)
                    dm = r["distance_match"]
                    require(lb == layout and r["learning_rate"] == f32(RATE) and abs(norm(delta)/RADIUS-1) <= 1e-4 and
                            close(norm(delta),r["applied_l2"]) and close(norm(delta),dm["realized_l2"]) and
                            (dm["requested_l2"],dm["origin_id"],dm["result_id"]) == (RADIUS,owner,candidate) and
                            dm["relative_vector_error"] <= 1e-4, "unmatched proposal distance")
                    cert = certify(groups,delta,r)
                    counts["guards"] += 1
                    event = take(tags,"refusal","adopt")
                    phase = "guard"
                else:
                    phase = "proposal"
                if event["event"] == "refusal":
                    reason = event["reason"]
                    require(reason["phase"] == phase and reason["message"] in REFUSALS[phase], "unrecognized refusal")
                    break
                g = event["receipt"]
                verify_receipt(g,p,owner,guarded,[108]*3+[3]*6)
                next_id = g["post_geometry_id"]
                z,lz = coords(next_id)
                require(lz == layout and g["candidate_geometry_id"] == candidate, "changed candidate or layout")
                if arm == "guard_only":
                    require((candidate,next_id,fingerprint(r),fingerprint(g)) == (ref["candidate"],ref["chosen"],ref["proposal"],ref["guard"]),
                            "baseline step no longer replays")
                predicted = verify_guard(g,guarded,delta,subtract(z,a))
                after = event["rows"]
                validate_rows(after,p["groups"])
                finite = evaluate(before,after,p["groups"],predicted,MARGIN,.15)
                steps.append(dict(qualified=finite["qualifies"],strict=finite["strict_finite"],harmful=finite["harmful"],
                                  fifth_weight=cert["unit_weights"][4] if len(groups) == 5 else None))
                counts["adopted"] += 1
                counts["score_menus"] += 18
                counts["score_singletons"] += 6
                owner,a,before = next_id,z,after
            ends[key] = dict(owner=owner,adopted=len(steps),refusal=reason)
            snapshots[arm] = dict(complete=len(steps) == STEPS, rows=before, steps=steps, max_signed_vjp_error=max_err,
                                  changes=contrast(initial,before,p["triples"]), **ends[key])
        summary[state] = dict(gates=decisions(initial,snapshots,p["triples"]),arms=snapshots)
    require(next(events,None) is None and not binary.read(1) and ends == m["ends"] and counts == m["counts"] and
            len(ends) == BUDGET["path_count"] and counts["adopted"] <= BUDGET["max_steps"] and
            all(counts[k] <= BUDGET["max_"+k] for k in counts if k != "adopted"), "extra, missing or over-budget work")
    out.mkdir(mode=0o700)
    return put(out/"results.json",dict(plan=pp,measurement=mp,counts=counts,states=summary,
        gates={g:all(v["gates"][g] for v in summary.values()) for g in ("primary","entity_binding","stopped_line","contained")},
        scope="Training-only synthetic decoder. A relative lexical advantage does not establish wrong-context suppression, "
              "transfer to new prompts or improved general generation."))
