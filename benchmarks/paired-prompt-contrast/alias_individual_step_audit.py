"""Model-free finite-step readback against the precomputed nine-objective witnesses."""
import io
import json
import math

from alias_common_trajectory_audit import read_scores, subtract
from alias_interference_audit import avg, close, coordinates, dot, norm
from shared_prefix_study import obj, put, read, require


def nine(rows, groups, menu=False):
    def loss(i):
        r = rows[i]
        if menu:
            return r["lines"][5]
        return r["single"] if "single" in r else r["singletons"][1]["cost"]
    return [avg([rows[i]["cost"] for i in g]) for g in groups] + [loss(i) for i in groups[2]]


def passes(predicted, measured, margin, rel):
    return (predicted < -margin and measured < -margin and
            abs(measured-predicted) <= max(margin,rel*abs(predicted)))


def audit(previous, protocol, pp, measured, mp, out):
    from alias_individual_step import BUDGET, CROSS_TOL, FLOOR, MARGIN, STATES, STEPS, archive, means, pins, weight_path, witness
    pl = obj(protocol/"plan.json", pp)
    m = obj(measured/"measured.json", mp)
    proc = json.loads((measured/"process.json").read_bytes())
    require(json.loads((measured/"attempt.json").read_bytes()) == dict(attempt=1,plan=pp,native=pl["native"]) and
            proc["exit"] == 0 and 0 < proc["seconds"] <= pl["seconds"] + 1 and
            read(measured/"process.log",proc["log"]).decode().splitlines()[-1:] == [mp], "not a complete bounded trial")
    require(pl["code"] == pins() and m["plan"] == pp and m["native"] == pl["native"] and
            m["counts"] == pl["budget"] == BUDGET and (pl["margin"],pl["min_cosine"]) == (MARGIN,FLOOR), "wrong measurement")
    p, paths = archive(previous, pl["old_plan"], pl["old_measured"])
    groups = p["groups"]
    labels = {i:c for c,g in enumerate(groups) for i in g}
    it = iter(io.BytesIO(read(measured/"observations.jsonl", m["journal"], 8 << 20)))
    def take(event, **tags):
        row = json.loads(next(it))
        require(row["event"] == event and all(row[k] == v for k,v in tags.items()), "changed observation order")
        return row
    result = {}
    for state in STATES:
        points = []
        path = paths[state]
        for t, ((owner, old, control), (post, later, _)) in enumerate(zip(path[:-1],path[1:],strict=True)):
            tags = dict(state=state,t=t,owner=owner)
            take("start",**tags)
            step = take("guard",**tags)
            r = step["receipt"]
            a,layout = coordinates(weight_path(previous,state,t),owner[7:])
            b,layout_b = coordinates(weight_path(previous,state,t+1),post[7:])
            z,layout_z = coordinates(measured/f"{state}-{t:02}.json",step["post"])
            require(layout == layout_b == layout_z and (r["pre_geometry_id"],r["candidate_geometry_id"],r["post_geometry_id"]) ==
                    (owner, post, "sha256:"+step["post"]) and post == "sha256:"+control["post"] and
                    (r["base_model_identity"],r["lora_artifact_id"],r["readout_accumulator"]) ==
                    (p["base"],p["lora"],"f32_wide_backward") and r["trainable_values"] == 64 and
                    r["group_sizes"] == [6]*3+[1]*6 and r["group_target_tokens"] == [108]*3+[3]*6 and
                    r["min_cosine"] == FLOOR, "different frozen coordinates or ownership")
            candidate, guarded = subtract(b,a), subtract(z,a)
            length = norm(candidate)
            gs = means(old, groups)
            losses = [avg([old[i]["loss"] for i in ids]) for ids in groups] + [old[i]["singletons"][1]["loss"] for i in groups[2]]
            ref = witness(gs, candidate)
            require(ref == pl["refs"][state][t] and ref["worst"] == r["worst_candidate_group"] and
                    abs(ref["mix"]-r["mix"]) < 1e-12 and
                    (r["feasible_mix"] is None) == (ref["interval"] is None), "guard policy differs from frozen witness")
            if ref["interval"] is None:
                require(step["post"] == control["post"] and r["mix"] == 0., "pass-through changed weights")
            else:
                require(all(abs(x-y)<1e-12 for x,y in zip(ref["interval"],r["feasible_mix"],strict=True)), "wrong feasible interval")
            error = norm([v/length-u for v,u in zip(guarded,ref["direction"],strict=True)])
            require(close(length,r["candidate_l2"]) and close(norm(guarded),r["applied_l2"]) and
                    abs(norm(guarded)/length-1) <= 1e-4 and error <= 1e-4 and
                    abs(error-r["relative_vector_error"]) < 2e-12, "different realized direction or distance")
            arms, rows = {}, {}
            for label, delta, prefix in (("candidate",candidate,"candidate"),("guarded",guarded,"applied")):
                predicted = [dot(g,delta) for g in gs]
                norms = [norm(g) for g in gs]
                min_cos = min(-d/(ng*norm(delta)) for d,ng in zip(predicted,norms,strict=True))
                require(all(close(x,y) for field,values in (("group_mean_losses",losses),("group_gradient_l2",norms),
                                                            (prefix+"_loss_dots",predicted))
                            for x,y in zip(values,r[field],strict=True)) and close(min_cos,r[prefix+"_min_cosine"]), "false slope receipt")
                if label == "guarded":
                    require(min_cos > FLOOR, "rounded direction is not descending")
                scored = [take("score",**tags,arm=label,slot=i) for i in range(18)]
                for i,row in enumerate(scored):
                    read_scores(row,labels[i],labels[i]+3)
                    require(("single" in row) == (labels[i] == 2), "wrong score membership")
                    if "single" in row:
                        require(abs(row["single"]-row["lines"][5]) <= CROSS_TOL, "inconsistent line likelihood")
                    if label == "candidate" or ref["interval"] is None:
                        require(all(row[k] == later[i][k] for k in ("lines","cost","absolute")) and
                                ("single" not in row or row["single"] == later[i]["singletons"][1]["cost"]), "control did not replay")
                finite = subtract(nine(scored,groups),nine(old,groups))
                menu = subtract(nine(scored,groups,True),nine(old,groups,True))
                ok = [passes(d,x,MARGIN,pl["relative_prediction_tolerance"]) and passes(d,y,MARGIN,pl["relative_prediction_tolerance"])
                      for d,x,y in zip(predicted,finite,menu,strict=True)]
                arms[label] = dict(predicted=predicted, finite=finite, menu=menu, pass_per_objective=ok,
                                   minimum_cosine=min_cos, realized_length=norm(delta))
                rows[label] = scored
            terms = {}
            if t in (0,STEPS-1):
                for name in ("before","candidate","guarded"):
                    term = take("tokens",**tags,arm=name,slot=6)["terms"]
                    full = old[6]["lines"][5] if name == "before" else rows[name][6]["single"]
                    require(len(term)==3 and all(math.isfinite(x) and x>=0 for x in term) and abs(math.fsum(term)-full)<=CROSS_TOL,
                            "token decomposition does not reconstruct the standby line")
                    terms[name] = term
            points.append(dict(time=t, unchanged=ref["interval"] is None, direction_error=error, arms=arms, terms=terms))
        result[state] = dict(points=points, unchanged=sum(q["unchanged"] for q in points),
            candidate_all_pass=sum(all(q["arms"]["candidate"]["pass_per_objective"]) for q in points),
            guarded_all_pass=sum(all(q["arms"]["guarded"]["pass_per_objective"]) for q in points),
            largest_guarded_loss_change=max(max(q["arms"]["guarded"]["finite"]) for q in points),
            max_direction_error=max(q["direction_error"] for q in points))
    require(next(it,None) is None,"extra observations")
    out.mkdir(mode=0o700)
    return put(out/"results.json",dict(plan=pp,measurement=mp,states=result,
        primary_pass=all(s["guarded_all_pass"] == STEPS for s in result.values()),
        scope="64 independent single steps from previously observed synthetic TRAINING snapshots; "
              "not a continued guarded path, nine-group maximin optimum, held-out or general-language quality."))
