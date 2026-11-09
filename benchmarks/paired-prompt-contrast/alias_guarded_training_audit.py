"""Model-free audit of state chaining, prospective continuation and actual per-step losses."""
import io
import json
import math

from alias_common_trajectory_audit import read_scores, subtract
from alias_individual_step import witness
from alias_individual_step_audit import nine, passes
from alias_interference_audit import avg, close, coordinates, dot, f32, mean_vec, norm
from alias_protection_audit import certify, token_costs
from shared_prefix_study import obj, put, read, require, sha


def evaluate(before, after, groups, predicted, margin, rel):
    single, menu = (subtract(nine(after,groups,x),nine(before,groups,x)) for x in (False,True))
    require(len(single)==len(menu)==len(predicted)==9 and all(math.isfinite(v) for v in (*single,*menu,*predicted)),
            "invalid nine-objective values")
    gates = [passes(d,a,margin,rel) and passes(d,b,margin,rel) for d,a,b in zip(predicted,single,menu,strict=True)]
    return dict(finite=single, menu=menu, predicted=predicted, gates=gates, qualifies=all(gates),
                strict_finite=all(max(a,b)<-margin for a,b in zip(single,menu,strict=True)),
                harmful=any(max(a,b)>margin for a,b in zip(single,menu,strict=True)))


def validate_scores(rows, groups, tol):
    require([r["slot"] for r in rows]==list(range(18)), "incorrect scored cohort")
    for c,group in enumerate(groups):
        for i in group:
            row = rows[i]
            read_scores(row,c,c+3)
            require(("single" in row)==(c==2), "incorrect singleton membership")
            if c==2:
                require(abs(row["single"]-row["lines"][5])<=tol, "singleton and full-menu readouts differ")


def streaks(flags):
    current = longest = 0
    for v in flags:
        current = current + 1 if v else 0
        longest = max(longest,current)
    return dict(longest=longest,trailing=current)


def summary(steps, checkpoints, margin):
    accepted = [s["selected"] for s in steps]
    repaired = [s["repaired"] for s in steps]
    measures = [a["qualifies"] for a in accepted]
    start, end = checkpoints["0"], checkpoints[str(len(steps))]
    counts = dict(total=len(steps), qualified=sum(measures), finite_descents=sum(a["strict_finite"] for a in accepted),
                  harmful=sum(a["harmful"] for a in accepted), first_miss=next((i for i,v in enumerate(measures,1) if not v),None),
                  first_harm=next((i for i,a in enumerate(accepted,1) if a["harmful"]),None),
                  repaired=repaired.count(True), archive_replays=sum(s["replayed"] for s in steps),
                  strict_prefix32=all(measures[:32]), all_qualified=all(measures))
    if all(r is not None for r in repaired):
        counts.update(pass_through=repaired.count(False), safe_streak=streaks([not r and ok for r,ok in zip(repaired,measures,strict=True)]),
                      rescued=sum(not s["candidate"]["qualifies"] and s["selected"]["qualifies"] for s in steps),
                      prevented_harm=sum(s["candidate"]["harmful"] and not s["selected"]["harmful"] for s in steps),
                      first_pass_through=next((s["step"] for s in steps if not s["repaired"]),None),
                      last_repair=next((s["step"] for s in reversed(steps) if s["repaired"]),None),
                      switches=sum(a!=b for a,b in zip(repaired[:-1],repaired[1:],strict=True)))
        counts["last16_without_repair_and_qualified"] = counts["safe_streak"]["trailing"] >= 16
    diff = subtract(end["nine"],start["nine"])
    return dict(counts=counts, endpoint_delta=diff, standby_prompts_better=sum(d < -margin for d in diff[3:]),
                idle_accepted_delta=end["absolute_idle"]-start["absolute_idle"],
                other_idle_alias_delta=end["other_idle_line"]-start["other_idle_line"],
                token_component_delta=mean_vec([subtract(a,b) for a,b in zip(end["terms"],start["terms"],strict=True)]))


def audit(protocol, pp, root, mp, out):
    from alias_guarded_training import (ARMS, BLOCK, BUDGET, CROSS_TOL, FLOOR, HORIZON, MARGIN, MILESTONES,
                                        RATE, RADIUS, SPLIT, STATES, VECTOR, WALL, boundary, check_replay,
                                        fingerprint, geometry_path, sources)
    p, m = obj(protocol/"plan.json",pp), obj(root/"complete.json",mp)
    require(m["plan"]==pp and p["code"]==sources() and m["native"]==p["native"] and len(m["parts"])==2 and
            p["arms"]==list(ARMS) and p["states"]==list(STATES) and (p["horizon"],p["split"])==(HORIZON,SPLIT) and
            p["margin"]==MARGIN and p["floor"]==FLOOR and p["budget"]==BUDGET, "mismatched plan")
    require(json.loads((root/"attempt.json").read_bytes())==dict(plan=pp,native=p["native"],attempt=1), "not the recorded attempt")
    groups = p["groups"]
    require(list(map(len,groups))==[6]*3 and sorted(i for g in groups for i in g)==list(range(18)), "not the training cohort")
    seen = {ref["pre"] for refs in p["refs"].values() for ref in refs}
    history = {s+"/"+a:dict(steps=[],checkpoints={}) for s in STATES for a in ARMS}
    state_table, counts, prior = {}, dict.fromkeys(BUDGET,0), None
    for stage,pin in enumerate(m["parts"]):
        folder = root/f"part{stage}"
        meta = obj(folder/"complete.json",pin)
        ex = json.loads((root/f"exit{stage}.json").read_bytes())
        predecessor = m["parts"][0] if stage else "-"
        require(meta["previous"]==ex["previous"]==predecessor and meta["plan"]==pp and meta["stage"]==stage and
                ex["exit"]==0 and 0<ex["elapsed"]<=WALL and read(root/f"worker{stage}.log",ex["log"]).splitlines()[-1:]==[pin.encode()] and
                (prior is None or meta["pid"]!=prior["pid"]), "unverified process boundary")
        require(set(meta["files"])=={"steps.jsonl","gradients.f32le"} and
                set(meta["endpoints"])==set(history), "incomplete stage")
        records = iter(io.BytesIO(read(folder/"steps.jsonl",meta["files"]["steps.jsonl"],32<<20)))
        binary = io.BytesIO(read(folder/"gradients.f32le",meta["files"]["gradients.f32le"],4<<20))
        def take(event, tags):
            r = json.loads(next(records))
            require(r["event"]==event and all(r[k]==x for k,x in tags.items()), "journal chain broken")
            return r
        stage_counts = dict.fromkeys(BUDGET,0)
        for s in STATES:
            for arm in ARMS:
                key = s+"/"+arm
                hist = history[key]
                last = prior["endpoints"][key] if prior else None
                if last:
                    boundary(last,pp,0,s,arm)
                owner = last["owner"] if last else p["origins"][s]
                previous_scores = last["scores"] if last else None
                a,layout = coordinates(geometry_path(root,owner),owner[7:])
                for t in range(stage*SPLIT,(stage+1)*SPLIT):
                    tags = dict(state=s,arm=arm,t=t,owner=owner)
                    take("start",tags)
                    r = take("adopt",tags)
                    proof, proposal, guard = r["proof"],r["proposal"],r["guard"]
                    before, cs = r["before"],r["candidate_scores"]
                    guarded = arm=="individual_guard"
                    zs = r["guard_scores"] if guarded else cs
                    require(guarded==(guard is not None)==(r["guard_scores"] is not None), "wrong policy")
                    for rows in (before,cs,zs):
                        validate_scores(rows,groups,CROSS_TOL)
                    raw = binary.read(BLOCK)
                    require(len(raw)==BLOCK and sha(raw)==proof["gradient"], "unbound gradients")
                    vs = list(VECTOR.iter_unpack(raw))
                    require(len(vs)==24 and all(math.isfinite(x) for v in vs for x in v), "invalid f32 gradient block")
                    require(proof["pre"]==owner==proposal["pre_geometry_id"] and proof["before"]==fingerprint(before) and
                            (previous_scores is None or previous_scores==proof["before"]) and proof["proposal"]==fingerprint(proposal) and
                            proof["candidate"]==proposal["post_geometry_id"] and proof["guard"]==(fingerprint(guard) if guarded else None),
                            "update does not follow last selected state")
                    if t==SPLIT:
                        require(proof==last["shadow"], "cold continuation lost the live next-step witness")
                    candidate = proof["candidate"]
                    b,l = coordinates(geometry_path(root,candidate),candidate[7:])
                    delta = subtract(b,a)
                    length = norm(delta)
                    require(l==layout and abs(length/RADIUS-1)<=1e-4 and close(length,proposal["applied_l2"]) and
                            (proposal["base_model_identity"],proposal["lora_artifact_id"],proposal["readout_accumulator"]) ==
                            (p["base"],p["lora"],"f32_wide_backward") and proposal["group_sizes"]==[6]*4 and
                            proposal["group_target_tokens"]==[108]*3+[18] and proposal["trainable_values"]==64 and
                            proposal["learning_rate"]==f32(RATE), "incompatible proposal")
                    dm=proposal["distance_match"]
                    require((dm["origin_id"],dm["result_id"],dm["requested_l2"])==(owner,candidate,RADIUS) and
                            close(dm["realized_l2"],length) and dm["relative_vector_error"]<=1e-4, "inconsistent path length")
                    classes = [[dict(v=vs[i],loss=f32(before[i]["cost"])) for i in ids] for ids in groups]
                    tail = [dict(v=v,loss=f32(before[i]["single"])) for i,v in zip(groups[2],vs[18:],strict=True)]
                    certify([*classes,tail],delta,proposal)
                    means = [mean_vec([r["v"] for r in g]) for g in classes]+[r["v"] for r in tail]
                    predictions = [dot(g,delta) for g in means]
                    cand = evaluate(before,cs,groups,predictions,MARGIN,p["prediction_rel"])
                    chosen_id=proof["chosen"]
                    selected=cand
                    z=b
                    repair=None
                    if guarded:
                        z,l = coordinates(geometry_path(root,chosen_id),chosen_id[7:])
                        dz = subtract(z,a)
                        effective = norm(dz)
                        ref = witness(means,delta)
                        norms = [norm(g) for g in means]
                        applied = [dot(g,dz) for g in means]
                        slopes = [min(-d/(n*k) for d,n in zip(ds,norms,strict=True)) for ds,k in ((predictions,length),(applied,effective))]
                        err = norm([v/length - u for v,u in zip(dz,ref["direction"],strict=True)])
                        require(l==layout and (guard["pre_geometry_id"],guard["candidate_geometry_id"],guard["post_geometry_id"]) ==
                                (owner,candidate,chosen_id) and guard["group_sizes"]==[6]*3+[1]*6 and
                                guard["group_target_tokens"]==[108]*3+[3]*6 and guard["trainable_values"]==64 and
                                (guard["base_model_identity"],guard["lora_artifact_id"],guard["readout_accumulator"]) ==
                                (p["base"],p["lora"],"f32_wide_backward") and guard["min_cosine"]==FLOOR and
                                close(length,guard["candidate_l2"]) and close(effective,guard["applied_l2"]) and
                                abs(effective/length-1)<=1e-4 and err<=1e-4 and close(err,guard["relative_vector_error"]) and
                                ref["worst"]==guard["worst_candidate_group"] and close(ref["mix"],guard["mix"]) and
                                close(slopes[0],guard["candidate_min_cosine"]) and close(slopes[1],guard["applied_min_cosine"]) and
                                slopes[1]>FLOOR, "wrong guard direction, rounding or owner")
                        for name,expected in (("candidate_loss_dots",predictions),("applied_loss_dots",applied),
                                              ("group_gradient_l2",norms),("group_mean_losses",
                                               [avg([r["loss"] for r in g]) for g in classes]+[r["loss"] for r in tail])):
                            require(all(close(x,y) for x,y in zip(guard[name],expected,strict=True)), "false guard evidence: "+name)
                        repair = ref["interval"] is not None
                        require(repair==(guard["feasible_mix"] is not None), "wrong pass-through decision")
                        if repair:
                            require(all(close(x,y) for x,y in zip(ref["interval"],guard["feasible_mix"],strict=True)), "wrong interval")
                        else:
                            require(candidate==chosen_id and cs==zs, "pass-through changed endpoint or scores")
                        selected=evaluate(before,zs,groups,applied,MARGIN,p["prediction_rel"])
                    else:
                        require(chosen_id==candidate,"control adopted a guarded model")
                    cs_pin, zs_pin = fingerprint(cs),fingerprint(zs)
                    same = check_replay(p,s,arm,t,proof,cs_pin,zs_pin)
                    require(r["archived_replay"]==same, "archive replay status inaccurate")
                    expected_times = ([0] if t==0 else []) + ([t+1] if t+1 in MILESTONES else [])
                    require(set(r["terms"])=={str(k) for k in expected_times}, "unexpected token scoring")
                    for k in expected_times:
                        rows = before if k==0 else zs
                        terms = r["terms"][str(k)]
                        require(len(terms)==len(groups[2]), "missing term score")
                        for i,v in zip(groups[2],terms,strict=True):
                            token_costs(dict(rows[i],terms=v),CROSS_TOL)
                        hist["checkpoints"][str(k)] = dict(nine=nine(rows,groups),terms=terms,owner=owner if k==0 else chosen_id,
                            absolute_idle=avg([rows[i]["absolute"] for i in groups[2]]),other_idle_line=avg([rows[i]["lines"][2] for i in groups[2]]))
                    hist["steps"].append(dict(step=t+1,owner=owner,candidate_owner=candidate,selected_owner=chosen_id,
                        candidate=cand,selected=selected,repaired=repair,replayed=same,previously_observed_owner=owner in seen,
                        worst_group=guard["worst_candidate_group"] if guarded else None, mix=guard["mix"] if guarded else None))
                    stage_counts["backwards"]+=24
                    stage_counts["proposals"]+=1
                    stage_counts["guards"]+=int(guarded)
                    stage_counts["adopted"]+=1
                    stage_counts["forwards"]+=48 if guarded else 24
                    stage_counts["token_scores"]+=30*len(expected_times)
                    owner,a,previous_scores=chosen_id,z,zs_pin
                marker=meta["endpoints"][key]
                require((marker["state"],marker["arm"],marker["plan"],marker["stage"],marker["cursor"],marker["owner"],marker["scores"]) ==
                        (s,arm,pp,stage,(stage+1)*SPLIT,owner,previous_scores), "wrong saved endpoint")
                if stage==0:
                    boundary(marker,pp,stage,s,arm)
                    stage_counts["uncommitted_probes"]+=1
                    stage_counts["backwards"]+=24
                    stage_counts["proposals"]+=1
                    stage_counts["guards"]+=int(arm=="individual_guard")
                else:
                    require("shadow" not in marker, "unexpected additional probe")
        require(next(records,None) is None and not binary.read(1) and meta["counts"]==stage_counts, "incomplete/extra work")
        for k,n in stage_counts.items():
            counts[k]+=n
        prior=meta
    require(counts==BUDGET, "full work budget mismatch")
    for s in STATES:
        arms = {a:summary(history[s+"/"+a]["steps"],history[s+"/"+a]["checkpoints"],MARGIN) for a in ARMS}
        require(arms["four_group"]["counts"]["archive_replays"]>=32, "lost historical control")
        state_table[s]=arms
    out.mkdir(mode=0o700)
    return put(out/"results.json",dict(plan=pp,measurement=mp,counts=counts,states=state_table,history=history,
        all_guarded_qualified=all(result["individual_guard"]["counts"]["all_qualified"] for result in state_table.values()),
        bridges_verified=4,scope="Four chained geometry-only synthetic training paths and four non-adopted continuation witnesses. "
                                "No held-out scoring, pretrained weights or generated language."))
