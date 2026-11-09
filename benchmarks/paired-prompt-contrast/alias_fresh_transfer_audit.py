"""Model-free paired transfer: verify readout parity first, then unblind the context key."""
import io
import json
import math

from shared_prefix_study import obj, put, read, require


def avg(values):
    return math.fsum(values)/len(values)


def log_mass_nll(costs):
    low=min(costs)
    return low-math.log(math.fsum(math.exp(low-v) for v in costs))


def check_scores(new, ref, tol):
    domain = dict(stop_id=1, target_counts=[3]*6, prefix_nodes=8, decoder_lanes=6)
    require(all(new[k]==ref[k]==v for k,v in domain.items()) and new["shared_content_tokens"]==1 and
            set(new)==set(domain)|{"shared_content_tokens","winner_index","line_nlls","conditional_nlls","token_nlls","menu_set_nll"},
            "not the label-free six-line readout")
    require(all(new[k]==ref[k] for k in ("line_nlls","winner_index","menu_set_nll")), "old shared-trie readout changed")
    line, cond, token, old = new["line_nlls"],new["conditional_nlls"],new["token_nlls"],ref["token_nlls"]
    require(len(line)==len(cond)==len(token)==len(old)==len(ref["menu_probabilities"])==6 and
            0<=new["winner_index"]<6 and all(len(v)==3 for v in [*token,*old]) and
            all(math.isfinite(x) and x>=0 for x in [*line,*cond,*[x for t in token+old for x in t]]),
            "invalid finite score array")
    errs=[abs(x-y) for t,u in zip(token,old,strict=True) for x,y in zip(t,u,strict=True)]
    direct=[math.fsum(ts) for ts in old]
    line_err=max(abs(x-y) for x,y in zip(line,direct,strict=True))
    require(max(errs)<=tol and line_err<=tol and direct[new["winner_index"]]<=min(direct)+tol and
            len({row[0] for row in token})==1 and
            all(abs(math.fsum(ts)-cost)<=1e-12 for ts,cost in zip(token,line,strict=True)), "independent stopped-line scores disagree")
    mass=log_mass_nll(line)
    require(abs(mass-new["menu_set_nll"])<=1e-11 and
            all(abs(v-(x-mass))<=1e-11 and math.isclose(math.exp(-v),q,rel_tol=1e-12,abs_tol=0.)
                for x,v,q in zip(line,cond,ref["menu_probabilities"],strict=True)), "absolute/conditional partition mismatch")
    return max(errs), line_err


def paired(blocks):
    """For six matched triples of log gains, cancel a class-independent output shift."""
    require(len(blocks)==6 and all(len(row)==3 and all(map(math.isfinite,row)) for row in blocks),
            "need six complete on/off/idle blocks")
    gc=[avg(v) for v in zip(*blocks,strict=True)]
    d=[idle-(on+off)/2 for on,off,idle in blocks]
    return dict(on=gc[0],off=gc[1],correct=gc[2],wrong=(gc[0]+gc[1])/2,selectivity=avg(d),block_differences=d,
                block_class_gains=blocks)


def decisions(absolute, menu, eps, effect):
    same=sum(a>eps and b>eps for a,b in zip(absolute["block_differences"],menu["block_differences"],strict=True))
    return dict(same_blocks_better=same,
                specific=all(v["correct"]>eps and v["selectivity"]>effect for v in (absolute,menu)) and same>=4,
                contained=all(v["correct"]>eps and max(v["on"],v["off"])<=eps for v in (absolute,menu)),
                broad_absolute=absolute["correct"]>effect and absolute["wrong"]>effect and absolute["wrong"]>=.75*absolute["correct"],
                broad_menu=menu["correct"]>effect and menu["wrong"]>effect and menu["wrong"]>=.75*menu["correct"])


def profile(report):
    k=5
    return dict(absolute=report["line_nlls"][k],menu=report["conditional_nlls"][k],
                prefix=report["token_nlls"][k][0],word=report["token_nlls"][k][1],stop=report["token_nlls"][k][2])


def arm_summary(rows, keys):
    by_class=[0,0,0]
    accepted=[[],[],[]]
    gaps, wins, ties = {}, {}, {}
    for label in ("absolute","menu"):
        gaps[label], wins[label], ties[label]=[],0,0
        for triple in keys:
            costs=[profile(rows[k])[label] for k in triple]
            diffs=[v-costs[2] for v in costs[:2]]
            gaps[label].append(avg(diffs))
            wins[label]+=sum(v>0. for v in diffs)
            ties[label]+=sum(v==0. for v in diffs)
    for triple in keys:
        for c,k in enumerate(triple):
            r=rows[k]
            by_class[c]+=r["winner_index"] in (c,c+3)
            accepted[c].append(log_mass_nll([r["conditional_nlls"][c],r["conditional_nlls"][c+3]]))
    return dict(selectivity={k:avg(v) for k,v in gaps.items()},pair_wins_of_12=wins,pair_ties_of_12=ties,
                accepted_top1_of_6=by_class,accepted_conditional_nll=list(map(avg,accepted)))


def contrast(baseline, candidate, keys, eps, effect):
    gains={}; per_block={}
    for label in ("absolute","menu","prefix","word","stop"):
        per_block[label]=[[profile(baseline[k])[label]-profile(candidate[k])[label] for k in triple] for triple in keys]
        gains[label]=paired(per_block[label])
    parts=zip(*(gains[k]["block_differences"] for k in ("prefix","word","stop")),strict=True)
    require(all(abs(math.fsum(terms)-v)<1e-11 for terms,v in zip(parts,gains["absolute"]["block_differences"],strict=True)),
            "components do not reconstruct the absolute selectivity change")
    return dict(gains=gains,wrong_prompts_boosted=sum(v>eps for b in per_block["absolute"] for v in b[:2]),
                correct_prompts_boosted=sum(b[2]>eps for b in per_block["absolute"]),
                gates=decisions(gains["absolute"],gains["menu"],eps,effect))


def audit(protocol, pp, root, mp, out):
    from alias_fresh_transfer import BUDGET, EFFECT, EPS, ORDERS, PHASES, POOLS, STARTS, TOKEN_TOL, WALL, dataset, sources
    p,m=obj(protocol/"plan.json",pp),obj(root/"complete.json",mp)
    att=dict(plan=pp,query_pin=p["query_pin"],phases=list(PHASES),attempt=1)
    require(p["code"]==sources() and m["plan"]==pp and p["budget"]==BUDGET and
            (p["eps"],p["effect"],p["token_tolerance"],p["wall"],p["retries"])==(EPS,EFFECT,TOKEN_TOL,WALL,0) and
            json.loads((root/"attempt.json").read_bytes())==att==json.loads((protocol/"consumed.json").read_bytes()),
            "wrong admission")
    queries=obj(protocol/"inputs/queries.json",p["query_pin"])
    reports, workers, previous_elapsed = {}, [], 0.
    for phase in PHASES:
        pin=m["phases"][phase]
        part,ex=root/phase,json.loads((root/(phase+"-exit.json")).read_bytes())
        meta=obj(part/"complete.json",pin)
        require(ex["exit"]==0 and previous_elapsed<ex["elapsed"]<=WALL and
                read(root/(phase+".log"),ex["log"]).decode().splitlines()[-1:]==[pin] and
                (meta["plan"],meta["phase"],meta["extension"],meta["query_pin"],meta["owners"])==
                (pp,phase,p["runtimes"][phase]["extension"],p["query_pin"],p["owners"]) and
                meta["menus"]==360 and meta["separate_tokens"]==(4680 if phase=="reference" else 0), "incomplete worker")
        workers.append(meta["pid"]); previous_elapsed=ex["elapsed"]
        it=iter(io.BytesIO(read(part/"scores.jsonl",meta["journal"],4<<20)))
        by_owner={}
        for owner in p["owners"]:
            rs={}
            for q in queries:
                tags=dict(owner=owner,id=q["id"])
                begin=json.loads(next(it)); end=json.loads(next(it))
                require(begin==dict(event="start",**tags) and end["event"]=="return" and all(end[k]==v for k,v in tags.items())
                        and q["id"] not in rs, "missing/duplicate/foreign score")
                rs[q["id"]]=end["report"]
            by_owner[owner]=rs
        require(next(it,None) is None,"extra model observations")
        reports[phase]=by_owner
    require(len(set(workers))==2 and len(queries)==72 and len(set(p["owners"]))==5,"invalid comparison shape")
    raw=reports["native"]
    errors=[check_scores(raw[owner][q["id"]],reports["reference"][owner][q["id"]],TOKEN_TOL)
            for owner in p["owners"] for q in queries]
    # No ground-truth file is needed to verify either native readout or their agreement.
    key=obj(protocol/"answer-key.json",p["key_pin"])
    expected_q,expected_k=dataset()
    require(queries==expected_q and key==expected_k,"unblinding key differs from frozen grammar")
    cells={}
    roles=p["roles"]
    for pool in POOLS:
        for order in ORDERS:
            subset=[row for row in key if (row["pool"],row["order"])==(pool,order)]
            lookup={(r["block"],r["meaning"]):r["id"] for r in subset}
            require(len(lookup)==len(subset)==18,"broken matched cell")
            triples=[[lookup[b,c] for c in range(3)] for b in range(6)]
            cell={"role_summaries":{role:arm_summary(raw[owner],triples) for role,owner in roles.items()}}
            for start in STARTS:
                origin,four,guard=(raw[roles[start+"/"+k]] for k in ("origin","four_group","individual_guard"))
                cell[start]=dict(four_minus_origin=contrast(origin,four,triples,EPS,EFFECT),
                                 guard_minus_origin=contrast(origin,guard,triples,EPS,EFFECT),
                                 guard_minus_four=contrast(four,guard,triples,EPS,EFFECT))
            zero=cell["subset_rank"]["guard_minus_four"]
            require(roles["subset_rank/four_group"]==roles["subset_rank/individual_guard"] and
                    all(x==0. for g in zero["gains"].values() for j in g["block_class_gains"] for x in j),
                    "identical-artifact control is not zero")
            cells[pool+"/"+order]=cell
    gates=[c["initial"]["guard_minus_four"]["gates"] for c in cells.values()]
    out.mkdir(mode=0o700)
    return put(out/"results.json",dict(plan=pp,measurement=mp,roles=roles,cells=cells,
        all_cells={k:all(v[k] for v in gates) for k in ("specific","contained","broad_absolute","broad_menu")},
        readout_check=dict(compared_menus=len(errors),distinct_weights=len(raw),reference_token_calls=4680,
                           max_token_nll_error=max(v[0] for v in errors),max_line_nll_error=max(v[1] for v in errors)),
        scope="Prospective symbolic context shifts on six matched blocks, not natural language, independent model seeds, "
              "label-free discovery of meanings, training, generation, or inference about general population effects."))
