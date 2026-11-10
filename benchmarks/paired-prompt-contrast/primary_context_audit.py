"""Model-free readback of the actual anchor, primary direction, protected slopes, and finite losses."""
import io
import json
import math

from primary_context_training import ARMS, LIMITS, RESOLVED, STATES, admitted, prior
from alias_common_trajectory_audit import subtract
from alias_guarded_training import VECTOR, fingerprint, geometry_path
from alias_guarded_training_audit import evaluate
from alias_interference_audit import avg, close, coordinates, dot, f32, mean_vec, norm
from alias_protection_audit import certify
from byte_release_artifacts import PACKAGE_ROOTS
from context_token_training_audit import audit_vectors, contrast, decisions, proposal_groups, validate_rows, verify_guard, verify_receipt
from shared_prefix_study import obj, put, read, require


def primary_cost(rows, arm, pairs):
    edges=pairs["matched_pair" if arm=="positive_ce" else arm]
    word=[r["terms"][1] for r in rows]
    if arm=="positive_ce":
        return avg([word[i] for i,_ in edges])
    def softplus(x):
        return max(0.,x)+math.log1p(math.exp(-abs(x)))
    return avg([softplus(word[i]-word[j]) for i,j in edges])


def direction(base, anchor, means, floor):
    delta=subtract(anchor,base); length=norm(delta); norms=[norm(g) for g in means]
    require(length>0 and min(norms)>0 and 0<floor<1 and len(means)>=2,"invalid steering domain")
    u=[x/length for x in delta]; v=[-x/norms[0] for x in means[0]]
    units=[[-x/n for x in g] for g,n in zip(means,norms,strict=True)]
    old=[dot(g,u) for g in units]; direct=[dot(g,v) for g in units[1:]]
    require(min(old[1:])>floor,"infeasible anchor")
    candidates=[((a-floor)/(a-b),i) for i,(a,b) in enumerate(zip(old[1:],direct,strict=True)) if b<=floor]
    cap,lim=min(candidates) if candidates else (1.,None)
    mix=cap/2 if candidates else 1.
    aimed=[(1-mix)*a+mix*b for a,b in zip(u,v,strict=True)]
    n=norm(aimed); aimed=[x/n for x in aimed]
    rounded=[f32(x+length*y) if y else x for x,y in zip(base,aimed,strict=True)]
    return rounded,aimed,old,direct,cap,lim,mix


def check_steer(r,p,owner,anchor_id,groups,arm,base,anchor,end):
    verify_receipt(r,p,owner,groups,[12 if arm=="positive_ce" else 24]+[108]*3+[3]*6)
    means=[mean_vec([x["v"] for x in group]) for group in groups]
    rounded,aimed,old,direct,cap,lim,mix=direction(base,anchor,means,p["floor"])
    dz,da=subtract(end,base),subtract(anchor,base); size,ref=norm(dz),norm(da)
    dots=[dot(g,dz) for g in means]; anchor_dots=[dot(g,da) for g in means]; norms=list(map(norm,means))
    cos=[-x/(size*n) for x,n in zip(dots,norms,strict=True)]
    err=norm([v/ref-y for v,y in zip(dz,aimed,strict=True)])
    require(end==rounded and r["anchor_geometry_id"]==anchor_id and r["min_cosine"]==p["floor"] and
            r["limiting_protected_group"]==lim and min(cos[1:])>p["floor"] and
            cos[0]>old[0]+1e-12 and dots[0]<min(0.,anchor_dots[0]) and err<=1e-4 and abs(size/ref-1)<=1e-4,
            "not the rounded primary-improving protected step")
    for field,expected in (("mix_limit",cap),("mix",mix),("relative_vector_error",err),("anchor_l2",ref),("applied_l2",size),
                           ("direct_min_protected_cosine",min(direct)),("anchor_min_protected_cosine",min(old[1:])),
                           ("applied_min_protected_cosine",min(cos[1:])),("anchor_primary_cosine",old[0]),("applied_primary_cosine",cos[0])):
        require(close(r[field],expected),"incorrect steering "+field)
    for field,expected in (("anchor_loss_dots",anchor_dots),("applied_loss_dots",dots),("group_gradient_l2",norms),
                           ("group_mean_losses",[avg([z["loss"] for z in g]) for g in groups])):
        require(all(close(a,b) for a,b in zip(r[field],expected,strict=True)),"incorrect steering "+field)
    return dots[1:], dict(mix=mix,primary_cosine_gain=cos[0]-old[0],realized_min_protected=min(cos[1:]),primary_loss_dot=dots[0])


def audit(protocol,pp,root,mp,out):
    p,m=obj(protocol/"plan.json",pp),obj(root/"measured.json",mp)
    inv=admitted(protocol,p)
    require(m["plan"]==pp and m["installed"]==dict(version=inv["version"],wheel_sha256=p["wheel"],native_sha256=p["native"],
            package_files=sum(k.split("/",1)[0] in PACKAGE_ROOTS for k in inv["files"]),all_original_files_match=True,python_bytecode_absent=True),
            "foreign training environment")
    attempt=json.loads((root/"attempt.json").read_bytes()); ex=json.loads((root/"exit.json").read_bytes())
    require(attempt==dict(plan=pp,attempt=1,native=p["native"]) and ex["exit"]==0 and 0<ex["elapsed"]<=p["wall"] and
            read(root/"process.log",ex["log"]).splitlines()[-1:]==[mp.encode()],"no complete one-shot trial")
    require(prior.pairs(obj(protocol/"inputs/train.json",p["files"]["train.json"]))==(p["triples"],p["pairs"]),"unbound cohorts")
    journal=io.BytesIO(read(root/"journal.jsonl",m["journal"],64<<20))
    binary=io.BytesIO(read(root/"vectors.f32le",m["binary"],LIMITS["max_proposals"]*prior.GROUPS*VECTOR.size))
    def take(tags,*kinds):
        row=json.loads(journal.readline() or b"null")
        require(isinstance(row,dict) and row["event"] in kinds and all(row[k]==v for k,v in tags.items()),"broken event chain")
        return row
    def weights(owner):
        return coordinates(geometry_path(root,owner),owner[7:])
    def stopped(event,phase):
        if event["event"]!="refusal":return None
        reason=event["reason"]; allowed=RESOLVED if phase=="steer" else prior.REFUSALS[phase]
        require(reason["phase"]==phase and reason["message"] in allowed,"wrong refusal")
        return reason
    counts=dict.fromkeys(("adopted","backwards","proposals","guards","steers","score_menus","score_singletons"),0)
    ends={}; results={}
    for state in STATES:
        initial=None; paths={}
        for arm in ARMS:
            key=state+"/"+arm; owner=p["origins"][state]; a,layout=weights(owner)
            rows=take(dict(key=key,owner=owner),"origin")["rows"];validate_rows(rows,p["groups"])
            if initial is None:initial=rows
            require(initial==rows,"different starting scores")
            counts["score_menus"]+=18;counts["score_singletons"]+=6
            steps=[];reason=None;max_err=0.
            for t in range(p["steps"]):
                tags=dict(key=key,t=t,owner=owner); take(tags,"start")
                evt=take(tags,"gradients")
                gs,err=audit_vectors(binary.read(prior.GROUPS*VECTOR.size),evt,rows,p);max_err=max(err,max_err)
                groups,protected=proposal_groups(gs,p,"guard_only")
                counts["backwards"]+=prior.GROUPS;counts["proposals"]+=1
                ref=p["refs"][state][t]
                replay=arm=="guard_only" or owner==ref["pre"]
                if replay:require((owner,evt["old_scores"],evt["old_gradient"])==(ref["pre"],ref["before"],ref["gradient"]),"unmatched historical gradients")
                evt=take(tags,"proposal","refusal");reason=stopped(evt,"proposal")
                if reason: break
                r=evt["receipt"];cid=r["post_geometry_id"];b,lb=weights(cid); dc=subtract(b,a);l=norm(dc);dm=r["distance_match"]
                verify_receipt(r,p,owner,groups,[108]*3+[18])
                require(lb==layout and abs(l/p["radius"]-1)<=1e-4 and close(l,r["applied_l2"]) and
                        r["learning_rate"]==f32(p["rate"]) and (dm["requested_l2"],dm["origin_id"],dm["result_id"])==(p["radius"],owner,cid)
                        and close(dm["realized_l2"],l) and dm["relative_vector_error"]<=1e-4,"wrong proposal length")
                certify(groups,dc,r);counts["guards"]+=1
                evt=take(tags,"anchor","refusal");reason=stopped(evt,"guard")
                if reason: break
                g=evt["receipt"];aid=g["post_geometry_id"];z,lz=weights(aid)
                verify_receipt(g,p,owner,protected,[108]*3+[3]*6)
                require(lz==layout and g["candidate_geometry_id"]==cid,"foreign guard anchor")
                predicted=verify_guard(g,protected,dc,subtract(z,a))
                if replay:require((cid,aid,fingerprint(r),fingerprint(g))==(ref["candidate"],ref["chosen"],ref["proposal"],ref["guard"]),"old guard does not replay")
                if arm!="guard_only":counts["steers"]+=1
                evt=take(tags,"adopt",*(["refusal"] if arm!="guard_only" else []));reason=stopped(evt,"steer")
                if reason: break
                end=evt["post"];chosen,lc=weights(end);after=evt["rows"];validate_rows(after,p["groups"])
                require(lc==layout,"changed frozen coordinates")
                if arm=="guard_only":
                    require(end==aid and evt["steering"] is None,"guard-only treatment changed")
                    detail={}
                else:
                    primary=proposal_groups(gs,p,arm)[0][-1]
                    sr=evt["steering"];require(sr["post_geometry_id"]==end,"foreign selected state")
                    predicted,detail=check_steer(sr,p,owner,aid,[primary,*protected],arm,a,z,chosen)
                    detail["primary_finite_delta"]=primary_cost(after,arm,p["pairs"])-primary_cost(rows,arm,p["pairs"])
                finite=evaluate(rows,after,p["groups"],predicted,p["eps"],.15)
                steps.append(detail | dict(qualified=finite["qualifies"],strict=finite["strict_finite"],harmful=finite["harmful"]))
                a,owner,rows=chosen,end,after
                counts["adopted"]+=1;counts["score_menus"]+=18;counts["score_singletons"]+=6
            ends[key]=dict(owner=owner,adopted=len(steps),refusal=reason)
            paths[arm]=dict(rows=rows,steps=steps,complete=len(steps)==p["steps"],changes=contrast(initial,rows,p["triples"]),
                            max_signed_vjp_error=max_err,**ends[key])
        results[state]=dict(arms=paths,gates=decisions(initial,paths,p["triples"]))
    require(not journal.read(1) and not binary.read(1) and counts==m["counts"] and ends==m["ends"] and
            len(ends)==LIMITS["path_count"] and counts["adopted"]<=LIMITS["max_steps"] and
            all(counts[k]<=LIMITS["max_"+k] for k in counts if k!="adopted"),"incomplete or over-budget trial")
    gates={k:all(v["gates"][k] for v in results.values()) for k in ("primary","entity_binding","stopped_line","contained")}
    gates["all_paths_nine_finite"]=all(a["complete"] and all(x["qualified"] for x in a["steps"]) for v in results.values() for a in v["arms"].values())
    out.mkdir(mode=0o700)
    return put(out/"results.json",dict(plan=pp,measurement=mp,counts=counts,gates=gates,states=results,
        scope="Continuous training on 18 previously used synthetic training prompts. Direction certificates, measured finite "
              "losses, binding benefit versus rotation and absolute spillover are separate. No transfer or real-model claim."))
