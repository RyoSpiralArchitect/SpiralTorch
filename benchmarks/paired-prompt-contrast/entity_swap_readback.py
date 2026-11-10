"""Audit training without the test set; only a separately sealed scoring phase unlocks the answer join."""
import io
import json
import math

import entity_swap_task as task
from entity_swap_study import DIRECT_REFUSALS, LIMITS, LOGITS, NGRAD, VECTOR, check_claim, complete, fingerprint, geometry_path, inventory, prev, primary
from alias_guarded_training_audit import evaluate, validate_scores
from alias_interference_audit import avg,close,coordinates,dot,f32,mean_vec,norm
from alias_protection_audit import certify
from byte_release_artifacts import PACKAGE_ROOTS
from context_token_training_audit import verify_guard,verify_receipt
from primary_context_audit import direction
from shared_prefix_study import obj,put,read,require,sha


def minus(a,b):return [x-y for x,y in zip(a,b,strict=True)]


def admitted_model(m,p,inv):
    require(m["installed"]==dict(version=inv["version"],wheel_sha256=p["wheel"],native_sha256=p["native"],
            package_files=sum(k.split("/",1)[0] in PACKAGE_ROOTS for k in inv["files"]),all_original_files_match=True,python_bytecode_absent=True),
            "foreign installation")


def gradients(block,j,batch,ns,p):
    losses=j["losses"];rows=j["before"]
    require(len(block)==VECTOR.size*NGRAD and sha(block)==j["sha"] and len(losses)==NGRAD,"unbound gradients")
    gs=[dict(v=v,loss=y) for v,y in zip(VECTOR.iter_unpack(block),losses,strict=True)]
    require(all(math.isfinite(g["loss"]) and all(map(math.isfinite,g["v"])) for g in gs),"nonfinite geometry handle")
    validate_scores(rows,p["groups"],prev.CROSS_TOL)
    classes=[[gs[i] for i in ids] for ids in p["groups"]];tail=gs[18:24]
    require(all(g["loss"]==f32(r["cost"]) for g,r in zip(gs[:18],rows,strict=True)) and
            all(g["loss"]==f32(rows[i]["single"]) for g,i in zip(tail,p["groups"][2],strict=True)),"wrong old task")
    def ce(i,k):return gs[24+2*i+task.ANSWERS.index(k)]
    targets={"ce":[ce(i,r["gold"]) for i,r in enumerate(batch)]}; largest=0.
    for name,start in (("matched",56),("rotated",72)):
        targets[name]=gs[start:start+16]
        for i,(k,g,r) in enumerate(zip(ns[name],targets[name],j["reports"][name],strict=True)):
            y=batch[i]["gold"];a,b=ce(i,y),ce(k,y);gap=r["other_nll"]-r["preferred_nll"]
            z=math.exp(-abs(gap));weight=z/(1+z) if gap>=0 else 1/(1+z)
            require((r["target_id"],r["vocabulary"],r["margin"])==(y,64,0.) and close(r["log_probability_gap"],gap) and
                    close(weight,r["gradient_weight"]) and close(max(-gap,0.)+math.log1p(z),r["loss"]) and f32(r["loss"])==g["loss"] and
                    abs(a["loss"]-r["preferred_nll"])<=prev.CROSS_TOL and abs(b["loss"]-r["other_nll"])<=prev.CROSS_TOL,"incorrect pair readout")
            high,_=prev.check_seed(g["v"],a["v"],b["v"],weight);largest=max(largest,high)
    return [*classes,tail],[*classes,*([g] for g in tail)],targets,largest


def direct_step(r,p,owner,post,a,end,gs,token_count):
    v=list(map(f32,mean_vec([g["v"] for g in gs])));rate=f32(p["rate"])
    unscaled=[f32(x-f32(rate*y))-x for x,y in zip(a,v,strict=True)]
    magnitude=norm(unscaled);require(magnitude>0,"zero direct SGD")
    scale=p["radius"]/magnitude;delta=minus(end,a);dm=r["distance_match"]
    require([f32(x+y*scale) if y else x for x,y in zip(a,unscaled,strict=True)]==end and
            (r["pre_geometry_id"],r["post_geometry_id"],r["base_model_identity"],r["lora_artifact_id"],r["readout_accumulator"])==
            (owner,post,p["base"],p["lora"],"f32_wide_backward") and (r["members"],r["total_target_tokens"],r["trainable_values"],r["learning_rate"])==
            (16,token_count,64,rate) and close(norm(v),r["mean_gradient_l2"]) and close(avg([g["loss"] for g in gs]),r["mean_input_loss"]) and
            close(norm(delta),r["applied_l2"]) and abs(norm(delta)/p["radius"]-1)<=1e-4 and
            (dm["origin_id"],dm["result_id"],dm["requested_l2"])==(owner,post,p["radius"]) and close(magnitude,dm["direction_l2"]) and
            close(norm(delta),dm["realized_l2"]) and dm["relative_vector_error"]<=1e-4 and
            all(close(dot(g["v"],delta),y) for g,y in zip(gs,r["member_loss_dots"],strict=True)),"not the recorded direct SGD step")


def steered_step(r,p,owner,anchor_id,post,a,z,end,gs,protected,token_count):
    groups=[gs,*protected];verify_receipt(r,p,owner,groups,[token_count]+[108]*3+[3]*6)
    means=[mean_vec([g["v"] for g in group]) for group in groups]
    rounded,aim,old,direct,cap,lim,mix=direction(a,z,means,p["floor"])
    delta,base=minus(end,a),minus(z,a);l,b=norm(delta),norm(base)
    dots=[dot(g,delta) for g in means];old_dots=[dot(g,base) for g in means]
    cs=[-v/(l*norm(g)) for v,g in zip(dots,means,strict=True)]
    err=norm([x/b-y for x,y in zip(delta,aim,strict=True)])
    require(rounded==end and (r["anchor_geometry_id"],r["post_geometry_id"],r["limiting_protected_group"],r["min_cosine"])==
            (anchor_id,post,lim,p["floor"]) and min(cs[1:])>p["floor"] and cs[0]>old[0]+1e-12 and dots[0]<min(0.,old_dots[0]) and
            err<=1e-4 and abs(l/b-1)<=1e-4,"unsafe or non-primary steering endpoint")
    for field,val in (("mix",mix),("mix_limit",cap),("relative_vector_error",err),("anchor_l2",b),("applied_l2",l),("anchor_primary_cosine",old[0]),
                      ("applied_primary_cosine",cs[0]),("anchor_min_protected_cosine",min(old[1:])),("direct_min_protected_cosine",min(direct)),
                      ("applied_min_protected_cosine",min(cs[1:]))):
        require(close(r[field],val),"steering witness differs: "+field)
    for field,val in (("anchor_loss_dots",old_dots),("applied_loss_dots",dots),("group_gradient_l2",list(map(norm,means))),
                      ("group_mean_losses",[avg([g["loss"] for g in group]) for group in groups])):
        require(all(close(x,y) for x,y in zip(r[field],val,strict=True)),"wrong steering group: "+field)


def audit_train(proto,pp,root,tp,out):
    check_claim(proto,"train",pp,root,[])
    p=obj(proto/"plan.json",pp);inv=inventory(proto,p);m=complete(root,tp,"train",pp,[]);admitted_model(m,p,inv)
    data=obj(proto/"fit/task.json",p["fit"]);require(data==task.deck("train"),"wrong training inputs")
    journal=io.BytesIO(read(root/"journal.jsonl",m["journal"],96<<20)); binary=io.BytesIO(read(root/"vectors.f32le",m["binary"],LIMITS["backwards"]*VECTOR.size))
    def take(tags,*kinds):
        x=json.loads(journal.readline() or b"null")
        require(isinstance(x,dict) and x["event"] in kinds and all(x[k]==v for k,v in tags.items()),"invalid event chain")
        return x
    def coords(h):return coordinates(geometry_path(root,h),h[7:])
    def refused(e,phase):
        if e["event"]!="refusal":return None
        rs=e["reason"];allowed=DIRECT_REFUSALS if phase=="direct" else primary.RESOLVED if phase=="steer" else prev.REFUSALS[phase]
        require(rs["phase"]==phase and rs["message"] in allowed,"unrecognized numerical stop")
        return rs
    counts=dict.fromkeys(m["counts"],0);require(set(counts)==set(LIMITS)-{"score_roles","score_rows","score_batches"},"unknown counts")
    roles={};ends={};paths={};calibration={}
    for state in task.STATES:
        roles[state+"/origin"]=p["origins"][state]
        for arm in task.ARMS:
            key=state+"/"+arm;owner=p["origins"][state];a,layout=coords(owner)
            last=None;reason=None;adopted=qualified=harmful=0;err=0.
            for t in range(task.STEPS):
                tags=dict(key=key,t=t,owner=owner);batch=task.batch(data,t);ns=task.negatives(batch)
                require(take(tags,"start")["ids"]==[r["id"] for r in batch],"wrong minibatch")
                j=take(tags,"gradients");block=binary.read(NGRAD*VECTOR.size);before=j["before"]
                groups,protected,targets,hi=gradients(block,j,batch,ns,p);err=max(err,hi);counts["backwards"]+=NGRAD
                if t==0 and arm=="retention":
                    calibration[state]={b["id"]:j["losses"][24+2*i:26+2*i] for i,b in enumerate(batch)}
                require(last is None or last==before,"lost score boundary")
                ref=p["refs"][state][t];same=owner==ref["pre"]
                if same or arm=="retention":
                    require(same and sha(block[:24*VECTOR.size])==ref["gradient"] and fingerprint(before)==ref["before"],"archive mismatch")
                if arm.startswith("direct_"):
                    counts["direct"]+=1;e=take(tags,"adopt","refusal");reason=refused(e,"direct")
                    if reason:break
                else:
                    counts["anchor_proposals"]+=1;e=take(tags,"proposal","refusal");reason=refused(e,"proposal")
                    if reason:break
                    r=e["receipt"];cid=r["post_geometry_id"];b,lb=coords(cid);delta=minus(b,a);size=norm(delta);dm=r["distance_match"]
                    verify_receipt(r,p,owner,groups,[108]*3+[18])
                    require(lb==layout and abs(size/p["radius"]-1)<=1e-4 and close(size,r["applied_l2"]) and r["learning_rate"]==f32(p["rate"]) and
                            (dm["origin_id"],dm["result_id"],dm["requested_l2"])==(owner,cid,p["radius"]) and close(dm["realized_l2"],size) and
                            dm["relative_vector_error"]<=1e-4,"unmatched anchor proposal")
                    certify(groups,delta,r);counts["guards"]+=1;e=take(tags,"guard","refusal");reason=refused(e,"guard")
                    if reason:break
                    g=e["receipt"];aid=g["post_geometry_id"];z,lz=coords(aid)
                    verify_receipt(g,p,owner,protected,[108]*3+[3]*6)
                    require(lz==layout and g["candidate_geometry_id"]==cid,"guard identity")
                    verify_guard(g,protected,delta,minus(z,a))
                    if same:require((cid,aid,fingerprint(r),fingerprint(g))==(ref["candidate"],ref["chosen"],ref["proposal"],ref["guard"]),"old update mismatch")
                    if arm!="retention":counts["steers"]+=1
                    e=take(tags,"adopt",*(["refusal"] if arm!="retention" else []));reason=refused(e,"steer")
                    if reason:break
                post=e["post"];end,le=coords(post);require(le==layout,"frozen coordinate changed")
                if arm=="retention":require(e["receipt"] is None and post==aid,"wrong control")
                else:
                    mode,name=arm.split("_");tokens=16 if name=="ce" else 32
                    if mode=="direct":direct_step(e["receipt"],p,owner,post,a,end,targets[name],tokens)
                    else:steered_step(e["receipt"],p,owner,aid,post,a,z,end,targets[name],protected,tokens)
                after=e["after"];validate_scores(after,p["groups"],prev.CROSS_TOL)
                d=minus(end,a);dots=[dot(mean_vec([v["v"] for v in group]),d) for group in protected]
                f=evaluate(before,after,p["groups"],dots,p["eps"],.15);qualified+=f["qualifies"];harmful+=f["harmful"]
                adopted+=1;counts["updates"]+=1;counts["retention_forwards"]+=24;last=after;owner=post;a=end
            ends[key]=dict(adopted=adopted,owner=owner,refusal=reason)
            roles[key]=owner if adopted==task.STEPS else None
            paths[key]=dict(qualified_old_nine=qualified,harmed_old_objective=harmful,max_signed_vjp_error=err)
    require(not journal.read(1) and not binary.read(1) and counts==m["counts"] and ends==m["ends"] and
            all(v<=LIMITS[k] for k,v in counts.items()),"counts or retained endpoints differ")
    out.mkdir(mode=0o700)
    return put(out/"verified.json",dict(plan=pp,training=tp,roles=roles,ends=ends,paths=paths,counts=counts,calibration=calibration))


def report(proto,pp,trained,tp,audited,ap,scored,sp,out):
    check_claim(proto,"train",pp,trained,[])
    check_claim(proto,"score",pp,scored,[tp,ap])
    p=obj(proto/"plan.json",pp);inv=inventory(proto,p);m=complete(trained,tp,"train",pp,[])
    a=obj(audited/"verified.json",ap);s=complete(scored,sp,"score",pp,[tp,ap])
    require((a["plan"],a["training"],a["ends"])==(pp,tp,m["ends"]) and (s["training"],s["training_audit"],s["roles"])==(tp,ap,a["roles"]),"score lineage")
    admitted_model(s,p,inv);c=obj(scored/"test-consumed.json",s["consumed"]);require((c["plan"],c["training"],c["training_audit"])==(pp,tp,ap),"missing first look")
    qs={k:obj(proto/"blind"/(k+".json"),p["decks"][k]["queries"]) for k in ("train","test")}
    log=iter(io.BytesIO(read(scored/"batches.jsonl",s["log"],2<<20)))
    raw=io.BytesIO(read(scored/"logits.f32le",s["binary"],LIMITS["score_rows"]*LOGITS.size))
    owners=sorted({v for v in a["roles"].values() if v is not None});observed={};calls=0
    require(owners==s["owners"] and len(owners)<=LIMITS["score_roles"],"unfrozen scoring roster")
    for owner in owners:
        observed[owner]={}
        for split,rows in qs.items():
            values={};observed[owner][split]=values
            for i in range(0,64,8):
                part=rows[i:i+8];ids=[x["id"] for x in part];block=raw.read(8*LOGITS.size)
                require(len(block)==8*LOGITS.size and json.loads(next(log))==dict(owner=owner,split=split,first=i,ids=ids,
                        inputs=fingerprint(part),logits=sha(block)),"unbound raw logits")
                values.update(zip(ids,LOGITS.iter_unpack(block),strict=True));calls+=1
    require(next(log,None) is None and not raw.read(1) and s["forwards"]==calls<=LIMITS["score_batches"] and s["rows"]==8*calls,"extra or missing scores")
    error=0.
    for state in task.STATES:
        checks=a["calibration"][state];require(len(checks)==16,"missing origin calibration")
        for row_id,ce in checks.items():
            v=observed[p["origins"][state]]["train"][row_id]; mx=max(v)
            logz=math.log(math.fsum(math.exp(x-mx) for x in v))
            error=max(error,*(abs((mx-v[k])+logz-loss) for k,loss in zip(task.ANSWERS,ce,strict=True)))
    require(error<=prev.CROSS_TOL,"scored full-vocabulary logits do not reconstruct native word CE")
    # No answer file is opened until all roles, inputs, logits and the scoring exit have been verified.
    decks=[task.deck(k) for k in ("train","test")];task.design(*decks)
    result={};gates={}
    for split,deck in zip(("train","test"),decks,strict=True):
        labels=obj(proto/"keys"/(split+".json"),p["decks"][split]["key"])
        require(qs[split]==task.queries(deck) and labels==task.key(deck),"answer/input roster differs")
        scores={owner:task.metrics(labels,v[split]) for owner,v in observed.items()};result[split]={}
        for state in task.STATES:
            arms={name:scores.get(a["roles"][state+"/"+name]) for name in ("origin",*task.ARMS)}
            result[split][state]=dict(arms=arms,decision=task.decisions(arms))
        gates[split]={route:{gate:all(v["decision"][route][gate] for v in result[split].values()) for gate in ("advantage","competence")} for route in ("safe","direct")}
    out.mkdir(mode=0o700)
    return put(out/"results.json",dict(plan=pp,training=tp,verified=ap,scores=sp,states=result,gates=gates,counts=a["counts"],
        scoring=dict(unique_owners=len(owners),calls=calls,rows=8*calls,test_consumed=True,max_calibration_error=error),scope="New combinations of seen synthetic identities; "
        "two initial geometries, not independent seeds. Strict two-choice, full-vocabulary and relative role ranking are distinct outcomes."))
