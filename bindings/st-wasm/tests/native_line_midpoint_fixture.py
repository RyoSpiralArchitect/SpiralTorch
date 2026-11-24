"""Raw-first angular midpoint on generated training prompts. Rust is the only optimizer."""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import runpy
import sys

prior = runpy.run_path(str(Path(__file__).with_name("native_line_secant_fixture.py")))
common, base, prev, ray, cases = (prior[k] for k in ("common","base","prev","ray","cases"))
Session, T, H = base["Session"], prior["T"], prior["H"]
KIND = "raw_then_secant_midpoint"
JOURNAL = "aa227f6808e8ee6992ef9638bcca3fab54fbf2b17b8ec2865bfad97e36f4c98a"


def dump(p):
    return json.dumps(p, sort_keys=True, separators=(",",":"), allow_nan=False)


def shadow_plan(plan, cursor):
    # The shadow's first loss example changes, never the anchor prompts or their order.
    q = json.loads(plan)
    q["epochs"] = 1
    q["update"].pop("direction")
    examples = q["examples"]
    offset = cursor % len(examples)
    q["examples"] = examples[offset:] + examples[:offset]
    q["update"]["anchor_examples"] = [(j-offset) % len(examples) for j in range(len(examples))]
    assert [q["examples"][j] for j in q["update"]["anchor_examples"]] == cases
    return dump(q)


def entry(a):
    assert a["status"] in ("applied","unmatched")
    return a["step"]["anchor_kl"] if a["status"] == "applied" else a["refusal"]


def run(old_journal, out, cuts):
    rows = old_journal.read_bytes()
    assert hashlib.sha256(rows).hexdigest() == JOURNAL
    old = defaultdict(list)
    for r in map(json.loads,rows.splitlines()):
        old[r["case"]].append(r["attempt"])
    origin,_=common["host"](common["HOST"])
    initial = common["losses"](origin)
    results, bridge = {}, {}
    with out.with_suffix(".jsonl").open("x") as log:
        for selection in ("largest","successor"):
            for mode in base["MODES"]:
                name=selection+"/"+mode
                p=json.loads(prior["plan"](mode,"raw"))
                p["update"]["direction"]=dict(kind=KIND,anchor=selection,probe_l2=H)
                plan=dump(p)
                live=Session(origin,plan)
                metrics, before, saved=[],initial,base["cut"](live,plan)
                on_prior=True
                for i in range(base["TOTAL"]):
                    m=live.model
                    pre=live.checkpoint_with_id()
                    old_z=ray["logits"](m)
                    a=live.attempt_step()
                    log.write(dump(dict(case=name,update=i+1,primary=a))+"\n")
                    log.flush()
                    k=entry(a)
                    applied=a["status"]=="applied"
                    shadow=Session(m,shadow_plan(plan,i))
                    ctrl=shadow.attempt_step()
                    raw=entry(ctrl)
                    log.write(dump(dict(case=name,update=i+1,shadow=ctrl))+"\n")
                    log.flush()
                    is_raw=ctrl["status"]=="applied"
                    base["check_trials"](raw,"capped",is_raw)
                    if on_prior:
                        orig=old["raw/"+mode][i]
                        assert ctrl["status"]==orig["status"] and raw==entry(orig)
                        if is_raw:
                            assert ctrl["step"]["parameter_ray"]==orig["step"]["parameter_ray"]
                    summary=dict(applied=applied,raw=is_raw,raw_trials=len(raw["trials"]),
                                 trials=len(k["trials"]),readouts=k["anchor_readouts"])
                    if is_raw:
                        assert applied and k==raw and a["step"]["parameter_ray"]==ctrl["step"]["parameter_ray"]
                        assert common["owners"](live.model)==common["owners"](shadow.model)
                        assert common["state"](live.model)==common["state"](shadow.model)
                        assert "direction" not in k and "raw_miss" not in k
                    else:
                        miss=k["raw_miss"]
                        assert miss==dict(reason=raw["reason"],trials=raw["trials"],
                                          anchor_readouts=raw["anchor_readouts"],rescue_policy=p["update"]["direction"])
                        summary.update(raw_reason=miss["reason"],raw_readouts=miss["anchor_readouts"])
                        assert k["anchor_readouts"]==raw["anchor_readouts"]+len(cases)*(1+len(k["trials"]))
                        if "direction" in k:
                            info=k["direction"]
                            # Existing single-ray trial validator expects the origin counted once.
                            local=dict(k,anchor_readouts=k["anchor_readouts"]-raw["anchor_readouts"]+len(cases))
                            base["check_trials"](local,"capped",applied,probe_banks=1)
                            grads=prior["audit_probe"](m,old_z,info,i,mode,selection,KIND)
                            mid=info["midpoint"]
                            assert abs(mid["source_cosine"]-mid["projected_cosine"])<1e-6
                            assert abs(mid["source_cosine"]-math.sqrt((1+info["retained_fraction"])/2))<1e-6
                            summary.update(anchor=info["selected_anchor"],probe_peak=info["peak_anchor"],
                                           retained=info["retained_fraction"],source_cosine=mid["source_cosine"])
                        else:
                            assert not applied and not k["trials"] and k["reason"] in (
                                "unresolved secant normal or source gradient",
                                "secant projection retains less than 0.001 of the source norm")
                        on_prior=False
                    metrics.append(summary)
                    if not applied:
                        assert a["next_update"]==i+1 and a["example_index"]==i%len(cases)
                        assert live.checkpoint_with_id()==pre and common["losses"](live.model)==before
                        summary["reason"]=k["reason"]
                        break
                    step=a["step"]
                    assert step["update"]==i+1 and step["example_index"]==i%len(cases)
                    new_z=ray["logits"](live.model)
                    kl=prev["readout"](old_z,new_z)
                    mean=base["check_candidate"](kl,k["trials"][-1])
                    assert 0.98*T<=mean<=T and max(kl)<=2*T
                    assert len(kl)==len(k["per_anchor_kl"]) and all(prev["near"](x,y) for x,y in zip(kl,k["per_anchor_kl"]))
                    x,y=ray["blocks"](m),ray["blocks"](live.model)
                    movement=ray["distance"](x,y)["total"]
                    assert prev["near"](movement,step["parameter_ray"]["realized_l2"])
                    if not is_raw:
                        dot=math.fsum((n-o)*g for (path,nm),block in x.items() if path in grads
                                      for o,n,g in zip(block,y[path,nm],grads[path][nm]))
                        assert dot<0 and prev["near"](dot,info["source_loss_dot_step"])
                        assert abs(-dot/(movement*info["source_gradient_l2"])-mid["source_cosine"])<1e-4
                        assert step["parameter_ray"]["method"]=="selected_parameter_secant_midpoint_ray_f32.v1"
                        assert prev["near"](mid["gradient_l2"],step["parameter_ray"]["gradient_l2"])
                    after=common["losses"](live.model)
                    idx=i%len(cases)
                    assert math.isclose(step["loss"],before[idx],abs_tol=2e-6)
                    summary.update(loss=step["loss"],loss_after=after[idx],mean_kl=mean,peak_kl=max(kl))
                    before=after
                    if i<2: saved=base["cut"](live,plan)
                n=live.completed_updates
                assert n==sum(r["applied"] for r in metrics)
                assert live.exhausted == (n==base["TOTAL"])
                if mode=="lora":
                    assert on_prior and n==6
                results[name]=dict(completed=n, attempts=metrics, stop=metrics[-1].get("reason"),
                                   final_scores=before, owners=common["owners"](live.model))
                bridge[name]=dict(saved, completed=n,reason=metrics[-1].get("reason"), last_rows=ray["logits"](live.model))
    common["save"](out,dict(schema="spiraltorch.synthetic_line_midpoint.v1",cases=results,
                             before=initial,previous_journal=JOURNAL,pretrained_calls=0))
    common["save"](cuts,dict(cases=bridge))
    print(json.dumps({k:[v["completed"],sum(not a["raw"] and a["applied"] for a in v["attempts"])]
                      for k,v in results.items()},sort_keys=True))


def verify_saved(result, cuts, journal):
    source=json.loads(result.read_bytes())
    assert source["schema"]=="spiraltorch.synthetic_line_midpoint.v1"
    bridge=json.loads(cuts.read_bytes())["cases"]
    primary=defaultdict(list)
    for r in map(json.loads,journal.read_bytes().splitlines()):
        if "primary" in r:
            assert r["update"]==len(primary[r["case"]])+1
            primary[r["case"]].append(r["primary"])
    assert set(source["cases"])==set(bridge)==set(primary)
    origin,_=common["host"](common["HOST"])
    suffix_updates=suffix_refusals=prefix_checks=0
    for name,ref in source["cases"].items():
        c=bridge[name]
        # A native cut can be downstream of a successful rescue. Verify that
        # prefix through its last rescued step as well, without rerunning the study.
        rescued=[i for i,a in enumerate(primary[name][:c["cut"]]) if a["status"]=="applied" and "raw_miss" in entry(a)]
        if rescued:
            live=Session(origin,c["plan"])
            for a in primary[name][:rescued[-1]+1]:
                assert live.attempt_step()==a
                prefix_checks+=1
        cold=Session.from_json(origin,c["plan"],c["payload"],c["id"])
        assert cold.completed_updates==c["cut"] and cold.checkpoint_with_id()==(c["payload"],c["id"])
        for a in primary[name][c["cut"]:]:
            pre=cold.checkpoint_with_id()
            assert cold.attempt_step()==a
            if a["status"]=="applied": suffix_updates+=1
            else:
                suffix_refusals+=1
                assert cold.checkpoint_with_id()==pre
        assert cold.completed_updates==c["completed"] and cold.exhausted==(ref["stop"] is None)
        assert list(common["owners"](cold.model))==ref["owners"] and common["losses"](cold.model)==ref["final_scores"]
    print(dump(dict(verified_prefix_steps=prefix_checks,suffix_updates=suffix_updates,suffix_refusals=suffix_refusals)))


if __name__=="__main__":
    if len(sys.argv)==5 and sys.argv[1]=="verify":
        verify_saved(*map(Path,sys.argv[2:]))
    elif len(sys.argv)==4:
        run(*map(Path,sys.argv[1:]))
    else:
        sys.exit("PINNED_PRIOR_JOURNAL NEW_RESULT NEW_CUTS | verify RESULT CUTS JOURNAL")
