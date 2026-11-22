"""Mean, capped mean and peak KL on the same tiny synthetic language tasks."""
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
previous = runpy.run_path(str(HERE/"native_line_kl_fixture.py"))
ray, common, Session, MODES, CASES = (previous[k] for k in ("ray","common","Session","MODES","CASES"))
T, C, DISTANCE, TOTAL = 1/65536, 2/65536, 1/8, 6
RULES = ("mean", "capped", "peak")
OLD_RAW = "9a60f7c3592e1844dd15d5a1f71f2e2119794c635c5ab13c7561543fabd07aa9"
OLD_CUTS = "3c43174babb99fcd6e2c84b439b4e144af0cf518252770e9bb8d3b9b405bb4e8"


def pinned(path, digest):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    return json.loads(raw)


def make_plan(mode, rule):
    base = previous["plan"](mode,"kl")
    if rule == "mean":
        return base
    p = json.loads(base)
    p["update"]["criterion"] = (dict(kind="mean_with_cap", max_anchor_kl=C) if rule=="capped"
                                else dict(kind="peak"))
    p["update"]["target_kl"] = C if rule=="peak" else T
    return json.dumps(p,sort_keys=True,separators=(",",":"))


def position(rule, mean, peak):
    target, metric = (C,peak) if rule=="peak" else (T,mean)
    if metric>target or (rule=="capped" and peak>C):
        return "above"
    return "matched" if metric/target >= 0.98 else "below"


def check_trials(report, rule, applied):
    assert report["policy"]==dict(target_kl=C if rule=="peak" else T,max_parameter_l2=DISTANCE)
    if rule=="mean":
        assert "criterion" not in report
    else:
        expected = dict(kind="mean_with_cap",max_anchor_kl=C) if rule=="capped" else dict(kind="peak")
        assert report["criterion"]==expected
    trials=report["trials"]
    assert 1<=len(trials)<=20 and report["anchor_readouts"]==len(CASES)*(1+len(trials))
    lo, hi = 0., DISTANCE
    for i,t in enumerate(trials):
        d, mean = t["requested_l2"], t["mean_kl"]
        assert math.isfinite(mean) and mean>=0 and d==(hi if i==0 else lo+(hi-lo)*0.5)
        if rule=="mean":
            assert "peak" not in t
            peak=mean
        else:
            peak=t["peak"]["kl"]
            assert 0<=t["peak"]["anchor_index"]<len(CASES) and peak+1e-15>=mean
        side=position(rule,mean,peak)
        assert (side=="matched") == (applied and i==len(trials)-1)
        if side=="above": hi=d
        if side=="below": lo=d
    if not applied:
        assert (len(trials)==1 and report["reason"]=="ceiling below matching interval" or
                len(trials)==20 and report["reason"]=="bounded bracket exhausted")


def check_candidate(values, trial):
    mean=math.fsum(values)/len(values)
    assert previous["near"](trial["mean_kl"],mean)
    if "peak" in trial:
        index=max(range(len(values)),key=values.__getitem__)
        assert trial["peak"]["anchor_index"]==index and previous["near"](trial["peak"]["kl"],values[index])
    return mean


def cut(session, plan):
    text, pin=session.checkpoint_with_id()
    assert common["pin"](text)==pin
    return dict(plan=plan,cut=session.completed_updates,payload=text,id=pin)


def run(old_dir, out, cuts):
    raw=pinned(old_dir/"raw.json",OLD_RAW)
    old_cuts=pinned(old_dir/"bridge.json",OLD_CUTS)["cases"]
    origin,_=common["host"](common["HOST"])
    z0,x0=ray["logits"](origin),ray["blocks"](origin)
    before=common["losses"](origin)
    assert raw["before"]==before
    cases,bridge={},{}
    with out.with_suffix(".jsonl").open("x") as log:
        def journal(entry):
            log.write(json.dumps(entry,sort_keys=True,allow_nan=False)+"\n")
            log.flush()
        for rule in RULES:
            for mode in MODES:
                name=rule+"/"+mode
                plan=make_plan(mode,rule)
                s=Session(origin,plan)
                z,x=z0,x0
                steps,refusal,saved=[],None,cut(s,plan)
                for i in range(TOTAL):
                    current=s.model
                    pre=s.checkpoint_with_id()
                    attempt=s.attempt_step()
                    if attempt["status"]=="unmatched":
                        assert (s.completed_updates,attempt["next_update"],attempt["example_index"])==(i,i+1,i%len(CASES))
                        assert attempt["plan_id"]==s.plan_id and s.checkpoint_with_id()==pre
                        refusal=attempt["refusal"]
                        check_trials(refusal,rule,False)
                        journal(dict(case=name,next_update=i+1,refusal=refusal))
                        # A frozen-state observation of rejected candidates, never an optimizer retry.
                        g=common["grad"](current,CASES[i%len(CASES)],mode)
                        observations=[]
                        for t in refusal["trials"]:
                            probe,_=current.step_adaptation_gradient_at_distance(g,mode=mode,target_l2=t["requested_l2"])
                            values=previous["readout"](z,ray["logits"](probe))
                            check_candidate(values,t)
                            assert position(rule,math.fsum(values)/len(values),max(values))!="matched"
                            observations.append(values)
                        refusal=dict(refusal,verified_per_anchor_kl=observations)
                        assert common["owners"](current)==common["owners"](s.model)
                        break
                    assert attempt["status"]=="applied"
                    r=attempt["step"]
                    assert r["update"]==i+1 and r["mode"]==mode
                    k=r["anchor_kl"]
                    check_trials(k,rule,True)
                    nz,nx=ray["logits"](s.model),ray["blocks"](s.model)
                    kl=previous["readout"](z,nz)
                    mean=check_candidate(kl,k["trials"][-1])
                    assert len(k["per_anchor_kl"])==len(kl) and all(previous["near"](a,b) for a,b in zip(kl,k["per_anchor_kl"]))
                    assert position(rule,mean,max(kl))=="matched"
                    dx=ray["distance"](x,nx)["total"]
                    assert previous["near"](r["parameter_ray"]["realized_l2"],dx)
                    assert r["parameter_ray"]["requested_l2"]==k["trials"][-1]["requested_l2"]
                    step=dict(loss=r["loss"],parameter_l2=dx,mean_kl=mean,per_anchor_kl=kl,
                              trials=k["trials"],anchor_readouts=k["anchor_readouts"])
                    if rule=="mean":
                        assert step==raw["cases"]["kl/"+mode]["steps"][i]
                    steps.append(step)
                    journal(dict(case=name,update=i+1,step=step))
                    if i+1<=2: saved=cut(s,plan)
                    z,x=nz,nx
                assert s.completed_updates==len(steps)
                if refusal is None: assert s.exhausted and s.attempt_step() is None
                after=common["losses"](s.model)
                end=previous["readout"](z0,z)
                pin=s.checkpoint_with_id()[1]
                cases[name]=dict(steps=steps,completed=len(steps),status="complete" if refusal is None else "unmatched",
                    refusal=refusal,scores_at_stop=after,parameter_path=math.fsum(q["parameter_l2"] for q in steps),
                    endpoint_l2=ray["distance"](x0,x)["total"],endpoint_kl=math.fsum(end)/len(end))
                if rule=="mean":
                    assert refusal is None and after==raw["cases"]["kl/"+mode]["after"] and pin==old_cuts[mode]["final_pin"]
                else:
                    bridge[name]=dict(saved,completed=len(steps),status=cases[name]["status"],steps=[q["loss"] for q in steps],
                                      final_pin=pin,endpoint=common["state"](s.model),refusal=refusal and refusal["reason"])
    common["save"](out,dict(schema="spiraltorch.synthetic_kl_envelope.v1",host=common["HOST"],before=before,
        mean_target=T,peak_limit=C,max_parameter_l2=DISTANCE,in_sample=True,pretrained_calls=0,cases=cases))
    common["save"](cuts,dict(cases=bridge))
    print(json.dumps({k:[v["completed"],v["status"],v["scores_at_stop"]] for k,v in cases.items()},sort_keys=True))


def resume(cuts):
    source,_=common["host"](common["HOST"])
    updates=refusals=0
    for name,c in json.loads(cuts.read_bytes())["cases"].items():
        s=Session.from_json(source,c["plan"],c["payload"],c["id"])
        assert s.completed_updates==c["cut"] and s.checkpoint_with_id()==(c["payload"],c["id"])
        rule,mode=name.split("/")
        for i in range(c["cut"],c["completed"]):
            a=s.attempt_step()
            assert a["status"]=="applied" and a["step"]["mode"]==mode and a["step"]["loss"]==c["steps"][i]
            check_trials(a["step"]["anchor_kl"],rule,True)
            updates+=1
        if c["refusal"]:
            a=s.attempt_step()
            assert a["status"]=="unmatched" and a["refusal"]["reason"]==c["refusal"]
            check_trials(a["refusal"],rule,False)
            refusals+=1
        else:
            assert s.exhausted and s.attempt_step() is None
        assert common["state"](s.model)==c["endpoint"] and s.checkpoint_with_id()[1]==c["final_pin"]
    print(json.dumps(dict(suffix_updates=updates,verified_refusals=refusals,pretrained_calls=0)))


if __name__=="__main__":
    if len(sys.argv)==5 and sys.argv[1]=="run": run(*map(Path,sys.argv[2:]))
    elif len(sys.argv)==3 and sys.argv[1]=="resume": resume(Path(sys.argv[2]))
    else: sys.exit("run PRIOR_KL_DIRECTORY NEW_RESULT NEW_CUTS | resume CUTS")
