"""Recompute admission, refusal boundaries and cross-client counts without running a model."""
from hashlib import sha256
import json
import math
from pathlib import Path
import runpy
import sys

HERE=Path(__file__).resolve().parent
TESTS=HERE.parents[4]/"bindings"/"st-wasm"/"tests"
MODES=("lora","geometry","joint")
T=1/65536
C=2*T


def measured(rule, mean, peak):
    if rule=="peak":
        return 0.98<=peak/C<=1
    return 0.98<=mean/T<=1 and (rule=="mean" or peak<=C)


def summarize(raw, node, dom):
    native=json.loads(raw)
    assert (native["schema"],native["host"],native["in_sample"],native["pretrained_calls"])==(
        "spiraltorch.synthetic_kl_envelope.v1","ordinary/wave",True,0)
    assert (native["mean_target"],native["peak_limit"],native["max_parameter_l2"])==(T,C,1/8)
    original=json.loads((HERE/"line_anchor_kl_results.json").read_bytes())
    parser=runpy.run_path(str(TESTS/"native_line_session_capture.py"))["Capture"]()
    parser.feed(dom)
    assert parser.passed
    browser=json.loads("".join(parser.chunks["result"]))
    clients=[node,browser]
    assert [c["line_envelope"]["checkpoint_source"] for c in clients]==["native","wasm"]
    rules=("mean","capped","peak")
    cases=native["cases"]
    assert set(cases)=={r+"/"+m for r in rules for m in MODES}
    counts=dict(accepted=0,refusals=0,resumed=0,resumed_refusals=0,verified_rejected_candidates=0)
    worst_loss=worst_kl=0.
    rows={}
    for name,r in sorted(cases.items()):
        rule,mode=name.split("/")
        steps=r["steps"]
        miss=r["refusal"]
        n=len(steps)
        assert 0<=n<=6 and n==r["completed"] and (n==6)==(miss is None)
        assert r["status"]==("complete" if n==6 else "unmatched")
        assert len(r["scores_at_stop"])==3
        if not n: assert r["scores_at_stop"]==native["before"] and r["parameter_path"]==r["endpoint_l2"]==r["endpoint_kl"]==0
        if rule=="mean":
            old=original["cases"]["kl/"+mode]
            assert r["parameter_path"]==old["parameter_path"] and r["scores_at_stop"]==old["after"]
            assert [s["per_anchor_kl"] for s in steps]==old["per_anchor_kl_by_step"]
        ratios=[]
        for i,s in enumerate(steps):
            kls=s["per_anchor_kl"]
            assert len(kls)==3 and min(kls)>=0 and all(math.isfinite(x) for x in kls)
            mean=math.fsum(kls)/3
            peak=max(kls); which=kls.index(peak)
            assert mean==s["mean_kl"] and measured(rule,mean,peak)
            assert s["anchor_readouts"]==3*(1+len(s["trials"]))
            if rule!="mean":
                p=s["trials"][-1]["peak"]
                assert p["anchor_index"]==which and abs(p["kl"]-peak)<4e-15
            ratios.append([mean/T,peak/T,which])
            for client in clients:
                js=(client["line_anchor_kl"]["cases"][mode] if rule=="mean"
                    else client["line_envelope"]["cases"][name])["steps"][i]
                loss_error,kl_error=abs(js["loss"]-s["loss"]),abs(js["mean_kl"]-mean)
                assert loss_error<=3e-6 and kl_error<=1e-8
                if rule!="mean":
                    worst_loss=max(worst_loss,loss_error)
                    worst_kl=max(worst_kl,kl_error,abs(js["peak_kl"]-peak))
                    assert measured(rule,js["mean_kl"],js["peak_kl"])
        stop=None
        work=sum(s["anchor_readouts"] for s in steps)
        if miss is not None:
            ts=miss["trials"]
            assert rule!="mean" and 1<=len(ts)<=20 and len(ts)==len(miss["verified_per_anchor_kl"])
            assert miss["anchor_readouts"]==3*(1+len(ts))
            work+=miss["anchor_readouts"]
            for t,kls in zip(ts,miss["verified_per_anchor_kl"]):
                assert len(kls)==3 and abs(t["mean_kl"]-math.fsum(kls)/3)<4e-15
                assert abs(t["peak"]["kl"]-max(kls))<4e-15 and t["peak"]["anchor_index"]==kls.index(max(kls))
                assert not measured(rule,math.fsum(kls)/3,max(kls))
            safe=[(t,kls) for t,kls in zip(ts,miss["verified_per_anchor_kl"]) if max(kls)<=C]
            best=max(safe,key=lambda pair: pair[0]["mean_kl"]) if safe else None
            stop=dict(next_update=n+1,example_index=n%3,reason=miss["reason"],tested=len(ts),
                      last_peak_anchor=ts[-1]["peak"]["anchor_index"],
                      highest_observed_mean_under_cap_ratio=best[0]["mean_kl"]/T if best else None,
                      per_anchor_at_that_candidate_over_T=[x/T for x in best[1]] if best else None)
            counts["refusals"]+=1;counts["resumed_refusals"]+=1;counts["verified_rejected_candidates"]+=len(ts)
        if rule!="mean":
            counts["accepted"]+=n;counts["resumed"]+=max(0,n-2)
            for client in clients:
                c=client["line_envelope"]["cases"][name]
                assert (len(c["steps"]),c["reason"],c["anchor_readouts"])==(n,miss and miss["reason"],work)
                assert c["limiting_anchor_at_stop"]==(stop and stop["last_peak_anchor"])
        rows[name]=dict(status=r["status"],completed=n,scores_at_stop=r["scores_at_stop"],stop=stop,
                        parameter_path=r["parameter_path"],endpoint_mean_kl=r["endpoint_kl"],readouts=work,
                        steps_mean_over_T_peak_over_T_and_index=ratios)
    for client in clients:
        assert all(client["line_envelope"][k]==v for k,v in counts.items())
    assert worst_loss<=3e-6 and worst_kl<=1e-8
    # Never rank an untouched or partial endpoint beside six-update endpoints.
    rankings={}
    for rule in rules:
        group={m:rows[rule+"/"+m] for m in MODES}
        rankings[rule]=None if any(r["status"]!="complete" for r in group.values()) else {
            task:sorted(MODES,key=lambda m:group[m]["scores_at_stop"][i])
            for i,task in enumerate(("set","menu","subset"))}
    return dict(schema="spiraltorch.measured_kl_envelope_summary.v1",before=native["before"],
                target_mean=T,cap=C,host=native["host"],pretrained_calls=0,in_sample=True,
                native_raw_sha256=sha256(raw).hexdigest(),cases=rows,
                complete_three_owner_rankings=rankings,wasm_new_policies=dict(counts,
                max_abs_loss_error=worst_loss,max_abs_kl_error=worst_kl))


if __name__=="__main__":
    if len(sys.argv)!=5: sys.exit("NATIVE_RAW NODE_REPORT CHROME_DOM NEW_OUTPUT")
    raw,node,dom,out=map(Path,sys.argv[1:])
    result=summarize(raw.read_bytes(),json.loads(node.read_bytes()),dom.read_text())
    with out.open("x") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
