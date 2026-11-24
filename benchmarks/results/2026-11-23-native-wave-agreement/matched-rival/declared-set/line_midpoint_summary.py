"""Offline reduction: compare the raw shadow, primary journal and cross-runtime observations."""
import hashlib
import json
from pathlib import Path
import sys

T = 1/65536


def report(a):
    return a["step"]["anchor_kl"] if a["status"] == "applied" else a["refusal"]


def safe_mean(k):
    safe = [t["mean_kl"]/T for t in k["trials"] if t["peak"]["kl"] <= 2*T]
    return max(safe, default=None)


def reduce_run(raw, journal, cuts, wasm):
    read = lambda f: json.loads(f.read_bytes())
    source, node, checkpoints = read(raw), read(wasm), read(cuts)["cases"]
    assert source["schema"] == "spiraltorch.synthetic_line_midpoint.v1"
    assert source["pretrained_calls"] == node["pretrained_calls"] == 0
    rows = {}
    for r in map(json.loads,journal.read_bytes().splitlines()):
        key = (r["case"],r["update"])
        stage = "primary" if "primary" in r else "shadow"
        entry = rows.setdefault(key,{})
        assert stage not in entry
        entry[stage] = r[stage]
    names = {a+"/"+m for a in ("largest","successor") for m in ("lora","geometry","joint")}
    assert set(source["cases"]) == set(node["cases"]) == set(checkpoints) == names
    outcomes = {}
    shadow_work = dict(attempts=0,scalar_candidates=0,anchor_readouts=0,independent_probe_gradients=0)
    for name in sorted(names):
        arm = source["cases"][name]
        n, bridge = node["cases"][name], checkpoints[name]
        assert arm["completed"] == n["completed"] == bridge["completed"]
        assert arm["stop"] == n["reason"] == bridge["reason"]
        assert len(arm["attempts"]) == len(n["events"])
        accepted, corrections, candidates, reads, probes = 0, [], 0, 0, 0
        for i,(r,event) in enumerate(zip(arm["attempts"],n["events"]),1):
            pair = rows.pop((name,i))
            a, ctrl = pair["primary"], pair["shadow"]
            k, raw_k = report(a), report(ctrl)
            shadow_work["attempts"] += 1
            shadow_work["scalar_candidates"] += len(raw_k["trials"])
            shadow_work["anchor_readouts"] += raw_k["anchor_readouts"]
            ok = a["status"]=="applied"
            triggered = ctrl["status"]=="unmatched"
            assert ok == r["applied"] and triggered == (not r["raw"]) == ("raw_miss" in k)
            assert len(raw_k["trials"]) <= 20 and len(k["trials"]) <= 20
            assert r["readouts"] == k["anchor_readouts"]
            if triggered:
                probes += 1
                miss = k["raw_miss"]
                for field in ("trials","reason","anchor_readouts"):
                    assert miss[field] == raw_k[field]
                assert k["anchor_readouts"] == raw_k["anchor_readouts"]+3*(1+len(k["trials"]))
                d = k.get("direction")
                correction = dict(update=i,applied=ok,raw_max_cap_safe_mean_over_target=safe_mean(raw_k),
                                  midpoint_max_cap_safe_mean_over_target=safe_mean(k),raw_trials=len(raw_k["trials"]),
                                  midpoint_trials=len(k["trials"]),anchor_readouts=k["anchor_readouts"])
                if d:
                    shadow_work["independent_probe_gradients"] += 1
                    correction.update(selected_bank_index=d["selected_anchor"],peak_at_probe=d["peak_anchor"],
                                      source_cosine=d["midpoint"]["source_cosine"])
                if ok:
                    assert d["source_loss_dot_step"] < 0
                    correction.update(kl_over_target=[v/T for v in k["per_anchor_kl"]],
                                      accepted_mean_over_target=k["mean_kl"]/T,
                                      scheduled_loss=[r["loss"],r["loss_after"]])
                corrections.append(correction)
                candidates += len(raw_k["trials"])
            else:
                assert ok and k == raw_k and a["step"]["parameter_ray"] == ctrl["step"]["parameter_ray"]
            if ok:
                accepted += 1
                assert 0.98 <= k["mean_kl"]/T <= 1 and max(k["per_anchor_kl"])/T <= 2
            else:
                assert a["next_update"] == i and i==len(arm["attempts"])
            assert event["kind"] == ("refused" if not ok else "midpoint" if triggered else "raw")
            candidates += len(k["trials"])
            reads += k["anchor_readouts"]
        rescued = sum(c["applied"] for c in corrections)
        assert (accepted-rescued,rescued) == (n["raw"],n["rescued"])
        outcomes[name] = dict(raw_updates=accepted-rescued,midpoint_updates=rescued,completed=accepted,
                              stop=arm["stop"],corrections=corrections,final_training_losses=arm["final_scores"],
                              scalar_candidates=candidates,anchor_readouts=reads,anchor_vjps=probes)
    assert not rows
    return dict(schema="spiraltorch.synthetic_line_midpoint_summary.v1", before=source["before"],
                base_journal_sha256=source["previous_journal"],pretrained_calls=0,cases=outcomes,
                shadow_work=shadow_work,
                wasm_native_cut_suffix_updates=node["suffixUpdates"],wasm_native_cut_refusals=node["suffixRefusals"],
                sums={k:sum(r[k] for r in outcomes.values()) for k in
                      ("completed","raw_updates","midpoint_updates","scalar_candidates","anchor_readouts","anchor_vjps")},
                sha256={k:hashlib.sha256(p.read_bytes()).hexdigest()
                        for k,p in zip(("native","journal","cuts","wasm"),(raw,journal,cuts,wasm))})


if __name__=="__main__":
    if len(sys.argv)!=6: sys.exit("RAW JOURNAL CUTS WASM NEW_SUMMARY")
    result=reduce_run(*map(Path,sys.argv[1:5]))
    with Path(sys.argv[5]).open("x") as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print(json.dumps(result["sums"],sort_keys=True))
