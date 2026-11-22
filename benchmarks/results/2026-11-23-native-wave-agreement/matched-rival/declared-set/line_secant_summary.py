"""Reduce raw observations, first-failure journal and independent WASM restore results."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys


def reduce_run(result, journal, cuts, wasm):
    def load(path):
        return json.loads(path.read_bytes())

    primary, bridge, node = load(result), load(cuts)["cases"], load(wasm)
    assert primary["schema"] == "spiraltorch.synthetic_line_secant.v1" and primary["pretrained_calls"] == 0
    assert node["schema"] == "spiraltorch.synthetic_line_secant_wasm.v1" and node["pretrained_calls"] == 0
    rows = defaultdict(list)
    for line in journal.read_bytes().splitlines():
        r = json.loads(line)
        rows[r["case"]].append(r["attempt"])
    names = {a+"/"+b for a in ("raw","largest","successor") for b in ("lora","geometry","joint")}
    assert names == set(primary["cases"]) == set(rows) == set(bridge) == set(node["cases"])
    outcomes = {}
    t = 1/65536
    for name in sorted(names):
        p = primary["cases"][name]
        c, n = bridge[name], node["cases"][name]
        assert p["completed"] == n["completed"] == c["completed"]
        assert p["stop"] == c["reason"] and len(p["attempts"]) == len(rows[name])
        assert n["resumed"] == p["completed"] - c["cut"] and n["refusals"] == int(p["stop"] is not None)
        assert len(rows[name]) == p["completed"] + n["refusals"]
        best = None
        shaped = not name.startswith("raw/")
        for i,(a,r) in enumerate(zip(rows[name],p["attempts"])):
            applied = a["status"] == "applied"
            assert applied == r["applied"] == (i < p["completed"])
            k = a["step"]["anchor_kl"] if applied else a["refusal"]
            assert len(k["trials"]) == r["trials"]
            assert k["anchor_readouts"] == r["readouts"] == 3*(1+int(shaped)+r["trials"])
            assert k["criterion"] == dict(kind="mean_with_cap", max_anchor_kl=2*t)
            assert k["policy"] == dict(target_kl=t,max_parameter_l2=1/8)
            if shaped:
                assert k["direction"]["retained_fraction"] == r["retained_fraction"]
                assert abs(k["direction"]["projected_normal_cosine"]) < 1e-6
            elif "direction" in k:
                raise ValueError("raw gradient unexpectedly shaped")
            if applied:
                assert 0.98*t <= k["mean_kl"] <= t and max(k["per_anchor_kl"]) <= 2*t
            else:
                assert a["next_update"] == i+1 and k["reason"] == p["stop"]
                safe = [q for q in k["trials"] if q["peak"]["kl"] <= 2*t]
                if safe:
                    q = max(safe,key=lambda q:q["mean_kl"])
                    best = dict(mean_over_target=q["mean_kl"]/t,peak_bank_index=q["peak"]["anchor_index"],
                                requested_l2=q["requested_l2"])
        first = p["attempts"][0]
        outcomes[name] = dict(completed=p["completed"],reason=p["stop"],
                              best_cap_safe_at_stop=best,
                              first_probe_peak=first.get("peak_anchor"),first_selected=first.get("selected_anchor"),
                              first_retained_fraction=first.get("retained_fraction"),
                              candidate_trials=sum(r["trials"] for r in p["attempts"]),
                              anchor_readouts=sum(r["readouts"] for r in p["attempts"]),
                              additional_anchor_vjps=len(rows[name]) if shaped else 0)
    sums = {k:sum(r[k] for r in outcomes.values()) for k in ("candidate_trials","anchor_readouts","additional_anchor_vjps")}
    return dict(schema="spiraltorch.synthetic_line_secant_summary.v1",outcomes=outcomes,work=sums,
                wasm_suffix_updates=sum(n["resumed"] for n in node["cases"].values()),
                wasm_refusals=sum(n["refusals"] for n in node["cases"].values()),pretrained_calls=0,
                sha256={name:hashlib.sha256(path.read_bytes()).hexdigest()
                        for name,path in zip(("raw","journal","cuts","wasm"),(result,journal,cuts,wasm))})


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("RESULT JOURNAL CUTS WASM NEW_OUTPUT")
    result = reduce_run(*map(Path,sys.argv[1:5]))
    with Path(sys.argv[5]).open("x") as out:
        json.dump(result,out,sort_keys=True,indent=2,allow_nan=False)
        out.write("\n")
    print(json.dumps(result["work"],sort_keys=True))
