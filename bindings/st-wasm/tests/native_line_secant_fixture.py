"""Frozen synthetic direction controls; the Rust session alone chooses/applies updates."""
import json
import math
from pathlib import Path
import runpy
import sys

base = runpy.run_path(str(Path(__file__).with_name("native_line_envelope_fixture.py")))
common, ray, prev, cases = (base[k] for k in ("common", "ray", "previous", "CASES"))
H, T, C = 1/256, base["T"], base["C"]
RAW = "26e9ebe0491d01ea64d4921deaf1fe58fa0c6ed5bff71cd77269dee9739247f4"
CUTS = "3d97d0e887c256f662ae9bfa45876a5d1f08c948ab75b307cf1c6abd98b6e115"


def plan(mode, selection):
    text = base["make_plan"](mode, "capped")
    if selection == "raw":
        return text
    p = json.loads(text)
    p["update"]["direction"] = dict(kind="secant_projection", probe_l2=H, anchor=selection)
    return json.dumps(p, sort_keys=True, separators=(",", ":"))


def audit_probe(origin, old_z, info, i, mode, selection, kind="secant_projection"):
    g = common["grad"](origin, cases[i % len(cases)], mode)
    candidate, r = origin.step_adaptation_gradient_at_distance(g, mode=mode, target_l2=H)
    kl = prev["readout"](old_z, ray["logits"](candidate))
    assert len(kl) == len(info["probe_per_anchor_kl"])
    assert all(prev["near"](a,b) for a,b in zip(kl,info["probe_per_anchor_kl"]))
    peak = max(range(len(kl)), key=kl.__getitem__)
    index = peak if selection == "largest" else (peak+1) % len(kl)
    assert (info["peak_anchor"], info["selected_anchor"]) == (peak,index)
    assert info["policy"] == dict(kind=kind, probe_l2=H, anchor=selection)
    assert prev["near"](info["source_gradient_l2"], r["gradient_l2"])
    assert info["normal_l2"] > 0 and abs(info["projected_normal_cosine"]) < 1e-6
    assert 0.001 <= info["retained_fraction"] <= 1+1e-6
    return g.parameter_gradients()


def run(prior, out, cuts):
    ref = base["pinned"](prior/"raw.json", RAW)["cases"]
    pins = base["pinned"](prior/"cuts.json", CUTS)["cases"]
    origin, _ = common["host"](common["HOST"])
    z0 = ray["logits"](origin)
    results, bridge = {}, {}
    with out.with_suffix(".jsonl").open("x") as journal:
        for selection in ("raw", "largest", "successor"):
            for mode in base["MODES"]:
                name = selection+"/"+mode
                previous = ref["capped/"+mode]
                text = plan(mode, selection)
                session = base["Session"](origin,text)
                before = z0
                reports = []
                saved = base["cut"](session,text)
                for i in range(base["TOTAL"]):
                    old = session.model
                    pre = session.checkpoint_with_id()
                    a = session.attempt_step()
                    journal.write(json.dumps(dict(case=name, attempt=a),allow_nan=False)+"\n")
                    journal.flush()
                    applied = a["status"] == "applied"
                    k = a["step"]["anchor_kl"] if applied else a["refusal"]
                    shaped = selection != "raw"
                    base["check_trials"](k,"capped",applied,int(shaped))
                    assert ("direction" in k) == shaped
                    if shaped:
                        g = audit_probe(old,before,k["direction"],i,mode,selection)
                    row = dict(applied=applied, trials=len(k["trials"]), readouts=k["anchor_readouts"])
                    if shaped:
                        row.update(k["direction"])
                    if not applied:
                        assert a["status"] == "unmatched" and a["next_update"] == i+1
                        assert a["example_index"] == i%len(cases) and a["plan_id"] == session.plan_id
                        assert session.checkpoint_with_id() == pre and session.completed_updates == i
                        assert "source_loss_dot_step" not in row
                        row["reason"] = k["reason"]
                        if not shaped:
                            assert all(k[x]==previous["refusal"][x] for x in k)
                        reports.append(row)
                        break
                    r = a["step"]
                    assert r["update"] == i+1 and r["mode"] == mode
                    after = ray["logits"](session.model)
                    kl = prev["readout"](before,after)
                    mean = base["check_candidate"](kl,k["trials"][-1])
                    assert 0.98*T <= mean <= T and max(kl) <= C
                    assert len(kl) == len(k["per_anchor_kl"]) and all(prev["near"](a,b) for a,b in zip(kl,k["per_anchor_kl"]))
                    x, y = ray["blocks"](old), ray["blocks"](session.model)
                    actual = ray["distance"](x,y)["total"]
                    assert prev["near"](actual,r["parameter_ray"]["realized_l2"])
                    if shaped:
                        dot = []
                        for (p,n), old_block in x.items():
                            if p not in g:
                                assert old_block == y[p,n]
                            else:
                                dot.extend((b-a)*v for a,b,v in zip(old_block,y[p,n],g[p][n]))
                        assert prev["near"](math.fsum(dot),row["source_loss_dot_step"]) and math.fsum(dot) < 0
                        assert r["parameter_ray"]["method"] == "selected_parameter_secant_projected_ray_f32.v1"
                    else:
                        old_step = previous["steps"][i]
                        assert (r["loss"],k["trials"],actual,kl) == (old_step["loss"],old_step["trials"],old_step["parameter_l2"],old_step["per_anchor_kl"])
                    row.update(loss=r["loss"],mean_kl=mean,per_anchor_kl=kl,parameter_l2=actual)
                    reports.append(row)
                    before = after
                    if i < 2:
                        saved = base["cut"](session,text)
                completed = session.completed_updates
                stop = reports[-1].get("reason")
                assert completed == sum(r["applied"] for r in reports)
                assert (completed == base["TOTAL"]) == (session.exhausted and stop is None)
                final_pin = session.checkpoint_with_id()[1]
                if selection == "raw":
                    assert completed == previous["completed"] and final_pin == pins["capped/"+mode]["final_pin"]
                results[name] = dict(completed=completed, attempts=reports, stop=stop, final_pin=final_pin)
                bridge[name] = dict(saved, completed=completed, reason=stop, last_rows=before)
    common["save"](out, dict(schema="spiraltorch.synthetic_line_secant.v1",pretrained_calls=0,
                             raw_control_sha256=RAW,control_cuts_sha256=CUTS,cases=results))
    common["save"](cuts,dict(cases=bridge))
    print(json.dumps({k:v["completed"] for k,v in results.items()},sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("PRIOR_ENVELOPE_DIR NEW_RESULT NEW_CUTS")
    run(*map(Path,sys.argv[1:]))
