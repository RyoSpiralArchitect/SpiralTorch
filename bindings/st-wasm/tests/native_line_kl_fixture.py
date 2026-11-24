"""Full-answer training with matched *measured* next-token KL on three training anchors."""
import json
import math
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
ray = runpy.run_path(str(HERE / "native_line_ray_fixture.py"))
common, Session, MODES, CASES = (ray[k] for k in ("common", "Session", "MODES", "CASES"))
TARGET, CEILING, FLOOR, TOTAL = 1/65536, 1/8, 0.98, 6


def readout(old, new):
    """Independent forward KL via normalized full-vocabulary log probabilities."""
    def log_probs(row):
        highest = max(row)
        shifted = [x-highest for x in row]
        log_sum = math.log(math.fsum(math.exp(x) for x in shifted))
        return [x-log_sum for x in shifted]
    result = []
    assert len(old) == len(new) == len(CASES)
    for x, y in zip(old, new):
        assert len(x) == len(y) == 11
        lp, lq = log_probs(x), log_probs(y)
        result.append(math.fsum(math.exp(p)*(p-q) for p, q in zip(lp, lq)))
    return result


def plan(mode, policy):
    p = json.loads(ray["plan"](mode, "ray"))
    if policy == "kl":
        p["update"] = dict(kind="anchor_forward_kl", target_kl=TARGET,
                           max_parameter_l2=CEILING, anchor_examples=list(range(len(CASES))))
    return json.dumps(p, sort_keys=True, separators=(",", ":"))


def near(x, y):
    return math.isclose(x, y, rel_tol=1e-10, abs_tol=3e-15)


def trial_check(report):
    assert report["policy"] == dict(target_kl=TARGET, max_parameter_l2=CEILING)
    trials = report["trials"]
    assert 1 <= len(trials) <= 20 and report["anchor_readouts"] == (len(trials)+1)*len(CASES)
    lo, hi = 0, CEILING
    for i, t in enumerate(trials):
        d, kl = t["requested_l2"], t["mean_kl"]
        assert d == (CEILING if i == 0 else lo+0.5*(hi-lo)) and math.isfinite(kl) and kl >= 0
        if i == len(trials)-1:
            assert kl == report["mean_kl"] and FLOOR <= kl/TARGET <= 1
        elif kl > TARGET:
            hi = d
        else:
            assert i != 0 and kl/TARGET < FLOOR
            lo = d


def run(out, bridge_path):
    origin, _ = common["host"](common["HOST"])
    x0, z0, base = ray["blocks"](origin), ray["logits"](origin), common["losses"](origin)
    old = json.loads((HERE / "native_line_ray_bridge.json").read_bytes())["cases"]
    cases, bridges = {}, {}
    for policy in ("ray", "kl"):
        for mode in MODES:
            p = plan(mode, policy)
            s = Session(origin, p)
            x, z = x0, z0
            steps = []
            status, refusal, cut = "matched" if policy == "kl" else "control", None, None
            for i in range(TOTAL):
                before_pin = s.checkpoint_with_id()[1]
                try:
                    r = s.step()
                except (ValueError, RuntimeError) as error:
                    if policy != "kl" or "anchor KL not matched:" not in str(error):
                        raise
                    assert s.completed_updates == i and s.checkpoint_with_id()[1] == before_pin
                    status, refusal = "unmatched", str(error)
                    break
                assert r["mode"] == mode and r["update"] == i+1
                post = s.model
                new_x, new_z = ray["blocks"](post), ray["logits"](post)
                dx = ray["distance"](x, new_x)["total"]
                assert near(dx, r["parameter_ray"]["realized_l2"])
                kl = readout(z, new_z)
                mean = math.fsum(kl)/len(kl)
                sample = dict(loss=r["loss"], parameter_l2=dx, mean_kl=mean, per_anchor_kl=kl)
                if policy == "kl":
                    k = r["anchor_kl"]
                    trial_check(k)
                    assert near(k["mean_kl"], mean) and all(near(a,b) for a,b in zip(kl,k["per_anchor_kl"]))
                    assert k["trials"][-1]["requested_l2"] == r["parameter_ray"]["requested_l2"]
                    sample.update(trials=k["trials"], anchor_readouts=k["anchor_readouts"])
                else:
                    assert "anchor_kl" not in r and r["loss"] == old[mode]["steps"][i]
                steps.append(sample)
                x, z = new_x, new_z
                if i+1 == 2 and policy == "kl":
                    payload, pin = s.checkpoint_with_id()
                    cut = dict(plan=p, payload=payload, id=pin)
            end = s.model
            if status != "unmatched":
                assert s.exhausted and s.step() is None
            if policy == "ray":
                assert common["state"](end) == old[mode]["endpoint"] and s.checkpoint_with_id()[1] == old[mode]["final_pin"]
            if policy == "kl" and status == "matched":
                bridges[mode] = dict(cut, losses=[r["loss"] for r in steps],
                                     endpoint=common["state"](end), final_pin=s.checkpoint_with_id()[1])
            end_kl = readout(z0,z)
            cases[policy+"/"+mode] = dict(status=status, refusal=refusal, steps=steps, after=common["losses"](end),
                parameter_path=math.fsum(r["parameter_l2"] for r in steps),
                endpoint_l2=ray["distance"](x0,x)["total"], endpoint_per_anchor_kl=end_kl,
                endpoint_kl=math.fsum(end_kl)/len(end_kl))
    result = dict(schema="spiraltorch.synthetic_owner_anchor_kl.v1", host=common["HOST"],
                  target_kl=TARGET, ceiling_l2=CEILING, lower_target_ratio=FLOOR,
                  training_anchor_ids=[e["prompt_ids"] for e in CASES], before=base,
                  in_sample=True, pretrained_calls=0, cases=cases)
    common["save"](out, result)
    common["save"](bridge_path, dict(cases=bridges))
    print(json.dumps({k: dict(status=c["status"], updates=len(c["steps"]), after=c["after"])
                      for k,c in cases.items()}, sort_keys=True))


def resume(path):
    origin,_ = common["host"](common["HOST"])
    updates = 0
    for mode, c in json.loads(path.read_bytes())["cases"].items():
        s = Session.from_json(origin, c["plan"], c["payload"], c["id"])
        assert s.completed_updates == 2 and s.checkpoint_with_id() == (c["payload"], c["id"])
        for i in range(2,TOTAL):
            r=s.step()
            trial_check(r["anchor_kl"])
            assert r["loss"] == c["losses"][i] and r["mode"] == mode
            updates+=1
        assert common["state"](s.model) == c["endpoint"] and s.checkpoint_with_id()[1] == c["final_pin"]
    print(json.dumps(dict(suffix_updates=updates, pretrained_calls=0)))


if __name__ == "__main__":
    if len(sys.argv)==4 and sys.argv[1]=="run":
        run(*map(Path,sys.argv[2:]))
    elif len(sys.argv)==3 and sys.argv[1]=="resume":
        resume(Path(sys.argv[2]))
    else:
        sys.exit("run NEW_RESULT NEW_BRIDGE | resume BRIDGE")
