"""Read only the generated study receipts. No extension or models needed."""
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import permutations, product
import json
import math
from pathlib import Path
import sys


def digest(raw):
    return sha256(raw).hexdigest()


def identity(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False)


def groups(values):
    return dict(sorted(Counter(values).items()))


def checked(native, journal, legacy):
    raw, saved, old = (p.read_bytes() for p in (native, journal, legacy))
    ref = json.loads(raw)
    assert ref["schema"] == "spiraltorch.synthetic_line_prompt_identity.v1" and not ref["pretrained_calls"]
    assert digest(old) == ref["old_journal_sha256"] == "f44d4144ed79d7060295c86e723f58fcd9caa10a31c644f4cc3e8e205e66260a"
    legacy_steps = defaultdict(list)
    for row in map(json.loads, old.splitlines()):
        if "primary" in row:
            legacy_steps[row["case"]].append(row["primary"])
    cases = ref["cases"]
    schedule = defaultdict(list)
    for row in map(json.loads, saved.splitlines()):
        assert row["cursor"] == len(schedule[row["case"]])
        schedule[row["case"]].append(row["attempt"])
    modes = ("lora", "geometry", "joint")
    choices = ("largest", "successor", "loss_prompt")
    orders = ["".join(p) for p in permutations("012")]
    expected = {"/".join((str(off), m, s, bank)) for off, m, s in product(range(3), modes, choices)
                for bank in (orders if off == 0 else ("012", "021"))}
    assert set(schedule) == set(cases) == expected
    prior_checks = count = accepted = refusals = shaped = tie_count = 0
    for key, steps in schedule.items():
        c = cases[key]
        plan, events, bank = json.loads(c["plan"]), c["events"], c["bank"]
        encoded_bank = "".join(map(str, bank))
        assert key == "/".join((str(c["offset"]), c["mode"], c["choice"], encoded_bank))
        assert plan["mode"] == c["mode"] and plan["update"]["direction"]["anchor"] == c["choice"]
        assert [(idx+c["offset"])%3 for idx in plan["update"]["anchor_examples"]] == bank
        horizon = 1 if c["offset"] else 6
        assert 0 < len(steps) == len(events) <= horizon
        assert sum(e["applied"] for e in events) == c["completed"]
        assert len(events) == horizon or not events[-1]["applied"]
        if c["offset"] == 0 and encoded_bank == "012" and c["choice"] != "loss_prompt":
            assert steps == legacy_steps[c["choice"]+"/"+c["mode"]]
            prior_checks += len(steps)
        for i, (a,e) in enumerate(zip(steps,events)):
            count += 1
            ok = a["status"] == "applied"
            k = a["step"]["anchor_kl"] if ok else a["refusal"]
            assert ok == e["applied"] and a["status"] in ("applied","unmatched")
            assert (e["loss_case"],e["trials"],e["readouts"]) == ((i+c["offset"])%3,[t["requested_l2"] for t in k["trials"]],k["anchor_readouts"])
            pin = a["step"]["plan_id"] if ok else a["plan_id"]
            assert pin == "sha256:"+digest(c["plan"].encode())
            assert (e["branch"] == "midpoint") == ("raw_miss" in k)
            if "raw_miss" in k:
                assert e["raw_trials"] == [t["requested_l2"] for t in k["raw_miss"]["trials"]]
            info = k.get("direction")
            assert bool(info) == ("selected" in e)
            if info:
                shaped += 1
                probe = info["probe_per_anchor_kl"]
                peak = probe.index(max(probe))
                assert e["peak_ties"] == probe.count(max(probe))
                tie_count += e["peak_ties"] > 1
                assert e["peak"] == bank[peak] and e["selected"] == bank[info["selected_anchor"]]
                assert e["selected"] == {"largest":bank[peak], "successor":bank[(peak+1)%3], "loss_prompt":e["loss_case"]}[c["choice"]]
            if ok:
                accepted += 1
                s = a["step"]
                assert s["update"] == i+1 and s["loss"] == e["loss"]
                assert [s["post_lora_id"],s["post_geometry_id"]] == e["owners"]
                assert all(math.isclose(x,e["kl"][j],abs_tol=3e-15,rel_tol=1e-10) for j,x in zip(bank,k["per_anchor_kl"]))
                assert .98/65536 <= math.fsum(e["kl"])/3 <= 1/65536 and max(e["kl"]) <= 2/65536
            else:
                refusals += 1
                assert i == len(steps)-1 and a["next_update"] == i+1 and k["reason"] == e["reason"]
    main, secondary = {}, {}
    for mode, choice in product(modes,choices):
        selected = [cases["0/"+mode+"/"+choice+"/"+b] for b in orders]
        main[mode+"/"+choice] = dict(completed_by_bank={b:c["completed"] for b,c in zip(orders,selected)},
            successful_first_anchors=groups(str(c["events"][0].get("selected","raw")) for c in selected if c["events"][0]["applied"]),
            endpoint_states=len({c["state"] for c in selected}), trajectories=len({
                identity([c["events"],c["owners"],c["state"],c["scores"],c["endpoint"]]) for c in selected}))
    for mode, offset in product(modes,(1,2)):
        selected = [c for c in cases.values() if c["mode"]==mode and c["offset"]==offset]
        secondary[mode+"/"+str(offset)] = groups(
            "raw applied" if e["branch"]=="raw" else ("anchor "+str(e.get("selected"))+(" applied" if e["applied"] else " refused"))
            for c in selected for e in c["events"])
    # Compare actual joint weights/outputs, not just the number of accepted updates.
    def endpoint(c):
        return identity([c["state"],c["owners"],c["scores"],c["endpoint"]])
    successful_joint = {endpoint(c) for c in cases.values() if c["offset"]==0 and c["mode"]=="joint" and c["completed"]==6}
    return dict(schema="spiraltorch.synthetic_line_prompt_summary.v1", source_sha256=digest(raw), journal_sha256=digest(saved),
                previous_journal_sha256=digest(old), pretrained_calls=0, main=main, alternate_first_tasks=secondary,
                counts=dict(main=54, alternate_first=36, attempts=count, applied=accepted, refused=refusals,
                            measured_directions=shaped, tied_probe_peaks=tie_count, canonical_receipts=prior_checks,
                            distinct_successful_joint_endpoints=len(successful_joint)))


if __name__ == "__main__":
    if len(sys.argv) != 4 or not __debug__:
        sys.exit("RESULT_JSON RAW_JSONL PRIOR_JOURNAL (assertions required)")
    print(json.dumps(checked(*map(Path,sys.argv[1:])), sort_keys=True, indent=2))
