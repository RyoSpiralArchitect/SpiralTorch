"""Permute constraint-bank positions, not the loss order. Synthetic weights only."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import runpy
import struct
import sys

mid = runpy.run_path(str(Path(__file__).with_name("native_line_midpoint_fixture.py")))
base, common, prev, ray = (mid[k] for k in ("base", "common", "prev", "ray"))
cases, dump, T = mid["cases"], mid["dump"], mid["T"]
SELECTIONS = ("largest", "successor", "loss_prompt")
OLD = "f44d4144ed79d7060295c86e723f58fcd9caa10a31c644f4cc3e8e205e66260a"


def prompt_id(tokens):
    data = b"spiraltorch.native_causal_lm.line_prompt.v1\0"
    return "sha256:" + hashlib.sha256(data + b"".join(struct.pack("<Q", n) for n in [len(tokens), *tokens])).hexdigest()


def plans():
    for offset, orders, horizon in ((0, list(itertools.permutations(range(3))), 6),
                                    (1, [(0, 1, 2), (0, 2, 1)], 1),
                                    (2, [(0, 1, 2), (0, 2, 1)], 1)):
        for mode, choice, bank in itertools.product(base["MODES"], SELECTIONS, orders):
            p = json.loads(mid["prior"]["plan"](mode, "raw"))
            p["examples"] = cases[offset:] + cases[:offset]
            p["update"]["anchor_examples"] = [(i-offset) % 3 for i in bank]
            p["update"]["direction"] = dict(kind=mid["KIND"], probe_l2=mid["H"], anchor=choice)
            assert [p["examples"][i] for i in p["update"]["anchor_examples"]] == [cases[i] for i in bank]
            yield offset, horizon, mode, choice, bank, dump(p)


def trajectory(origin, cfg, old, journal, key):
    offset, horizon, mode, choice, bank, plan = cfg
    s = mid["Session"](origin, plan)
    z = ray["logits"](origin)
    events = []
    canonical = offset == 0 and bank == (0, 1, 2) and choice != "loss_prompt"
    for cursor in range(horizon):
        loss = (offset + cursor) % 3
        before, checkpoint = s.model, s.checkpoint_with_id()
        a = s.attempt_step()
        journal.write(dump(dict(case=key, cursor=cursor, attempt=a)) + "\n")
        journal.flush()
        if canonical:
            assert a == old[choice+"/"+mode][cursor], "legacy receipt changed"
        k = mid["entry"](a)
        ok = a["status"] == "applied"
        info, miss = k.get("direction"), k.get("raw_miss")
        e = dict(loss_case=loss, applied=ok, branch="midpoint" if miss else "raw",
                 trials=[t["requested_l2"] for t in k["trials"]], readouts=k["anchor_readouts"])
        if miss:
            raw = dict(k, **{n: miss[n] for n in ("anchor_readouts", "reason", "trials")})
            base["check_trials"](raw, "capped", False)
            e["raw_trials"] = [t["requested_l2"] for t in miss["trials"]]
            assert k["anchor_readouts"] == miss["anchor_readouts"] + 3*(1+len(k["trials"]))
            if info:
                last = dict(k, anchor_readouts=k["anchor_readouts"]-miss["anchor_readouts"]+3)
                base["check_trials"](last, "capped", ok, probe_banks=1)
                # An independent, uncommitted finite probe on the measured loss ray.
                g = common["grad"](before, cases[loss], mode)
                assert g.line_prompt_id == prompt_id(cases[loss]["prompt_ids"])
                probe, _ = before.step_adaptation_gradient_at_distance(g, mode=mode, target_l2=mid["H"])
                measured = prev["readout"](z, ray["logits"](probe))
                assert all(prev["near"](measured[i], x) for i, x in zip(bank, info["probe_per_anchor_kl"]))
                peak = max(range(3), key=info["probe_per_anchor_kl"].__getitem__)
                ix = dict(largest=peak, successor=(peak+1)%3, loss_prompt=bank.index(loss))[choice]
                assert (info["peak_anchor"], info["selected_anchor"]) == (peak, ix)
                assert info.get("loss_prompt_id") == (g.line_prompt_id if choice == "loss_prompt" else None)
                e.update(peak=bank[peak], selected=bank[ix], peak_ties=info["probe_per_anchor_kl"].count(max(info["probe_per_anchor_kl"])))
                if ok:
                    assert info["source_loss_dot_step"] < 0
            else:
                assert not ok and not k["trials"]
        else:
            assert ok and info is None
            base["check_trials"](k, "capped", True)
        if not ok:
            assert s.checkpoint_with_id() == checkpoint and s.completed_updates == cursor
            assert a["example_index"] == cursor%3 and a["next_update"] == cursor+1
            e["reason"] = k["reason"]
        else:
            step, z2 = a["step"], ray["logits"](s.model)
            assert (step["example_index"], step["update"]) == (cursor%3, cursor+1)
            kl = prev["readout"](z, z2)
            in_bank_order = [kl[i] for i in bank]
            mean = base["check_candidate"](in_bank_order, k["trials"][-1])
            assert .98*T <= mean <= T and max(kl) <= 2*T
            assert all(prev["near"](x,y) for x,y in zip(in_bank_order, k["per_anchor_kl"]))
            old_ids, ids = common["owners"](before), common["owners"](s.model)
            assert old_ids == (step["pre_lora_id"], step["pre_geometry_id"])
            assert ids == (step["post_lora_id"], step["post_geometry_id"])
            assert [x!=y for x,y in zip(old_ids,ids)] == [mode!="geometry", mode!="lora"]
            e.update(loss=step["loss"], owners=ids, kl=kl)
            z = z2
        events.append(e)
        if not ok:
            break
    if canonical:
        assert len(events) == len(old[choice+"/"+mode])
    n = s.completed_updates
    assert n == sum(e["applied"] for e in events)
    assert s.exhausted == (n == 6)
    state = common["state"](s.model)
    return dict(plan=plan, mode=mode, offset=offset, bank=bank, choice=choice, events=events,
                completed=n, owners=common["owners"](s.model), state=hashlib.sha256(dump(state).encode()).hexdigest(),
                scores=common["losses"](s.model), endpoint=z)


def run(prior_journal, out):
    raw = prior_journal.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == OLD
    old = {}
    for r in map(json.loads, raw.splitlines()):
        if "primary" in r:
            old.setdefault(r["case"], []).append(r["primary"])
    configs = list(plans())
    assert len(configs) == 90
    origin, _ = common["host"](common["HOST"])
    results = {}
    with out.with_suffix(".jsonl").open("x") as journal:
        for cfg in configs:
            offset, _, mode, choice, bank, _ = cfg
            key = "%d/%s/%s/%s" % (offset, mode, choice, "".join(map(str, bank)))
            assert key not in results
            results[key] = trajectory(origin, cfg, old, journal, key)
    common["save"](out, dict(schema="spiraltorch.synthetic_line_prompt_identity.v1", cases=results,
                             old_journal_sha256=OLD, pretrained_calls=0))
    print(dump(dict(trajectories=54, first_step_controls=36, total_attempts=sum(len(c["events"]) for c in results.values()),
                    legacy_receipts=sum(map(len, old.values())), pretrained_calls=0)))


if __name__ == "__main__":
    if len(sys.argv) != 3 or not __debug__:
        sys.exit("PINNED_MIDPOINT_JOURNAL NEW_RESULT (assertions required)")
    run(*map(Path, sys.argv[1:]))
