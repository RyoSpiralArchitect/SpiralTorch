"""Read balanced verb-choice contrasts from sealed outputs, without a model or a new gate."""
import argparse
import json
import math
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
E = runpy.run_path(str(HERE / "experiment.py"))
T = runpy.run_path(str(HERE.parent / "parameter-vjp/radius-sweep/cross-prefix/transfer.py"))
need, sha = E["need"], E["sha"]
PLANS = {
    "rehearsal": ("e7a486aadc12503cbaa6828aa9ced4ea575900c51de470a1f0abf9c1ae2ab636", 60),
    "gpt2": ("5a0e2edfb537bc76ccdf88ef1ed05a00520f013404e8954286c243f151fb2bab", 600),
}
TERMS = ("verb_bias", "subject", "attractor", "interaction")
OUTCOMES = ("both_correct", "both_wrong", "always_singular", "always_plural", "boundary_tie")


def mean(xs):
    return math.fsum(xs) / len(xs)


def close(x, y):
    need(math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12), "factorial and paired margins disagree")


def outcome(singular, plural):
    if singular == 0 or plural == 0:
        return "boundary_tie"
    if singular < 0 < plural:
        return "both_correct"
    if plural < 0 < singular:
        return "both_wrong"
    return "always_singular" if singular < 0 else "always_plural"


def describe(cases, margins):
    groups = T["checked_cases"](cases)
    need(set(margins) == {c["key"] for c in cases} and all(map(math.isfinite, margins.values())),
         "missing, extra or nonfinite margins")
    grids = {group: {} for group in groups}
    for c in cases:
        s = 1 if c["plural"] else -1
        a = -s if c["conflict"] else s
        grids[c["pair"]][s, a] = s * margins[c["key"]]  # plural logit minus singular logit
    quartets, counts = {}, dict.fromkeys(OUTCOMES, 0)
    orders = dict.fromkeys(("correct_order", "reverse_order", "tied"), 0)
    for group, z in grids.items():
        coeff = dict(zip(TERMS, (mean([v * (1, s, a, s*a)[i] for (s, a), v in z.items()])
                                 for i in range(4))))
        pairs = {}
        for a, label in [(-1, "singular"), (1, "plural")]:
            shift = z[1, a] - z[-1, a]
            which = outcome(z[-1, a], z[1, a])
            counts[which] += 1
            orders["correct_order" if shift > 0 else "reverse_order" if shift < 0 else "tied"] += 1
            close(shift * .5, coeff["subject"] + a * coeff["interaction"])
            pairs[label] = dict(singular_gap=z[-1, a], plural_gap=z[1, a], switch=shift, outcome=which)
        quartets[group] = dict(coefficients=coeff, by_attractor=pairs)
    avg = {k: mean([v["coefficients"][k] for v in quartets.values()]) for k in TERMS}
    close(mean(list(margins.values())), avg["subject"])
    return dict(quartets=quartets, coefficients=avg, orders=orders, pair_outcomes=counts,
                pairwise_gold_wins=sum(m > 0 for m in margins.values()),
                pairwise_gold_ties=sum(m == 0 for m in margins.values()), rows=len(cases))


def captured(root, kind):
    # inputs() reads four hash-pinned files for the old *train-only* margin study.
    p, rows, pins = T["S"]["inputs"](root, kind)
    need("evaluation" not in p, "capture must be training only")
    hosts = {}
    for h in T["S"]["HOSTS"]:
        margins, greedy = {}, dict.fromkeys(("gold", "rival", "other"), 0)
        for row in rows:
            if row["host"] != h:
                continue
            v = row["receipt"]["contrast"]
            m = v["target_logit"] - v["alternative_logit"]
            need(m == v["target_minus_alternative"], "stored margin differs from logits")
            margins[row["key"]] = m
            choice = v["greedy_id"]
            need(type(choice) is int and 0 <= choice < p["shape"][-1], "invalid stored greedy id")
            greedy["gold" if choice == v["target_id"] else "rival" if choice == v["alternative_id"]
                   else "other"] += 1
        hosts[h] = dict(**describe(p["train"], margins), recorded_vocab_greedy=greedy)
    return dict(scope="initial_training_capture", kind=kind, input_hashes=pins, hosts=hosts)


def completed(root, kind, result_pin):
    # Refuse an absent or failed attempt before even opening the held-out plan.
    attempt = root / "attempt"
    plan_pin, seconds = PLANS[kind]
    receipt_raw = (attempt / "receipt.json").read_bytes()
    r = json.loads(receipt_raw)
    need(r["success"] is True and r["retry"] is False and r["exit_code"] == 0
         and r["seconds_limit"] == seconds and 0 < r["elapsed"] <= seconds
         and r["min_free"] >= 4 << 30, "no successful bounded attempt")
    result_raw = (attempt / "result.json").read_bytes()
    need(sha(result_raw) == result_pin, "wrong sealed result")
    plan_raw = (root / "plan.json").read_bytes()
    need(sha(plan_raw) == plan_pin, "wrong frozen plan")
    p = json.loads(plan_raw)
    need(p["kind"] == kind and p["seconds"] == seconds and p["sources"] == E["sources"](), "wrong sources")
    result = E["replay"](p, attempt)
    need(E["enc"](result) == result_raw, "original protocol does not replay")
    scores = {stage: {key: row["margin"] for key, row in scores.items()}
              for stage, scores in result["by_item"].items()}
    cases = p["evaluation"]
    stages = {stage: describe(cases, values) for stage, values in scores.items()}
    changes = {}
    for order in result["orders"]:
        seed = order["seed"]
        arms = {}
        for arm in E["ARMS"]:
            before, after = scores["initial-" + arm], scores["%s-%s" % (seed, arm)]
            vals = {k: after[k] - before[k] for k in before}
            coeff = describe(cases, vals)["coefficients"]
            for g, effect in [("all", coeff["subject"]), ("congruent", coeff["subject"] + coeff["attractor"]),
                              ("conflict", coeff["subject"] - coeff["attractor"])]:
                close(effect, order["arms"][arm][g]["change"])
            arms[arm] = coeff
        close(arms["r4"]["subject"] - arms["r1"]["subject"], order["paired_gain"]["all"])
        changes[str(seed)] = arms
    return dict(scope="completed_radius_fit", kind=kind, plan=plan_pin, result=result_pin,
                receipt=sha(receipt_raw), original_advance=result["advance"], stages=stages, changes=changes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("capture", "completed"):
        sub = subs.add_parser(name)
        sub.add_argument("root", type=Path)
        sub.add_argument("kind", choices=PLANS)
        if name == "completed":
            sub.add_argument("result_sha256")
        sub.add_argument("new_output", type=Path)
    args = parser.parse_args()
    report = (captured(args.root, args.kind) if args.command == "capture"
              else completed(args.root, args.kind, args.result_sha256))
    report.update(schema="spiraltorch.paired_verb_readout.v1", new_model_calls=0,
                  source_sha256=sha(Path(__file__).read_bytes()))
    with args.new_output.open("xb") as out:
        out.write(E["enc"](report))
