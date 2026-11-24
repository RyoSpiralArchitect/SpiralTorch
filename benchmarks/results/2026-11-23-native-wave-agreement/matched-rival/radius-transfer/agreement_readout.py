"""Read balanced verb-choice contrasts from sealed outputs, without a model or a new gate."""
import argparse
import json
import math
from pathlib import Path
import runpy
import sys

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
    need(math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12), "independent contrasts disagree")


def outcome(singular, plural):
    if singular == 0 or plural == 0:
        return "boundary_tie"
    if singular < 0 < plural:
        return "both_correct"
    if plural < 0 < singular:
        return "both_wrong"
    return "always_singular" if singular < 0 else "always_plural"


def softplus(x):
    return max(x, 0.) + math.log1p(math.exp(-abs(x)))


def vocabulary_row(row, gold, rival):
    need(gold != rival and 0 <= min(gold, rival) <= max(gold, rival) < len(row)
         and all(map(math.isfinite, row)), "invalid full-vocabulary row")
    choice = softplus(row[rival] - row[gold])
    coverage = 0.
    if len(row) > 2:
        # Separate pivots preserve pair mass even when full and conditional gold NLL
        # are both huge and subtracting the rounded losses would erase the difference.
        pair_peak = max(row[gold], row[rival])
        def outside():
            return (v for i, v in enumerate(row) if i != gold and i != rival)
        out_peak = max(outside())
        out_norm = math.log(math.fsum(math.exp(v - out_peak) for v in outside()))
        pair_norm = math.log1p(math.exp(-abs(row[rival] - row[gold])))
        coverage = softplus((out_peak - pair_peak) + (out_norm - pair_norm))
    peak = max(row)
    best = next(i for i, v in enumerate(row) if v == peak)  # first maximum, as in the native decoder
    return dict(choice_nll=choice, coverage_nll=coverage, pair_mass=math.exp(-coverage),
                gold_nll=choice + coverage, greedy="gold" if best == gold else "rival" if best == rival else "other",
                top_tie=sum(v == peak for v in row) > 1)


COSTS = ("choice_nll", "coverage_nll", "gold_nll", "pair_mass")


def vocabulary_stage(attempt, stage, cases, vocab, digest, scores):
    raw = (attempt / (stage + ".f32le")).read_bytes()
    need(sys.byteorder == "little" and len(raw) == 4*len(cases)*vocab and sha(raw) == digest,
         "wrong archived logits")
    buf = memoryview(raw).cast("f")
    rows = {}
    for i, c in enumerate(cases):
        row = buf[i*vocab:(i+1)*vocab]
        v = vocabulary_row(row, c["gold"], c["rival"])
        need(row[c["gold"]]-row[c["rival"]] == scores[c["key"]]["margin"]
             and math.isclose(v["gold_nll"], scores[c["key"]]["nll"], rel_tol=1e-12, abs_tol=1e-12),
             "probability split disagrees with frozen score")
        rows[c["key"]] = v
    counts = {k:sum(v["greedy"] == k for v in rows.values()) for k in ("gold", "rival", "other")}
    return rows, dict(means={k:mean([v[k] for v in rows.values()]) for k in COSTS},
                      recorded_vocab_rows=len(cases), greedy=counts, top_ties=sum(v["top_tie"] for v in rows.values()))


def describe(cases, margins):
    groups = T["checked_cases"](cases)
    need(set(margins) == {c["key"] for c in cases} and all(map(math.isfinite, margins.values())),
         "missing, extra or nonfinite margins")
    grids = {group: {} for group in groups}
    prefixes = {group: {} for group in groups}
    for c in cases:
        s = 1 if c["plural"] else -1
        a = -s if c["conflict"] else s
        grids[c["pair"]][s, a] = s * margins[c["key"]]  # plural logit minus singular logit
        prefix = c["prefix"]
        need(len(prefix) == 6 and all(type(v) is int and v >= 0 for v in prefix), "wrong tokenized template")
        prefixes[c["pair"]][s, a] = prefix
    quartets, counts = {}, dict.fromkeys(OUTCOMES, 0)
    orders = dict.fromkeys(("correct_order", "reverse_order", "tied"), 0)
    for group, z in grids.items():
        tokens = prefixes[group]
        for n in (-1, 1):
            for first, second, slot in [((-1, n), (1, n), 1), ((n, -1), (n, 1), 4)]:
                changed = [i for i, (x, y) in enumerate(zip(tokens[first], tokens[second])) if x != y]
                need(changed == [slot], "not a one-token subject/attractor switch")
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
    scores = {stage: {key: row["margin"] for key, row in observed.items()}
              for stage, observed in result["by_item"].items()}
    cases = p["evaluation"]
    stages = {stage: describe(cases, values) for stage, values in scores.items()}
    vocab_rows = {}
    for stage in stages:
        vocab_rows[stage], stages[stage]["vocabulary"] = vocabulary_stage(
            attempt, stage, cases, p["shape"][-1], result["logit_sha256"][stage], result["by_item"][stage])
    changes, vocab_changes = {}, {}
    for order in result["orders"]:
        seed = order["seed"]
        arms = {}
        vocab_changes[str(seed)] = {}
        for arm in E["ARMS"]:
            before, after = scores["initial-" + arm], scores["%s-%s" % (seed, arm)]
            vals = {k: after[k] - before[k] for k in before}
            coeff = describe(cases, vals)["coefficients"]
            for g, effect in [("all", coeff["subject"]), ("congruent", coeff["subject"] + coeff["attractor"]),
                              ("conflict", coeff["subject"] - coeff["attractor"])]:
                close(effect, order["arms"][arm][g]["change"])
            arms[arm] = coeff
            a, b = vocab_rows["initial-"+arm], vocab_rows["%s-%s"%(seed,arm)]
            deltas = {k:mean([b[key][k]-a[key][k] for key in a]) for k in COSTS}
            close(deltas["gold_nll"], order["arms"][arm]["all"]["nll_change"])
            close(deltas["gold_nll"], deltas["coverage_nll"] + deltas["choice_nll"])
            vocab_changes[str(seed)][arm] = deltas
        close(arms["r4"]["subject"] - arms["r1"]["subject"], order["paired_gain"]["all"])
        changes[str(seed)] = arms
    return dict(scope="completed_radius_fit", kind=kind, plan=plan_pin, result=result_pin,
                receipt=sha(receipt_raw), original_advance=result["advance"], stages=stages, changes=changes,
                vocabulary_changes=vocab_changes)


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
