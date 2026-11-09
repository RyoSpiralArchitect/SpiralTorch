"""Frozen prompt pairs and an opt-in reader. The CLI never loads or scores a model."""
import ast
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import permutations, product
import json
import math
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
PILOT = RESULTS / "2026-11-03-native-field-extraction-sft/pilot.py"
MANIFEST = RESULTS / "2026-11-03-native-factorial-answer-training/prepared-receipts/manifest.json"
MANIFEST_PIN = "72151805040cddbaa9d2149abc90ba5d424da8513eaa62fe2ca240363c7843c9"
FRESH = """
0-1-0 0-1-4 0-4-7 0-6-7 0-7-2 1-2-5 2-2-3 2-3-2 2-5-3 3-1-2 3-4-0
3-5-3 3-5-7 4-1-0 4-2-6 4-2-7 4-3-3 4-4-1 4-6-1 5-0-4 5-3-5 5-5-0
5-7-2 6-4-5 6-5-3 7-0-4 7-1-0 7-4-7 7-5-1 7-6-1 7-7-0 7-7-7
""".split()


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def check(ok, message):
    if not ok:
        raise ValueError(message)


def pinned(path, pin):
    raw = path.read_bytes()
    check(sha256(raw).hexdigest() == pin, f"unexpected bytes in {path.name}")
    return raw


@dataclass(frozen=True)
class Pair:
    key: str
    field: str
    display: tuple
    question: str
    record: str
    answer: str
    first_form: str


def matrix():
    m = json.loads(pinned(MANIFEST, MANIFEST_PIN))
    recent = {k for split in ("train", "eval") for k, *_ in m["lengths"][split]}
    universe = {"-".join(map(str, t)) for t in product(range(8), repeat=3)}
    fresh = set(FRESH)
    older = universe - recent - fresh
    check(len(recent) == 192 and len(fresh) == len(FRESH) == 32 and fresh <= universe and not fresh & recent,
          "invalid fresh record roster")
    check(len(older) == 288 and digest(sorted(older)) == m["frozen"]["excluded_keys_sha256"],
          "roster does not exclude exactly the two older studies")
    # Extract immutable vocabulary literals; do not import or execute old training drivers.
    src = ast.parse(pinned(PILOT, "30c0fe798951749b7e4925561b16555cb1ecbd9374dc27ae82da4178addbe383"))
    vocab = {n.targets[0].id: ast.literal_eval(n.value) for n in src.body if isinstance(n, ast.Assign)
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id in {"FIELDS", "VALUES"}}
    fields = vocab["FIELDS"]
    orders = list(permutations(fields))
    pairs = []
    for i, key in enumerate(sorted(fresh, key=lambda k: digest(["paired-forms-v1", k]))):
        values = {f: vocab["VALUES"][j][int(v)] for j, (f, v) in enumerate(zip(fields, key.split("-"), strict=True))}
        display = tuple((f, values[f]) for f in orders[i % len(orders)])
        for j, field in enumerate(fields):
            q = " | ".join(f"{f}: {v}" for f, v in display) + f"\nWhat is the {field}?\nAnswer:"
            r = "Record: " + "; ".join(f"{f} = {v}" for f, v in display) + f".\nReturn the {field}.\nAnswer:"
            pairs.append(Pair(key, field, display, q, r, " " + values[field], ("question", "record")[(i + j) % 2]))
    return pairs


def manifest(pairs):
    return dict(schema="spiraltorch.paired_prompt_contrast_design.v1", prior_manifest=MANIFEST_PIN,
                unit="record", records=len({p.key for p in pairs}), pairs=len(pairs), prompts=2 * len(pairs),
                pair_sha256=digest([asdict(p) for p in pairs]))


def prepare_pair(pair, encode):
    """encode(text) must omit special tokens. Neither answer nor LF is sent to the model."""
    prefixes, candidates = {}, []
    for form in ("question", "record"):
        prompt = getattr(pair, form)
        prefix, answer, lf = (tuple(encode(prompt + tail)) for tail in ("", pair.answer, "\n"))
        check(prefix and answer[:len(prefix)] == lf[:len(prefix)] == prefix, "prompt token boundary changed")
        content = answer[len(prefix):]
        stop = lf[len(prefix):]
        check(content and len(stop) == 1 and content[0] != stop[0], "no distinct first answer and single LF tokens")
        prefixes[form] = prefix
        candidates.append((content[0], stop[0]))
    check(candidates[0] == candidates[1], "candidate IDs differ between prompt forms")
    return prefixes, candidates[0]


def inspect_pair(pair, encode, model):
    """Two independent prompt-only forwards; no sampling, labels, updates or reference continuation."""
    prefixes, (target, lf) = prepare_pair(pair, encode)
    return inspect_prepared(pair, prefixes, target, lf, lambda _form, ids, a, b: model.next_token_contrast(ids, a, b))


def inspect_prepared(pair, prefixes, target, lf, observe):
    """Read both forms from an admitted token matrix; observe receives the form for journaling."""
    forms = ("question", "record") if pair.first_form == "question" else ("record", "question")
    results = {}
    for form in forms:
        score = observe(form, prefixes[form], target, lf)
        winner = score["greedy_id"]
        results[form] = {**score, "winner": "target" if winner == target else "LF" if winner == lf else "other"}
    return dict(key=pair.key, field=pair.field, pair_sha256=digest(asdict(pair)), forms=results,
                record_minus_question_margin=(results["record"]["target_minus_alternative"]
                                              - results["question"]["target_minus_alternative"]))


def summarize(pairs, observations):
    """Summarize one model, giving equal weight to each complete record."""
    fields = ("color", "shape", "material")
    planned = {(p.key, p.field): digest(asdict(p)) for p in pairs}
    keys = sorted({p.key for p in pairs})
    check(keys and len(planned) == len(pairs) and planned.keys() == set(product(keys, fields)),
          "expected one complete field triplet per record")
    seen, margins, transitions = set(), defaultdict(dict), Counter()
    for row in observations:
        key, field = row["key"], row["field"]
        check((key, field) in planned and (key, field) not in seen, "duplicate or unplanned field pair")
        check(row["pair_sha256"] == planned[key, field], "mislabeled prompt pair")
        seen.add((key, field))
        winners, values, ids = [], [], []
        for form in ("question", "record"):
            score = row["forms"][form]
            a, b, g = score["target_id"], score["alternative_id"], score["greedy_id"]
            check(a != b and all(type(i) is int and i >= 0 for i in (a, b, g)), "invalid token IDs")
            winner = "target" if g == a else "LF" if g == b else "other"
            margin = score["target_logit"] - score["alternative_logit"]
            check(math.isfinite(margin) and margin == score["target_minus_alternative"]
                  and winner == score["winner"], "inconsistent next-token score")
            winners.append(winner)
            values.append(margin)
            ids.append((a, b))
        delta = values[1] - values[0]
        check(ids[0] == ids[1] and math.isfinite(delta) and delta == row["record_minus_question_margin"],
              "inconsistent pair")
        margins[key][field] = delta
        transitions[tuple(winners)] += 1
    check(seen == planned.keys(), "missing field pairs")
    means = {k: math.fsum(margins[k][f] for f in fields) / len(fields) for k in keys}
    categories = ("target", "LF", "other")
    return dict(records=len(keys), pairs=len(seen), mean_record_delta=math.fsum(means.values()) / len(keys),
                record_means=means, field_deltas=dict(margins),
                transition_pairs={a: {b: transitions[a, b] for b in categories} for a in categories})


if __name__ == "__main__":
    print(json.dumps(manifest(matrix()), indent=2))
