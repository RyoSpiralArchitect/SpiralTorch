"""Cross the presentation and request from an observed pair. No model calls on import or CLI."""
from collections import Counter, defaultdict
from dataclasses import replace
import argparse
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matrix as m
from admit_tokenizer import TOKEN_INPUT_PIN

CELLS = ("pipe_what", "pipe_return", "record_what", "record_return")
CACHED = {"question": "pipe_what", "record": "record_return"}
NEW = ("pipe_return", "record_what")
FULL_ORDERS = tuple(tuple(CELLS[(i + j) % 4] for j in (0, 1, 3, 2)) for i in range(4))
GRID_INPUT_PIN = "7f37c14a9a9816039d9c0ff9ad1456276a647b482f0c26007cadc661ad1826e9"
FIELDS = ("color", "shape", "material")
CATEGORIES = ("target", "LF", "other")
EDGES = {
    "presentation_at_what": ("pipe_what", "record_what"),
    "presentation_at_return": ("pipe_return", "record_return"),
    "request_with_pipe": ("pipe_what", "pipe_return"),
    "request_with_record": ("record_what", "record_return"),
}


def texts(pair):
    pipe, what, end = pair.question.split("\n")
    record, request, other_end = pair.record.split("\n")
    m.check(end == other_end == "Answer:" and what != request and pipe != record, "unexpected parent forms")
    return dict(pipe_what=pair.question, pipe_return="\n".join((pipe, request, end)),
                record_what="\n".join((record, what, end)), record_return=pair.record)


def new_call_order(pair):
    return NEW if pair.first_form == "question" else NEW[::-1]


def admit(pairs, encode, prior_input_pin=TOKEN_INPUT_PIN):
    old_rows, rows, distinct = [], [], set()
    for pair in pairs:
        forms = texts(pair)
        old, ids = m.prepare_pair(pair, encode)
        cross = replace(pair, question=forms["pipe_return"], record=forms["record_what"])
        new, cross_ids = m.prepare_pair(cross, encode)
        m.check(ids == cross_ids, "the two candidate IDs must be identical in all four cells")
        old_rows.append(dict(key=pair.key, field=pair.field, target=ids[0], lf=ids[1], prefixes=old))
        prefixes = dict(pipe_what=old["question"], pipe_return=new["question"],
                        record_what=new["record"], record_return=old["record"])
        m.check(all(0 < len(p) <= 64 for p in prefixes.values()), "prompt outside registered context")
        distinct.update(prefixes.values())
        rows.append(dict(key=pair.key, field=pair.field, target=ids[0], lf=ids[1], prefixes=prefixes))
    m.check(len(distinct) == 4 * len(rows), "different cell identities share an encoded prompt")
    m.check(m.digest(old_rows) == prior_input_pin, "cached diagonals differ from the registered input")
    return rows


def admit_gpt2(path):
    prior = json.loads(m.pinned(m.MANIFEST, m.MANIFEST_PIN))
    with path.open("rb") as f:
        raw = f.read(4 * 1024 * 1024 + 1)
    m.check(len(raw) <= 4 * 1024 * 1024 and sha256(raw).hexdigest() == prior["tokenizer"], "wrong tokenizer")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_str(raw.decode())
    rows = admit(m.matrix(), lambda s: tok.encode(s, add_special_tokens=False).ids)
    m.check(m.digest(rows) == GRID_INPUT_PIN, "registered four-cell token matrix changed")
    return rows


def effects(y):
    """Factor differences in signed target-minus-LF margin; unrestricted winners are separate."""
    pq, pr, rq, rr = (y[k] for k in CELLS)
    return {**{name: y[b] - y[a] for name, (a, b) in EDGES.items()},
            "presentation": math.fsum([rq, rr, -pq, -pr]) / 2,
            "request": math.fsum([pr, rr, -pq, -rq]) / 2,
            "interaction": math.fsum([rr, -rq, -pr, pq]), "diagonal": rr - pq}


def checked_score(raw, row, cell):
    ids = row["prefixes"][cell]
    m.check(raw["prefix_sha256"] == m.digest(ids), "mislabeled score prefix")
    s = raw["score"]
    a, b, g = s["target_id"], s["alternative_id"], s["greedy_id"]
    m.check(all(type(i) is int and 0 <= i < 50257 for i in (a,b,g)) and type(s["prompt_tokens"]) is int and a != b and
            (a, b, s["prompt_tokens"]) == (row["target"], row["lf"], len(ids)), "score token identity differs")
    x, y = s["target_logit"], s["alternative_logit"]
    m.check(all(type(v) in (int, float) and math.isfinite(v) for v in (x, y, s["target_minus_alternative"])),
            "non-numeric or non-finite native score")
    margin = x - y
    m.check(math.isfinite(margin) and margin == s["target_minus_alternative"], "inconsistent native margin")
    m.check((g != a or x >= y) and (g != b or y >= x) and (x != y or g not in (a, b) or g == min(a, b)),
            "greedy winner contradicts candidate logits")
    return margin, "target" if g == a else "LF" if g == b else "other"


def measure_crossed(pairs, rows, cached, model, observe):
    """Validate all cached cells first. Only off-diagonal prompt IDs reach the callback."""
    slots = {(p.key, p.field) for p in pairs}
    keys = {k for k, _ in slots}
    m.check(keys and len(pairs) == len(rows) == len(slots) and slots == set(product(keys, FIELDS)),
            "incomplete input field triplets")
    m.check(cached.keys() == {(*slot, c) for slot in slots for c in CACHED.values()}, "missing or foreign cached cells")
    prefixes = set()
    for p, row in zip(pairs, rows, strict=True):
        m.check((p.key, p.field) == (row["key"], row["field"]) and set(row["prefixes"]) == set(CELLS),
                "input identity differs")
        for cell in CELLS:
            ids = tuple(row["prefixes"][cell])
            m.check(0 < len(ids) <= 64 and ids not in prefixes and all(type(i) is int and 0 <= i < 50257 for i in ids),
                    "invalid or reused prefix")
            prefixes.add(ids)
        for cell in CACHED.values():
            item = cached[p.key, p.field, cell]
            m.check((item["model"], item["source"]) == (model, "cached"), "wrong cached model or source")
            checked_score(item, row, cell)
    evidence = dict(cached)
    for p, row in zip(pairs, rows, strict=True):
        for cell in new_call_order(p):
            ids = tuple(row["prefixes"][cell])
            score = observe(p, cell, ids, row["target"], row["lf"])
            item = dict(model=model, source="new", prefix_sha256=m.digest(ids), score=score)
            checked_score(item, row, cell)
            evidence[p.key, p.field, cell] = item
    return summarize(rows, evidence, model)


def measure_all(pairs, rows, model, observe, *, grid_pin=GRID_INPUT_PIN):
    """Read all four cells for each field in this execution; no cached scores are accepted."""
    m.check(m.digest(rows) == grid_pin, "wrong full input grid")
    slots = [(p.key, p.field) for p in pairs]
    keys = {k for k, _ in slots}
    m.check(keys and len(slots) == len(set(slots)) and set(slots) == set(product(keys, FIELDS))
            and slots == [(r["key"], r["field"]) for r in rows], "wrong record/field coverage")
    unique = set()
    for r in rows:
        a, b = r["target"], r["lf"]
        m.check(set(r["prefixes"]) == set(CELLS) and a != b and
                all(type(i) is int and 0 <= i < 50257 for i in (a,b)), "invalid candidates or cells")
        for seq in r["prefixes"].values():
            ids = tuple(seq)
            m.check(0 < len(ids) <= 64 and all(type(i) is int and 0 <= i < 50257 for i in ids)
                    and ids not in unique, "invalid or repeated prefix")
            unique.add(ids)
    evidence = {}
    for i, (p, r) in enumerate(zip(pairs, rows, strict=True)):
        for cell in FULL_ORDERS[i % 4]:
            ids = tuple(r["prefixes"][cell])
            s = observe(p, cell, ids, r["target"], r["lf"])
            item = dict(model=model, source="new", prefix_sha256=m.digest(ids), score=s)
            checked_score(item, r, cell)
            evidence[p.key, p.field, cell] = item
    return summarize(rows, evidence, model, mode="all_measured")


def summarize(rows, evidence, model, *, mode="reuse_diagonals"):
    """One model, one observation per key/field/cell, one equally weighted triplet per record."""
    m.check(mode in ("reuse_diagonals", "all_measured"), "unknown observation mode")
    named = {(r["key"], r["field"]): r for r in rows}
    keys = sorted({k for k, _ in named})
    m.check(keys and len(named) == len(rows) and named.keys() == set(product(keys, FIELDS)),
            "expected each record exactly once per field")
    m.check(evidence.keys() == {(*kf, cell) for kf in named for cell in CELLS}, "missing or extra cells")
    by_key, field_effects = {}, defaultdict(dict)
    winners = {c: Counter() for c in CELLS}
    transitions = {k: Counter() for k in EDGES}
    for (key, field), row in named.items():
        y, w = {}, {}
        for cell in CELLS:
            raw = evidence[key, field, cell]
            m.check(raw["model"] == model and raw["source"] == ("new" if mode == "all_measured" or cell in NEW else "cached"),
                    "model or score source differs")
            y[cell], w[cell] = checked_score(raw, row, cell)
            winners[cell][w[cell]] += 1
        field_effects[key][field] = effects(y)
        for name, (a,b) in EDGES.items():
            transitions[name][w[a], w[b]] += 1
    for k in keys:
        by_key[k] = {e: math.fsum(field_effects[k][f][e] for f in FIELDS) / len(FIELDS)
                     for e in field_effects[k][FIELDS[0]]}
    return dict(model=model, records=len(keys), field_pairs=len(rows), field_effects=dict(field_effects),
                record_means=by_key, mean={e: math.fsum(by_key[k][e] for k in keys)/len(keys) for e in by_key[keys[0]]},
                winners={c: {w: winners[c][w] for w in CATEGORIES} for c in CELLS},
                transitions={name: {a: {b: counts[a,b] for b in CATEGORIES} for a in CATEGORIES}
                             for name, counts in transitions.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, help="also verify token IDs using an existing tokenizer.json")
    args = parser.parse_args()
    pairs = m.matrix()
    report = dict(unit="previously observed record", records=len({p.key for p in pairs}), field_pairs=len(pairs),
                  cached_prefixes=2*len(pairs), new_prefixes=2*len(pairs),
                  text_sha256=m.digest([dict(key=p.key,field=p.field,forms=texts(p)) for p in pairs]))
    if args.tokenizer:
        report["input_sha256"] = m.digest(admit_gpt2(args.tokenizer))
    print(json.dumps(report, indent=2))
