"""Closed, token-aligned English number agreement pilot. No ML dependencies."""
import hashlib
import json
import math


PINS = {
    "config.json": "0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4",
    "model.safetensors": "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
    "vocab.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
}
EXTENSION = "69245aec6cb242848e282ee71228a0687bf8fb86d925648a063634a6047ad2d5"
ARMS = ("wave_fixed", "wave_sgd", "affine_sgd")
TRAIN = (("cup", "cups", "plate", "plates"), ("box", "boxes", "door", "doors"),
         ("lamp", "lamps", "chair", "chairs"), ("key", "keys", "lock", "locks"),
         ("boat", "boats", "bridge", "bridges"), ("book", "books", "desk", "desks"))
TEST = (("bird", "birds", "tree", "trees"), ("truck", "trucks", "gate", "gates"),
        ("cat", "cats", "window", "windows"), ("sign", "signs", "road", "roads"))


def need(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def single_word_id(word, vocab, ranks):
    # Only our listed ASCII words (and the GPT-2 space marker), not a general tokenizer.
    pieces = list(word)
    while len(pieces) > 1:
        pairs = zip(pieces, pieces[1:])
        pair = min(pairs, key=lambda p: ranks.get(p, math.inf))
        need(pair in ranks, "fixture word no longer encodes to one token: " + word)
        out, i = [], 0
        while i < len(pieces):
            if tuple(pieces[i:i+2]) == pair:
                out.append("".join(pair))
                i += 2
            else:
                out.append(pieces[i])
                i += 1
        pieces = out
    return vocab[pieces[0]]


def cases(groups, ids):
    result = []
    for sg, pl, near_sg, near_pl in groups:
        for h, subject in enumerate((sg, pl)):
            for n, neighbor in enumerate((near_sg, near_pl)):
                words = ("The", subject, "near", "the", neighbor)
                result.append(dict(key=sg+"."+str(h)+str(n), pair=sg, plural=bool(h),
                                   conflict=h != n, text=" ".join(words),
                                   prefix=[ids[w] for w in words],
                                   gold=ids[("is", "are")[h]], rival=ids[("are", "is")[h]]))
    return result


def build_plan(kind, source, model_dir=None):
    if kind == "gpt2":
        need(model_dir is not None, "local model directory required")
        for name, pin in PINS.items():
            need(file_hash(model_dir/name) == pin, "wrong local file: " + name)
        train, test, orders = TRAIN, TEST, (17, 31, 73)
        vocab = json.loads((model_dir/"vocab.json").read_bytes())
        lines = (model_dir/"merges.txt").read_text().splitlines()
        ranks = {tuple(line.split()): i for i, line in enumerate(lines[1:])}
        words = {"The", "near", "the", "is", "are"} | {w for group in train+test for w in group}
        ids = {w: single_word_id(w if w == "The" else chr(288)+w, vocab, ranks) for w in sorted(words)}
        shape = [12, 768, 12, 50257]
    else:
        need(kind == "rehearsal" and model_dir is None, "wrong model scope")
        train = (("a", "as", "b", "bs"),)
        test = (("c", "cs", "d", "ds"),)
        words = ["The", "near", "the", "is", "are", *train[0], *test[0]]
        ids, orders, shape = dict(zip(words, range(1, len(words)+1))), (11,), [2, 16, 2, 32]
    need(len(ids) == len(set(ids.values())), "colliding fixture tokens")
    tr, ev = cases(train, ids), cases(test, ids)
    order = [sorted(range(len(tr)), key=lambda i: digest(("%s/%s" % (seed, tr[i]["key"])).encode()))
             for seed in orders]
    return dict(schema="spiraltorch.wave_language_agreement.v1", kind=kind, shape=shape,
                source_sha256=source, model_pins=PINS if kind == "gpt2" else None,
                extension_pin=EXTENSION if kind == "gpt2" else None,
                train=tr, evaluation=ev, seeds=list(orders), order=order, arms=list(ARMS),
                lr=0.025, strength=0.25, gradient_calls=len(tr)*len(order)*3,
                applied_updates=len(tr)*len(order)*2,
                scored_rows=len(ev)*(2+len(order)*3), wall_seconds=600 if kind == "gpt2" else 60)


def metrics(row, gold, rival):
    need(gold != rival and 0 <= min(gold, rival) <= max(gold, rival) < len(row), "invalid target pair")
    need(all(math.isfinite(x) for x in row), "nonfinite logits")
    peak = max(row)
    return dict(margin=float(row[gold])-float(row[rival]),
                nll=(peak-float(row[gold]))+math.log(math.fsum(math.exp(x-peak) for x in row)))


def summarize(plan, scores):
    keys = [x["key"] for x in plan["evaluation"]]
    rows = {x["key"]: x for x in plan["evaluation"]}
    def averages(name, start, group):
        a, b = scores[name], scores[start]
        need(set(a) == set(b) == set(keys), "mismatched evaluation prompts")
        chosen = [k for k in keys if group == "all" or rows[k]["conflict"] == (group == "conflict")]
        n = len(chosen)
        return dict(n=n, margin_delta=math.fsum(a[k]["margin"]-b[k]["margin"] for k in chosen)/n,
                    nll_delta=math.fsum(a[k]["nll"]-b[k]["nll"] for k in chosen)/n,
                    final_correct=sum(a[k]["margin"]>0 for k in chosen),
                    final_tied=sum(a[k]["margin"]==0 for k in chosen))
    per_order = []
    for seed in plan["seeds"]:
        result = {}
        for arm in ARMS:
            end = "%s-%s" % (seed, arm)
            start = "initial-" + ("affine" if arm == "affine_sgd" else "wave")
            result[arm] = {g: averages(end, start, g) for g in ("all", "conflict", "congruent")}
        wave, fixed, affine = [result[a]["all"]["margin_delta"] for a in ("wave_sgd", "wave_fixed", "affine_sgd")]
        per_order.append(dict(seed=seed, arms=result, wave_minus_fixed=wave-fixed,
                              wave_minus_affine=wave-affine))
    return dict(per_order=per_order, pilot_positive=all(x["wave_minus_fixed"]>0 and
                x["wave_minus_affine"]>0 for x in per_order))
