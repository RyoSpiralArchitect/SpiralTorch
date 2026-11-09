"""Model-free preparation of matched native-QKV length/position training."""
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

PARENT = "641689cc0a960fe058ab3541a91f1e557033d22c32edd7f602083afe5a035bf5"
DRIVER = "d6895e12017dfd75b4345beac239f389bc3b12262dac77dd70135793fbaaf310"
POLICY = {
    "kind": "spiraltorch.length_position_training.v1",
    "lanes": [[lag, view] for lag in (0, 1) for view in ("fixed", "varied")],
    "initialization": "Identical saved synthetic numerical weights; explicit lag ownership. One initialization, not pretrained GPT-2.",
    "train": "256 updates per lane, 96 answers/update, cap 4, lr .01, AdamW .9/.999 epsilon 1e-8, decay 0. Train block-0 QKV only. All other weights/LoRA/geometry frozen.",
    "order": "One Random(20261110) permutation of 32 old fit rows, shared by all lanes. Each update contains lengths 9,11,13 in that order. Varied gap=(permuted row index + zero-based update)%3; fixed gap=0.",
    "endpoint": "Score origin and final 256 only, no best-checkpoint selection. Save intermediate step 128 solely for one prescribed lag1-varied exact continuation witness (128 extra updates).",
    "evaluation": "Lengths 10/12, gaps 0/1/2, filler58 primary, filler59 secondary; 32 trained-pair and 96 untrained-pair views per cell. No overlap with training lengths. Associations are already consumed research data; no globally fresh holdout claim.",
    "primary": "On the six filler58 untrained-pair cells, lag1-varied must strictly improve aggregate unrestricted top1 over lag1-fixed with no worse aggregate gold NLL. Absolute competence separately requires >=87/96 in EVERY cell. Report both even on failure.",
    "controls": "Report all four lanes, both pair groups and all cells. Factorial interaction in correct counts is descriptive. No geometry attribution: frozen geometry is common. Filler59 cannot rescue the primary.",
    "budget": "CPU only; 1024 main updates plus 128 prescribed resume; 9216 origin/final question rows with full/cache logits, no retries. Training phase <=600s; scoring phase <=240s; stop at first error; disk floor 4GiB. No GPU, rebuild, publish or overwrite.",
    "status": "prepared_only_no_model_execution",
}


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def present(q, extra, gap, filler):
    require(len(q) == 9 and q[0] == 60 and q[3] == 46 and q[6] == 47
            and q[-1] == 23 and q[-2] in (q[1], q[4]) and {q[2], q[5]} == {24, 26},
            "invalid original question")
    require(type(extra) is int and 0 <= extra <= 4 and gap in (0, 1, 2)
            and filler in (58, 59), "invalid presentation")
    where = (1, 4, 7)[gap]
    return q[:where] + [filler] * extra + q[where:]


def gold(q):
    present(q, 0, 0, 58)
    return 16 if q[2 if q[1] == q[7] else 5] == 26 else 2


def questions(pairs):
    out = []
    for a, b in pairs:
        for order in ((a, b), (b, a)):
            for cue in (26, 24):
                assigned = {a: cue, b: 50 - cue}
                for query in (a, b):
                    out.append([60, order[0], assigned[order[0]], 46,
                                order[1], assigned[order[1]], 47, query, 23])
    return out


def decks(fit):
    require(len(fit) == 32 and len({tuple(q) for q in fit}) == 32, "wrong fit roster")
    for q in fit:
        gold(q)
    pairs = [(fit[i][1], fit[i][4]) for i in range(0, 32, 8)]
    require(questions(pairs) == fit and len(set(sum(([a, b] for a, b in pairs), []))) == 8,
            "not the four fixed identity pairs")
    other = [(a, b) for a, b in itertools.product([a for a, _ in pairs], [b for _, b in pairs])
             if (a, b) not in pairs]
    return {"trained_pairs": fit, "untrained_pairs": questions(other)}


def training(fit, view, update):
    require(view in ("fixed", "varied") and type(update) is int and 0 <= update < 256,
            "invalid training schedule")
    order = list(range(32))
    random.Random(20261110).shuffle(order)
    rows, labels = [], []
    for extra in (0, 2, 4):
        for i, index in enumerate(order):
            q = fit[index]
            rows.append(present(q, extra, 0 if view == "fixed" else (i + update) % 3, 58))
            labels.append(gold(q))
    return {"prompts": rows, "gold": labels, "microbatch": 4, "batch_kind": "unpadded_sequences"}


def evaluation(fit):
    return {f"f{filler}-n{extra}-g{gap}-{group}": {
        "prompts": [present(q, extra, gap, filler) for q in rows],
        "gold": [gold(q) for q in rows], "prefix_len": 7 + extra,
    } for filler, extra, gap in itertools.product((58, 59), (1, 3), range(3))
        for group, rows in decks(fit).items()}


def freeze(parent, driver, destination):
    protocol = parent / "protocol"
    require(sha(protocol / "plan.json") == PARENT and sha(driver) == DRIVER,
            "wrong source model or executable")
    p = json.loads((protocol / "plan.json").read_bytes())
    files = {str(protocol / "plan.json"): PARENT, str(driver.resolve()): DRIVER,
             str(protocol / "fit.json"): p["fit"],
             str(protocol / "model.safetensors"): p["weights"], str(Path(__file__).resolve()): sha(__file__)}
    for lag in (0, 1):
        for filename, key in (("config.json", "config"), ("lora.json", "lo"), ("geometry.json", "ge")):
            files[str(protocol / f"lag{lag}" / filename)] = p["routes"][str(lag)][key]
    require(all(sha(path) == pin for path, pin in files.items()), "changed saved input")
    fit = json.loads((protocol / "fit.json").read_bytes())
    decks(fit)
    payload = dict(policy=POLICY, inputs=files, routes={str(k): p["routes"][str(k)] for k in (0, 1)},
                   training={view: [training(fit, view, step) for step in range(256)]
                             for view in ("fixed", "varied")}, evaluation=evaluation(fit))
    # Exclusive creation: a later amendment must preserve this sealed protocol.
    with destination.open("x") as f:
        json.dump(payload, f, sort_keys=True, separators=(",", ":"), allow_nan=False)
        f.write("\n")
    return sha(destination)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("PARENT_DIRECTORY SAVED_EXECUTABLE NEW_PLAN_JSON")
    print(freeze(*(Path(x).resolve() for x in sys.argv[1:])))
