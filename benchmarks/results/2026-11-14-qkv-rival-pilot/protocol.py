"""Freeze a fresh field-extraction comparison before any QKV model execution.

The two prior studies supply only excluded record identities. Their scores,
checkpoints and selected hyperparameters do not enter record selection.
"""

from collections import Counter
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
PREDECESSORS = {
    "field_pilot": (
        RESULTS / "2026-11-03-native-field-extraction-sft" / "pilot.py",
        "30c0fe798951749b7e4925561b16555cb1ecbd9374dc27ae82da4178addbe383",
    ),
    "affine_control": (
        RESULTS / "2026-11-03-native-wave-affine-extraction" / "protocol.py",
        "11d81ac0145ff488fcd6db4b5003177e251da86554341b870d07fe439efe3548",
    ),
}
SCHEMA = "spiraltorch.gpt2_qkv_rival_pilot.v1"
SOURCE_SHA256 = {
    "config.json": "0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4",
    "model.safetensors": "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
    "tokenizer.json": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
    "frozen_wave_lora-096.ck": "dc1dbd48507e505c179707edc8a1bf505fd7e90e8d2ae72ef0439e720574fd4c",
    "frozen_wave_lora-096-boundary.json": "174699cef128650b9d7502dd29b08b7236966b0677f3f736a29952d2538d43b4",
}
SPLIT_SIZES = {"train": 24, "eval": 48}
ORDERS = ("order-31", "order-47", "order-59")
RECIPE = {
    "arms": ["ordinary_ce", "top_rival_ce"],
    "layer": 11,
    "updates_per_arm_order": 24,
    "learning_rate": 0.0001,
    "weight_decay": 0.01,
    "max_gradient_norm": 1.0,
    "rivals": 4,
    "mix": 0.5,
    "label_smoothing": 0.0,
    "context_length": 64,
    "evaluation_microbatch_windows": 1,
    "readout_accumulator": "f32",
    "device": "cpu",
    "autotune": "disabled",
    "source": "GPT-2 small with frozen LoRA and WaveGate, before QKV updates",
    "optimizer": "fresh NativeCausalLmQkvAdamW per arm and order, default betas and epsilon",
    "stop_rule": "one attempt; stop on first input, execution, or receipt mismatch",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def pin(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _module(name, path, expected_sha):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha:
        raise ValueError(f"frozen predecessor changed: {name}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _predecessors():
    pilot = _module("qkv_rival_field_pilot", *PREDECESSORS["field_pilot"])
    affine = _module("qkv_rival_affine_control", *PREDECESSORS["affine_control"])
    pilot_keys = {row["key"] for rows in pilot.records().values() for row in rows}
    affine_splits = affine.make_splits(pilot_keys)
    affine_keys = {row["key"] for rows in affine_splits.values() for row in rows}
    if len(pilot_keys) != 144 or len(affine_keys) != 144 or pilot_keys & affine_keys:
        raise ValueError("predecessor exclusion inventory changed")
    return pilot, pilot_keys | affine_keys


def order_indices(rows, order):
    if order not in ORDERS:
        raise ValueError("unknown frozen training order")
    return sorted(range(len(rows)), key=lambda i: pin([SCHEMA, order, rows[i]["key"]]))


def freeze():
    pilot, excluded = _predecessors()
    fields, values = pilot.FIELDS, pilot.VALUES
    used = set(excluded)
    pool = list(itertools.product(range(8), repeat=3))
    splits = {}
    for split, size in SPLIT_SIZES.items():
        rows = []
        for field_index, field in enumerate(fields):
            for value_index in range(8):
                for repeat in range(size // 24):
                    candidates = (
                        combo for combo in pool
                        if combo[field_index] == value_index
                        and "-".join(map(str, combo)) not in used
                    )
                    triple = min(
                        candidates,
                        key=lambda combo: pin([
                            SCHEMA, split, field_index, value_index, repeat, combo
                        ]),
                    )
                    key = "-".join(map(str, triple))
                    used.add(key)
                    selected = {name: values[i][triple[i]] for i, name in enumerate(fields)}
                    position = (field_index + value_index + repeat) % 3
                    template = (field_index + value_index + repeat) % 2
                    arranged = [name for name in fields if name != field]
                    if template:
                        arranged.reverse()
                    arranged.insert(position, field)
                    if template:
                        prompt = "Record: " + "; ".join(
                            name + " = " + selected[name] for name in arranged
                        ) + f".\nReturn the {field}.\nAnswer:"
                    else:
                        prompt = " | ".join(
                            name + ": " + selected[name] for name in arranged
                        ) + f"\nWhat is the {field}?\nAnswer:"
                    rows.append({
                        "key": key, "field": field, "values": selected,
                        "position": position, "template": template,
                        "prompt": prompt, "answer": " " + selected[field],
                    })
        rows.sort(key=lambda row: row["key"])
        if len(rows) != size or len({row["key"] for row in rows}) != size:
            raise ValueError("split size or uniqueness changed")
        counts = Counter((row["field"], row["answer"]) for row in rows)
        if len(counts) != 24 or set(counts.values()) != {size // 24}:
            raise ValueError("field/answer balance changed")
        if Counter(row["position"] for row in rows) != {p: size // 3 for p in range(3)}:
            raise ValueError("field position balance changed")
        if Counter(row["template"] for row in rows) != {t: size // 2 for t in range(2)}:
            raise ValueError("template balance changed")
        splits[split] = rows
    if len(used) != 288 + sum(SPLIT_SIZES.values()):
        raise ValueError("fresh train/eval keys overlap frozen predecessors")
    orders = {
        order: pin([splits["train"][i]["key"] for i in order_indices(splits["train"], order)])
        for order in ORDERS
    }
    if len(set(orders.values())) != len(ORDERS):
        raise ValueError("frozen training orders are not distinct")
    protocol = {
        "schema": SCHEMA,
        "predecessor_sha256": {name: expected for name, (_, expected) in PREDECESSORS.items()},
        "source_sha256": SOURCE_SHA256,
        "excluded_keys_sha256": pin(sorted(excluded)),
        "splits": {
            split: {"rows": len(rows), "text_sha256": pin(rows),
                    "keys_sha256": pin([row["key"] for row in rows])}
            for split, rows in splits.items()
        },
        "orders_sha256": orders,
        "recipe": RECIPE,
        "primary": "rival minus ordinary CE: count of fresh evaluation answers whose every gold token is top-1",
        "secondary": ["target-weighted unsmoothed NLL", "answer-weighted NLL",
                      "selected-prefix top-1 target count", "per-answer win/tie/loss"],
        "interpretation_limit": "same synthetic task and frozen LoRA/WaveGate source; "
                                "order variants are not independent data replications; "
                                "top-1 teacher-forced correctness is not free generation; "
                                "no geometry-causality or general language-quality claim",
    }
    return splits, protocol


if __name__ == "__main__":
    splits, receipt = freeze()
    print(json.dumps({"protocol": receipt, "splits": splits}, sort_keys=True, indent=2))
