"""Prospective field-extraction control. Dataset construction never opens a model or scores."""
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "2026-11-03-native-field-extraction-sft" / "pilot.py"
PILOT_SHA = "30c0fe798951749b7e4925561b16555cb1ecbd9374dc27ae82da4178addbe383"
if hashlib.sha256(PILOT.read_bytes()).hexdigest() != PILOT_SHA:
    raise ValueError("the frozen predecessor changed")
spec = importlib.util.spec_from_file_location("field_pilot_v1", PILOT)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

ARMS = ("affine", "wave_fixed", "wave_learned")
ORDERS = ("order-109", "order-113", "order-127")
SPLITS = {"train": 96, "eval": 48}
CONTEXT, WIDTH, LAYER, RANK, ALPHA = 64, 768, 11, 2, 4.0
STRENGTH, LR, MAX_VALUES = 0.25, 0.01, 65536
WHEEL_SHA = "e3bbd8861b29294c5e3dc23e13c60e82ebc9648a4bbd244c5cfb5115b6b6867a"
NATIVE_SHA = "bd19a487e8acd986009d6054dc43b0de059765198d7d970308b141d371ffabce"
# A descriptive reporting threshold, not an equivalence test or a p-value.
MARGIN = math.log1p(.001)
PAIRS = {"fixed_minus_affine": (0, 1), "learned_minus_fixed": (1, 2)}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def pin(value):
    return old.digest(canonical(value))


def previous_keys():
    # Only record identities enter the exclusion set; never load old scores or checkpoints.
    return {r["key"] for rows in old.records().values() for r in rows}


def make_splits(excluded):
    used = set(excluded)
    pool = list(itertools.product(range(8), repeat=3))
    splits = {}
    for split, n in SPLITS.items():
        rows = []
        for f, field in enumerate(old.FIELDS):
            for value in range(8):
                for rep in range(n // 24):
                    possible = ("-".join(map(str, x)) for x in pool if x[f] == value)
                    key = min((k for k in possible if k not in used),
                              key=lambda k: pin(["field-control-v1", split, field, value, rep, k]))
                    used.add(key)
                    values = {name: old.VALUES[j][int(k)]
                              for j, (name, k) in enumerate(zip(old.FIELDS, key.split("-"), strict=True))}
                    position, template = (f + value + rep) % 3, (f + value + rep) % 2
                    fields = [name for name in old.FIELDS if name != field]
                    if template:
                        fields.reverse()
                    fields.insert(position, field)
                    if template:
                        prompt = "Record: " + "; ".join(k + " = " + values[k] for k in fields)
                        prompt += f".\nReturn the {field}.\nAnswer:"
                    else:
                        prompt = " | ".join(k + ": " + values[k] for k in fields)
                        prompt += f"\nWhat is the {field}?\nAnswer:"
                    rows.append(dict(key=key, prompt=prompt, answer=" " + values[field], field=field,
                                     values=values, position=position, template=template))
        splits[split] = sorted(rows, key=lambda r: r["key"])
    return splits


def train_order(rows, order):
    old.require(order in ORDERS, "unknown order; no post-hoc permutations")
    return sorted(range(len(rows)), key=lambda i: pin(["field-order-v1", order, rows[i]["key"]]))


def freeze():
    excluded = previous_keys()
    splits = make_splits(excluded)
    keys = set(excluded)
    stats, order_pins = {}, {}
    for split, rows in splits.items():
        n = SPLITS[split]
        counts = Counter((r["field"], r["answer"]) for r in rows)
        old.require(len(rows) == n and len(counts) == 24 and set(counts.values()) == {n // 24}
                    and Counter(r["position"] for r in rows) == {p: n // 3 for p in range(3)}
                    and Counter(r["template"] for r in rows) == {t: n // 2 for t in range(2)},
                    "field, target, template or position imbalance")
        for r in rows:
            old.require(r["key"] not in keys and r["answer"] == " " + r["values"][r["field"]],
                        "overlap or incorrect target")
            keys.add(r["key"])
        stats[split] = dict(records=n, text_sha256=pin(rows), key_sha256=pin([r["key"] for r in rows]))
    for order in ORDERS:
        order_pins[order] = pin([splits["train"][i]["key"] for i in train_order(splits["train"], order)])
    old.require(len(excluded) == 144 and len(keys) == 288 and len(set(order_pins.values())) == 3,
                "exclusion or order cardinality")
    return splits, dict(schema="spiraltorch.field_extraction_affine_controls.v1", predecessor=PILOT_SHA,
                        excluded_key_sha256=pin(sorted(excluded)), splits=stats, orders=order_pins,
                        arms=list(ARMS), trainable_counts=[4608, 4608, 4609], context=CONTEXT, layer=LAYER,
                        width=WIDTH, rank=RANK, alpha=ALPHA, strength=STRENGTH, lr=LR, updates=96,
                        endpoints=[0, 96], eval_batch=2, mode="joint", practical_nll_margin=MARGIN,
                        wheel_sha256=WHEEL_SHA, native_sha256=NATIVE_SHA, base_id=old.BASE_ID)


def geometry(st, base_id, arm, *, width=WIDTH, layer=LAYER):
    old.require(arm in ARMS, "unknown geometry arm")
    tensors = [dict(name=k, role="parameter", requires_grad=True, shape=[width], values=[0.] * width)
               for k in ("gate", "bias")]
    if arm == "affine":
        recipe = dict(schema="spiraltorch.channel_affine_adapter.v1", max_values=MAX_VALUES)
    else:
        learned = arm == "wave_learned"
        recipe = dict(schema="spiraltorch.wave_gate_adapter.v2", learnable_radius=learned,
                      kernel=dict(curvature=-1., saturation=1., porosity=.05, max_values=MAX_VALUES))
        tensors.append(dict(name="log_radius", role="parameter" if learned else "buffer",
                            requires_grad=learned, shape=[], values=[0.]))
    return st.GeometryAdapterArtifact.create_json(json.dumps(dict(
        schema="spiraltorch.geometry_adapter_artifact.v1", base_model_identity=base_id,
        policy="full_prefix_f32_no_cache.v1", training=True, placements=[dict(
            path=f"transformer.h.{layer}.attn", training=True, tensors=tensors,
            recipe=dict(recipe, features=width, strength=STRENGTH))])))


def parameter_state(artifact):
    tensors = json.loads(artifact.state_json())["placements"][0]["tensors"]
    return {t["name"]: t["values"] for t in tensors}, sum(len(t["values"]) for t in tensors
                                                          if t["requires_grad"] and t["role"] == "parameter")


def first_step(states):
    values = [parameter_state(a)[0] for a in states]
    old.require(set(values[0]) == {"gate", "bias"} and all(set(v) == {"gate", "bias", "log_radius"}
                and v["log_radius"] == [0.] for v in values[1:]), "radius moved at the origin")
    differences = {}
    for name in ("gate", "bias"):
        triples = list(zip(*(v[name] for v in values), strict=True))
        old.require(triples and any(x != 0. for x, _, _ in triples), "no first-step " + name)
        differences[name] = max(max(t) - min(t) for t in triples)
        old.require(all(math.isclose(a, b, abs_tol=2e-8, rel_tol=2e-6) and b == c for a, b, c in triples),
                    "different initial update for " + name)
    return differences


def classify(contrasts):
    old.require(len(contrasts) == len(ORDERS), "incomplete order replication")
    ds = [c["answer_weighted_delta_nll"] for c in contrasts]
    old.require(all(math.isfinite(d) for d in ds), "nonfinite primary result")
    counts = [c["candidate_all_correct_answers"] - c["reference_all_correct_answers"] for c in contrasts]
    if all(d < -MARGIN and n >= 0 for d, n in zip(ds, counts)):
        return "fixed_wave_advantage_on_this_split"
    if all(d > MARGIN and n <= 0 for d, n in zip(ds, counts)):
        return "affine_advantage_on_this_split"
    if all(abs(d) <= MARGIN for d in ds):
        return "inside_reporting_margin_not_equivalence"
    return "mixed_or_below_gate"


if __name__ == "__main__":
    print(json.dumps(freeze()[1], sort_keys=True, indent=2))
