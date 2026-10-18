"""Validate complete aggregate-only records; never publish private resume payloads."""
import argparse
import hashlib
import math
from pathlib import Path
import re
import runpy
import statistics

Q = runpy.run_path(str(Path(__file__).with_name("experiment.py")))
digest, load, save = Q["digest"], Q["load"], Q["save"]


def fields(value, names):
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise ValueError("unexpected record fields")


def sha(value, prefix=""):
    if not isinstance(value, str) or not re.fullmatch(re.escape(prefix) + "[0-9a-f]{64}", value):
        raise ValueError("invalid identity")


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid numeric measurement")


def metric(row):
    finite(row["max_logit_abs"])
    if type(row["argmax_exact"]) is not bool or type(row["within_gate"]) is not bool:
        raise ValueError("invalid metric flags")
    if row["within_gate"] != (row["max_logit_abs"] <= Q["contract"]()["logit_atol"] and row["argmax_exact"]):
        raise ValueError("computed gate disagrees")


def timing(value):
    fields(value, "contract extension_sha256 torch transformers python platform torch_threads torch_interop_threads rows accuracy")
    if value["contract"] != Q["contract"]():
        raise ValueError("timing protocol changed")
    sha(value["extension_sha256"])
    for name in ("torch", "transformers", "python", "platform"):
        if not isinstance(value[name], str) or not 0 < len(value[name]) < 256:
            raise ValueError("metadata must be a bounded scalar string")
    if type(value["torch_threads"]) is not int or value["torch_threads"] != 2 or type(value["torch_interop_threads"]) is not int or value["torch_interop_threads"] != 1:
        raise ValueError("thread policy differs")
    expected = {(n, arm, i) for n in Q["LENGTHS"] for arm in Q["ARMS"]
                for i in range(Q["WARMUPS"] + Q["REPEATS"])}
    seen = set()
    for row in value["rows"]:
        fields(row, "length arm round warmup prefill_ms decode_ms total_ms max_logit_abs argmax_exact within_gate")
        if type(row["length"]) is not int or type(row["round"]) is not int or type(row["arm"]) is not str:
            raise ValueError("invalid timing condition")
        key = (row["length"], row["arm"], row["round"])
        if key not in expected or key in seen or row["warmup"] is not (row["round"] < Q["WARMUPS"]):
            raise ValueError("invalid or duplicate condition")
        seen.add(key)
        for name in ("prefill_ms", "decode_ms", "total_ms"):
            finite(row[name])
            if row[name] == 0:
                raise ValueError("zero duration")
        if not math.isclose(row["total_ms"], row["prefill_ms"] + row["decode_ms"], abs_tol=1e-6):
            raise ValueError("duration accounting mismatch")
        metric(row)
    if seen != expected:
        raise ValueError("missing conditions")
    seen = set()
    expected = {(n, arm, i) for n in Q["LENGTHS"] for arm in Q["ARMS"] for i in range(Q["APPENDS"]+1)}
    for row in value["accuracy"]:
        fields(row, "length arm step max_logit_abs argmax_exact within_gate")
        if type(row["length"]) is not int or type(row["step"]) is not int or type(row["arm"]) is not str:
            raise ValueError("invalid accuracy condition")
        key = (row["length"], row["arm"], row["step"])
        if key not in expected or key in seen:
            raise ValueError("invalid/duplicate accuracy prefix")
        seen.add(key)
        metric(row)
    if seen != expected:
        raise ValueError("missing accuracy prefix")


def public_runtime(value):
    # Free-form runtime metadata stays in hashed private records, never the public envelope.
    return {key: item for key, item in value.items() if key not in {"torch", "transformers", "python", "platform"}}


def validate_seals(before, after, baseline_seal, candidate_seal):
    expected = Q["contract"]()
    if any(value != expected for value in (before["contract"], after["contract"], baseline_seal, candidate_seal)):
        raise ValueError("saved protocol seals differ from records or current executable closure")


def summaries(times):
    result = []
    for length in Q["LENGTHS"]:
        for arm in Q["ARMS"]:
            samples = [r for r in times["rows"] if r["length"] == length and r["arm"] == arm and not r["warmup"]]
            accuracy = [r for r in times["accuracy"] if r["length"] == length and r["arm"] == arm]
            result.append({"length": length, "arm": arm,
                **{name: statistics.median(r[name] for r in samples) for name in ("prefill_ms", "decode_ms", "total_ms")},
                "max_logit_abs": max(r["max_logit_abs"] for r in [*samples, *accuracy]),
                "all_logit_gates_pass": all(r["within_gate"] for r in [*samples, *accuracy])})
    return result


def status(equal, cache_exact, resume_exact, rows):
    wide = all(r["within_gate"] for r in rows if not r["arm"].startswith("f32"))
    control = all(r["within_gate"] for r in rows if r["arm"].startswith("f32"))
    if not (equal and cache_exact and resume_exact and wide):
        return "UNQUALIFIED", control
    return ("SCOPED_READOUT_PASS" if control else "SCOPED_WIDE_READOUT_PASS_WITH_F32_FAILING_CONTROL"), control


def native(value, *, baseline=False):
    fields(value, "contract extension_sha256 python platform foreign_imports_forbidden rows" +
           ("" if baseline else " baseline_sha256 f32_bitwise_compatible"))
    if value["contract"] != Q["contract"]() or value["foreign_imports_forbidden"] is not True:
        raise ValueError("native protocol/import guard differs")
    sha(value["extension_sha256"])
    if not baseline:
        sha(value["baseline_sha256"])
        if type(value["f32_bitwise_compatible"]) is not bool:
            raise ValueError("invalid compatibility flag")
    for key in ("python", "platform"):
        if type(value[key]) is not str or not 0 < len(value[key]) < 256:
            raise ValueError("invalid native metadata")
    expected = {(policy, owner) for policy in (["f32"] if baseline else Q["POLICIES"]) for owner in ("base", "learned")}
    seen = set()
    for row in value["rows"]:
        fields(row, "policy owner projections generation" + (" training" if row.get("owner") == "learned" else ""))
        key = (row["policy"], row["owner"])
        if key not in expected or key in seen:
            raise ValueError("invalid native condition")
        seen.add(key)
        conditions = set()
        for p in row["projections"]:
            fields(p, "length step full_sha256 cache_sha256 max_logit_abs argmax_exact within_gate")
            if type(p["length"]) is not int or type(p["step"]) is not int:
                raise ValueError("invalid projection condition")
            condition = (p["length"], p["step"])
            if condition in conditions:
                raise ValueError("duplicate projection")
            conditions.add(condition)
            sha(p["full_sha256"])
            sha(p["cache_sha256"])
            metric(p)
        if conditions != {(n, i) for n in Q["LENGTHS"] for i in range(Q["APPENDS"]+1)}:
            raise ValueError("missing projection")
        recipes = []
        for g in row["generation"]:
            fields(g, "recipe steps_sha256 checkpoint_id tokens")
            recipes.append(g["recipe"])
            sha(g["steps_sha256"])
            sha(g["checkpoint_id"], "sha256:")
            if type(g["tokens"]) is not int or not 3 < g["tokens"] <= 8:
                raise ValueError("invalid continuation count")
        if sorted(recipes) != sorted(Q["OLD"]["CONFIGS"]):
            raise ValueError("missing/duplicate generation")
        if row["owner"] == "learned":
            t = row["training"]
            fields(t, "steps_sha256 checkpoint_id updates initial_loss second_loss")
            sha(t["steps_sha256"])
            sha(t["checkpoint_id"], "sha256:")
            if type(t["updates"]) is not int or t["updates"] != 2:
                raise ValueError("invalid update count")
            finite(t["initial_loss"])
            finite(t["second_loss"])
    if seen != expected:
        raise ValueError("missing native condition")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = load(args.baseline)
    after, resumed, times = [load(args.cohort / f) for f in ("native.json", "resume.json", "timing.json")]
    native(before, baseline=True)
    native(after)
    timing(times)
    validate_seals(before, after, load(args.baseline.parent / "contract.json"),
                   load(args.cohort / "contract.json"))
    equal = [r for r in after["rows"] if r["policy"] == "f32"] == before["rows"]
    if after["baseline_sha256"] != digest(args.baseline) or after["f32_bitwise_compatible"] is not equal:
        raise ValueError("baseline evidence changed or flag disagrees")
    fields(resumed, "contract input_sha256 extension_sha256 foreign_imports_forbidden rows")
    if resumed["contract"] != Q["contract"]() or resumed["foreign_imports_forbidden"] is not True:
        raise ValueError("resume protocol/import guard differs")
    if resumed["extension_sha256"] != after["extension_sha256"] or times["extension_sha256"] != after["extension_sha256"]:
        raise ValueError("mixed candidate binaries")
    source = args.cohort / "resume-inputs.json"
    if resumed["input_sha256"] != digest(source):
        raise ValueError("resume evidence is stale")
    raw = load(source)
    if raw["contract"] != Q["contract"]():
        raise ValueError("resume input protocol differs")
    expected = {("generation", p, owner, recipe) for p in Q["POLICIES"]
                for owner in ("base", "learned") for recipe in Q["OLD"]["CONFIGS"]}
    expected |= {("training", p, None, None) for p in Q["POLICIES"]}
    seen = set()
    for row in resumed["rows"]:
        fields(row, "kind policy owner recipe steps_exact checkpoint_exact")
        key = tuple(row[k] for k in ("kind", "policy", "owner", "recipe"))
        if key not in expected or key in seen or any(type(row[k]) is not bool for k in ("steps_exact", "checkpoint_exact")):
            raise ValueError("invalid resume condition")
        seen.add(key)
        cases = [c for c in raw["cases"] if tuple(c.get(k) for k in ("kind", "policy", "owner", "recipe")) == key]
        if len(cases) != 1:
            raise ValueError("missing/duplicate raw continuation")
        case = cases[0]
        owner = next(r for r in after["rows"] if r["policy"] == key[1] and r["owner"] == (key[2] or "learned"))
        final = next(g for g in owner["generation"] if g["recipe"] == key[3]) if key[0] == "generation" else owner["training"]
        if "sha256:" + hashlib.sha256(case["final"].encode()).hexdigest() != final["checkpoint_id"]:
            raise ValueError("continuation is for a different final checkpoint")
        if len(case["remaining_steps"]) != (final["tokens"] - 3 if key[0] == "generation" else 1):
            raise ValueError("wrong continuation length")
    if seen != expected or len(raw["cases"]) != len(expected):
        raise ValueError("incomplete resume evidence")
    medians = summaries(times)
    cache_exact = all(p["full_sha256"] == p["cache_sha256"] and p["within_gate"] for row in after["rows"] for p in row["projections"])
    resume_exact = all(r["steps_exact"] and r["checkpoint_exact"] for r in resumed["rows"])
    qualified, control = status(equal, cache_exact, resume_exact, [*times["rows"], *times["accuracy"]])
    save(args.output, {"schema": "spiraltorch.native_readout_comparison.v1", "contract": Q["contract"](),
        "publisher_sha256": digest(Path(__file__)),
        "status": qualified, "f32_control_all_gates_pass": control,
        "f32_bitwise_compatible": equal, "cache_full_exact": cache_exact, "resume_exact": resume_exact,
        "baseline": public_runtime(before), "native": public_runtime(after), "resume": resumed,
        "timing": public_runtime(times), "medians": medians,
        "raw_sha256": {name: digest(args.cohort / name) for name in ("contract.json", "native.json", "resume.json", "resume-inputs.json", "timing.json")},
        "baseline_sha256": digest(args.baseline),
        "baseline_contract_sha256": digest(args.baseline.parent / "contract.json")})


if __name__ == "__main__":
    main()
