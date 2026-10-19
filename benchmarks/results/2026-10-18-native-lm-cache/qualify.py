"""Offline pinned KV qualification; Python 3.11+, raw material stays private."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import runpy
import statistics
import time

ROOT = Path(__file__).resolve().parents[3]
PREVIOUS = ROOT / "benchmarks/results/2026-10-18-native-lm-generation"
PRIOR = runpy.run_path(str(PREVIOUS / "qualify.py"))
BASE = PRIOR["PRIOR"]["BASE"]
OLD_REPORT = "f1e61ab2876dcb5ae05f8816d1d1c68760e4534cf39c6aa7bc50a1daacb0e480"
CONFIGS = copy.deepcopy(PRIOR["CONFIGS"])
for _config in CONFIGS.values():
    _config["execution"] = {"kind": "kv_cache", "max_values": 16_777_216}
GATES = {"logit_atol": 3e-4, "argmax_exact": True, "resume_exact": True, "old_checkpoint_exact": True}
LENGTHS = [8, 32, 64]
TIMING_LENGTHS = [16, 64]
ARMS = ["native_full", "native_cache", "torch_full", "torch_cache"]
APPENDS = 8
WARMUP = 2
REPEATS = 7
SOURCES = [
    "crates/st-nn/src/causal_lm.rs", "crates/st-nn/src/causal_lm/context.rs",
    "crates/st-nn/src/causal_lm/decoder.rs", "crates/st-nn/src/causal_lm/generation.rs",
    "crates/st-nn/src/geometry_runtime.rs", "bindings/st-py/src/causal_lm/context.rs",
    "crates/st-core/src/inference/token_decoding.rs",
]
digest, save, load = PRIOR["digest"], PRIOR["save"], PRIOR["load"]


def source_hashes():
    files = [Path(__file__).resolve(), PREVIOUS / "qualify.py", PRIOR["TRAINING"],
             PRIOR["PRIOR"]["HELPER"], *[ROOT / name for name in SOURCES]]
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def protocol():
    return {"schema": "spiraltorch.native_kv_qualification.v1", "source_sha256": source_hashes(),
        "configs": CONFIGS, "gates": GATES, "model_sha256": PRIOR["PRIOR"]["MODEL"],
        "checkpoint_sha256": PRIOR["CHECKPOINT"], "data_sha256": PRIOR["DATA"],
        "base_model_identity": BASE, "geometry_artifact_id": PRIOR["GEOMETRY"],
        "old_report_sha256": OLD_REPORT, "teacher_forcing_lengths": LENGTHS,
        "prompt_offsets": [0, 128, 512], "prompt_length": 8, "resume_after_tokens": 3,
        "timing": {"arms": ARMS, "lengths": TIMING_LENGTHS, "appends": APPENDS,
                   "warmups": WARMUP, "repetitions": REPEATS, "requested_threads": 2,
                   "order": "listed on even rounds, reversed on odd rounds",
                   "readout": "last row only, all arms; inputs prepared before timing"},
        "scope": "fixed train-chapter continuations; CPU f32 batch=1; no held-out quality or Torch RNG claim"}


def checked(args):
    PRIOR["PRIOR"]["check_model"](args.model_dir)
    if digest(args.data) != PRIOR["DATA"] or digest(args.checkpoint) != PRIOR["CHECKPOINT"]:
        raise ValueError("only the sealed earlier training data/checkpoint are admitted")
    if digest(PREVIOUS / "comparison.json") != OLD_REPORT:
        raise ValueError("old generation evidence changed")
    if args.mode != "prepare" and load(args.cohort / "contract.json") != protocol():
        raise ValueError("protocol or source changed after sealing")
    return load(args.data)["token_documents"][0]


def prepare(args):
    PRIOR["PRIOR"]["forbid_ml"]()
    checked(args)
    args.cohort.mkdir(mode=0o700, parents=True, exist_ok=False)
    save(args.cohort / "contract.json", protocol())


def metrics(actual, expected):
    if len(actual) != len(expected) or not actual or not all(math.isfinite(v) for v in [*actual, *expected]):
        raise ValueError("invalid logit vectors")
    error = max(abs(a-b) for a, b in zip(actual, expected))
    same = max(range(len(actual)), key=actual.__getitem__) == max(range(len(expected)), key=expected.__getitem__)
    return {"max_logit_abs": error, "argmax_exact": same, "within_gate": error <= GATES["logit_atol"] and same}


def native(args):
    PRIOR["PRIOR"]["forbid_ml"]()
    data = checked(args)
    import spiraltorch as st
    models = PRIOR["model"](args, st)
    output = args.cohort / "native"
    output.mkdir(mode=0o700, exist_ok=False)
    teacher = []
    for owner, model in models.items():
        for length in LENGTHS:
            history = data[:length]
            context = st.NativeCausalLmContext(model, history)
            rows = []
            for step in range(APPENDS + 1):
                if step:
                    context.append(data[length + step - 1])
                    history.append(data[length + step - 1])
                expected = model.forward(history, [1, len(history)], last_only=True).values()
                rows.append({"step": step, **metrics(context.logits().values(), expected),
                             "cache_values": context.cache_values})
            teacher.append({"owner": owner, "length": length, "steps": rows})
            print(f"{owner} teacher-forcing {length} complete", flush=True)
        for offset in [0, 128, 512]:
            for kind, config in CONFIGS.items():
                generator = st.NativeCausalLmGenerator(model, data[offset:offset+8], json.dumps(config))
                steps, mid = [], None
                while not generator.finished:
                    steps.append(json.loads(generator.step().to_json()))
                    if len(steps) == 3:
                        mid = {"payload": generator.to_json(), "pin": generator.checkpoint_id}
                name = f"{owner}-{offset}-{kind}"
                save(output / (name + ".json"), {"owner": owner, "offset": offset, "kind": kind,
                    "prompt": data[offset:offset+8], "steps": steps, "mid": mid,
                    "tokens": generator.token_ids, "generated": generator.generated_ids,
                    "payload": generator.to_json(), "pin": generator.checkpoint_id})
                print(name + " complete", flush=True)
    save(args.cohort / "teacher.json", teacher)
    save(args.cohort / "native-runtime.json", {"python": platform.python_version(), "platform": platform.platform(),
        "extension_sha256": digest(Path(st._rs.__file__)), "optional_ml_imports_forbidden": True})
    if not all(row["within_gate"] for case in teacher for row in case["steps"]):
        raise ValueError("cached/full-prefix logit gate failed; preserve this attempt")


def resume(args):
    PRIOR["PRIOR"]["forbid_ml"]()
    checked(args)
    import spiraltorch as st
    models = PRIOR["model"](args, st)
    old_hashes = {row["case"]: row["raw_sha256"] for row in load(PREVIOUS / "comparison.json")["cases"]}
    rows = []
    for kind, directory in [("cached", args.cohort), ("old_full_prefix", args.previous_cohort)]:
        for name in sorted(old_hashes):
            path = directory / "native" / (name + ".json")
            if kind == "old_full_prefix" and digest(path) != old_hashes[name]:
                raise ValueError("old raw checkpoint hash mismatch")
            case = load(path)
            if case["mid"] is None:
                raise ValueError("EOS before the sealed resume boundary")
            generator = st.NativeCausalLmGenerator.from_json(models[case["owner"]], case["mid"]["payload"], case["mid"]["pin"])
            before_exact = generator.to_json() == case["mid"]["payload"]
            steps = []
            while not generator.finished:
                steps.append(json.loads(generator.step().to_json()))
            rows.append({"mode": kind, "case": name, "before_exact": before_exact,
                "steps_exact": steps == case["steps"][3:], "checkpoint_exact": generator.to_json() == case["payload"],
                "continued_tokens": len(steps), "source_sha256": digest(path)})
        print(kind + " resumed", flush=True)
    save(args.cohort / "resume.json", rows)
    if not all(row["before_exact"] and row["steps_exact"] and row["checkpoint_exact"] for row in rows):
        raise ValueError("fresh-process/old checkpoint compatibility failure")


def benchmark(args):
    data = checked(args)
    import torch
    import transformers
    import spiraltorch as st
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    base = PRIOR["model"](args, st)["base"]
    reference = transformers.GPT2LMHeadModel.from_pretrained(args.model_dir, local_files_only=True,
        dtype=torch.float32, attn_implementation="eager").eval().requires_grad_(False)
    results = []
    for length in TIMING_LENGTHS:
        prefixes = [data[:length+i] for i in range(APPENDS+1)]
        full_inputs = [torch.tensor([ids]) for ids in prefixes]
        new_inputs = [torch.tensor([[data[length+i]]]) for i in range(APPENDS)]
        with torch.inference_mode():
            expected = reference.transformer(full_inputs[-1], use_cache=False)
            expected = reference.lm_head(expected.last_hidden_state[:, -1:]).flatten().tolist()
            for round_index in range(WARMUP + REPEATS):
                arms = ARMS if round_index % 2 == 0 else list(reversed(ARMS))
                for arm in arms:
                    # Release derived state outside each measured region. Torch's cache is mutable.
                    context = cache = logits = output = None
                    start = time.perf_counter_ns()
                    if arm == "native_cache":
                        context = st.NativeCausalLmContext(base, prefixes[0])
                        logits = context.logits()
                    elif arm == "native_full":
                        logits = base.forward(prefixes[0], [1, length], last_only=True)
                    else:
                        output = reference.transformer(full_inputs[0], use_cache=arm == "torch_cache")
                        cache = output.past_key_values
                        logits = reference.lm_head(output.last_hidden_state[:, -1:])
                    prefill_end = time.perf_counter_ns()
                    for i in range(APPENDS):
                        if arm == "native_cache":
                            logits = context.append(data[length+i])
                        elif arm == "native_full":
                            logits = base.forward(prefixes[i+1], [1, length+i+1], last_only=True)
                        elif arm == "torch_cache":
                            output = reference.transformer(new_inputs[i], past_key_values=cache, use_cache=True)
                            cache = output.past_key_values
                            logits = reference.lm_head(output.last_hidden_state[:, -1:])
                        else:
                            output = reference.transformer(full_inputs[i+1], use_cache=False)
                            logits = reference.lm_head(output.last_hidden_state[:, -1:])
                    end = time.perf_counter_ns()
                    values = logits.values() if arm.startswith("native") else logits.flatten().tolist()
                    results.append({"length": length, "arm": arm, "round": round_index,
                        "warmup": round_index < WARMUP, "prefill_ms": (prefill_end-start)/1e6,
                        "decode_ms": (end-prefill_end)/1e6, "total_ms": (end-start)/1e6,
                        **metrics(values, expected)})
                print(f"length={length} round={round_index} complete", flush=True)
    save(args.cohort / "timing.json", {"torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "extension_sha256": digest(Path(st._rs.__file__)), "rows": results})
    if not all(row["within_gate"] for row in results):
        raise ValueError("benchmark logit gate failed; retain timings as unqualified")


def exact_fields(value, fields):
    if not isinstance(value, dict) or set(value) != set(fields.split()):
        raise ValueError("unknown or missing receipt fields")


def validate_runtime(value, *, native=False):
    fields = "python platform extension_sha256 optional_ml_imports_forbidden" if native else (
        "torch transformers python platform torch_threads torch_interop_threads extension_sha256 rows")
    exact_fields(value, fields)
    for key in (["python", "platform"] if native else ["torch", "transformers", "python", "platform"]):
        if not isinstance(value[key], str) or not 0 < len(value[key]) <= 1024:
            raise ValueError("runtime metadata must be bounded scalar strings")
    pin = value["extension_sha256"]
    if not isinstance(pin, str) or len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin):
        raise ValueError("invalid extension digest")
    if native:
        if value["optional_ml_imports_forbidden"] is not True:
            raise ValueError("native import guard was not enabled")
    elif (type(value["torch_threads"]) is not int or value["torch_threads"] != 2
          or type(value["torch_interop_threads"]) is not int or value["torch_interop_threads"] != 1):
        raise ValueError("unexpected reference thread policy")


def publication_inputs(args):
    PRIOR["PRIOR"]["forbid_ml"]()
    data = checked(args)
    import spiraltorch as st
    models = PRIOR["model"](args, st)
    cases, hashes, identities = {}, {}, {}
    for path in (args.cohort / "native").glob("*.json"):
        raw = path.read_bytes()
        cases[path.stem] = json.loads(raw)
        hashes[path.stem] = hashlib.sha256(raw).hexdigest()
    for owner, model in models.items():
        for offset in [0, 128, 512]:
            for kind, config in CONFIGS.items():
                name = f"{owner}-{offset}-{kind}"
                prompt = data[offset:offset+8]
                fresh = st.NativeCausalLmGenerator(model, prompt, json.dumps(config))
                identities[name] = {"owner": owner, "offset": offset, "kind": kind, "prompt": prompt,
                    "config": json.loads(fresh.config_json()), "geometry": model.geometry_artifact_id}
                case = cases[name]
                # Rust remains the authority for checkpoint/config validity, not this publisher.
                for payload, pin in [(case["payload"], case["pin"]), (case["mid"]["payload"], case["mid"]["pin"])]:
                    restored = st.NativeCausalLmGenerator.from_json(model, payload, pin)
                    if restored.to_json() != payload or restored.config_json() != fresh.config_json():
                        raise ValueError("checkpoint/config does not match the declared native execution")
    old = {r["case"]: r for r in load(PREVIOUS / "comparison.json")["cases"]}
    return cases, identities, {"cached": hashes, "old_full_prefix": {name: r["raw_sha256"] for name, r in old.items()}}, {
        name: r["generated_tokens"] - 3 for name, r in old.items()}


def validate(cases, teacher, resumed, timing, identities, hashes, old_continued):
    names = {f"{owner}-{offset}-{kind}" for owner in ["base", "learned"] for offset in [0, 128, 512] for kind in CONFIGS}
    if set(cases) != names or set(identities) != names or any(set(hashes[m]) != names for m in ["cached", "old_full_prefix"]) or set(old_continued) != names:
        raise ValueError("incomplete native identities")
    if len(teacher) != 6 or {(r["owner"], r["length"]) for r in teacher} != {
            (owner, length) for owner in ["base", "learned"] for length in LENGTHS}:
        raise ValueError("incomplete native cases")
    for name, case in cases.items():
        exact_fields(case, "owner offset kind prompt steps mid tokens generated payload pin")
        expected = identities[name]
        if any(case[key] != expected[key] for key in ["owner", "offset", "kind", "prompt"]):
            raise ValueError("case identity mismatch")
        n = len(case["generated"])
        if not 3 <= n <= 8 or len(case["steps"]) != n or case["tokens"] != case["prompt"] + case["generated"]:
            raise ValueError("invalid native token accounting")
        if any(type(token) is not int or not 0 <= token < 50257 for token in case["tokens"]):
            raise ValueError("invalid token IDs")
        if n < 8 and case["generated"][-1] != 50256:
            raise ValueError("invalid early stop")
        if any(r["step"] != i or r["token_id"] != case["generated"][i-1] for i, r in enumerate(case["steps"], 1)):
            raise ValueError("invalid native step identity")
        exact_fields(case["mid"], "payload pin")
        for payload, pin, tokens in [(case["payload"], case["pin"], case["tokens"]),
                                     (case["mid"]["payload"], case["mid"]["pin"], case["prompt"] + case["generated"][:3])]:
            if pin != "sha256:" + hashlib.sha256(payload.encode()).hexdigest():
                raise ValueError("invalid checkpoint pin")
            state = json.loads(payload)
            exact_fields(state, "schema rng base_model_identity geometry_artifact_id config prompt_length tokens")
            if state != {"schema": "spiraltorch.native_causal_lm_generation.v1",
                         "rng": "rand_chacha_0_3_chacha20_seed_u64_top53.v1", "base_model_identity": BASE,
                         "geometry_artifact_id": expected["geometry"], "config": expected["config"],
                         "prompt_length": len(expected["prompt"]), "tokens": tokens}:
                raise ValueError("checkpoint does not represent this case or midpoint")
    if len(resumed) != 36 or {(r["mode"], r["case"]) for r in resumed} != {
            (mode, name) for mode in ["cached", "old_full_prefix"] for name in names}:
        raise ValueError("incomplete resumes")
    for row in resumed:
        exact_fields(row, "mode case before_exact steps_exact checkpoint_exact continued_tokens source_sha256")
        if row["source_sha256"] != hashes[row["mode"]][row["case"]]:
            raise ValueError("replay source hash mismatch")
        if any(row[k] is not True for k in ["before_exact", "steps_exact", "checkpoint_exact"]):
            raise ValueError("resume failed")
        if type(row["continued_tokens"]) is not int or not 0 <= row["continued_tokens"] <= 5:
            raise ValueError("invalid resumed step count")
        if row["mode"] == "cached" and row["continued_tokens"] != len(cases[row["case"]]["steps"]) - 3:
            raise ValueError("incomplete resumed sequence")
        if row["mode"] == "old_full_prefix" and row["continued_tokens"] != old_continued[row["case"]]:
            raise ValueError("incomplete old checkpoint continuation")
    validate_runtime(timing)
    rows = timing["rows"]
    expected = {(length, arm, r) for length in TIMING_LENGTHS for arm in ARMS for r in range(WARMUP+REPEATS)}
    if len(rows) != len(expected) or {(r["length"], r["arm"], r["round"]) for r in rows} != expected:
        raise ValueError("incomplete timing conditions")
    for row in rows:
        exact_fields(row, "length arm round warmup prefill_ms decode_ms total_ms max_logit_abs argmax_exact within_gate")
        if row["warmup"] is not (row["round"] < WARMUP):
            raise ValueError("warmup classification mismatch")
        values = [row[k] for k in ["prefill_ms", "decode_ms", "total_ms"]]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("invalid timing values")
        if abs(values[0] + values[1] - values[2]) > 1e-6:
            raise ValueError("timing accounting mismatch")
    for case in teacher:
        exact_fields(case, "owner length steps")
        if [r["step"] for r in case["steps"]] != list(range(APPENDS+1)):
            raise ValueError("incomplete teacher-forced steps")
        for row in case["steps"]:
            exact_fields(row, "step max_logit_abs argmax_exact within_gate cache_values")
            if type(row["cache_values"]) is not int or row["cache_values"] <= 0:
                raise ValueError("invalid cache accounting")
    failures = False
    for row in rows + [r for case in teacher for r in case["steps"]]:
        value = row["max_logit_abs"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid logit metric")
        if type(row["argmax_exact"]) is not bool or type(row["within_gate"]) is not bool:
            raise ValueError("invalid metric flags")
        passed = value <= GATES["logit_atol"] and row["argmax_exact"]
        if passed != row["within_gate"]:
            raise ValueError("metric/decision mismatch")
        failures |= not passed
    if failures:
        raise ValueError("logit gate failed")


def report(args):
    cases, identities, hashes, old_continued = publication_inputs(args)
    teacher, resumed, timing = [load(args.cohort / name) for name in ["teacher.json", "resume.json", "timing.json"]]
    validate(cases, teacher, resumed, timing, identities, hashes, old_continued)
    native_runtime = load(args.cohort / "native-runtime.json")
    validate_runtime(native_runtime, native=True)
    if timing["extension_sha256"] != native_runtime["extension_sha256"]:
        raise ValueError("native/reference extensions differ")
    summaries = []
    for length in TIMING_LENGTHS:
        for arm in ARMS:
            rows = [r for r in timing["rows"] if r["length"] == length and r["arm"] == arm and not r["warmup"]]
            summaries.append({"length": length, "arm": arm, "repetitions": len(rows),
                **{key: statistics.median(r[key] for r in rows) for key in ["prefill_ms", "decode_ms", "total_ms"]}})
    save(args.output, {"contract": load(args.cohort / "contract.json"), "teacher_forcing": teacher,
        "native_runtime": native_runtime, "timing": timing, "medians": summaries, "resume": resumed,
        "cases": [{"case": name, "generated_tokens": len(case["generated"]),
                   "final_checkpoint_id": case["pin"], "raw_sha256": hashes["cached"][name]}
                  for name, case in sorted(cases.items())],
        "raw_sha256": {name: digest(args.cohort / name) for name in ["contract.json", "teacher.json", "resume.json", "timing.json"]},
        "claim_ceiling": "CPU KV mechanics and last-row readout timing on one local GPT-2 train chapter; not quality, GPU, serving, or arbitrary-HF support"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "native", "resume", "benchmark", "report"])
    for name in ["model-dir", "data", "checkpoint", "cohort"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--previous-cohort", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "resume" and args.previous_cohort is None:
        parser.error("resume requires --previous-cohort")
    if args.mode == "report" and args.output is None:
        parser.error("report requires --output")
    globals()[args.mode](args)


if __name__ == "__main__":
    main()
