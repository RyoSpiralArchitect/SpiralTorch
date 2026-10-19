"""Pinned native generation qualification; private raw outputs, no quality/speed claim."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import runpy

ROOT = Path(__file__).resolve().parents[3]
TRAINING = ROOT / "benchmarks/results/2026-10-18-native-corpus-training/qualify.py"
PRIOR = runpy.run_path(str(TRAINING))
CHECKPOINT = "1ed13924961efe03330d1e16407382e3fcb0b3255504939215215a9375f62e6e"
DATA = "d95c8c6161a6ad91b67be1d477890daf3adc15b90f804a83a1350b955bd40fee"
GEOMETRY = "sha256:429f2f2a85fd2e870e81ff23e1f294b24370100e0dd8adccf770e5f7aec3d4a2"
SOURCES = ["crates/st-core/src/inference/token_decoding.rs", "crates/st-core/src/inference/generation_control.rs",
           "crates/st-nn/src/causal_lm/generation.rs", "bindings/st-py/src/causal_lm/generation.rs",
           "bindings/st-py/examples/native_geometry_generation.py"]
CONFIGS = {
    "greedy": {"max_new_tokens": 8, "eos_token_ids": [50256]},
    "sample": {"max_new_tokens": 8, "eos_token_ids": [50256], "seed": 714, "decoding": {
        "selection": "sample", "top_k": 40, "top_p": .9,
        "distribution": {"kind": "softmax", "temperature": .8}}},
    "zspace": {"max_new_tokens": 8, "eos_token_ids": [50256], "seed": 714, "decoding": {
        "selection": "sample", "top_k": 40, "top_p": .9,
        "distribution": {"kind": "zspace", "config": {"curvature": -1., "temperature": .8,
            "repression_window": 32, "repression_strength": .6, "last_token_repression": .3,
            "ngram_size": 3, "ngram_repression_strength": .4}}}},
}
GATES = {"probability_atol": 2e-5, "probability_rtol": 2e-4, "entropy_atol": 2e-4,
         "greedy_ids_exact": True, "resume_steps_and_checkpoint_exact": True}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load(path):
    return json.loads(path.read_bytes())


def sources():
    return {str(p.relative_to(ROOT)): digest(p) for p in
            [Path(__file__).resolve(), TRAINING, *[ROOT / p for p in SOURCES]]}


def checked(args):
    PRIOR["check_model"](args.model_dir)
    if digest(args.data) != DATA or digest(args.checkpoint) != CHECKPOINT:
        raise ValueError("only the sealed earlier training data/checkpoint are admitted")
    if args.mode != "prepare":
        contract = load(args.cohort / "contract.json")
        if contract["source_sha256"] != sources() or contract["configs"] != CONFIGS or contract["gates"] != GATES:
            raise ValueError("source/config/gates changed after sealing")
    return load(args.data)["token_documents"][0]


def model(args, st):
    base = st.NativeCausalLm.from_gpt2((args.model_dir / "config.json").read_text(),
                                      st.SafeTensorArchive(args.model_dir / "model.safetensors"))
    if base.base_model_identity != PRIOR["BASE"]:
        raise ValueError("native base identity mismatch")
    learned = st.NativeCausalLm.from_training_checkpoint(base, args.checkpoint.read_text(), "sha256:" + CHECKPOINT)
    if learned.geometry_artifact_id != GEOMETRY:
        raise ValueError("learned geometry mismatch")
    return {"base": base, "learned": learned}


def prepare(args):
    PRIOR["forbid_ml"]()
    checked(args)
    args.cohort.mkdir(mode=0o700, parents=True, exist_ok=False)
    save(args.cohort / "contract.json", {"schema": "spiraltorch.native_generation_qualification.v1",
        "source_sha256": sources(), "configs": CONFIGS, "gates": GATES,
        "model_sha256": PRIOR["MODEL"], "checkpoint_sha256": CHECKPOINT, "data_sha256": DATA,
        "base_model_identity": PRIOR["BASE"], "learned_geometry_id": GEOMETRY,
        "prompt_offsets": [0, 128, 512], "prompt_length": 8, "resume_after_tokens": 3,
        "scope": "18 fixed train-chapter continuations; no held-out quality, speed, or Torch RNG claim"})


def native(args):
    PRIOR["forbid_ml"]()
    data = checked(args)
    import spiraltorch as st
    models = model(args, st)
    output = args.cohort / "native"
    output.mkdir(mode=0o700, exist_ok=False)
    for owner, active in models.items():
        for offset in [0, 128, 512]:
            prompt = data[offset:offset+8]
            for kind, config in CONFIGS.items():
                name = f"{owner}-{offset}-{kind}"
                generator = st.NativeCausalLmGenerator(active, prompt, json.dumps(config))
                steps, mid = [], None
                while not generator.finished:
                    steps.append(json.loads(generator.step().to_json()))
                    if len(steps) == 3:
                        mid = {"payload": generator.to_json(), "pin": generator.checkpoint_id}
                save(output / (name + ".json"), {"owner": owner, "offset": offset, "kind": kind,
                    "prompt": prompt, "steps": steps, "tokens": generator.token_ids,
                    "generated": generator.generated_ids, "mid": mid, "config": json.loads(generator.config_json()),
                    "payload": generator.to_json(), "pin": generator.checkpoint_id})
                print(name + " complete", flush=True)
    save(args.cohort / "native-runtime.json", {"python": platform.python_version(), "platform": platform.platform(),
        "extension_sha256": digest(Path(st._rs.__file__)), "optional_ml_imports_forbidden": True})


def resume(args):
    PRIOR["forbid_ml"]()
    checked(args)
    import spiraltorch as st
    models = model(args, st)
    rows = []
    for path in sorted((args.cohort / "native").glob("*.json")):
        case = load(path)
        if case["mid"] is None:
            raise ValueError("EOS before predeclared resume boundary; retain as unqualified")
        generator = st.NativeCausalLmGenerator.from_json(models[case["owner"]], case["mid"]["payload"], case["mid"]["pin"])
        steps = []
        while not generator.finished:
            steps.append(json.loads(generator.step().to_json()))
        rows.append({"case": path.stem, "steps_exact": steps == case["steps"][3:],
                     "checkpoint_exact": generator.to_json() == case["payload"], "continued_tokens": len(steps)})
    save(args.cohort / "resume.json", rows)
    if not all(row["steps_exact"] and row["checkpoint_exact"] for row in rows):
        raise ValueError("fresh-process continuation mismatch")


def reference(args):
    checked(args)
    import torch
    import transformers
    import spiraltorch as st
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    reference = transformers.GPT2LMHeadModel.from_pretrained(args.model_dir, local_files_only=True,
        dtype=torch.float32, attn_implementation="eager").eval().requires_grad_(False)
    saved = load(args.checkpoint)
    artifact = st.GeometryAdapterArtifact.from_json(saved["geometry_json"], GEOMETRY, PRIOR["BASE"])
    stack = st.GeometryAdapterStack.from_artifact(artifact, expected_base_model_identity=PRIOR["BASE"])
    attach = runpy.run_path(str(PRIOR["HELPER"]))["attach_reference"]
    from contextlib import nullcontext
    results = []
    for path in sorted((args.cohort / "native").glob("*.json")):
        case = load(path)
        if case["kind"] == "zspace":
            continue  # Shared Rust control is qualified separately, not an independent Torch oracle.
        rows = []
        history = case["prompt"][:]
        config = case["config"]["decoding"]
        for step in case["steps"]:
            with torch.no_grad(), (attach(stack, reference) if case["owner"] == "learned" else nullcontext()):
                logits = reference(torch.tensor([history]), use_cache=False).logits[0, -1].double()
            ids = torch.argsort(logits, descending=True, stable=True)[:config["top_k"]]
            probs = torch.softmax(logits[ids] / config["distribution"]["temperature"], dim=0)
            if config["top_p"] < 1.:
                keep = int(torch.searchsorted(probs.cumsum(0), torch.tensor(config["top_p"], dtype=torch.float64))) + 1
                ids, probs = ids[:keep], probs[:keep]
            positive = probs > 0
            ids, probs = ids[positive], probs[positive]
            probs /= probs.sum()
            expected = float(probs[ids.tolist().index(step["token_id"])]) if step["token_id"] in ids else 0.
            probability_error = abs(expected - step["probability"])
            entropy_error = abs(float(-(probs * probs.log()).sum()) - step["entropy"])
            greedy_exact = config["selection"] != "greedy" or int(ids[0]) == step["token_id"]
            rows.append({"step": step["step"], "probability_abs": probability_error, "expected_probability": expected,
                         "entropy_abs": entropy_error,
                         "greedy_exact": greedy_exact, "candidate_count_exact": len(ids) == step["candidates"],
                         "within_gate": probability_error <= GATES["probability_atol"] + GATES["probability_rtol"] * abs(expected)
                         and entropy_error <= GATES["entropy_atol"] and greedy_exact and len(ids) == step["candidates"]})
            history.append(step["token_id"])
        results.append({"case": path.stem, "steps": rows})
        print(path.stem + " compared", flush=True)
    save(args.cohort / "reference.json", {"torch": torch.__version__, "transformers": transformers.__version__, "cases": results})
    if not all(row["within_gate"] for case in results for row in case["steps"]):
        raise ValueError("ordinary decoder/probability comparison failed")


def validate_results(cases, resumed, reference):
    expected = {f"{owner}-{offset}-{kind}" for owner in ["base", "learned"] for offset in [0, 128, 512] for kind in CONFIGS}
    if set(cases) != expected or len(resumed) != len(expected) or {r["case"] for r in resumed} != expected:
        raise ValueError("incomplete native/resume case set")
    expected_reference = {name for name in expected if not name.endswith("zspace")}
    if len(reference["cases"]) != len(expected_reference) or {r["case"] for r in reference["cases"]} != expected_reference:
        raise ValueError("incomplete reference case set")
    for name, case in cases.items():
        n = len(case["generated"])
        if not 3 <= n <= 8 or len(case["steps"]) != n or case["tokens"] != case["prompt"] + case["generated"]:
            raise ValueError("invalid native sequence accounting")
        if n < 8 and case["generated"][-1] != 50256:
            raise ValueError("native sequence stopped without EOS or length")
        if any(r["step"] != i or r["token_id"] != case["generated"][i-1] for i, r in enumerate(case["steps"], 1)):
            raise ValueError("native step identity mismatch")
        if "sha256:" + hashlib.sha256(case["payload"].encode()).hexdigest() != case["pin"]:
            raise ValueError("native checkpoint pin mismatch")
    for row in resumed:
        if row["steps_exact"] is not True or row["checkpoint_exact"] is not True or row["continued_tokens"] != len(cases[row["case"]]["steps"]) - 3:
            raise ValueError("cannot publish an incomplete or failed resume")
    for case in reference["cases"]:
        if len(case["steps"]) != len(cases[case["case"]]["steps"]):
            raise ValueError("reference omitted steps")
        for i, row in enumerate(case["steps"], 1):
            numbers = [row[k] for k in ["probability_abs", "expected_probability", "entropy_abs"]]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in numbers):
                raise ValueError("invalid reference metrics")
            if row["step"] != i or row["greedy_exact"] is not True or row["candidate_count_exact"] is not True or row["within_gate"] is not True or not (
                    row["probability_abs"] <= GATES["probability_atol"] + GATES["probability_rtol"] * row["expected_probability"]
                    and row["expected_probability"] <= 1. and row["entropy_abs"] <= GATES["entropy_atol"]):
                raise ValueError("cannot publish a failed reference comparison")


def report(args):
    checked(args)
    cases = {path.stem: load(path) for path in sorted((args.cohort / "native").glob("*.json"))}
    resumed = load(args.cohort / "resume.json")
    reference = load(args.cohort / "reference.json")
    validate_results(cases, resumed, reference)
    rows = []
    for name, case in cases.items():
        rows.append({"case": name, "generated_tokens": len(case["generated"]),
            "mean_entropy": sum(r["entropy"] for r in case["steps"]) / len(case["steps"]),
            "control_top_changes": sum(r["raw_top_token"] != r["controlled_top_token"] for r in case["steps"]),
            "final_checkpoint_id": case["pin"], "raw_sha256": digest(args.cohort / "native" / (name + ".json"))})
    deltas = []
    for offset in [0, 128, 512]:
        for kind in CONFIGS:
            left, right = cases[f"base-{offset}-{kind}"]["generated"], cases[f"learned-{offset}-{kind}"]["generated"]
            deltas.append({"offset": offset, "kind": kind, "sequence_identical": left == right,
                "different_token_positions": sum(a != b for a, b in zip(left, right)) + abs(len(left) - len(right))})
    values = [row for case in reference["cases"] for row in case["steps"]]
    save(args.output, {"contract": load(args.cohort / "contract.json"), "cases": rows, "base_learned_deltas": deltas,
        "resume_cases": len(resumed), "resume_exact": True, "reference_cases": len(reference["cases"]),
        "reference_steps": len(values), "max_probability_abs": max(r["probability_abs"] for r in values),
        "max_entropy_abs": max(r["entropy_abs"] for r in values), "reference_all_pass": True,
        "runtime": load(args.cohort / "native-runtime.json"), "torch": reference["torch"], "transformers": reference["transformers"],
        "receipt_sha256": {name: digest(args.cohort / name) for name in ["contract.json", "resume.json", "reference.json"]}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "native", "resume", "reference", "report"])
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "report" and args.output is None:
        parser.error("report requires a new output file")
    globals()[args.mode](args)


if __name__ == "__main__":
    main()
