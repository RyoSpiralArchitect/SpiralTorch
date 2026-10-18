"""Offline pinned readout experiment. Outputs/checkpoints are private until reviewed."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import runpy
import struct
import time

ROOT = Path(__file__).resolve().parents[3]
OLD = runpy.run_path(str(ROOT / "benchmarks/results/2026-10-18-native-lm-cache/qualify.py"))
PRIOR = OLD["PRIOR"]
digest, load, save = OLD["digest"], OLD["load"], OLD["save"]
LENGTHS, APPENDS, WARMUPS, REPEATS = [16, 64], 8, 2, 7
ARMS = ["f32_full", "f32_cache", "f64_full", "f64_cache", "torch_full", "torch_cache"]
POLICIES = ["f32", "f64"]


def contract():
    return {"schema": "spiraltorch.native_readout_experiment.v1",
        "script_sha256": digest(Path(__file__)), "model_sha256": PRIOR["PRIOR"]["MODEL"],
        "data_sha256": PRIOR["DATA"], "checkpoint_sha256": PRIOR["CHECKPOINT"],
        "base_model_identity": OLD["BASE"], "geometry_artifact_id": PRIOR["GEOMETRY"],
        "logit_atol": OLD["GATES"]["logit_atol"], "argmax_exact": True,
        "lengths": LENGTHS, "appends": APPENDS, "warmups": WARMUPS, "repeats": REPEATS,
        "arms": ARMS, "timing_order": "listed on even rounds, reversed on odd rounds",
        "timing_scope": "CPU batch1, last-row readout; precreated input IDs; model loading excluded",
        "native_scope": "base/learned, two lengths, nine full/cache comparisons; three decoding recipes; two training updates",
        "claim_ceiling": "one pinned train chapter; not model quality, arbitrary HF architectures, GPU or full-weight FT"}


def checked(args):
    PRIOR["PRIOR"]["check_model"](args.model_dir)
    if digest(args.data) != PRIOR["DATA"] or digest(args.checkpoint) != PRIOR["CHECKPOINT"]:
        raise ValueError("only previously admitted data/checkpoint are permitted")
    return load(args.data)["token_documents"][0]


def hashed(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def vector_hash(values):
    return hashlib.sha256(struct.pack("<" + "f" * len(values), *values)).hexdigest()


def native(args):
    PRIOR["PRIOR"]["forbid_ml"]()
    data = checked(args)
    import spiraltorch as st
    models = PRIOR["model"](args, st)
    policies = ["f32"] if args.mode == "baseline" else POLICIES
    rows, resumes = [], []
    for policy in policies:
        for owner, original in models.items():
            model = original if policy == "f32" else original.with_readout_accumulator(policy)
            projections, generation = [], []
            for length in LENGTHS:
                ids = data[:length]
                context = st.NativeCausalLmContext(model, ids)
                for step in range(APPENDS + 1):
                    if step:
                        ids.append(data[length + step - 1])
                        context.append(ids[-1])
                    full = model.forward(ids, [1, len(ids)], last_only=True).values()
                    cache = context.logits().values()
                    projections.append({"length": length, "step": step,
                        "full_sha256": vector_hash(full), "cache_sha256": vector_hash(cache),
                        **OLD["metrics"](cache, full)})
            for name, config in OLD["CONFIGS"].items():
                g = st.NativeCausalLmGenerator(model, data[:8], json.dumps(config))
                steps, mid, pin = [], None, None
                while not g.finished:
                    steps.append(g.step().to_json())
                    if len(steps) == 3:
                        mid, pin = g.to_json(), g.checkpoint_id
                if mid is None:
                    raise ValueError("generation ended before continuation boundary")
                generation.append({"recipe": name, "steps_sha256": hashed(steps),
                    "checkpoint_id": g.checkpoint_id, "tokens": len(steps)})
                resumes.append({"kind": "generation", "policy": policy, "owner": owner, "recipe": name,
                    "payload": mid, "pin": pin, "remaining_steps": steps[3:], "final": g.to_json()})
            row = {"policy": policy, "owner": owner, "projections": projections, "generation": generation}
            if owner == "learned":
                trainer = st.NativeCausalLmTrainer(model, st.NativeGeometryAdamW(model.geometry_artifact()),
                    [data[:33]], 16, learning_rate=.0001, microbatch_windows=1)
                first = trainer.step().to_json()
                mid, pin = trainer.to_json(), trainer.checkpoint_id
                second = trainer.step().to_json()
                if not trainer.exhausted:
                    raise ValueError("unexpected training update count")
                row["training"] = {"steps_sha256": hashed([first, second]),
                    "checkpoint_id": trainer.checkpoint_id, "updates": 2,
                    "initial_loss": json.loads(first)["loss"], "second_loss": json.loads(second)["loss"]}
                resumes.append({"kind": "training", "policy": policy, "payload": mid,
                    "pin": pin, "remaining_steps": [second], "final": trainer.to_json()})
            rows.append(row)
            print(f"{policy}/{owner} native checks complete", flush=True)
    result = {"contract": contract(), "extension_sha256": digest(Path(st._rs.__file__)),
        "python": platform.python_version(), "platform": platform.platform(),
        "foreign_imports_forbidden": True, "rows": rows}
    if args.mode != "baseline":
        baseline = load(args.baseline)
        if baseline["contract"] != contract():
            raise ValueError("baseline protocol differs")
        result["baseline_sha256"] = digest(args.baseline)
        result["f32_bitwise_compatible"] = rows[:2] == baseline["rows"]
    save(args.output / "native.json", result)
    save(args.output / "resume-inputs.json", {"contract": contract(), "cases": resumes})
    if not all(r["within_gate"] and r["full_sha256"] == r["cache_sha256"]
               for row in rows for r in row["projections"]):
        raise ValueError("native cache/full exactness gate failed; retained")
    if args.mode != "baseline" and not result["f32_bitwise_compatible"]:
        raise ValueError("default f32 behavior changed; retained")


def resume(args):
    PRIOR["PRIOR"]["forbid_ml"]()
    data = checked(args)
    import spiraltorch as st
    models = PRIOR["model"](args, st)
    source = args.output / "resume-inputs.json"
    record = load(source)
    if record["contract"] != contract():
        raise ValueError("resume protocol differs")
    rows = []
    for case in record["cases"]:
        policy = case["policy"]
        if case["kind"] == "generation":
            model = models[case["owner"]].with_readout_accumulator(policy)
            session = st.NativeCausalLmGenerator.from_json(model, case["payload"], case["pin"])
        else:
            model = models["base"].with_readout_accumulator(policy)
            session = st.NativeCausalLmTrainer.from_json(model, [data[:33]], case["payload"], case["pin"])
        steps = []
        while (step := session.step()) is not None:
            steps.append(step.to_json())
        rows.append({"kind": case["kind"], "policy": policy, "owner": case.get("owner"),
            "recipe": case.get("recipe"), "steps_exact": steps == case["remaining_steps"],
            "checkpoint_exact": session.to_json() == case["final"]})
    save(args.output / "resume.json", {"contract": contract(), "input_sha256": digest(source),
        "extension_sha256": digest(Path(st._rs.__file__)), "foreign_imports_forbidden": True, "rows": rows})
    if not all(r["steps_exact"] and r["checkpoint_exact"] for r in rows):
        raise ValueError("fresh-process native continuation failed; retained")


def benchmark(args):
    data = checked(args)
    import torch
    import transformers
    import spiraltorch as st
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    base = PRIOR["model"](args, st)["base"]
    models = {p: base.with_readout_accumulator(p) for p in POLICIES}
    reference = transformers.GPT2LMHeadModel.from_pretrained(args.model_dir, local_files_only=True,
        dtype=torch.float32, attn_implementation="eager").eval().requires_grad_(False)
    rows = []
    with torch.inference_mode():
        for length in LENGTHS:
            prefixes = [data[:length+i] for i in range(APPENDS+1)]
            inputs = [torch.tensor([ids]) for ids in prefixes]
            singles = [torch.tensor([[data[length+i]]]) for i in range(APPENDS)]
            hidden = reference.transformer(inputs[-1], use_cache=False).last_hidden_state
            expected = reference.lm_head(hidden[:, -1:]).flatten().tolist()
            for repeat in range(WARMUPS + REPEATS):
                for arm in ARMS if repeat % 2 == 0 else reversed(ARMS):
                    context = cache = logits = output = None
                    native_arm = not arm.startswith("torch")
                    cached = arm.endswith("cache")
                    model = models[arm[:3]] if native_arm else None
                    start = time.perf_counter_ns()
                    if native_arm:
                        if cached:
                            context = st.NativeCausalLmContext(model, prefixes[0])
                            logits = context.logits()
                        else:
                            logits = model.forward(prefixes[0], [1, length], last_only=True)
                    else:
                        output = reference.transformer(inputs[0], use_cache=cached)
                        cache = output.past_key_values
                        logits = reference.lm_head(output.last_hidden_state[:, -1:])
                    prefill = time.perf_counter_ns()
                    for i in range(APPENDS):
                        if native_arm:
                            logits = context.append(data[length+i]) if cached else model.forward(
                                prefixes[i+1], [1, length+i+1], last_only=True)
                        else:
                            output = reference.transformer(singles[i] if cached else inputs[i+1],
                                past_key_values=cache if cached else None, use_cache=cached)
                            cache = output.past_key_values
                            logits = reference.lm_head(output.last_hidden_state[:, -1:])
                    end = time.perf_counter_ns()
                    values = logits.values() if native_arm else logits.flatten().tolist()
                    rows.append({"length": length, "arm": arm, "round": repeat, "warmup": repeat < WARMUPS,
                        "prefill_ms": (prefill-start)/1e6, "decode_ms": (end-prefill)/1e6,
                        "total_ms": (end-start)/1e6, **OLD["metrics"](values, expected)})
                print(f"length={length} round={repeat} complete", flush=True)
    save(args.output / "timing.json", {"contract": contract(),
        "extension_sha256": digest(Path(st._rs.__file__)), "torch": torch.__version__,
        "transformers": transformers.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(), "rows": rows})
    # The unchanged f32 path is a deliberately retained failing control.
    if not all(r["within_gate"] for r in rows if not r["arm"].startswith("f32")):
        raise ValueError("wide/torch logit gate failed; retain all timings as unqualified")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=["baseline", "native", "resume", "benchmark"])
    for name in ["model-dir", "data", "checkpoint", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    args = p.parse_args()
    if args.mode == "native" and args.baseline is None:
        p.error("native phase requires the old binary baseline")
    if args.mode in {"baseline", "native"}:
        args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        native(args)
    else:
        (resume if args.mode == "resume" else benchmark)(args)


if __name__ == "__main__":
    main()
