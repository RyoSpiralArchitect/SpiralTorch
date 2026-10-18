#!/usr/bin/env python3
"""Training-only strength calibration through existing Rust/HF adapter paths."""

import argparse
import json
import math
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
POLICY = runpy.run_path(str(HERE / "protocol.py"))
OBS = runpy.run_path(str(HERE.parent / "2026-10-18-hf-geometry-observation/replay.py"))
BOOK = HERE.parent / "2026-10-18-hf-book-controls"
RADIUS = HERE.parent / "2026-10-18-hf-learned-radius"
PINNED, DIGEST, READ = OBS["load_pinned"], OBS["digest"], OBS["READ"]
SITES = POLICY["SITES"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-plan-id", required=True)
    parser.add_argument("--book-private", type=Path, required=True)
    parser.add_argument("--radius-runs", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new private directory")
    plan = PINNED(HERE / "plan.json", args.expected_plan_id)
    cases = POLICY["cases_for"](plan)
    book_results = PINNED(BOOK / "results.json", plan["book_results_id"])
    radius_results = PINNED(RADIUS / "results.json", plan["radius_results_id"])
    preflight = PINNED(args.book_private / "preflight-bound/preflight.json", plan["book_preflight_id"])
    runtime = PINNED(HERE.parent / "2026-10-18-hf-geometry-observation/execution.json",
                     plan["qualified_runtime_execution_id"])

    def inputs():
        return OBS["PREFLIGHT"]["inputs"](BOOK / "plan.json", plan["book_plan_id"],
                                           args.book_private / "chapters-balanced", plan["book_manifest_id"])

    book, _, _, texts, _ = inputs()
    require(book["sites"] == SITES, "forward layer order differs")
    ordinary = [row for row in book_results["runs"] if row["arm"] == "ordinary"]
    geometric = [row for row in radius_results["runs"] if row["phase"] == "full"]
    require(book_results["status"] == "twelve_runs_completed" and radius_results["status"] == "completed"
            and [r["seed"] for r in ordinary] == plan["seeds"]
            and [(r["arm"], r["seed"]) for r in geometric]
            == [(arm, seed) for seed in plan["seeds"] for arm in plan["geometry_arms"]],
            "source study cases differ")
    sources, paths = {}, {}
    for row in ordinary + geometric:
        key = (row["arm"], row["seed"])
        directory = (args.book_private / f"runs-fixed/ordinary-seed-{row['seed']}"
                     if row["arm"] == "ordinary" else args.radius_runs / f"{row['arm']}-seed-{row['seed']}-full")
        path = directory / "geometry-export/spiraltorch-geometry.json"
        PINNED(path, row["artifact_id"])
        sources[key], paths[key] = row, path

    import spiraltorch as st
    import torch
    import transformers as hf
    import accelerate

    package = Path(st.__file__).parent
    versions = {"python": sys.version.split()[0], "torch": torch.__version__,
                "transformers": hf.__version__, "accelerate": accelerate.__version__}
    require(versions == runtime["versions"] == preflight["versions"]
            and OBS["PREFLIGHT"]["package_identity"](package) == runtime["installed_package_ids"]
            and torch.backends.mps.is_available(), "runtime differs from qualified installed wheel")
    torch.set_num_threads(2)
    api = runpy.run_path(str(OBS["EXAMPLE"]))
    config, tokenizer, actual = api["local_runtime"](args.model_dir)
    base_id = actual["observed_identity_id"]
    require(base_id == preflight["runtime_identity"] and config.model_type == "gpt2"
            and config.n_layer == 12 and config.n_embd == 768, "model/tokenizer identity differs")
    datasets = {split: st.HfGeometryDataset.from_texts(tokenizer, rows, plan["context_length"])
                for split, rows in texts.items()}
    identity = st.hf_tokenized_dataset_identity_report(
        train_dataset=datasets["train"], eval_dataset=datasets["development"],
        expected_identity_id=preflight["tokenized_dataset_identity"])
    require(identity["status"] == "ready" and len(datasets["train"]) == plan["train_windows"]
            and len(datasets["development"]) == plan["development_windows"], "tokenized windows differ")
    indices = POLICY["calibration_indices"](len(datasets["train"]), plan["calibration_windows"])
    calibration = torch.utils.data.Subset(datasets["train"], indices)
    source_ids = {str(path.relative_to(ROOT)): DIGEST(path) for path in (
        Path(__file__), HERE / "protocol.py", HERE / "plan.json", BOOK / "preflight.py", BOOK / "prepare.py",
        HERE.parent / "2026-10-18-hf-geometry-observation/replay.py", OBS["EXAMPLE"], OBS["HELPER"])}
    args.output.mkdir(parents=True, exist_ok=False)
    execution = {"schema": "spiraltorch.hf_residual_calibration_execution.v1",
                 "status": "declared_before_measurement", "plan_id": args.expected_plan_id,
                 "sources": source_ids, "installed_package_ids": runtime["installed_package_ids"],
                 "versions": versions, "runtime_identity": base_id,
                 "dataset_identity": identity["observed_identity_id"], "calibration_train_indices": indices,
                 "cases": cases, "source_artifacts": [
                     {"arm": arm, "seed": seed, "artifact_id": row["artifact_id"],
                      "expected_loss": row["trained_geometry_eval_loss"]}
                     for (arm, seed), row in sources.items()],
                 "optimizer_updates": 0, "reserve_opened": False}
    (args.output / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")

    def unchanged():
        require(all(DIGEST(ROOT / name) == pin for name, pin in source_ids.items())
                and OBS["PREFLIGHT"]["package_identity"](package) == runtime["installed_package_ids"]
                and all(DIGEST(path) == sources[key]["artifact_id"] for key, path in paths.items()),
                "source, installed package or input artifact changed")
        inputs()

    config.use_cache = False
    model = hf.AutoModelForCausalLM.from_pretrained(
        args.model_dir, config=config, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32, attn_implementation="eager").requires_grad_(False).to("mps").eval()
    base_hash = api["parameter_hash"](model)
    require(all(row["base_parameter_hash"] == base_hash for row in sources.values()), "frozen base differs")
    references = {}

    def execute(case):
        unchanged()
        key = f"{case['arm']}-seed-{case['seed']}-{case['mode']}"
        print(json.dumps({"key": key, "status": "running"}), flush=True)
        destination = args.output / key
        destination.mkdir(exist_ok=False)
        matched = case["mode"] == "calibrated_ordinary"
        source_key = ("ordinary" if matched else case["arm"], case["seed"])
        source = sources[source_key]
        artifact = st.GeometryAdapterArtifact.from_json(
            READ(paths[source_key], 1_048_576).decode(), source["artifact_id"], base_id)
        geometry = st.GeometryAdapterStack.from_artifact(
            artifact, expected_base_model_identity=base_id, device="mps").eval()
        require(list(geometry.paths) == SITES, "adapter order differs")
        before = geometry.export_artifact(base_model_identity=base_id)
        before_state = json.loads(before.state_json())
        arguments = hf.TrainingArguments(
            output_dir=str(destination / "hf"), report_to=[], disable_tqdm=True,
            per_device_eval_batch_size=plan["eval_batch_size"], dataloader_pin_memory=False,
            seed=case["seed"])
        require(arguments.device.type == "mps", "HF selected another device")
        trainer = st.hf_geometry_trainer_class()(
            model=model, geometry=geometry, base_model_identity=base_id, args=arguments,
            data_collator=hf.default_data_collator, processing_class=tokenizer)

        def measure(label, dataset):
            state_id = geometry.export_artifact(base_model_identity=base_id).artifact_id
            calls = math.ceil(len(dataset) / plan["eval_batch_size"]) * len(SITES)
            with geometry.observe(max_events=calls, max_values=plan["max_values"]) as observed:
                loss = trainer.evaluate(eval_dataset=dataset)["eval_loss"]
            require(math.isfinite(loss) and state_id == geometry.export_artifact(
                base_model_identity=base_id).artifact_id, "evaluation changed state or returned nonfinite loss")
            trace = observed.to_dict()
            layers = POLICY["public_layers"](OBS["aggregate"](
                trace, SITES, native=source_key[0] != "ordinary", expected_calls=calls,
                expected_rows=len(dataset) * plan["context_length"]))
            trace_path = destination / f"{label}.json"
            trace_path.write_text(json.dumps(trace, indent=2, allow_nan=False) + "\n")
            return loss, {"stage": label, "layers": layers, "trace_id": DIGEST(trace_path),
                          "geometry_artifact_id": state_id}

        calibration_stages = []
        target = references[(case["arm"], case["seed"])] if matched else None
        if matched:
            for index, adapter in enumerate(geometry.adapters):
                _, stage = measure(f"calibration-before-layer-{index}", calibration)
                calibration_stages.append(stage)
                adapter.strength = POLICY["fitted_strength"](
                    adapter.strength, target["calibration"]["layers"][index]["relative_residual_rms"],
                    stage["layers"][index]["relative_residual_rms"])
        _, calibrated = measure("calibration-final", calibration)
        errors = POLICY["mismatch"](calibrated["layers"], target["calibration"]["layers"]) if matched else None
        if matched:
            require(all(abs(error["relative_rms_error"]) <= plan["calibration_relative_tolerance"]
                        for error in errors), "training-calibration RMS match failed; no retry")
        candidate = geometry.export_artifact(base_model_identity=base_id)
        final_state = json.loads(candidate.state_json())
        require(POLICY["without_strength"](before_state) == POLICY["without_strength"](final_state)
                and (matched or before.artifact_id == candidate.artifact_id), "learned state changed")
        (destination / "candidate.json").write_text(candidate.to_json())
        loss, development = measure("development", datasets["development"])
        if not matched:
            require(loss == source["trained_geometry_eval_loss"], "untouched development CE did not reproduce")
        # Restore the actual exported recipe, not a hand-built approximation.
        restored_artifact = st.GeometryAdapterArtifact.from_json(
            READ(destination / "candidate.json", 1_048_576).decode(), candidate.artifact_id, base_id)
        restored = st.GeometryAdapterStack.from_artifact(
            restored_artifact, expected_base_model_identity=base_id, device="mps").eval()
        sample = {name: value.to("mps") for name, value in hf.default_data_collator(
            [datasets["development"][0]]).items()}
        with torch.no_grad(), geometry.attach(model, require_all=True):
            expected = model(**sample, use_cache=False).logits
        with torch.no_grad(), restored.attach(model, require_all=True):
            actual = model(**sample, use_cache=False).logits
        require(torch.equal(expected, actual), "export/reload changed first-window logits")
        require(trainer.optimizer is None and trainer.lr_scheduler is None
                and all(p.grad is None for p in geometry.parameters())
                and all(p.grad is None and not p.requires_grad for p in model.parameters())
                and api["parameter_hash"](model) == base_hash
                and geometry.export_artifact(base_model_identity=base_id).artifact_id == candidate.artifact_id,
                "read-only evaluation mutated learned state")
        result = {**case, "source_artifact_id": source["artifact_id"], "candidate_artifact_id": candidate.artifact_id,
                  "target_artifact_id": sources[(case["arm"], case["seed"])]["artifact_id"] if matched else None,
                  "development_loss": loss, "reference_loss_exact": None if matched else True,
                  "calibration_stages": calibration_stages, "calibration": calibrated,
                  "calibration_mismatch": errors, "development": development,
                  "development_mismatch": POLICY["mismatch"](development["layers"], target["development"]["layers"])
                  if matched else None, "all_learned_tensors_unchanged": True,
                  "only_recipe_strengths_may_change": matched, "restored_first_window_logits_exact": True,
                  "base_parameter_hash": base_hash, "optimizer_updates": 0}
        (destination / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        unchanged()
        if not matched:
            references[(case["arm"], case["seed"])] = result
        print(json.dumps({"key": key, "status": "completed", "development_loss": loss}), flush=True)
        return {**result, "private_result_id": DIGEST(destination / "result.json")}

    def validate(records):
        unchanged()
        require(len(records) == plan["case_count"], "missing planned cases")

    records = runpy.run_path(str(OBS["HELPER"]))["execute_cases"](
        args.output / "run-manifest.json", cases, execute, validate)
    contrasts = [{"arm": row["arm"], "seed": row["seed"],
                  "geometry_minus_matched_ordinary_ce": references[(row["arm"], row["seed"])]["development_loss"]
                  - row["development_loss"],
                  "matched_minus_original_ordinary_ce": row["development_loss"]
                  - references[("ordinary", row["seed"])]["development_loss"]}
                 for row in records if row["mode"] == "calibrated_ordinary"]
    summary = {"schema": "spiraltorch.hf_residual_calibration_results.v1", "status": "completed",
               "plan_id": args.expected_plan_id, "execution_id": DIGEST(args.output / "execution.json"),
               "cases": [POLICY["public_record"](row) for row in records], "contrasts": contrasts,
               "private_manifest_id": DIGEST(args.output / "run-manifest.json"),
               "publication_projection": "calibration_and_matched_control_RMS_private_mismatch_bands_only",
               "optimizer_updates": 0, "reserve_opened": False,
               "quality_status": "posttraining_magnitude_diagnosis_not_matched_FT_or_causal_quality_evidence"}
    (args.output / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
