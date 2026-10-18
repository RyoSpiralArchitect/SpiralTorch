#!/usr/bin/env python3
"""Read-only saved-state diagnosis through the public HF evaluation loop."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BOOK = HERE.parent / "2026-10-18-hf-book-controls"
PREFLIGHT = runpy.run_path(str(BOOK / "preflight.py"))
READ = runpy.run_path(str(BOOK / "prepare.py"))["read_bounded"]
EXAMPLE = ROOT / "bindings/st-py/examples/hf_geometry_finetune.py"
HELPER = HERE.parent / "2026-10-18-hf-affine-control/replay.py"
ARMS = ["ordinary", "wave_gate", "wave_gate_radius1", "wave_gate_radius4"]


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_pinned(path, pin):
    raw = READ(path, 8_388_608)
    if "sha256:" + hashlib.sha256(raw).hexdigest() != pin:
        raise ValueError("input does not match independently retained identity")
    return json.loads(raw)


def wheel_package_identity(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            parts = member.filename.split("/")
            if (len(parts) != 2 or parts[0] != "spiraltorch"
                    or not parts[1].endswith((".py", ".so"))):
                continue
            if parts[1] in result:
                raise ValueError("duplicate package member in wheel")
            sha = hashlib.sha256()
            with archive.open(member) as stream:
                for block in iter(lambda: stream.read(1_048_576), b""):
                    sha.update(block)
            result[parts[1]] = "sha256:" + sha.hexdigest()
    return result


def cases_for(results, saved, preflight):
    expected = [(arm, seed) for seed in (41, 43, 47) for arm in ARMS]
    if (results["status"] != "twelve_runs_completed"
            or [(r["arm"], r["seed"]) for r in results["runs"]] != expected
            or saved["status"] != "passed"
            or [(r["arm"], r["seed"], r["step"]) for r in saved["checkpoints"]]
            != [(a, s, step) for a, s in expected for step in (256, 512)]):
        raise ValueError("diagnosis requires all completed book cases in declared order")
    cases = [{"arm": arm, "seed": None, "step": 0,
              "artifact_id": preflight["artifacts"][arm]["artifact_id"],
              "expected_loss": results["runs"][0]["initial_geometry_eval_loss"]}
             for arm in ARMS]
    for record, checkpoint in zip(results["runs"], saved["checkpoints"][::2]):
        for step in (256, 512):
            pin = (checkpoint["private_file_ids"]["spiraltorch-geometry.json"]
                   if step == 256 else record["artifact_id"])
            cases.append({"arm": record["arm"], "seed": record["seed"], "step": step,
                          "artifact_id": pin,
                          "expected_loss": record["trained_geometry_eval_loss"] if step == 512 else None})
    return cases


def aggregate(trace, sites, *, native, expected_calls=320, expected_rows=20352):
    if (trace["schema"] != "spiraltorch.geometry_observation.v1"
            or trace["coverage_complete"] is not True
            or trace["calls_seen"] != expected_calls
            or len(trace["events"]) != expected_calls
            or {e["path"] for e in trace["events"]} != set(sites)):
        raise ValueError("incomplete observed coverage")
    output = []
    for path in sites:
        events = [e for e in trace["events"] if e["path"] == path]
        if (len(events) * len(sites) != expected_calls
                or any(e["training"] or e["grad_enabled"] or e["status"] != "observed"
                       or not e["residual"]["finite"] or e["shape"][-1] != 768
                       or e["device"] not in ("mps", "mps:0") or e["native_reports_omitted"]
                       or len(e["native_reports"]) != int(native) for e in events)):
            raise ValueError("observation is not the complete declared evaluation")
        rows = sum(math.prod(e["shape"][:-1]) for e in events)
        if rows != expected_rows:
            raise ValueError("evaluation row coverage differs")
        input_sq = math.fsum(e["residual"]["input_l2"] ** 2 for e in events)
        delta_sq = math.fsum(e["residual"]["residual_l2"] ** 2 for e in events)
        item = {"path": path, "calls": len(events), "rows": rows,
                "relative_residual_energy": math.sqrt(delta_sq / input_sq) if input_sq else None,
                "residual_max_abs": max(e["residual"]["residual_max_abs"] for e in events),
                "native_conditioning": None}
        if native:
            reports = [e["native_reports"][0] for e in events]
            if any(r["rows"] != math.prod(e["shape"][:-1]) or r["features"] != 768
                   or r["affine_nonfinite_values"] for e, r in zip(events, reports)):
                raise ValueError("native report dimensions or finiteness differ")
            means = ("affine_abs_slope_mean", "dimensionless_norm_mean",
                     "relative_radial_gain_mean", "relative_tangential_gain_mean")
            stats = {k: math.fsum(r[k] * r["rows"] for r in reports) / rows for k in means}
            stats.update({k: min(r[k] for r in reports) for k in (
                "relative_radial_gain_min", "relative_tangential_gain_min")})
            stats.update(dimensionless_norm_max=max(r["dimensionless_norm_max"] for r in reports),
                         affine_outside_fraction=sum(r["affine_outside_values"] for r in reports) / (rows * 768),
                         gate_outside_values_max=max(r["gate_outside_values"] for r in reports))
            for key in ("schema", "projection_radius", "log_radius"):
                values = {r.get(key) for r in reports}
                if len(values) != 1:
                    raise ValueError("native conditioning recipe changed during evaluation")
                stats[key] = values.pop()
            item["native_conditioning"] = stats
        output.append(item)
    json.dumps(output, allow_nan=False)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-plan-id", required=True)
    parser.add_argument("--book-private", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new private directory")
    plan = load_pinned(HERE / "plan.json", args.expected_plan_id)
    results = load_pinned(BOOK / "results.json", plan["book_results_id"])
    saved = load_pinned(BOOK / "saved-state-check.json", plan["book_saved_state_check_id"])
    preflight = load_pinned(args.book_private / "preflight-bound/preflight.json", plan["book_preflight_id"])
    book_plan, _, _, texts, _ = PREFLIGHT["inputs"](
        BOOK / "plan.json", plan["book_plan_id"], args.book_private / "chapters-balanced", plan["book_manifest_id"])
    cases = cases_for(results, saved, preflight)
    if len(cases) != plan["case_count"]:
        raise ValueError("case count differs from declaration")
    paths = {}
    for case in cases:
        key = f"{case['arm']}-seed-{case['seed']}-step-{case['step']}"
        if case["step"] == 0:
            path = args.book_private / f"preflight-bound/{case['arm']}-initial.json"
        else:
            directory = args.book_private / f"runs-fixed/{case['arm']}-seed-{case['seed']}"
            path = directory / ("checkpoint-256" if case["step"] == 256 else "geometry-export") / "spiraltorch-geometry.json"
        load_pinned(path, case["artifact_id"])
        paths[key] = path
        case["key"] = key

    import spiraltorch as st
    import torch
    import transformers as hf
    import accelerate

    package = Path(st.__file__).parent
    package_ids = PREFLIGHT["package_identity"](package)
    if package_ids != wheel_package_identity(args.wheel):
        raise ValueError("installed package bytes do not match the supplied wheel")
    versions = {"python": sys.version.split()[0], "torch": torch.__version__,
                "transformers": hf.__version__, "accelerate": accelerate.__version__}
    if versions != preflight["versions"] or not torch.backends.mps.is_available():
        raise ValueError("runtime versions or device differ from book run")
    torch.set_num_threads(2)
    api = runpy.run_path(str(EXAMPLE))
    config, tokenizer, runtime = api["local_runtime"](args.model_dir)
    base_id = runtime["observed_identity_id"]
    if base_id != preflight["runtime_identity"]:
        raise ValueError("frozen model/tokenizer identity changed")
    data = {split: st.HfGeometryDataset.from_texts(tokenizer, rows, plan["context_length"])
            for split, rows in texts.items()}
    data_id = st.hf_tokenized_dataset_identity_report(
        train_dataset=data["train"], eval_dataset=data["development"],
        expected_identity_id=preflight["tokenized_dataset_identity"])
    if data_id["status"] != "ready" or len(data["development"]) != 159:
        raise ValueError("development tokenization/window identity changed")
    source_ids = {str(p.relative_to(ROOT)): digest(p) for p in (
        Path(__file__), HERE / "plan.json", BOOK / "preflight.py", BOOK / "prepare.py", EXAMPLE, HELPER)}
    args.output.mkdir(parents=True, exist_ok=False)
    execution = {"schema": "spiraltorch.hf_geometry_observation_execution.v1",
                 "status": "declared_before_measurement", "plan_id": args.expected_plan_id,
                 "sources": source_ids, "installed_package_ids": package_ids,
                 "wheel_id": digest(args.wheel), "versions": versions, "cases": cases,
                 "runtime_identity": base_id, "dataset_identity": data_id["observed_identity_id"],
                 "reserve_opened": False, "optimizer_updates": 0}
    (args.output / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")

    def unchanged():
        if (any(digest(ROOT / name) != pin for name, pin in source_ids.items())
                or PREFLIGHT["package_identity"](package) != package_ids
                or any(digest(paths[c["key"]]) != c["artifact_id"] for c in cases)):
            raise ValueError("source, installed runtime or saved artifact changed")

    config.use_cache = False
    model = hf.AutoModelForCausalLM.from_pretrained(
        args.model_dir, config=config, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32, attn_implementation="eager").requires_grad_(False).to("mps").eval()
    base_hash = api["parameter_hash"](model)
    if any(r["base_parameter_hash"] != base_hash for r in results["runs"]):
        raise ValueError("loaded frozen parameters differ from published book runs")

    def execute(case):
        unchanged()
        print(json.dumps({"key": case["key"], "status": "running"}), flush=True)
        artifact = st.GeometryAdapterArtifact.from_json(
            paths[case["key"]].read_text(), case["artifact_id"], base_id)
        geometry = st.GeometryAdapterStack.from_artifact(
            artifact, expected_base_model_identity=base_id, device="mps").eval()
        before = geometry.export_artifact(base_model_identity=base_id).artifact_id
        arguments = hf.TrainingArguments(
            output_dir=str(args.output / case["key"]), report_to=[], disable_tqdm=True,
            per_device_eval_batch_size=plan["eval_batch_size"], dataloader_pin_memory=False,
            seed=case["seed"] or 41)
        if arguments.device.type != "mps":
            raise ValueError("HF selected the wrong evaluation device")
        trainer = st.hf_geometry_trainer_class()(
            model=model, geometry=geometry, base_model_identity=base_id, args=arguments,
            eval_dataset=data["development"], data_collator=hf.default_data_collator,
            processing_class=tokenizer)
        cpu_rng, mps_rng = torch.get_rng_state(), torch.mps.get_rng_state()
        reference = trainer.evaluate()["eval_loss"]
        cpu_after, mps_after = torch.get_rng_state(), torch.mps.get_rng_state()
        torch.set_rng_state(cpu_rng)
        torch.mps.set_rng_state(mps_rng)
        with geometry.observe(max_events=plan["max_events"], max_values=plan["max_values"]) as trace:
            observed = trainer.evaluate()["eval_loss"]
        if (reference != observed or not math.isfinite(observed)
                or not torch.equal(cpu_after, torch.get_rng_state())
                or not torch.equal(mps_after, torch.mps.get_rng_state())
                or (case["expected_loss"] is not None and reference != case["expected_loss"])):
            raise ValueError("observation or runtime changed evaluation loss or RNG")
        sample = {k: v.to("mps") for k, v in hf.default_data_collator([data["development"][0]]).items()}
        with torch.no_grad(), geometry.attach(model, require_all=True):
            expected_logits = model(**sample, use_cache=False).logits
        with torch.no_grad(), geometry.observe(max_events=4), geometry.attach(model, require_all=True):
            actual_logits = model(**sample, use_cache=False).logits
        if not torch.equal(expected_logits, actual_logits):
            raise ValueError("observation changed first-window logits")
        if (geometry.export_artifact(base_model_identity=base_id).artifact_id != before
                or trainer.optimizer is not None or trainer.lr_scheduler is not None
                or any(p.grad is not None for p in geometry.parameters())
                or any(p.grad is not None or p.requires_grad for p in model.parameters())
                or api["parameter_hash"](model) != base_hash):
            raise ValueError("read-only observation changed model state")
        report = trace.to_dict()
        layers = aggregate(report, book_plan["sites"], native=case["arm"] != "ordinary")
        trace_path = args.output / f"{case['key']}.json"
        trace_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        unchanged()
        record = {**case, "eval_loss": observed, "layers": layers, "trace_id": digest(trace_path),
                  "observed_loss_exact": True, "first_window_logits_exact": True,
                  "rng_exact": True, "base_and_geometry_unchanged": True,
                  "coverage_complete": True, "optimizer_updates": 0}
        print(json.dumps({"key": case["key"], "eval_loss": observed, "status": "completed"}), flush=True)
        return record

    def validate(records):
        unchanged()
        if len(records) != 28:
            raise ValueError("missing planned cases")
        PREFLIGHT["inputs"](BOOK / "plan.json", plan["book_plan_id"],
                            args.book_private / "chapters-balanced", plan["book_manifest_id"])

    records = runpy.run_path(str(HELPER))["execute_cases"](
        args.output / "run-manifest.json", cases, execute, validate)
    result = {"schema": "spiraltorch.hf_geometry_observation_study.v1", "status": "completed",
              "plan_id": args.expected_plan_id, "execution_id": digest(args.output / "execution.json"),
              "cases": records, "reserve_opened": False, "optimizer_updates": 0,
              "quality_status": "posthoc_mechanism_diagnosis_not_quality_confirmation"}
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
