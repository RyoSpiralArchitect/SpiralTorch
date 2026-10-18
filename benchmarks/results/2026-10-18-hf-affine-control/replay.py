#!/usr/bin/env python3
"""Bounded matched-control probe; delegates every update to the public HF example.

Outputs (including generated text and learned artifacts) stay in the caller's
new private directory. Publish only summary.json and run-manifest.json after
checking their contents. A failed run remains in the manifest; no retries.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import subprocess
import sys


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def execute_cases(path, cases, execute, validate):
    """Retain every planned case even on failure; this is not crash recovery."""
    manifest = {"schema": "spiraltorch.hf_affine_run_manifest.v1", "status": "running",
                "replay_source_id": digest(Path(__file__)),
                "runs": [dict(case, status="not_started") for case in cases]}

    def persist():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    persist()
    records = []
    current = None
    try:
        for current in manifest["runs"]:
            current["status"] = "running"
            persist()
            record = execute({key: value for key, value in current.items() if key != "status"})
            current.update(status="completed", result=record)
            records.append(record)
            persist()
        current = None
        manifest["status"] = "qualifying"
        persist()
        validate(records)
    except BaseException as error:
        if current is not None:
            current["status"] = "failed"
        # Exception messages may contain private paths or data; preserve them
        # in the local log, not in the publishable manifest.
        manifest.update(status="failed", error_type=type(error).__name__)
        persist()
        raise
    manifest["status"] = "completed"
    persist()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train-text", type=Path, action="append", required=True)
    parser.add_argument("--eval-text", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new private directory")
    files = args.train_text + args.eval_text
    hashes = [digest(path) for path in files]
    if len(set(hashes)) != len(hashes):
        parser.error("each document must be distinct")
    root = Path(__file__).resolve().parents[3]
    example = root / "bindings/st-py/examples/hf_geometry_finetune.py"
    api = runpy.run_path(str(example))
    config, _, runtime = api["local_runtime"](args.model_dir)
    if config.model_type != "gpt2" or config.n_embd != 768 or config.n_layer != 12:
        parser.error("this bounded recipe requires local GPT-2 small (12 layers, width 768)")
    import spiraltorch as st
    import torch
    import transformers
    import accelerate

    args.output.mkdir(parents=True, exist_ok=False)
    base_id = runtime["observed_identity_id"]
    paths = [f"transformer.h.{i}.mlp" for i in (8, 9, 10, 11)]
    artifacts = {}
    for arm, kind in (("ordinary", st.ChannelAffineAdapter), ("wave_gate", st.WaveGateAdapter)):
        options = {"curvature": -1.} if arm == "wave_gate" else {}
        stack = st.GeometryAdapterStack({site: kind(768, strength=.1, **options) for site in paths})
        count = sum(p.numel() for p in stack.parameters() if p.requires_grad)
        if count != 6144 or any(torch.count_nonzero(p) for p in stack.parameters()):
            raise RuntimeError("arms must have 6144 zero-initialized trainable parameters")
        artifact = stack.export_artifact(base_model_identity=base_id)
        path = args.output / f"{arm}-initial.json"
        path.write_text(artifact.to_json(), encoding="utf-8")
        artifacts[arm] = (path, artifact.artifact_id)
    common = ["--model-dir", str(args.model_dir), "--context-length", "32",
              "--max-steps", "32", "--save-steps", "16", "--batch-size", "2",
              "--learning-rate", ".001", "--max-grad-norm", ".1", "--device", args.device]
    for kind, files_ in (("--train-text", args.train_text), ("--eval-text", args.eval_text)):
        for path in files_:
            common.extend((kind, str(path)))
    def execute(case):
        arm, seed = case["arm"], case["seed"]
        path, pin = artifacts[arm]
        destination = args.output / f"{arm}-seed-{seed}"
        command = [sys.executable, "-I", "-B", str(example), *common,
                   "--geometry-artifact", str(path), "--geometry-artifact-id", pin,
                   "--seed", str(seed), "--output", str(destination)]
        with (args.output / f"{arm}-seed-{seed}.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        result_path = destination / "result.json"
        result = json.loads(result_path.read_text())
        if not (result["step"] == 32 and result["base_unchanged"] and result["restored_logits_exact"]):
            raise RuntimeError("workflow contract did not complete")
        if result["bare_eval_loss"] != result["initial_geometry_eval_loss"]:
            raise RuntimeError("identity-start reference differs from frozen base")
        if not math.isfinite(result["trained_geometry_eval_loss"]):
            raise RuntimeError("non-finite evaluation loss")
        final_path = destination / "geometry-export/spiraltorch-geometry.json"
        final = st.GeometryAdapterArtifact.from_json(
            final_path.read_text(), result["artifact_id"], base_id)
        state = json.loads(final.state_json())
        changed_sites = sum(any(v != 0. for t in site["tensors"] for v in t["values"])
                            for site in state["placements"])
        if changed_sites != len(paths):
            raise RuntimeError("not every intended adapter learned")
        record = {key: result[key] for key in (
            "runtime_identity", "tokenized_dataset_identity", "initial_artifact_id",
            "initial_layout_id", "artifact_id", "checkpoint_artifact_id", "seed", "step",
            "base_parameter_hash", "base_unchanged", "restored_logits_exact", "bare_eval_loss",
            "initial_geometry_eval_loss", "trained_geometry_eval_loss",
            "train_window_plan", "eval_window_plan")}
        record.update(arm=arm, trained_sites=changed_sites,
                      private_result_id=digest(result_path), private_artifact_id=digest(final_path))
        print(json.dumps({k: record[k] for k in ("arm", "seed", "step", "trained_geometry_eval_loss")}),
              flush=True)
        return record

    def validate(records):
        if (len({r["tokenized_dataset_identity"] for r in records}) != 1
                or len({r["base_parameter_hash"] for r in records}) != 1
                or hashes != [digest(path) for path in files]):
            raise RuntimeError("data or base changed across the matched runs")

    cases = [{"arm": arm, "seed": seed} for seed in (41, 43, 47) for arm in artifacts]
    records = execute_cases(args.output / "run-manifest.json", cases, execute, validate)
    paired = []
    for seed in (41, 43, 47):
        pair = {r["arm"]: r for r in records if r["seed"] == seed}
        paired.append({"seed": seed, "wave_minus_ordinary_eval_loss":
                       pair["wave_gate"]["trained_geometry_eval_loss"] -
                       pair["ordinary"]["trained_geometry_eval_loss"]})
    summary = {
        "schema": "spiraltorch.hf_affine_matched_probe.v1",
        "status": "six_runs_completed", "quality_status": "workflow_probe_not_quality_evidence",
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "transformers": transformers.__version__, "accelerate": accelerate.__version__},
        "recipe": {"model": "local_gpt2_small", "sites": paths, "trainable_values_per_arm": 6144,
                   "geometry_backend": "rust_f32_cpu", "hf_device": args.device,
                   "strength": .1, "wave_curvature": -1., "wave_learnable_radius": False,
                   "context_length": 32, "batch_size": 2, "max_steps": 32, "save_steps": 16,
                   "learning_rate": .001, "max_grad_norm": .1, "seeds": [41, 43, 47],
                   "attention": "eager", "dtype": "float32", "cache": False,
                   "optimizer": "adamw_torch", "scheduler": "linear", "warmup_steps": 0,
                   "weight_decay": 0., "gradient_accumulation_steps": 1},
        "documents": [{"split": "train" if i < len(args.train_text) else "eval", "sha256": h}
                      for i, h in enumerate(hashes)],
        "runs": records, "paired_differences": paired,
        "claims": {"general_language_quality": False, "gpu_speed": False, "promoted_default": False},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
