#!/usr/bin/env python3
"""Read-only verification of the retained learned-radius study; no model loads."""

import argparse
import json
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
API = runpy.run_path(str(HERE / "replay.py"))
PUBLIC = runpy.run_path(str(HERE / "publication.py"))["public_results"]
DIGEST = API["DIGEST"]
PINNED = API["PINNED"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-runs", type=Path, required=True)
    parser.add_argument("--expected-results-id", required=True)
    parser.add_argument("--expected-manifest-id", required=True)
    args = parser.parse_args()
    result = PINNED(HERE / "results.json", args.expected_results_id)
    execution = PINNED(HERE / "execution.json", result["execution_id"])
    plan = PINNED(HERE / "plan.json", result["plan_id"])
    manifest = PINNED(args.private_runs / "run-manifest.json", args.expected_manifest_id)
    require(result["status"] == manifest["status"] == "completed", "study incomplete")
    require(execution["cases"] == API["cases_for"](plan), "case declaration changed")
    require(len(manifest["runs"]) == len(result["runs"]) == 14, "missing outcomes")
    private_id = result["publication"]["private_source_results_id"]
    private = PINNED(args.private_runs / "results.json", private_id)
    require(PUBLIC(private, private_id) == result, "public result projection differs")
    require((HERE / "execution.json").read_bytes() == (args.private_runs / "execution.json").read_bytes(),
            "execution bytes differ")
    for name, pin in execution["sources"].items():
        require(DIGEST(API["ROOT"] / name) == pin, "frozen source changed")
    import spiraltorch as st
    import torch
    require(API["PREFLIGHT"]["package_identity"](Path(st.__file__).parent)
            == execution["installed_package_ids"], "installed SpiralTorch changed")
    base_id = execution["runtime_identity"]
    initial = {}
    for arm, identity in execution["initial_artifacts"].items():
        path = args.private_runs / "preflight" / f"{arm}-initial.json"
        initial[arm] = json.loads(st.GeometryAdapterArtifact.from_json(
            API["READ"](path, 1_048_576).decode(), identity["artifact_id"], base_id).state_json())
    full = {}
    checkpoint_count = companion_count = resumed = 0
    require(len(private["runs"]) == 14, "missing private outcomes")
    for case, retained, published, original_result in zip(
            execution["cases"], manifest["runs"], result["runs"], private["runs"]):
        require({k: published[k] for k in case} == case, "case identity differs")
        require(retained["status"] == "completed" and retained["result"] == original_result,
                "manifest outcome differs")
        key = f"{case['arm']}-seed-{case['seed']}-{case['phase']}"
        directory = args.private_runs / key
        raw = PINNED(directory / "result.json", published["result_id"])
        require(DIGEST(args.private_runs / f"{key}.log") == published["log_id"], "case log differs")
        for field in ("artifact_id", "checkpoint_artifact_id", "base_parameter_hash",
                      "trained_geometry_eval_loss", "bare_eval_loss", "base_unchanged", "restored_logits_exact"):
            require(raw[field] == published[field], "result field differs")
        export = json.loads(st.GeometryAdapterArtifact.from_json(
            API["READ"](directory / "geometry-export/spiraltorch-geometry.json", 1_048_576).decode(),
            published["artifact_id"], base_id).state_json())
        require(API["radius_metrics"](initial[case["arm"]], export) == published["radii"],
                "radius aggregate differs")
        require(API["BOOK_API"]["state_metrics"](initial[case["arm"]], export)
                == original_result["learned_sites"], "private parameter aggregate differs")
        require([c["step"] for c in published["checkpoints"]]
                == ([256, 512] if case["phase"] == "full" else [512]), "checkpoint set differs")
        for checkpoint in published["checkpoints"]:
            observed = API["saved_state"](directory, checkpoint["step"],
                learned=case["arm"].startswith("learned"), base_id=base_id, exported=export)
            require(observed == checkpoint, "checkpoint hashes or optimizer metrics differ")
            checkpoint_count += 1
            companion_count += len(checkpoint["private_file_ids"])
        pair = (case["arm"], case["seed"])
        if case["phase"] == "full":
            full[pair] = published
            continue
        original = full[pair]
        require(published["resume_exact"] is True and published["artifact_id"] == original["artifact_id"]
                and published["trained_geometry_eval_loss"] == original["trained_geometry_eval_loss"],
                "resume final state differs")
        original_dir = args.private_runs / f"{case['arm']}-seed-{case['seed']}-full/checkpoint-512"
        for name in ("optimizer.pt", "scheduler.pt", "spiraltorch-mps-rng.pt"):
            require(API["tensor_equal"](
                torch.load(original_dir / name, map_location="cpu", weights_only=True),
                torch.load(directory / "checkpoint-512" / name, map_location="cpu", weights_only=True)),
                "resumed optimizer/scheduler/MPS RNG differs")
        require(DIGEST(original_dir / "rng_state.pth")
                == DIGEST(directory / "checkpoint-512/rng_state.pth"), "resumed HF RNG bytes differ")
        resumed += 1
    differences = [{"seed": seed, "contrast": f"{right}_minus_{left}",
                    "development_loss_difference": full[(right, seed)]["trained_geometry_eval_loss"]
                    - full[(left, seed)]["trained_geometry_eval_loss"]}
                   for seed in plan["seeds"] for left, right in plan["primary_contrasts"]]
    require(differences == result["paired_differences"], "paired differences changed")
    print(json.dumps({"schema": "spiraltorch.hf_learned_radius_saved_check.v1", "status": "passed",
                      "results_id": args.expected_results_id, "execution_id": result["execution_id"],
                      "private_manifest_id": args.expected_manifest_id, "verifier_id": DIGEST(Path(__file__)),
                      "runs": 14, "checkpoints": checkpoint_count, "checkpoint_companions": companion_count,
                      "exact_resumes_rechecked": resumed, "all_manifest_outcomes_published": True,
                      "private_result_and_log_hashes_exact": True, "public_result_projection_exact": True,
                      "private_source_results_id": private_id, "execution_bytes_exact": True,
                      "publication_source_id": DIGEST(HERE / "publication.py"),
                      "radius_and_parameter_aggregates_exact": True, "source_ids_unchanged": True,
                      "optimizer_updates": 0, "model_loaded": False, "reserve_opened": False}, indent=2))


if __name__ == "__main__":
    main()
