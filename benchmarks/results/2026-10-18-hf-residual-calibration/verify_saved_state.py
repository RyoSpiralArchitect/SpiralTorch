#!/usr/bin/env python3
"""Verify retained calibration traces and artifacts without loading a model."""

import argparse
import copy
import json
import math
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
API = runpy.run_path(str(HERE / "replay.py"))
POLICY, OBS = API["POLICY"], API["OBS"]
DIGEST, PINNED, REQUIRE = API["DIGEST"], API["PINNED"], API["require"]


def verify_case(case, record, original, candidate, target, plan):
    """Recompute the declared strength rule; never print its private operands."""
    matched = case["mode"] == "calibrated_ordinary"
    REQUIRE({key: record[key] for key in case} == case, "case identity differs")
    REQUIRE(record["all_learned_tensors_unchanged"] is True
            and record["restored_first_window_logits_exact"] is True
            and record["only_recipe_strengths_may_change"] is matched
            and type(record["optimizer_updates"]) is int and record["optimizer_updates"] == 0,
            "runtime state guards were not satisfied")
    REQUIRE(math.isfinite(record["development_loss"]), "nonfinite development CE")
    REQUIRE(POLICY["without_strength"](original) == POLICY["without_strength"](candidate),
            "learned tensors or other recipe fields differ")
    expected_stages = [f"calibration-before-layer-{i}" for i in range(4)] if matched else []
    REQUIRE([s["stage"] for s in record["calibration_stages"]] == expected_stages
            and record["calibration"]["stage"] == "calibration-final"
            and record["development"]["stage"] == "development", "stage order differs")
    if not matched:
        REQUIRE(original == candidate and record["reference_loss_exact"] is True
                and record["source_artifact_id"] == record["candidate_artifact_id"]
                and record["target_artifact_id"] is None
                and record["calibration_mismatch"] is None and record["development_mismatch"] is None,
                "untouched reference changed")
        return
    REQUIRE(record["reference_loss_exact"] is None
            and record["target_artifact_id"] == target["candidate_artifact_id"], "target identity differs")
    for index, (before, after) in enumerate(zip(original["placements"], candidate["placements"])):
        expected = POLICY["fitted_strength"](
            before["recipe"]["strength"], target["calibration"]["layers"][index]["relative_residual_rms"],
            record["calibration_stages"][index]["layers"][index]["relative_residual_rms"])
        REQUIRE(after["recipe"]["strength"] == expected, "fitted strength rule differs")
    for stage in ("calibration", "development"):
        errors = POLICY["mismatch"](record[stage]["layers"], target[stage]["layers"])
        REQUIRE(record[f"{stage}_mismatch"] == errors, "stored mismatch differs")
        if stage == "calibration":
            REQUIRE(all(abs(e["relative_rms_error"]) <= plan["calibration_relative_tolerance"]
                        for e in errors), "training calibration did not qualify")


def verify_stage_ids(record, original, candidate, identify):
    """Each pre-fit stage must contain exactly the preceding layer updates."""
    current = copy.deepcopy(original)
    REQUIRE(identify(current) == record["source_artifact_id"], "source normalization changed identity")
    for index, stage in enumerate(record["calibration_stages"]):
        REQUIRE(stage.get("geometry_artifact_id") == identify(current), "intermediate stage identity differs")
        current["placements"][index]["recipe"]["strength"] = candidate["placements"][index]["recipe"]["strength"]
    REQUIRE(current == candidate and identify(current) == record["candidate_artifact_id"],
            "sequential calibration did not reach the candidate")
    for stage in (record["calibration"], record["development"]):
        REQUIRE(stage.get("geometry_artifact_id") == record["candidate_artifact_id"],
                "final stage identity differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-runs", type=Path, required=True)
    parser.add_argument("--book-private", type=Path, required=True)
    parser.add_argument("--radius-runs", type=Path, required=True)
    parser.add_argument("--expected-results-id", required=True)
    parser.add_argument("--expected-manifest-id", required=True)
    args = parser.parse_args()
    public = PINNED(HERE / "results.json", args.expected_results_id)
    execution = PINNED(HERE / "execution.json", public["execution_id"])
    plan = PINNED(HERE / "plan.json", public["plan_id"])
    manifest = PINNED(args.private_runs / "run-manifest.json", args.expected_manifest_id)
    REQUIRE(public["private_manifest_id"] == args.expected_manifest_id
            and public["status"] == manifest["status"] == "completed", "study incomplete")
    for name in ("results.json", "execution.json"):
        REQUIRE((HERE / name).read_bytes() == (args.private_runs / name).read_bytes(), "publication bytes differ")
    cases = POLICY["cases_for"](plan)
    REQUIRE(execution["cases"] == cases and len(manifest["runs"]) == len(public["cases"]) == len(cases),
            "case coverage differs")
    REQUIRE(execution["calibration_train_indices"] == POLICY["calibration_indices"](
        plan["train_windows"], plan["calibration_windows"]), "calibration selection differs")
    REQUIRE(execution["plan_id"] == public["plan_id"]
            and execution["optimizer_updates"] == public["optimizer_updates"] == 0
            and execution["reserve_opened"] is public["reserve_opened"] is False, "protocol differs")
    for name, pin in execution["sources"].items():
        REQUIRE(DIGEST(API["ROOT"] / name) == pin, "frozen source changed")
    REQUIRE(manifest["replay_source_id"] == DIGEST(OBS["HELPER"]), "case owner changed")
    book = PINNED(API["BOOK"] / "results.json", plan["book_results_id"])
    radius = PINNED(API["RADIUS"] / "results.json", plan["radius_results_id"])
    originals = [r for r in book["runs"] if r["arm"] == "ordinary"]
    originals += [r for r in radius["runs"] if r["phase"] == "full"]
    REQUIRE(execution["source_artifacts"] == [
        {"arm": r["arm"], "seed": r["seed"], "artifact_id": r["artifact_id"],
         "expected_loss": r["trained_geometry_eval_loss"]} for r in originals], "source lineage differs")
    import spiraltorch as st
    REQUIRE(OBS["PREFLIGHT"]["package_identity"](Path(st.__file__).parent)
            == execution["installed_package_ids"], "installed SpiralTorch differs")

    def state(path, identity):
        return json.loads(st.GeometryAdapterArtifact.from_json(
            API["READ"](path, 1_048_576).decode(), identity, execution["runtime_identity"]).state_json())

    source_states, sources = {}, {}
    for source in originals:
        key = (source["arm"], source["seed"])
        directory = (args.book_private / f"runs-fixed/ordinary-seed-{source['seed']}"
                     if source["arm"] == "ordinary"
                     else args.radius_runs / f"{source['arm']}-seed-{source['seed']}-full")
        source_states[key] = state(directory / "geometry-export/spiraltorch-geometry.json", source["artifact_id"])
        sources[key] = source
    references, records = {}, []
    trace_count = 0
    for case, retained, published in zip(cases, manifest["runs"], public["cases"]):
        REQUIRE({k: retained[k] for k in case} == case and retained["status"] == "completed",
                "manifest disposition differs")
        key = f"{case['arm']}-seed-{case['seed']}-{case['mode']}"
        directory = args.private_runs / key
        record = PINNED(directory / "result.json", published["private_result_id"])
        augmented = {**record, "private_result_id": published["private_result_id"]}
        REQUIRE(retained["result"] == augmented and POLICY["public_record"](augmented) == published,
                "manifest or public projection differs")
        matched = case["mode"] == "calibrated_ordinary"
        source_key = ("ordinary" if matched else case["arm"], case["seed"])
        REQUIRE(record["source_artifact_id"] == sources[source_key]["artifact_id"]
                and record["base_parameter_hash"] == sources[source_key]["base_parameter_hash"],
                "source or base identity differs")
        candidate = state(directory / "candidate.json", record["candidate_artifact_id"])
        target = references[(case["arm"], case["seed"])] if matched else None
        verify_case(case, record, source_states[source_key], candidate, target, plan)
        verify_stage_ids(record, source_states[source_key], candidate,
                         lambda payload: st.GeometryAdapterArtifact.create_json(
                             json.dumps(payload, allow_nan=False)).artifact_id)
        for stage in [*record["calibration_stages"], record["calibration"], record["development"]]:
            trace = PINNED(directory / f"{stage['stage']}.json", stage["trace_id"])
            windows = plan["development_windows"] if stage["stage"] == "development" else plan["calibration_windows"]
            layers = POLICY["public_layers"](OBS["aggregate"](
                trace, POLICY["SITES"], native=source_key[0] != "ordinary",
                expected_calls=math.ceil(windows / plan["eval_batch_size"]) * len(POLICY["SITES"]),
                expected_rows=windows * plan["context_length"]))
            REQUIRE(layers == stage["layers"], "trace aggregate differs")
            trace_count += 1
        if not matched:
            REQUIRE(record["development_loss"] == sources[source_key]["trained_geometry_eval_loss"],
                    "reference CE differs")
            references[(case["arm"], case["seed"])] = record
        records.append(record)
    contrasts = [{"arm": r["arm"], "seed": r["seed"],
                  "geometry_minus_matched_ordinary_ce": references[(r["arm"], r["seed"])]["development_loss"]
                  - r["development_loss"],
                  "matched_minus_original_ordinary_ce": r["development_loss"]
                  - references[("ordinary", r["seed"])]["development_loss"]}
                 for r in records if r["mode"] == "calibrated_ordinary"]
    REQUIRE(contrasts == public["contrasts"] and trace_count == 102, "contrast or trace count differs")
    print(json.dumps({"schema": "spiraltorch.hf_residual_calibration_saved_check.v1", "status": "passed",
                      "results_id": args.expected_results_id, "execution_id": public["execution_id"],
                      "private_manifest_id": args.expected_manifest_id, "verifier_id": DIGEST(Path(__file__)),
                      "cases": len(records), "traces": trace_count, "candidate_artifacts": len(records),
                      "source_artifacts": len(source_states), "public_projection_exact": True,
                      "all_retained_outcomes_published": True, "trace_aggregates_exact": True,
                      "sequential_stage_artifact_ids_exact": True,
                      "calibration_rule_and_mismatch_rechecked": True, "learned_tensors_unchanged": True,
                      "reference_ce_receipts_exact": True, "source_ids_unchanged": True,
                      "model_loaded": False, "optimizer_updates": 0, "reserve_opened": False}, indent=2))


if __name__ == "__main__":
    main()
