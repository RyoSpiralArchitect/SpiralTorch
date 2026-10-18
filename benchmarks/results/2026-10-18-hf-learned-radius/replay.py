#!/usr/bin/env python3
"""Exercise existing Rust learned-radius VJPs through the public HF loop."""

import argparse
import copy
import json
import math
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BOOK = HERE.parent / "2026-10-18-hf-book-controls"
BOOK_API = runpy.run_path(str(BOOK / "replay.py"))
PREFLIGHT = BOOK_API["PREFLIGHT"]
READ = BOOK_API["READ"]
PINNED = BOOK_API["load_pinned"]
DIGEST = BOOK_API["digest"]
OBSERVATION = HERE.parent / "2026-10-18-hf-geometry-observation"
EXAMPLE = BOOK_API["EXAMPLE"]
OWNER = BOOK_API["OWNER"]
HELPER = BOOK_API["HELPER"]
ARMS = ["fixed_r1", "learned_r1", "fixed_r4", "learned_r4"]
SOURCE_ARMS = {name: "wave_gate_radius" + name[-1] for name in ARMS}


def cases_for(plan):
    if (plan["schema"] != "spiraltorch.hf_learned_radius_plan.v1"
            or plan["arms"] != ARMS or plan["seeds"] != [41, 43, 47]
            or plan["primary_contrasts"] != [["fixed_r1", "learned_r1"], ["fixed_r4", "learned_r4"]]
            or plan["full_runs"] != 12 or plan["trainable_values"] != {"fixed": 6144, "learned": 6148}
            or plan["resume_cases"] != [{"arm": "learned_r1", "seed": 41}, {"arm": "learned_r4", "seed": 41}]
            or plan["resume_from_step"] != 256 or plan["resume_to_step"] != 512
            or plan["total_optimizer_updates_including_resume_replays"] != 6656):
        raise ValueError("unsupported learned-radius study declaration")
    cases = [{"arm": arm, "seed": seed, "phase": "full"} for seed in plan["seeds"] for arm in ARMS]
    return cases + [{**case, "phase": "resume"} for case in plan["resume_cases"]]


def learned_record(fixed):
    record = copy.deepcopy(fixed)
    for placement in record["placements"]:
        if (placement["recipe"]["schema"] != "spiraltorch.wave_gate_adapter.v2"
                or placement["recipe"]["learnable_radius"] is not False):
            raise ValueError("treatment requires an existing fixed-radius WaveGate recipe")
        radii = [t for t in placement["tensors"] if t["name"] == "log_radius"]
        if len(radii) != 1 or radii[0]["role"] != "buffer" or radii[0]["requires_grad"] is not False:
            raise ValueError("expected one frozen radius buffer at every site")
        placement["recipe"]["learnable_radius"] = True
        radii[0].update(role="parameter", requires_grad=True)
    return record


def radius_metrics(initial, final):
    before = {p["path"]: next(t["values"][0] for t in p["tensors"] if t["name"] == "log_radius")
              for p in initial["placements"]}
    after = {p["path"]: next(t["values"][0] for t in p["tensors"] if t["name"] == "log_radius")
             for p in final["placements"]}
    if not before or before.keys() != after.keys():
        raise ValueError("radius site coverage differs")
    delta = [after[name] - value for name, value in before.items()]
    if not all(math.isfinite(value) for value in delta):
        raise ValueError("radius changes must be finite")
    return {"sites": len(delta), "changed_sites": sum(value != 0 for value in delta),
            "increased_sites": sum(value > 0 for value in delta),
            "log_radius_delta_l2": math.sqrt(math.fsum(value * value for value in delta)),
            "radius_multiplier_geometric_mean": math.exp(math.fsum(delta) / len(delta))}


def tensor_equal(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(tensor_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(tensor_equal(a, b) for a, b in zip(left, right))
    return left == right


def saved_state(directory, step, *, learned, base_id, exported):
    import spiraltorch as st
    import torch
    checkpoint = directory / f"checkpoint-{step}"
    names = ["spiraltorch-geometry.json", "optimizer.pt", "scheduler.pt", "trainer_state.json",
             "rng_state.pth", "spiraltorch-mps-rng.pt"]
    ids = {name: DIGEST(checkpoint / name) for name in names}
    state = json.loads(READ(checkpoint / "trainer_state.json", 1_048_576))
    if state["global_step"] != step or state["max_steps"] != 512:
        raise ValueError("checkpoint step changed")
    optimizer = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    scheduler = torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True)
    moments = list(optimizer["state"].values())
    if (len(moments) != (12 if learned else 8)
            or sum(s["exp_avg"].numel() for s in moments) != (6148 if learned else 6144)
            or any(s["step"].item() != step or not torch.isfinite(s["exp_avg"]).all()
                   or not torch.isfinite(s["exp_avg_sq"]).all() for s in moments)
            or scheduler["last_epoch"] != step
            or any(g["lr"] != .001 * (1 - step / 512) for g in optimizer["param_groups"])):
        raise ValueError("optimizer/scheduler does not own the declared finite learned state")
    artifact = st.GeometryAdapterArtifact.from_json(
        READ(checkpoint / "spiraltorch-geometry.json", 1_048_576).decode(),
        ids["spiraltorch-geometry.json"], base_id)
    def normalized(record):
        return [{k: v for k, v in p.items() if k != "training"} for p in record["placements"]]
    if step == 512 and normalized(json.loads(artifact.state_json())) != normalized(exported):
        raise ValueError("checkpoint and final export differ")
    return {"step": step, "artifact_id": artifact.artifact_id, "private_file_ids": ids,
            "finite_optimizer_values": sum(s["exp_avg"].numel() for s in moments),
            "radius_nonzero_first_moments": sum(s["exp_avg"].ndim == 0 and s["exp_avg"].item() != 0 for s in moments),
            "optimizer_scheduler_step_exact": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-plan-id", required=True)
    parser.add_argument("--book-private", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new private directory")
    plan = PINNED(HERE / "plan.json", args.expected_plan_id)
    cases = cases_for(plan)
    book_results = PINNED(BOOK / "results.json", plan["book_results_id"])
    previous = PINNED(args.book_private / "preflight-bound/preflight.json", plan["book_preflight_id"])
    runtime = PINNED(OBSERVATION / "execution.json", plan["qualified_runtime_execution_id"])

    def inputs():
        return PREFLIGHT["inputs"](BOOK / "plan.json", plan["book_plan_id"],
                                   args.book_private / "chapters-balanced", plan["book_manifest_id"])
    book, _, selected, _, _ = inputs()
    BOOK_API["cases_for"](book)
    import spiraltorch as st
    import torch
    import transformers as hf
    import accelerate
    package = Path(st.__file__).parent
    versions = {"python": sys.version.split()[0], "torch": torch.__version__,
                "transformers": hf.__version__, "accelerate": accelerate.__version__}
    if (versions != previous["versions"] or versions != runtime["versions"]
            or PREFLIGHT["package_identity"](package) != runtime["installed_package_ids"]
            or not torch.backends.mps.is_available()):
        raise ValueError("runtime differs from the qualified installed wheel")
    torch.set_num_threads(2)
    _, _, actual = runpy.run_path(str(EXAMPLE))["local_runtime"](args.model_dir)
    base_id = actual["observed_identity_id"]
    if base_id != previous["runtime_identity"] or base_id != book["model"]["runtime_identity"]:
        raise ValueError("local model/tokenizer changed")
    artifacts, stacks = {}, {}
    for arm in ARMS:
        source = SOURCE_ARMS[arm]
        fixed = st.GeometryAdapterArtifact.from_json(
            READ(args.book_private / f"preflight-bound/{source}-initial.json", 1_048_576).decode(),
            previous["artifacts"][source]["artifact_id"], base_id)
        artifact = (st.GeometryAdapterArtifact.create_json(json.dumps(learned_record(json.loads(fixed.state_json()))))
                    if arm.startswith("learned") else fixed)
        stack = st.GeometryAdapterStack.from_artifact(artifact, expected_base_model_identity=base_id)
        count = sum(p.numel() for p in stack.parameters() if p.requires_grad)
        if count != plan["trainable_values"][arm.split("_")[0]]:
            raise ValueError("initial trainable budget differs")
        artifacts[arm], stacks[arm] = artifact, stack
    value = torch.linspace(-1., 1., 2 * 4 * 768).reshape(2, 4, 768)
    for fixed_name, learned_name in plan["primary_contrasts"]:
        for fixed, learned in zip(stacks[fixed_name].adapters, stacks[learned_name].adapters):
            if not torch.equal(fixed(value), value) or not torch.equal(learned(value), value):
                raise ValueError("initial output is not the shared identity")
            fixed(value).square().sum().backward()
            learned(value).square().sum().backward()
            if (not torch.equal(fixed.gate.grad, learned.gate.grad)
                    or not torch.equal(fixed.bias.grad, learned.bias.grad)
                    or learned.log_radius.grad.item() != 0):
                raise ValueError("initial parameter gradients differ")
    args.output.mkdir(parents=True, exist_ok=False)
    preflight_dir = args.output / "preflight"
    preflight_dir.mkdir()
    preflight = {**previous, "artifacts": {}}
    for arm, artifact in artifacts.items():
        (preflight_dir / f"{arm}-initial.json").write_text(artifact.to_json())
        preflight["artifacts"][arm] = {"artifact_id": artifact.artifact_id, "layout_id": artifact.layout_id}
    source_ids = {str(p.relative_to(ROOT)): DIGEST(p) for p in (
        Path(__file__), HERE / "plan.json", BOOK / "replay.py", BOOK / "preflight.py",
        BOOK / "prepare.py", EXAMPLE, HELPER, OWNER)}
    execution = {"schema": "spiraltorch.hf_learned_radius_execution.v1", "status": "declared_before_training",
                 "plan_id": args.expected_plan_id, "sources": source_ids, "versions": versions,
                 "installed_package_ids": runtime["installed_package_ids"], "runtime_identity": base_id,
                 "dataset_identity": previous["tokenized_dataset_identity"], "cases": cases,
                 "initial_artifacts": preflight["artifacts"], "initial_outputs_and_common_gradients_exact": True,
                 "initial_radius_gradients_zero": True, "reserve_opened": False, "case_timeout_seconds": 900}
    (args.output / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    completed = {}
    owned = runpy.run_path(str(OWNER))["run"]

    def unchanged():
        inputs()
        if (any(DIGEST(ROOT / name) != pin for name, pin in source_ids.items())
                or PREFLIGHT["package_identity"](package) != runtime["installed_package_ids"]
                or any(DIGEST(preflight_dir / f"{arm}-initial.json") != artifact.artifact_id
                       for arm, artifact in artifacts.items())):
            raise ValueError("study source/runtime/initial-state bytes changed")

    def execute(case):
        unchanged()
        key = f"{case['arm']}-seed-{case['seed']}-{case['phase']}"
        destination = args.output / key
        command = BOOK_API["command_for"](book, preflight, selected, args.model_dir,
                                           preflight_dir, destination, case)
        original = None
        if case["phase"] == "resume":
            original = completed[(case["arm"], case["seed"])]
            checkpoint = args.output / f"{case['arm']}-seed-{case['seed']}-full/checkpoint-256"
            for name, pin in original["checkpoints"][0]["private_file_ids"].items():
                if DIGEST(checkpoint / name) != pin:
                    raise ValueError("independently retained resume companion changed")
            command += ["--resume-from-checkpoint", str(checkpoint.resolve()),
                        "--geometry-checkpoint-id", original["checkpoints"][0]["artifact_id"]]
        print(json.dumps(dict(case, status="running")), flush=True)
        log_path = args.output / f"{key}.log"
        with log_path.open("x") as log:
            owned(command, timeout=900, stdout=log, cwd=ROOT).check_returncode()
        result = json.loads(READ(destination / "result.json", 1_048_576))
        BOOK_API["validate_result"](result, book, preflight, case)
        exported = st.GeometryAdapterArtifact.from_json(
            READ(destination / "geometry-export/spiraltorch-geometry.json", 1_048_576).decode(), result["artifact_id"], base_id)
        final = json.loads(exported.state_json())
        learned = case["arm"].startswith("learned")
        metrics = BOOK_API["state_metrics"](json.loads(artifacts[case["arm"]].state_json()), final)
        radii = radius_metrics(json.loads(artifacts[case["arm"]].state_json()), final)
        checks = [saved_state(destination, step, learned=learned, base_id=base_id, exported=final)
                  for step in ([512] if original else [256, 512])]
        fixed_exact = None
        if not learned:
            prior = next(r for r in book_results["runs"] if r["arm"] == SOURCE_ARMS[case["arm"]] and r["seed"] == case["seed"])
            fixed_exact = all(result[k] == prior[k] for k in (
                "trained_geometry_eval_loss", "artifact_id", "checkpoint_artifact_id", "base_parameter_hash"))
            if not fixed_exact:
                raise ValueError("fixed control did not reproduce the previous complete final state")
        resume_exact = None
        if original:
            prior_dir = args.output / f"{case['arm']}-seed-{case['seed']}-full/checkpoint-512"
            if any(DIGEST(prior_dir / name) != pin for name, pin in original["checkpoints"][1]["private_file_ids"].items()):
                raise ValueError("uninterrupted comparison companion changed")
            resume_exact = result["artifact_id"] == original["artifact_id"]
            for name in ("optimizer.pt", "scheduler.pt", "spiraltorch-mps-rng.pt"):
                resume_exact = resume_exact and tensor_equal(
                    torch.load(prior_dir / name, map_location="cpu", weights_only=True),
                    torch.load(destination / "checkpoint-512" / name, map_location="cpu", weights_only=True))
            resume_exact = resume_exact and DIGEST(prior_dir / "rng_state.pth") == checks[0]["private_file_ids"]["rng_state.pth"]
            if not resume_exact:
                raise ValueError("resumed final weights or optimizer/scheduler/RNG differ")
        record = {k: result[k] for k in ("artifact_id", "checkpoint_artifact_id", "base_parameter_hash",
                  "trained_geometry_eval_loss", "bare_eval_loss", "base_unchanged", "restored_logits_exact")}
        record.update(case, radii=radii, learned_sites=metrics, checkpoints=checks,
                      fixed_replay_exact=fixed_exact, resume_exact=resume_exact,
                      result_id=DIGEST(destination / "result.json"), log_id=DIGEST(log_path))
        unchanged()
        if case["phase"] == "full":
            completed[(case["arm"], case["seed"])] = record
        print(json.dumps({**case, "status": "completed", "loss": result["trained_geometry_eval_loss"], "radii": radii}), flush=True)
        return record

    def validate(records):
        unchanged()
        if len(records) != 14 or len({r["base_parameter_hash"] for r in records}) != 1:
            raise ValueError("missing cases or changed frozen base")

    records = runpy.run_path(str(HELPER))["execute_cases"](
        args.output / "run-manifest.json", cases, execute, validate)
    differences = [{"seed": seed, "contrast": f"{right}_minus_{left}",
                    "development_loss_difference": completed[(right, seed)]["trained_geometry_eval_loss"]
                    - completed[(left, seed)]["trained_geometry_eval_loss"]}
                   for seed in plan["seeds"] for left, right in plan["primary_contrasts"]]
    summary = {"schema": "spiraltorch.hf_learned_radius_results.v1", "status": "completed",
               "plan_id": args.expected_plan_id, "execution_id": DIGEST(args.output / "execution.json"),
               "runs": records, "paired_differences": differences, "reserve_opened": False,
               "quality_status": "exploratory_one_corpus_not_matched_energy_or_equal_parameter_budget"}
    (args.output / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
