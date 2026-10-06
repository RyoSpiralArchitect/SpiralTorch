#!/usr/bin/env python3
"""Read saved adapters, never train or score the base model.

Use the frozen study client/native package on PYTHONPATH. A repeat that differs
is recorded as a failed replay criterion, not discarded as an invalid result.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.shape == right.shape
                and left.dtype == right.dtype
                and torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                                right.contiguous().reshape(-1).view(torch.uint8)))
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(equal(a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def verify_files(root, manifest):
    require(isinstance(manifest, dict) and bool(manifest), "empty file manifest")
    actual = set()
    for path in root.rglob("*"):
        require(not path.is_symlink(), "invalid frozen file path")
        if path.is_file():
            relative = path.relative_to(root)
            # Ignore descendants of real cache directories, not cache-named files.
            cached = any(parent.name == "__pycache__" and (root / parent).is_dir()
                         for parent in relative.parents)
            if not cached:
                actual.add(str(relative))
    require(actual == set(manifest), "frozen file inventory differs")
    for name, expected in manifest.items():
        path = root / name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts
                and not path.is_symlink() and root.resolve() in path.resolve().parents,
                "invalid frozen file path")
        require(digest(path) == expected, f"frozen file differs: {name}")
    return len(manifest)


def named_adam(saved, adapter, optimizer, steps):
    parameters = dict(adapter.named_parameters())
    state = saved["optimizer"]
    require(len(state["param_groups"]) == 1, "Adam group count differs")
    group = state["param_groups"][0]
    expected = optimizer.state_dict()["param_groups"][0]
    options = lambda row: {key: value for key, value in row.items() if key != "params"}
    require(equal(options(group), options(expected)), "Adam recipe differs")
    ids = group["params"]
    require(len(ids) == len(parameters) == len(set(ids))
            and set(ids) == set(state["state"]), "Adam parameter mapping differs")
    named = {}
    for (name, parameter), identifier in zip(parameters.items(), ids):
        value = state["state"][identifier]
        require(set(value) == {"step", "exp_avg", "exp_avg_sq"}, "Adam state fields differ")
        for field in ("exp_avg", "exp_avg_sq"):
            tensor = value[field]
            require(isinstance(tensor, torch.Tensor) and tensor.shape == parameter.shape
                    and tensor.dtype == parameter.dtype and bool(torch.isfinite(tensor).all()),
                    f"invalid Adam moment: {name}/{field}")
        require(bool((value["exp_avg_sq"] >= 0).all()), "negative Adam second moment")
        require(isinstance(value["step"], torch.Tensor) and value["step"].shape == torch.Size([])
                and float(value["step"]) == steps, "Adam step differs")
        named[name] = value
    optimizer.load_state_dict(copy.deepcopy(state))
    require(equal(optimizer.state_dict(), state), "Adam roundtrip differs")
    return named


def inspect_run(client, plan, row, entry, saved):
    config, driver = plan["config"], client.study
    seed, arm = row["run_key"].split(":")
    require(saved["run_key"] == row["run_key"] and saved["study_id"] == plan["study_id"],
            "saved run identity differs")
    adapter = client.adapter_for(arm, config, int(seed))
    require(driver.pilot.model_digest(adapter) == saved["initial_parameter_sha256"]
            == row["initial_parameter_sha256"], "saved initialization differs")
    parameters = dict(adapter.named_parameters())
    require(set(saved["adapter"]) == set(parameters) | {"_extra_state"}
            and set(parameters) == {"gate", "local_gate", "log_alpha"},
            "adapter parameter names differ")
    for name, parameter in parameters.items():
        value = saved["adapter"][name]
        require(isinstance(value, torch.Tensor) and value.shape == parameter.shape
                and value.dtype == parameter.dtype and bool(torch.isfinite(value).all()),
                f"invalid saved parameter: {name}")
    adapter.load_state_dict(saved["adapter"])
    require(equal(adapter.state_dict(), saved["adapter"]), "adapter roundtrip differs")
    optimizer = driver.make_optimizer(adapter, arm, config, None)
    moments = named_adam(saved, adapter, optimizer, config["steps"])
    require(row["checkpoint"] == entry["checkpoint"], "endpoint checkpoint differs")
    require(saved["records"] == row["records"] and saved["development"] == row["development"],
            "saved trajectory differs")
    require(saved["cursor"] == entry["cursor"] == config["steps"] == len(row["records"]),
            "saved cursor differs")
    require([r["batch_indices"] for r in row["records"]] == plan["batch_schedules"][seed][:-1]
            and [r["step"] for r in row["records"]] == list(range(1, config["steps"] + 1)),
            "saved batch schedule differs")
    require(row["parameter_count"] == row["trainable_parameter_count"]
            == sum(p.numel() for p in parameters.values()) == 2 * config["features"] + 1
            and all(p.requires_grad for p in parameters.values()), "saved capacity differs")
    require(float(adapter.log_alpha.detach()) == row["final_log_alpha"]
            == row["records"][-1]["log_alpha_after_update"]
            and float(adapter.log_alpha.detach().exp()) == row["final_alpha"]
            == row["records"][-1]["alpha_after_update"], "saved final order differs")
    require(saved.get("frozen_base_verified") is True
            and entry["resume_next_update_equal"] is True
            and entry["frozen_base_unchanged"] is True
            and entry["extra_validation_updates"] == 2, "missing saved invariants")
    return adapter, moments


def completed(directory, client):
    plan = json.loads((directory / "plan.json").read_bytes())
    journal = json.loads((directory / "journal.json").read_bytes())
    require(client.study.identity({key: value for key, value in plan.items()
                                   if key not in {"study_id", "source_revision", "batch_schedules"}})
            == plan["study_id"], "plan identity differs")
    result = client.study.completed_result(directory, journal, plan)
    require(result is not None, "study is not completed")
    rows = {row["run_key"]: row for row in result["runs"]}
    require(len(rows) == len(result["runs"]) and set(rows) == set(journal["runs"]),
            "endpoint run inventory differs")
    return plan, journal, result, rows


def filter_description(adapter):
    """Descriptive post-hoc probe; Rust produces every coefficient."""
    import spiraltorch as st

    recipe = adapter.get_extra_state()
    kernel = st.FractionalGlKernel(**recipe["kernel"])
    length = recipe["kernel"]["kernel_len"]
    impulse = [1.] + [0.] * (length - 1)
    alpha = float(adapter.log_alpha.detach().exp())
    if "gain" in recipe:
        snapshot = kernel.forward_history_l2(impulse, [length], 0, alpha, recipe["gain"])
    else:
        snapshot = kernel.forward_history(impulse, [length], 0, alpha)
    values = snapshot.output
    require(len(values) == length and values[0] == 0. and all(math.isfinite(x) for x in values),
            "invalid native impulse response")
    energy = math.fsum(x * x for x in values[1:])
    return {"backend": kernel.execution_backend, "kernel_len": length,
            "alpha": alpha, "coefficient_l2": math.sqrt(energy),
            "lag_3_plus_energy_fraction": math.fsum(x * x for x in values[3:]) / energy if energy else None,
            "first_two_past_coefficients": values[1:3],
            "scope": "Post-hoc native unit-impulse description, not a hidden-state variance or memory-quality test."}


def replay(current, previous, client, adapters, moments):
    plan, journal, result, rows = current
    old_plan, old_journal, old_result, old_rows, directory = previous
    require(old_plan["config"]["schema"] == "spiraltorch.fractional_two_lag_protocol.v1",
            "expected prior two-lag study")
    for key in ("data", "base_parameter_sha256", "model_config_sha256", "batch_schedules",
                "torch", "transformers"):
        require(equal(plan[key], old_plan[key]), f"historical comparison input differs: {key}")
    for key in ("model_snapshot", "corpus_sha256", "transfer_sha256", "block", "features",
                "seeds", "kernel", "steps", "batch_size", "block_size", "learning_rate", "strength", "threads"):
        require(equal(plan["config"][key], old_plan["config"][key]), f"historical recipe differs: {key}")
    report = {}
    for seed in plan["config"]["seeds"]:
        key, old_key = f"{seed}:history_raw_full", f"{seed}:history_learned_two"
        row, old_row = rows[key], old_rows[old_key]
        receipt = old_journal["runs"][old_key]["checkpoint"]
        saved = client.study.load_checkpoint(directory, receipt, old_plan["study_id"], old_key)
        require(receipt == old_row["checkpoint"] and saved["records"] == old_row["records"]
                and saved["development"] == old_row["development"], "historical checkpoint differs")
        require(saved["cursor"] == old_plan["config"]["steps"], "historical cursor differs")
        adapter = adapters[key]
        expected_recipe = {**adapter.get_extra_state(),
                           "study_schema": "spiraltorch.fractional_two_lag_control.v1",
                           "arm": "history_learned_two"}
        require(equal(saved["adapter"]["_extra_state"], expected_recipe), "historical adapter recipe differs")
        require(set(saved["adapter"]) == set(adapter.state_dict()), "historical parameters differ")
        old_moments = named_adam(saved, adapter,
                                client.study.make_optimizer(adapter, key.split(":")[1], plan["config"], None),
                                plan["config"]["steps"])
        parity = {"baseline_equal": equal(result["baseline"], old_result["baseline"]),
                  "records_equal": equal(row["records"], old_row["records"]),
                  "development_equal": equal(row["development"], old_row["development"]),
                  "endpoints_equal": equal(row["scores"], old_row["scores"]),
                  "initial_parameters_equal": row["initial_parameter_sha256"] == old_row["initial_parameter_sha256"],
                  "saved_parameters_equal": all(equal(value, saved["adapter"][name])
                                                for name, value in adapter.named_parameters()),
                  "named_adam_equal": equal(moments[key], old_moments)}
        report[str(seed)] = {"parity": parity, "current_checkpoint": row["checkpoint"],
                             "previous_checkpoint": receipt}
    return {"status": "exact_replay" if all(all(r["parity"].values()) for r in report.values()) else "mismatch",
            "current_study_id": plan["study_id"], "previous_study_id": old_plan["study_id"],
            "counts_as_additional_independent_seeds": False, "runs": report}


def verify(study, prior_study, summary_path, client, summary):
    current = completed(study, client)
    plan, journal, result, rows = current
    client.validate_protocol(plan["config"])
    input_hashes = {key: digest(study / f"{key}.json") for key in ("plan", "results", "journal")}
    rebuilt = summary.summarize(plan, result, journal, input_hashes["results"])
    rebuilt["input_sha256"] = input_hashes
    require((json.dumps(rebuilt, indent=2, allow_nan=False) + "\n").encode() == summary_path.read_bytes(),
            "summary artifact differs from reconstruction")
    adapters, moments, runs = {}, {}, {}
    for key, row in rows.items():
        entry = journal["runs"][key]
        saved = client.study.load_checkpoint(study, entry["checkpoint"], plan["study_id"], key)
        adapters[key], moments[key] = inspect_run(client, plan, row, entry, saved)
        runs[key] = {"checkpoint": entry["checkpoint"], "cursor": saved["cursor"],
                     "restored_adapter_and_adam_equal": True, "finite_parameters_and_moments": True,
                     "final_parameter_sha256": client.study.pilot.model_digest(adapters[key]),
                     "final_alpha": row["final_alpha"], "filter_description": filter_description(adapters[key])}
    historical = replay(current, (*completed(prior_study, client), prior_study), client, adapters, moments)
    return {"schema": "spiraltorch.fractional_history_factorial_checkpoint_verification.v1",
            "status": "passed", "study_id": plan["study_id"], "training_source_revision": plan["source_revision"],
            "planned_runs_verified": len(runs), "primary_updates": sum(r["cursor"] for r in runs.values()),
            "continuation_only_updates": sum(r["extra_validation_updates"] for r in journal["runs"].values()),
            "artifacts": input_hashes, "summary_artifact_sha256": digest(summary_path),
            "summary_source_sha256": digest(Path(summary.__file__)), "verifier_sha256": digest(Path(__file__)),
            "summary_rebuilt_byte_identical": True, "historical_raw_full": historical, "runs": runs,
            "scope": "Saved state and sealed receipts, not reexecution of continuation or model scoring. "
                     "State validity and historical replay are separate criteria; inspect both statuses."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "prior-study", "client-manifest", "runtime-manifest", "summary", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "verification output already exists")
    import hf_fractional_history_factorial as client
    import summarize_wave_gate_long_horizon as summary
    import spiraltorch as st
    import spiraltorch.spiraltorch as native
    import spiraltorch.geometry_autograd as geometry

    client_root, package_root = Path(client.__file__).parent, Path(st.__file__).parent.parent
    counts = {"client": verify_files(client_root, json.loads(args.client_manifest.read_bytes())),
              "runtime": verify_files(package_root, json.loads(args.runtime_manifest.read_bytes()))}
    plan = json.loads((args.study / "plan.json").read_bytes())
    for key, path in (("native_sha256", Path(native.__file__)), ("bridge_sha256", Path(geometry.__file__)),
                      ("script_sha256", Path(client.study.__file__)),
                      ("helper_sha256", Path(client.study.pilot.__file__)),
                      ("config_sha256", client_root / "hf_fractional_pride_history_factorial.json")):
        require(plan[key] == digest(path), f"runtime differs: {key}")
    require(plan["adapter_sources_sha256"] == {"history_factorial": digest(Path(client.__file__)),
                                               "fractional_bridge": digest(Path(client.fractional_bridge.__file__))},
            "adapter source differs")
    require(plan["config"] == json.loads((client_root / "hf_fractional_pride_history_factorial.json").read_bytes()),
            "plan recipe differs from frozen config")
    require(plan["torch"] == str(torch.__version__)
            and plan["transformers"] == str(client.study.pilot.transformers.__version__),
            "verification framework version differs")
    require(Path(summary.__file__).parent == client_root, "summary must come from the frozen client")
    torch.set_num_threads(plan["config"]["threads"])
    report = verify(args.study, args.prior_study, args.summary, client, summary)
    report["frozen_files_verified"] = counts
    report["manifest_sha256"] = {"client": digest(args.client_manifest), "runtime": digest(args.runtime_manifest)}
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"saved_state": report["status"], "historical_replay": report["historical_raw_full"]["status"],
                      "runs": report["planned_runs_verified"]}), flush=True)


if __name__ == "__main__":
    main()
