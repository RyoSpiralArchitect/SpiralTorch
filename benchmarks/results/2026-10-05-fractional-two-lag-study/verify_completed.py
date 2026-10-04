"""Read-only saved-state verification; never trains or scores a model."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import torch
import spiraltorch as st
import spiraltorch.spiraltorch as native
import spiraltorch.geometry_autograd as geometry
import hf_fractional_two_lag_study as client
import summarize_wave_gate_long_horizon as summary

if not __debug__:
    raise RuntimeError("checkpoint verification requires assertions enabled")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bitwise_equal(left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.shape == right.shape and left.dtype == right.dtype
                and torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                                right.contiguous().reshape(-1).view(torch.uint8)))
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(bitwise_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(bitwise_equal(a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        assert bool(torch.isfinite(value).all()), "nonfinite saved tensor"
    elif isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            finite_tree(child)
    elif isinstance(value, float):
        assert math.isfinite(value), "nonfinite saved scalar"


def main():
    parser = argparse.ArgumentParser()
    for name in ("study", "manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--analysis-manifest", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "verification output already exists"
    driver, pilot = client.study, client.study.pilot
    plan = json.loads((args.study / "plan.json").read_bytes())
    journal = json.loads((args.study / "journal.json").read_bytes())
    result = driver.completed_result(args.study, journal, plan)
    assert result is not None, "study is not completed"
    config = plan["config"]
    assert config["arms"] == client.ARMS
    torch.set_num_threads(config["threads"])
    manifest = json.loads(args.manifest.read_bytes())
    roots = {"client": Path(client.__file__).parent, "package": Path(st.__file__).parent}
    for group, root in roots.items():
        names = {str(path.relative_to(root)) for path in root.rglob("*")
                 if path.suffix in {".py", ".pyi", ".so", ".json"} or path.name == "py.typed"}
        assert names == set(manifest[group])
        for name, expected in manifest[group].items():
            assert digest(root / name) == expected, (group, name)
    for field, path in (
        ("native_sha256", Path(native.__file__)),
        ("bridge_sha256", Path(geometry.__file__)),
        ("script_sha256", Path(driver.__file__)),
        ("helper_sha256", Path(pilot.__file__)),
        ("config_sha256", roots["client"] / "hf_fractional_pride_two_lag.json"),
    ):
        assert digest(path) == plan[field], field
    assert plan["adapter_sources_sha256"] == {
        "two_lag_study": digest(Path(client.__file__)),
        "one_lag_control": digest(Path(client.one_lag.__file__)),
        "fractional_bridge": digest(Path(client.fractional_bridge.__file__)),
    }
    launch = json.loads((args.manifest.parent / "launch.json").read_text())
    assert launch["source_revision"] == plan["source_revision"]
    assert launch["study_id"] == plan["study_id"]
    analysis_path = args.analysis_manifest
    analysis = json.loads(analysis_path.read_text())
    assert Path(summary.__file__).parent == Path(__file__).parent
    assert analysis["summary_source_sha256"] == digest(Path(summary.__file__))
    summary_path = analysis_path.parent / "summary.json"
    receipt_summary_path = analysis_path.parent / "summary-receipts.json"
    assert analysis["summary_artifact_sha256"] == digest(summary_path)
    assert analysis["receipt_summary_artifact_sha256"] == digest(receipt_summary_path)
    assert analysis["training_source_revision"] == plan["source_revision"]
    assert analysis["study_id"] == plan["study_id"]
    assert analysis["training_client_manifest_sha256"] == digest(args.manifest.parent / "client-sha256.json")
    rows = {row["run_key"]: row for row in result["runs"]}
    keys = driver.endpoint_gate(journal, plan, args.study)
    assert len(rows) == len(result["runs"]) == len(keys)
    checked, saved_by_key, named_adam = {}, {}, {}
    for key in keys:
        seed, arm = key.split(":")
        entry, row = journal["runs"][key], rows[key]
        saved = driver.load_checkpoint(args.study, entry["checkpoint"], plan["study_id"], key)
        finite_tree(saved)
        adapter = client.adapter_for(arm, config, int(seed))
        initial = copy.deepcopy(adapter.state_dict())
        assert saved["initial_parameter_sha256"] == pilot.model_digest(adapter)
        adapter.load_state_dict(saved["adapter"])
        assert pilot.equal_state(adapter.state_dict(), saved["adapter"])
        optimizer = driver.make_optimizer(adapter, arm, config, None)
        optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
        assert pilot.equal_state(optimizer.state_dict(), saved["optimizer"])
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]["lr"] == config["learning_rate"]
        count = 2 * config["features"] + int(arm != "lag2")
        trainable = count - int(arm == "history_fixed_two")
        assert sum(p.numel() for p in adapter.parameters()) == row["parameter_count"] == count
        assert sum(p.numel() for p in adapter.parameters() if p.requires_grad) == row["trainable_parameter_count"] == trainable
        assert row["checkpoint"] == entry["checkpoint"]
        assert row["records"] == saved["records"]
        assert row["development"] == saved["development"]
        assert saved["cursor"] == entry["cursor"] == config["steps"] == len(saved["records"])
        records = saved["records"]
        assert [r["batch_indices"] for r in records] == plan["batch_schedules"][seed][:-1]
        assert [r["step"] for r in records] == list(range(1, config["steps"] + 1))
        parameters = dict(adapter.named_parameters())
        assert set(parameters) == {"gate", "local_gate"} | ({"log_alpha"} if arm != "lag2" else set())
        assert len(optimizer.state) == sum(p.requires_grad for p in parameters.values())
        named_adam[key] = {name: copy.deepcopy(optimizer.state[p])
                           for name, p in parameters.items() if p in optimizer.state}
        for parameter in parameters.values():
            if parameter.requires_grad:
                assert float(optimizer.state[parameter]["step"]) == config["steps"]
            else:
                assert parameter not in optimizer.state
        scalar = {}
        if arm != "lag2":
            alpha = adapter.log_alpha.detach().exp()
            learned = arm != "history_fixed_two"
            assert adapter.log_alpha.requires_grad is learned
            assert float(initial["log_alpha"]) == records[0]["log_alpha_before_update"]
            assert float(adapter.log_alpha.detach()) == row["final_log_alpha"] == records[-1]["log_alpha_after_update"]
            assert float(alpha) == row["final_alpha"] == records[-1]["alpha_after_update"]
            summary.order_trajectory(row, client.INITIAL_ORDERS[arm], learned)
            if not learned:
                assert torch.equal(adapter.log_alpha.detach(), initial["log_alpha"])
            scalar = {"final_alpha": float(alpha),
                      "scalar_changed": not torch.equal(adapter.log_alpha.detach(), initial["log_alpha"]),
                      "nonzero_order_gradient_steps": sum(r["log_alpha_gradient"] not in (None, 0) for r in records),
                      "frozen_scalar_has_no_adam_state": None if learned else "log_alpha" not in named_adam[key]}
        gates = {name: {"l2": float(parameters[name].detach().norm()),
                        "changed_from_initial": not torch.equal(parameters[name].detach(), initial[name]),
                        "nonzero_gradient_steps": sum(r[f"{name}_gradient_l2"] != 0 for r in records)}
                 for name in ("gate", "local_gate")}
        assert entry["extra_validation_updates"] == 2
        saved_by_key[key] = saved
        checked[key] = {"checkpoint": entry["checkpoint"], "cursor": saved["cursor"],
                        "parameter_count": count, "trainable_parameter_count": trainable,
                        "finite_adapter_and_adam": True, "restored_adam_equal": True,
                        "checkpoint_records_and_endpoint_receipts_match": True,
                        "final_parameter_sha256": pilot.model_digest(adapter), "gates": gates,
                        "frozen_base_unchanged": entry["frozen_base_unchanged"],
                        "resume_next_update_equal": entry["resume_next_update_equal"], **scalar}
    report_summary = summary.summarize(plan, result, journal, digest(args.study / "results.json"),
                                       checkpoint_dir=args.study)
    assert report_summary["same_math_parity_status"] == "passed"
    input_hashes = {name: digest(args.study / f"{name}.json") for name in ("plan", "results", "journal")}
    report_summary["input_sha256"] = input_hashes
    rebuilt = (json.dumps(report_summary, indent=2, allow_nan=False) + "\n").encode()
    assert rebuilt == summary_path.read_bytes(), "published full-state summary differs from saved-state reconstruction"
    receipt_summary = summary.summarize(plan, result, journal, input_hashes["results"])
    receipt_summary["input_sha256"] = input_hashes
    rebuilt_receipts = (json.dumps(receipt_summary, indent=2, allow_nan=False) + "\n").encode()
    assert rebuilt_receipts == receipt_summary_path.read_bytes(), "published receipt-only summary differs"
    parity = copy.deepcopy(report_summary["same_math_receipt_parity"])
    for seed in config["seeds"]:
        lag_key, fixed_key = (f"{seed}:{arm}" for arm in ("lag2", "history_fixed_two"))
        lag, fixed = saved_by_key[lag_key], saved_by_key[fixed_key]
        parity[str(seed)]["saved_gates_equal"] = all(
            bitwise_equal(lag["adapter"][name], fixed["adapter"][name]) for name in ("gate", "local_gate"))
        parity[str(seed)]["saved_named_adam_equal"] = bitwise_equal(named_adam[lag_key], named_adam[fixed_key])
        state_parity = report_summary["same_math_state_parity"][str(seed)]
        assert state_parity["saved_gates_equal"] == parity[str(seed)]["saved_gates_equal"]
        assert state_parity["saved_named_adam_equal"] == parity[str(seed)]["saved_named_adam_equal"]
        assert all(v is True for k, v in parity[str(seed)].items() if k != "endpoint_block_losses_equal")
        assert all(parity[str(seed)]["endpoint_block_losses_equal"].values())
    report = {"schema": "spiraltorch.fractional_two_lag_checkpoint_verification.v2",
              "status": "passed", "study_id": plan["study_id"],
              "planned_runs_verified": len(checked), "primary_updates": sum(r["cursor"] for r in checked.values()),
              "extra_continuation_updates": sum(r["extra_validation_updates"] for r in journal["runs"].values()),
              "artifacts": {name: digest(args.study / name) for name in ("plan.json", "journal.json", "results.json")},
              "frozen_client_files_verified": len(manifest["client"]),
              "frozen_package_files_verified": len(manifest["package"]),
              "analysis_source_revision": analysis["analysis_source_revision"],
              "summary_source_sha256": analysis["summary_source_sha256"],
              "summary_artifact_sha256": digest(summary_path),
              "receipt_summary_artifact_sha256": digest(receipt_summary_path),
              "published_summary_rebuilt_from_saved_states_byte_identical": True,
              "published_receipt_summary_rebuilt_byte_identical": True,
              "verifier_sha256": digest(Path(__file__)),
              "analysis_manifest_sha256": digest(analysis_path),
              "supersedes": analysis["supersedes"],
              "same_math_parity": parity, "runs": checked,
              "scope": "Saved contents and sealed receipts only; no training or model scoring. Not speed, significance or general-LLM evidence."}
    driver.atomic_json(args.output, report)
    print(json.dumps({"status": "passed", "runs": len(checked), "primary_updates": report["primary_updates"]}), flush=True)


if __name__ == "__main__":
    main()
