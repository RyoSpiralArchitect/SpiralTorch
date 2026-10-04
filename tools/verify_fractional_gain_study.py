#!/usr/bin/env python3
"""Verify saved log-order/angular gain states without training or scoring a model.

Use the frozen client/runtime before the repository tools on PYTHONPATH.
The shared verifier helpers are read-only; previous studies are not inputs.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

import verify_fractional_history_factorial as common

require = common.require
digest = common.digest
equal = common.equal


def tensor_receipt(value):
    require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
            and value.layout == torch.strided and value.dtype == torch.float32
            and bool(torch.isfinite(value).all()), "invalid saved float32 tensor")
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return {"dtype": "float32", "shape": list(value.shape),
            "sha256": hashlib.sha256(raw).hexdigest()}


def compare_tensor_maps(left, right):
    """Fixed preflight tolerance, reported separately from saved-state validity."""
    require(bool(left) and left.keys() == right.keys(), "paired tensor names differ")
    errors, byte_equal, value_equal, close = {}, True, True, True
    for name, a in left.items():
        b = right[name]
        ra, rb = tensor_receipt(a), tensor_receipt(b)
        require(ra["shape"] == rb["shape"], "paired tensor shapes differ")
        byte_equal = byte_equal and ra == rb
        value_equal = value_equal and torch.equal(a, b)
        close = close and torch.allclose(a, b, rtol=3e-6, atol=3e-7)
        errors[name] = float((a.detach().double() - b.detach().double()).abs().max())
    return {"status": "passed" if close else "failed", "allclose": bool(close),
            "byte_equal": bool(byte_equal), "value_equal": bool(value_equal),
            "rtol": 3e-6, "atol": 3e-7, "max_abs_error": errors}


def inspect_run(client, plan, row, entry, saved):
    config, driver = plan["config"], client.study
    seed, arm = row["run_key"].split(":")
    require(saved["run_key"] == row["run_key"] and saved["study_id"] == plan["study_id"],
            "saved run identity differs")
    adapter = client.adapter_for(arm, config, int(seed))
    require(driver.pilot.model_digest(adapter) == saved["initial_parameter_sha256"]
            == row["initial_parameter_sha256"], "saved initialization differs")
    parameters = dict(adapter.named_parameters())
    angular = config.get("schema") == "spiraltorch.fractional_angle_protocol.v1"
    shape_name = "history_angle" if angular or arm == client.ARMS[0] else "log_alpha"
    require(set(parameters) == {"gate", "local_gate", "log_gain", shape_name}
            and set(saved["adapter"]) == set(parameters) | {"_extra_state"},
            "adapter parameter names differ")
    require(equal(saved["adapter"]["_extra_state"], adapter.get_extra_state()),
            "saved adapter recipe differs")
    for name, parameter in parameters.items():
        tensor_receipt(saved["adapter"][name])
        require(saved["adapter"][name].shape == parameter.shape
                and saved["adapter"][name].dtype == parameter.dtype,
                f"invalid saved parameter: {name}")
    adapter.load_state_dict(saved["adapter"])
    require(equal(adapter.state_dict(), saved["adapter"]), "adapter roundtrip differs")
    optimizer = driver.make_optimizer(adapter, arm, config, None)
    # This fixed driver always constructs Adam from the complete registration order.
    require(equal(saved["optimizer"]["param_groups"][0]["params"],
                  optimizer.state_dict()["param_groups"][0]["params"]),
            "Adam parameter registration differs")
    moments = common.named_adam(saved, adapter, optimizer, config["steps"])
    for value in moments.values():
        for tensor in value.values():
            tensor_receipt(tensor)
    require(equal(row["checkpoint"], entry["checkpoint"]), "endpoint checkpoint differs")
    require(equal(saved["records"], row["records"])
            and equal(saved["development"], row["development"]), "saved trajectory differs")
    require(saved["cursor"] == entry["cursor"] == config["steps"] == len(row["records"]),
            "saved cursor differs")
    require([r["batch_indices"] for r in row["records"]] == plan["batch_schedules"][seed][:-1]
            and [r["step"] for r in row["records"]] == list(range(1, config["steps"] + 1)),
            "saved batch schedule differs")
    require(row["parameter_count"] == row["trainable_parameter_count"]
            == sum(p.numel() for p in parameters.values()) == 2 * config["features"] + 2
            and all(p.requires_grad for p in parameters.values()), "saved capacity differs")
    final = driver.pilot.gain_snapshot(adapter)
    if angular:
        final.update(driver.pilot.angle_snapshot(adapter))
    elif shape_name == "history_angle":
        final[shape_name] = float(adapter.history_angle.detach())
    else:
        final[shape_name] = float(adapter.log_alpha.detach())
        final["alpha"] = float(adapter.log_alpha.detach().exp())
    for name, value in final.items():
        require(equal(value, row[f"final_{name}"])
                and equal(value, row["records"][-1][f"{name}_after_update"]),
                f"saved final coordinate differs: {name}")
    require(saved.get("frozen_base_verified") is True and entry["status"] == "completed"
            and entry["resume_next_update_equal"] is True
            and row["resume_next_update_equal"] is True
            and entry["frozen_base_unchanged"] is True
            and entry["extra_validation_updates"] == 2, "missing saved invariants")
    return adapter, moments, final


def verify(study, summary_path, client, summary):
    plan, journal, result, rows = common.completed(study, client)
    client.validate_protocol(plan["config"])
    family = ("angle" if plan["config"]["schema"] == "spiraltorch.fractional_angle_protocol.v1"
              else "gain")
    require(result["schema"] == f"spiraltorch.fractional_{family}_study.v1",
            "gain study result schema differs")
    input_hashes = {key: digest(study / f"{key}.json") for key in ("plan", "results", "journal")}
    rebuilt = summary.summarize(plan, result, journal, input_hashes["results"])
    rebuilt["input_sha256"] = input_hashes
    require((json.dumps(rebuilt, indent=2, allow_nan=False) + "\n").encode() == summary_path.read_bytes(),
            "summary artifact differs from reconstruction")
    runs, states = {}, {}
    for key, row in rows.items():
        entry = journal["runs"][key]
        saved = client.study.load_checkpoint(study, entry["checkpoint"], plan["study_id"], key)
        adapter, moments, final = inspect_run(client, plan, row, entry, saved)
        if family == "angle":
            states[key] = {
                "parameters": {name: p.detach() for name, p in adapter.named_parameters()},
                "adam": {f"{name}/{field}": tensor for name, value in moments.items()
                         for field, tensor in value.items()},
            }
        runs[key] = {"checkpoint": entry["checkpoint"], "cursor": saved["cursor"],
                     "restored_adapter_and_adam_equal": True,
                     "final_parameter_sha256": client.study.pilot.model_digest(adapter),
                     "final_coordinates": final,
                     "parameters": {name: tensor_receipt(value)
                                    for name, value in adapter.named_parameters()},
                     "named_adam": {name: {field: tensor_receipt(tensor)
                                           for field, tensor in value.items()}
                                    for name, value in moments.items()}}
    paired = {}
    if family == "angle":
        for seed in plan["config"]["seeds"]:
            left, right = (states[f"{seed}:{arm}"] for arm in client.ARMS[:2])
            paired[str(seed)] = {field: compare_tensor_maps(left[field], right[field])
                                 for field in ("parameters", "adam")}
    report = {"schema": f"spiraltorch.fractional_{family}_checkpoint_verification.v1",
            "status": "passed", "study_id": plan["study_id"],
            "training_source_revision": plan["source_revision"], "planned_runs_verified": len(runs),
            "primary_updates": sum(r["cursor"] for r in runs.values()),
            "continuation_only_updates": sum(r["extra_validation_updates"] for r in journal["runs"].values()),
            "artifacts": input_hashes, "summary_artifact_sha256": digest(summary_path),
            "summary_source_sha256": digest(Path(summary.__file__)),
            "verifier_sha256": digest(Path(__file__)),
            "verifier_helper_sha256": digest(Path(common.__file__)),
            "summary_rebuilt_byte_identical": True, "runs": runs,
            "scope": "Saved tensors, recipes, named Adam and sealed receipts; not reexecution of "
                     "continuation or model scoring. No historical replay or cross-arm state parity is claimed."}
    if family == "angle":
        report.update(paired_short_states=paired,
                      paired_short_state_status=("passed" if all(
                          item["allclose"] for row in paired.values() for item in row.values()) else "failed"),
                      scope="Saved recipes/tensors/named Adam and sealed receipts, plus a separately reported "
                            "fixed-tolerance comparison of final ordinary/GL short states. Not replay, full "
                            "gradient trajectory parity, bitwise equivalence, quality or process termination.")
    return report


def verify_environment(plan, client, summary, client_manifest, runtime_manifest):
    import spiraltorch as st
    import spiraltorch.spiraltorch as native
    import spiraltorch.geometry_autograd as geometry

    client_root, package_root = Path(client.__file__).parent, Path(st.__file__).parent.parent
    angular = hasattr(client, "gain")
    require((plan["config"].get("schema") == "spiraltorch.fractional_angle_protocol.v1") == angular,
            "client coordinate schema differs")
    control = client.gain if angular else client
    sources = ({"angle_study": client, "gain_control": control} if angular
               else {"gain_study": client})
    sources.update(lag_control=control.lag, fractional_bridge=control.fractional_bridge)
    config_name = "hf_fractional_pride_angle.json" if angular else "hf_fractional_pride_gain.json"
    runtime = json.loads(runtime_manifest.read_bytes())
    require(set(runtime) == {"source_revision", "files"}
            and isinstance(runtime["source_revision"], str)
            and len(runtime["source_revision"]) == 40
            and all(c in "0123456789abcdef" for c in runtime["source_revision"]),
            "invalid frozen runtime manifest")
    counts = {"client": common.verify_files(client_root, json.loads(client_manifest.read_bytes())),
              "runtime": common.verify_files(package_root, runtime["files"])}
    require(all(Path(module.__file__).parent == client_root
                for module in (control, control.lag, client.study, client.study.pilot, summary)),
            "helpers and summary must come from the frozen client")
    require(all(Path(module.__file__).parent == package_root / "spiraltorch"
                for module in (native, geometry, control.fractional_bridge)),
            "native and bridges must come from the frozen package")
    for key, path in (("native_sha256", Path(native.__file__)), ("bridge_sha256", Path(geometry.__file__)),
                      ("script_sha256", Path(client.study.__file__)),
                      ("helper_sha256", Path(client.study.pilot.__file__)),
                      ("config_sha256", client_root / config_name)):
        require(plan[key] == digest(path), f"runtime differs: {key}")
    require(plan["adapter_sources_sha256"] == {key: digest(Path(module.__file__))
                                              for key, module in sources.items()}, "adapter source differs")
    require(plan["config"] == json.loads((client_root / config_name).read_bytes()),
            "plan recipe differs from frozen config")
    require(plan["torch"] == str(torch.__version__)
            and plan["transformers"] == str(client.study.pilot.transformers.__version__),
            "verification framework version differs")
    # Build and launch revisions can differ by documentation-only commits; bytes bind execution.
    return {"frozen_files_verified": counts, "runtime_build_source_revision": runtime["source_revision"],
            "manifest_sha256": {"client": digest(client_manifest), "runtime": digest(runtime_manifest)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "client-manifest", "runtime-manifest", "summary", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--coordinate", choices=("log-order", "angle"), default="log-order")
    args = parser.parse_args()
    require(not args.output.exists(), "verification output already exists")
    if args.coordinate == "angle":
        import hf_fractional_angle_study as client
    else:
        import hf_fractional_gain_study as client
    import summarize_wave_gate_long_horizon as summary
    import spiraltorch as st

    for root in (args.study, Path(client.__file__).parent, Path(st.__file__).parent.parent):
        require(not args.output.resolve().is_relative_to(root.resolve()),
                "verification output must be outside frozen study/client/runtime")
    plan = json.loads((args.study / "plan.json").read_bytes())
    environment = verify_environment(plan, client, summary, args.client_manifest, args.runtime_manifest)
    torch.set_num_threads(plan["config"]["threads"])
    report = verify(args.study, args.summary, client, summary)
    report.update(environment)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"saved_state": report["status"], "runs": report["planned_runs_verified"],
                      "primary_updates": report["primary_updates"]}), flush=True)


if __name__ == "__main__":
    main()
