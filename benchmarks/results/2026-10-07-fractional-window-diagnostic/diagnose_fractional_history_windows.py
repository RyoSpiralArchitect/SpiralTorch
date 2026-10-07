#!/usr/bin/env python3
"""Read-only angular-checkpoint interventions, gated on exact full-score replay.

Put the frozen study client and the candidate native package on PYTHONPATH.
Original runtime files are verified separately, not imported or substituted
into the original plan. No optimizer, training, checkpoint write or selection.
"""

import argparse
import json
import math
from pathlib import Path

import torch

import verify_fractional_history_factorial as common

require, digest, equal = common.require, common.digest, common.equal
ARM = "history_angle_full"


def windows(length):
    require(type(length) is int and length > 3, "full history must have long taps")
    return {"full": None, "retained_short": (1, 3),
            "retained_tail": (3, length), "local_only": (1, 1)}


def intervention(client, config, saved, mode):
    """Explicit parameter-only transform; never rewrite the saved recipe."""
    window = windows(config["kernel"]["kernel_len"])[mode]
    source = client.adapter_for(ARM, config, 0)
    require(client.study.pilot.model_digest(source) == saved["initial_parameter_sha256"],
            "saved initialization differs")
    state = saved["adapter"]
    require(set(state) == set(source.state_dict())
            and equal(state["_extra_state"], source.get_extra_state()), "saved full recipe differs")
    for name, parameter in source.named_parameters():
        value = state[name]
        require(isinstance(value, torch.Tensor) and value.shape == parameter.shape
                and value.dtype == parameter.dtype and value.device.type == "cpu"
                and bool(torch.isfinite(value).all()), f"invalid saved parameter: {name}")
    source.load_state_dict(state)
    require(equal(source.state_dict(), state), "saved full state roundtrip differs")
    target = source
    if window is not None:
        target = client.StudyAngularHistory(config["features"], arm=ARM,
                    strength=config["strength"], lag_window=window, **config["kernel"])
        require(dict(target.named_parameters()).keys() == dict(source.named_parameters()).keys(),
                "intervention parameter names differ")
        with torch.no_grad():
            for name, value in target.named_parameters():
                value.copy_(state[name])
    require(all(equal(value, state[name]) for name, value in target.named_parameters()),
            "intervention changed parameter bits")
    target.eval().requires_grad_(False)
    target.alpha  # Reject chart exits before scoring, including an empty history.
    target.gain
    return target, {
        "lag_window": None if window is None else list(window),
        "normalization_kernel_len": config["kernel"]["kernel_len"],
        "source_recipe_sha256": client.study.identity(state["_extra_state"]),
        "intervention_recipe_sha256": client.study.identity(target.get_extra_state()),
        "parameter_sha256": client.study.pilot.model_digest(target),
        "parameter_bits_preserved": True,
    }


def score(model, config, evaluation, adapter, driver):
    parent_name, child = config["block"].rsplit(".", 1)
    parent = model.get_submodule(parent_name)
    original = parent.get_submodule(child)
    before = driver.pilot.model_digest(adapter)
    recipe = adapter.get_extra_state()
    parent.add_module(child, torch.nn.Sequential(original, adapter))
    try:
        scores = {label: driver.per_block_loss(model, blocks, config["batch_size"])
                  for label, blocks in evaluation.items()}
    finally:
        parent.add_module(child, original)
    require(before == driver.pilot.model_digest(adapter)
            and equal(recipe, adapter.get_extra_state()), "scoring mutated adapter")
    return scores


def replay_gate(actual, expected):
    require(actual.keys() == expected.keys(), "endpoint labels differ")
    errors = {}
    for label, row in actual.items():
        a, b = row["block_losses"], expected[label]["block_losses"]
        require(len(a) == len(b) > 0
                and all(type(x) in (int, float) and math.isfinite(x) for x in a + b),
                "invalid replay losses")
        errors[label] = max(abs(x - y) for x, y in zip(a, b))
    return {"exact": equal(actual, expected), "rtol": 0., "atol": 0.,
            "max_abs_block_error": errors}


def deltas(actual, full):
    require(actual.keys() == full.keys(), "delta endpoint labels differ")
    result = {}
    for label, row in actual.items():
        a, b = row["block_losses"], full[label]["block_losses"]
        require(len(a) == len(b) > 0, "delta block counts differ")
        values = [x - y for x, y in zip(a, b)]
        require(all(math.isfinite(x) for x in values), "nonfinite loss delta")
        result[label] = {"mean": sum(values) / len(values), "block_deltas": values}
    return result


def evaluate(model, evaluation, current, directory, client, report, persist):
    plan, journal, _, rows = current
    config, driver = plan["config"], client.study
    require(driver.pilot.model_digest(model) == plan["base_parameter_sha256"], "base differs")
    require({key: driver.block_hashes(value) for key, value in evaluation.items()}
            == plan["data"]["evaluation_block_hashes"], "evaluation block identity differs")
    report.update(status="full_replay", runs={})
    saved_runs = {}
    # All seeds pass the unchanged full route before ANY intervention is scored.
    for seed in config["seeds"]:
        key = f"{seed}:{ARM}"
        entry, row = journal["runs"][key], rows[key]
        require(equal(entry["checkpoint"], row["checkpoint"]), "endpoint checkpoint differs")
        saved = driver.load_checkpoint(directory, entry["checkpoint"], plan["study_id"], key)
        require(saved["cursor"] == config["steps"] and equal(saved["records"], row["records"])
                and equal(saved["development"], row["development"]), "saved endpoint differs")
        saved_runs[key] = saved
        adapter, receipt = intervention(client, config, saved, "full")
        full = score(model, config, evaluation, adapter, driver)
        report["runs"][key] = {"checkpoint": entry["checkpoint"],
            "full_replay": replay_gate(full, row["scores"]),
            "modes": {"full": {"scores": full, "receipt": receipt}}}
        persist(report)
    if not all(row["full_replay"]["exact"] for row in report["runs"].values()):
        report["status"] = "blocked_full_replay"
    else:
        report["status"] = "interventions"
        for key, saved in saved_runs.items():
            row = report["runs"][key]
            for mode in windows(config["kernel"]["kernel_len"]):
                if mode == "full":
                    continue
                adapter, receipt = intervention(client, config, saved, mode)
                scores = score(model, config, evaluation, adapter, driver)
                row["modes"][mode] = {"scores": scores, "receipt": receipt,
                    "delta_from_full": deltas(scores, row["modes"]["full"]["scores"])}
                persist(report)
        report["status"] = "completed"
    require(driver.pilot.model_digest(model) == plan["base_parameter_sha256"], "base changed")
    report["frozen_base_unchanged"] = True
    persist(report)
    return report


def verify_environment(args, plan, client):
    import spiraltorch as st
    import spiraltorch.spiraltorch as native
    import spiraltorch.fractional_autograd as fractional
    import spiraltorch.geometry_autograd as geometry

    client_root = Path(client.__file__).resolve().parent
    package_root = Path(st.__file__).resolve().parent.parent
    original = json.loads(args.original_runtime_manifest.read_bytes())
    candidate = json.loads(args.candidate_runtime_manifest.read_bytes())
    counts = {
        "study": common.verify_files(args.study, json.loads(args.study_manifest.read_bytes())),
        "client": common.verify_files(client_root, json.loads(args.client_manifest.read_bytes())),
        "original_runtime": common.verify_files(args.original_runtime, original["files"]),
        "candidate_runtime": common.verify_files(package_root, candidate["files"]),
    }
    require(all(Path(module.__file__).resolve().parent == client_root for module in
                (client.gain, client.gain.lag, client.study, client.study.pilot)), "client roots differ")
    require(all(Path(module.__file__).resolve().parent == package_root / "spiraltorch"
                for module in (native, fractional, geometry)), "candidate package roots differ")
    original_files = original["files"]
    for key, path in (("script_sha256", Path(client.study.__file__)),
                      ("helper_sha256", Path(client.study.pilot.__file__)),
                      ("config_sha256", client_root / "hf_fractional_pride_angle.json")):
        require(plan[key] == digest(path), f"original client differs: {key}")
    require(plan["config"] == json.loads((client_root / "hf_fractional_pride_angle.json").read_bytes()),
            "original configuration differs")
    require(plan["native_sha256"] == original_files["spiraltorch/spiraltorch.abi3.so"]
            and plan["bridge_sha256"] == original_files["spiraltorch/geometry_autograd.py"],
            "original native binding differs")
    require(plan["adapter_sources_sha256"] == {
        "angle_study": digest(Path(client.__file__)), "gain_control": digest(Path(client.gain.__file__)),
        "lag_control": digest(Path(client.gain.lag.__file__)),
        "fractional_bridge": original_files["spiraltorch/fractional_autograd.py"],
    }, "original adapter binding differs")
    require(plan["torch"] == str(torch.__version__)
            and plan["transformers"] == str(client.study.pilot.transformers.__version__),
            "framework versions differ")
    return {"frozen_files_verified": counts,
            "original_runtime_source_revision": original["source_revision"],
            "candidate_runtime_source_revision": candidate["source_revision"],
            "original_native_sha256": plan["native_sha256"],
            "candidate_native_sha256": digest(Path(native.__file__)),
            "candidate_fractional_bridge_sha256": digest(Path(fractional.__file__)),
            "torch": str(torch.__version__), "transformers": plan["transformers"]}


def outside_inputs(output, roots):
    output = output.resolve()
    require(all(output != root.resolve() and root.resolve() not in output.parents for root in roots),
            "diagnostic output must be outside all input roots")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "study-manifest", "client-manifest", "original-runtime",
                 "original-runtime-manifest", "candidate-runtime-manifest", "model-dir",
                 "corpus", "transfer-corpus", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--diagnostic-source-revision", required=True)
    args = parser.parse_args()
    import hf_fractional_angle_study as client
    import spiraltorch as st

    outside_inputs(args.output_dir, (args.study, Path(client.__file__).parent, args.original_runtime,
                   Path(st.__file__).parent.parent, args.model_dir, args.corpus.parent,
                   args.transfer_corpus.parent, args.study_manifest.parent,
                   args.client_manifest.parent, args.original_runtime_manifest.parent,
                   args.candidate_runtime_manifest.parent, Path(__file__).parent))
    require(not args.output_dir.exists(), "diagnostic output already exists; never overwrite")
    require(len(args.diagnostic_source_revision) == 40
            and all(c in "0123456789abcdef" for c in args.diagnostic_source_revision),
            "invalid diagnostic source revision")
    current = common.completed(args.study, client)
    plan, _, _, _ = current
    client.validate_protocol(plan["config"])
    environment = verify_environment(args, plan, client)
    config, driver = plan["config"], client.study
    require(args.model_dir.name == config["model_snapshot"], "model snapshot differs")
    require(len(config["seeds"]) > 0 and len(set(config["seeds"])) == len(config["seeds"]),
            "empty or duplicate seeds")
    torch.set_num_threads(config["threads"])
    transformers = driver.pilot.transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.model_max_length = 10**9
    _, _, evaluation, data = driver.prepare_data(tokenizer, args.corpus.read_bytes(),
                                                args.transfer_corpus.read_bytes(), config)
    require(equal(data, plan["data"]), "reconstructed data differs")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.float32).cpu().eval().requires_grad_(False)
    model.config.use_cache = False
    require(driver.identity(model.config.to_dict()) == plan["model_config_sha256"], "model config differs")
    source = {"diagnostic": digest(Path(__file__)), "helper": digest(Path(common.__file__))}
    inputs = {name: digest(getattr(args, name)) for name in (
        "study_manifest", "client_manifest", "original_runtime_manifest", "candidate_runtime_manifest")}
    binding = {"schema": "spiraltorch.fractional_history_window_diagnostic.v1",
        "source_study_id": plan["study_id"], "training_source_revision": plan["source_revision"],
        "diagnostic_source_revision": args.diagnostic_source_revision,
        "input_sha256": {name: digest(args.study / f"{name}.json") for name in ("plan", "journal", "results")},
        "manifest_sha256": inputs, "source_sha256": source, "environment": environment,
        "seeds": config["seeds"], "arm": ARM, "windows": windows(config["kernel"]["kernel_len"]),
        "evaluation_block_hashes": data["evaluation_block_hashes"],
        "gate": {"all_seeds_before_interventions": True, "rtol": 0., "atol": 0.},
        "training_updates": 0, "optimizer_steps": 0,
        "scope": "Post-training dependence at fixed learned parameters, not matched-training causation, "
                 "independent heldout evidence, unique long-memory benefit, speed or general LLM quality."}
    report = {**binding, "diagnostic_id": driver.identity(binding), "status": "prepared"}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    driver.atomic_json(args.output_dir / "plan.json", report)

    def persist(value):
        driver.atomic_json(args.output_dir / "results.json", value)
        print(json.dumps({"status": value["status"], "completed_modes": {
            key: list(row["modes"]) for key, row in value.get("runs", {}).items()}}), flush=True)

    try:
        evaluate(model, evaluation, current, args.study, client, report, persist)
        require(environment == verify_environment(args, plan, client), "input files changed during scoring")
        require(source == {"diagnostic": digest(Path(__file__)), "helper": digest(Path(common.__file__))}
                and inputs == {name: digest(getattr(args, name)) for name in inputs}, "diagnostic sources changed")
        report["frozen_manifest_inputs_preserved"] = True
        persist(report)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        persist(report)
        raise
    require(report["status"] == "completed", "full-score replay failed; interventions withheld")


if __name__ == "__main__":
    main()
