"""Auxiliary updates on checkpoint copies, for separate native-binary comparison."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
import transformers
import spiraltorch as st
import spiraltorch.spiraltorch as native
import hf_fractional_lag_study as lag
import hf_fractional_memory_study as memory


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ("validation-root", "data-root", "model-dir", "package-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    assert Path(st.__file__).resolve().parent == (args.package_root / "spiraltorch").resolve()
    assert not args.output.with_suffix(".pt").exists() and not args.output.with_suffix(".json").exists()
    driver, pilot = lag.study, lag.study.pilot
    torch.set_num_threads(2)
    first = args.validation_root / "2026-10-03-fractional-lag-study/study"
    previous = json.loads((first / "plan.json").read_bytes())
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.model_max_length = 10**9
    train, _, _, metadata = driver.prepare_data(
        tokenizer, (args.data_root / "pride_and_prejudice.txt").read_bytes(),
        (args.data_root / "alices_adventure_in_wonderland.txt").read_bytes(), previous["config"])
    assert metadata == previous["data"]
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.float32).cpu().eval().requires_grad_(False)
    model.config.use_cache = False
    base_hash = pilot.model_digest(model)
    assert base_hash == previous["base_parameter_sha256"]
    parent_name, child = previous["config"]["block"].rsplit(".", 1)
    parent = model.get_submodule(parent_name)
    original = parent.get_submodule(child)
    payload = {"base_parameter_sha256": base_hash, "runs": {}}
    report = {"schema": "spiraltorch.fractional_forward_pretrained_probe.v1",
              "native_sha256": digest(Path(native.__file__)), "probe_sha256": digest(Path(__file__)),
              "actual_shape": [2, 128, 768], "threads": 2, "runs": []}
    for client, family, arm in ((lag, "fractional-lag-study", "history_learned_one"),
                                (memory, "fractional-memory-study", "fractional_learned")):
        directory = args.validation_root / f"2026-10-03-{family}/study"
        plan = json.loads((directory / "plan.json").read_bytes())
        journal = json.loads((directory / "journal.json").read_bytes())
        assert driver.completed_result(directory, journal, plan) is not None
        assert plan["data"] == metadata and plan["base_parameter_sha256"] == base_hash
        before = {name: digest(directory / name) for name in ("plan.json", "journal.json", "results.json")}
        for seed in plan["config"]["seeds"]:
            key = f"{seed}:{arm}"
            saved = driver.load_checkpoint(directory, journal["runs"][key]["checkpoint"], plan["study_id"], key)
            batch = train[plan["batch_schedules"][str(seed)][-1]]
            adapter = client.adapter_for(arm, plan["config"], seed)
            adapter.load_state_dict(copy.deepcopy(saved["adapter"]))
            optimizer = driver.make_optimizer(adapter, arm, plan["config"], None)
            optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
            parent.add_module(child, torch.nn.Sequential(original, adapter))
            records, gradients = [], []
            try:
                for _ in range(2):
                    records.append(pilot.update(model, adapter, optimizer, batch))
                    gradients.append({name: parameter.grad.clone() for name, parameter in adapter.named_parameters()})
            finally:
                parent.add_module(child, original)
            assert all(record["log_alpha_gradient"] != 0 for record in records)
            name = family + "/" + key
            payload["runs"][name] = {"records": records, "gradients": gradients,
                                     "adapter": copy.deepcopy(adapter.state_dict()),
                                     "optimizer": copy.deepcopy(optimizer.state_dict())}
            report["runs"].append({"family": family, "run_key": key,
                                  "source_study_id": plan["study_id"], "checkpoint": journal["runs"][key]["checkpoint"],
                                  "auxiliary_updates": 2, "records": records})
            print(f"{name}: two auxiliary updates completed", flush=True)
        assert before == {name: digest(directory / name) for name in before}
    assert pilot.model_digest(model) == base_hash
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters())
    report.update(status="probe_completed_not_yet_compared", frozen_base_unchanged=True,
                  original_study_files_unchanged=True, heldout_losses_computed=False)
    torch.save(payload, args.output.with_suffix(".pt"))
    with args.output.with_suffix(".json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
