"""Frozen-checkpoint interventions, not another training or speed comparison.

Rust still supplies the nonlinear features and the unmodified native condition.
Torch-only replacement terms are experimental controls, never production routes.
"""

import argparse
import copy
import json
import math
import subprocess
from pathlib import Path

import torch
import transformers
import spiraltorch as st

import hf_elliptic_gated_study as gated


study = gated.study
pilot = study.pilot
MODES = ["native", "local", "gain_only", "anchor", "prefix_mean"]
ARMS = ["gated_tangent", "gated_elliptic"]
SCHEMA = "spiraltorch.elliptic_context_ablation.v1"


class _AblationMixin:
    def __init__(self, features, *, mode, **options):
        study.require(mode in MODES, "unknown context intervention")
        self.mode = mode
        super().__init__(features, **options)

    def forward(self, value):
        study.require(
            not self.training and not torch.is_grad_enabled(),
            "context interventions require eval() and no_grad(); no retraining",
        )
        return super().forward(value)

    def _map_features(self, orientation):
        if self.mode == "native":
            return super()._map_features(orientation)
        if hasattr(self, "tangent"):
            features = self.anchor + orientation[..., 1:] @ self.tangent.T
        else:
            features = st.elliptic_warp_autograd(self._warp, orientation)
        if self.mode == "local":
            return features
        local = features.double()
        if self.mode == "gain_only":
            context = torch.zeros_like(local)
        elif self.mode == "anchor":
            context = torch.tensor(
                self._warp.map_orientations_batch([1.0, 0.0, 0.0]).features,
                dtype=torch.float64, device=features.device,
            )
        else:
            counts = torch.arange(
                1, features.shape[1] + 1, device=features.device, dtype=torch.float64
            ).reshape(1, -1, 1)
            # Inclusive prefix only, reset independently at each batch boundary.
            # Round the context to f32 just as the native attention output does.
            context = (local.cumsum(dim=1) / counts).float().double()
        mix = self.raw_mix.double().tanh()
        return ((1 - mix) * local + mix * context).float()


class AblatedTangent(_AblationMixin, gated.GatedTangentControl):
    pass


class AblatedElliptic(_AblationMixin, st.EllipticGatedCausalResidualAdapter):
    pass


def adapter_for(arm, config, seed, mode):
    study.require(arm in ARMS, "context ablation requires a gated checkpoint")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(seed)
        kind = AblatedTangent if arm == "gated_tangent" else AblatedElliptic
        return kind(
            config["features"], mode=mode, strength=config["strength"], **config["warp"]
        ).eval().requires_grad_(False)


def condition_order(config):
    # Reproduce EVERY original condition before allowing any intervention.
    return [
        (f"{seed}:{arm}", mode)
        for mode in MODES for seed in config["seeds"] for arm in ARMS
    ]


def validate_report(report, binding, parent_result):
    study.require(report["schema"] == SCHEMA, "wrong ablation schema")
    study.require(report["ablation_id"] == binding["ablation_id"], "ablation identity differs")
    study.require(report["status"] in {"evaluating", "completed"}, "invalid ablation status")
    rows = report["conditions"]
    expected = condition_order(binding["config"])
    study.require(
        [(r["run_key"], r["mode"]) for r in rows] == expected[:len(rows)]
        and len(rows) <= len(expected), "invalid or duplicate condition order",
    )
    source = {row["run_key"]: row for row in parent_result["runs"]}
    for row in rows:
        original = source[row["run_key"]]
        study.require(row["checkpoint"] == original["checkpoint"], "condition checkpoint differs")
        study.require(row["raw_mix"] == original["final_raw_mix"], "condition gate differs")
        study.require(row["adapter_state_unchanged"] is True, "condition mutated adapter")
        study.require(set(row["scores"]) == set(original["scores"]), "endpoint sets differ")
        for label, score in row["scores"].items():
            values = score["block_losses"]
            study.require(
                len(values) == len(original["scores"][label]["block_losses"])
                and bool(values) and all(math.isfinite(v) for v in values)
                and score["mean"] == sum(values) / len(values), "invalid block scores",
            )
        if row["mode"] == "native":
            study.require(row["scores"] == original["scores"], "native replay differs")
    if report["status"] == "completed":
        study.require(len(rows) == len(expected), "incomplete ablation")
        study.require(report.get("base_unchanged") is True, "base not verified")


def summarize(report):
    native = {
        row["run_key"]: row for row in report["conditions"] if row["mode"] == "native"
    }
    output = {}
    for arm in ARMS:
        output[arm] = {}
        for mode in MODES:
            rows = [r for r in report["conditions"]
                    if r["run_key"].endswith(":" + arm) and r["mode"] == mode]
            output[arm][mode] = {}
            for label in rows[0]["scores"]:
                paired = [
                    {"seed": int(r["run_key"].split(":")[0]),
                     "cross_entropy": r["scores"][label]["mean"],
                     "delta_from_native": r["scores"][label]["mean"]
                     - native[r["run_key"]]["scores"][label]["mean"]}
                    for r in rows
                ]
                output[arm][mode][label] = {
                    "per_seed": paired,
                    "mean_cross_entropy": sum(r["cross_entropy"] for r in paired) / len(paired),
                    "mean_delta_from_native": sum(r["delta_from_native"] for r in paired) / len(paired),
                }
    return {"schema": SCHEMA, "ablation_id": report["ablation_id"], "contrasts": output}


def persist_report(directory, report):
    payload = {k: v for k, v in report.items() if k != "payload_sha256"}
    report["payload_sha256"] = study.identity(payload)
    study.atomic_json(directory / "results.json", report)


def run_ablation(
    model, parent, child, original, evaluation, parent_plan, parent_journal,
    parent_result, parent_dir, binding, directory, *, after_condition=None,
):
    path = directory / "results.json"
    if path.exists():
        report = json.loads(path.read_text())
        payload = {k: v for k, v in report.items() if k != "payload_sha256"}
        study.require(report.get("payload_sha256") == study.identity(payload), "ablation result hash mismatch")
    else:
        report = {"schema": SCHEMA, "ablation_id": binding["ablation_id"],
                  "status": "evaluating", "conditions": []}
    validate_report(report, binding, parent_result)
    study.require(pilot.model_digest(model) == parent_plan["base_parameter_sha256"], "base differs")
    if report["status"] == "completed":
        print("Ablation already completed; conditions were not rerun.", flush=True)
        return report
    config = parent_plan["config"]
    for key, mode in condition_order(config)[len(report["conditions"]):]:
        receipt = parent_journal["runs"][key]["checkpoint"]
        saved = study.load_checkpoint(parent_dir, receipt, parent_plan["study_id"], key)
        seed, arm = key.split(":")
        adapter = adapter_for(arm, config, int(seed), mode)
        study.require(pilot.model_digest(adapter) == saved["initial_parameter_sha256"], "initialization differs")
        adapter.load_state_dict(saved["adapter"])
        before = copy.deepcopy(adapter.state_dict())
        parent.add_module(child, torch.nn.Sequential(original, adapter))
        try:
            scores = {label: study.per_block_loss(model, blocks, config["batch_size"])
                      for label, blocks in evaluation.items()}
        finally:
            parent.add_module(child, original)
        study.require(pilot.equal_state(before, adapter.state_dict()), "intervention mutated adapter")
        report["conditions"].append({
            "run_key": key, "mode": mode, "checkpoint": receipt,
            "raw_mix": float(adapter.raw_mix), "adapter_state_unchanged": True,
            "scores": scores,
        })
        validate_report(report, binding, parent_result)
        persist_report(directory, report)
        print(json.dumps({"run_key": key, "mode": mode, "completed": len(report["conditions"])}), flush=True)
        if after_condition is not None:
            after_condition(key, mode)
    study.require(pilot.model_digest(model) == parent_plan["base_parameter_sha256"], "base changed")
    report.update(status="completed", base_unchanged=True)
    validate_report(report, binding, parent_result)
    persist_report(directory, report)
    return report


def source_binding():
    import spiraltorch.spiraltorch as native
    import spiraltorch.geometry_autograd as bridge

    sha = lambda module: pilot.digest(Path(module.__file__).read_bytes())
    return {
        "script_sha256": sha(study), "helper_sha256": sha(pilot),
        "native_sha256": sha(native), "bridge_sha256": sha(bridge),
        "torch": str(torch.__version__), "transformers": str(transformers.__version__),
        "adapter_sources_sha256": {
            "study_adapter": sha(gated), "causal_control": sha(gated.causal),
            "pointwise_adapter": sha(gated.causal.pointwise),
            "elliptic_bridge": sha(gated.causal.pointwise.elliptic_bridge),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-study-dir", "model-dir", "corpus", "transfer-corpus", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    parent_plan = json.loads((args.parent_study_dir / "plan.json").read_text())
    plan_payload = {k: v for k, v in parent_plan.items()
                    if k not in {"study_id", "source_revision", "batch_schedules"}}
    study.require(study.identity(plan_payload) == parent_plan["study_id"], "parent plan hash differs")
    study.require(parent_plan["result_schema"] == "spiraltorch.elliptic_gated_study.v1", "wrong parent study")
    config = parent_plan["config"]
    study.require(config["arms"] == gated.ARMS, "parent arms differ")
    sources = source_binding()
    study.require(all(parent_plan.get(k) == v for k, v in sources.items()), "parent runtime/source binding differs")
    journal = json.loads((args.parent_study_dir / "journal.json").read_text())
    parent_result = study.completed_result(args.parent_study_dir, journal, parent_plan)
    study.require(parent_result is not None, "parent study must be completed")
    study.require(args.model_dir.name == config["model_snapshot"], "model snapshot differs")
    if args.resume:
        study.require(args.output_dir.is_dir(), "resume needs an existing directory")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    with study.study_lock(args.output_dir):
        torch.set_num_threads(config["threads"])
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        tokenizer.model_max_length = 10**9
        _, _, evaluation, data = study.prepare_data(
            tokenizer, args.corpus.read_bytes(), args.transfer_corpus.read_bytes(), config,
        )
        study.require(data == parent_plan["data"], "evaluation data binding differs")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_dir, local_files_only=True, torch_dtype=torch.float32,
        ).cpu().eval().requires_grad_(False)
        model.config.use_cache = False
        study.require(study.identity(model.config.to_dict()) == parent_plan["model_config_sha256"], "model config differs")
        binding = {
            "schema": SCHEMA, "parent_study_id": parent_plan["study_id"],
            "parent_file_sha256": {name: pilot.digest((args.parent_study_dir / name).read_bytes())
                                   for name in ("plan.json", "journal.json", "results.json")},
            "sources": sources, "intervention_sha256": pilot.digest(Path(__file__).read_bytes()),
            "config": config, "modes": MODES,
        }
        binding["ablation_id"] = study.identity(binding)
        if args.resume:
            existing = json.loads((args.output_dir / "plan.json").read_text())
            study.require({k: existing.get(k) for k in binding} == binding, "ablation resume binding differs")
            binding = existing
        else:
            binding["source_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            study.atomic_json(args.output_dir / "plan.json", binding)
        parent_name, child = config["block"].rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        original = parent.get_submodule(child)
        result = run_ablation(
            model, parent, child, original, evaluation, parent_plan, journal,
            parent_result, args.parent_study_dir, binding, args.output_dir,
        )
        study.atomic_json(args.output_dir / "summary.json", summarize(result))


if __name__ == "__main__":
    main()
