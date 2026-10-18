#!/usr/bin/env python3
"""Bind the declared book split and zero-state geometry before any model update."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy


def identity(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def inputs(plan_path, expected_plan_id, data):
    api = runpy.run_path(str(Path(__file__).with_name("prepare.py")))
    raw = api["read_bounded"](plan_path, 65_536)
    if identity(raw) != expected_plan_id:
        raise ValueError("plan does not match the separately pinned declaration")
    plan = json.loads(raw)
    if plan["schema"] != "spiraltorch.hf_book_control_plan.v1" or plan["data_policy"] != api["DATA_POLICY"]:
        raise ValueError("unsupported book plan")
    manifest_bytes = api["read_bounded"](data / "manifest.json", 65_536)
    manifest = json.loads(manifest_bytes)
    if (manifest["schema"] != "spiraltorch.hf_book_preparation.v1"
            or manifest["plan_id"] != expected_plan_id
            or manifest["raw_source_id"] != "sha256:" + plan["source"]["raw_sha256"]
            or manifest["reserve_used"] is not False):
        raise ValueError("preparation is not the pinned unused-reserve corpus")
    documents = manifest["documents"]
    if (any(type(doc["chapter"]) is not int for doc in documents)
            or [doc["chapter"] for doc in documents] != list(range(1, 62))):
        raise ValueError("preparation must contain every ordered chapter exactly once")
    selected = {"train": [], "development": []}
    texts = {"train": [], "development": []}
    for doc in documents:
        chapter = doc["chapter"]
        split = "train" if chapter <= 49 else "development" if chapter <= 55 else "reserve"
        relative = f"{split}/chapter-{chapter:02d}.txt"
        if doc["split"] != split or doc["relative_path"] != relative:
            raise ValueError("chapter split or path changed")
        if split == "reserve":
            continue
        path = data / relative
        if path.is_symlink() or not path.resolve().is_relative_to(data.resolve()):
            raise ValueError("chapter must remain inside its prepared corpus directory")
        raw = api["read_bounded"](path, 5_000_000)
        if len(raw) != doc["bytes"] or hashlib.sha256(raw).hexdigest() != doc["sha256"]:
            raise ValueError("prepared chapter changed")
        selected[split].append(path)
        texts[split].append(raw.decode("utf-8"))
    return plan, manifest, selected, texts, identity(manifest_bytes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("plan.json"))
    parser.add_argument("--expected-plan-id", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preflight output must be a new private directory")
    plan, manifest, selected, texts, manifest_id = inputs(args.plan, args.expected_plan_id, args.data)
    root = Path(__file__).resolve().parents[3]
    example = root / "bindings/st-py/examples/hf_geometry_finetune.py"
    example_api = runpy.run_path(str(example))
    config, tokenizer, runtime = example_api["local_runtime"](args.model_dir)
    if (runtime["observed_identity_id"] != plan["model"]["runtime_identity"]
            or config.model_type != "gpt2" or config.n_embd != 768 or config.n_layer != 12):
        raise ValueError("local base/tokenizer does not match the declared GPT-2 runtime")
    import spiraltorch as st
    import torch
    import transformers
    import accelerate

    torch.set_num_threads(2)
    datasets = {split: st.HfGeometryDataset.from_texts(tokenizer, rows, plan["context_length"])
                for split, rows in texts.items()}
    dataset_id = st.hf_tokenized_dataset_identity_report(
        train_dataset=datasets["train"], eval_dataset=datasets["development"])
    if dataset_id["status"] != "ready":
        raise ValueError("tokenized dataset identity is unavailable")
    stacks, artifacts, gradients = {}, {}, {}
    kinds = {"ChannelAffineAdapter": st.ChannelAffineAdapter, "WaveGateAdapter": st.WaveGateAdapter}
    value = torch.linspace(-1., 1., 2 * 4 * 768).reshape(2, 4, 768)
    for arm in plan["arms"]:
        options = {key: value for key, value in arm.items() if key not in ("name", "type")}
        stack = st.GeometryAdapterStack({site: kinds[arm["type"]](**options) for site in plan["sites"]})
        if sum(p.numel() for p in stack.parameters() if p.requires_grad) != plan["trainable_values_per_arm"]:
            raise ValueError("trainable parameter budgets differ")
        for adapter in stack.adapters:
            output = adapter(value)
            if not torch.equal(output, value):
                raise ValueError("adapter is not identity-start")
            output.square().sum().backward()
        gradients[arm["name"]] = [p.grad.clone() for p in stack.parameters()]
        stack.zero_grad(set_to_none=True)
        artifacts[arm["name"]] = stack.export_artifact(base_model_identity=runtime["observed_identity_id"])
        stacks[arm["name"]] = stack
    ordinary = gradients["ordinary"]
    if not all(len(g) == len(ordinary) and all(torch.equal(a, b) for a, b in zip(ordinary, g))
               for g in gradients.values()):
        raise ValueError("initial branch parameter gradients differ")
    # Revalidate selected inputs after tokenization; never open reserve files.
    if inputs(args.plan, args.expected_plan_id, args.data)[4] != manifest_id:
        raise ValueError("preparation manifest changed during preflight")
    args.output.mkdir(parents=True, exist_ok=False)
    artifact_info = {}
    for arm, artifact in artifacts.items():
        (args.output / f"{arm}-initial.json").write_text(artifact.to_json(), encoding="utf-8")
        artifact_info[arm] = {"artifact_id": artifact.artifact_id, "layout_id": artifact.layout_id}
    result = {
        "schema": "spiraltorch.hf_book_preflight.v1", "plan_id": args.expected_plan_id,
        "preparation_manifest_id": manifest_id,
        "source_id": manifest["raw_source_id"], "runtime_identity": runtime["observed_identity_id"],
        "tokenized_dataset_identity": dataset_id["observed_identity_id"], "artifacts": artifact_info,
        "window_plans": {split: json.loads(data.plan.to_json()) for split, data in datasets.items()},
        "identity_start_exact": True, "initial_parameter_gradients_exact": True,
        "reserve_opened": False, "model_loaded": False, "optimizer_updates": 0,
        "versions": {"python": __import__("sys").version.split()[0], "torch": torch.__version__,
                     "transformers": transformers.__version__, "accelerate": accelerate.__version__},
        "sources": {str(p.relative_to(root)): identity(p.read_bytes()) for p in (
            Path(__file__), Path(__file__).with_name("prepare.py"), example)},
    }
    (args.output / "preflight.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("plan_id", "tokenized_dataset_identity",
                     "identity_start_exact", "initial_parameter_gradients_exact", "reserve_opened", "optimizer_updates")}))


if __name__ == "__main__":
    main()
