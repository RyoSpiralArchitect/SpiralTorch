"""Offline encoded-document geometry FT; Rust owns training and checkpoint semantics."""
import argparse
import hashlib
import json
from pathlib import Path

import spiraltorch as st


def read(path, limit):
    with path.open("rb") as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError("input file exceeds byte budget")
    return value


def digest(value):
    return "sha256:" + hashlib.sha256(value).hexdigest()


def save(directory, trainer):
    payload = trainer.to_json()
    identity = trainer.checkpoint_id
    if digest(payload.encode()) != identity:
        raise ValueError("checkpoint bytes and native identity disagree")
    path = directory / f"checkpoint-{trainer.completed_updates:06d}.json"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return {"file": path.name, "checkpoint_id": identity, "updates": trainer.completed_updates}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe-json", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--artifact-id")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--stop-after", type=int, help="absolute completed-update boundary, not a changed recipe")
    args = parser.parse_args(argv)
    if args.resume:
        if not args.checkpoint_id or any((args.recipe_json, args.artifact, args.artifact_id)):
            parser.error("resume needs a pinned checkpoint ID and no replacement recipe/artifact")
    elif not all((args.recipe_json, args.artifact, args.artifact_id)) or args.checkpoint_id:
        parser.error("fresh training needs recipe-json, artifact and artifact-id")
    if args.save_every < 1 or (args.stop_after is not None and args.stop_after < 0):
        parser.error("save-every must be positive and stop-after nonnegative")
    if args.output.exists():
        parser.error("output must be a new private directory")
    data_bytes = read(args.data_json, 256 * 1024 * 1024)
    data = json.loads(data_bytes)
    if not isinstance(data, dict) or set(data) - {"token_documents", "labels"} or "token_documents" not in data:
        raise ValueError("data JSON needs token_documents and optional unshifted document labels")
    weights = st.SafeTensorArchive(args.model_dir / "model.safetensors")
    base = st.NativeCausalLm.from_gpt2(read(args.model_dir / "config.json", 1_048_576).decode(), weights)
    del weights
    if args.resume:
        trainer = st.NativeCausalLmTrainer.from_json(base, data["token_documents"],
                    read(args.resume, 128 * 1024 * 1024).decode(), args.checkpoint_id, labels=data.get("labels"))
    else:
        recipe = json.loads(read(args.recipe_json, 65_536))
        if not isinstance(recipe, dict) or set(recipe) != {"context_length", "microbatch_windows", "accumulation_steps",
                "epochs", "learning_rate", "weight_decay", "label_smoothing", "max_gradient_norm"}:
            raise ValueError("provide the complete explicit native training and clipping recipe")
        artifact = st.GeometryAdapterArtifact.from_json(read(args.artifact, 64 * 1024 * 1024).decode(),
                                                        args.artifact_id, base.base_model_identity)
        optimizer = st.NativeGeometryAdamW(artifact, max_gradient_norm=recipe.pop("max_gradient_norm"))
        trainer = st.NativeCausalLmTrainer(base.with_geometry(artifact), optimizer, data["token_documents"],
                                          labels=data.get("labels"), **recipe)
    stop = trainer.total_updates if args.stop_after is None else args.stop_after
    if not trainer.completed_updates <= stop <= trainer.total_updates:
        raise ValueError("stop-after must be between the current cursor and the sealed run end")
    args.output.mkdir(parents=True, exist_ok=False)
    initial = save(args.output, trainer)
    saved = [initial]
    with (args.output / "steps.jsonl").open("x", encoding="utf-8") as log:
        while trainer.completed_updates < stop:
            report = trainer.step()
            log.write(report.to_json() + "\n")
            log.flush()
            print(report.to_json(), flush=True)
            if trainer.completed_updates % args.save_every == 0 or trainer.completed_updates == stop:
                saved.append(save(args.output, trainer))
    summary = {"schema": "spiraltorch.native_corpus_example.v1", "base_model_identity": base.base_model_identity,
               "data_file_id": digest(data_bytes), "data_id": trainer.data_id,
               "config": json.loads(trainer.config_json()), "plan": json.loads(trainer.plan_json()),
               "initial": initial, "checkpoints": saved, "completed_updates": trainer.completed_updates,
               "total_updates": trainer.total_updates, "exhausted": trainer.exhausted,
               "final_geometry_artifact_id": trainer.model.geometry_artifact_id,
               "scope": "geometry-only frozen-GPT2 native CPU training; not full-model FT or quality evidence"}
    with (args.output / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
