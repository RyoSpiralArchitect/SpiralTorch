"""Offline native GPT-2 generation with optional learned geometry, without Torch."""
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
    return value.decode("utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-json", type=Path)
    parser.add_argument("--recipe-json", type=Path)
    parser.add_argument("--training-checkpoint", type=Path)
    parser.add_argument("--training-checkpoint-id")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--artifact-id")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-id")
    parser.add_argument("--stop-after", type=int, help="absolute generated-token boundary; recipe stays sealed")
    args = parser.parse_args(argv)
    if bool(args.training_checkpoint) != bool(args.training_checkpoint_id) or bool(args.artifact) != bool(args.artifact_id):
        parser.error("every training checkpoint or geometry artifact needs its pinned identity")
    if args.training_checkpoint and args.artifact:
        parser.error("choose a training checkpoint OR an artifact")
    if args.resume:
        if not args.resume_id or args.prompt_json or args.recipe_json:
            parser.error("resume needs a pinned generation ID and no replacement prompt/recipe")
    elif args.resume_id or not args.prompt_json or not args.recipe_json:
        parser.error("fresh generation needs prompt-json and recipe-json")
    if args.output.exists():
        parser.error("output must be a new private directory")
    weights = st.SafeTensorArchive(args.model_dir / "model.safetensors")
    model = st.NativeCausalLm.from_gpt2(read(args.model_dir / "config.json", 1_048_576), weights)
    del weights
    if args.training_checkpoint:
        model = st.NativeCausalLm.from_training_checkpoint(model, read(args.training_checkpoint, 128 * 1024 * 1024),
                                                          args.training_checkpoint_id)
    elif args.artifact:
        artifact = st.GeometryAdapterArtifact.from_json(read(args.artifact, 64 * 1024 * 1024),
                                                        args.artifact_id, model.base_model_identity)
        model = model.with_geometry(artifact)
    if args.resume:
        generator = st.NativeCausalLmGenerator.from_json(model, read(args.resume, 1_048_576), args.resume_id)
    else:
        generator = st.NativeCausalLmGenerator(model, json.loads(read(args.prompt_json, 1_048_576)),
                                               read(args.recipe_json, 65_536))
    recipe = json.loads(generator.config_json())
    stop = recipe["max_new_tokens"] if args.stop_after is None else args.stop_after
    if not len(generator.generated_ids) <= stop <= recipe["max_new_tokens"]:
        raise ValueError("stop-after must lie between the current cursor and sealed generation end")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    initial_id = generator.checkpoint_id
    with (args.output / "steps.jsonl").open("x", encoding="utf-8") as stream:
        while not generator.finished and len(generator.generated_ids) < stop:
            stream.write(generator.step().to_json() + "\n")
            stream.flush()
    payload = generator.to_json()
    if "sha256:" + hashlib.sha256(payload.encode()).hexdigest() != generator.checkpoint_id:
        raise ValueError("generation bytes and native identity disagree")
    with (args.output / "generation.json").open("x", encoding="utf-8") as stream:
        stream.write(payload)
    result = {"schema": "spiraltorch.native_generation_example.v1", "base_model_identity": model.base_model_identity,
              "geometry_artifact_id": model.geometry_artifact_id, "config": recipe, "initial_id": initial_id,
              "execution": recipe.get("execution", {"kind": "full_prefix"}),
              "checkpoint_id": generator.checkpoint_id, "generated_tokens": len(generator.generated_ids),
              "finished": generator.finished, "finish_reason": generator.finish_reason,
              "scope": "frozen GPT-2 CPU generation; no corpus validation or quality claim"}
    with (args.output / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
