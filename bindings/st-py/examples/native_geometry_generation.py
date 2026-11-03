"""Offline native GPT-2 generation with optional learned geometry, without Torch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

import spiraltorch as st


def read_bytes(path, limit):
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= limit:
            raise ValueError("input must be a regular file within byte budget")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            value = stream.read(info.st_size)
            if len(value) != info.st_size or stream.read(1):
                raise ValueError("input size changed while reading")
    finally:
        os.close(fd)
    return value


def read(path, limit):
    return read_bytes(path, limit).decode("utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-json", type=Path)
    parser.add_argument("--recipe-json", type=Path)
    parser.add_argument("--training-checkpoint", type=Path)
    parser.add_argument("--training-checkpoint-id")
    parser.add_argument("--session-checkpoint", type=Path, help="native SGD LoRA/geometry checkpoint")
    parser.add_argument("--session-checkpoint-id", help="separately retained checkpoint SHA-256 identity")
    parser.add_argument("--session-boundary", type=Path, help="bounded metadata from the native training session example")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--artifact-id")
    parser.add_argument("--observed-run", type=Path, help="private source run with an explicitly selected observed pair")
    parser.add_argument("--observed-update", type=int)
    parser.add_argument("--observation-id")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-id")
    parser.add_argument("--stop-after", type=int, help="absolute generated-token boundary; recipe stays sealed")
    args = parser.parse_args(argv)
    if bool(args.training_checkpoint) != bool(args.training_checkpoint_id) or bool(args.artifact) != bool(args.artifact_id):
        parser.error("every training checkpoint or geometry artifact needs its pinned identity")
    observed = (args.observed_run is not None, args.observed_update is not None, args.observation_id is not None)
    if any(observed) and (not all(observed) or not args.observation_id or not 0 < args.observed_update < 2**64):
        parser.error("observed inference needs observed-run, a positive observed-update and observation-id")
    session = (args.session_checkpoint is not None, args.session_checkpoint_id is not None, args.session_boundary is not None)
    if any(session) and (not all(session) or not args.session_checkpoint_id):
        parser.error("session inference needs session-checkpoint, session-checkpoint-id and session-boundary")
    if sum((args.training_checkpoint is not None, args.artifact is not None, all(observed), all(session))) > 1:
        parser.error("choose just one inference model source")
    boundary = None
    session_payload = None
    if all(session):
        if not hasattr(st.NativeCausalLm, "from_session_checkpoint_at"):
            parser.error("session inference needs a matching source build after 0.4.33")
        meta = json.loads(read(args.session_boundary, 1_048_576))
        keys = ("recorded_data_id", "post_lora_id", "post_geometry_id")
        if (not isinstance(meta, dict) or meta.get("checkpoint_id") != args.session_checkpoint_id
                or "completed_updates" not in meta or not all(k in meta for k in keys)):
            raise ValueError("session boundary is missing or belongs to another checkpoint")
        boundary = dict(update=meta["completed_updates"], **{k:meta[k] for k in keys})
        session_payload = read_bytes(args.session_checkpoint, st.NativeCausalLmTrainingSession.MAX_CHECKPOINT_BYTES)
        if "sha256:" + hashlib.sha256(session_payload).hexdigest() != args.session_checkpoint_id:
            raise ValueError("session checkpoint differs from pinned identity")
    if all(observed) and not hasattr(st.NativeCausalLm, "from_training_checkpoint_at"):
        parser.error("observed inference needs a matching SpiralTorch source build after 0.4.33")
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
    elif boundary is not None:
        model = st.NativeCausalLm.from_session_checkpoint_at(model, session_payload, args.session_checkpoint_id, **boundary)
        del session_payload
    elif args.artifact:
        artifact = st.GeometryAdapterArtifact.from_json(read(args.artifact, 64 * 1024 * 1024),
                                                        args.artifact_id, model.base_model_identity)
        model = model.with_geometry(artifact)
    elif all(observed):
        from spiraltorch.native_corpus_runs import load_observed_model
        model = load_observed_model(args.observed_run, args.observed_update,
                                    expected_observation_id=args.observation_id, base=model)
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
    if all(observed):
        result["observed_training_boundary"] = {"update": args.observed_update, "observation_id": args.observation_id,
                                                "training_documents_verified": False}
    if boundary is not None:
        result["lora_artifact_id"] = model.lora_artifact_id
        result["session_boundary"] = {**boundary, "checkpoint_id":args.session_checkpoint_id,
                                      "training_documents_verified":False}
    with (args.output / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
