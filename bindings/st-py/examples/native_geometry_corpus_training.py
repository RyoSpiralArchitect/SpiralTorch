"""Offline encoded-document geometry FT; Rust owns training and checkpoint semantics."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

import spiraltorch as st


def read(path, limit):
    # Model inputs may be symlinks in a local HF cache; inspect the opened target.
    with open(path, "rb", opener=lambda p, f: os.open(p, f | getattr(os, "O_NONBLOCK", 0))) as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= limit:
            raise ValueError("expected a regular input file within byte budget")
        value = stream.read(info.st_size)
        if len(value) != info.st_size or stream.read(1):
            raise ValueError("input size changed during read")
        return value


def digest(value):
    return "sha256:" + hashlib.sha256(value).hexdigest()


def private_output(path):
    path.mkdir(mode=0o700, parents=True, exist_ok=False)


def read_documents(path):
    raw = read(path, 256 * 1024 * 1024)
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) - {"token_documents", "labels"} or "token_documents" not in data:
        raise ValueError("data JSON needs token_documents and optional unshifted document labels")
    return raw, data


def save(directory, trainer):
    path = directory / f"checkpoint-{trainer.completed_updates:06d}.json"
    if hasattr(trainer, "save_checkpoint_new"):
        identity = trainer.save_checkpoint_new(path)
    else:  # Plain runs also support the published 0.4.33 wheel.
        payload, identity = (trainer.checkpoint_with_id() if hasattr(trainer, "checkpoint_with_id")
                             else (trainer.to_json(), trainer.checkpoint_id))
        raw = payload.encode()
        if digest(raw) != identity:
            raise ValueError("checkpoint bytes and native identity disagree")
        fd, pending = tempfile.mkstemp(dir=directory, prefix=".pending-")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(pending, path)
        finally:
            os.unlink(pending)
        if os.name == "posix":
            fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    return {"file": path.name, "checkpoint_id": identity, "updates": trainer.completed_updates}


def advance(trainer, directory, stop, save_every, observe_every, eval_every):
    start = trainer.completed_updates
    end = min([stop] + [(start // n + 1) * n for n in (save_every, observe_every, eval_every) if n])
    observed = observe_every and end % observe_every == 0
    save_due = observed or end == stop or end % save_every == 0
    if save_due and end - start <= getattr(trainer, "MAX_CHECKPOINT_INTERVAL", 0):
        if observed:
            return trainer.advance_to_observed_checkpoint_new(end, directory)
        path = directory / f"checkpoint-{end:06d}.json"
        reports, pin = trainer.advance_to_checkpoint_new(end, path)
        return reports, {"file": path.name, "checkpoint_id": pin, "updates": end}
    # Preserve evaluation-only cuts and old-wheel cadence; no extra file or update.
    report = trainer.step()
    checkpoint = save(directory, trainer) if save_due and trainer.completed_updates == end else None
    return [report], checkpoint


def index_pin(stream, entry):
    stream.write(json.dumps(entry, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


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
    parser.add_argument("--observe-every", type=int, help="save an observed checkpoint every N absolute updates (opt-in)")
    parser.add_argument("--stop-after", type=int, help="absolute completed-update boundary, not a changed recipe")
    parser.add_argument("--eval-data-json", type=Path, help="separate encoded holdout corpus; native NLL, no smoothing")
    parser.add_argument("--eval-every", type=int, help="evaluation interval; defaults to save-every, plus initial/final")
    args = parser.parse_args(argv)
    if args.resume:
        if not args.checkpoint_id or any((args.recipe_json, args.artifact, args.artifact_id)):
            parser.error("resume needs a pinned checkpoint ID and no replacement recipe/artifact")
    elif not all((args.recipe_json, args.artifact, args.artifact_id)) or args.checkpoint_id:
        parser.error("fresh training needs recipe-json, artifact and artifact-id")
    if args.save_every < 1 or (args.stop_after is not None and args.stop_after < 0):
        parser.error("save-every must be positive and stop-after nonnegative")
    if args.observe_every is not None:
        if args.observe_every < 1:
            parser.error("observe-every must be positive")
        if not callable(getattr(st.NativeCausalLmTrainer, "advance_to_observed_checkpoint_new", None)):
            parser.error("observe-every needs a matching SpiralTorch source build after 0.4.33")
    if args.eval_every is not None and (args.eval_every < 1 or args.eval_data_json is None):
        parser.error("eval-every must be positive and requires eval-data-json")
    if args.output.exists():
        parser.error("output must be a new private directory")
    data_bytes, data = read_documents(args.data_json)
    evaluation_data = read_documents(args.eval_data_json) if args.eval_data_json else None
    weights = st.SafeTensorArchive(args.model_dir / "model.safetensors")
    base = st.NativeCausalLm.from_gpt2(read(args.model_dir / "config.json", 1_048_576).decode(), weights)
    del weights
    if args.resume:
        payload = read(args.resume, 128 * 1024 * 1024)
        trainer = st.NativeCausalLmTrainer.from_json(base, data["token_documents"],
                    payload if hasattr(st.NativeCausalLmTrainer, "save_checkpoint_new") else payload.decode(),
                    args.checkpoint_id, labels=data.get("labels"))
        del payload
    else:
        recipe = json.loads(read(args.recipe_json, 65_536))
        required = {"context_length", "microbatch_windows", "accumulation_steps",
                    "epochs", "learning_rate", "weight_decay", "label_smoothing", "max_gradient_norm"}
        if not isinstance(recipe, dict) or not required <= set(recipe) or set(recipe) - required - {"repetition", "max_log_radius_drift"}:
            raise ValueError("provide the complete explicit native training and clipping recipe")
        artifact = st.GeometryAdapterArtifact.from_json(read(args.artifact, 64 * 1024 * 1024).decode(),
                                                        args.artifact_id, base.base_model_identity)
        options = {"max_gradient_norm": recipe.pop("max_gradient_norm")}
        if "max_log_radius_drift" in recipe:
            if not hasattr(st.NativeGeometryAdamW, "max_log_radius_drift"):
                raise ValueError("max_log_radius_drift requires a matching source build after 0.4.33")
            options["max_log_radius_drift"] = recipe.pop("max_log_radius_drift")
        optimizer = st.NativeGeometryAdamW(artifact, **options)
        trainer = st.NativeCausalLmTrainer(base.with_geometry(artifact), optimizer, data["token_documents"],
                                          labels=data.get("labels"), **recipe)
    stop = trainer.total_updates if args.stop_after is None else args.stop_after
    if not trainer.completed_updates <= stop <= trainer.total_updates:
        raise ValueError("stop-after must be between the current cursor and the sealed run end")
    config = json.loads(trainer.config_json())

    def evaluate():
        if evaluation_data is None:
            return None
        heldout = evaluation_data[1]
        score = trainer.model.evaluate_documents(heldout["token_documents"], config["context_length"],
                    microbatch_windows=config["microbatch_windows"], labels=heldout.get("labels"))
        return {"completed_updates": trainer.completed_updates, "evaluation": json.loads(score.to_json())}

    latest_evaluation = evaluate()  # Validate and score the holdout before creating a run directory.
    evaluation_count = int(latest_evaluation is not None)
    private_output(args.output)
    observation_count = 0
    with ExitStack() as stack:
        checkpoint_index = stack.enter_context((args.output / "checkpoints.jsonl").open("x", encoding="utf-8"))
        log = stack.enter_context((args.output / "steps.jsonl").open("x", encoding="utf-8"))
        evaluation_log = (stack.enter_context((args.output / "evaluation.jsonl").open("x", encoding="utf-8"))
                          if evaluation_data else None)
        observation_index = (stack.enter_context((args.output / "observed-checkpoints.jsonl").open("x", encoding="utf-8"))
                             if args.observe_every is not None else None)
        # Open the indices first so the checkpoint's directory sync also covers their names.
        initial = save(args.output, trainer)
        saved = [initial]
        index_pin(checkpoint_index, initial)
        if evaluation_log:
            evaluation_log.write(json.dumps(latest_evaluation, allow_nan=False) + "\n")
            evaluation_log.flush()
        while trainer.completed_updates < stop:
            reports, checkpoint = advance(trainer, args.output, stop, args.save_every, args.observe_every,
                                          (args.eval_every or args.save_every) if evaluation_log else None)
            if checkpoint is not None:
                index_pin(checkpoint_index, checkpoint)
                saved.append(checkpoint)
                if "observation_id" in checkpoint:
                    index_pin(observation_index, checkpoint)
                    observation_count += 1
            for report in reports:
                raw = report.to_json()
                log.write(raw + "\n")
                log.flush()
                print(raw, flush=True)
            if evaluation_log and (trainer.completed_updates % (args.eval_every or args.save_every) == 0
                                   or trainer.completed_updates == stop):
                latest_evaluation = evaluate()
                evaluation_count += 1
                evaluation_log.write(json.dumps(latest_evaluation, allow_nan=False) + "\n")
                evaluation_log.flush()
    summary = {"schema": "spiraltorch.native_corpus_example.v1", "base_model_identity": base.base_model_identity,
               "data_file_id": digest(data_bytes), "data_id": trainer.data_id,
               "config": json.loads(trainer.config_json()), "plan": json.loads(trainer.plan_json()),
               "initial": initial, "checkpoints": saved, "checkpoint_index": "checkpoints.jsonl",
               "completed_updates": trainer.completed_updates,
               "total_updates": trainer.total_updates, "exhausted": trainer.exhausted,
               "final_geometry_artifact_id": trainer.model.geometry_artifact_id,
               "evaluation": None if evaluation_data is None else {
                   "data_file_id": digest(evaluation_data[0]), "file": "evaluation.jsonl",
                   "count": evaluation_count, "last": latest_evaluation,
                   "policy": "read-only native NLL; caller supplies the holdout; no automatic model selection"},
               "scope": "geometry-only frozen-GPT2 native CPU training; not full-model FT or quality evidence"}
    if args.observe_every is not None:
        summary["observations"] = {"every": args.observe_every, "count": observation_count,
                                   "file": "observed-checkpoints.jsonl"}
    with (args.output / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
