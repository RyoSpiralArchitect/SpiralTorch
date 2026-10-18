"""Offline public-example integration; no new corpus or language-quality claim."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[3]
PRIOR_PATH = Path(__file__).with_name("qualify.py")
PRIOR = runpy.run_path(str(PRIOR_PATH))
TRAIN = ROOT / "bindings/st-py/examples/native_geometry_corpus_training.py"
GENERATE = ROOT / "bindings/st-py/examples/native_geometry_generation.py"
digest, save, load = PRIOR["digest"], PRIOR["save"], PRIOR["load"]
TRAIN_RECIPE = {"context_length": 32, "microbatch_windows": 1, "accumulation_steps": 2,
                "epochs": 2, "learning_rate": .001, "weight_decay": .01,
                "label_smoothing": .1, "max_gradient_norm": .1,
                "repetition": PRIOR["CONFIGS"]["model_topk_periodic"]}
GEN_RECIPE = {"max_new_tokens": 4, "seed": 714, "decoding": {
    "selection": "sample", "top_k": 40, "top_p": .9,
    "distribution": {"kind": "zspace", "config": {"curvature": -1., "temperature": .8,
        "repression_window": 32, "repression_strength": .6, "last_token_repression": .3,
        "ngram_size": 3, "ngram_repression_strength": .4}}}}
EXECUTIONS = {"full_prefix": {"kind": "full_prefix"},
              "kv_cache": {"kind": "kv_cache", "max_values": 1_000_000}}


def contract(args, st):
    documents = PRIOR["pinned_input"](args)
    artifact = load(args.checkpoint)["geometry_json"]
    if "sha256:" + hashlib.sha256(artifact.encode()).hexdigest() != PRIOR["PRIOR"]["GEOMETRY"]:
        raise ValueError("unexpected saved geometry")
    inputs = {"data.json": json.dumps({"token_documents": documents}),
              "artifact.json": artifact, "train.json": json.dumps(TRAIN_RECIPE),
              "prompt.json": json.dumps(documents[0][:8])}
    inputs.update({kind + ".json": json.dumps({**GEN_RECIPE, "execution": execution})
                   for kind, execution in EXECUTIONS.items()})
    sources = [Path(__file__).resolve(), PRIOR_PATH, TRAIN, GENERATE,
               PRIOR["PRIOR_PATH"], PRIOR["PRIOR"]["TRAINING"],
               *[ROOT / p for p in PRIOR["SOURCES"]]]
    record = {"schema": "spiraltorch.native_example_integration.v1",
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
              "extension_sha256": digest(Path(st._rs.__file__)),
              "model_sha256": PRIOR["PRIOR"]["PRIOR"]["MODEL"],
              "data_sha256": PRIOR["PRIOR"]["DATA"], "checkpoint_sha256": PRIOR["PRIOR"]["CHECKPOINT"],
              "input_sha256": {name: hashlib.sha256(value.encode()).hexdigest() for name, value in inputs.items()},
              "training_recipe": TRAIN_RECIPE, "generation_recipe": GEN_RECIPE, "executions": EXECUTIONS,
              "readout": "f32", "documents": ["train_doc0[0:32]",
                  "synthetic:train_doc0[128:130]*15+train_doc0[130:132]"],
              "gates": {"training_resume_bytes_exact": True, "generation_resume_bytes_exact": True,
                        "all_step_reports_exact": True, "learned_geometry_changes": True},
              "scope": "public CLI mechanics on frozen GPT-2 with learned geometry; not quality, speed, or full FT"}
    return record, inputs


def check_run(directory, expected, files):
    actual = load(directory / "result.json")
    for name, value in expected.items():
        if actual.get(name) != value:
            raise ValueError("summary mismatch: " + directory.name + ":" + name)
    for name, payload in files.items():
        if (directory / name).read_bytes() != payload.encode():
            raise ValueError("native replay mismatch: " + directory.name + ":" + name)


def report(cohort, st, base):
    # Re-execute the sealed tiny study through Rust, rather than trusting mutually
    # consistent summaries or reconstructing training/generation semantics here.
    data, recipe = load(cohort / "data.json"), load(cohort / "train.json")
    artifact_path = cohort / "artifact.json"
    artifact = st.GeometryAdapterArtifact.from_json(artifact_path.read_text(),
                    "sha256:" + digest(artifact_path), base.base_model_identity)
    optimizer = st.NativeGeometryAdamW(artifact, max_gradient_norm=recipe.pop("max_gradient_norm"))
    trainer = st.NativeCausalLmTrainer(base.with_geometry(artifact), optimizer, data["token_documents"],
                                      labels=data.get("labels"), **recipe)
    if trainer.total_updates != 2:
        raise ValueError("qualification requires exactly two native updates")
    training = {"schema": "spiraltorch.native_corpus_example.v1", "base_model_identity": base.base_model_identity,
                "data_file_id": "sha256:" + digest(cohort / "data.json"), "data_id": trainer.data_id,
                "config": json.loads(trainer.config_json()), "plan": json.loads(trainer.plan_json()), "total_updates": 2}
    checkpoints, steps = [], []
    for index in range(3):
        checkpoints.append({"payload": trainer.to_json(), "identity": trainer.checkpoint_id,
                            "geometry": trainer.model.geometry_artifact_id, "exhausted": trainer.exhausted})
        if index < 2:
            steps.append(trainer.step().to_json() + "\n")
    if checkpoints[2]["geometry"] == artifact.artifact_id:
        raise ValueError("geometry did not change")
    for arm, start, end in (("full", 0, 2), ("part", 0, 1), ("resume", 1, 2)):
        entries = [{"file": "checkpoint-%06d.json" % i, "checkpoint_id": checkpoints[i]["identity"], "updates": i}
                   for i in range(start, end + 1)]
        expected = {**training, "initial": entries[0], "checkpoints": entries, "completed_updates": end,
                    "exhausted": checkpoints[end]["exhausted"], "final_geometry_artifact_id": checkpoints[end]["geometry"]}
        files = {entry["file"]: checkpoints[entry["updates"]]["payload"] for entry in entries}
        files["steps.jsonl"] = "".join(steps[start:end])
        check_run(cohort / ("train-" + arm), expected, files)
    generation = {}
    for kind, execution in EXECUTIONS.items():
        prompt, config = load(cohort / "prompt.json"), (cohort / (kind + ".json")).read_text()
        generator = st.NativeCausalLmGenerator(trainer.model, prompt, config)
        normalized = json.loads(generator.config_json())
        if normalized["max_new_tokens"] != 4 or normalized.get("execution", {"kind": "full_prefix"}) != execution:
            raise ValueError("unexpected generation qualification recipe")
        saved, reports = [], []
        for index in range(5):
            saved.append({"payload": generator.to_json(), "identity": generator.checkpoint_id,
                          "finished": generator.finished, "finish_reason": generator.finish_reason})
            if index < 4:
                step = generator.step()
                if step is None:
                    raise ValueError("generation ended before the declared boundary")
                reports.append(step.to_json() + "\n")
        for arm, start, end in (("full", 0, 4), ("part", 0, 2), ("resume", 2, 4)):
            expected = {"schema": "spiraltorch.native_generation_example.v1", "base_model_identity": base.base_model_identity,
                        "geometry_artifact_id": trainer.model.geometry_artifact_id, "config": normalized,
                        "execution": execution, "checkpoint_id": saved[end]["identity"],
                        "initial_id": saved[start]["identity"], "generated_tokens": end,
                        "finished": saved[end]["finished"], "finish_reason": saved[end]["finish_reason"]}
            check_run(cohort / (kind + "-" + arm), expected,
                      {"generation.json": saved[end]["payload"], "steps.jsonl": "".join(reports[start:end])})
        generation[kind] = {"generated_tokens": 4, "resume_checkpoint_exact": True,
                            "resume_reports_exact": True, "checkpoint_id": saved[4]["identity"]}
    return {"contract": load(cohort / "contract.json"),
            "validated_by": "Rust re-execution from sealed inputs; all summaries, checkpoints and ordered reports match",
            "training": {"updates": 2, "resume_checkpoint_exact": True, "resume_reports_exact": True,
                         "geometry_artifact_id": trainer.model.geometry_artifact_id,
                         "reports": [json.loads(line) for line in steps]},
            "generation": generation,
            "raw_sha256": {str(path.relative_to(cohort)): digest(path) for path in sorted(cohort.rglob("*")) if path.is_file()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prepare", "train", "generate", "report"], required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--arm", choices=["full", "part", "resume"])
    parser.add_argument("--execution", choices=EXECUTIONS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.mode in ("train", "generate")) != bool(args.arm) or (args.mode == "generate") != bool(args.execution):
        parser.error("train/generate need an arm, and only generate needs an execution")
    if (args.mode == "report") != bool(args.output):
        parser.error("only report needs an output path")
    PRIOR["PRIOR"]["PRIOR"]["forbid_ml"]()
    import spiraltorch as st
    expected, inputs = contract(args, st)
    if args.mode == "prepare":
        args.cohort.mkdir(mode=0o700, parents=True, exist_ok=False)
        for name, payload in inputs.items():
            with (args.cohort / name).open("x", encoding="utf-8") as stream:
                stream.write(payload)
        save(args.cohort / "contract.json", expected)
        return
    if load(args.cohort / "contract.json") != expected or any(
            digest(args.cohort / name) != sha for name, sha in expected["input_sha256"].items()):
        raise ValueError("sealed conditions changed")
    if args.mode == "report":
        base = st.NativeCausalLm.from_gpt2((args.model_dir / "config.json").read_text(),
                                          st.SafeTensorArchive(args.model_dir / "model.safetensors"))
        save(args.output, report(args.cohort, st, base))
        return
    prefix = "train" if args.mode == "train" else args.execution
    argv = ["--model-dir", str(args.model_dir), "--output", str(args.cohort / (prefix + "-" + args.arm))]
    if args.mode == "train":
        argv += ["--data-json", str(args.cohort / "data.json"), "--save-every", "1"]
        if args.arm != "resume":
            argv += ["--recipe-json", str(args.cohort / "train.json"), "--artifact", str(args.cohort / "artifact.json"),
                     "--artifact-id", PRIOR["PRIOR"]["GEOMETRY"]]
        else:
            path = args.cohort / "train-part/checkpoint-000001.json"
            argv += ["--resume", str(path), "--checkpoint-id", "sha256:" + digest(path)]
        if args.arm == "part":
            argv += ["--stop-after", "1"]
        runpy.run_path(str(TRAIN))["main"](argv)
    else:
        path = args.cohort / "train-full/checkpoint-000002.json"
        argv += ["--training-checkpoint", str(path), "--training-checkpoint-id", "sha256:" + digest(path)]
        if args.arm != "resume":
            argv += ["--prompt-json", str(args.cohort / "prompt.json"), "--recipe-json", str(args.cohort / (args.execution + ".json"))]
        else:
            path = args.cohort / (args.execution + "-part/generation.json")
            argv += ["--resume", str(path), "--resume-id", "sha256:" + digest(path)]
        if args.arm == "part":
            argv += ["--stop-after", "2"]
        runpy.run_path(str(GENERATE))["main"](argv)


if __name__ == "__main__":
    main()
