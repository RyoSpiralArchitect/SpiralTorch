#!/usr/bin/env python3
"""Offline geometry FT using public SpiralTorch dataset/artifact/Trainer APIs.

Uses a caller-pinned initial geometry artifact and a local HF base. This is a
bounded executable example, not a replacement for the existing full/LoRA CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import spiraltorch as st


def local_runtime(model_dir):
    import transformers as hf
    config = hf.AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    tokenizer = hf.AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    report = st.hf_causal_lm_runtime_identity_report(
        base_model_source=model_dir, base_model_revision=None,
        tokenizer_source=model_dir, tokenizer_source_kind="local",
        config=config, tokenizer=tokenizer)
    if report["status"] != "ready":
        raise ValueError(f"runtime identity is not ready: {report.get('errors')}")
    return config, tokenizer, report


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.named_parameters():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--geometry-artifact", type=Path, required=True)
    parser.add_argument("--geometry-artifact-id", required=True)
    parser.add_argument("--train-text", type=Path, action="append", required=True)
    parser.add_argument("--eval-text", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--geometry-checkpoint-id")
    parser.add_argument("--expected-dataset-id")
    parser.add_argument("--prompt", default="SpiralTorch is")
    args = parser.parse_args(argv)
    if args.max_steps <= 0 or args.save_steps <= 0 or args.batch_size <= 0:
        parser.error("step and batch counts must be positive")
    if bool(args.resume_from_checkpoint) != bool(args.geometry_checkpoint_id):
        parser.error("resume needs both checkpoint path and separately pinned geometry ID")
    if args.resume_from_checkpoint is not None and not args.expected_dataset_id:
        parser.error("resume needs --expected-dataset-id from the previous run")
    if args.stop_after is not None and not 0 < args.stop_after < args.max_steps:
        parser.error("stop-after must be strictly inside the declared run")
    if args.output.exists():
        parser.error("output must be a new directory; use a new destination for resume")
    if {p.resolve() for p in args.train_text} & {p.resolve() for p in args.eval_text}:
        parser.error("train and evaluation files must be disjoint")

    import torch
    import transformers as hf
    torch.set_num_threads(2)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("requested MPS device is unavailable")
    hf.set_seed(args.seed)
    config, tokenizer, runtime = local_runtime(args.model_dir)
    base_id = runtime["observed_identity_id"]
    initial = st.GeometryAdapterArtifact.from_json(
        args.geometry_artifact.read_text(), args.geometry_artifact_id, base_id)
    geometry = st.GeometryAdapterStack.from_artifact(initial, expected_base_model_identity=base_id)
    train_texts = [path.read_text(encoding="utf-8") for path in args.train_text]
    eval_texts = [path.read_text(encoding="utf-8") for path in args.eval_text]
    if set(train_texts) & set(eval_texts):
        raise ValueError("exact duplicate text across train/evaluation")
    train = st.HfGeometryDataset.from_texts(tokenizer, train_texts, args.context_length)
    evaluation = st.HfGeometryDataset.from_texts(tokenizer, eval_texts, args.context_length)
    data_id = st.hf_tokenized_dataset_identity_report(
        train_dataset=train, eval_dataset=evaluation, expected_identity_id=args.expected_dataset_id)
    if data_id["status"] != "ready":
        raise ValueError(f"dataset identity is not ready: {data_id.get('errors')}")
    config.use_cache = False
    model = hf.AutoModelForCausalLM.from_pretrained(
        args.model_dir, config=config, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32, attn_implementation="eager").requires_grad_(False)
    base_before = parameter_hash(model)
    args.output.mkdir(parents=True, exist_ok=False)
    training_args = hf.TrainingArguments(
        output_dir=str(args.output), use_cpu=args.device == "cpu", report_to=[], disable_tqdm=True,
        max_steps=args.max_steps, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, learning_rate=args.learning_rate,
        optim="adamw_torch", max_grad_norm=args.max_grad_norm, seed=args.seed, data_seed=args.seed,
        save_steps=args.save_steps, save_total_limit=2, logging_steps=args.save_steps,
        dataloader_pin_memory=False)
    if training_args.device.type != args.device:
        raise RuntimeError("HF selected a different device from the explicit request")
    bare = hf.Trainer(model=model, args=training_args, eval_dataset=evaluation,
                      data_collator=hf.default_data_collator).evaluate()

    class StopAfter(hf.TrainerCallback):
        def on_step_end(self, args_, state, control, **kwargs):
            if args.stop_after is not None and state.global_step == args.stop_after:
                control.should_save = control.should_training_stop = True

    trainer = st.hf_geometry_trainer_class()(
        model=model, geometry=geometry, base_model_identity=base_id, args=training_args,
        train_dataset=train, eval_dataset=evaluation, data_collator=hf.default_data_collator,
        callbacks=[StopAfter()], processing_class=tokenizer)
    frozen = trainer.evaluate()
    trained = trainer.train(
        resume_from_checkpoint=None if args.resume_from_checkpoint is None else str(args.resume_from_checkpoint),
        geometry_checkpoint_id=args.geometry_checkpoint_id)
    checkpoint_id = trainer.last_geometry_artifact_id
    after = trainer.evaluate()
    trainer.save_model(str(args.output / "geometry-export"))
    exported = st.GeometryAdapterArtifact.from_json(
        (args.output / "geometry-export/spiraltorch-geometry.json").read_text(),
        trainer.last_geometry_artifact_id, base_id)
    restored = st.GeometryAdapterStack.from_artifact(
        exported, expected_base_model_identity=base_id, device=training_args.device).eval()
    model.eval()
    prompt = tokenizer(args.prompt, return_tensors="pt").to(training_args.device)
    with torch.no_grad(), geometry.attach(model, require_all=True):
        expected_logits = model(**prompt, use_cache=False).logits
    with torch.no_grad(), restored.attach(model, require_all=True):
        actual_logits = model(**prompt, use_cache=False).logits
        ids = model.generate(**prompt, use_cache=False, do_sample=False, max_new_tokens=24,
                             pad_token_id=tokenizer.eos_token_id)
    if not torch.equal(expected_logits, actual_logits):
        raise RuntimeError("restored geometry changes inference logits")
    base_after = parameter_hash(model)
    if base_before != base_after:
        raise RuntimeError("frozen base parameters changed")
    result = {
        "schema": "spiraltorch.hf_geometry_example.v1", "device": args.device,
        "runtime_identity": base_id, "tokenized_dataset_identity": data_id["observed_identity_id"],
        "initial_artifact_id": initial.artifact_id, "initial_layout_id": initial.layout_id,
        "checkpoint_artifact_id": checkpoint_id, "artifact_id": exported.artifact_id,
        "step": trainer.state.global_step, "max_steps": args.max_steps, "seed": args.seed,
        "context_length": args.context_length, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "max_grad_norm": args.max_grad_norm,
        "train_window_plan": json.loads(train.plan.to_json()), "eval_window_plan": json.loads(evaluation.plan.to_json()),
        "base_parameter_hash": base_after, "base_unchanged": True, "restored_logits_exact": True,
        "bare_eval_loss": bare["eval_loss"], "initial_geometry_eval_loss": frozen["eval_loss"],
        "trained_geometry_eval_loss": after["eval_loss"], "train_metrics": trained.metrics,
        "generated_text": tokenizer.decode(ids[0], skip_special_tokens=True),
        "quality_status": "small_local_workflow_probe_not_general_quality_evidence",
        "run_config": {"attention": "eager", "dtype": "float32", "cache": False,
                       "optimizer": "adamw_torch", "data_seed": args.seed},
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("runtime_identity", "tokenized_dataset_identity", "step",
                      "artifact_id", "bare_eval_loss", "initial_geometry_eval_loss", "trained_geometry_eval_loss")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
