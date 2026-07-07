#!/usr/bin/env python3
"""Local GPT-2 fine-tuning bridge with SpiralTorch runtime/Z-Space preflight."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import spiraltorch as st
from spiraltorch.hf_ft import (
    HF_GPT2_FT_DEFAULT_DEVICE_BACKENDS,
    hf_gpt2_finetune_preflight_report,
    hf_gpt2_finetune_summary_lines,
    hf_gpt2_finetune_trainer_trace_callback,
    hf_gpt2_finetune_zspace_probe,
    summarize_hf_gpt2_finetune_trainer_trace,
    write_hf_gpt2_finetune_run_card,
)


DEFAULT_MODEL = "gpt2"
DEFAULT_DATASET = "wikitext"
DEFAULT_DATASET_CONFIG = "wikitext-2-raw-v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/hf-gpt2-ft"))
    parser.add_argument("--run-card", type=Path, default=None)
    parser.add_argument("--trainer-trace-jsonl", type=Path, default=None)
    parser.add_argument("--trainer-trace-run-id", default=None)
    parser.add_argument("--no-trainer-trace", action="store_true")
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--train", action="store_true", help="Actually run Trainer.train().")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Load model/tokenizer/dataset metadata but do not run Trainer.train().",
    )
    parser.add_argument("--max-train-samples", type=int, default=4096)
    parser.add_argument("--max-eval-samples", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--runtime-device-backend",
        action="append",
        default=[],
        help=(
            "SpiralTorch backend to report. Defaults to wgpu,cpu. This audits "
            "SpiralTorch runtime readiness; PyTorch still owns HF model kernels."
        ),
    )
    parser.add_argument(
        "--require-runtime-device-ready-backend",
        action="append",
        default=[],
        help="Fail preflight unless this SpiralTorch backend is runtime-ready.",
    )
    parser.add_argument(
        "--require-wgpu-ready",
        action="store_true",
        help="Shortcut for --require-runtime-device-ready-backend wgpu.",
    )
    parser.add_argument(
        "--no-require-hf-gpt2-ft",
        action="store_true",
        help="Report missing HF FT imports without failing the preflight gate.",
    )
    parser.add_argument("--zspace-probe", action="store_true")
    parser.add_argument("--zspace-probe-dim", type=int, default=64)
    parser.add_argument("--zspace-curvature", type=float, default=-0.04)
    parser.add_argument("--zspace-frequency", type=float, default=0.65)
    parser.add_argument("--zspace-strength", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.max_train_samples is not None and args.max_train_samples < 0:
        parser.error("--max-train-samples must be non-negative")
    if args.max_eval_samples is not None and args.max_eval_samples < 0:
        parser.error("--max-eval-samples must be non-negative")
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    if args.per_device_train_batch_size <= 0:
        parser.error("--per-device-train-batch-size must be positive")
    if args.per_device_eval_batch_size <= 0:
        parser.error("--per-device-eval-batch-size must be positive")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient-accumulation-steps must be positive")
    if args.metadata_only and args.train:
        parser.error("--metadata-only and --train are mutually exclusive")
    return args


def _module(name: str) -> Any:
    return importlib.import_module(name)


def _select_rows(dataset: Any, limit: int | None) -> Any:
    if limit is None or limit <= 0:
        return dataset
    count = min(int(limit), len(dataset))
    return dataset.select(range(count))


def _load_dataset_split(
    datasets: Any,
    args: argparse.Namespace,
    split: str | None,
) -> Any | None:
    if not split:
        return None
    kwargs = {"split": split}
    if args.dataset_config:
        return datasets.load_dataset(args.dataset_name, args.dataset_config, **kwargs)
    return datasets.load_dataset(args.dataset_name, **kwargs)


def _loader_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "local_files_only": not args.allow_remote,
        "trust_remote_code": args.trust_remote_code,
    }


def _tokenizer_vocab_size(tokenizer: Any) -> int | None:
    try:
        return int(len(tokenizer))
    except TypeError:
        vocab_size = getattr(tokenizer, "vocab_size", None)
        return None if vocab_size is None else int(vocab_size)


def _text_rows(dataset: Any, column: str, limit: int = 8) -> list[str]:
    rows = []
    for index in range(min(limit, len(dataset))):
        value = dataset[index].get(column)
        if isinstance(value, str) and value.strip():
            rows.append(value.strip())
    return rows


def _tokenize_dataset(dataset: Any, tokenizer: Any, args: argparse.Namespace) -> Any:
    if args.text_column not in getattr(dataset, "column_names", []):
        raise KeyError(
            f"dataset split does not contain text column {args.text_column!r}; "
            f"columns={getattr(dataset, 'column_names', [])!r}"
        )

    def tokenize(batch: Mapping[str, list[Any]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch[args.text_column])

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=list(getattr(dataset, "column_names", [])),
        desc="Tokenizing text",
    )

    def group_texts(
        examples: Mapping[str, list[list[int]]],
    ) -> dict[str, list[list[int]]]:
        concatenated = {
            key: sum((list(row) for row in rows), [])
            for key, rows in examples.items()
            if rows
        }
        if "input_ids" not in concatenated:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        total_length = (
            len(concatenated["input_ids"]) // args.block_size
        ) * args.block_size
        result = {
            key: [
                values[index : index + args.block_size]
                for index in range(0, total_length, args.block_size)
            ]
            for key, values in concatenated.items()
        }
        result["labels"] = list(result["input_ids"])
        return result

    return tokenized.map(group_texts, batched=True, desc="Grouping token blocks")


def _strategy_argument_name(training_arguments_cls: type) -> str:
    params = inspect.signature(training_arguments_cls.__init__).parameters
    if "eval_strategy" in params:
        return "eval_strategy"
    return "evaluation_strategy"


def _training_arguments_kwargs(
    args: argparse.Namespace,
    *,
    has_eval: bool,
    cls: type,
) -> dict[str, Any]:
    strategy_key = _strategy_argument_name(cls)
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "overwrite_output_dir": True,
        "do_train": bool(args.train),
        "do_eval": bool(has_eval),
        "num_train_epochs": float(args.num_train_epochs),
        "learning_rate": float(args.learning_rate),
        "per_device_train_batch_size": int(args.per_device_train_batch_size),
        "per_device_eval_batch_size": int(args.per_device_eval_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "logging_steps": int(args.logging_steps),
        "save_steps": int(args.save_steps),
        "save_total_limit": 2,
        "report_to": ["none"],
        "seed": int(args.seed),
        "save_strategy": "steps" if args.train else "no",
        strategy_key: "steps" if has_eval and args.train else "no",
    }
    if has_eval:
        kwargs["eval_steps"] = int(args.eval_steps)
    if args.max_steps is not None and args.max_steps > 0:
        kwargs["max_steps"] = int(args.max_steps)
    return kwargs


def _set_seed(torch: Any, transformers: Any, seed: int) -> None:
    random.seed(seed)
    set_seed = getattr(transformers, "set_seed", None)
    if callable(set_seed):
        set_seed(seed)
    manual_seed = getattr(torch, "manual_seed", None)
    if callable(manual_seed):
        manual_seed(seed)


def _runtime_backends(args: argparse.Namespace) -> list[str]:
    return args.runtime_device_backend or list(HF_GPT2_FT_DEFAULT_DEVICE_BACKENDS)


def _required_ready_backends(args: argparse.Namespace) -> list[str]:
    backends = list(args.require_runtime_device_ready_backend or [])
    if args.require_wgpu_ready and "wgpu" not in backends:
        backends.append("wgpu")
    return backends


def _write_card(card: Mapping[str, Any], args: argparse.Namespace) -> None:
    path = args.run_card or (args.output_dir / "spiraltorch-hf-gpt2-ft-run-card.json")
    write_hf_gpt2_finetune_run_card(card, path)
    print(f"run_card {path}")


def _trainer_trace_path(args: argparse.Namespace) -> Path | None:
    if args.no_trainer_trace:
        return None
    return args.trainer_trace_jsonl or (
        args.output_dir / "spiraltorch-hf-gpt2-ft-trainer-trace.jsonl"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preflight = hf_gpt2_finetune_preflight_report(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        train_split=args.train_split,
        eval_split=args.eval_split,
        text_column=args.text_column,
        runtime_device_backends=_runtime_backends(args),
        required_runtime_device_ready_backends=_required_ready_backends(args),
        require_hf_gpt2_ft=not args.no_require_hf_gpt2_ft,
    )
    for line in hf_gpt2_finetune_summary_lines(preflight):
        print(line)
    if not preflight["runtime_import_preflight_passed"]:
        _write_card(preflight, args)
        return 1
    if not args.train and not args.metadata_only:
        _write_card(preflight, args)
        return 0

    transformers = _module("transformers")
    torch = _module("torch")
    datasets = _module("datasets")
    _set_seed(torch, transformers, args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name,
        **_loader_kwargs(args),
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **_loader_kwargs(args),
    )
    if getattr(tokenizer, "pad_token_id", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    raw_train = _select_rows(
        _load_dataset_split(datasets, args, args.train_split),
        args.max_train_samples,
    )
    raw_eval = _load_dataset_split(datasets, args, args.eval_split)
    raw_eval = None if raw_eval is None else _select_rows(raw_eval, args.max_eval_samples)
    preview_texts = _text_rows(raw_train, args.text_column)

    zspace_probe = None
    preview_token_ids: list[int | float] = []
    if args.zspace_probe and preview_texts:
        encoded = tokenizer(preview_texts[0])
        preview_token_ids = list(encoded.get("input_ids", []))
        zspace_probe = hf_gpt2_finetune_zspace_probe(
            preview_token_ids,
            dim=args.zspace_probe_dim,
            vocab_size=_tokenizer_vocab_size(tokenizer),
            curvature=args.zspace_curvature,
            frequency=args.zspace_frequency,
            strength=args.zspace_strength,
        )

    card: dict[str, Any] = {
        "row_type": "hf_gpt2_finetune_run_card",
        "preflight": preflight,
        "spiraltorch_version": getattr(st, "__version__", None),
        "transformers_version": getattr(transformers, "__version__", None),
        "torch_version": getattr(torch, "__version__", None),
        "datasets_version": getattr(datasets, "__version__", None),
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "text_column": args.text_column,
        "raw_train_rows": len(raw_train),
        "raw_eval_rows": None if raw_eval is None else len(raw_eval),
        "block_size": args.block_size,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "zspace_probe": zspace_probe,
        "train_requested": bool(args.train),
        "metadata_only": bool(args.metadata_only),
        "trainer_trace_jsonl": (
            None if args.metadata_only else str(_trainer_trace_path(args))
        ),
    }
    if args.metadata_only:
        _write_card(card, args)
        return 0

    train_dataset = _tokenize_dataset(raw_train, tokenizer, args)
    eval_dataset = None if raw_eval is None else _tokenize_dataset(raw_eval, tokenizer, args)
    collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args_cls = transformers.TrainingArguments
    training_args = training_args_cls(
        **_training_arguments_kwargs(
            args,
            has_eval=eval_dataset is not None,
            cls=training_args_cls,
        )
    )
    trace_path = _trainer_trace_path(args)
    callbacks = []
    if trace_path is not None:
        callbacks.append(
            hf_gpt2_finetune_trainer_trace_callback(
                trace_path,
                run_id=args.trainer_trace_run_id or args.output_dir.name,
                zspace_probe_tokens=preview_token_ids if args.zspace_probe else None,
                zspace_probe_kwargs={
                    "dim": args.zspace_probe_dim,
                    "vocab_size": _tokenizer_vocab_size(tokenizer),
                    "curvature": args.zspace_curvature,
                    "frequency": args.zspace_frequency,
                    "strength": args.zspace_strength,
                },
            )
        )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    train_result = trainer.train()
    trainer.save_model(str(args.output_dir))
    trainer_trace_summary = (
        None
        if trace_path is None or not trace_path.exists()
        else summarize_hf_gpt2_finetune_trainer_trace(trace_path)
    )
    card.update(
        {
            "tokenized_train_rows": len(train_dataset),
            "tokenized_eval_rows": None if eval_dataset is None else len(eval_dataset),
            "trainer_metrics": dict(getattr(train_result, "metrics", {}) or {}),
            "trainer_trace_jsonl": None if trace_path is None else str(trace_path),
            "trainer_trace_summary": trainer_trace_summary,
            "model_saved": True,
        }
    )
    _write_card(card, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
