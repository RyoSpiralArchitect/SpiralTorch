#!/usr/bin/env python3
"""Local GPT-2 fine-tuning bridge with SpiralTorch runtime/Z-Space preflight."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import json
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import spiraltorch as st
from spiraltorch.hf_ft import (
    HF_GPT2_FT_DEFAULT_DEVICE_BACKENDS,
    hf_gpt2_finetune_corpus_file_report,
    hf_gpt2_finetune_corpus_scan_report,
    hf_gpt2_finetune_dataset_fit_report,
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
HF_OFFLINE_ENV_VARS = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--train-file",
        action="append",
        type=Path,
        default=[],
        help=(
            "Local corpus file for training. May be repeated. When present, "
            "--dataset-name/--dataset-config are bypassed."
        ),
    )
    parser.add_argument(
        "--validation-file",
        action="append",
        type=Path,
        default=[],
        help="Local corpus file for validation/eval. May be repeated.",
    )
    parser.add_argument(
        "--dataset-format",
        choices=("text", "json", "csv"),
        default="text",
        help="datasets.load_dataset builder used for --train-file inputs.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.0,
        help=(
            "If using --train-file without --validation-file, split this "
            "fraction from train as validation."
        ),
    )
    parser.add_argument(
        "--corpus-scan",
        action="store_true",
        help=(
            "Stream local corpus files before loading HF datasets and record "
            "line/byte/sample stats in the run card."
        ),
    )
    parser.add_argument(
        "--corpus-scan-max-bytes-per-file",
        type=int,
        default=0,
        help=(
            "Bound --corpus-scan per file. The default 0 scans each local "
            "file fully."
        ),
    )
    parser.add_argument(
        "--corpus-scan-sample-lines",
        type=int,
        default=8,
        help="Number of nonempty preview lines to retain per scanned local file.",
    )
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
    if args.validation_file and not args.train_file:
        parser.error("--validation-file requires --train-file")
    if args.validation_file and args.validation_fraction > 0.0:
        parser.error(
            "--validation-file and --validation-fraction are mutually exclusive"
        )
    if args.validation_fraction < 0.0 or args.validation_fraction >= 1.0:
        parser.error("--validation-fraction must be in [0.0, 1.0)")
    if args.corpus_scan and not args.train_file:
        parser.error("--corpus-scan requires --train-file")
    if args.corpus_scan_max_bytes_per_file < 0:
        parser.error("--corpus-scan-max-bytes-per-file must be non-negative")
    if args.corpus_scan_sample_lines < 0:
        parser.error("--corpus-scan-sample-lines must be non-negative")
    for path in [*args.train_file, *args.validation_file]:
        if not path.is_file():
            parser.error(f"local corpus file does not exist: {path}")
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


def _has_local_corpus(args: argparse.Namespace) -> bool:
    return bool(args.train_file)


def _local_data_files(args: argparse.Namespace) -> dict[str, list[str]]:
    data_files = {"train": [str(path) for path in args.train_file]}
    if args.validation_file:
        data_files["validation"] = [str(path) for path in args.validation_file]
    return data_files


def _corpus_file_report(args: argparse.Namespace) -> dict[str, object] | None:
    if not _has_local_corpus(args):
        return None
    return hf_gpt2_finetune_corpus_file_report(
        train_files=args.train_file,
        validation_files=args.validation_file,
        dataset_format=args.dataset_format,
        text_column=args.text_column,
    )


def _corpus_scan_report(args: argparse.Namespace) -> dict[str, object] | None:
    if not _has_local_corpus(args) or not args.corpus_scan:
        return None
    max_bytes = (
        None
        if args.corpus_scan_max_bytes_per_file <= 0
        else int(args.corpus_scan_max_bytes_per_file)
    )
    return hf_gpt2_finetune_corpus_scan_report(
        train_files=args.train_file,
        validation_files=args.validation_file,
        dataset_format=args.dataset_format,
        text_column=args.text_column,
        sample_line_limit=args.corpus_scan_sample_lines,
        max_bytes_per_file=max_bytes,
    )


def _attach_local_corpus_reports(
    card: dict[str, Any],
    args: argparse.Namespace,
    *,
    corpus_file_report: Mapping[str, object] | None,
    corpus_scan_report: Mapping[str, object] | None,
) -> dict[str, Any]:
    if not _has_local_corpus(args):
        return card
    card.update(
        {
            "dataset_source": "local_files",
            "dataset_format": args.dataset_format,
            "corpus_file_report": corpus_file_report,
            "corpus_scan_report": corpus_scan_report,
            "validation_fraction": args.validation_fraction,
        }
    )
    return card


def _load_raw_datasets(
    datasets: Any,
    args: argparse.Namespace,
) -> tuple[Any, Any | None, dict[str, object] | None]:
    if not _has_local_corpus(args):
        return (
            _load_dataset_split(datasets, args, args.train_split),
            _load_dataset_split(datasets, args, args.eval_split),
            None,
        )

    corpus_report = _corpus_file_report(args)
    loaded = datasets.load_dataset(
        args.dataset_format,
        data_files=_local_data_files(args),
    )
    raw_train = loaded["train"]
    raw_eval = loaded["validation"] if "validation" in loaded else None
    if raw_eval is None and args.validation_fraction > 0.0:
        split = raw_train.train_test_split(
            test_size=float(args.validation_fraction),
            seed=int(args.seed),
        )
        raw_train = split["train"]
        raw_eval = split["test"]
    return raw_train, raw_eval, corpus_report


def _loader_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "local_files_only": not args.allow_remote,
        "trust_remote_code": args.trust_remote_code,
    }


def _hf_remote_access_report(args: argparse.Namespace) -> dict[str, object]:
    return {
        "allow_remote": bool(args.allow_remote),
        "offline_env": {name: os.environ.get(name) for name in HF_OFFLINE_ENV_VARS},
        "offline_env_overridden": bool(args.allow_remote)
        and any(os.environ.get(name) for name in HF_OFFLINE_ENV_VARS),
    }


@contextlib.contextmanager
def _hf_remote_access(args: argparse.Namespace):
    if not args.allow_remote:
        yield
        return

    old_env = {name: os.environ.get(name) for name in HF_OFFLINE_ENV_VARS}
    patched_attrs = []
    try:
        for name in HF_OFFLINE_ENV_VARS:
            os.environ[name] = "0"
        for module_name, attr_names in (
            ("huggingface_hub.constants", ("HF_HUB_OFFLINE",)),
            ("transformers.utils.hub", ("HF_HUB_OFFLINE",)),
            ("datasets.config", ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")),
        ):
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            for attr_name in attr_names:
                if hasattr(module, attr_name):
                    patched_attrs.append((module, attr_name, getattr(module, attr_name)))
                    setattr(module, attr_name, False)
        yield
    finally:
        for module, attr_name, old_value in reversed(patched_attrs):
            setattr(module, attr_name, old_value)
        for name, old_value in old_env.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value


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


def _training_argument_parameter_names(training_arguments_cls: type) -> set[str] | None:
    params = inspect.signature(training_arguments_cls.__init__).parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return None
    return set(params)


def _filter_training_arguments_kwargs(
    training_arguments_cls: type,
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    names = _training_argument_parameter_names(training_arguments_cls)
    if names is None:
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in names}


def _dropped_training_arguments_kwargs(
    training_arguments_cls: type,
    kwargs: Mapping[str, Any],
) -> list[str]:
    names = _training_argument_parameter_names(training_arguments_cls)
    if names is None:
        return []
    return sorted(key for key in kwargs if key not in names)


def _raw_training_arguments_kwargs(
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


def _training_arguments_kwargs(
    args: argparse.Namespace,
    *,
    has_eval: bool,
    cls: type,
) -> dict[str, Any]:
    return _filter_training_arguments_kwargs(
        cls,
        _raw_training_arguments_kwargs(args, has_eval=has_eval, cls=cls),
    )


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


def _preflight_dataset_name(args: argparse.Namespace) -> str:
    return "local-files" if _has_local_corpus(args) else args.dataset_name


def _preflight_dataset_config(args: argparse.Namespace) -> str | None:
    return args.dataset_format if _has_local_corpus(args) else args.dataset_config


def _base_run_card(
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
    *,
    corpus_file_report: Mapping[str, object] | None,
    corpus_scan_report: Mapping[str, object] | None,
    transformers: Any = None,
    torch: Any = None,
    datasets: Any = None,
) -> dict[str, Any]:
    return {
        "row_type": "hf_gpt2_finetune_run_card",
        "preflight": preflight,
        "spiraltorch_version": getattr(st, "__version__", None),
        "transformers_version": getattr(transformers, "__version__", None),
        "torch_version": getattr(torch, "__version__", None),
        "datasets_version": getattr(datasets, "__version__", None),
        "model_name": args.model_name,
        "dataset_name": _preflight_dataset_name(args),
        "dataset_config": _preflight_dataset_config(args),
        "dataset_source": "local_files" if _has_local_corpus(args) else "hf_dataset",
        "dataset_format": args.dataset_format if _has_local_corpus(args) else None,
        "corpus_file_report": corpus_file_report,
        "corpus_scan_report": corpus_scan_report,
        "validation_fraction": (
            args.validation_fraction if _has_local_corpus(args) else None
        ),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "text_column": args.text_column,
        "block_size": args.block_size,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "zspace_probe": None,
        "train_requested": bool(args.train),
        "metadata_only": bool(args.metadata_only),
        "trainer_trace_jsonl": (
            None if args.metadata_only else str(_trainer_trace_path(args))
        ),
        "load_status": "pending",
        "failure_stage": None,
        "failure_error": None,
        "dataset_fit_report": None,
    }


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
    remote_access_report = _hf_remote_access_report(args)
    with _hf_remote_access(args):
        return _main_with_runtime_access(args, remote_access_report)


def _main_with_runtime_access(
    args: argparse.Namespace,
    remote_access_report: Mapping[str, object],
) -> int:
    corpus_file_report = _corpus_file_report(args)
    corpus_scan_report = _corpus_scan_report(args)
    preflight = hf_gpt2_finetune_preflight_report(
        model_name=args.model_name,
        dataset_name=_preflight_dataset_name(args),
        dataset_config=_preflight_dataset_config(args),
        train_split=args.train_split,
        eval_split=args.eval_split,
        text_column=args.text_column,
        runtime_device_backends=_runtime_backends(args),
        required_runtime_device_ready_backends=_required_ready_backends(args),
        require_hf_gpt2_ft=not args.no_require_hf_gpt2_ft,
    )
    preflight["hf_remote_access"] = dict(remote_access_report)
    _attach_local_corpus_reports(
        preflight,
        args,
        corpus_file_report=corpus_file_report,
        corpus_scan_report=corpus_scan_report,
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

    card = _base_run_card(
        args,
        preflight,
        corpus_file_report=corpus_file_report,
        corpus_scan_report=corpus_scan_report,
        transformers=transformers,
        torch=torch,
        datasets=datasets,
    )
    try:
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
    except Exception as exc:
        card.update(
            {
                "load_status": "error",
                "failure_stage": "model_tokenizer_load",
                "failure_error": f"{exc.__class__.__name__}: {exc}",
            }
        )
        _write_card(card, args)
        return 1
    card["load_status"] = "ok"

    raw_train, raw_eval, loaded_corpus_report = _load_raw_datasets(datasets, args)
    if loaded_corpus_report is not None:
        corpus_file_report = loaded_corpus_report
        card["corpus_file_report"] = corpus_file_report
    raw_train = _select_rows(raw_train, args.max_train_samples)
    raw_eval = (
        None if raw_eval is None else _select_rows(raw_eval, args.max_eval_samples)
    )
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

    card.update(
        {
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "text_column": args.text_column,
            "raw_train_rows": len(raw_train),
            "raw_eval_rows": None if raw_eval is None else len(raw_eval),
            "block_size": args.block_size,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "zspace_probe": zspace_probe,
        }
    )
    if args.metadata_only:
        _write_card(card, args)
        return 0

    try:
        train_dataset = _tokenize_dataset(raw_train, tokenizer, args)
        eval_dataset = (
            None if raw_eval is None else _tokenize_dataset(raw_eval, tokenizer, args)
        )
    except Exception as exc:
        card.update(
            {
                "failure_stage": "dataset_tokenize",
                "failure_error": f"{exc.__class__.__name__}: {exc}",
            }
        )
        _write_card(card, args)
        return 1
    tokenized_train_rows = len(train_dataset)
    tokenized_eval_rows = None if eval_dataset is None else len(eval_dataset)
    dataset_fit_report = hf_gpt2_finetune_dataset_fit_report(
        raw_train_rows=len(raw_train),
        raw_eval_rows=None if raw_eval is None else len(raw_eval),
        tokenized_train_rows=tokenized_train_rows,
        tokenized_eval_rows=tokenized_eval_rows,
        block_size=args.block_size,
    )
    card.update(
        {
            "tokenized_train_rows": tokenized_train_rows,
            "tokenized_eval_rows": tokenized_eval_rows,
            "dataset_fit_report": dataset_fit_report,
        }
    )
    if dataset_fit_report["train_ready"] is not True:
        card.update(
            {
                "failure_stage": "dataset_fit",
                "failure_error": (
                    "tokenized train split produced too few blocks: "
                    f"{dataset_fit_report['warnings']}"
                ),
            }
        )
        _write_card(card, args)
        return 1
    if dataset_fit_report["eval_dropped_empty"] is True:
        eval_dataset = None
        card["tokenized_eval_rows"] = tokenized_eval_rows
    collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    training_args_cls = transformers.TrainingArguments
    raw_training_kwargs = _raw_training_arguments_kwargs(
        args,
        has_eval=eval_dataset is not None,
        cls=training_args_cls,
    )
    training_kwargs = _filter_training_arguments_kwargs(
        training_args_cls,
        raw_training_kwargs,
    )
    card["training_arguments_kwargs"] = sorted(training_kwargs)
    card["training_arguments_dropped_kwargs"] = _dropped_training_arguments_kwargs(
        training_args_cls,
        raw_training_kwargs,
    )
    try:
        training_args = training_args_cls(**training_kwargs)
    except Exception as exc:
        card.update(
            {
                "failure_stage": "training_arguments_init",
                "failure_error": f"{exc.__class__.__name__}: {exc}",
            }
        )
        _write_card(card, args)
        return 1
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
    try:
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
    except Exception as exc:
        card.update(
            {
                "failure_stage": "trainer_train",
                "failure_error": f"{exc.__class__.__name__}: {exc}",
            }
        )
        _write_card(card, args)
        return 1
    trainer_trace_summary = (
        None
        if trace_path is None or not trace_path.exists()
        else summarize_hf_gpt2_finetune_trainer_trace(trace_path)
    )
    card.update(
        {
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
