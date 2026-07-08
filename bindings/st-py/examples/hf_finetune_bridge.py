#!/usr/bin/env python3
"""Generic Hugging Face fine-tuning bridge with SpiralTorch Z-Space preflight."""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

import hf_gpt2_finetune_bridge as _legacy  # noqa: E402
from hf_gpt2_finetune_bridge import *  # noqa: F401,F403,E402
from hf_gpt2_finetune_bridge import parse_args as _legacy_parse_args  # noqa: E402
import spiraltorch as st  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("runs/hf-finetune")
DEFAULT_RUN_CARD_FILENAME = "spiraltorch-hf-finetune-run-card.json"
DEFAULT_TRAINER_TRACE_FILENAME = "spiraltorch-hf-finetune-trainer-trace.jsonl"
DEFAULT_MODEL_PROFILE = "causal-lm-local-smoke"


def _argv_has_option(raw_argv: list[str], *names: str) -> bool:
    prefixes = tuple(f"{name}=" for name in names)
    return any(arg in names or arg.startswith(prefixes) for arg in raw_argv)


def _with_default_model_profile(raw_argv: list[str]) -> list[str]:
    if _argv_has_option(raw_argv, "-h", "--help"):
        return raw_argv
    if _argv_has_option(raw_argv, "--model-configs", "--model-profile", "--model-name"):
        return raw_argv
    return ["--model-profile", DEFAULT_MODEL_PROFILE, *raw_argv]


def parse_args(argv: list[str] | None = None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _legacy_parse_args(_with_default_model_profile(raw_argv))
    if not _argv_has_option(raw_argv, "--output-dir"):
        args.output_dir = DEFAULT_OUTPUT_DIR
    if not _argv_has_option(raw_argv, "--run-card"):
        args.run_card = args.output_dir / DEFAULT_RUN_CARD_FILENAME
    if not _argv_has_option(raw_argv, "--trainer-trace-jsonl"):
        args.trainer_trace_jsonl = args.output_dir / DEFAULT_TRAINER_TRACE_FILENAME
    return args


def _install_generic_bindings() -> None:
    """Route legacy bridge internals through model-neutral SpiralTorch helpers."""

    bindings = {
        "hf_gpt2_finetune_corpus_file_report": st.hf_finetune_corpus_file_report,
        "hf_gpt2_finetune_corpus_scan_report": st.hf_finetune_corpus_scan_report,
        "hf_gpt2_finetune_dataset_fit_report": st.hf_finetune_dataset_fit_report,
        "hf_gpt2_finetune_disk_headroom_plan": st.hf_finetune_disk_headroom_plan,
        "hf_gpt2_finetune_eval_report": st.hf_finetune_eval_report,
        "hf_gpt2_finetune_generation_report": st.hf_finetune_generation_report,
        "hf_gpt2_finetune_inference_distortion_handoff_report": (
            st.hf_finetune_inference_distortion_handoff_report
        ),
        "hf_gpt2_finetune_inference_distortion_handoff_lines": (
            st.hf_finetune_inference_distortion_handoff_lines
        ),
        "hf_gpt2_finetune_preflight_report": st.hf_finetune_preflight_report,
        "hf_gpt2_finetune_summary_lines": st.hf_finetune_summary_lines,
        "hf_gpt2_finetune_trainer_trace_callback": (
            st.hf_finetune_trainer_trace_callback
        ),
        "hf_gpt2_finetune_zspace_probe": st.hf_finetune_zspace_probe,
        "summarize_hf_gpt2_finetune_trainer_trace": (
            st.summarize_hf_finetune_trainer_trace
        ),
        "write_hf_gpt2_finetune_run_card": st.write_hf_finetune_run_card,
    }
    for name, helper in bindings.items():
        setattr(_legacy, name, helper)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _install_generic_bindings()
    remote_access_report = _legacy._hf_remote_access_report(args)
    with _legacy._hf_remote_access(args):
        return _legacy._main_with_runtime_access(args, remote_access_report)


if __name__ == "__main__":
    raise SystemExit(main())
