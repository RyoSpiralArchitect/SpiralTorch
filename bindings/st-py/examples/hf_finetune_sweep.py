#!/usr/bin/env python3
"""Run Hugging Face fine-tune sweeps and compare SpiralTorch run cards."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

EXAMPLES_ROOT = Path(__file__).resolve().parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

import hf_gpt2_finetune_sweep as _legacy  # noqa: E402
from hf_gpt2_finetune_sweep import *  # noqa: F401,F403,E402
from hf_gpt2_finetune_sweep import parse_args as _legacy_parse_args  # noqa: E402
import spiraltorch as st  # noqa: E402

DEFAULT_BRIDGE = Path(__file__).resolve().with_name("hf_finetune_bridge.py")
DEFAULT_OUT_DIR = Path("runs/hf-finetune-sweep")
DEFAULT_RUN_PREFIX = "hf-ft"
DEFAULT_RUN_CARD_FILENAME = "spiraltorch-hf-finetune-run-card.json"
DEFAULT_TRAINER_TRACE_FILENAME = "spiraltorch-hf-finetune-trainer-trace.jsonl"
DEFAULT_MODEL_PROFILE = st.HF_FINETUNE_DEFAULT_MODEL_PROFILE


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
    if not _argv_has_option(raw_argv, "--bridge-script"):
        args.bridge_script = DEFAULT_BRIDGE
    if not _argv_has_option(raw_argv, "--out-dir"):
        args.out_dir = DEFAULT_OUT_DIR
    if not _argv_has_option(raw_argv, "--run-prefix"):
        args.run_prefix = DEFAULT_RUN_PREFIX
    args.run_card_filename = DEFAULT_RUN_CARD_FILENAME
    args.trainer_trace_filename = DEFAULT_TRAINER_TRACE_FILENAME
    return args


def _genericize_text(value: str) -> str:
    if value.startswith("hf_gpt2_finetune_"):
        return "hf_finetune_" + value.removeprefix("hf_gpt2_finetune_")
    if value.startswith("hf_gpt2_ft_"):
        return "hf_ft_" + value.removeprefix("hf_gpt2_ft_")
    return value


def _genericize_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        payload = {str(key): _genericize_payload(item) for key, item in value.items()}
        row_type = payload.get("row_type")
        if isinstance(row_type, str):
            payload["row_type"] = _genericize_text(row_type)
        return payload
    if isinstance(value, list):
        return [_genericize_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_genericize_payload(item) for item in value]
    if isinstance(value, str):
        return _genericize_text(value)
    return value


def _rewrite_json(path: Path) -> None:
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(
        json.dumps(
            _genericize_payload(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _is_reusable_run_card(
    path: Path,
    *,
    require_adapter_promotion: bool = False,
) -> bool:
    if not path.is_file():
        return False
    try:
        card = st.load_hf_finetune_run_card(path)
    except (OSError, ValueError):
        return False
    if card.get("row_type") not in {
        "hf_finetune_run_card",
        "hf_gpt2_finetune_run_card",
    }:
        return False
    if card.get("failure_stage") or card.get("failure_error"):
        return False
    promotion = card.get("adapter_promotion")
    if require_adapter_promotion and (
        not isinstance(promotion, Mapping)
        or promotion.get("promotion_ready") is not True
    ):
        return False
    return True


def _install_generic_bindings() -> None:
    _legacy._is_reusable_run_card = _is_reusable_run_card


def run_sweep(args) -> dict[str, Any]:
    _install_generic_bindings()
    report = _genericize_payload(_legacy.run_sweep(args))
    _rewrite_json(args.out_dir / "sweep-plan.json")
    _rewrite_json(args.out_dir / "sweep-report.json")
    scale_up_path = report.get("scale_up_command_path")
    if isinstance(scale_up_path, str):
        _rewrite_json(Path(scale_up_path))
    return dict(report)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_sweep(args)
    report_path = args.out_dir / "sweep-report.json"
    print(f"sweep_report {report_path}")
    scale_up_command = report.get("scale_up_command")
    if isinstance(scale_up_command, dict):
        print(
            "scale_up_command "
            f"{report.get('scale_up_command_path')} "
            f"status={scale_up_command.get('status')}"
        )
        command_display = scale_up_command.get("command_display")
        if command_display:
            print(f"scale_up_replay {command_display}")
    failed = int(report.get("failed_run_count") or 0)
    summary = report.get("summary")
    if (
        args.adapter_promotion_gate
        and not args.dry_run
        and isinstance(summary, Mapping)
        and summary.get("status") == "no_promotion_ready_runs"
    ):
        return 1
    return 1 if failed and args.fail_fast else 0


if __name__ == "__main__":
    raise SystemExit(main())
