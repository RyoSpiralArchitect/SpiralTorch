#!/usr/bin/env python3
"""Build a Hugging Face fine-tune generation-control curve report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EXAMPLES_ROOT = Path(__file__).resolve().parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import spiraltorch as st  # noqa: E402
from hf_gpt2_ft_generation_curve import *  # noqa: F401,F403,E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser.add_argument(
        "sweeps",
        nargs="+",
        type=Path,
        help="Generation-control sweep JSON artifacts to join into the curve.",
    )
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument(
        "--model-configs",
        type=Path,
        default=None,
        help=(
            "Optional JSON config with Hugging Face model profiles. "
            "Useful for trace-only curves while a run card is still live."
        ),
    )
    parser.add_argument(
        "--model-profile",
        default=None,
        help=(
            "Model profile id used to fill model/dataset metadata unless "
            "--model-name/--dataset-* override it."
        ),
    )
    parser.add_argument(
        "--run-card",
        type=Path,
        default=None,
        help="Completed HF fine-tune run-card JSON. Optional while a run is live.",
    )
    parser.add_argument(
        "--trainer-trace-jsonl",
        type=Path,
        default=None,
        help="Live or completed trainer trace JSONL used when --run-card is absent.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory for live trace-only reports. Defaults to trace parent.",
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--lines-out", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args(argv)
    _apply_model_profile_defaults(args, parser=parser, raw_argv=raw_argv)
    if args.top_n < 0:
        parser.error("--top-n must be non-negative")
    if args.label and len(args.label) != len(args.sweeps):
        parser.error("--label must be repeated exactly once per sweep path")
    missing_sweeps = [path for path in args.sweeps if not path.is_file()]
    if missing_sweeps:
        parser.error(
            "sweep artifact does not exist: "
            + ", ".join(map(str, missing_sweeps))
        )
    if args.run_card is not None and not args.run_card.is_file():
        parser.error(f"run card does not exist: {args.run_card}")
    if args.trainer_trace_jsonl is not None and not args.trainer_trace_jsonl.is_file():
        parser.error(f"trainer trace does not exist: {args.trainer_trace_jsonl}")
    if args.run_card is None and args.trainer_trace_jsonl is None:
        parser.error("provide --run-card or --trainer-trace-jsonl")
    if args.run_dir is not None and not args.run_dir.exists():
        parser.error(f"run dir does not exist: {args.run_dir}")
    return args


def _argv_has_option(raw_argv: list[str], *names: str) -> bool:
    prefixes = tuple(f"{name}=" for name in names)
    return any(arg in names or arg.startswith(prefixes) for arg in raw_argv)


def _profile_section(profile: dict[str, Any], section: str) -> dict[str, Any]:
    value = profile.get(section)
    return dict(value) if isinstance(value, dict) else {}


def _apply_model_profile_defaults(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
    raw_argv: list[str],
) -> None:
    args._hf_finetune_model_profile = None
    if args.model_configs is None and args.model_profile is None:
        return
    try:
        profile = st.resolve_hf_finetune_model_profile(
            args.model_configs,
            profile=args.model_profile,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(f"failed to resolve model profile: {exc}")
    args._hf_finetune_model_profile = profile
    dataset = _profile_section(profile, "dataset")
    if args.model_name is None and not _argv_has_option(raw_argv, "--model-name"):
        args.model_name = profile.get("model_name")
    if args.dataset_name is None and not _argv_has_option(raw_argv, "--dataset-name"):
        args.dataset_name = dataset.get("name")
    if args.dataset_config is None and not _argv_has_option(
        raw_argv,
        "--dataset-config",
    ):
        args.dataset_config = dataset.get("config")


def _resolved_model_profile(args: argparse.Namespace) -> dict[str, Any] | None:
    profile = getattr(args, "_hf_finetune_model_profile", None)
    return dict(profile) if isinstance(profile, dict) else None


def _apply_profile_metadata(
    payload: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    profile = _resolved_model_profile(args)
    if profile is None:
        return payload
    payload.setdefault("model_profile_id", profile.get("profile_id"))
    payload.setdefault("model_profile_extends", profile.get("extends"))
    payload.setdefault("model_profile", profile)
    payload.setdefault("model_name", profile.get("model_name"))
    dataset = _profile_section(profile, "dataset")
    payload.setdefault("dataset_name", dataset.get("name"))
    payload.setdefault("dataset_config", dataset.get("config"))
    return payload


def _trace_only_card(args: argparse.Namespace) -> dict[str, Any]:
    trace_path = args.trainer_trace_jsonl
    if trace_path is None:
        raise TypeError("trace-only card requires --trainer-trace-jsonl")
    run_dir = args.run_dir or trace_path.parent
    return _apply_profile_metadata(
        {
            "row_type": "hf_finetune_run_card",
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "output_dir": str(run_dir),
            "run_dir": str(run_dir),
            "trainer_trace_jsonl": str(trace_path),
        },
        args,
    )


def _card_source(args: argparse.Namespace) -> str | Path | dict[str, Any]:
    if args.run_card is None:
        return _trace_only_card(args)
    if args.trainer_trace_jsonl is None and args.run_dir is None:
        payload = st.load_hf_finetune_run_card(args.run_card)
        return _apply_profile_metadata(payload, args)
    payload = st.load_hf_finetune_run_card(args.run_card)
    if args.trainer_trace_jsonl is not None:
        payload["trainer_trace_jsonl"] = str(args.trainer_trace_jsonl)
    if args.run_dir is not None:
        payload["run_dir"] = str(args.run_dir)
        payload.setdefault("output_dir", str(args.run_dir))
    if args.model_name is not None:
        payload["model_name"] = args.model_name
    if args.dataset_name is not None:
        payload["dataset_name"] = args.dataset_name
    if args.dataset_config is not None:
        payload["dataset_config"] = args.dataset_config
    return _apply_profile_metadata(payload, args)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    return st.hf_finetune_generation_curve_report(
        _card_source(args),
        list(args.sweeps),
        labels=list(args.label) or None,
        top_n=int(args.top_n),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    lines = st.hf_finetune_generation_curve_lines(
        report,
        top_n=int(args.top_n),
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"hf_ft_generation_curve_json {args.out}")
    if args.lines_out is not None:
        args.lines_out.parent.mkdir(parents=True, exist_ok=True)
        args.lines_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"hf_ft_generation_curve_lines {args.lines_out}")
    if args.out is None and args.lines_out is None:
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
