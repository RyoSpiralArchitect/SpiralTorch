#!/usr/bin/env python3
"""Run chained char VAE context sweeps and summarize follow-up decisions."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "models" / "python" / "llm_char_vae_context.py"
SCHEMA = "st.llm_char_vae_context.chain.v1"
DEFAULT_FEATURES = "raw,reconstruction,latent,raw_latent,reconstruction_latent"
FOCUSED_HYBRID_FEATURES = "raw,latent,raw_latent,reconstruction_latent"
DEFAULT_NORMALIZE_MODES = "blocks,vector"
DEFAULT_FAIL_ON_VERDICT = "regressed,unknown"
SMOKE_LATENT_SCALES = "0.5,1.0"
SCOUT_LATENT_SCALES = "0.5,1.0,2.0,4.0"
HYBRID4_LATENT_SCALES = "2.0,4.0"
HYBRID4_DEEP_LATENT_SCALES = "4.0"

PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "features": DEFAULT_FEATURES,
        "feature_normalize_modes": DEFAULT_NORMALIZE_MODES,
        "head_init": "legacy",
        "window_chars": 20,
        "latent_dim": 5,
        "hidden": 8,
        "epochs": 1,
        "batches": 1,
        "batch_size": 2,
        "vae_epochs": 1,
        "vae_batches": 1,
        "vae_batch_size": 2,
        "eval_samples": 8,
        "gen": 0,
        "seeds": "7,13",
        "hybrid_latent_scales": SMOKE_LATENT_SCALES,
        "follow_up_seed_groups": "17;19",
    },
    "small": {
        "features": DEFAULT_FEATURES,
        "feature_normalize_modes": DEFAULT_NORMALIZE_MODES,
        "head_init": "legacy",
        "window_chars": 32,
        "latent_dim": 8,
        "hidden": 16,
        "epochs": 2,
        "batches": 4,
        "batch_size": 4,
        "vae_epochs": 2,
        "vae_batches": 4,
        "vae_batch_size": 4,
        "eval_samples": 32,
        "gen": 0,
        "seeds": "7,13,17",
        "hybrid_latent_scales": SCOUT_LATENT_SCALES,
        "follow_up_seed_groups": "19,23;29,31",
    },
    "hybrid4": {
        "features": FOCUSED_HYBRID_FEATURES,
        "feature_normalize_modes": "blocks",
        "head_init": "xavier",
        "window_chars": 32,
        "latent_dim": 8,
        "hidden": 16,
        "epochs": 8,
        "batches": 16,
        "batch_size": 4,
        "vae_epochs": 8,
        "vae_batches": 16,
        "vae_batch_size": 4,
        "eval_samples": 128,
        "gen": 0,
        "seeds": "2001,2003,2005",
        "hybrid_latent_scales": HYBRID4_LATENT_SCALES,
        "follow_up_seed_groups": "2007,2009,2011;2013,2015,2017",
    },
    "hybrid4_deep": {
        "features": FOCUSED_HYBRID_FEATURES,
        "feature_normalize_modes": "blocks",
        "head_init": "xavier",
        "window_chars": 32,
        "latent_dim": 8,
        "hidden": 16,
        "epochs": 10,
        "batches": 24,
        "batch_size": 4,
        "vae_epochs": 8,
        "vae_batches": 16,
        "vae_batch_size": 4,
        "eval_samples": 256,
        "gen": 0,
        "seeds": "2001,2003,2005",
        "hybrid_latent_scales": HYBRID4_DEEP_LATENT_SCALES,
        "follow_up_seed_groups": "2007,2009,2011;2013,2015,2017",
    },
    "base": {
        "features": DEFAULT_FEATURES,
        "feature_normalize_modes": DEFAULT_NORMALIZE_MODES,
        "head_init": "legacy",
        "window_chars": 48,
        "latent_dim": 16,
        "hidden": 32,
        "epochs": 3,
        "batches": 8,
        "batch_size": 8,
        "vae_epochs": 3,
        "vae_batches": 8,
        "vae_batch_size": 8,
        "eval_samples": 64,
        "gen": 0,
        "seeds": "7,13,17,19",
        "hybrid_latent_scales": SCOUT_LATENT_SCALES,
        "follow_up_seed_groups": "23,29,31;37,41,43",
    },
}


def _timestamp_slug() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_summary(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_groups(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def _follow_up_command_record(
    summary: dict[str, Any],
    *,
    index: int,
) -> tuple[dict[str, Any] | None, str | None]:
    guided = summary.get("guided_next_follow_up_command")
    if isinstance(guided, dict) and guided.get("enabled"):
        return guided, "guided_next_follow_up_command"
    next_command = summary.get("next_follow_up_command")
    if index == 1 and isinstance(next_command, dict):
        return next_command, "next_follow_up_command"
    return None, None


def _follow_up_new_seeds(
    command_record: dict[str, Any] | None,
    seed_groups: list[str],
    *,
    index: int,
    explicit_seed_groups: bool,
) -> tuple[str | None, str]:
    seed_group_index = index - 1
    if explicit_seed_groups:
        if seed_group_index < len(seed_groups):
            return seed_groups[seed_group_index], "explicit_seed_group"
        return None, "script_default"
    if isinstance(command_record, dict):
        default_new_seeds = command_record.get("default_new_seeds")
        if isinstance(default_new_seeds, str) and default_new_seeds.strip():
            return default_new_seeds.strip(), "command_default"
    if seed_group_index < len(seed_groups):
        return seed_groups[seed_group_index], "preset_seed_group"
    return None, "script_default"


def _append_flag(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _preset_value(args: argparse.Namespace, key: str) -> Any:
    value = getattr(args, key)
    if value is not None:
        return value
    return PRESETS[args.preset][key]


def _parent_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    command = [
        args.python,
        "-S",
        "-s",
        str(SCRIPT.relative_to(ROOT)),
        *[str(path) for path in args.text_or_dir],
    ]
    _append_flag(command, "--features", _preset_value(args, "features"))
    _append_flag(
        command,
        "--feature-normalize-modes",
        _preset_value(args, "feature_normalize_modes"),
    )
    _append_flag(
        command,
        "--hybrid-latent-scales",
        _preset_value(args, "hybrid_latent_scales"),
    )
    _append_flag(command, "--head-init", _preset_value(args, "head_init"))
    _append_flag(command, "--seeds", _preset_value(args, "seeds"))
    _append_flag(command, "--run-dir", run_dir)
    _append_flag(command, "--follow-up-fail-on-verdict", args.follow_up_fail_on_verdict)
    for key, flag in (
        ("window_chars", "--window-chars"),
        ("latent_dim", "--latent-dim"),
        ("hidden", "--hidden"),
        ("epochs", "--epochs"),
        ("batches", "--batches"),
        ("batch_size", "--batch-size"),
        ("eval_samples", "--eval-samples"),
        ("gen", "--gen"),
        ("vae_epochs", "--vae-epochs"),
        ("vae_batches", "--vae-batches"),
        ("vae_batch_size", "--vae-batch-size"),
    ):
        _append_flag(command, flag, _preset_value(args, key))
    return command


def _run_command(command: list[str], *, log_path: Path, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, **(env or {})},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(process.stdout or "", encoding="utf-8")
    if process.stdout:
        print(process.stdout, end="")
    return int(process.returncode)


def _value(summary: dict[str, Any] | None, *keys: str) -> Any:
    item: Any = summary
    for key in keys:
        if not isinstance(item, dict):
            return None
        item = item.get(key)
    return item


def _config_label(config: dict[str, Any] | None) -> str | None:
    if not isinstance(config, dict):
        return None
    feature = config.get("best_feature")
    normalize = config.get("feature_normalize")
    scale = config.get("hybrid_latent_scale")
    if feature is None:
        return None
    return f"{feature}@normalize={normalize},scale={scale}"


def _step_record(
    *,
    index: int,
    role: str,
    run_dir: Path,
    command: list[str],
    exit_code: int | None,
    dry_run: bool,
) -> dict[str, Any]:
    summary = None if dry_run else _load_summary(run_dir)
    summary_path = run_dir / "summary.json"
    best_config = _value(summary, "best_config")
    delta_vs_raw = _value(summary, "best_config", "mean_best_nll_delta_vs_raw")
    delta_vs_source = _value(
        summary,
        "follow_up_result",
        "mean_best_nll_delta_vs_source",
    )
    source_feature_delta_vs_source = _value(
        summary,
        "follow_up_result",
        "source_feature_mean_best_nll_delta_vs_source",
    )
    source_feature_retained = _value(
        summary,
        "follow_up_result",
        "source_best_feature_retained",
    )
    gate_failed = _value(summary, "follow_up_gate", "failed")
    seed_policy = _value(summary, "next_follow_up_command", "seed_confirmation_policy")
    if not isinstance(seed_policy, dict):
        seed_policy = {}
    return {
        "index": index,
        "role": role,
        "run_dir": str(run_dir),
        "summary_path": str(summary_path) if summary_path.exists() or dry_run else None,
        "exit_code": exit_code,
        "dry_run": dry_run,
        "command": command,
        "command_line": shlex.join(command),
        "status": _value(summary, "status"),
        "best_feature": _value(summary, "best_feature"),
        "best_config": best_config,
        "best_config_label": _config_label(
            best_config if isinstance(best_config, dict) else None
        ),
        "best_config_feature": _value(summary, "best_config", "best_feature"),
        "mean_best_nll": _value(summary, "best_config", "mean_best_nll"),
        "mean_best_accuracy": _value(summary, "best_config", "mean_best_accuracy"),
        "mean_best_nll_delta_vs_raw": delta_vs_raw,
        "runner_up_feature": _value(summary, "best_config", "runner_up_feature"),
        "runner_up_mean_best_nll": _value(
            summary,
            "best_config",
            "runner_up_mean_best_nll",
        ),
        "margin_to_runner_up": _value(summary, "best_config", "margin_to_runner_up"),
        "combined_runner_up_margin_stderr": _value(
            summary,
            "best_config",
            "combined_runner_up_margin_stderr",
        ),
        "runner_up_within_uncertainty": _value(
            summary,
            "best_config",
            "runner_up_within_uncertainty",
        ),
        "mean_best_nll_delta_vs_source": delta_vs_source,
        "source_feature_mean_best_nll_delta_vs_source": source_feature_delta_vs_source,
        "follow_up_verdict": _value(summary, "follow_up_result", "verdict"),
        "source_best_feature_retained": source_feature_retained,
        "follow_up_gate_failed": gate_failed,
        "trajectory_action": _value(summary, "follow_up_trajectory", "trajectory_action"),
        "trajectory_verdict": _value(summary, "follow_up_trajectory", "trajectory_verdict"),
        "guidance_action": _value(summary, "follow_up_guidance", "action"),
        "unsafe_promotion": _value(summary, "follow_up_guidance", "unsafe_promotion"),
        "guided_enabled": _value(summary, "guided_next_follow_up_command", "enabled"),
        "next_default_new_seed_count": _value(
            summary,
            "next_follow_up_command",
            "default_new_seed_count",
        ),
        "next_default_new_seeds": _value(
            summary,
            "next_follow_up_command",
            "default_new_seeds",
        ),
        "seed_policy_reason": seed_policy.get("reason"),
        "uncertainty_tie_seed_boost": seed_policy.get("uncertainty_tie_seed_boost"),
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _selection_step_record(step: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(step, dict):
        return None
    summary_path = step.get("summary_path")
    if not summary_path:
        return None
    return {
        "index": step.get("index"),
        "role": step.get("role"),
        "run_dir": step.get("run_dir"),
        "summary_path": summary_path,
        "status": step.get("status"),
        "best_feature": step.get("best_feature"),
        "best_config": step.get("best_config"),
        "best_config_label": step.get("best_config_label") or step.get("best_feature"),
        "mean_best_nll": step.get("mean_best_nll"),
        "mean_best_accuracy": step.get("mean_best_accuracy"),
        "mean_best_nll_delta_vs_raw": step.get("mean_best_nll_delta_vs_raw"),
        "runner_up_feature": step.get("runner_up_feature"),
        "runner_up_mean_best_nll": step.get("runner_up_mean_best_nll"),
        "margin_to_runner_up": step.get("margin_to_runner_up"),
        "combined_runner_up_margin_stderr": step.get(
            "combined_runner_up_margin_stderr"
        ),
        "runner_up_within_uncertainty": step.get("runner_up_within_uncertainty"),
        "mean_best_nll_delta_vs_source": step.get("mean_best_nll_delta_vs_source"),
        "follow_up_verdict": step.get("follow_up_verdict"),
        "source_best_feature_retained": step.get("source_best_feature_retained"),
        "follow_up_gate_failed": step.get("follow_up_gate_failed"),
        "next_default_new_seed_count": step.get("next_default_new_seed_count"),
        "next_default_new_seeds": step.get("next_default_new_seeds"),
        "seed_policy_reason": step.get("seed_policy_reason"),
        "uncertainty_tie_seed_boost": step.get("uncertainty_tie_seed_boost"),
        "follow_up_command_source": step.get("follow_up_command_source"),
        "new_seed_source": step.get("new_seed_source"),
        "new_seeds": step.get("new_seeds"),
    }


def _refresh_chain_selection(manifest: dict[str, Any]) -> None:
    if manifest.get("dry_run"):
        return
    steps = [step for step in manifest.get("steps", []) if isinstance(step, dict)]
    accepted: dict[str, Any] | None = None
    for step in steps:
        if not step.get("summary_path"):
            continue
        exit_code = step.get("exit_code")
        if exit_code == 0:
            accepted = step
            continue
        if step.get("follow_up_gate_failed"):
            # A gate-stopped follow-up is evidence, not a promotion.
            continue
        break

    scored_steps = [step for step in steps if _is_number(step.get("mean_best_nll"))]
    best = (
        min(scored_steps, key=lambda step: float(step["mean_best_nll"]))
        if scored_steps
        else None
    )

    accepted_record = _selection_step_record(accepted)
    best_record = _selection_step_record(best)
    if accepted_record is not None:
        manifest["accepted_step"] = accepted_record
        manifest["accepted_summary_path"] = accepted_record["summary_path"]
        manifest["accepted_best_config"] = accepted_record.get("best_config")
        manifest["accepted_best_config_label"] = accepted_record.get(
            "best_config_label"
        )
    if best_record is not None:
        manifest["best_step"] = best_record
        manifest["best_summary_path"] = best_record["summary_path"]
        manifest["best_config"] = best_record.get("best_config")
        manifest["best_config_label"] = best_record.get("best_config_label")


def _can_continue_after_gate_stop(
    summary: dict[str, Any] | None,
    *,
    allow_gate_stop: bool,
    index: int,
    follow_up_count: int,
) -> bool:
    if not allow_gate_stop or index >= follow_up_count:
        return False
    return bool(
        _value(summary, "follow_up_gate", "failed")
        and _value(summary, "guided_next_follow_up_command", "enabled")
    )


def _render_report(manifest: dict[str, Any]) -> str:
    accepted = manifest.get("accepted_step")
    best = manifest.get("best_step")
    lines = [
        "# Char VAE Context Chain Report",
        "",
        f"- schema: {manifest['schema']}",
        f"- preset: {manifest['preset']}",
        f"- run_root: `{manifest['run_root']}`",
        f"- stopped_reason: {manifest.get('stopped_reason') or '-'}",
        f"- allowed_gate_stop: {manifest.get('allowed_gate_stop') or False}",
        "- follow_up_seed_groups: {source} ({groups})".format(
            source=_fmt(manifest.get("follow_up_seed_group_source")),
            groups=(
                ", ".join(
                    str(group)
                    for group in manifest.get("planned_follow_up_seed_groups", [])
                )
                or "-"
            ),
        ),
        "- accepted: {label} (step {index}, mean_best_nll={nll})".format(
            label=_fmt(_value(accepted, "best_config_label")),
            index=_fmt(_value(accepted, "index")),
            nll=_fmt(_value(accepted, "mean_best_nll")),
        ),
        "- best: {label} (step {index}, mean_best_nll={nll})".format(
            label=_fmt(_value(best, "best_config_label")),
            index=_fmt(_value(best, "index")),
            nll=_fmt(_value(best, "mean_best_nll")),
        ),
        "",
        "| step | role | exit | status | best_config | mean_best_nll | "
        "runner_up | margin | margin_stderr | within_uncertainty | "
        "next_seed_count | tie_seed_boost | seed_policy | "
        "run_seeds | run_seed_source | "
        "delta_vs_raw | delta_vs_source | verdict | retained | gate | "
        "trajectory | guidance | unsafe | guided |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for step in manifest.get("steps", []):
        lines.append(
            "| {index} | {role} | {exit_code} | {status} | {best_feature} | "
            "{mean_best_nll} | {runner_up_feature} | {margin_to_runner_up} | "
            "{combined_runner_up_margin_stderr} | {runner_up_within_uncertainty} | "
            "{next_default_new_seed_count} | {uncertainty_tie_seed_boost} | "
            "{seed_policy_reason} | "
            "{new_seeds} | {new_seed_source} | "
            "{mean_best_nll_delta_vs_raw} | "
            "{mean_best_nll_delta_vs_source} | {follow_up_verdict} | "
            "{source_best_feature_retained} | {follow_up_gate_failed} | "
            "{trajectory_action} | {guidance_action} | {unsafe_promotion} | "
            "{guided_enabled} |".format(
                index=step.get("index"),
                role=_fmt(step.get("role")),
                exit_code=_fmt(step.get("exit_code")),
                status=_fmt(step.get("status")),
                best_feature=_fmt(
                    step.get("best_config_label") or step.get("best_feature")
                ),
                mean_best_nll=_fmt(step.get("mean_best_nll")),
                runner_up_feature=_fmt(step.get("runner_up_feature")),
                margin_to_runner_up=_fmt(step.get("margin_to_runner_up")),
                combined_runner_up_margin_stderr=_fmt(
                    step.get("combined_runner_up_margin_stderr")
                ),
                runner_up_within_uncertainty=_fmt(
                    step.get("runner_up_within_uncertainty")
                ),
                next_default_new_seed_count=_fmt(
                    step.get("next_default_new_seed_count")
                ),
                uncertainty_tie_seed_boost=_fmt(
                    step.get("uncertainty_tie_seed_boost")
                ),
                seed_policy_reason=_fmt(step.get("seed_policy_reason")),
                new_seeds=_fmt(step.get("new_seeds")),
                new_seed_source=_fmt(step.get("new_seed_source")),
                mean_best_nll_delta_vs_raw=_fmt(
                    step.get("mean_best_nll_delta_vs_raw")
                ),
                mean_best_nll_delta_vs_source=_fmt(
                    step.get("mean_best_nll_delta_vs_source")
                ),
                follow_up_verdict=_fmt(step.get("follow_up_verdict")),
                source_best_feature_retained=_fmt(
                    step.get("source_best_feature_retained")
                ),
                follow_up_gate_failed=_fmt(step.get("follow_up_gate_failed")),
                trajectory_action=_fmt(step.get("trajectory_action")),
                guidance_action=_fmt(step.get("guidance_action")),
                unsafe_promotion=_fmt(step.get("unsafe_promotion")),
                guided_enabled=_fmt(step.get("guided_enabled")),
            )
        )
    follow_up_steps = [
        step for step in manifest.get("steps", []) if step.get("role") == "follow_up"
    ]
    if follow_up_steps:
        lines.extend(
            [
                "",
                "## Follow-Up Deltas",
                "",
                "Negative `delta_vs_raw` means the selected feature still beats the "
                "raw-feature baseline; positive `delta_vs_source` means the follow-up "
                "softened or regressed versus the source run.",
                "",
                "| step | best_config | delta_vs_raw | delta_vs_source | "
                "source_feature_delta_vs_source | retained | gate | verdict |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for step in follow_up_steps:
            lines.append(
                "| {index} | {best_feature} | {delta_raw} | {delta_source} | "
                "{source_feature_delta} | {retained} | {gate} | {verdict} |".format(
                    index=step.get("index"),
                    best_feature=_fmt(
                        step.get("best_config_label") or step.get("best_feature")
                    ),
                    delta_raw=_fmt(step.get("mean_best_nll_delta_vs_raw")),
                    delta_source=_fmt(step.get("mean_best_nll_delta_vs_source")),
                    source_feature_delta=_fmt(
                        step.get("source_feature_mean_best_nll_delta_vs_source")
                    ),
                    retained=_fmt(step.get("source_best_feature_retained")),
                    gate=_fmt(step.get("follow_up_gate_failed")),
                    verdict=_fmt(step.get("follow_up_verdict")),
                )
            )
    lines.extend(["", "## Commands", ""])
    for step in manifest.get("steps", []):
        lines.extend(
            [
                f"### Step {step.get('index')} ({step.get('role')})",
                "",
                "```bash",
                str(step.get("command_line")),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run parent + guided follow-up char VAE context sweeps."
    )
    parser.add_argument("text_or_dir", nargs="+", help="input text file(s) or directories")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="small")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--follow-ups", type=int, default=1)
    parser.add_argument("--follow-up-seed-groups", default=None)
    parser.add_argument("--follow-up-fail-on-verdict", default=DEFAULT_FAIL_ON_VERDICT)
    parser.add_argument(
        "--allow-gate-stop",
        action="store_true",
        help="return success when a follow-up gate intentionally stops promotion",
    )
    parser.add_argument("--features", default=None)
    parser.add_argument("--feature-normalize-modes", default=None)
    parser.add_argument("--hybrid-latent-scales", default=None)
    parser.add_argument("--head-init", choices=("legacy", "xavier"), default=None)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="print chain manifest JSON")
    for key in (
        "window_chars",
        "latent_dim",
        "hidden",
        "epochs",
        "batches",
        "batch_size",
        "eval_samples",
        "gen",
        "vae_epochs",
        "vae_batches",
        "vae_batch_size",
    ):
        parser.add_argument(f"--{key.replace('_', '-')}", type=int, default=None)
    parser.add_argument("--seeds", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.follow_ups < 0:
        raise ValueError("--follow-ups must be >= 0")
    run_root = args.run_root or ROOT / "models" / "runs" / f"char_vae_context_chain_{_timestamp_slug()}"
    explicit_seed_groups = args.follow_up_seed_groups is not None
    seed_groups = _csv_groups(
        args.follow_up_seed_groups
        if explicit_seed_groups
        else PRESETS[args.preset]["follow_up_seed_groups"]
    )

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "preset": args.preset,
        "run_root": str(run_root),
        "dry_run": bool(args.dry_run),
        "allow_gate_stop": bool(args.allow_gate_stop),
        "planned_follow_ups": int(args.follow_ups),
        "planned_follow_up_seed_groups": seed_groups,
        "follow_up_seed_group_source": (
            "explicit" if explicit_seed_groups else "preset_fallback"
        ),
        "steps": [],
    }

    parent_dir = run_root / "parent"
    parent_command = _parent_command(args, parent_dir)
    parent_exit = 0 if args.dry_run else _run_command(
        parent_command,
        log_path=parent_dir / "process.log",
    )
    manifest["steps"].append(
        _step_record(
            index=0,
            role="parent",
            run_dir=parent_dir,
            command=parent_command,
            exit_code=parent_exit,
            dry_run=args.dry_run,
        )
    )
    current_dir = parent_dir
    if parent_exit != 0:
        manifest["stopped_reason"] = f"parent exited {parent_exit}"

    follow_up_count = 0 if args.dry_run else args.follow_ups
    for index in range(1, follow_up_count + 1):
        if manifest.get("stopped_reason"):
            break
        summary = _load_summary(current_dir)
        if summary is None:
            manifest["stopped_reason"] = "missing summary for follow-up source"
            break
        command_record, command_record_source = _follow_up_command_record(
            summary,
            index=index,
        )
        if isinstance(command_record, dict):
            script_path = command_record.get("script_path")
        else:
            manifest["stopped_reason"] = "guided follow-up disabled"
            break
        if not script_path:
            manifest["stopped_reason"] = "missing follow-up script"
            break

        follow_dir = run_root / f"follow_up_{index:02d}"
        env = {
            "NEXT_RUN_DIR": str(follow_dir),
            "FOLLOW_UP_FROM": str(current_dir / "summary.json"),
            "FOLLOW_UP_FAIL_ON_VERDICT": args.follow_up_fail_on_verdict,
        }
        new_seeds, new_seed_source = _follow_up_new_seeds(
            command_record,
            seed_groups,
            index=index,
            explicit_seed_groups=explicit_seed_groups,
        )
        if new_seeds is not None:
            env["NEW_SEEDS"] = new_seeds
        command = ["bash", str(script_path)]
        exit_code = 0 if args.dry_run else _run_command(
            command,
            log_path=follow_dir / "process.log",
            env=env,
        )
        step = _step_record(
            index=index,
            role="follow_up",
            run_dir=follow_dir,
            command=[f"{key}={value}" for key, value in sorted(env.items())] + command,
            exit_code=exit_code,
            dry_run=args.dry_run,
        )
        step["follow_up_command_source"] = command_record_source
        step["new_seed_source"] = new_seed_source
        step["new_seeds"] = new_seeds
        manifest["steps"].append(step)
        current_dir = follow_dir
        if exit_code != 0:
            summary = _load_summary(follow_dir)
            if _value(summary, "follow_up_gate", "failed") and args.allow_gate_stop:
                manifest["allowed_gate_stop"] = True
            if _can_continue_after_gate_stop(
                summary,
                allow_gate_stop=args.allow_gate_stop,
                index=index,
                follow_up_count=follow_up_count,
            ):
                continue
            manifest["stopped_reason"] = f"follow-up {index} exited {exit_code}"
            break

    chain_path = run_root / "chain.json"
    report_path = run_root / "chain_report.md"
    _refresh_chain_selection(manifest)
    _write_json(chain_path, manifest)
    _write_text(report_path, _render_report(manifest))
    print(f"chain_json={chain_path}")
    print(f"chain_report={report_path}")
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.dry_run or not manifest.get("stopped_reason") or manifest.get("allowed_gate_stop"):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
