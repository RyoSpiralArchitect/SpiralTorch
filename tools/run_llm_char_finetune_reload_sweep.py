#!/usr/bin/env python3
"""Run outcome-oriented Python char-LM reload-pair sweeps.

The sweep is intentionally thin: each cell shells out to
tools/run_llm_char_finetune_reload_pair.py, then collects outcome.json so reload
FT can be ranked by measured best-NLL movement instead of eyeballing logs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA = "st.llm_char_finetune.reload_sweep.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
PAIR_SCRIPT = REPO_ROOT / "tools" / "run_llm_char_finetune_reload_pair.py"
RUNTIME_IMPORT_PRESET_CHOICES = [
    "transformers",
    "torch",
    "tokenizers",
    "torch-transformers",
    "hf-runtime",
    "hf-datasets",
    "hf-finetune",
    "hf-peft",
]


def default_run_root() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "models" / "runs" / f"llm_char_finetune_reload_sweep_{stamp}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def parse_int_values(raw: str, *, label: str) -> list[int]:
    values: list[int] = []
    for cell in raw.split(","):
        text = cell.strip()
        if not text:
            continue
        try:
            values.append(int(text))
        except ValueError as exc:
            raise ValueError(f"--{label} must be a comma-separated integer list") from exc
    if not values:
        raise ValueError(f"--{label} must contain at least one value")
    return values


def parse_float_values(raw: str, *, label: str) -> list[float]:
    values: list[float] = []
    for cell in raw.split(","):
        text = cell.strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"--{label} must be a comma-separated float list") from exc
        if not math.isfinite(value):
            raise ValueError(f"--{label} must contain finite values")
        values.append(value)
    if not values:
        raise ValueError(f"--{label} must contain at least one value")
    return values


def positive_int(value: int, *, label: str, allow_zero: bool = False) -> int:
    if value < 0 or (value == 0 and not allow_zero):
        suffix = "non-negative" if allow_zero else "positive"
        raise ValueError(f"--{label} must be {suffix}")
    return value


def float_slug(value: float) -> str:
    text = f"{value:.12g}"
    return text.replace("-", "m").replace(".", "p")


def shell_command(command: list[str]) -> str:
    return "PYTHONNOUSERSITE=1 " + shlex.join(command)


def runtime_import_contract_requested(source: argparse.Namespace) -> bool:
    return bool(
        getattr(source, "runtime_imports", None)
        or getattr(source, "runtime_import_presets", None)
        or getattr(source, "required_runtime_imports", None)
        or getattr(source, "required_runtime_import_presets", None)
        or getattr(source, "runtime_device_backends", None)
        or getattr(source, "required_runtime_device_backends", None)
        or getattr(source, "required_runtime_device_ready_backends", None)
        or getattr(source, "require_runtime_imports", False)
    )


def runtime_import_contract_settings(source: argparse.Namespace) -> dict[str, Any]:
    return {
        "runtime_imports": list(getattr(source, "runtime_imports", []) or []),
        "runtime_import_presets": list(
            getattr(source, "runtime_import_presets", []) or []
        ),
        "required_runtime_imports": list(
            getattr(source, "required_runtime_imports", []) or []
        ),
        "required_runtime_import_presets": list(
            getattr(source, "required_runtime_import_presets", []) or []
        ),
        "runtime_device_backends": list(
            getattr(source, "runtime_device_backends", []) or []
        ),
        "required_runtime_device_backends": list(
            getattr(source, "required_runtime_device_backends", []) or []
        ),
        "required_runtime_device_ready_backends": list(
            getattr(source, "required_runtime_device_ready_backends", []) or []
        ),
        "require_runtime_imports": bool(
            getattr(source, "require_runtime_imports", False)
        ),
        "runtime_import_preflight_requested": runtime_import_contract_requested(
            source
        ),
    }


def runtime_import_cli_flags(source: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    for name in getattr(source, "runtime_imports", []) or []:
        flags.extend(["--runtime-import", str(name)])
    for preset in getattr(source, "runtime_import_presets", []) or []:
        flags.extend(["--runtime-import-preset", str(preset)])
    for name in getattr(source, "required_runtime_imports", []) or []:
        flags.extend(["--require-runtime-import", str(name)])
    for preset in getattr(source, "required_runtime_import_presets", []) or []:
        flags.extend(["--require-runtime-import-preset", str(preset)])
    if getattr(source, "require_runtime_imports", False):
        flags.append("--require-runtime-imports")
    for backend in getattr(source, "runtime_device_backends", []) or []:
        flags.extend(["--runtime-device-backend", str(backend)])
    for backend in getattr(source, "required_runtime_device_backends", []) or []:
        flags.extend(["--require-runtime-device-backend", str(backend)])
    for backend in getattr(source, "required_runtime_device_ready_backends", []) or []:
        flags.extend(["--require-runtime-device-ready-backend", str(backend)])
    return flags


def md_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def fmt_float(value: Any, *, digits: int = 6) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    out = float(value)
    if not math.isfinite(out):
        return "-"
    return f"{out:.{digits}f}"


def fmt_label(value: Any) -> str:
    if isinstance(value, (int, float)):
        out = float(value)
        if math.isfinite(out):
            return f"{out:.12g}"
    return str(value)


def run_command(command: list[str], log_path: Path) -> tuple[int, float]:
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + shell_command(command) + "\n")
        log_file.flush()
        env = dict(os.environ)
        env["PYTHONNOUSERSITE"] = "1"
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return process.returncode, time.time() - started


def pair_command(
    data_paths: list[Path],
    *,
    reload_data_paths: list[Path],
    run_root: Path,
    seed: int,
    reload_seed: int,
    eval_seed: int,
    reload_lr: float,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        "-S",
        "-s",
        str(PAIR_SCRIPT),
        *[str(path) for path in data_paths],
    ]
    for reload_data in reload_data_paths:
        command.extend(["--reload-data", str(reload_data)])
    command.extend(
        [
            "--run-root",
            str(run_root),
            "--base-epochs",
            str(args.base_epochs),
            "--reload-epochs",
            str(args.reload_epochs),
            "--batches",
            str(args.batches),
            "--batch",
            str(args.batch),
            "--steps",
            str(args.steps),
            "--embed-dim",
            str(args.embed_dim),
            "--hidden",
            str(args.hidden),
            "--lr",
            str(args.lr),
            "--reload-lr",
            str(reload_lr),
            "--eval-samples",
            str(args.eval_samples),
            "--val-split",
            str(args.val_split),
            "--gen",
            str(args.gen),
            "--topk",
            str(args.topk),
            "--seed",
            str(seed),
            "--reload-seed",
            str(reload_seed),
            "--eval-seed",
            str(eval_seed),
            "--backend",
            args.backend,
            "--summary-limit",
            str(args.summary_limit),
        ]
    )
    if args.early_stop_patience > 0:
        command.extend(["--early-stop-patience", str(args.early_stop_patience)])
    if args.restore_best_at_end:
        command.append("--restore-best-at-end")
    if args.rollback_on_validation_regression:
        command.append("--rollback-on-validation-regression")
    command.extend(runtime_import_cli_flags(args))
    if args.curves:
        command.append("--curves")
    if args.ignore_preflight:
        command.append("--ignore-preflight")
    if args.dry_run:
        command.append("--dry-run")
    return command


def planned_cells(args: argparse.Namespace) -> list[dict[str, Any]]:
    seeds = parse_int_values(args.seed_values, label="seed-values")
    reload_lrs = parse_float_values(args.reload_lr_values, label="reload-lr-values")
    cells: list[dict[str, Any]] = []
    for seed in seeds:
        for reload_lr in reload_lrs:
            reload_seed = seed + int(args.reload_seed_offset)
            eval_seed = seed + int(args.eval_seed_offset)
            name = f"seed{seed}_reloadlr{float_slug(reload_lr)}"
            cells.append(
                {
                    "name": name,
                    "seed": seed,
                    "reload_seed": reload_seed,
                    "eval_seed": eval_seed,
                    "reload_lr": reload_lr,
                    "run_root": str(args.run_root / name),
                }
            )
    return cells


def run_cell(
    cell: dict[str, Any],
    *,
    args: argparse.Namespace,
    data_paths: list[Path],
    reload_data_paths: list[Path],
) -> dict[str, Any]:
    run_root = Path(str(cell["run_root"]))
    command = pair_command(
        data_paths,
        reload_data_paths=reload_data_paths,
        run_root=run_root,
        seed=int(cell["seed"]),
        reload_seed=int(cell["reload_seed"]),
        eval_seed=int(cell["eval_seed"]),
        reload_lr=float(cell["reload_lr"]),
        args=args,
    )
    log_path = run_root / "reload_pair_process.log"
    item = {
        **cell,
        "command": command,
        "shell_command": shell_command(command),
        "log_path": str(log_path),
        "manifest_path": str(run_root / "reload_pair.json"),
        "outcome_path": str(run_root / "outcome.json"),
        "returncode": None,
        "elapsed_seconds": 0.0,
        "status": "planned",
        "pair_manifest": None,
        "outcome": None,
    }
    if args.dry_run:
        item["returncode"] = 0
        item["status"] = "dry_run"
        return item

    returncode, elapsed = run_command(command, log_path)
    item["returncode"] = returncode
    item["elapsed_seconds"] = elapsed
    item["status"] = "ok" if returncode == 0 else "failed"
    manifest_path = Path(str(item["manifest_path"]))
    outcome_path = Path(str(item["outcome_path"]))
    if manifest_path.exists():
        item["pair_manifest"] = read_json(manifest_path)
    if outcome_path.exists():
        item["outcome"] = read_json(outcome_path)
    elif isinstance(item["pair_manifest"], dict):
        embedded = item["pair_manifest"].get("outcome")
        if isinstance(embedded, dict):
            item["outcome"] = embedded
    if returncode == 0 and not isinstance(item["outcome"], dict):
        item["status"] = "missing_outcome"
    return item


def outcome_status(cell: dict[str, Any]) -> str:
    outcome = cell.get("outcome")
    if isinstance(outcome, dict):
        return str(outcome.get("status") or "unknown")
    return str(cell.get("status") or "unknown")


def outcome_training_status(cell: dict[str, Any]) -> str:
    outcome = cell.get("outcome")
    if isinstance(outcome, dict):
        return str(outcome.get("reload_training_status") or "unknown")
    return str(cell.get("status") or "unknown")


def outcome_adoption_status(cell: dict[str, Any]) -> str:
    outcome = cell.get("outcome")
    if isinstance(outcome, dict):
        return str(outcome.get("reload_adoption_status") or "unknown")
    return str(cell.get("status") or "unknown")


def csv_parts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        parts: list[str] = []
        for item in value:
            parts.extend(csv_parts(item))
        return parts
    if isinstance(value, dict):
        return []
    parts = [part.strip() for part in str(value).split(",")]
    return [part for part in parts if part and part != "none"]


def runtime_preflight_payload(
    cell: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    pair_manifest = cell.get("pair_manifest")
    if not isinstance(pair_manifest, dict):
        return None, None
    preflight = pair_manifest.get("preflight")
    if not isinstance(preflight, dict):
        return None, None
    runtime = preflight.get("child_runtime_preflight")
    if not isinstance(runtime, dict):
        runtime = preflight.get("runtime_import_preflight")
    return preflight, runtime if isinstance(runtime, dict) else None


def runtime_preflight_status(cell: dict[str, Any]) -> str:
    pair_manifest = cell.get("pair_manifest")
    if not isinstance(pair_manifest, dict):
        status = str(cell.get("status") or "unknown")
        return status if status in {"dry_run", "failed"} else "unobserved"
    preflight, runtime = runtime_preflight_payload(cell)
    if not isinstance(preflight, dict):
        return "missing_preflight"
    requested = preflight.get("runtime_import_preflight_requested")
    if isinstance(runtime, dict):
        requested = runtime.get("runtime_import_preflight_requested", requested)
        passed = runtime.get("runtime_import_preflight_passed")
    else:
        passed = preflight.get("runtime_import_preflight_passed")
    if requested is not True:
        return "not_requested"
    if passed is True:
        return "passed"
    if passed is False:
        return "failed"
    return "unknown"


def runtime_preflight_detail(cell: dict[str, Any]) -> str:
    status = runtime_preflight_status(cell)
    preflight, runtime = runtime_preflight_payload(cell)
    details: list[str] = []
    if isinstance(runtime, dict):
        details.extend(csv_parts(runtime.get("runtime_import_preflight_failures")))
        details.extend(csv_parts(runtime.get("runtime_device_report_statuses")))
        details.extend(csv_parts(runtime.get("runtime_imports_failed")))
        details.extend(csv_parts(runtime.get("runtime_import_failed_install_hints")))
    if isinstance(preflight, dict):
        details.extend(csv_parts(preflight.get("reason")))
        details.extend(csv_parts(preflight.get("issues")))
    unique = list(dict.fromkeys(details))
    if unique:
        return ";".join(unique)
    if status in {"passed", "not_requested", "dry_run"}:
        return "none"
    return status


def runtime_preflight_trusted(cell: dict[str, Any]) -> bool:
    return runtime_preflight_status(cell) in {"passed", "not_requested"}


def trusted_cells(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [cell for cell in cells if runtime_preflight_trusted(cell)]


def outcome_delta(cell: dict[str, Any], key: str) -> float | None:
    outcome = cell.get("outcome")
    if not isinstance(outcome, dict):
        return None
    value = outcome.get(key)
    if not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def delta_stats(cells: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [outcome_delta(cell, key) for cell in cells]
    finite = [value for value in values if value is not None]
    if not finite:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(finite),
        "mean": sum(finite) / len(finite),
        "min": min(finite),
        "max": max(finite),
    }


def best_cell_by_delta(
    cells: list[dict[str, Any]],
    key: str,
    *,
    secondary_key: str | None = None,
) -> dict[str, Any] | None:
    ranked = [cell for cell in cells if outcome_delta(cell, key) is not None]

    def sort_delta(cell: dict[str, Any], metric: str) -> float:
        value = outcome_delta(cell, metric)
        return value if value is not None else float("inf")

    ranked.sort(
        key=lambda cell: (
            sort_delta(cell, key),
            sort_delta(cell, secondary_key) if secondary_key is not None else 0.0,
            str(cell.get("name") or ""),
        )
    )
    return ranked[0] if ranked else None


def aggregate_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(outcome_status(cell) for cell in cells)
    training_status_counts = Counter(outcome_training_status(cell) for cell in cells)
    adoption_status_counts = Counter(outcome_adoption_status(cell) for cell in cells)
    run_status_counts = Counter(str(cell.get("status") or "unknown") for cell in cells)
    runtime_preflight_status_counts = Counter(
        runtime_preflight_status(cell) for cell in cells
    )
    runtime_preflight_detail_counts = Counter(
        runtime_preflight_detail(cell) for cell in cells
    )
    trusted = trusted_cells(cells)
    best = best_cell_by_delta(
        cells,
        "reload_best_minus_base_best_nll",
        secondary_key="reload_final_minus_base_final_nll",
    )
    best_training = best_cell_by_delta(
        cells,
        "reload_training_final_minus_base_best_nll",
    )
    trusted_best = best_cell_by_delta(
        trusted,
        "reload_best_minus_base_best_nll",
        secondary_key="reload_final_minus_base_final_nll",
    )
    trusted_best_training = best_cell_by_delta(
        trusted,
        "reload_training_final_minus_base_best_nll",
    )
    return {
        "cells": len(cells),
        "status_counts": dict(sorted(status_counts.items())),
        "training_status_counts": dict(sorted(training_status_counts.items())),
        "adoption_status_counts": dict(sorted(adoption_status_counts.items())),
        "run_status_counts": dict(sorted(run_status_counts.items())),
        "runtime_preflight_status_counts": dict(
            sorted(runtime_preflight_status_counts.items())
        ),
        "runtime_preflight_detail_counts": dict(
            sorted(runtime_preflight_detail_counts.items())
        ),
        "runtime_trusted_cells": len(trusted),
        "runtime_untrusted_cells": len(cells) - len(trusted),
        "best_cell": best.get("name") if best else None,
        "best_reload_best_minus_base_best_nll": (
            outcome_delta(best, "reload_best_minus_base_best_nll") if best else None
        ),
        "best_training_cell": best_training.get("name") if best_training else None,
        "best_reload_training_final_minus_base_best_nll": (
            outcome_delta(best_training, "reload_training_final_minus_base_best_nll")
            if best_training
            else None
        ),
        "trusted_best_cell": trusted_best.get("name") if trusted_best else None,
        "trusted_best_reload_best_minus_base_best_nll": (
            outcome_delta(trusted_best, "reload_best_minus_base_best_nll")
            if trusted_best
            else None
        ),
        "trusted_best_training_cell": (
            trusted_best_training.get("name") if trusted_best_training else None
        ),
        "trusted_best_reload_training_final_minus_base_best_nll": (
            outcome_delta(
                trusted_best_training,
                "reload_training_final_minus_base_best_nll",
            )
            if trusted_best_training
            else None
        ),
        "improved_cells": int(status_counts.get("improved", 0)),
        "regressed_cells": int(status_counts.get("regressed", 0)),
        "accepted_improved_cells": int(
            adoption_status_counts.get("accepted_improved", 0)
        ),
        "protected_noop_cells": int(adoption_status_counts.get("protected_noop", 0)),
        "rejected_regressed_cells": int(
            adoption_status_counts.get("rejected_regressed", 0)
        ),
        "reload_best_minus_base_best_nll_stats": delta_stats(
            cells,
            "reload_best_minus_base_best_nll",
        ),
        "reload_training_final_minus_base_best_nll_stats": delta_stats(
            cells,
            "reload_training_final_minus_base_best_nll",
        ),
        "reload_training_final_minus_reload_initial_nll_stats": delta_stats(
            cells,
            "reload_training_final_minus_reload_initial_nll",
        ),
        "reload_validation_rollback_count_stats": delta_stats(
            cells,
            "reload_validation_rollback_count",
        ),
    }


def reload_lr_group_summaries(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[float, list[dict[str, Any]]] = {}
    for cell in cells:
        reload_lr = cell.get("reload_lr")
        if not isinstance(reload_lr, (int, float)):
            continue
        value = float(reload_lr)
        if not math.isfinite(value):
            continue
        groups.setdefault(value, []).append(cell)
    summaries: list[dict[str, Any]] = []
    for reload_lr in sorted(groups):
        group_cells = groups[reload_lr]
        summary = aggregate_cells(group_cells)
        summary["reload_lr"] = reload_lr
        summary["reload_lr_label"] = fmt_label(reload_lr)
        summaries.append(summary)
    return summaries


def sweep_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    summary = aggregate_cells(cells)
    summary["reload_lr_groups"] = reload_lr_group_summaries(cells)
    return summary


def stats_value(stats: Any, key: str) -> Any:
    if isinstance(stats, dict):
        return stats.get(key)
    return None


def render_markdown(manifest: dict[str, Any]) -> str:
    summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
    lines = [
        "# LLM Char Finetune Reload Sweep",
        "",
        f"- schema: `{manifest.get('schema')}`",
        f"- cells: `{summary.get('cells', 0)}`",
        f"- status_counts: `{summary.get('status_counts', {})}`",
        f"- training_status_counts: `{summary.get('training_status_counts', {})}`",
        f"- adoption_status_counts: `{summary.get('adoption_status_counts', {})}`",
        f"- runtime_preflight_status_counts: `{summary.get('runtime_preflight_status_counts', {})}`",
        f"- runtime_preflight_detail_counts: `{summary.get('runtime_preflight_detail_counts', {})}`",
        f"- runtime_trusted_cells: `{summary.get('runtime_trusted_cells', 0)}`",
        f"- runtime_untrusted_cells: `{summary.get('runtime_untrusted_cells', 0)}`",
        f"- best_cell: `{summary.get('best_cell', '-')}`",
        f"- best_training_cell: `{summary.get('best_training_cell', '-')}`",
        f"- trusted_best_cell: `{summary.get('trusted_best_cell', '-')}`",
        f"- trusted_best_training_cell: `{summary.get('trusted_best_training_cell', '-')}`",
        "",
    ]
    reload_lr_groups = summary.get("reload_lr_groups")
    if isinstance(reload_lr_groups, list) and reload_lr_groups:
        lines.extend(
            [
                "## Reload LR Groups",
                "",
                "| reload_lr | cells | runtime_status | runtime_detail | trusted_cells | status_counts | training_status_counts | adoption_status_counts | best_cell | trusted_best_cell | best_delta | trusted_best_delta | best_training_cell | trusted_best_training_cell | training_delta_mean | training_delta_min | training_delta_max | reload_delta_mean | rollback_count_mean |",
                "| ---: | ---: | --- | --- | ---: | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for group in reload_lr_groups:
            if not isinstance(group, dict):
                continue
            training_stats = group.get(
                "reload_training_final_minus_base_best_nll_stats"
            )
            reload_stats = group.get(
                "reload_training_final_minus_reload_initial_nll_stats"
            )
            rollback_stats = group.get("reload_validation_rollback_count_stats")
            lines.append(
                "| "
                + " | ".join(
                    [
                        md_cell(group.get("reload_lr_label")),
                        md_cell(group.get("cells")),
                        md_cell(group.get("runtime_preflight_status_counts")),
                        md_cell(group.get("runtime_preflight_detail_counts")),
                        md_cell(group.get("runtime_trusted_cells")),
                        md_cell(group.get("status_counts")),
                        md_cell(group.get("training_status_counts")),
                        md_cell(group.get("adoption_status_counts")),
                        md_cell(group.get("best_cell")),
                        md_cell(group.get("trusted_best_cell")),
                        fmt_float(group.get("best_reload_best_minus_base_best_nll")),
                        fmt_float(
                            group.get(
                                "trusted_best_reload_best_minus_base_best_nll"
                            )
                        ),
                        md_cell(group.get("best_training_cell")),
                        md_cell(group.get("trusted_best_training_cell")),
                        fmt_float(stats_value(training_stats, "mean")),
                        fmt_float(stats_value(training_stats, "min")),
                        fmt_float(stats_value(training_stats, "max")),
                        fmt_float(stats_value(reload_stats, "mean")),
                        fmt_float(stats_value(rollback_stats, "mean")),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "| cell | status | training_status | adoption_status | run_status | runtime_status | runtime_detail | trusted | seed | reload_seed | eval_seed | reload_lr | best_delta | training_delta | final_delta | reload_delta | rollback_count | manifest | outcome |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for cell in manifest.get("cells", []):
        outcome = cell.get("outcome") if isinstance(cell.get("outcome"), dict) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    md_cell(cell.get("name")),
                    md_cell(outcome_status(cell)),
                    md_cell(outcome_training_status(cell)),
                    md_cell(outcome_adoption_status(cell)),
                    md_cell(cell.get("status")),
                    md_cell(runtime_preflight_status(cell)),
                    md_cell(runtime_preflight_detail(cell)),
                    md_cell(runtime_preflight_trusted(cell)),
                    md_cell(cell.get("seed")),
                    md_cell(cell.get("reload_seed")),
                    md_cell(cell.get("eval_seed")),
                    fmt_float(cell.get("reload_lr")),
                    fmt_float(outcome.get("reload_best_minus_base_best_nll")),
                    fmt_float(
                        outcome.get("reload_training_final_minus_base_best_nll")
                    ),
                    fmt_float(outcome.get("reload_final_minus_base_final_nll")),
                    fmt_float(outcome.get("reload_best_minus_reload_initial_nll")),
                    fmt_float(outcome.get("reload_validation_rollback_count")),
                    md_cell(cell.get("manifest_path")),
                    md_cell(cell.get("outcome_path")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run several Python char-LM reload-pair cells and summarize outcome.json."
    )
    parser.add_argument("data_paths", nargs="+", type=Path)
    parser.add_argument("--reload-data", dest="reload_data_paths", action="append", type=Path, default=[])
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--seed-values", default="42")
    parser.add_argument("--reload-seed-offset", type=int, default=1)
    parser.add_argument("--eval-seed-offset", type=int, default=0)
    parser.add_argument("--base-epochs", type=int, default=2)
    parser.add_argument("--reload-epochs", type=int, default=2)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--reload-lr-values", default="0.02")
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--gen", type=int, default=120)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--restore-best-at-end", action="store_true")
    parser.add_argument("--rollback-on-validation-regression", action="store_true")
    parser.add_argument(
        "--runtime-import",
        dest="runtime_imports",
        action="append",
        default=[],
        help="optional module forwarded to each reload-pair child LLM FT runtime preflight",
    )
    parser.add_argument(
        "--runtime-import-preset",
        dest="runtime_import_presets",
        action="append",
        choices=RUNTIME_IMPORT_PRESET_CHOICES,
        default=[],
        help="named runtime import bundle forwarded to each reload-pair cell",
    )
    parser.add_argument(
        "--require-runtime-import",
        dest="required_runtime_imports",
        action="append",
        default=[],
        help="module that must import successfully in each reload-pair cell",
    )
    parser.add_argument(
        "--require-runtime-import-preset",
        dest="required_runtime_import_presets",
        action="append",
        choices=RUNTIME_IMPORT_PRESET_CHOICES,
        default=[],
        help="preset that must be satisfied in each reload-pair cell",
    )
    parser.add_argument(
        "--require-runtime-imports",
        action="store_true",
        help="require every requested runtime import/preset in each reload-pair cell",
    )
    parser.add_argument(
        "--runtime-device-backend",
        "--device-backend",
        dest="runtime_device_backends",
        action="append",
        default=[],
        help="runtime backend to inspect in each reload-pair cell",
    )
    parser.add_argument(
        "--require-runtime-device-backend",
        "--require-device-backend",
        dest="required_runtime_device_backends",
        action="append",
        default=[],
        help="require this runtime backend report in each reload-pair cell",
    )
    parser.add_argument(
        "--require-runtime-device-ready-backend",
        "--require-device-ready-backend",
        dest="required_runtime_device_ready_backends",
        action="append",
        default=[],
        help="require this runtime backend to be ready in each reload-pair cell",
    )
    parser.add_argument("--curves", action="store_true")
    parser.add_argument("--summary-limit", type=int, default=8)
    parser.add_argument("--ignore-preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    parse_int_values(args.seed_values, label="seed-values")
    reload_lrs = parse_float_values(args.reload_lr_values, label="reload-lr-values")
    positive_int(args.base_epochs, label="base-epochs", allow_zero=True)
    positive_int(args.reload_epochs, label="reload-epochs", allow_zero=True)
    positive_int(args.batches, label="batches")
    positive_int(args.batch, label="batch")
    positive_int(args.steps, label="steps")
    positive_int(args.embed_dim, label="embed-dim")
    positive_int(args.hidden, label="hidden")
    positive_int(args.eval_samples, label="eval-samples", allow_zero=True)
    positive_int(args.gen, label="gen", allow_zero=True)
    positive_int(args.topk, label="topk")
    positive_int(args.early_stop_patience, label="early-stop-patience", allow_zero=True)
    positive_int(args.summary_limit, label="summary-limit", allow_zero=True)
    if not (0.0 <= float(args.val_split) < 0.95):
        raise ValueError("--val-split must be within [0, 0.95)")
    if float(args.lr) <= 0.0:
        raise ValueError("--lr must be positive")
    if any(value <= 0.0 for value in reload_lrs):
        raise ValueError("--reload-lr-values must be positive")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    args.run_root = args.run_root.resolve() if args.run_root else default_run_root()
    try:
        validate_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.run_root.mkdir(parents=True, exist_ok=True)
    data_paths = list(args.data_paths)
    reload_data_paths = list(args.reload_data_paths)
    cells = planned_cells(args)
    manifest_path = args.run_root / "reload_sweep.json"
    markdown_path = args.run_root / "reload_sweep.md"
    started = time.time()
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "started_at_unix": started,
        "run_root": str(args.run_root),
        "dry_run": bool(args.dry_run),
        "data_paths": [str(path) for path in data_paths],
        "reload_data_paths": [str(path) for path in reload_data_paths],
        "settings": {
            "backend": args.backend,
            "seed_values": parse_int_values(args.seed_values, label="seed-values"),
            "reload_seed_offset": int(args.reload_seed_offset),
            "eval_seed_offset": int(args.eval_seed_offset),
            "base_epochs": args.base_epochs,
            "reload_epochs": args.reload_epochs,
            "batches": args.batches,
            "batch": args.batch,
            "steps": args.steps,
            "embed_dim": args.embed_dim,
            "hidden": args.hidden,
            "lr": args.lr,
            "reload_lr_values": parse_float_values(args.reload_lr_values, label="reload-lr-values"),
            "eval_samples": args.eval_samples,
            "val_split": args.val_split,
            "gen": args.gen,
            "topk": args.topk,
            "early_stop_patience": int(args.early_stop_patience),
            "restore_best_at_end": bool(args.restore_best_at_end),
            "rollback_on_validation_regression": bool(
                args.rollback_on_validation_regression
            ),
            **runtime_import_contract_settings(args),
        },
        "cells": [],
        "summary": {},
        "failed": False,
    }
    write_json(manifest_path, manifest)

    for cell in cells:
        item = run_cell(
            cell,
            args=args,
            data_paths=data_paths,
            reload_data_paths=reload_data_paths,
        )
        manifest["cells"].append(item)
        if item.get("returncode") not in (0, None) or item.get("status") in {
            "failed",
            "missing_outcome",
        }:
            manifest["failed"] = True
        manifest["summary"] = sweep_summary(manifest["cells"])
        write_json(manifest_path, manifest)
        markdown_path.write_text(render_markdown(manifest), encoding="utf-8")

    manifest["finished_at_unix"] = time.time()
    manifest["elapsed_seconds"] = manifest["finished_at_unix"] - started
    manifest["summary"] = sweep_summary(manifest["cells"])
    manifest["markdown_path"] = str(markdown_path)
    write_json(manifest_path, manifest)
    markdown_path.write_text(render_markdown(manifest), encoding="utf-8")
    print(f"manifest: {manifest_path}")
    print(f"summary: {markdown_path}")
    return 1 if manifest["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
