#!/usr/bin/env python3
"""Run a fixed char-LM ablation benchmark and emit report artifacts.

This is the small, opinionated front door for "does the Spiral char-LM stack
help?" experiments. It reuses the lower-level sweep helpers for process
launching, trace summarization, compare.json generation, and compare_summary.md
rendering while keeping the default variant plan intentionally compact.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import run_char_lm_sweep as sweep


DEFAULT_SEEDS = "7,13"
DEFAULT_VARIANT_SET = "guarded"


@dataclass(frozen=True)
class AblationVariant:
    label: str
    description: str
    architecture: str
    head_prior: str = "learned-bigram"
    char_feature: str = "token-bigram"
    head_residual_scale: float | None = 0.5
    bigram_topk_guard: float | None = 0.0
    bigram_topk_guard_k: int | None = 5
    bigram_rank_guard: float | None = None
    bigram_rank_guard_margin: float | None = None
    bigram_rank_guard_band: float | None = None
    bigram_rank_guard_min_candidates: int | None = None
    bigram_soft_guard: float | None = None


CORE_VARIANTS = (
    AblationVariant(
        label="lstm_baseline",
        description="Stateless batched LSTM baseline with the learned bigram prior.",
        architecture="lstm",
    ),
    AblationVariant(
        label="spiral_rnn",
        description="SpiralRNN recurrent core under the same learned bigram prior.",
        architecture="finetune",
    ),
    AblationVariant(
        label="coherence_scan",
        description="Z-space coherence scan context mixer.",
        architecture="scan",
    ),
    AblationVariant(
        label="coherence_wave",
        description="Z-space coherence wave mixer.",
        architecture="wave",
    ),
)

TOPK_GUARD_VARIANTS = tuple(
    AblationVariant(
        label=f"{variant.label}_topk_guard",
        description=f"{variant.description} Adds a light previous-token top-k guard.",
        architecture=variant.architecture,
        head_prior=variant.head_prior,
        char_feature=variant.char_feature,
        head_residual_scale=variant.head_residual_scale,
        bigram_topk_guard=0.1,
        bigram_topk_guard_k=5,
    )
    for variant in CORE_VARIANTS
)

RANK_MIN_VARIANTS = (
    AblationVariant(
        label="lstm_rank_min_guard",
        description="LSTM with top-k plus adaptive rank-min bigram guard.",
        architecture="lstm",
        bigram_topk_guard=0.1,
        bigram_topk_guard_k=5,
        bigram_rank_guard=0.1,
        bigram_rank_guard_margin=0.05,
        bigram_rank_guard_band=0.003,
        bigram_rank_guard_min_candidates=1,
    ),
    AblationVariant(
        label="scan_rank_min_guard",
        description="Coherence scan with top-k plus adaptive rank-min bigram guard.",
        architecture="scan",
        bigram_topk_guard=0.1,
        bigram_topk_guard_k=5,
        bigram_rank_guard=0.1,
        bigram_rank_guard_margin=0.05,
        bigram_rank_guard_band=0.003,
        bigram_rank_guard_min_candidates=1,
    ),
    AblationVariant(
        label="wave_rank_min_guard",
        description="Coherence wave with top-k plus adaptive rank-min bigram guard.",
        architecture="wave",
        bigram_topk_guard=0.1,
        bigram_topk_guard_k=5,
        bigram_rank_guard=0.1,
        bigram_rank_guard_margin=0.05,
        bigram_rank_guard_band=0.003,
        bigram_rank_guard_min_candidates=1,
    ),
)

VARIANT_SETS = {
    "core": CORE_VARIANTS,
    "guarded": (*CORE_VARIANTS, *TOPK_GUARD_VARIANTS),
    "rank-min": (*CORE_VARIANTS, *TOPK_GUARD_VARIANTS, *RANK_MIN_VARIANTS),
}


def default_run_root() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return sweep.REPO_ROOT / "models" / "runs" / f"char_lm_ablation_{stamp}"


def parse_csv_int(raw: str, *, label: str) -> list[int]:
    return sweep.parse_csv_int(raw, label=label)


def md_table(headers: list[str], rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_none_"
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| "
        + " | ".join(
            sweep.md_cell("-" if row.get(header) is None else row.get(header, "-"))
            for header in headers
        )
        + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body])


def settings_from_args(args: argparse.Namespace) -> sweep.SweepSettings:
    preset = sweep.PRESETS[args.preset]
    return sweep.SweepSettings(
        epochs=args.epochs if args.epochs is not None else preset["epochs"],
        batches=args.batches if args.batches is not None else preset["batches"],
        batch=args.batch if args.batch is not None else preset["batch"],
        eval_samples=(
            args.eval_samples if args.eval_samples is not None else preset["eval_samples"]
        ),
        gen=args.gen if args.gen is not None else preset["gen"],
        early_stop_patience=(
            args.early_stop_patience
            if args.early_stop_patience is not None
            else preset["early_stop_patience"]
        ),
        steps=args.steps,
        embed_dim=args.embed_dim,
        hidden=args.hidden,
        memory=args.memory,
        lr=args.lr,
        curvature=args.curvature,
        temperature=args.temperature,
        head_residual_scale=args.head_residual_scale,
        backend=args.backend,
        val_start_fraction=args.val_start_fraction,
    )


def settings_for_variant(
    settings: sweep.SweepSettings,
    variant: AblationVariant,
) -> sweep.SweepSettings:
    return replace(
        settings,
        head_residual_scale=(
            settings.head_residual_scale
            if settings.head_residual_scale is not None
            else variant.head_residual_scale
        ),
        bigram_topk_guard=variant.bigram_topk_guard,
        bigram_topk_guard_k=variant.bigram_topk_guard_k,
        bigram_rank_guard=variant.bigram_rank_guard,
        bigram_rank_guard_margin=variant.bigram_rank_guard_margin,
        bigram_rank_guard_band=variant.bigram_rank_guard_band,
        bigram_rank_guard_min_candidates=variant.bigram_rank_guard_min_candidates,
        bigram_soft_guard=variant.bigram_soft_guard,
    )


def slugged_run_name(
    *,
    variant: AblationVariant,
    seed: int,
    settings: sweep.SweepSettings,
) -> str:
    parts = [
        sweep.slug(variant.label),
        f"arch-{sweep.slug(variant.architecture)}",
        f"head-{sweep.slug(variant.head_prior)}",
        f"backend-{sweep.slug(settings.backend)}",
        f"seed-{seed}",
    ]
    return "__".join(parts)


def variant_rows(variants: tuple[AblationVariant, ...]) -> list[dict[str, Any]]:
    return [
        {
            "label": variant.label,
            "arch": variant.architecture,
            "head_prior": variant.head_prior,
            "topk_guard": variant.bigram_topk_guard,
            "rank_guard": variant.bigram_rank_guard,
            "rank_band": variant.bigram_rank_guard_band,
            "rank_min": variant.bigram_rank_guard_min_candidates,
            "description": variant.description,
        }
        for variant in variants
    ]


def build_report_markdown(
    *,
    manifest: dict[str, Any],
    compare_summary: dict[str, Any] | None,
) -> str:
    rows = []
    recommendations = []
    guard_recommendations = []
    route_counts = {}
    if isinstance(compare_summary, dict):
        rows = compare_summary.get("rows") if isinstance(compare_summary.get("rows"), list) else []
        recommendations = (
            compare_summary.get("paired_recurrent_recommendations")
            if isinstance(compare_summary.get("paired_recurrent_recommendations"), list)
            else []
        )
        guard_recommendations = (
            compare_summary.get("bigram_guard_recommendations")
            if isinstance(compare_summary.get("bigram_guard_recommendations"), list)
            else []
        )
        route_counts = (
            compare_summary.get("route_status_counts")
            if isinstance(compare_summary.get("route_status_counts"), dict)
            else {}
        )

    run_rows = [
        {
            "variant": run.get("variant"),
            "seed": run.get("seed"),
            "status": run.get("run_status"),
            "final_nll": run.get("final_validation_mean_nll"),
            "best_nll": run.get("best_validation_mean_nll"),
            "elapsed_s": run.get("elapsed_seconds"),
            "run_dir": run.get("run_dir"),
        }
        for run in manifest.get("runs", [])
        if isinstance(run, dict)
    ]
    lines = [
        "# Char-LM Ablation Benchmark",
        "",
        "## Artifacts",
        "",
        md_table(
            ["key", "value"],
            [
                {"key": "run_root", "value": manifest.get("run_root")},
                {"key": "manifest", "value": manifest.get("manifest_path")},
                {"key": "compare", "value": manifest.get("compare_path")},
                {"key": "compare_json", "value": manifest.get("compare_json_path")},
                {"key": "compare_summary", "value": manifest.get("compare_summary_path")},
                {
                    "key": "compare_summary_json",
                    "value": manifest.get("compare_summary_json_path"),
                },
            ],
        ),
        "",
        "## Variant Plan",
        "",
        md_table(
            [
                "label",
                "arch",
                "head_prior",
                "topk_guard",
                "rank_guard",
                "rank_band",
                "rank_min",
                "description",
            ],
            manifest.get("variants", []),
        ),
        "",
        "## Top Aggregate Rows",
        "",
        md_table(
            [
                "rank",
                "arch",
                "recurrent",
                "head_prior",
                "bigram_guard",
                "final_nll_mean",
                "best_nll_mean",
                "final_vs_bigram_mean",
                "final_top5_bigram_overlap_mean",
                "trace_step_ms_mean_mean",
                "route_status",
            ],
            rows[: int(manifest.get("summary_limit") or 8)],
        ),
        "",
        "## Recurrent Recommendations",
        "",
        md_table(
            [
                "rank",
                "recommendation",
                "candidate_recurrent",
                "baseline_recurrent",
                "quality_status",
                "efficiency_verdict",
                "final_nll_delta",
                "trace_step_ms_ratio",
                "cpu_debt_ratio",
            ],
            recommendations[: int(manifest.get("summary_limit") or 8)],
        ),
        "",
        "## Bigram Guard Recommendations",
        "",
        md_table(
            [
                "rank",
                "recommendation",
                "arch",
                "recurrent",
                "candidate_bigram_guard",
                "baseline_bigram_guard",
                "guard_verdict",
                "final_nll_delta",
                "top5_bigram_overlap_delta_pp",
                "candidate_route_status",
                "baseline_route_status",
            ],
            guard_recommendations[: int(manifest.get("summary_limit") or 8)],
        ),
        "",
        "## Route Status Counts",
        "",
        md_table(
            ["status", "count"],
            [
                {"status": status, "count": count}
                for status, count in sorted(route_counts.items())
            ],
        ),
        "",
        "## Runs",
        "",
        md_table(
            [
                "variant",
                "seed",
                "status",
                "final_nll",
                "best_nll",
                "elapsed_s",
                "run_dir",
            ],
            run_rows,
        ),
        "",
    ]
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_paths", nargs="+", type=Path, help="text files or corpus directories")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="report output directory (default: models/runs/char_lm_ablation_<timestamp>)",
    )
    parser.add_argument("--variant-set", choices=sorted(VARIANT_SETS), default=DEFAULT_VARIANT_SET)
    parser.add_argument("--preset", choices=sorted(sweep.PRESETS), default="smoke")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help="comma-separated integer seeds")
    parser.add_argument("--backend", default=sweep.DEFAULT_BACKEND, help="auto|wgpu|cuda|hip|cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batches", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--gen", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--memory", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--curvature", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--head-residual-scale", type=float, default=None)
    parser.add_argument("--val-start-fraction", type=float, default=None)
    parser.add_argument("--cargo-bin", default="cargo")
    parser.add_argument("--cargo-features", default=None)
    parser.add_argument("--no-default-features", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--quiet-runs", action="store_true")
    parser.add_argument("--curves", action="store_true")
    parser.add_argument("--summary-limit", type=int, default=8)
    parser.add_argument(
        "--summary-sort-metric",
        choices=sorted(sweep.SUMMARY_SORT_METRICS),
        default="final_nll",
    )
    parser.add_argument("--no-print-report", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        seeds = parse_csv_int(args.seeds, label="seeds")
        if args.backend not in {"auto", "wgpu", "cuda", "hip", "cpu"}:
            raise ValueError("--backend must be one of auto,wgpu,cuda,hip,cpu")
        if args.summary_limit < 0:
            raise ValueError("--summary-limit must be non-negative")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    variants = VARIANT_SETS[args.variant_set]
    settings = settings_from_args(args)
    run_root = args.run_root.resolve() if args.run_root else default_run_root()
    run_root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    manifest_path = run_root / "ablation_report.json"
    report_path = run_root / "ablation_report.md"
    runs: list[dict[str, Any]] = []
    successful_run_dirs: list[Path] = []
    failed = False
    manifest: dict[str, Any] = {
        "schema": "st.char_lm.ablation_report.v1",
        "started_at_unix": started,
        "run_root": str(run_root),
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "variant_set": args.variant_set,
        "variants": variant_rows(variants),
        "data_paths": [str(path) for path in args.data_paths],
        "seeds": seeds,
        "preset": args.preset,
        "settings": settings.__dict__,
        "summary_limit": args.summary_limit,
        "summary_sort_metric": args.summary_sort_metric,
        "cargo_features": args.cargo_features,
        "no_default_features": args.no_default_features,
        "dry_run": args.dry_run,
        "planned_runs": len(variants) * len(seeds),
        "runs": runs,
    }
    write_json(manifest_path, manifest)

    total = len(variants) * len(seeds)
    index = 0
    for variant in variants:
        variant_settings = settings_for_variant(settings, variant)
        for seed in seeds:
            index += 1
            run_name = slugged_run_name(
                variant=variant,
                seed=seed,
                settings=variant_settings,
            )
            run_dir = run_root / run_name
            log_path = run_dir / "process.log"
            command = sweep.build_command(
                cargo_bin=args.cargo_bin,
                cargo_features=(
                    args.cargo_features
                    if args.cargo_features is not None
                    else ("wgpu" if variant_settings.backend == "wgpu" else None)
                ),
                no_default_features=args.no_default_features,
                architecture=variant.architecture,
                data_paths=args.data_paths,
                run_dir=run_dir,
                char_feature=variant.char_feature,
                head_prior=variant.head_prior,
                seed=seed,
                settings=variant_settings,
                extra_args=args.extra_arg,
            )
            skipped = False
            failure_kind = None
            failure_detail = None
            if args.skip_existing and (run_dir / "summary.json").exists():
                returncode = 0
                elapsed = 0.0
                skipped = True
            else:
                if not args.quiet_runs:
                    print(f"[{index}/{total}] {variant.label} seed={seed}")
                    print("  " + shlex.join(command))
                returncode, elapsed = sweep.run_command(
                    command,
                    log_path,
                    dry_run=args.dry_run,
                )
                if returncode != 0:
                    failure_kind, failure_detail = sweep.classify_failure(
                        returncode,
                        log_path,
                    )

            trace_summary_path = None
            trace_summary_error = None
            if returncode == 0 and not args.dry_run:
                trace_summary_path, trace_summary_error = sweep.write_trainer_trace_summary(
                    run_dir
                )
            summary = sweep.read_json(run_dir / "summary.json")
            run_payload = sweep.read_json(run_dir / "run.json")
            missing_summary = returncode == 0 and summary is None and not args.dry_run
            if missing_summary:
                failure_kind = "missing_summary"
                failure_detail = "summary.json missing after successful command"
            run_failed = returncode != 0 or missing_summary
            if run_failed:
                failed = True
            run_record: dict[str, Any] = {
                "variant": variant.label,
                "variant_description": variant.description,
                "architecture": variant.architecture,
                "example": sweep.EXAMPLES[variant.architecture],
                "char_feature": variant.char_feature,
                "head_prior": variant.head_prior,
                "backend": variant_settings.backend,
                "seed": seed,
                "steps": variant_settings.steps,
                "embed_dim": variant_settings.embed_dim,
                "hidden": variant_settings.hidden,
                "memory": variant_settings.memory,
                "epochs": variant_settings.epochs,
                "batches": variant_settings.batches,
                "batch": variant_settings.batch,
                "eval_samples": variant_settings.eval_samples,
                "validation_start_fraction": variant_settings.val_start_fraction,
                "lr": variant_settings.lr,
                "head_residual_scale": variant_settings.head_residual_scale,
                "bigram_topk_guard": variant_settings.bigram_topk_guard,
                "bigram_topk_guard_k": variant_settings.bigram_topk_guard_k,
                "bigram_rank_guard": variant_settings.bigram_rank_guard,
                "bigram_rank_guard_margin": variant_settings.bigram_rank_guard_margin,
                "bigram_rank_guard_band": variant_settings.bigram_rank_guard_band,
                "bigram_rank_guard_min_candidates": (
                    variant_settings.bigram_rank_guard_min_candidates
                ),
                "bigram_soft_guard": variant_settings.bigram_soft_guard,
                "run_dir": str(run_dir),
                "log_path": str(log_path),
                "command": command,
                "returncode": returncode,
                "elapsed_seconds": elapsed,
                "skipped": skipped,
                "failed": run_failed,
                "run_status": (
                    "failed"
                    if run_failed
                    else ("dry_run" if args.dry_run else ("skipped" if skipped else "ok"))
                ),
                "failure_kind": failure_kind,
                "failure_detail": failure_detail,
                "summary_path": str(run_dir / "summary.json"),
                "has_summary": summary is not None,
                "run_json_path": str(run_dir / "run.json"),
                "has_run_json": run_payload is not None,
                "trainer_trace_summary_path": (
                    str(trace_summary_path) if trace_summary_path is not None else None
                ),
                "has_trainer_trace_summary": trace_summary_path is not None,
                "trainer_trace_summary_error": trace_summary_error,
            }
            if isinstance(summary, dict):
                run_record["best_validation_mean_nll"] = summary.get(
                    "best_validation_mean_nll"
                )
                final = summary.get("final_validation")
                if isinstance(final, dict):
                    run_record["final_validation_mean_nll"] = final.get("mean_nll")
            if isinstance(run_payload, dict):
                backend_runtime = run_payload.get("backend_runtime")
                if isinstance(backend_runtime, dict):
                    run_record["backend_runtime"] = backend_runtime
                tensor_policy = run_payload.get("tensor_policy")
                if isinstance(tensor_policy, dict):
                    run_record["tensor_policy"] = tensor_policy
                roundtable_summary = sweep.roundtable_wgpu_summary(run_payload)
                if roundtable_summary:
                    run_record["roundtable_wgpu"] = roundtable_summary
            runs.append(run_record)
            if returncode == 0 and summary is not None:
                successful_run_dirs.append(run_dir)
            write_json(manifest_path, manifest)
            if run_failed and not args.continue_on_error:
                print(f"run failed: {variant.label} seed={seed}; see {log_path}", file=sys.stderr)
                manifest["finished_at_unix"] = time.time()
                manifest["elapsed_seconds"] = manifest["finished_at_unix"] - started
                manifest["failed"] = True
                write_json(manifest_path, manifest)
                return returncode or 1

    compare_output = None
    compare_summary_output = None
    if not args.dry_run:
        compare_output = sweep.render_compare(
            successful_run_dirs,
            run_root,
            curves=args.curves,
        )
        if compare_output is not None:
            compare_summary_output = sweep.render_compare_summary(
                run_root,
                options=sweep.CompareSummaryOptions(
                    limit=args.summary_limit,
                    route_clean_only=False,
                    prefer_clean_route=True,
                    sort_metric=args.summary_sort_metric,
                ),
            )
            if compare_summary_output is None:
                failed = True

    manifest["finished_at_unix"] = time.time()
    manifest["elapsed_seconds"] = manifest["finished_at_unix"] - started
    manifest["compare_path"] = str(run_root / "compare.md") if compare_output is not None else None
    manifest["compare_json_path"] = (
        str(run_root / "compare.json") if compare_output is not None else None
    )
    compare_summary_json_path = run_root / "compare_summary.json"
    manifest["compare_summary_path"] = (
        str(run_root / "compare_summary.md")
        if (run_root / "compare_summary.md").exists()
        else None
    )
    manifest["compare_summary_json_path"] = (
        str(compare_summary_json_path) if compare_summary_json_path.exists() else None
    )
    manifest["failed"] = failed
    compare_summary_payload = sweep.read_json(compare_summary_json_path)
    if isinstance(compare_summary_payload, dict):
        manifest["compare_summary_route_status_counts"] = compare_summary_payload.get(
            "route_status_counts"
        )
        manifest["compare_summary_paired_recurrent_recommendations"] = (
            compare_summary_payload.get("paired_recurrent_recommendations")
        )
        manifest["compare_summary_bigram_guard_recommendations"] = (
            compare_summary_payload.get("bigram_guard_recommendations")
        )
    write_json(manifest_path, manifest)
    report_path.write_text(
        build_report_markdown(
            manifest=manifest,
            compare_summary=compare_summary_payload,
        ),
        encoding="utf-8",
    )

    print(f"ablation_report: {report_path}")
    print(f"manifest: {manifest_path}")
    if manifest.get("compare_path") is not None:
        print(f"compare: {manifest['compare_path']}")
    if manifest.get("compare_summary_path") is not None:
        print(f"compare_summary: {manifest['compare_summary_path']}")
    if not args.no_print_report:
        print(report_path.read_text(encoding="utf-8"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
