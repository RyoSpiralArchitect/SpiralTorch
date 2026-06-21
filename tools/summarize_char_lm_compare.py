#!/usr/bin/env python3
"""Summarize route-aware char-LM compare.json artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from backend_sweep_meta import md_cell


COHERENCE_GROUP_COLUMNS = [
    "context_scale",
    "self_score",
    "query_resid",
    "wave_kernel",
    "wave_dilations",
]

SUMMARY_HEADERS = [
    "rank",
    "source",
    "route_status",
    "arch",
    "recurrent",
    "backend",
    "head_prior",
    "head_resid",
    "bigram_guard",
    "bigram_guard_k",
    "bigram_rank_guard",
    "bigram_rank_margin",
    "bigram_rank_band",
    "bigram_rank_min",
    "bigram_soft_guard",
    "char_feature",
    "mode",
    *COHERENCE_GROUP_COLUMNS,
    "steps",
    "hidden",
    "embed_dim",
    "epochs",
    "batches",
    "batch",
    "eval_samples",
    "val_start",
    "lr",
    "runs",
    "val_start_actual_mean",
    "final_windows_mean",
    "unigram_windows_mean",
    "bigram_windows_mean",
    "rank_cov_windows_mean",
    "rank_cov_unbounded_mean",
    "rank_cov_band_mean",
    "rank_cov_min_mean",
    "rank_cov_guarded_mean",
    "rank_cov_effective_band_mean",
    "rank_cov_adaptive_fill_ratio_mean",
    "rank_cov_filled_mean",
    "rank_cov_zero_ratio_mean",
    "rank_cov_mass_mean",
    "rank_cov_band_ratio_mean",
    "rank_cov_topk_ratio_mean",
    "final_nll_mean",
    "best_nll_mean",
    "delta_nll_mean",
    "unigram_nll_mean",
    "bigram_nll_mean",
    "final_vs_unigram_mean",
    "final_vs_bigram_mean",
    "best_vs_unigram_mean",
    "best_vs_bigram_mean",
    "final_bigram_logprob_lift_mean",
    "final_bigram_rank_lift_mean",
    "final_bigram_target_rank_mean",
    "final_bigram_rank_debt_mean",
    "final_kl_bigram_mean",
    "final_top5_bigram_overlap_mean",
    "trace_step_ms_mean_mean",
    "trace_update_ratio_mean",
    "cpu_debt_ops_mean",
    "lstm_est_cpu_debt_ops_mean",
    "coherence_route_status",
    "coherence_route_status_counts",
    "coherence_route_debt_mean",
    "lstm_scan_backend_counts",
    "lstm_scan_fallback_counts",
]

PAIR_GROUP_COLUMNS = [
    "backend",
    "head_prior",
    "head_resid",
    "bigram_guard",
    "bigram_guard_k",
    "bigram_rank_guard",
    "bigram_rank_margin",
    "bigram_rank_band",
    "bigram_rank_min",
    "bigram_soft_guard",
    "char_feature",
    "mode",
    *COHERENCE_GROUP_COLUMNS,
    "steps",
    "hidden",
    "embed_dim",
    "epochs",
    "batches",
    "batch",
    "eval_samples",
    "val_start",
    "lr",
]

PAIR_DELTA_HEADERS = [
    "source",
    *PAIR_GROUP_COLUMNS,
    "candidate_recurrent",
    "baseline_recurrent",
    "candidate_runs",
    "baseline_runs",
    "candidate_final_nll",
    "baseline_final_nll",
    "final_nll_delta",
    "candidate_delta_nll",
    "baseline_delta_nll",
    "candidate_learning_status",
    "baseline_learning_status",
    "candidate_final_vs_bigram",
    "baseline_final_vs_bigram",
    "final_vs_bigram_delta",
    "candidate_trace_step_ms",
    "baseline_trace_step_ms",
    "trace_step_ms_delta",
    "trace_step_ms_ratio",
    "candidate_cpu_debt",
    "baseline_cpu_debt",
    "cpu_debt_delta",
    "cpu_debt_ratio",
    "quality_status",
    "latency_status",
    "cpu_debt_status",
    "efficiency_verdict",
    "candidate_route_status",
    "baseline_route_status",
]

PAIR_RECOMMENDATION_HEADERS = [
    "rank",
    "recommendation",
    "source",
    *PAIR_GROUP_COLUMNS,
    "candidate_recurrent",
    "baseline_recurrent",
    "quality_status",
    "latency_status",
    "cpu_debt_status",
    "efficiency_verdict",
    "final_nll_delta",
    "candidate_delta_nll",
    "baseline_delta_nll",
    "candidate_learning_status",
    "baseline_learning_status",
    "final_vs_bigram_delta",
    "trace_step_ms_ratio",
    "cpu_debt_ratio",
    "trace_step_ms_delta",
    "cpu_debt_delta",
    "candidate_route_status",
    "baseline_route_status",
]

BIGRAM_GUARD_GROUP_COLUMNS = [
    "arch",
    "recurrent",
    "backend",
    "head_prior",
    "head_resid",
    "bigram_guard_k",
    "bigram_rank_guard",
    "bigram_rank_margin",
    "bigram_rank_band",
    "bigram_rank_min",
    "bigram_soft_guard",
    "char_feature",
    "mode",
    *COHERENCE_GROUP_COLUMNS,
    "steps",
    "hidden",
    "embed_dim",
    "epochs",
    "batches",
    "batch",
    "eval_samples",
    "val_start",
    "lr",
]

BIGRAM_GUARD_DELTA_HEADERS = [
    "source",
    *BIGRAM_GUARD_GROUP_COLUMNS,
    "candidate_bigram_guard",
    "baseline_bigram_guard",
    "candidate_runs",
    "baseline_runs",
    "candidate_final_nll",
    "baseline_final_nll",
    "final_nll_delta",
    "candidate_final_vs_bigram",
    "baseline_final_vs_bigram",
    "final_vs_bigram_delta",
    "candidate_bigram_logprob_lift",
    "baseline_bigram_logprob_lift",
    "bigram_logprob_lift_delta",
    "candidate_bigram_rank_lift",
    "baseline_bigram_rank_lift",
    "bigram_rank_lift_delta",
    "candidate_top5_bigram_overlap",
    "baseline_top5_bigram_overlap",
    "top5_bigram_overlap_delta_pp",
    "nll_status",
    "bigram_gap_status",
    "bigram_logprob_status",
    "bigram_rank_status",
    "top5_bigram_status",
    "guard_verdict",
    "quality_status",
    "candidate_route_status",
    "baseline_route_status",
]

BIGRAM_GUARD_RECOMMENDATION_HEADERS = [
    "rank",
    "recommendation",
    "source",
    *BIGRAM_GUARD_GROUP_COLUMNS,
    "candidate_bigram_guard",
    "baseline_bigram_guard",
    "guard_verdict",
    "nll_status",
    "bigram_gap_status",
    "bigram_rank_status",
    "top5_bigram_status",
    "final_nll_delta",
    "final_vs_bigram_delta",
    "bigram_logprob_lift_delta",
    "bigram_rank_lift_delta",
    "top5_bigram_overlap_delta_pp",
    "candidate_route_status",
    "baseline_route_status",
]

BIGRAM_RANK_GUARD_GROUP_COLUMNS = [
    "arch",
    "recurrent",
    "backend",
    "head_prior",
    "head_resid",
    "bigram_guard",
    "bigram_guard_k",
    "bigram_rank_margin",
    "bigram_rank_band",
    "bigram_rank_min",
    "bigram_soft_guard",
    "char_feature",
    "mode",
    *COHERENCE_GROUP_COLUMNS,
    "steps",
    "hidden",
    "embed_dim",
    "epochs",
    "batches",
    "batch",
    "eval_samples",
    "val_start",
    "lr",
]

BIGRAM_RANK_GUARD_DELTA_HEADERS = [
    "source",
    *BIGRAM_RANK_GUARD_GROUP_COLUMNS,
    "candidate_bigram_rank_guard",
    "baseline_bigram_rank_guard",
    "candidate_runs",
    "baseline_runs",
    "candidate_final_nll",
    "baseline_final_nll",
    "final_nll_delta",
    "candidate_final_vs_bigram",
    "baseline_final_vs_bigram",
    "final_vs_bigram_delta",
    "candidate_bigram_logprob_lift",
    "baseline_bigram_logprob_lift",
    "bigram_logprob_lift_delta",
    "candidate_bigram_rank_debt",
    "baseline_bigram_rank_debt",
    "bigram_rank_debt_delta",
    "candidate_bigram_rank_lift",
    "baseline_bigram_rank_lift",
    "bigram_rank_lift_delta",
    "candidate_top5_bigram_overlap",
    "baseline_top5_bigram_overlap",
    "top5_bigram_overlap_delta_pp",
    "nll_status",
    "bigram_gap_status",
    "bigram_logprob_status",
    "rank_debt_status",
    "bigram_rank_status",
    "top5_bigram_status",
    "guard_verdict",
    "quality_status",
    "rank_status",
    "candidate_route_status",
    "baseline_route_status",
]

BIGRAM_RANK_GUARD_RECOMMENDATION_HEADERS = [
    "rank",
    "recommendation",
    "source",
    *BIGRAM_RANK_GUARD_GROUP_COLUMNS,
    "candidate_bigram_rank_guard",
    "baseline_bigram_rank_guard",
    "guard_verdict",
    "nll_status",
    "bigram_gap_status",
    "bigram_logprob_status",
    "rank_debt_status",
    "bigram_rank_status",
    "top5_bigram_status",
    "final_nll_delta",
    "final_vs_bigram_delta",
    "bigram_logprob_lift_delta",
    "bigram_rank_debt_delta",
    "bigram_rank_lift_delta",
    "top5_bigram_overlap_delta_pp",
    "candidate_route_status",
    "baseline_route_status",
]

BIGRAM_RANK_GUARD_SEED_GROUP_COLUMNS = [
    *BIGRAM_RANK_GUARD_GROUP_COLUMNS,
    "seed",
]

BIGRAM_RANK_GUARD_SEED_DELTA_HEADERS = [
    "source",
    *BIGRAM_RANK_GUARD_SEED_GROUP_COLUMNS,
    "candidate_bigram_rank_guard",
    "baseline_bigram_rank_guard",
    "candidate_final_nll",
    "baseline_final_nll",
    "final_nll_delta",
    "candidate_final_vs_bigram",
    "baseline_final_vs_bigram",
    "final_vs_bigram_delta",
    "candidate_bigram_rank_debt",
    "baseline_bigram_rank_debt",
    "bigram_rank_debt_delta",
    "candidate_bigram_rank_lift",
    "baseline_bigram_rank_lift",
    "bigram_rank_lift_delta",
    "candidate_top5_bigram_overlap",
    "baseline_top5_bigram_overlap",
    "top5_bigram_overlap_delta_pp",
    "nll_status",
    "bigram_gap_status",
    "rank_debt_status",
    "bigram_rank_status",
    "top5_bigram_status",
    "guard_verdict",
    "rank_status",
]

BIGRAM_RANK_GUARD_STABILITY_GROUP_COLUMNS = [
    *BIGRAM_RANK_GUARD_GROUP_COLUMNS,
    "candidate_bigram_rank_guard",
    "baseline_bigram_rank_guard",
]

BIGRAM_RANK_GUARD_STABILITY_HEADERS = [
    "source",
    *BIGRAM_RANK_GUARD_STABILITY_GROUP_COLUMNS,
    "seed_pairs",
    "rank_improved_seeds",
    "rank_neutral_seeds",
    "rank_regressed_seeds",
    "rank_missing_seeds",
    "mean_bigram_rank_debt_delta",
    "min_bigram_rank_debt_delta",
    "max_bigram_rank_debt_delta",
    "mean_bigram_rank_lift_delta",
    "mean_final_nll_delta",
    "mean_final_vs_bigram_delta",
    "mean_top5_bigram_overlap_delta_pp",
    "stability_verdict",
]

BIGRAM_RANK_BAND_GROUP_COLUMNS = [
    "arch",
    "recurrent",
    "backend",
    "head_prior",
    "head_resid",
    "bigram_guard",
    "bigram_guard_k",
    "bigram_rank_guard",
    "bigram_rank_margin",
    "bigram_rank_min",
    "bigram_soft_guard",
    "char_feature",
    "mode",
    *COHERENCE_GROUP_COLUMNS,
    "steps",
    "hidden",
    "embed_dim",
    "epochs",
    "batches",
    "batch",
    "eval_samples",
    "val_start",
    "lr",
]

BIGRAM_RANK_BAND_DELTA_HEADERS = [
    "source",
    *BIGRAM_RANK_BAND_GROUP_COLUMNS,
    "candidate_bigram_rank_band",
    "baseline_bigram_rank_band",
    "candidate_runs",
    "baseline_runs",
    "candidate_final_nll",
    "baseline_final_nll",
    "final_nll_delta",
    "candidate_final_vs_bigram",
    "baseline_final_vs_bigram",
    "final_vs_bigram_delta",
    "candidate_bigram_logprob_lift",
    "baseline_bigram_logprob_lift",
    "bigram_logprob_lift_delta",
    "candidate_bigram_rank_debt",
    "baseline_bigram_rank_debt",
    "bigram_rank_debt_delta",
    "candidate_bigram_rank_lift",
    "baseline_bigram_rank_lift",
    "bigram_rank_lift_delta",
    "candidate_top5_bigram_overlap",
    "baseline_top5_bigram_overlap",
    "top5_bigram_overlap_delta_pp",
    "nll_status",
    "bigram_gap_status",
    "bigram_logprob_status",
    "rank_debt_status",
    "bigram_rank_status",
    "top5_bigram_status",
    "band_verdict",
    "quality_status",
    "alignment_status",
    "candidate_route_status",
    "baseline_route_status",
]

BIGRAM_RANK_BAND_RECOMMENDATION_HEADERS = [
    "rank",
    "recommendation",
    "source",
    *BIGRAM_RANK_BAND_GROUP_COLUMNS,
    "candidate_bigram_rank_band",
    "baseline_bigram_rank_band",
    "band_verdict",
    "nll_status",
    "bigram_gap_status",
    "bigram_logprob_status",
    "rank_debt_status",
    "bigram_rank_status",
    "top5_bigram_status",
    "final_nll_delta",
    "final_vs_bigram_delta",
    "bigram_logprob_lift_delta",
    "bigram_rank_debt_delta",
    "bigram_rank_lift_delta",
    "top5_bigram_overlap_delta_pp",
    "candidate_route_status",
    "baseline_route_status",
]

BIGRAM_RANK_BAND_SEED_GROUP_COLUMNS = [
    *BIGRAM_RANK_BAND_GROUP_COLUMNS,
    "seed",
]

BIGRAM_RANK_BAND_SEED_DELTA_HEADERS = [
    "source",
    *BIGRAM_RANK_BAND_SEED_GROUP_COLUMNS,
    "candidate_bigram_rank_band",
    "baseline_bigram_rank_band",
    "candidate_final_nll",
    "baseline_final_nll",
    "final_nll_delta",
    "candidate_final_vs_bigram",
    "baseline_final_vs_bigram",
    "final_vs_bigram_delta",
    "candidate_bigram_rank_debt",
    "baseline_bigram_rank_debt",
    "bigram_rank_debt_delta",
    "candidate_bigram_rank_lift",
    "baseline_bigram_rank_lift",
    "bigram_rank_lift_delta",
    "candidate_top5_bigram_overlap",
    "baseline_top5_bigram_overlap",
    "top5_bigram_overlap_delta_pp",
    "nll_status",
    "bigram_gap_status",
    "rank_debt_status",
    "bigram_rank_status",
    "top5_bigram_status",
    "band_verdict",
    "alignment_status",
]

BIGRAM_RANK_BAND_STABILITY_GROUP_COLUMNS = [
    *BIGRAM_RANK_BAND_GROUP_COLUMNS,
    "candidate_bigram_rank_band",
    "baseline_bigram_rank_band",
]

BIGRAM_RANK_BAND_STABILITY_HEADERS = [
    "source",
    *BIGRAM_RANK_BAND_STABILITY_GROUP_COLUMNS,
    "seed_pairs",
    "alignment_improved_seeds",
    "alignment_neutral_seeds",
    "alignment_regressed_seeds",
    "alignment_missing_seeds",
    "mean_bigram_rank_debt_delta",
    "min_bigram_rank_debt_delta",
    "max_bigram_rank_debt_delta",
    "mean_bigram_rank_lift_delta",
    "mean_final_nll_delta",
    "mean_final_vs_bigram_delta",
    "mean_top5_bigram_overlap_delta_pp",
    "stability_verdict",
]

BIGRAM_RANK_MIN_GROUP_COLUMNS = [
    "arch",
    "recurrent",
    "backend",
    "head_prior",
    "head_resid",
    "bigram_guard",
    "bigram_guard_k",
    "bigram_rank_guard",
    "bigram_rank_margin",
    "bigram_rank_band",
    "bigram_soft_guard",
    "char_feature",
    "mode",
    *COHERENCE_GROUP_COLUMNS,
    "steps",
    "hidden",
    "embed_dim",
    "epochs",
    "batches",
    "batch",
    "eval_samples",
    "val_start",
    "lr",
]

BIGRAM_RANK_MIN_DELTA_HEADERS = [
    "source",
    *BIGRAM_RANK_MIN_GROUP_COLUMNS,
    "candidate_bigram_rank_min",
    "baseline_bigram_rank_min",
    "candidate_runs",
    "baseline_runs",
    "candidate_rank_cov_guarded",
    "baseline_rank_cov_guarded",
    "rank_cov_guarded_delta",
    "candidate_rank_cov_zero_ratio",
    "baseline_rank_cov_zero_ratio",
    "rank_cov_zero_ratio_delta",
    "candidate_rank_cov_filled",
    "baseline_rank_cov_filled",
    "rank_cov_filled_delta",
    "candidate_final_nll",
    "baseline_final_nll",
    "final_nll_delta",
    "candidate_final_vs_bigram",
    "baseline_final_vs_bigram",
    "final_vs_bigram_delta",
    "candidate_bigram_logprob_lift",
    "baseline_bigram_logprob_lift",
    "bigram_logprob_lift_delta",
    "candidate_bigram_rank_debt",
    "baseline_bigram_rank_debt",
    "bigram_rank_debt_delta",
    "candidate_bigram_rank_lift",
    "baseline_bigram_rank_lift",
    "bigram_rank_lift_delta",
    "candidate_top5_bigram_overlap",
    "baseline_top5_bigram_overlap",
    "top5_bigram_overlap_delta_pp",
    "nll_status",
    "bigram_gap_status",
    "bigram_logprob_status",
    "rank_debt_status",
    "bigram_rank_status",
    "top5_bigram_status",
    "rank_cov_guarded_status",
    "rank_cov_zero_status",
    "min_verdict",
    "quality_status",
    "alignment_status",
    "candidate_route_status",
    "baseline_route_status",
]

BIGRAM_RANK_MIN_RECOMMENDATION_HEADERS = [
    "rank",
    "recommendation",
    "source",
    *BIGRAM_RANK_MIN_GROUP_COLUMNS,
    "candidate_bigram_rank_min",
    "baseline_bigram_rank_min",
    "min_verdict",
    "nll_status",
    "bigram_gap_status",
    "bigram_logprob_status",
    "rank_debt_status",
    "bigram_rank_status",
    "top5_bigram_status",
    "rank_cov_guarded_status",
    "rank_cov_zero_status",
    "rank_cov_zero_ratio_delta",
    "rank_cov_guarded_delta",
    "rank_cov_filled_delta",
    "final_nll_delta",
    "final_vs_bigram_delta",
    "bigram_logprob_lift_delta",
    "bigram_rank_debt_delta",
    "bigram_rank_lift_delta",
    "top5_bigram_overlap_delta_pp",
    "candidate_route_status",
    "baseline_route_status",
]

BIGRAM_RANK_MIN_SEED_GROUP_COLUMNS = [
    *BIGRAM_RANK_MIN_GROUP_COLUMNS,
    "seed",
]

BIGRAM_RANK_MIN_SEED_DELTA_HEADERS = [
    "source",
    *BIGRAM_RANK_MIN_SEED_GROUP_COLUMNS,
    "candidate_bigram_rank_min",
    "baseline_bigram_rank_min",
    "candidate_rank_cov_guarded",
    "baseline_rank_cov_guarded",
    "rank_cov_guarded_delta",
    "candidate_rank_cov_zero_ratio",
    "baseline_rank_cov_zero_ratio",
    "rank_cov_zero_ratio_delta",
    "candidate_rank_cov_filled",
    "baseline_rank_cov_filled",
    "rank_cov_filled_delta",
    "candidate_final_nll",
    "baseline_final_nll",
    "final_nll_delta",
    "candidate_final_vs_bigram",
    "baseline_final_vs_bigram",
    "final_vs_bigram_delta",
    "candidate_bigram_rank_debt",
    "baseline_bigram_rank_debt",
    "bigram_rank_debt_delta",
    "candidate_bigram_rank_lift",
    "baseline_bigram_rank_lift",
    "bigram_rank_lift_delta",
    "candidate_top5_bigram_overlap",
    "baseline_top5_bigram_overlap",
    "top5_bigram_overlap_delta_pp",
    "nll_status",
    "bigram_gap_status",
    "rank_debt_status",
    "bigram_rank_status",
    "top5_bigram_status",
    "rank_cov_guarded_status",
    "rank_cov_zero_status",
    "min_verdict",
    "alignment_status",
]

BIGRAM_RANK_MIN_STABILITY_GROUP_COLUMNS = [
    *BIGRAM_RANK_MIN_GROUP_COLUMNS,
    "candidate_bigram_rank_min",
    "baseline_bigram_rank_min",
]

BIGRAM_RANK_MIN_STABILITY_HEADERS = [
    "source",
    *BIGRAM_RANK_MIN_STABILITY_GROUP_COLUMNS,
    "seed_pairs",
    "alignment_improved_seeds",
    "alignment_neutral_seeds",
    "alignment_regressed_seeds",
    "alignment_missing_seeds",
    "mean_rank_cov_zero_ratio_delta",
    "mean_rank_cov_guarded_delta",
    "mean_rank_cov_filled_delta",
    "mean_bigram_rank_debt_delta",
    "min_bigram_rank_debt_delta",
    "max_bigram_rank_debt_delta",
    "mean_bigram_rank_lift_delta",
    "mean_final_nll_delta",
    "mean_final_vs_bigram_delta",
    "mean_top5_bigram_overlap_delta_pp",
    "stability_verdict",
]

BIGRAM_RANK_MIN_STABLE_RECOMMENDATION_HEADERS = [
    "rank",
    "recommendation",
    "source",
    *BIGRAM_RANK_MIN_STABILITY_GROUP_COLUMNS,
    "seed_pairs",
    "alignment_improved_seeds",
    "alignment_neutral_seeds",
    "alignment_regressed_seeds",
    "mean_rank_cov_zero_ratio_delta",
    "mean_rank_cov_guarded_delta",
    "mean_rank_cov_filled_delta",
    "mean_bigram_rank_debt_delta",
    "min_bigram_rank_debt_delta",
    "max_bigram_rank_debt_delta",
    "mean_bigram_rank_lift_delta",
    "mean_final_nll_delta",
    "mean_final_vs_bigram_delta",
    "mean_top5_bigram_overlap_delta_pp",
    "stability_verdict",
]

BIGRAM_RANK_MIN_PROMOTION_GATE_HEADERS = [
    "decision",
    "failed",
    "fail_on_decisions",
    "total_rows",
    "strict_promotions",
    "bounded_promotions",
    "non_promoted_rows",
    "recommendation_rows",
    "verdict_counts",
]

BIGRAM_SOFT_GUARD_GROUP_COLUMNS = [
    "arch",
    "recurrent",
    "backend",
    "head_prior",
    "head_resid",
    "bigram_guard",
    "bigram_guard_k",
    "bigram_rank_guard",
    "bigram_rank_margin",
    "bigram_rank_band",
    "bigram_rank_min",
    "char_feature",
    "mode",
    *COHERENCE_GROUP_COLUMNS,
    "steps",
    "hidden",
    "embed_dim",
    "epochs",
    "batches",
    "batch",
    "eval_samples",
    "val_start",
    "lr",
]

BIGRAM_SOFT_GUARD_DELTA_HEADERS = [
    "source",
    *BIGRAM_SOFT_GUARD_GROUP_COLUMNS,
    "candidate_bigram_soft_guard",
    "baseline_bigram_soft_guard",
    "candidate_runs",
    "baseline_runs",
    "candidate_final_nll",
    "baseline_final_nll",
    "final_nll_delta",
    "candidate_final_vs_bigram",
    "baseline_final_vs_bigram",
    "final_vs_bigram_delta",
    "candidate_bigram_logprob_lift",
    "baseline_bigram_logprob_lift",
    "bigram_logprob_lift_delta",
    "candidate_bigram_rank_debt",
    "baseline_bigram_rank_debt",
    "bigram_rank_debt_delta",
    "candidate_bigram_rank_lift",
    "baseline_bigram_rank_lift",
    "bigram_rank_lift_delta",
    "candidate_top5_bigram_overlap",
    "baseline_top5_bigram_overlap",
    "top5_bigram_overlap_delta_pp",
    "nll_status",
    "bigram_gap_status",
    "bigram_logprob_status",
    "rank_debt_status",
    "bigram_rank_status",
    "top5_bigram_status",
    "guard_verdict",
    "quality_status",
    "alignment_status",
    "candidate_route_status",
    "baseline_route_status",
]

BIGRAM_SOFT_GUARD_RECOMMENDATION_HEADERS = [
    "rank",
    "recommendation",
    "source",
    *BIGRAM_SOFT_GUARD_GROUP_COLUMNS,
    "candidate_bigram_soft_guard",
    "baseline_bigram_soft_guard",
    "guard_verdict",
    "nll_status",
    "bigram_gap_status",
    "bigram_logprob_status",
    "rank_debt_status",
    "bigram_rank_status",
    "top5_bigram_status",
    "final_nll_delta",
    "final_vs_bigram_delta",
    "bigram_logprob_lift_delta",
    "bigram_rank_debt_delta",
    "bigram_rank_lift_delta",
    "top5_bigram_overlap_delta_pp",
    "candidate_route_status",
    "baseline_route_status",
]

BIGRAM_SOFT_GUARD_SEED_GROUP_COLUMNS = [
    *BIGRAM_SOFT_GUARD_GROUP_COLUMNS,
    "seed",
]

BIGRAM_SOFT_GUARD_SEED_DELTA_HEADERS = [
    "source",
    *BIGRAM_SOFT_GUARD_SEED_GROUP_COLUMNS,
    "candidate_bigram_soft_guard",
    "baseline_bigram_soft_guard",
    "candidate_final_nll",
    "baseline_final_nll",
    "final_nll_delta",
    "candidate_final_vs_bigram",
    "baseline_final_vs_bigram",
    "final_vs_bigram_delta",
    "candidate_bigram_rank_debt",
    "baseline_bigram_rank_debt",
    "bigram_rank_debt_delta",
    "candidate_bigram_rank_lift",
    "baseline_bigram_rank_lift",
    "bigram_rank_lift_delta",
    "candidate_top5_bigram_overlap",
    "baseline_top5_bigram_overlap",
    "top5_bigram_overlap_delta_pp",
    "nll_status",
    "bigram_gap_status",
    "rank_debt_status",
    "bigram_rank_status",
    "top5_bigram_status",
    "guard_verdict",
    "alignment_status",
]

BIGRAM_SOFT_GUARD_STABILITY_GROUP_COLUMNS = [
    *BIGRAM_SOFT_GUARD_GROUP_COLUMNS,
    "candidate_bigram_soft_guard",
    "baseline_bigram_soft_guard",
]

BIGRAM_SOFT_GUARD_STABILITY_HEADERS = [
    "source",
    *BIGRAM_SOFT_GUARD_STABILITY_GROUP_COLUMNS,
    "seed_pairs",
    "alignment_improved_seeds",
    "alignment_neutral_seeds",
    "alignment_regressed_seeds",
    "alignment_missing_seeds",
    "mean_bigram_rank_debt_delta",
    "min_bigram_rank_debt_delta",
    "max_bigram_rank_debt_delta",
    "mean_bigram_rank_lift_delta",
    "mean_final_nll_delta",
    "mean_final_vs_bigram_delta",
    "mean_top5_bigram_overlap_delta_pp",
    "stability_verdict",
]

BASELINE_DIFFICULTY_HEADERS = [
    "rank",
    "source",
    "arch",
    "recurrent",
    "backend",
    "head_prior",
    "head_resid",
    "bigram_guard",
    "bigram_rank_guard",
    "bigram_rank_margin",
    "bigram_rank_band",
    "bigram_rank_min",
    "bigram_soft_guard",
    "char_feature",
    "mode",
    *COHERENCE_GROUP_COLUMNS,
    "steps",
    "hidden",
    "epochs",
    "batches",
    "eval_samples",
    "val_start",
    "lr",
    "runs",
    "val_start_actual_mean",
    "final_windows_mean",
    "unigram_windows_mean",
    "bigram_windows_mean",
    "unigram_nll_mean",
    "bigram_nll_mean",
    "bigram_vs_unigram_delta",
    "bigram_baseline_status",
    "final_nll_mean",
    "final_vs_bigram_mean",
    "best_vs_bigram_mean",
    "model_vs_bigram_status",
    "delta_nll_mean",
    "learning_status",
    "route_status",
]

LEARNING_SCOREBOARD_HEADERS = [
    "rank",
    "source",
    "arch",
    "recurrent",
    "backend",
    "head_prior",
    "head_resid",
    "char_feature",
    *COHERENCE_GROUP_COLUMNS,
    "epochs",
    "batches",
    "batch",
    "eval_samples",
    "lr",
    "runs",
    "learning_gain",
    "learning_status",
    "final_nll_mean",
    "best_nll_mean",
    "final_minus_best",
    "bigram_gap",
    "bigram_gap_status",
    "trace_step_ms_mean",
    "cpu_debt_ops_mean",
    "coherence_route_status",
    "coherence_route_debt_mean",
    "gain_per_ms",
    "route_status",
]

ROUTE_DEBT_GROUP_COLUMNS = [
    "arch",
    "recurrent",
    "backend",
    "head_prior",
    "head_resid",
    "bigram_guard",
    "bigram_guard_k",
    "bigram_rank_guard",
    "bigram_rank_margin",
    "bigram_rank_band",
    "bigram_rank_min",
    "bigram_soft_guard",
    "char_feature",
    "mode",
    "context_scale",
    "self_score",
    "query_resid",
    "wave_kernel",
    "steps",
    "hidden",
    "embed_dim",
    "epochs",
    "batches",
    "batch",
    "eval_samples",
    "val_start",
    "lr",
]

ROUTE_DEBT_RECOMMENDATION_HEADERS = [
    "rank",
    "recommendation",
    "source",
    *ROUTE_DEBT_GROUP_COLUMNS,
    "candidate_wave_dilations",
    "baseline_wave_dilations",
    "candidate_runs",
    "baseline_runs",
    "candidate_final_nll",
    "baseline_final_nll",
    "final_nll_delta",
    "candidate_best_nll",
    "baseline_best_nll",
    "best_nll_delta",
    "candidate_final_vs_bigram",
    "baseline_final_vs_bigram",
    "final_vs_bigram_delta",
    "candidate_trace_step_ms",
    "baseline_trace_step_ms",
    "trace_step_ms_delta",
    "trace_step_ms_ratio",
    "candidate_cpu_debt",
    "baseline_cpu_debt",
    "cpu_debt_delta",
    "cpu_debt_ratio",
    "candidate_route_debt",
    "baseline_route_debt",
    "route_debt_delta",
    "route_debt_ratio",
    "quality_status",
    "latency_status",
    "cpu_debt_status",
    "route_debt_status",
    "route_debt_verdict",
    "candidate_coherence_route_status",
    "baseline_coherence_route_status",
    "candidate_route_status",
    "baseline_route_status",
]

ROUTE_DEBT_SUMMARY_HEADERS = [
    "decision",
    "failed",
    "fail_on_decisions",
    "recommendation_rows",
    "top_recommendation",
    "top_candidate_wave_dilations",
    "top_baseline_wave_dilations",
    "top_quality_status",
    "top_final_nll_delta",
    "top_route_debt_ratio",
    "top_cpu_debt_ratio",
    "top_trace_step_ms_ratio",
]

DEFAULT_PAIR_BASELINE_RECURRENT = "spiral"
DEFAULT_PAIR_CANDIDATE_RECURRENT = "lstm"
DEFAULT_BIGRAM_GUARD_BASELINE = 0.0
DEFAULT_BIGRAM_RANK_GUARD_BASELINE = 0.0
DEFAULT_BIGRAM_RANK_BAND_BASELINE = 0.0
DEFAULT_BIGRAM_RANK_MIN_BASELINE = 0.0
DEFAULT_BIGRAM_SOFT_GUARD_BASELINE = 0.0

ROUTE_COUNT_HEADERS = [
    "scope",
    "rows",
    "clean_route",
    "scan_route_mixed",
    "scan_fallback",
    "scan_route_mismatch",
    "no_scan_route",
]

VALID_ROUTE_STATUSES = tuple(
    header for header in ROUTE_COUNT_HEADERS if header not in {"scope", "rows"}
)

VALID_PAIR_QUALITY_STATUSES = ("improved", "missing", "neutral", "regressed")

VALID_EFFICIENCY_VERDICTS = (
    "candidate_better_quality_and_cost",
    "candidate_cost_regressed",
    "candidate_neutral",
    "candidate_quality_better_cost_neutral",
    "candidate_quality_neutral_cost_better",
    "candidate_quality_regressed",
    "inconclusive",
)

VALID_RANK_MIN_PROMOTION_DECISIONS = (
    "no_rank_min_evidence",
    "needs_tuning",
    "partial_promote_needs_tuning",
    "promote",
    "promote_with_bounded_watch",
)

VALID_ROUTE_DEBT_DECISIONS = (
    "no_route_debt_recommendation",
    "promote_lite_wave",
)

RECOMMENDED_EFFICIENCY_VERDICTS = {
    "candidate_better_quality_and_cost": "quality_improved_cost_improved",
    "candidate_quality_better_cost_neutral": "quality_improved_cost_neutral",
    "candidate_quality_neutral_cost_better": "quality_neutral_cost_improved",
}

RECOMMENDED_BIGRAM_GUARD_VERDICTS = {
    "guard_quality_and_topk_improved": "quality_and_topk_improved",
    "guard_topk_improved": "topk_improved_quality_neutral",
    "guard_quality_improved": "quality_improved_topk_neutral",
    "guard_quality_improved_topk_mixed": "quality_improved_topk_mixed",
}

RECOMMENDED_BIGRAM_RANK_GUARD_VERDICTS = {
    "rank_guard_quality_and_rank_improved": "quality_and_rank_improved",
    "rank_guard_rank_improved": "rank_improved_quality_neutral",
    "rank_guard_quality_improved": "quality_improved_rank_neutral",
    "rank_guard_quality_improved_rank_mixed": "quality_improved_rank_mixed",
}

RECOMMENDED_BIGRAM_SOFT_GUARD_VERDICTS = {
    "soft_guard_quality_and_alignment_improved": "quality_and_alignment_improved",
    "soft_guard_alignment_improved": "alignment_improved_quality_neutral",
    "soft_guard_quality_improved": "quality_improved_alignment_neutral",
    "soft_guard_quality_improved_alignment_mixed": "quality_improved_alignment_mixed",
}

RECOMMENDED_BIGRAM_RANK_BAND_VERDICTS = {
    "rank_band_quality_and_alignment_improved": "quality_and_alignment_improved",
    "rank_band_alignment_improved": "alignment_improved_quality_neutral",
    "rank_band_quality_improved": "quality_improved_alignment_neutral",
    "rank_band_quality_improved_alignment_mixed": "quality_improved_alignment_mixed",
}

RECOMMENDED_BIGRAM_RANK_MIN_VERDICTS = {
    "rank_min_quality_and_alignment_improved": "quality_and_alignment_improved",
    "rank_min_alignment_improved": "alignment_improved_quality_neutral",
    "rank_min_quality_improved": "quality_improved_alignment_neutral",
    "rank_min_quality_improved_alignment_mixed": "quality_improved_alignment_mixed",
}

RECOMMENDED_BIGRAM_RANK_MIN_STABILITY_VERDICTS = {
    "rank_min_seed_stably_improved": "stable_alignment_improved",
    "rank_min_seed_improved_or_neutral": "stable_alignment_improved_or_neutral",
    "rank_min_seed_bounded_mixed": "bounded_alignment_improved",
}

RECOMMENDED_ROUTE_DEBT_VERDICTS = {
    "route_debt_quality_improved_lighter": "quality_improved_route_debt_lower",
    "route_debt_quality_neutral_lighter": "quality_neutral_route_debt_lower",
}

SORT_METRIC_COLUMNS = {
    "final_nll": "final_nll_mean",
    "best_nll": "best_nll_mean",
    "delta_nll": "delta_nll_mean",
    "final_vs_unigram": "final_vs_unigram_mean",
    "final_vs_bigram": "final_vs_bigram_mean",
    "final_bigram_logprob_lift": "final_bigram_logprob_lift_mean",
    "final_bigram_rank_lift": "final_bigram_rank_lift_mean",
    "final_bigram_target_rank": "final_bigram_target_rank_mean",
    "final_bigram_rank_debt": "final_bigram_rank_debt_mean",
    "final_top5_bigram_overlap": "final_top5_bigram_overlap_mean",
    "coherence_route_debt": "coherence_route_debt_mean",
    "cpu_debt": "cpu_debt_ops_mean",
    "lstm_cpu_debt": "lstm_est_cpu_debt_ops_mean",
}
HIGHER_IS_BETTER_SORT_METRICS = {
    "final_bigram_logprob_lift",
    "final_bigram_rank_lift",
    "final_top5_bigram_overlap",
}

DEFAULT_SORT_METRIC = "final_nll"
PAIR_DELTA_TOLERANCE = 0.0005
RANK_MIN_BOUNDED_MIN_SEED_PAIRS = 4
RANK_MIN_BOUNDED_MAX_REGRESSED_RATIO = 0.25
RANK_MIN_BOUNDED_MAX_RANK_DEBT_DELTA = 0.0105
RANK_MIN_BOUNDED_MAX_TOP5_DROP_PP = 0.25


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def resolve_compare_paths(inputs: list[Path], *, recursive: bool) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            if recursive:
                paths.extend(sorted(item.rglob("compare.json")))
                continue
            compare_path = item / "compare.json"
            if not compare_path.exists():
                raise FileNotFoundError(
                    f"{item} is a directory without compare.json; pass --recursive to search below it"
                )
            paths.append(compare_path)
        else:
            paths.append(item)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    if not deduped:
        raise FileNotFoundError("no compare.json files found")
    return deduped


def parse_number_cell(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped == "-":
        return None
    if stripped.endswith("%"):
        stripped = stripped[:-1]
    try:
        return float(stripped)
    except ValueError:
        return None


def preferred_number_cell(row: dict[str, Any], column: str) -> float | None:
    if column.endswith("_mean"):
        raw_column = f"{column[:-5]}_raw_mean"
    else:
        raw_column = f"{column}_raw"
    raw_value = parse_number_cell(row.get(raw_column))
    if raw_value is not None:
        return raw_value
    return parse_number_cell(row.get(column))


def parse_count_cell(value: Any) -> dict[str, int]:
    if not isinstance(value, str) or not value or value == "-":
        return {}
    counts: dict[str, int] = {}
    for item in value.split(","):
        label, sep, count = item.rpartition(":")
        if not sep or not label:
            continue
        try:
            counts[label] = int(count)
        except ValueError:
            continue
    return counts


def fmt_delta(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def fmt_ratio(numerator: float | None, denominator: float | None) -> str:
    if numerator is None or denominator is None or denominator == 0.0:
        return "-"
    return f"{numerator / denominator:.4f}"


def ratio_for_sort(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None or denominator == 0.0:
        return float("inf")
    return numerator / denominator


def classify_lower_is_better_delta(
    delta: float | None,
    *,
    tolerance: float = PAIR_DELTA_TOLERANCE,
) -> str:
    if delta is None:
        return "missing"
    if delta < -tolerance:
        return "improved"
    if delta > tolerance:
        return "regressed"
    return "neutral"


def classify_higher_is_better_delta(
    delta: float | None,
    *,
    tolerance: float = PAIR_DELTA_TOLERANCE,
) -> str:
    if delta is None:
        return "missing"
    if delta > tolerance:
        return "improved"
    if delta < -tolerance:
        return "regressed"
    return "neutral"


def combine_quality_status(*statuses: str) -> str:
    present = [status for status in statuses if status != "missing"]
    if not present:
        return "missing"
    if any(status == "regressed" for status in present):
        return "regressed"
    if any(status == "improved" for status in present):
        return "improved"
    return "neutral"


def efficiency_verdict(
    *,
    quality_status: str,
    latency_status: str,
    cpu_debt_status: str,
) -> str:
    cost_statuses = [latency_status, cpu_debt_status]
    present_costs = [status for status in cost_statuses if status != "missing"]
    if quality_status == "regressed":
        return "candidate_quality_regressed"
    if not present_costs or quality_status == "missing":
        return "inconclusive"
    if any(status == "regressed" for status in present_costs):
        return "candidate_cost_regressed"
    if any(status == "improved" for status in present_costs):
        if quality_status == "improved":
            return "candidate_better_quality_and_cost"
        return "candidate_quality_neutral_cost_better"
    if quality_status == "improved":
        return "candidate_quality_better_cost_neutral"
    return "candidate_neutral"


def route_status(row: dict[str, Any]) -> str:
    existing = row.get("route_status")
    if isinstance(existing, str) and existing and existing != "-":
        return existing

    scan_backends = parse_count_cell(row.get("lstm_scan_backend_counts"))
    if not scan_backends:
        return "no_scan_route"

    fallbacks = parse_count_cell(row.get("lstm_scan_fallback_counts"))
    active_fallbacks = {
        label: count
        for label, count in fallbacks.items()
        if label != "none" and count > 0
    }
    if active_fallbacks:
        return "scan_fallback"

    requested_backend = row.get("backend")
    if isinstance(requested_backend, str) and requested_backend in {
        "wgpu",
        "cuda",
        "hip",
        "mps",
    }:
        if requested_backend not in scan_backends:
            return "scan_route_mismatch"
    if len(scan_backends) > 1:
        return "scan_route_mixed"
    return "clean_route"


def route_penalty(status: str) -> int:
    order = {
        "clean_route": 0,
        "scan_route_mixed": 1,
        "scan_fallback": 2,
        "scan_route_mismatch": 3,
        "no_scan_route": 4,
    }
    return order.get(status, 5)


def scan_route_applicable(row: dict[str, Any]) -> bool:
    recurrent = row.get("recurrent")
    if isinstance(recurrent, str) and recurrent == "lstm":
        return True
    arch = row.get("arch")
    if isinstance(arch, str) and "lstm" in arch:
        return True
    return bool(parse_count_cell(row.get("lstm_scan_backend_counts")))


def route_penalty_for_row(row: dict[str, Any], status: str) -> int:
    if status == "no_scan_route" and not scan_route_applicable(row):
        return 0
    return route_penalty(status)


def source_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("aggregate_runs")
    if not isinstance(rows, list) or not rows:
        rows = payload.get("top_aggregate_runs")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def run_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("runs")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def paired_recurrent_deltas(
    payloads: list[tuple[str, dict[str, Any]]],
    *,
    baseline_recurrent: str = DEFAULT_PAIR_BASELINE_RECURRENT,
    candidate_recurrent: str = DEFAULT_PAIR_CANDIDATE_RECURRENT,
) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for source, payload in payloads:
        grouped: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}
        for row in source_rows(payload):
            recurrent = row.get("recurrent")
            if not isinstance(recurrent, str) or recurrent in {"", "-"}:
                continue
            key = tuple(str(row.get(column, "-")) for column in PAIR_GROUP_COLUMNS)
            grouped.setdefault(key, {})[recurrent] = row

        for key, rows_by_recurrent in sorted(grouped.items()):
            baseline = rows_by_recurrent.get(baseline_recurrent)
            candidate = rows_by_recurrent.get(candidate_recurrent)
            if baseline is None or candidate is None:
                continue

            candidate_final_nll = parse_number_cell(candidate.get("final_nll_mean"))
            baseline_final_nll = parse_number_cell(baseline.get("final_nll_mean"))
            candidate_delta_nll = parse_number_cell(candidate.get("delta_nll_mean"))
            baseline_delta_nll = parse_number_cell(baseline.get("delta_nll_mean"))
            candidate_bigram = parse_number_cell(candidate.get("final_vs_bigram_mean"))
            baseline_bigram = parse_number_cell(baseline.get("final_vs_bigram_mean"))
            candidate_step_ms = parse_number_cell(candidate.get("trace_step_ms_mean_mean"))
            baseline_step_ms = parse_number_cell(baseline.get("trace_step_ms_mean_mean"))
            candidate_cpu_debt = parse_number_cell(candidate.get("cpu_debt_ops_mean"))
            baseline_cpu_debt = parse_number_cell(baseline.get("cpu_debt_ops_mean"))
            final_nll_delta = (
                candidate_final_nll - baseline_final_nll
                if candidate_final_nll is not None and baseline_final_nll is not None
                else None
            )
            bigram_delta = (
                candidate_bigram - baseline_bigram
                if candidate_bigram is not None and baseline_bigram is not None
                else None
            )
            step_ms_delta = (
                candidate_step_ms - baseline_step_ms
                if candidate_step_ms is not None and baseline_step_ms is not None
                else None
            )
            cpu_debt_delta = (
                candidate_cpu_debt - baseline_cpu_debt
                if candidate_cpu_debt is not None and baseline_cpu_debt is not None
                else None
            )
            quality_status = combine_quality_status(
                classify_lower_is_better_delta(final_nll_delta),
                classify_lower_is_better_delta(bigram_delta),
            )
            candidate_learning_status = classify_lower_is_better_delta(
                candidate_delta_nll
            )
            baseline_learning_status = classify_lower_is_better_delta(
                baseline_delta_nll
            )
            latency_status = classify_lower_is_better_delta(step_ms_delta)
            cpu_debt_status = classify_lower_is_better_delta(cpu_debt_delta)

            pair = {
                "source": source,
                **dict(zip(PAIR_GROUP_COLUMNS, key, strict=True)),
                "candidate_recurrent": candidate_recurrent,
                "baseline_recurrent": baseline_recurrent,
                "candidate_runs": str(candidate.get("runs", "-")),
                "baseline_runs": str(baseline.get("runs", "-")),
                "candidate_final_nll": str(candidate.get("final_nll_mean", "-")),
                "baseline_final_nll": str(baseline.get("final_nll_mean", "-")),
                "final_nll_delta": fmt_delta(final_nll_delta),
                "candidate_delta_nll": str(candidate.get("delta_nll_mean", "-")),
                "baseline_delta_nll": str(baseline.get("delta_nll_mean", "-")),
                "candidate_learning_status": candidate_learning_status,
                "baseline_learning_status": baseline_learning_status,
                "candidate_final_vs_bigram": str(candidate.get("final_vs_bigram_mean", "-")),
                "baseline_final_vs_bigram": str(baseline.get("final_vs_bigram_mean", "-")),
                "final_vs_bigram_delta": fmt_delta(bigram_delta),
                "candidate_trace_step_ms": str(
                    candidate.get("trace_step_ms_mean_mean", "-")
                ),
                "baseline_trace_step_ms": str(baseline.get("trace_step_ms_mean_mean", "-")),
                "trace_step_ms_delta": fmt_delta(step_ms_delta),
                "trace_step_ms_ratio": fmt_ratio(candidate_step_ms, baseline_step_ms),
                "candidate_cpu_debt": str(candidate.get("cpu_debt_ops_mean", "-")),
                "baseline_cpu_debt": str(baseline.get("cpu_debt_ops_mean", "-")),
                "cpu_debt_delta": fmt_delta(cpu_debt_delta),
                "cpu_debt_ratio": fmt_ratio(candidate_cpu_debt, baseline_cpu_debt),
                "quality_status": quality_status,
                "latency_status": latency_status,
                "cpu_debt_status": cpu_debt_status,
                "efficiency_verdict": efficiency_verdict(
                    quality_status=quality_status,
                    latency_status=latency_status,
                    cpu_debt_status=cpu_debt_status,
                ),
                "candidate_route_status": route_status(candidate),
                "baseline_route_status": route_status(baseline),
            }
            pairs.append(pair)
    return pairs


def fmt_guard_value(value: float) -> str:
    return f"{value:.4f}"


def fmt_rank_min_value(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 1.0e-9:
        return str(int(rounded))
    return f"{value:g}"


def bigram_guard_verdict(
    *,
    nll_status: str,
    bigram_gap_status: str,
    bigram_logprob_status: str,
    bigram_rank_status: str,
    top5_bigram_status: str,
) -> str:
    quality_statuses = [nll_status, bigram_gap_status, bigram_logprob_status]
    topk_statuses = [bigram_rank_status, top5_bigram_status]
    quality_regressed = any(status == "regressed" for status in quality_statuses)
    quality_improved = any(status == "improved" for status in quality_statuses)
    topk_regressed = any(status == "regressed" for status in topk_statuses)
    topk_improved = any(status == "improved" for status in topk_statuses)
    quality_missing = all(status == "missing" for status in quality_statuses)
    topk_missing = all(status == "missing" for status in topk_statuses)

    if quality_regressed:
        return "guard_quality_regressed"
    if topk_improved and topk_regressed:
        if quality_improved:
            return "guard_quality_improved_topk_mixed"
        return "guard_topk_mixed"
    if topk_regressed:
        if quality_improved:
            return "guard_quality_improved_topk_regressed"
        return "guard_topk_regressed"
    if quality_improved and topk_improved:
        return "guard_quality_and_topk_improved"
    if quality_improved:
        return "guard_quality_improved"
    if topk_improved:
        return "guard_topk_improved"
    if quality_missing and topk_missing:
        return "guard_inconclusive"
    return "guard_neutral"


def paired_bigram_guard_deltas(
    payloads: list[tuple[str, dict[str, Any]]],
    *,
    baseline_guard: float = DEFAULT_BIGRAM_GUARD_BASELINE,
) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for source, payload in payloads:
        grouped: dict[tuple[str, ...], dict[float, dict[str, Any]]] = {}
        for row in source_rows(payload):
            guard = parse_number_cell(row.get("bigram_guard"))
            if guard is None:
                continue
            key = tuple(
                str(row.get(column, "-")) for column in BIGRAM_GUARD_GROUP_COLUMNS
            )
            grouped.setdefault(key, {})[guard] = row

        for key, rows_by_guard in sorted(grouped.items()):
            baseline = rows_by_guard.get(baseline_guard)
            if baseline is None:
                continue
            for guard in sorted(value for value in rows_by_guard if value != baseline_guard):
                candidate = rows_by_guard[guard]
                candidate_final_nll = parse_number_cell(candidate.get("final_nll_mean"))
                baseline_final_nll = parse_number_cell(baseline.get("final_nll_mean"))
                candidate_bigram = parse_number_cell(
                    candidate.get("final_vs_bigram_mean")
                )
                baseline_bigram = parse_number_cell(baseline.get("final_vs_bigram_mean"))
                candidate_logprob_lift = parse_number_cell(
                    candidate.get("final_bigram_logprob_lift_mean")
                )
                baseline_logprob_lift = parse_number_cell(
                    baseline.get("final_bigram_logprob_lift_mean")
                )
                candidate_rank_lift = parse_number_cell(
                    candidate.get("final_bigram_rank_lift_mean")
                )
                baseline_rank_lift = parse_number_cell(
                    baseline.get("final_bigram_rank_lift_mean")
                )
                candidate_top5 = parse_number_cell(
                    candidate.get("final_top5_bigram_overlap_mean")
                )
                baseline_top5 = parse_number_cell(
                    baseline.get("final_top5_bigram_overlap_mean")
                )
                final_nll_delta = (
                    candidate_final_nll - baseline_final_nll
                    if candidate_final_nll is not None
                    and baseline_final_nll is not None
                    else None
                )
                bigram_delta = (
                    candidate_bigram - baseline_bigram
                    if candidate_bigram is not None and baseline_bigram is not None
                    else None
                )
                logprob_lift_delta = (
                    candidate_logprob_lift - baseline_logprob_lift
                    if candidate_logprob_lift is not None
                    and baseline_logprob_lift is not None
                    else None
                )
                rank_lift_delta = (
                    candidate_rank_lift - baseline_rank_lift
                    if candidate_rank_lift is not None and baseline_rank_lift is not None
                    else None
                )
                top5_delta = (
                    candidate_top5 - baseline_top5
                    if candidate_top5 is not None and baseline_top5 is not None
                    else None
                )
                nll_status = classify_lower_is_better_delta(final_nll_delta)
                bigram_gap_status = classify_lower_is_better_delta(bigram_delta)
                bigram_logprob_status = classify_higher_is_better_delta(
                    logprob_lift_delta
                )
                bigram_rank_status = classify_higher_is_better_delta(rank_lift_delta)
                top5_bigram_status = classify_higher_is_better_delta(top5_delta)
                quality_status = combine_quality_status(
                    nll_status,
                    bigram_gap_status,
                    bigram_logprob_status,
                    bigram_rank_status,
                    top5_bigram_status,
                )
                pair = {
                    "source": source,
                    **dict(zip(BIGRAM_GUARD_GROUP_COLUMNS, key, strict=True)),
                    "candidate_bigram_guard": fmt_guard_value(guard),
                    "baseline_bigram_guard": fmt_guard_value(baseline_guard),
                    "candidate_runs": str(candidate.get("runs", "-")),
                    "baseline_runs": str(baseline.get("runs", "-")),
                    "candidate_final_nll": str(candidate.get("final_nll_mean", "-")),
                    "baseline_final_nll": str(baseline.get("final_nll_mean", "-")),
                    "final_nll_delta": fmt_delta(final_nll_delta),
                    "candidate_final_vs_bigram": str(
                        candidate.get("final_vs_bigram_mean", "-")
                    ),
                    "baseline_final_vs_bigram": str(
                        baseline.get("final_vs_bigram_mean", "-")
                    ),
                    "final_vs_bigram_delta": fmt_delta(bigram_delta),
                    "candidate_bigram_logprob_lift": str(
                        candidate.get("final_bigram_logprob_lift_mean", "-")
                    ),
                    "baseline_bigram_logprob_lift": str(
                        baseline.get("final_bigram_logprob_lift_mean", "-")
                    ),
                    "bigram_logprob_lift_delta": fmt_delta(logprob_lift_delta),
                    "candidate_bigram_rank_lift": str(
                        candidate.get("final_bigram_rank_lift_mean", "-")
                    ),
                    "baseline_bigram_rank_lift": str(
                        baseline.get("final_bigram_rank_lift_mean", "-")
                    ),
                    "bigram_rank_lift_delta": fmt_delta(rank_lift_delta),
                    "candidate_top5_bigram_overlap": str(
                        candidate.get("final_top5_bigram_overlap_mean", "-")
                    ),
                    "baseline_top5_bigram_overlap": str(
                        baseline.get("final_top5_bigram_overlap_mean", "-")
                    ),
                    "top5_bigram_overlap_delta_pp": fmt_delta(top5_delta),
                    "nll_status": nll_status,
                    "bigram_gap_status": bigram_gap_status,
                    "bigram_logprob_status": bigram_logprob_status,
                    "bigram_rank_status": bigram_rank_status,
                    "top5_bigram_status": top5_bigram_status,
                    "guard_verdict": bigram_guard_verdict(
                        nll_status=nll_status,
                        bigram_gap_status=bigram_gap_status,
                        bigram_logprob_status=bigram_logprob_status,
                        bigram_rank_status=bigram_rank_status,
                        top5_bigram_status=top5_bigram_status,
                    ),
                    "quality_status": quality_status,
                    "candidate_route_status": route_status(candidate),
                    "baseline_route_status": route_status(baseline),
                }
                pairs.append(pair)
    return pairs


def pair_metric_for_sort(row: dict[str, Any], key: str) -> float:
    value = parse_number_cell(row.get(key))
    if value is None:
        return float("inf")
    return value


def pair_metric_desc_for_sort(row: dict[str, Any], key: str) -> float:
    value = parse_number_cell(row.get(key))
    if value is None:
        return float("inf")
    return -value


def paired_bigram_guard_recommendations(
    pairs: list[dict[str, str]],
    *,
    limit: int,
) -> list[dict[str, str]]:
    candidates = [
        pair
        for pair in pairs
        if pair.get("guard_verdict") in RECOMMENDED_BIGRAM_GUARD_VERDICTS
    ]
    verdict_rank = {
        "guard_quality_and_topk_improved": 0,
        "guard_topk_improved": 1,
        "guard_quality_improved": 2,
        "guard_quality_improved_topk_mixed": 3,
    }
    sorted_pairs = sorted(
        candidates,
        key=lambda pair: (
            verdict_rank.get(str(pair.get("guard_verdict")), 99),
            pair_metric_for_sort(pair, "final_nll_delta"),
            pair_metric_for_sort(pair, "final_vs_bigram_delta"),
            pair_metric_desc_for_sort(pair, "top5_bigram_overlap_delta_pp"),
            pair_metric_desc_for_sort(pair, "bigram_rank_lift_delta"),
            pair_metric_for_sort(pair, "candidate_final_nll"),
            pair_metric_for_sort(pair, "candidate_bigram_guard"),
            str(pair.get("source", "")),
        ),
    )
    if limit > 0:
        sorted_pairs = sorted_pairs[:limit]

    recommendations: list[dict[str, str]] = []
    for rank, pair in enumerate(sorted_pairs, start=1):
        verdict = str(pair.get("guard_verdict", "-"))
        row = {
            header: str(pair.get(header, "-"))
            for header in BIGRAM_GUARD_RECOMMENDATION_HEADERS
        }
        row["rank"] = str(rank)
        row["recommendation"] = RECOMMENDED_BIGRAM_GUARD_VERDICTS.get(verdict, "-")
        recommendations.append(row)
    return recommendations


def bigram_rank_guard_verdict(
    *,
    nll_status: str,
    bigram_gap_status: str,
    bigram_logprob_status: str,
    rank_debt_status: str,
    bigram_rank_status: str,
    top5_bigram_status: str,
) -> str:
    quality_statuses = [nll_status, bigram_gap_status, bigram_logprob_status]
    rank_statuses = [rank_debt_status, bigram_rank_status, top5_bigram_status]
    quality_regressed = any(status == "regressed" for status in quality_statuses)
    quality_improved = any(status == "improved" for status in quality_statuses)
    rank_regressed = any(status == "regressed" for status in rank_statuses)
    rank_improved = any(status == "improved" for status in rank_statuses)
    quality_missing = all(status == "missing" for status in quality_statuses)
    rank_missing = all(status == "missing" for status in rank_statuses)

    if quality_regressed:
        return "rank_guard_quality_regressed"
    if rank_improved and rank_regressed:
        if quality_improved:
            return "rank_guard_quality_improved_rank_mixed"
        return "rank_guard_rank_mixed"
    if rank_regressed:
        if quality_improved:
            return "rank_guard_quality_improved_rank_regressed"
        return "rank_guard_rank_regressed"
    if quality_improved and rank_improved:
        return "rank_guard_quality_and_rank_improved"
    if rank_improved:
        return "rank_guard_rank_improved"
    if quality_improved:
        return "rank_guard_quality_improved"
    if quality_missing and rank_missing:
        return "rank_guard_inconclusive"
    return "rank_guard_neutral"


def paired_bigram_rank_guard_deltas(
    payloads: list[tuple[str, dict[str, Any]]],
    *,
    baseline_guard: float = DEFAULT_BIGRAM_RANK_GUARD_BASELINE,
) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for source, payload in payloads:
        grouped: dict[tuple[str, ...], dict[float, dict[str, Any]]] = {}
        for row in source_rows(payload):
            guard = parse_number_cell(row.get("bigram_rank_guard"))
            if guard is None:
                continue
            key = tuple(
                str(row.get(column, "-")) for column in BIGRAM_RANK_GUARD_GROUP_COLUMNS
            )
            grouped.setdefault(key, {})[guard] = row

        for key, rows_by_guard in sorted(grouped.items()):
            baseline = rows_by_guard.get(baseline_guard)
            if baseline is None:
                continue
            for guard in sorted(value for value in rows_by_guard if value != baseline_guard):
                candidate = rows_by_guard[guard]
                candidate_final_nll = parse_number_cell(candidate.get("final_nll_mean"))
                baseline_final_nll = parse_number_cell(baseline.get("final_nll_mean"))
                candidate_bigram = parse_number_cell(
                    candidate.get("final_vs_bigram_mean")
                )
                baseline_bigram = parse_number_cell(baseline.get("final_vs_bigram_mean"))
                candidate_logprob_lift = parse_number_cell(
                    candidate.get("final_bigram_logprob_lift_mean")
                )
                baseline_logprob_lift = parse_number_cell(
                    baseline.get("final_bigram_logprob_lift_mean")
                )
                candidate_rank_debt = preferred_number_cell(
                    candidate,
                    "final_bigram_rank_debt_mean",
                )
                baseline_rank_debt = preferred_number_cell(
                    baseline,
                    "final_bigram_rank_debt_mean",
                )
                candidate_rank_lift = preferred_number_cell(
                    candidate,
                    "final_bigram_rank_lift_mean",
                )
                baseline_rank_lift = preferred_number_cell(
                    baseline,
                    "final_bigram_rank_lift_mean",
                )
                candidate_top5 = preferred_number_cell(
                    candidate,
                    "final_top5_bigram_overlap_mean",
                )
                baseline_top5 = preferred_number_cell(
                    baseline,
                    "final_top5_bigram_overlap_mean",
                )
                final_nll_delta = (
                    candidate_final_nll - baseline_final_nll
                    if candidate_final_nll is not None
                    and baseline_final_nll is not None
                    else None
                )
                bigram_delta = (
                    candidate_bigram - baseline_bigram
                    if candidate_bigram is not None and baseline_bigram is not None
                    else None
                )
                logprob_lift_delta = (
                    candidate_logprob_lift - baseline_logprob_lift
                    if candidate_logprob_lift is not None
                    and baseline_logprob_lift is not None
                    else None
                )
                rank_debt_delta = (
                    candidate_rank_debt - baseline_rank_debt
                    if candidate_rank_debt is not None and baseline_rank_debt is not None
                    else None
                )
                rank_lift_delta = (
                    candidate_rank_lift - baseline_rank_lift
                    if candidate_rank_lift is not None and baseline_rank_lift is not None
                    else None
                )
                top5_delta = (
                    candidate_top5 - baseline_top5
                    if candidate_top5 is not None and baseline_top5 is not None
                    else None
                )
                nll_status = classify_lower_is_better_delta(final_nll_delta)
                bigram_gap_status = classify_lower_is_better_delta(bigram_delta)
                bigram_logprob_status = classify_higher_is_better_delta(
                    logprob_lift_delta
                )
                rank_debt_status = classify_lower_is_better_delta(rank_debt_delta)
                bigram_rank_status = classify_higher_is_better_delta(rank_lift_delta)
                top5_bigram_status = classify_higher_is_better_delta(top5_delta)
                quality_status = combine_quality_status(
                    nll_status,
                    bigram_gap_status,
                    bigram_logprob_status,
                )
                rank_status = combine_quality_status(
                    rank_debt_status,
                    bigram_rank_status,
                    top5_bigram_status,
                )
                pair = {
                    "source": source,
                    **dict(zip(BIGRAM_RANK_GUARD_GROUP_COLUMNS, key, strict=True)),
                    "candidate_bigram_rank_guard": fmt_guard_value(guard),
                    "baseline_bigram_rank_guard": fmt_guard_value(baseline_guard),
                    "candidate_runs": str(candidate.get("runs", "-")),
                    "baseline_runs": str(baseline.get("runs", "-")),
                    "candidate_final_nll": str(candidate.get("final_nll_mean", "-")),
                    "baseline_final_nll": str(baseline.get("final_nll_mean", "-")),
                    "final_nll_delta": fmt_delta(final_nll_delta),
                    "candidate_final_vs_bigram": str(
                        candidate.get("final_vs_bigram_mean", "-")
                    ),
                    "baseline_final_vs_bigram": str(
                        baseline.get("final_vs_bigram_mean", "-")
                    ),
                    "final_vs_bigram_delta": fmt_delta(bigram_delta),
                    "candidate_bigram_logprob_lift": str(
                        candidate.get("final_bigram_logprob_lift_mean", "-")
                    ),
                    "baseline_bigram_logprob_lift": str(
                        baseline.get("final_bigram_logprob_lift_mean", "-")
                    ),
                    "bigram_logprob_lift_delta": fmt_delta(logprob_lift_delta),
                    "candidate_bigram_rank_debt": str(
                        candidate.get("final_bigram_rank_debt_mean", "-")
                    ),
                    "baseline_bigram_rank_debt": str(
                        baseline.get("final_bigram_rank_debt_mean", "-")
                    ),
                    "bigram_rank_debt_delta": fmt_delta(rank_debt_delta),
                    "candidate_bigram_rank_lift": str(
                        candidate.get("final_bigram_rank_lift_mean", "-")
                    ),
                    "baseline_bigram_rank_lift": str(
                        baseline.get("final_bigram_rank_lift_mean", "-")
                    ),
                    "bigram_rank_lift_delta": fmt_delta(rank_lift_delta),
                    "candidate_top5_bigram_overlap": str(
                        candidate.get("final_top5_bigram_overlap_mean", "-")
                    ),
                    "baseline_top5_bigram_overlap": str(
                        baseline.get("final_top5_bigram_overlap_mean", "-")
                    ),
                    "top5_bigram_overlap_delta_pp": fmt_delta(top5_delta),
                    "nll_status": nll_status,
                    "bigram_gap_status": bigram_gap_status,
                    "bigram_logprob_status": bigram_logprob_status,
                    "rank_debt_status": rank_debt_status,
                    "bigram_rank_status": bigram_rank_status,
                    "top5_bigram_status": top5_bigram_status,
                    "guard_verdict": bigram_rank_guard_verdict(
                        nll_status=nll_status,
                        bigram_gap_status=bigram_gap_status,
                        bigram_logprob_status=bigram_logprob_status,
                        rank_debt_status=rank_debt_status,
                        bigram_rank_status=bigram_rank_status,
                        top5_bigram_status=top5_bigram_status,
                    ),
                    "quality_status": quality_status,
                    "rank_status": rank_status,
                    "candidate_route_status": route_status(candidate),
                    "baseline_route_status": route_status(baseline),
                }
                pairs.append(pair)
    return pairs


def paired_bigram_rank_guard_recommendations(
    pairs: list[dict[str, str]],
    *,
    limit: int,
) -> list[dict[str, str]]:
    candidates = [
        pair
        for pair in pairs
        if pair.get("guard_verdict") in RECOMMENDED_BIGRAM_RANK_GUARD_VERDICTS
    ]
    verdict_rank = {
        "rank_guard_quality_and_rank_improved": 0,
        "rank_guard_rank_improved": 1,
        "rank_guard_quality_improved": 2,
        "rank_guard_quality_improved_rank_mixed": 3,
    }
    sorted_pairs = sorted(
        candidates,
        key=lambda pair: (
            verdict_rank.get(str(pair.get("guard_verdict")), 99),
            pair_metric_for_sort(pair, "bigram_rank_debt_delta"),
            pair_metric_desc_for_sort(pair, "bigram_rank_lift_delta"),
            pair_metric_for_sort(pair, "final_nll_delta"),
            pair_metric_for_sort(pair, "final_vs_bigram_delta"),
            pair_metric_desc_for_sort(pair, "top5_bigram_overlap_delta_pp"),
            pair_metric_for_sort(pair, "candidate_final_nll"),
            pair_metric_for_sort(pair, "candidate_bigram_rank_guard"),
            str(pair.get("source", "")),
        ),
    )
    if limit > 0:
        sorted_pairs = sorted_pairs[:limit]

    recommendations: list[dict[str, str]] = []
    for rank, pair in enumerate(sorted_pairs, start=1):
        verdict = str(pair.get("guard_verdict", "-"))
        row = {
            header: str(pair.get(header, "-"))
            for header in BIGRAM_RANK_GUARD_RECOMMENDATION_HEADERS
        }
        row["rank"] = str(rank)
        row["recommendation"] = RECOMMENDED_BIGRAM_RANK_GUARD_VERDICTS.get(
            verdict,
            "-",
        )
        recommendations.append(row)
    return recommendations


def paired_bigram_rank_guard_seed_deltas(
    payloads: list[tuple[str, dict[str, Any]]],
    *,
    baseline_guard: float = DEFAULT_BIGRAM_RANK_GUARD_BASELINE,
) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for source, payload in payloads:
        grouped: dict[tuple[str, ...], dict[float, dict[str, Any]]] = {}
        for row in run_rows(payload):
            guard = parse_number_cell(row.get("bigram_rank_guard"))
            if guard is None:
                continue
            key = tuple(
                str(row.get(column, "-"))
                for column in BIGRAM_RANK_GUARD_SEED_GROUP_COLUMNS
            )
            grouped.setdefault(key, {})[guard] = row

        for key, rows_by_guard in sorted(grouped.items()):
            baseline = rows_by_guard.get(baseline_guard)
            if baseline is None:
                continue
            for guard in sorted(value for value in rows_by_guard if value != baseline_guard):
                candidate = rows_by_guard[guard]
                candidate_final_nll = parse_number_cell(candidate.get("final_nll"))
                baseline_final_nll = parse_number_cell(baseline.get("final_nll"))
                candidate_bigram = parse_number_cell(candidate.get("final_vs_bigram"))
                baseline_bigram = parse_number_cell(baseline.get("final_vs_bigram"))
                candidate_rank_debt = preferred_number_cell(
                    candidate,
                    "final_bigram_rank_debt",
                )
                baseline_rank_debt = preferred_number_cell(
                    baseline,
                    "final_bigram_rank_debt",
                )
                candidate_rank_lift = preferred_number_cell(
                    candidate,
                    "final_bigram_rank_lift",
                )
                baseline_rank_lift = preferred_number_cell(
                    baseline,
                    "final_bigram_rank_lift",
                )
                candidate_top5 = preferred_number_cell(
                    candidate,
                    "final_top5_bigram_overlap",
                )
                baseline_top5 = preferred_number_cell(
                    baseline,
                    "final_top5_bigram_overlap",
                )
                final_nll_delta = (
                    candidate_final_nll - baseline_final_nll
                    if candidate_final_nll is not None
                    and baseline_final_nll is not None
                    else None
                )
                bigram_delta = (
                    candidate_bigram - baseline_bigram
                    if candidate_bigram is not None and baseline_bigram is not None
                    else None
                )
                rank_debt_delta = (
                    candidate_rank_debt - baseline_rank_debt
                    if candidate_rank_debt is not None and baseline_rank_debt is not None
                    else None
                )
                rank_lift_delta = (
                    candidate_rank_lift - baseline_rank_lift
                    if candidate_rank_lift is not None and baseline_rank_lift is not None
                    else None
                )
                top5_delta = (
                    candidate_top5 - baseline_top5
                    if candidate_top5 is not None and baseline_top5 is not None
                    else None
                )
                nll_status = classify_lower_is_better_delta(final_nll_delta)
                bigram_gap_status = classify_lower_is_better_delta(bigram_delta)
                rank_debt_status = classify_lower_is_better_delta(rank_debt_delta)
                bigram_rank_status = classify_higher_is_better_delta(rank_lift_delta)
                top5_bigram_status = classify_higher_is_better_delta(top5_delta)
                rank_status = combine_quality_status(
                    rank_debt_status,
                    bigram_rank_status,
                    top5_bigram_status,
                )
                pair = {
                    "source": source,
                    **dict(
                        zip(
                            BIGRAM_RANK_GUARD_SEED_GROUP_COLUMNS,
                            key,
                            strict=True,
                        )
                    ),
                    "candidate_bigram_rank_guard": fmt_guard_value(guard),
                    "baseline_bigram_rank_guard": fmt_guard_value(baseline_guard),
                    "candidate_final_nll": str(candidate.get("final_nll", "-")),
                    "baseline_final_nll": str(baseline.get("final_nll", "-")),
                    "final_nll_delta": fmt_delta(final_nll_delta),
                    "candidate_final_vs_bigram": str(
                        candidate.get("final_vs_bigram", "-")
                    ),
                    "baseline_final_vs_bigram": str(
                        baseline.get("final_vs_bigram", "-")
                    ),
                    "final_vs_bigram_delta": fmt_delta(bigram_delta),
                    "candidate_bigram_rank_debt": str(
                        candidate.get("final_bigram_rank_debt", "-")
                    ),
                    "baseline_bigram_rank_debt": str(
                        baseline.get("final_bigram_rank_debt", "-")
                    ),
                    "bigram_rank_debt_delta": fmt_delta(rank_debt_delta),
                    "candidate_bigram_rank_lift": str(
                        candidate.get("final_bigram_rank_lift", "-")
                    ),
                    "baseline_bigram_rank_lift": str(
                        baseline.get("final_bigram_rank_lift", "-")
                    ),
                    "bigram_rank_lift_delta": fmt_delta(rank_lift_delta),
                    "candidate_top5_bigram_overlap": str(
                        candidate.get("final_top5_bigram_overlap", "-")
                    ),
                    "baseline_top5_bigram_overlap": str(
                        baseline.get("final_top5_bigram_overlap", "-")
                    ),
                    "top5_bigram_overlap_delta_pp": fmt_delta(top5_delta),
                    "nll_status": nll_status,
                    "bigram_gap_status": bigram_gap_status,
                    "rank_debt_status": rank_debt_status,
                    "bigram_rank_status": bigram_rank_status,
                    "top5_bigram_status": top5_bigram_status,
                    "guard_verdict": bigram_rank_guard_verdict(
                        nll_status=nll_status,
                        bigram_gap_status=bigram_gap_status,
                        bigram_logprob_status="missing",
                        rank_debt_status=rank_debt_status,
                        bigram_rank_status=bigram_rank_status,
                        top5_bigram_status=top5_bigram_status,
                    ),
                    "rank_status": rank_status,
                }
                pairs.append(pair)
    return pairs


def mean_number_cells(rows: list[dict[str, str]], key: str) -> float | None:
    values = [
        value
        for row in rows
        if (value := parse_number_cell(row.get(key))) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def min_number_cell(rows: list[dict[str, str]], key: str) -> float | None:
    values = [
        value
        for row in rows
        if (value := parse_number_cell(row.get(key))) is not None
    ]
    if not values:
        return None
    return min(values)


def max_number_cell(rows: list[dict[str, str]], key: str) -> float | None:
    values = [
        value
        for row in rows
        if (value := parse_number_cell(row.get(key))) is not None
    ]
    if not values:
        return None
    return max(values)


def rank_guard_seed_stability_verdict(
    *,
    improved: int,
    neutral: int,
    regressed: int,
    missing: int,
) -> str:
    observed = improved + neutral + regressed
    if observed == 0:
        return "rank_guard_seed_inconclusive" if missing else "rank_guard_seed_empty"
    if improved and regressed:
        return "rank_guard_seed_mixed"
    if regressed:
        return "rank_guard_seed_regressed"
    if improved and neutral:
        return "rank_guard_seed_improved_or_neutral"
    if improved:
        return "rank_guard_seed_stably_improved"
    return "rank_guard_seed_neutral"


def bigram_rank_guard_stability_rows(
    seed_deltas: list[dict[str, str]],
) -> list[dict[str, str]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in seed_deltas:
        key = tuple(
            str(row.get(column, "-"))
            for column in ["source", *BIGRAM_RANK_GUARD_STABILITY_GROUP_COLUMNS]
        )
        grouped.setdefault(key, []).append(row)

    rows: list[dict[str, str]] = []
    for key, group_rows in sorted(grouped.items()):
        statuses = [str(row.get("rank_status", "missing")) for row in group_rows]
        improved = statuses.count("improved")
        neutral = statuses.count("neutral")
        regressed = statuses.count("regressed")
        missing = sum(1 for status in statuses if status == "missing")
        source, *group_values = key
        rows.append(
            {
                "source": source,
                **dict(
                    zip(
                        BIGRAM_RANK_GUARD_STABILITY_GROUP_COLUMNS,
                        group_values,
                        strict=True,
                    )
                ),
                "seed_pairs": str(len(group_rows)),
                "rank_improved_seeds": str(improved),
                "rank_neutral_seeds": str(neutral),
                "rank_regressed_seeds": str(regressed),
                "rank_missing_seeds": str(missing),
                "mean_bigram_rank_debt_delta": fmt_delta(
                    mean_number_cells(group_rows, "bigram_rank_debt_delta")
                ),
                "min_bigram_rank_debt_delta": fmt_delta(
                    min_number_cell(group_rows, "bigram_rank_debt_delta")
                ),
                "max_bigram_rank_debt_delta": fmt_delta(
                    max_number_cell(group_rows, "bigram_rank_debt_delta")
                ),
                "mean_bigram_rank_lift_delta": fmt_delta(
                    mean_number_cells(group_rows, "bigram_rank_lift_delta")
                ),
                "mean_final_nll_delta": fmt_delta(
                    mean_number_cells(group_rows, "final_nll_delta")
                ),
                "mean_final_vs_bigram_delta": fmt_delta(
                    mean_number_cells(group_rows, "final_vs_bigram_delta")
                ),
                "mean_top5_bigram_overlap_delta_pp": fmt_delta(
                    mean_number_cells(group_rows, "top5_bigram_overlap_delta_pp")
                ),
                "stability_verdict": rank_guard_seed_stability_verdict(
                    improved=improved,
                    neutral=neutral,
                    regressed=regressed,
                    missing=missing,
                ),
            }
        )
    return rows


def bigram_rank_band_verdict(
    *,
    nll_status: str,
    bigram_gap_status: str,
    bigram_logprob_status: str,
    rank_debt_status: str,
    bigram_rank_status: str,
    top5_bigram_status: str,
) -> str:
    quality_statuses = [nll_status, bigram_gap_status, bigram_logprob_status]
    alignment_statuses = [rank_debt_status, bigram_rank_status, top5_bigram_status]
    quality_regressed = any(status == "regressed" for status in quality_statuses)
    quality_improved = any(status == "improved" for status in quality_statuses)
    alignment_regressed = any(
        status == "regressed" for status in alignment_statuses
    )
    alignment_improved = any(status == "improved" for status in alignment_statuses)
    quality_missing = all(status == "missing" for status in quality_statuses)
    alignment_missing = all(status == "missing" for status in alignment_statuses)

    if quality_regressed:
        return "rank_band_quality_regressed"
    if alignment_improved and alignment_regressed:
        if quality_improved:
            return "rank_band_quality_improved_alignment_mixed"
        return "rank_band_alignment_mixed"
    if alignment_regressed:
        if quality_improved:
            return "rank_band_quality_improved_alignment_regressed"
        return "rank_band_alignment_regressed"
    if quality_improved and alignment_improved:
        return "rank_band_quality_and_alignment_improved"
    if alignment_improved:
        return "rank_band_alignment_improved"
    if quality_improved:
        return "rank_band_quality_improved"
    if quality_missing and alignment_missing:
        return "rank_band_inconclusive"
    return "rank_band_neutral"


def paired_bigram_rank_band_deltas(
    payloads: list[tuple[str, dict[str, Any]]],
    *,
    baseline_band: float = DEFAULT_BIGRAM_RANK_BAND_BASELINE,
) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for source, payload in payloads:
        grouped: dict[tuple[str, ...], dict[float, dict[str, Any]]] = {}
        for row in source_rows(payload):
            band = parse_number_cell(row.get("bigram_rank_band"))
            if band is None:
                continue
            key = tuple(
                str(row.get(column, "-")) for column in BIGRAM_RANK_BAND_GROUP_COLUMNS
            )
            grouped.setdefault(key, {})[band] = row

        for key, rows_by_band in sorted(grouped.items()):
            baseline = rows_by_band.get(baseline_band)
            if baseline is None:
                continue
            for band in sorted(value for value in rows_by_band if value != baseline_band):
                candidate = rows_by_band[band]
                candidate_final_nll = parse_number_cell(candidate.get("final_nll_mean"))
                baseline_final_nll = parse_number_cell(baseline.get("final_nll_mean"))
                candidate_bigram = parse_number_cell(
                    candidate.get("final_vs_bigram_mean")
                )
                baseline_bigram = parse_number_cell(baseline.get("final_vs_bigram_mean"))
                candidate_logprob_lift = parse_number_cell(
                    candidate.get("final_bigram_logprob_lift_mean")
                )
                baseline_logprob_lift = parse_number_cell(
                    baseline.get("final_bigram_logprob_lift_mean")
                )
                candidate_rank_debt = preferred_number_cell(
                    candidate,
                    "final_bigram_rank_debt_mean",
                )
                baseline_rank_debt = preferred_number_cell(
                    baseline,
                    "final_bigram_rank_debt_mean",
                )
                candidate_rank_lift = preferred_number_cell(
                    candidate,
                    "final_bigram_rank_lift_mean",
                )
                baseline_rank_lift = preferred_number_cell(
                    baseline,
                    "final_bigram_rank_lift_mean",
                )
                candidate_top5 = preferred_number_cell(
                    candidate,
                    "final_top5_bigram_overlap_mean",
                )
                baseline_top5 = preferred_number_cell(
                    baseline,
                    "final_top5_bigram_overlap_mean",
                )
                final_nll_delta = (
                    candidate_final_nll - baseline_final_nll
                    if candidate_final_nll is not None
                    and baseline_final_nll is not None
                    else None
                )
                bigram_delta = (
                    candidate_bigram - baseline_bigram
                    if candidate_bigram is not None and baseline_bigram is not None
                    else None
                )
                logprob_lift_delta = (
                    candidate_logprob_lift - baseline_logprob_lift
                    if candidate_logprob_lift is not None
                    and baseline_logprob_lift is not None
                    else None
                )
                rank_debt_delta = (
                    candidate_rank_debt - baseline_rank_debt
                    if candidate_rank_debt is not None and baseline_rank_debt is not None
                    else None
                )
                rank_lift_delta = (
                    candidate_rank_lift - baseline_rank_lift
                    if candidate_rank_lift is not None and baseline_rank_lift is not None
                    else None
                )
                top5_delta = (
                    candidate_top5 - baseline_top5
                    if candidate_top5 is not None and baseline_top5 is not None
                    else None
                )
                nll_status = classify_lower_is_better_delta(final_nll_delta)
                bigram_gap_status = classify_lower_is_better_delta(bigram_delta)
                bigram_logprob_status = classify_higher_is_better_delta(
                    logprob_lift_delta
                )
                rank_debt_status = classify_lower_is_better_delta(rank_debt_delta)
                bigram_rank_status = classify_higher_is_better_delta(rank_lift_delta)
                top5_bigram_status = classify_higher_is_better_delta(top5_delta)
                quality_status = combine_quality_status(
                    nll_status,
                    bigram_gap_status,
                    bigram_logprob_status,
                )
                alignment_status = combine_quality_status(
                    rank_debt_status,
                    bigram_rank_status,
                    top5_bigram_status,
                )
                pair = {
                    "source": source,
                    **dict(zip(BIGRAM_RANK_BAND_GROUP_COLUMNS, key, strict=True)),
                    "candidate_bigram_rank_band": fmt_guard_value(band),
                    "baseline_bigram_rank_band": fmt_guard_value(baseline_band),
                    "candidate_runs": str(candidate.get("runs", "-")),
                    "baseline_runs": str(baseline.get("runs", "-")),
                    "candidate_final_nll": str(candidate.get("final_nll_mean", "-")),
                    "baseline_final_nll": str(baseline.get("final_nll_mean", "-")),
                    "final_nll_delta": fmt_delta(final_nll_delta),
                    "candidate_final_vs_bigram": str(
                        candidate.get("final_vs_bigram_mean", "-")
                    ),
                    "baseline_final_vs_bigram": str(
                        baseline.get("final_vs_bigram_mean", "-")
                    ),
                    "final_vs_bigram_delta": fmt_delta(bigram_delta),
                    "candidate_bigram_logprob_lift": str(
                        candidate.get("final_bigram_logprob_lift_mean", "-")
                    ),
                    "baseline_bigram_logprob_lift": str(
                        baseline.get("final_bigram_logprob_lift_mean", "-")
                    ),
                    "bigram_logprob_lift_delta": fmt_delta(logprob_lift_delta),
                    "candidate_bigram_rank_debt": str(
                        candidate.get("final_bigram_rank_debt_mean", "-")
                    ),
                    "baseline_bigram_rank_debt": str(
                        baseline.get("final_bigram_rank_debt_mean", "-")
                    ),
                    "bigram_rank_debt_delta": fmt_delta(rank_debt_delta),
                    "candidate_bigram_rank_lift": str(
                        candidate.get("final_bigram_rank_lift_mean", "-")
                    ),
                    "baseline_bigram_rank_lift": str(
                        baseline.get("final_bigram_rank_lift_mean", "-")
                    ),
                    "bigram_rank_lift_delta": fmt_delta(rank_lift_delta),
                    "candidate_top5_bigram_overlap": str(
                        candidate.get("final_top5_bigram_overlap_mean", "-")
                    ),
                    "baseline_top5_bigram_overlap": str(
                        baseline.get("final_top5_bigram_overlap_mean", "-")
                    ),
                    "top5_bigram_overlap_delta_pp": fmt_delta(top5_delta),
                    "nll_status": nll_status,
                    "bigram_gap_status": bigram_gap_status,
                    "bigram_logprob_status": bigram_logprob_status,
                    "rank_debt_status": rank_debt_status,
                    "bigram_rank_status": bigram_rank_status,
                    "top5_bigram_status": top5_bigram_status,
                    "band_verdict": bigram_rank_band_verdict(
                        nll_status=nll_status,
                        bigram_gap_status=bigram_gap_status,
                        bigram_logprob_status=bigram_logprob_status,
                        rank_debt_status=rank_debt_status,
                        bigram_rank_status=bigram_rank_status,
                        top5_bigram_status=top5_bigram_status,
                    ),
                    "quality_status": quality_status,
                    "alignment_status": alignment_status,
                    "candidate_route_status": route_status(candidate),
                    "baseline_route_status": route_status(baseline),
                }
                pairs.append(pair)
    return pairs


def paired_bigram_rank_band_recommendations(
    pairs: list[dict[str, str]],
    *,
    limit: int,
) -> list[dict[str, str]]:
    candidates = [
        pair
        for pair in pairs
        if pair.get("band_verdict") in RECOMMENDED_BIGRAM_RANK_BAND_VERDICTS
    ]
    verdict_rank = {
        "rank_band_quality_and_alignment_improved": 0,
        "rank_band_alignment_improved": 1,
        "rank_band_quality_improved": 2,
        "rank_band_quality_improved_alignment_mixed": 3,
    }
    sorted_pairs = sorted(
        candidates,
        key=lambda pair: (
            verdict_rank.get(str(pair.get("band_verdict")), 99),
            pair_metric_for_sort(pair, "bigram_rank_debt_delta"),
            pair_metric_desc_for_sort(pair, "bigram_rank_lift_delta"),
            pair_metric_for_sort(pair, "final_nll_delta"),
            pair_metric_for_sort(pair, "final_vs_bigram_delta"),
            pair_metric_desc_for_sort(pair, "top5_bigram_overlap_delta_pp"),
            pair_metric_for_sort(pair, "candidate_final_nll"),
            pair_metric_for_sort(pair, "candidate_bigram_rank_band"),
            str(pair.get("source", "")),
        ),
    )
    if limit > 0:
        sorted_pairs = sorted_pairs[:limit]

    recommendations: list[dict[str, str]] = []
    for rank, pair in enumerate(sorted_pairs, start=1):
        verdict = str(pair.get("band_verdict", "-"))
        row = {
            header: str(pair.get(header, "-"))
            for header in BIGRAM_RANK_BAND_RECOMMENDATION_HEADERS
        }
        row["rank"] = str(rank)
        row["recommendation"] = RECOMMENDED_BIGRAM_RANK_BAND_VERDICTS.get(
            verdict,
            "-",
        )
        recommendations.append(row)
    return recommendations


def paired_bigram_rank_band_seed_deltas(
    payloads: list[tuple[str, dict[str, Any]]],
    *,
    baseline_band: float = DEFAULT_BIGRAM_RANK_BAND_BASELINE,
) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for source, payload in payloads:
        grouped: dict[tuple[str, ...], dict[float, dict[str, Any]]] = {}
        for row in run_rows(payload):
            band = parse_number_cell(row.get("bigram_rank_band"))
            if band is None:
                continue
            key = tuple(
                str(row.get(column, "-"))
                for column in BIGRAM_RANK_BAND_SEED_GROUP_COLUMNS
            )
            grouped.setdefault(key, {})[band] = row

        for key, rows_by_band in sorted(grouped.items()):
            baseline = rows_by_band.get(baseline_band)
            if baseline is None:
                continue
            for band in sorted(value for value in rows_by_band if value != baseline_band):
                candidate = rows_by_band[band]
                candidate_final_nll = parse_number_cell(candidate.get("final_nll"))
                baseline_final_nll = parse_number_cell(baseline.get("final_nll"))
                candidate_bigram = parse_number_cell(candidate.get("final_vs_bigram"))
                baseline_bigram = parse_number_cell(baseline.get("final_vs_bigram"))
                candidate_rank_debt = preferred_number_cell(
                    candidate,
                    "final_bigram_rank_debt",
                )
                baseline_rank_debt = preferred_number_cell(
                    baseline,
                    "final_bigram_rank_debt",
                )
                candidate_rank_lift = preferred_number_cell(
                    candidate,
                    "final_bigram_rank_lift",
                )
                baseline_rank_lift = preferred_number_cell(
                    baseline,
                    "final_bigram_rank_lift",
                )
                candidate_top5 = preferred_number_cell(
                    candidate,
                    "final_top5_bigram_overlap",
                )
                baseline_top5 = preferred_number_cell(
                    baseline,
                    "final_top5_bigram_overlap",
                )
                final_nll_delta = (
                    candidate_final_nll - baseline_final_nll
                    if candidate_final_nll is not None
                    and baseline_final_nll is not None
                    else None
                )
                bigram_delta = (
                    candidate_bigram - baseline_bigram
                    if candidate_bigram is not None and baseline_bigram is not None
                    else None
                )
                rank_debt_delta = (
                    candidate_rank_debt - baseline_rank_debt
                    if candidate_rank_debt is not None and baseline_rank_debt is not None
                    else None
                )
                rank_lift_delta = (
                    candidate_rank_lift - baseline_rank_lift
                    if candidate_rank_lift is not None and baseline_rank_lift is not None
                    else None
                )
                top5_delta = (
                    candidate_top5 - baseline_top5
                    if candidate_top5 is not None and baseline_top5 is not None
                    else None
                )
                nll_status = classify_lower_is_better_delta(final_nll_delta)
                bigram_gap_status = classify_lower_is_better_delta(bigram_delta)
                rank_debt_status = classify_lower_is_better_delta(rank_debt_delta)
                bigram_rank_status = classify_higher_is_better_delta(rank_lift_delta)
                top5_bigram_status = classify_higher_is_better_delta(top5_delta)
                alignment_status = combine_quality_status(
                    rank_debt_status,
                    bigram_rank_status,
                    top5_bigram_status,
                )
                pair = {
                    "source": source,
                    **dict(
                        zip(
                            BIGRAM_RANK_BAND_SEED_GROUP_COLUMNS,
                            key,
                            strict=True,
                        )
                    ),
                    "candidate_bigram_rank_band": fmt_guard_value(band),
                    "baseline_bigram_rank_band": fmt_guard_value(baseline_band),
                    "candidate_final_nll": str(candidate.get("final_nll", "-")),
                    "baseline_final_nll": str(baseline.get("final_nll", "-")),
                    "final_nll_delta": fmt_delta(final_nll_delta),
                    "candidate_final_vs_bigram": str(
                        candidate.get("final_vs_bigram", "-")
                    ),
                    "baseline_final_vs_bigram": str(
                        baseline.get("final_vs_bigram", "-")
                    ),
                    "final_vs_bigram_delta": fmt_delta(bigram_delta),
                    "candidate_bigram_rank_debt": str(
                        candidate.get("final_bigram_rank_debt", "-")
                    ),
                    "baseline_bigram_rank_debt": str(
                        baseline.get("final_bigram_rank_debt", "-")
                    ),
                    "bigram_rank_debt_delta": fmt_delta(rank_debt_delta),
                    "candidate_bigram_rank_lift": str(
                        candidate.get("final_bigram_rank_lift", "-")
                    ),
                    "baseline_bigram_rank_lift": str(
                        baseline.get("final_bigram_rank_lift", "-")
                    ),
                    "bigram_rank_lift_delta": fmt_delta(rank_lift_delta),
                    "candidate_top5_bigram_overlap": str(
                        candidate.get("final_top5_bigram_overlap", "-")
                    ),
                    "baseline_top5_bigram_overlap": str(
                        baseline.get("final_top5_bigram_overlap", "-")
                    ),
                    "top5_bigram_overlap_delta_pp": fmt_delta(top5_delta),
                    "nll_status": nll_status,
                    "bigram_gap_status": bigram_gap_status,
                    "rank_debt_status": rank_debt_status,
                    "bigram_rank_status": bigram_rank_status,
                    "top5_bigram_status": top5_bigram_status,
                    "band_verdict": bigram_rank_band_verdict(
                        nll_status=nll_status,
                        bigram_gap_status=bigram_gap_status,
                        bigram_logprob_status="missing",
                        rank_debt_status=rank_debt_status,
                        bigram_rank_status=bigram_rank_status,
                        top5_bigram_status=top5_bigram_status,
                    ),
                    "alignment_status": alignment_status,
                }
                pairs.append(pair)
    return pairs


def rank_band_seed_stability_verdict(
    *,
    improved: int,
    neutral: int,
    regressed: int,
    missing: int,
) -> str:
    observed = improved + neutral + regressed
    if observed == 0:
        return "rank_band_seed_inconclusive" if missing else "rank_band_seed_empty"
    if improved and regressed:
        return "rank_band_seed_mixed"
    if regressed:
        return "rank_band_seed_regressed"
    if improved and neutral:
        return "rank_band_seed_improved_or_neutral"
    if improved:
        return "rank_band_seed_stably_improved"
    return "rank_band_seed_neutral"


def bigram_rank_band_stability_rows(
    seed_deltas: list[dict[str, str]],
) -> list[dict[str, str]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in seed_deltas:
        key = tuple(
            str(row.get(column, "-"))
            for column in ["source", *BIGRAM_RANK_BAND_STABILITY_GROUP_COLUMNS]
        )
        grouped.setdefault(key, []).append(row)

    rows: list[dict[str, str]] = []
    for key, group_rows in sorted(grouped.items()):
        statuses = [str(row.get("alignment_status", "missing")) for row in group_rows]
        improved = statuses.count("improved")
        neutral = statuses.count("neutral")
        regressed = statuses.count("regressed")
        missing = sum(1 for status in statuses if status == "missing")
        source, *group_values = key
        rows.append(
            {
                "source": source,
                **dict(
                    zip(
                        BIGRAM_RANK_BAND_STABILITY_GROUP_COLUMNS,
                        group_values,
                        strict=True,
                    )
                ),
                "seed_pairs": str(len(group_rows)),
                "alignment_improved_seeds": str(improved),
                "alignment_neutral_seeds": str(neutral),
                "alignment_regressed_seeds": str(regressed),
                "alignment_missing_seeds": str(missing),
                "mean_bigram_rank_debt_delta": fmt_delta(
                    mean_number_cells(group_rows, "bigram_rank_debt_delta")
                ),
                "min_bigram_rank_debt_delta": fmt_delta(
                    min_number_cell(group_rows, "bigram_rank_debt_delta")
                ),
                "max_bigram_rank_debt_delta": fmt_delta(
                    max_number_cell(group_rows, "bigram_rank_debt_delta")
                ),
                "mean_bigram_rank_lift_delta": fmt_delta(
                    mean_number_cells(group_rows, "bigram_rank_lift_delta")
                ),
                "mean_final_nll_delta": fmt_delta(
                    mean_number_cells(group_rows, "final_nll_delta")
                ),
                "mean_final_vs_bigram_delta": fmt_delta(
                    mean_number_cells(group_rows, "final_vs_bigram_delta")
                ),
                "mean_top5_bigram_overlap_delta_pp": fmt_delta(
                    mean_number_cells(group_rows, "top5_bigram_overlap_delta_pp")
                ),
                "stability_verdict": rank_band_seed_stability_verdict(
                    improved=improved,
                    neutral=neutral,
                    regressed=regressed,
                    missing=missing,
                ),
            }
        )
    return rows


def bigram_rank_min_verdict(
    *,
    nll_status: str,
    bigram_gap_status: str,
    bigram_logprob_status: str,
    rank_debt_status: str,
    bigram_rank_status: str,
    top5_bigram_status: str,
    rank_cov_guarded_status: str,
    rank_cov_zero_status: str,
) -> str:
    quality_statuses = [nll_status, bigram_gap_status, bigram_logprob_status]
    alignment_statuses = [
        rank_debt_status,
        bigram_rank_status,
        top5_bigram_status,
        rank_cov_guarded_status,
        rank_cov_zero_status,
    ]
    quality_regressed = any(status == "regressed" for status in quality_statuses)
    quality_improved = any(status == "improved" for status in quality_statuses)
    alignment_regressed = any(
        status == "regressed" for status in alignment_statuses
    )
    alignment_improved = any(status == "improved" for status in alignment_statuses)
    quality_missing = all(status == "missing" for status in quality_statuses)
    alignment_missing = all(status == "missing" for status in alignment_statuses)

    if quality_regressed:
        return "rank_min_quality_regressed"
    if alignment_improved and alignment_regressed:
        if quality_improved:
            return "rank_min_quality_improved_alignment_mixed"
        return "rank_min_alignment_mixed"
    if alignment_regressed:
        if quality_improved:
            return "rank_min_quality_improved_alignment_regressed"
        return "rank_min_alignment_regressed"
    if quality_improved and alignment_improved:
        return "rank_min_quality_and_alignment_improved"
    if alignment_improved:
        return "rank_min_alignment_improved"
    if quality_improved:
        return "rank_min_quality_improved"
    if quality_missing and alignment_missing:
        return "rank_min_inconclusive"
    return "rank_min_neutral"


def paired_bigram_rank_min_deltas(
    payloads: list[tuple[str, dict[str, Any]]],
    *,
    baseline_min: float = DEFAULT_BIGRAM_RANK_MIN_BASELINE,
) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for source, payload in payloads:
        grouped: dict[tuple[str, ...], dict[float, dict[str, Any]]] = {}
        for row in source_rows(payload):
            rank_min = parse_number_cell(row.get("bigram_rank_min"))
            if rank_min is None:
                continue
            key = tuple(
                str(row.get(column, "-")) for column in BIGRAM_RANK_MIN_GROUP_COLUMNS
            )
            grouped.setdefault(key, {})[rank_min] = row

        for key, rows_by_min in sorted(grouped.items()):
            baseline = rows_by_min.get(baseline_min)
            if baseline is None:
                continue
            for rank_min in sorted(
                value for value in rows_by_min if value != baseline_min
            ):
                candidate = rows_by_min[rank_min]
                candidate_rank_cov_guarded = parse_number_cell(
                    candidate.get("rank_cov_guarded_mean")
                )
                baseline_rank_cov_guarded = parse_number_cell(
                    baseline.get("rank_cov_guarded_mean")
                )
                candidate_rank_cov_zero = parse_number_cell(
                    candidate.get("rank_cov_zero_ratio_mean")
                )
                baseline_rank_cov_zero = parse_number_cell(
                    baseline.get("rank_cov_zero_ratio_mean")
                )
                candidate_rank_cov_filled = parse_number_cell(
                    candidate.get("rank_cov_filled_mean")
                )
                baseline_rank_cov_filled = parse_number_cell(
                    baseline.get("rank_cov_filled_mean")
                )
                candidate_final_nll = parse_number_cell(candidate.get("final_nll_mean"))
                baseline_final_nll = parse_number_cell(baseline.get("final_nll_mean"))
                candidate_bigram = parse_number_cell(
                    candidate.get("final_vs_bigram_mean")
                )
                baseline_bigram = parse_number_cell(baseline.get("final_vs_bigram_mean"))
                candidate_logprob_lift = parse_number_cell(
                    candidate.get("final_bigram_logprob_lift_mean")
                )
                baseline_logprob_lift = parse_number_cell(
                    baseline.get("final_bigram_logprob_lift_mean")
                )
                candidate_rank_debt = preferred_number_cell(
                    candidate,
                    "final_bigram_rank_debt_mean",
                )
                baseline_rank_debt = preferred_number_cell(
                    baseline,
                    "final_bigram_rank_debt_mean",
                )
                candidate_rank_lift = preferred_number_cell(
                    candidate,
                    "final_bigram_rank_lift_mean",
                )
                baseline_rank_lift = preferred_number_cell(
                    baseline,
                    "final_bigram_rank_lift_mean",
                )
                candidate_top5 = preferred_number_cell(
                    candidate,
                    "final_top5_bigram_overlap_mean",
                )
                baseline_top5 = preferred_number_cell(
                    baseline,
                    "final_top5_bigram_overlap_mean",
                )
                rank_cov_guarded_delta = (
                    candidate_rank_cov_guarded - baseline_rank_cov_guarded
                    if candidate_rank_cov_guarded is not None
                    and baseline_rank_cov_guarded is not None
                    else None
                )
                rank_cov_zero_delta = (
                    candidate_rank_cov_zero - baseline_rank_cov_zero
                    if candidate_rank_cov_zero is not None
                    and baseline_rank_cov_zero is not None
                    else None
                )
                rank_cov_filled_delta = (
                    candidate_rank_cov_filled - baseline_rank_cov_filled
                    if candidate_rank_cov_filled is not None
                    and baseline_rank_cov_filled is not None
                    else None
                )
                final_nll_delta = (
                    candidate_final_nll - baseline_final_nll
                    if candidate_final_nll is not None
                    and baseline_final_nll is not None
                    else None
                )
                bigram_delta = (
                    candidate_bigram - baseline_bigram
                    if candidate_bigram is not None and baseline_bigram is not None
                    else None
                )
                logprob_lift_delta = (
                    candidate_logprob_lift - baseline_logprob_lift
                    if candidate_logprob_lift is not None
                    and baseline_logprob_lift is not None
                    else None
                )
                rank_debt_delta = (
                    candidate_rank_debt - baseline_rank_debt
                    if candidate_rank_debt is not None and baseline_rank_debt is not None
                    else None
                )
                rank_lift_delta = (
                    candidate_rank_lift - baseline_rank_lift
                    if candidate_rank_lift is not None and baseline_rank_lift is not None
                    else None
                )
                top5_delta = (
                    candidate_top5 - baseline_top5
                    if candidate_top5 is not None and baseline_top5 is not None
                    else None
                )
                nll_status = classify_lower_is_better_delta(final_nll_delta)
                bigram_gap_status = classify_lower_is_better_delta(bigram_delta)
                bigram_logprob_status = classify_higher_is_better_delta(
                    logprob_lift_delta
                )
                rank_debt_status = classify_lower_is_better_delta(rank_debt_delta)
                bigram_rank_status = classify_higher_is_better_delta(rank_lift_delta)
                top5_bigram_status = classify_higher_is_better_delta(top5_delta)
                rank_cov_guarded_status = classify_higher_is_better_delta(
                    rank_cov_guarded_delta
                )
                rank_cov_zero_status = classify_lower_is_better_delta(
                    rank_cov_zero_delta
                )
                quality_status = combine_quality_status(
                    nll_status,
                    bigram_gap_status,
                    bigram_logprob_status,
                )
                alignment_status = combine_quality_status(
                    rank_debt_status,
                    bigram_rank_status,
                    top5_bigram_status,
                    rank_cov_guarded_status,
                    rank_cov_zero_status,
                )
                pair = {
                    "source": source,
                    **dict(zip(BIGRAM_RANK_MIN_GROUP_COLUMNS, key, strict=True)),
                    "candidate_bigram_rank_min": fmt_rank_min_value(rank_min),
                    "baseline_bigram_rank_min": fmt_rank_min_value(baseline_min),
                    "candidate_runs": str(candidate.get("runs", "-")),
                    "baseline_runs": str(baseline.get("runs", "-")),
                    "candidate_rank_cov_guarded": str(
                        candidate.get("rank_cov_guarded_mean", "-")
                    ),
                    "baseline_rank_cov_guarded": str(
                        baseline.get("rank_cov_guarded_mean", "-")
                    ),
                    "rank_cov_guarded_delta": fmt_delta(rank_cov_guarded_delta),
                    "candidate_rank_cov_zero_ratio": str(
                        candidate.get("rank_cov_zero_ratio_mean", "-")
                    ),
                    "baseline_rank_cov_zero_ratio": str(
                        baseline.get("rank_cov_zero_ratio_mean", "-")
                    ),
                    "rank_cov_zero_ratio_delta": fmt_delta(rank_cov_zero_delta),
                    "candidate_rank_cov_filled": str(
                        candidate.get("rank_cov_filled_mean", "-")
                    ),
                    "baseline_rank_cov_filled": str(
                        baseline.get("rank_cov_filled_mean", "-")
                    ),
                    "rank_cov_filled_delta": fmt_delta(rank_cov_filled_delta),
                    "candidate_final_nll": str(candidate.get("final_nll_mean", "-")),
                    "baseline_final_nll": str(baseline.get("final_nll_mean", "-")),
                    "final_nll_delta": fmt_delta(final_nll_delta),
                    "candidate_final_vs_bigram": str(
                        candidate.get("final_vs_bigram_mean", "-")
                    ),
                    "baseline_final_vs_bigram": str(
                        baseline.get("final_vs_bigram_mean", "-")
                    ),
                    "final_vs_bigram_delta": fmt_delta(bigram_delta),
                    "candidate_bigram_logprob_lift": str(
                        candidate.get("final_bigram_logprob_lift_mean", "-")
                    ),
                    "baseline_bigram_logprob_lift": str(
                        baseline.get("final_bigram_logprob_lift_mean", "-")
                    ),
                    "bigram_logprob_lift_delta": fmt_delta(logprob_lift_delta),
                    "candidate_bigram_rank_debt": str(
                        candidate.get("final_bigram_rank_debt_mean", "-")
                    ),
                    "baseline_bigram_rank_debt": str(
                        baseline.get("final_bigram_rank_debt_mean", "-")
                    ),
                    "bigram_rank_debt_delta": fmt_delta(rank_debt_delta),
                    "candidate_bigram_rank_lift": str(
                        candidate.get("final_bigram_rank_lift_mean", "-")
                    ),
                    "baseline_bigram_rank_lift": str(
                        baseline.get("final_bigram_rank_lift_mean", "-")
                    ),
                    "bigram_rank_lift_delta": fmt_delta(rank_lift_delta),
                    "candidate_top5_bigram_overlap": str(
                        candidate.get("final_top5_bigram_overlap_mean", "-")
                    ),
                    "baseline_top5_bigram_overlap": str(
                        baseline.get("final_top5_bigram_overlap_mean", "-")
                    ),
                    "top5_bigram_overlap_delta_pp": fmt_delta(top5_delta),
                    "nll_status": nll_status,
                    "bigram_gap_status": bigram_gap_status,
                    "bigram_logprob_status": bigram_logprob_status,
                    "rank_debt_status": rank_debt_status,
                    "bigram_rank_status": bigram_rank_status,
                    "top5_bigram_status": top5_bigram_status,
                    "rank_cov_guarded_status": rank_cov_guarded_status,
                    "rank_cov_zero_status": rank_cov_zero_status,
                    "min_verdict": bigram_rank_min_verdict(
                        nll_status=nll_status,
                        bigram_gap_status=bigram_gap_status,
                        bigram_logprob_status=bigram_logprob_status,
                        rank_debt_status=rank_debt_status,
                        bigram_rank_status=bigram_rank_status,
                        top5_bigram_status=top5_bigram_status,
                        rank_cov_guarded_status=rank_cov_guarded_status,
                        rank_cov_zero_status=rank_cov_zero_status,
                    ),
                    "quality_status": quality_status,
                    "alignment_status": alignment_status,
                    "candidate_route_status": route_status(candidate),
                    "baseline_route_status": route_status(baseline),
                }
                pairs.append(pair)
    return pairs


def paired_bigram_rank_min_recommendations(
    pairs: list[dict[str, str]],
    *,
    limit: int,
) -> list[dict[str, str]]:
    candidates = [
        pair
        for pair in pairs
        if pair.get("min_verdict") in RECOMMENDED_BIGRAM_RANK_MIN_VERDICTS
    ]
    verdict_rank = {
        "rank_min_quality_and_alignment_improved": 0,
        "rank_min_alignment_improved": 1,
        "rank_min_quality_improved": 2,
        "rank_min_quality_improved_alignment_mixed": 3,
    }
    sorted_pairs = sorted(
        candidates,
        key=lambda pair: (
            verdict_rank.get(str(pair.get("min_verdict")), 99),
            pair_metric_for_sort(pair, "rank_cov_zero_ratio_delta"),
            pair_metric_desc_for_sort(pair, "rank_cov_guarded_delta"),
            pair_metric_for_sort(pair, "bigram_rank_debt_delta"),
            pair_metric_desc_for_sort(pair, "bigram_rank_lift_delta"),
            pair_metric_for_sort(pair, "final_nll_delta"),
            pair_metric_for_sort(pair, "final_vs_bigram_delta"),
            pair_metric_desc_for_sort(pair, "top5_bigram_overlap_delta_pp"),
            pair_metric_for_sort(pair, "candidate_final_nll"),
            pair_metric_for_sort(pair, "candidate_bigram_rank_min"),
            str(pair.get("source", "")),
        ),
    )
    if limit > 0:
        sorted_pairs = sorted_pairs[:limit]

    recommendations: list[dict[str, str]] = []
    for rank, pair in enumerate(sorted_pairs, start=1):
        verdict = str(pair.get("min_verdict", "-"))
        row = {
            header: str(pair.get(header, "-"))
            for header in BIGRAM_RANK_MIN_RECOMMENDATION_HEADERS
        }
        row["rank"] = str(rank)
        row["recommendation"] = RECOMMENDED_BIGRAM_RANK_MIN_VERDICTS.get(
            verdict,
            "-",
        )
        recommendations.append(row)
    return recommendations


def paired_bigram_rank_min_seed_deltas(
    payloads: list[tuple[str, dict[str, Any]]],
    *,
    baseline_min: float = DEFAULT_BIGRAM_RANK_MIN_BASELINE,
) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for source, payload in payloads:
        grouped: dict[tuple[str, ...], dict[float, dict[str, Any]]] = {}
        for row in run_rows(payload):
            rank_min = parse_number_cell(row.get("bigram_rank_min"))
            if rank_min is None:
                continue
            key = tuple(
                str(row.get(column, "-"))
                for column in BIGRAM_RANK_MIN_SEED_GROUP_COLUMNS
            )
            grouped.setdefault(key, {})[rank_min] = row

        for key, rows_by_min in sorted(grouped.items()):
            baseline = rows_by_min.get(baseline_min)
            if baseline is None:
                continue
            for rank_min in sorted(
                value for value in rows_by_min if value != baseline_min
            ):
                candidate = rows_by_min[rank_min]
                candidate_rank_cov_guarded = parse_number_cell(
                    candidate.get("rank_cov_guarded")
                )
                baseline_rank_cov_guarded = parse_number_cell(
                    baseline.get("rank_cov_guarded")
                )
                candidate_rank_cov_zero = parse_number_cell(
                    candidate.get("rank_cov_zero_ratio")
                )
                baseline_rank_cov_zero = parse_number_cell(
                    baseline.get("rank_cov_zero_ratio")
                )
                candidate_rank_cov_filled = parse_number_cell(
                    candidate.get("rank_cov_filled")
                )
                baseline_rank_cov_filled = parse_number_cell(
                    baseline.get("rank_cov_filled")
                )
                candidate_final_nll = parse_number_cell(candidate.get("final_nll"))
                baseline_final_nll = parse_number_cell(baseline.get("final_nll"))
                candidate_bigram = parse_number_cell(candidate.get("final_vs_bigram"))
                baseline_bigram = parse_number_cell(baseline.get("final_vs_bigram"))
                candidate_rank_debt = preferred_number_cell(
                    candidate,
                    "final_bigram_rank_debt",
                )
                baseline_rank_debt = preferred_number_cell(
                    baseline,
                    "final_bigram_rank_debt",
                )
                candidate_rank_lift = preferred_number_cell(
                    candidate,
                    "final_bigram_rank_lift",
                )
                baseline_rank_lift = preferred_number_cell(
                    baseline,
                    "final_bigram_rank_lift",
                )
                candidate_top5 = preferred_number_cell(
                    candidate,
                    "final_top5_bigram_overlap",
                )
                baseline_top5 = preferred_number_cell(
                    baseline,
                    "final_top5_bigram_overlap",
                )
                rank_cov_guarded_delta = (
                    candidate_rank_cov_guarded - baseline_rank_cov_guarded
                    if candidate_rank_cov_guarded is not None
                    and baseline_rank_cov_guarded is not None
                    else None
                )
                rank_cov_zero_delta = (
                    candidate_rank_cov_zero - baseline_rank_cov_zero
                    if candidate_rank_cov_zero is not None
                    and baseline_rank_cov_zero is not None
                    else None
                )
                rank_cov_filled_delta = (
                    candidate_rank_cov_filled - baseline_rank_cov_filled
                    if candidate_rank_cov_filled is not None
                    and baseline_rank_cov_filled is not None
                    else None
                )
                final_nll_delta = (
                    candidate_final_nll - baseline_final_nll
                    if candidate_final_nll is not None
                    and baseline_final_nll is not None
                    else None
                )
                bigram_delta = (
                    candidate_bigram - baseline_bigram
                    if candidate_bigram is not None and baseline_bigram is not None
                    else None
                )
                rank_debt_delta = (
                    candidate_rank_debt - baseline_rank_debt
                    if candidate_rank_debt is not None and baseline_rank_debt is not None
                    else None
                )
                rank_lift_delta = (
                    candidate_rank_lift - baseline_rank_lift
                    if candidate_rank_lift is not None and baseline_rank_lift is not None
                    else None
                )
                top5_delta = (
                    candidate_top5 - baseline_top5
                    if candidate_top5 is not None and baseline_top5 is not None
                    else None
                )
                nll_status = classify_lower_is_better_delta(final_nll_delta)
                bigram_gap_status = classify_lower_is_better_delta(bigram_delta)
                rank_debt_status = classify_lower_is_better_delta(rank_debt_delta)
                bigram_rank_status = classify_higher_is_better_delta(rank_lift_delta)
                top5_bigram_status = classify_higher_is_better_delta(top5_delta)
                rank_cov_guarded_status = classify_higher_is_better_delta(
                    rank_cov_guarded_delta
                )
                rank_cov_zero_status = classify_lower_is_better_delta(
                    rank_cov_zero_delta
                )
                alignment_status = combine_quality_status(
                    rank_debt_status,
                    bigram_rank_status,
                    top5_bigram_status,
                    rank_cov_guarded_status,
                    rank_cov_zero_status,
                )
                pair = {
                    "source": source,
                    **dict(
                        zip(
                            BIGRAM_RANK_MIN_SEED_GROUP_COLUMNS,
                            key,
                            strict=True,
                        )
                    ),
                    "candidate_bigram_rank_min": fmt_rank_min_value(rank_min),
                    "baseline_bigram_rank_min": fmt_rank_min_value(baseline_min),
                    "candidate_rank_cov_guarded": str(
                        candidate.get("rank_cov_guarded", "-")
                    ),
                    "baseline_rank_cov_guarded": str(
                        baseline.get("rank_cov_guarded", "-")
                    ),
                    "rank_cov_guarded_delta": fmt_delta(rank_cov_guarded_delta),
                    "candidate_rank_cov_zero_ratio": str(
                        candidate.get("rank_cov_zero_ratio", "-")
                    ),
                    "baseline_rank_cov_zero_ratio": str(
                        baseline.get("rank_cov_zero_ratio", "-")
                    ),
                    "rank_cov_zero_ratio_delta": fmt_delta(rank_cov_zero_delta),
                    "candidate_rank_cov_filled": str(
                        candidate.get("rank_cov_filled", "-")
                    ),
                    "baseline_rank_cov_filled": str(
                        baseline.get("rank_cov_filled", "-")
                    ),
                    "rank_cov_filled_delta": fmt_delta(rank_cov_filled_delta),
                    "candidate_final_nll": str(candidate.get("final_nll", "-")),
                    "baseline_final_nll": str(baseline.get("final_nll", "-")),
                    "final_nll_delta": fmt_delta(final_nll_delta),
                    "candidate_final_vs_bigram": str(
                        candidate.get("final_vs_bigram", "-")
                    ),
                    "baseline_final_vs_bigram": str(
                        baseline.get("final_vs_bigram", "-")
                    ),
                    "final_vs_bigram_delta": fmt_delta(bigram_delta),
                    "candidate_bigram_rank_debt": str(
                        candidate.get("final_bigram_rank_debt", "-")
                    ),
                    "baseline_bigram_rank_debt": str(
                        baseline.get("final_bigram_rank_debt", "-")
                    ),
                    "bigram_rank_debt_delta": fmt_delta(rank_debt_delta),
                    "candidate_bigram_rank_lift": str(
                        candidate.get("final_bigram_rank_lift", "-")
                    ),
                    "baseline_bigram_rank_lift": str(
                        baseline.get("final_bigram_rank_lift", "-")
                    ),
                    "bigram_rank_lift_delta": fmt_delta(rank_lift_delta),
                    "candidate_top5_bigram_overlap": str(
                        candidate.get("final_top5_bigram_overlap", "-")
                    ),
                    "baseline_top5_bigram_overlap": str(
                        baseline.get("final_top5_bigram_overlap", "-")
                    ),
                    "top5_bigram_overlap_delta_pp": fmt_delta(top5_delta),
                    "nll_status": nll_status,
                    "bigram_gap_status": bigram_gap_status,
                    "rank_debt_status": rank_debt_status,
                    "bigram_rank_status": bigram_rank_status,
                    "top5_bigram_status": top5_bigram_status,
                    "rank_cov_guarded_status": rank_cov_guarded_status,
                    "rank_cov_zero_status": rank_cov_zero_status,
                    "min_verdict": bigram_rank_min_verdict(
                        nll_status=nll_status,
                        bigram_gap_status=bigram_gap_status,
                        bigram_logprob_status="missing",
                        rank_debt_status=rank_debt_status,
                        bigram_rank_status=bigram_rank_status,
                        top5_bigram_status=top5_bigram_status,
                        rank_cov_guarded_status=rank_cov_guarded_status,
                        rank_cov_zero_status=rank_cov_zero_status,
                    ),
                    "alignment_status": alignment_status,
                }
                pairs.append(pair)
    return pairs


def rank_min_seed_stability_verdict(
    *,
    improved: int,
    neutral: int,
    regressed: int,
    missing: int,
) -> str:
    observed = improved + neutral + regressed
    if observed == 0:
        return "rank_min_seed_inconclusive" if missing else "rank_min_seed_empty"
    if improved and regressed:
        return "rank_min_seed_mixed"
    if regressed:
        return "rank_min_seed_regressed"
    if improved and neutral:
        return "rank_min_seed_improved_or_neutral"
    if improved:
        return "rank_min_seed_stably_improved"
    return "rank_min_seed_neutral"


def rank_min_seed_bounded_mixed(
    *,
    improved: int,
    neutral: int,
    regressed: int,
    missing: int,
    mean_rank_cov_zero_ratio_delta: float | None,
    max_bigram_rank_debt_delta: float | None,
    mean_final_nll_delta: float | None,
    mean_final_vs_bigram_delta: float | None,
    mean_top5_bigram_overlap_delta: float | None,
) -> bool:
    observed = improved + neutral + regressed
    if observed < RANK_MIN_BOUNDED_MIN_SEED_PAIRS or missing:
        return False
    if not improved or not regressed:
        return False
    if regressed / observed > RANK_MIN_BOUNDED_MAX_REGRESSED_RATIO:
        return False
    if (
        mean_rank_cov_zero_ratio_delta is None
        or mean_rank_cov_zero_ratio_delta >= -PAIR_DELTA_TOLERANCE
    ):
        return False
    if (
        max_bigram_rank_debt_delta is not None
        and max_bigram_rank_debt_delta > RANK_MIN_BOUNDED_MAX_RANK_DEBT_DELTA
    ):
        return False
    if (
        mean_top5_bigram_overlap_delta is not None
        and mean_top5_bigram_overlap_delta < -RANK_MIN_BOUNDED_MAX_TOP5_DROP_PP
    ):
        return False
    if mean_final_nll_delta is not None and mean_final_nll_delta > PAIR_DELTA_TOLERANCE:
        return False
    if (
        mean_final_vs_bigram_delta is not None
        and mean_final_vs_bigram_delta > PAIR_DELTA_TOLERANCE
    ):
        return False
    return True


def merged_evidence_sources(group_rows: list[dict[str, str]]) -> list[str]:
    return sorted(
        {
            str(row.get("source", "-"))
            for row in group_rows
            if str(row.get("source", "-"))
        }
    )


def bigram_rank_min_stability_rows(
    seed_deltas: list[dict[str, str]],
    *,
    merge_sources: bool = False,
) -> list[dict[str, str]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in seed_deltas:
        group_values = tuple(
            str(row.get(column, "-"))
            for column in BIGRAM_RANK_MIN_STABILITY_GROUP_COLUMNS
        )
        source = str(row.get("source", "-"))
        key = group_values if merge_sources else (source, *group_values)
        grouped.setdefault(key, []).append(row)

    rows: list[dict[str, str]] = []
    for key, group_rows in sorted(grouped.items()):
        statuses = [str(row.get("alignment_status", "missing")) for row in group_rows]
        improved = statuses.count("improved")
        neutral = statuses.count("neutral")
        regressed = statuses.count("regressed")
        missing = sum(1 for status in statuses if status == "missing")
        if merge_sources:
            group_values = list(key)
            evidence_sources = merged_evidence_sources(group_rows)
            source = f"merged:{len(evidence_sources)}"
        else:
            source, *group_values = key
            evidence_sources = [source]
        mean_rank_cov_zero_ratio_delta = mean_number_cells(
            group_rows,
            "rank_cov_zero_ratio_delta",
        )
        mean_rank_cov_guarded_delta = mean_number_cells(
            group_rows,
            "rank_cov_guarded_delta",
        )
        mean_rank_cov_filled_delta = mean_number_cells(
            group_rows,
            "rank_cov_filled_delta",
        )
        mean_bigram_rank_debt_delta = mean_number_cells(
            group_rows,
            "bigram_rank_debt_delta",
        )
        min_bigram_rank_debt_delta = min_number_cell(
            group_rows,
            "bigram_rank_debt_delta",
        )
        max_bigram_rank_debt_delta = max_number_cell(
            group_rows,
            "bigram_rank_debt_delta",
        )
        mean_bigram_rank_lift_delta = mean_number_cells(
            group_rows,
            "bigram_rank_lift_delta",
        )
        mean_final_nll_delta = mean_number_cells(group_rows, "final_nll_delta")
        mean_final_vs_bigram_delta = mean_number_cells(
            group_rows,
            "final_vs_bigram_delta",
        )
        mean_top5_bigram_overlap_delta = mean_number_cells(
            group_rows,
            "top5_bigram_overlap_delta_pp",
        )
        stability_verdict = rank_min_seed_stability_verdict(
            improved=improved,
            neutral=neutral,
            regressed=regressed,
            missing=missing,
        )
        if stability_verdict == "rank_min_seed_mixed" and rank_min_seed_bounded_mixed(
            improved=improved,
            neutral=neutral,
            regressed=regressed,
            missing=missing,
            mean_rank_cov_zero_ratio_delta=mean_rank_cov_zero_ratio_delta,
            max_bigram_rank_debt_delta=max_bigram_rank_debt_delta,
            mean_final_nll_delta=mean_final_nll_delta,
            mean_final_vs_bigram_delta=mean_final_vs_bigram_delta,
            mean_top5_bigram_overlap_delta=mean_top5_bigram_overlap_delta,
        ):
            stability_verdict = "rank_min_seed_bounded_mixed"
        row = {
            "source": source,
            **dict(
                zip(
                    BIGRAM_RANK_MIN_STABILITY_GROUP_COLUMNS,
                    group_values,
                    strict=True,
                )
            ),
            "seed_pairs": str(len(group_rows)),
            "alignment_improved_seeds": str(improved),
            "alignment_neutral_seeds": str(neutral),
            "alignment_regressed_seeds": str(regressed),
            "alignment_missing_seeds": str(missing),
            "mean_rank_cov_zero_ratio_delta": fmt_delta(
                mean_rank_cov_zero_ratio_delta
            ),
            "mean_rank_cov_guarded_delta": fmt_delta(mean_rank_cov_guarded_delta),
            "mean_rank_cov_filled_delta": fmt_delta(mean_rank_cov_filled_delta),
            "mean_bigram_rank_debt_delta": fmt_delta(mean_bigram_rank_debt_delta),
            "min_bigram_rank_debt_delta": fmt_delta(min_bigram_rank_debt_delta),
            "max_bigram_rank_debt_delta": fmt_delta(max_bigram_rank_debt_delta),
            "mean_bigram_rank_lift_delta": fmt_delta(mean_bigram_rank_lift_delta),
            "mean_final_nll_delta": fmt_delta(mean_final_nll_delta),
            "mean_final_vs_bigram_delta": fmt_delta(mean_final_vs_bigram_delta),
            "mean_top5_bigram_overlap_delta_pp": fmt_delta(
                mean_top5_bigram_overlap_delta
            ),
            "stability_verdict": stability_verdict,
        }
        if merge_sources:
            row["evidence_source_count"] = str(len(evidence_sources))
            row["evidence_sources"] = ",".join(evidence_sources)
        rows.append(row)
    return rows


def bigram_rank_min_stable_recommendations(
    stability_rows: list[dict[str, str]],
    *,
    limit: int,
) -> list[dict[str, str]]:
    candidates = [
        row
        for row in stability_rows
        if row.get("stability_verdict")
        in RECOMMENDED_BIGRAM_RANK_MIN_STABILITY_VERDICTS
    ]
    verdict_rank = {
        "rank_min_seed_stably_improved": 0,
        "rank_min_seed_improved_or_neutral": 1,
        "rank_min_seed_bounded_mixed": 2,
    }
    sorted_rows = sorted(
        candidates,
        key=lambda row: (
            verdict_rank.get(str(row.get("stability_verdict")), 99),
            pair_metric_for_sort(row, "mean_rank_cov_zero_ratio_delta"),
            pair_metric_for_sort(row, "max_bigram_rank_debt_delta"),
            pair_metric_for_sort(row, "mean_bigram_rank_debt_delta"),
            pair_metric_desc_for_sort(row, "mean_rank_cov_guarded_delta"),
            pair_metric_desc_for_sort(row, "mean_bigram_rank_lift_delta"),
            pair_metric_for_sort(row, "candidate_bigram_rank_min"),
            pair_metric_for_sort(row, "val_start"),
            str(row.get("source", "")),
        ),
    )
    if limit > 0:
        sorted_rows = sorted_rows[:limit]

    recommendations: list[dict[str, str]] = []
    for rank, source_row in enumerate(sorted_rows, start=1):
        verdict = str(source_row.get("stability_verdict", "-"))
        row = {
            header: str(source_row.get(header, "-"))
            for header in BIGRAM_RANK_MIN_STABLE_RECOMMENDATION_HEADERS
        }
        row["rank"] = str(rank)
        row["recommendation"] = RECOMMENDED_BIGRAM_RANK_MIN_STABILITY_VERDICTS.get(
            verdict,
            "-",
        )
        recommendations.append(row)
    return recommendations


def format_count_map(counts: dict[str, int]) -> str:
    if not counts:
        return "-"
    return ",".join(f"{key}:{counts[key]}" for key in sorted(counts))


def bigram_rank_min_promotion_gate(
    stability_rows: list[dict[str, str]],
    stable_recommendations: list[dict[str, str]],
    *,
    fail_on_decisions: list[str] | None = None,
) -> dict[str, str]:
    verdict_counts: dict[str, int] = {}
    for row in stability_rows:
        verdict = str(row.get("stability_verdict", "-"))
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    total_rows = len(stability_rows)
    strict_promotions = verdict_counts.get("rank_min_seed_stably_improved", 0)
    bounded_promotions = verdict_counts.get("rank_min_seed_bounded_mixed", 0)
    promoted_rows = sum(
        count
        for verdict, count in verdict_counts.items()
        if verdict in RECOMMENDED_BIGRAM_RANK_MIN_STABILITY_VERDICTS
    )
    non_promoted_rows = total_rows - promoted_rows
    if total_rows == 0:
        decision = "no_rank_min_evidence"
    elif non_promoted_rows == 0 and bounded_promotions == 0:
        decision = "promote"
    elif non_promoted_rows == 0:
        decision = "promote_with_bounded_watch"
    elif promoted_rows:
        decision = "partial_promote_needs_tuning"
    else:
        decision = "needs_tuning"

    forbidden_decisions = fail_on_decisions or []
    failed = decision in set(forbidden_decisions)
    return {
        "decision": decision,
        "failed": str(failed).lower(),
        "fail_on_decisions": ",".join(forbidden_decisions),
        "total_rows": str(total_rows),
        "strict_promotions": str(strict_promotions),
        "bounded_promotions": str(bounded_promotions),
        "non_promoted_rows": str(non_promoted_rows),
        "recommendation_rows": str(len(stable_recommendations)),
        "verdict_counts": format_count_map(verdict_counts),
    }


def bigram_soft_guard_verdict(
    *,
    nll_status: str,
    bigram_gap_status: str,
    bigram_logprob_status: str,
    rank_debt_status: str,
    bigram_rank_status: str,
    top5_bigram_status: str,
) -> str:
    quality_statuses = [nll_status, bigram_gap_status, bigram_logprob_status]
    alignment_statuses = [rank_debt_status, bigram_rank_status, top5_bigram_status]
    quality_regressed = any(status == "regressed" for status in quality_statuses)
    quality_improved = any(status == "improved" for status in quality_statuses)
    alignment_regressed = any(
        status == "regressed" for status in alignment_statuses
    )
    alignment_improved = any(status == "improved" for status in alignment_statuses)
    quality_missing = all(status == "missing" for status in quality_statuses)
    alignment_missing = all(status == "missing" for status in alignment_statuses)

    if quality_regressed:
        return "soft_guard_quality_regressed"
    if alignment_improved and alignment_regressed:
        if quality_improved:
            return "soft_guard_quality_improved_alignment_mixed"
        return "soft_guard_alignment_mixed"
    if alignment_regressed:
        if quality_improved:
            return "soft_guard_quality_improved_alignment_regressed"
        return "soft_guard_alignment_regressed"
    if quality_improved and alignment_improved:
        return "soft_guard_quality_and_alignment_improved"
    if alignment_improved:
        return "soft_guard_alignment_improved"
    if quality_improved:
        return "soft_guard_quality_improved"
    if quality_missing and alignment_missing:
        return "soft_guard_inconclusive"
    return "soft_guard_neutral"


def paired_bigram_soft_guard_deltas(
    payloads: list[tuple[str, dict[str, Any]]],
    *,
    baseline_guard: float = DEFAULT_BIGRAM_SOFT_GUARD_BASELINE,
) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for source, payload in payloads:
        grouped: dict[tuple[str, ...], dict[float, dict[str, Any]]] = {}
        for row in source_rows(payload):
            guard = parse_number_cell(row.get("bigram_soft_guard"))
            if guard is None:
                continue
            key = tuple(
                str(row.get(column, "-")) for column in BIGRAM_SOFT_GUARD_GROUP_COLUMNS
            )
            grouped.setdefault(key, {})[guard] = row

        for key, rows_by_guard in sorted(grouped.items()):
            baseline = rows_by_guard.get(baseline_guard)
            if baseline is None:
                continue
            for guard in sorted(value for value in rows_by_guard if value != baseline_guard):
                candidate = rows_by_guard[guard]
                candidate_final_nll = parse_number_cell(candidate.get("final_nll_mean"))
                baseline_final_nll = parse_number_cell(baseline.get("final_nll_mean"))
                candidate_bigram = parse_number_cell(
                    candidate.get("final_vs_bigram_mean")
                )
                baseline_bigram = parse_number_cell(baseline.get("final_vs_bigram_mean"))
                candidate_logprob_lift = parse_number_cell(
                    candidate.get("final_bigram_logprob_lift_mean")
                )
                baseline_logprob_lift = parse_number_cell(
                    baseline.get("final_bigram_logprob_lift_mean")
                )
                candidate_rank_debt = parse_number_cell(
                    candidate.get("final_bigram_rank_debt_mean")
                )
                baseline_rank_debt = parse_number_cell(
                    baseline.get("final_bigram_rank_debt_mean")
                )
                candidate_rank_lift = parse_number_cell(
                    candidate.get("final_bigram_rank_lift_mean")
                )
                baseline_rank_lift = parse_number_cell(
                    baseline.get("final_bigram_rank_lift_mean")
                )
                candidate_top5 = parse_number_cell(
                    candidate.get("final_top5_bigram_overlap_mean")
                )
                baseline_top5 = parse_number_cell(
                    baseline.get("final_top5_bigram_overlap_mean")
                )
                final_nll_delta = (
                    candidate_final_nll - baseline_final_nll
                    if candidate_final_nll is not None
                    and baseline_final_nll is not None
                    else None
                )
                bigram_delta = (
                    candidate_bigram - baseline_bigram
                    if candidate_bigram is not None and baseline_bigram is not None
                    else None
                )
                logprob_lift_delta = (
                    candidate_logprob_lift - baseline_logprob_lift
                    if candidate_logprob_lift is not None
                    and baseline_logprob_lift is not None
                    else None
                )
                rank_debt_delta = (
                    candidate_rank_debt - baseline_rank_debt
                    if candidate_rank_debt is not None and baseline_rank_debt is not None
                    else None
                )
                rank_lift_delta = (
                    candidate_rank_lift - baseline_rank_lift
                    if candidate_rank_lift is not None and baseline_rank_lift is not None
                    else None
                )
                top5_delta = (
                    candidate_top5 - baseline_top5
                    if candidate_top5 is not None and baseline_top5 is not None
                    else None
                )
                nll_status = classify_lower_is_better_delta(final_nll_delta)
                bigram_gap_status = classify_lower_is_better_delta(bigram_delta)
                bigram_logprob_status = classify_higher_is_better_delta(
                    logprob_lift_delta
                )
                rank_debt_status = classify_lower_is_better_delta(rank_debt_delta)
                bigram_rank_status = classify_higher_is_better_delta(rank_lift_delta)
                top5_bigram_status = classify_higher_is_better_delta(top5_delta)
                quality_status = combine_quality_status(
                    nll_status,
                    bigram_gap_status,
                    bigram_logprob_status,
                )
                alignment_status = combine_quality_status(
                    rank_debt_status,
                    bigram_rank_status,
                    top5_bigram_status,
                )
                pair = {
                    "source": source,
                    **dict(zip(BIGRAM_SOFT_GUARD_GROUP_COLUMNS, key, strict=True)),
                    "candidate_bigram_soft_guard": fmt_guard_value(guard),
                    "baseline_bigram_soft_guard": fmt_guard_value(baseline_guard),
                    "candidate_runs": str(candidate.get("runs", "-")),
                    "baseline_runs": str(baseline.get("runs", "-")),
                    "candidate_final_nll": str(candidate.get("final_nll_mean", "-")),
                    "baseline_final_nll": str(baseline.get("final_nll_mean", "-")),
                    "final_nll_delta": fmt_delta(final_nll_delta),
                    "candidate_final_vs_bigram": str(
                        candidate.get("final_vs_bigram_mean", "-")
                    ),
                    "baseline_final_vs_bigram": str(
                        baseline.get("final_vs_bigram_mean", "-")
                    ),
                    "final_vs_bigram_delta": fmt_delta(bigram_delta),
                    "candidate_bigram_logprob_lift": str(
                        candidate.get("final_bigram_logprob_lift_mean", "-")
                    ),
                    "baseline_bigram_logprob_lift": str(
                        baseline.get("final_bigram_logprob_lift_mean", "-")
                    ),
                    "bigram_logprob_lift_delta": fmt_delta(logprob_lift_delta),
                    "candidate_bigram_rank_debt": str(
                        candidate.get("final_bigram_rank_debt_mean", "-")
                    ),
                    "baseline_bigram_rank_debt": str(
                        baseline.get("final_bigram_rank_debt_mean", "-")
                    ),
                    "bigram_rank_debt_delta": fmt_delta(rank_debt_delta),
                    "candidate_bigram_rank_lift": str(
                        candidate.get("final_bigram_rank_lift_mean", "-")
                    ),
                    "baseline_bigram_rank_lift": str(
                        baseline.get("final_bigram_rank_lift_mean", "-")
                    ),
                    "bigram_rank_lift_delta": fmt_delta(rank_lift_delta),
                    "candidate_top5_bigram_overlap": str(
                        candidate.get("final_top5_bigram_overlap_mean", "-")
                    ),
                    "baseline_top5_bigram_overlap": str(
                        baseline.get("final_top5_bigram_overlap_mean", "-")
                    ),
                    "top5_bigram_overlap_delta_pp": fmt_delta(top5_delta),
                    "nll_status": nll_status,
                    "bigram_gap_status": bigram_gap_status,
                    "bigram_logprob_status": bigram_logprob_status,
                    "rank_debt_status": rank_debt_status,
                    "bigram_rank_status": bigram_rank_status,
                    "top5_bigram_status": top5_bigram_status,
                    "guard_verdict": bigram_soft_guard_verdict(
                        nll_status=nll_status,
                        bigram_gap_status=bigram_gap_status,
                        bigram_logprob_status=bigram_logprob_status,
                        rank_debt_status=rank_debt_status,
                        bigram_rank_status=bigram_rank_status,
                        top5_bigram_status=top5_bigram_status,
                    ),
                    "quality_status": quality_status,
                    "alignment_status": alignment_status,
                    "candidate_route_status": route_status(candidate),
                    "baseline_route_status": route_status(baseline),
                }
                pairs.append(pair)
    return pairs


def paired_bigram_soft_guard_recommendations(
    pairs: list[dict[str, str]],
    *,
    limit: int,
) -> list[dict[str, str]]:
    candidates = [
        pair
        for pair in pairs
        if pair.get("guard_verdict") in RECOMMENDED_BIGRAM_SOFT_GUARD_VERDICTS
    ]
    verdict_rank = {
        "soft_guard_quality_and_alignment_improved": 0,
        "soft_guard_alignment_improved": 1,
        "soft_guard_quality_improved": 2,
        "soft_guard_quality_improved_alignment_mixed": 3,
    }
    sorted_pairs = sorted(
        candidates,
        key=lambda pair: (
            verdict_rank.get(str(pair.get("guard_verdict")), 99),
            pair_metric_for_sort(pair, "bigram_rank_debt_delta"),
            pair_metric_desc_for_sort(pair, "bigram_rank_lift_delta"),
            pair_metric_for_sort(pair, "final_nll_delta"),
            pair_metric_for_sort(pair, "final_vs_bigram_delta"),
            pair_metric_desc_for_sort(pair, "top5_bigram_overlap_delta_pp"),
            pair_metric_for_sort(pair, "candidate_final_nll"),
            pair_metric_for_sort(pair, "candidate_bigram_soft_guard"),
            str(pair.get("source", "")),
        ),
    )
    if limit > 0:
        sorted_pairs = sorted_pairs[:limit]

    recommendations: list[dict[str, str]] = []
    for rank, pair in enumerate(sorted_pairs, start=1):
        verdict = str(pair.get("guard_verdict", "-"))
        row = {
            header: str(pair.get(header, "-"))
            for header in BIGRAM_SOFT_GUARD_RECOMMENDATION_HEADERS
        }
        row["rank"] = str(rank)
        row["recommendation"] = RECOMMENDED_BIGRAM_SOFT_GUARD_VERDICTS.get(
            verdict,
            "-",
        )
        recommendations.append(row)
    return recommendations


def paired_bigram_soft_guard_seed_deltas(
    payloads: list[tuple[str, dict[str, Any]]],
    *,
    baseline_guard: float = DEFAULT_BIGRAM_SOFT_GUARD_BASELINE,
) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for source, payload in payloads:
        grouped: dict[tuple[str, ...], dict[float, dict[str, Any]]] = {}
        for row in run_rows(payload):
            guard = parse_number_cell(row.get("bigram_soft_guard"))
            if guard is None:
                continue
            key = tuple(
                str(row.get(column, "-"))
                for column in BIGRAM_SOFT_GUARD_SEED_GROUP_COLUMNS
            )
            grouped.setdefault(key, {})[guard] = row

        for key, rows_by_guard in sorted(grouped.items()):
            baseline = rows_by_guard.get(baseline_guard)
            if baseline is None:
                continue
            for guard in sorted(value for value in rows_by_guard if value != baseline_guard):
                candidate = rows_by_guard[guard]
                candidate_final_nll = parse_number_cell(candidate.get("final_nll"))
                baseline_final_nll = parse_number_cell(baseline.get("final_nll"))
                candidate_bigram = parse_number_cell(candidate.get("final_vs_bigram"))
                baseline_bigram = parse_number_cell(baseline.get("final_vs_bigram"))
                candidate_rank_debt = parse_number_cell(
                    candidate.get("final_bigram_rank_debt")
                )
                baseline_rank_debt = parse_number_cell(
                    baseline.get("final_bigram_rank_debt")
                )
                candidate_rank_lift = parse_number_cell(
                    candidate.get("final_bigram_rank_lift")
                )
                baseline_rank_lift = parse_number_cell(
                    baseline.get("final_bigram_rank_lift")
                )
                candidate_top5 = parse_number_cell(
                    candidate.get("final_top5_bigram_overlap")
                )
                baseline_top5 = parse_number_cell(
                    baseline.get("final_top5_bigram_overlap")
                )
                final_nll_delta = (
                    candidate_final_nll - baseline_final_nll
                    if candidate_final_nll is not None
                    and baseline_final_nll is not None
                    else None
                )
                bigram_delta = (
                    candidate_bigram - baseline_bigram
                    if candidate_bigram is not None and baseline_bigram is not None
                    else None
                )
                rank_debt_delta = (
                    candidate_rank_debt - baseline_rank_debt
                    if candidate_rank_debt is not None and baseline_rank_debt is not None
                    else None
                )
                rank_lift_delta = (
                    candidate_rank_lift - baseline_rank_lift
                    if candidate_rank_lift is not None and baseline_rank_lift is not None
                    else None
                )
                top5_delta = (
                    candidate_top5 - baseline_top5
                    if candidate_top5 is not None and baseline_top5 is not None
                    else None
                )
                nll_status = classify_lower_is_better_delta(final_nll_delta)
                bigram_gap_status = classify_lower_is_better_delta(bigram_delta)
                rank_debt_status = classify_lower_is_better_delta(rank_debt_delta)
                bigram_rank_status = classify_higher_is_better_delta(rank_lift_delta)
                top5_bigram_status = classify_higher_is_better_delta(top5_delta)
                alignment_status = combine_quality_status(
                    rank_debt_status,
                    bigram_rank_status,
                    top5_bigram_status,
                )
                pair = {
                    "source": source,
                    **dict(
                        zip(
                            BIGRAM_SOFT_GUARD_SEED_GROUP_COLUMNS,
                            key,
                            strict=True,
                        )
                    ),
                    "candidate_bigram_soft_guard": fmt_guard_value(guard),
                    "baseline_bigram_soft_guard": fmt_guard_value(baseline_guard),
                    "candidate_final_nll": str(candidate.get("final_nll", "-")),
                    "baseline_final_nll": str(baseline.get("final_nll", "-")),
                    "final_nll_delta": fmt_delta(final_nll_delta),
                    "candidate_final_vs_bigram": str(
                        candidate.get("final_vs_bigram", "-")
                    ),
                    "baseline_final_vs_bigram": str(
                        baseline.get("final_vs_bigram", "-")
                    ),
                    "final_vs_bigram_delta": fmt_delta(bigram_delta),
                    "candidate_bigram_rank_debt": str(
                        candidate.get("final_bigram_rank_debt", "-")
                    ),
                    "baseline_bigram_rank_debt": str(
                        baseline.get("final_bigram_rank_debt", "-")
                    ),
                    "bigram_rank_debt_delta": fmt_delta(rank_debt_delta),
                    "candidate_bigram_rank_lift": str(
                        candidate.get("final_bigram_rank_lift", "-")
                    ),
                    "baseline_bigram_rank_lift": str(
                        baseline.get("final_bigram_rank_lift", "-")
                    ),
                    "bigram_rank_lift_delta": fmt_delta(rank_lift_delta),
                    "candidate_top5_bigram_overlap": str(
                        candidate.get("final_top5_bigram_overlap", "-")
                    ),
                    "baseline_top5_bigram_overlap": str(
                        baseline.get("final_top5_bigram_overlap", "-")
                    ),
                    "top5_bigram_overlap_delta_pp": fmt_delta(top5_delta),
                    "nll_status": nll_status,
                    "bigram_gap_status": bigram_gap_status,
                    "rank_debt_status": rank_debt_status,
                    "bigram_rank_status": bigram_rank_status,
                    "top5_bigram_status": top5_bigram_status,
                    "guard_verdict": bigram_soft_guard_verdict(
                        nll_status=nll_status,
                        bigram_gap_status=bigram_gap_status,
                        bigram_logprob_status="missing",
                        rank_debt_status=rank_debt_status,
                        bigram_rank_status=bigram_rank_status,
                        top5_bigram_status=top5_bigram_status,
                    ),
                    "alignment_status": alignment_status,
                }
                pairs.append(pair)
    return pairs


def soft_guard_seed_stability_verdict(
    *,
    improved: int,
    neutral: int,
    regressed: int,
    missing: int,
) -> str:
    observed = improved + neutral + regressed
    if observed == 0:
        return "soft_guard_seed_inconclusive" if missing else "soft_guard_seed_empty"
    if improved and regressed:
        return "soft_guard_seed_mixed"
    if regressed:
        return "soft_guard_seed_regressed"
    if improved and neutral:
        return "soft_guard_seed_improved_or_neutral"
    if improved:
        return "soft_guard_seed_stably_improved"
    return "soft_guard_seed_neutral"


def bigram_soft_guard_stability_rows(
    seed_deltas: list[dict[str, str]],
) -> list[dict[str, str]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in seed_deltas:
        key = tuple(
            str(row.get(column, "-"))
            for column in ["source", *BIGRAM_SOFT_GUARD_STABILITY_GROUP_COLUMNS]
        )
        grouped.setdefault(key, []).append(row)

    rows: list[dict[str, str]] = []
    for key, group_rows in sorted(grouped.items()):
        statuses = [str(row.get("alignment_status", "missing")) for row in group_rows]
        improved = statuses.count("improved")
        neutral = statuses.count("neutral")
        regressed = statuses.count("regressed")
        missing = sum(1 for status in statuses if status == "missing")
        source, *group_values = key
        rows.append(
            {
                "source": source,
                **dict(
                    zip(
                        BIGRAM_SOFT_GUARD_STABILITY_GROUP_COLUMNS,
                        group_values,
                        strict=True,
                    )
                ),
                "seed_pairs": str(len(group_rows)),
                "alignment_improved_seeds": str(improved),
                "alignment_neutral_seeds": str(neutral),
                "alignment_regressed_seeds": str(regressed),
                "alignment_missing_seeds": str(missing),
                "mean_bigram_rank_debt_delta": fmt_delta(
                    mean_number_cells(group_rows, "bigram_rank_debt_delta")
                ),
                "min_bigram_rank_debt_delta": fmt_delta(
                    min_number_cell(group_rows, "bigram_rank_debt_delta")
                ),
                "max_bigram_rank_debt_delta": fmt_delta(
                    max_number_cell(group_rows, "bigram_rank_debt_delta")
                ),
                "mean_bigram_rank_lift_delta": fmt_delta(
                    mean_number_cells(group_rows, "bigram_rank_lift_delta")
                ),
                "mean_final_nll_delta": fmt_delta(
                    mean_number_cells(group_rows, "final_nll_delta")
                ),
                "mean_final_vs_bigram_delta": fmt_delta(
                    mean_number_cells(group_rows, "final_vs_bigram_delta")
                ),
                "mean_top5_bigram_overlap_delta_pp": fmt_delta(
                    mean_number_cells(group_rows, "top5_bigram_overlap_delta_pp")
                ),
                "stability_verdict": soft_guard_seed_stability_verdict(
                    improved=improved,
                    neutral=neutral,
                    regressed=regressed,
                    missing=missing,
                ),
            }
        )
    return rows


def paired_recurrent_recommendations(
    pairs: list[dict[str, str]],
    *,
    limit: int,
) -> list[dict[str, str]]:
    candidates = [
        pair
        for pair in pairs
        if pair.get("efficiency_verdict") in RECOMMENDED_EFFICIENCY_VERDICTS
        and pair.get("candidate_learning_status") in {"improved", "neutral"}
    ]
    verdict_rank = {
        "candidate_better_quality_and_cost": 0,
        "candidate_quality_neutral_cost_better": 1,
        "candidate_quality_better_cost_neutral": 2,
    }
    sorted_pairs = sorted(
        candidates,
        key=lambda pair: (
            verdict_rank.get(str(pair.get("efficiency_verdict")), 99),
            pair_metric_for_sort(pair, "cpu_debt_ratio"),
            pair_metric_for_sort(pair, "trace_step_ms_ratio"),
            pair_metric_for_sort(pair, "final_nll_delta"),
            pair_metric_for_sort(pair, "final_vs_bigram_delta"),
            str(pair.get("source", "")),
        ),
    )
    if limit > 0:
        sorted_pairs = sorted_pairs[:limit]

    recommendations: list[dict[str, str]] = []
    for rank, pair in enumerate(sorted_pairs, start=1):
        verdict = str(pair.get("efficiency_verdict", "-"))
        row = {header: str(pair.get(header, "-")) for header in PAIR_RECOMMENDATION_HEADERS}
        row["rank"] = str(rank)
        row["recommendation"] = RECOMMENDED_EFFICIENCY_VERDICTS.get(verdict, "-")
        recommendations.append(row)
    return recommendations


def bigram_baseline_status(delta: float | None) -> str:
    status = classify_lower_is_better_delta(delta)
    return {
        "improved": "bigram_stronger_than_unigram",
        "neutral": "bigram_near_unigram",
        "regressed": "bigram_weaker_than_unigram",
        "missing": "missing",
    }[status]


def model_vs_bigram_status(delta: float | None) -> str:
    status = classify_lower_is_better_delta(delta)
    return {
        "improved": "model_beats_bigram",
        "neutral": "model_near_bigram",
        "regressed": "model_lags_bigram",
        "missing": "missing",
    }[status]


def learning_status(delta: float | None) -> str:
    status = classify_lower_is_better_delta(delta)
    return {
        "improved": "loss_improved",
        "neutral": "loss_neutral",
        "regressed": "loss_regressed",
        "missing": "missing",
    }[status]


def baseline_difficulty_priority(
    *,
    bigram_status: str,
    model_status: str,
) -> int:
    if (
        bigram_status == "bigram_stronger_than_unigram"
        and model_status == "model_lags_bigram"
    ):
        return 0
    if model_status == "model_lags_bigram":
        return 1
    if bigram_status == "bigram_stronger_than_unigram":
        return 2
    if model_status == "model_near_bigram":
        return 3
    return 4


def baseline_difficulty_rows(
    payloads: list[tuple[str, dict[str, Any]]],
    *,
    limit: int,
) -> list[dict[str, str]]:
    candidates: list[tuple[tuple[float, ...], dict[str, str]]] = []
    for source, payload in payloads:
        for row in source_rows(payload):
            unigram_nll = parse_number_cell(row.get("unigram_nll_mean"))
            bigram_nll = parse_number_cell(row.get("bigram_nll_mean"))
            final_vs_bigram = parse_number_cell(row.get("final_vs_bigram_mean"))
            final_nll = parse_number_cell(row.get("final_nll_mean"))
            delta_nll = parse_number_cell(row.get("delta_nll_mean"))
            if (
                unigram_nll is None
                and bigram_nll is None
                and final_vs_bigram is None
            ):
                continue

            bigram_delta = (
                bigram_nll - unigram_nll
                if bigram_nll is not None and unigram_nll is not None
                else None
            )
            bigram_status = bigram_baseline_status(bigram_delta)
            model_status = model_vs_bigram_status(final_vs_bigram)
            route = route_status(row)
            summary = {
                header: str(row.get(header, "-"))
                for header in BASELINE_DIFFICULTY_HEADERS
                if header
                not in {
                    "rank",
                    "source",
                    "bigram_vs_unigram_delta",
                    "bigram_baseline_status",
                    "model_vs_bigram_status",
                    "learning_status",
                    "route_status",
                }
            }
            summary["source"] = source
            summary["bigram_vs_unigram_delta"] = fmt_delta(bigram_delta)
            summary["bigram_baseline_status"] = bigram_status
            summary["model_vs_bigram_status"] = model_status
            summary["learning_status"] = learning_status(delta_nll)
            summary["route_status"] = route
            candidates.append(
                (
                    (
                        float(
                            baseline_difficulty_priority(
                                bigram_status=bigram_status,
                                model_status=model_status,
                            )
                        ),
                        -(final_vs_bigram if final_vs_bigram is not None else float("-inf")),
                        bigram_delta if bigram_delta is not None else float("inf"),
                        final_nll if final_nll is not None else float("inf"),
                        float(route_penalty_for_row(row, route)),
                    ),
                    summary,
                )
            )

    candidates.sort(key=lambda item: item[0])
    selected = [row for _, row in candidates[: max(limit, 0)]]
    for rank, row in enumerate(selected, start=1):
        row["rank"] = str(rank)
    return selected


def positive_learning_gain(delta_nll: float | None) -> float | None:
    if delta_nll is None:
        return None
    return -delta_nll


def gain_per_ms(learning_gain: float | None, trace_step_ms: float | None) -> float | None:
    if learning_gain is None or trace_step_ms is None or trace_step_ms <= 0.0:
        return None
    return learning_gain / trace_step_ms


def learning_scoreboard_rows(
    payloads: list[tuple[str, dict[str, Any]]],
    *,
    limit: int,
) -> list[dict[str, str]]:
    candidates: list[tuple[tuple[float, ...], dict[str, str]]] = []
    for source, payload in payloads:
        for row in source_rows(payload):
            delta_nll = parse_number_cell(row.get("delta_nll_mean"))
            final_nll = parse_number_cell(row.get("final_nll_mean"))
            best_nll = parse_number_cell(row.get("best_nll_mean"))
            bigram_gap = parse_number_cell(row.get("final_vs_bigram_mean"))
            trace_step_ms = parse_number_cell(row.get("trace_step_ms_mean_mean"))
            cpu_debt = parse_number_cell(row.get("cpu_debt_ops_mean"))
            learning_gain = positive_learning_gain(delta_nll)
            per_ms = gain_per_ms(learning_gain, trace_step_ms)
            if learning_gain is None and final_nll is None:
                continue

            final_minus_best = (
                final_nll - best_nll
                if final_nll is not None and best_nll is not None
                else None
            )
            route = route_status(row)
            summary = {
                header: str(row.get(header, "-"))
                for header in LEARNING_SCOREBOARD_HEADERS
                if header
                not in {
                    "rank",
                    "source",
                    "learning_gain",
                    "learning_status",
                    "final_minus_best",
                    "bigram_gap",
                    "bigram_gap_status",
                    "trace_step_ms_mean",
                    "gain_per_ms",
                    "route_status",
                }
            }
            summary["source"] = source
            summary["learning_gain"] = fmt_delta(learning_gain)
            summary["learning_status"] = learning_status(delta_nll)
            summary["final_minus_best"] = fmt_delta(final_minus_best)
            summary["bigram_gap"] = fmt_delta(bigram_gap)
            summary["bigram_gap_status"] = model_vs_bigram_status(bigram_gap)
            summary["trace_step_ms_mean"] = fmt_delta(trace_step_ms)
            summary["gain_per_ms"] = fmt_delta(per_ms)
            summary["route_status"] = route
            candidates.append(
                (
                    (
                        -(per_ms if per_ms is not None else float("-inf")),
                        -(learning_gain if learning_gain is not None else float("-inf")),
                        bigram_gap if bigram_gap is not None else float("inf"),
                        final_nll if final_nll is not None else float("inf"),
                        trace_step_ms if trace_step_ms is not None else float("inf"),
                        cpu_debt if cpu_debt is not None else float("inf"),
                        float(route_penalty_for_row(row, route)),
                    ),
                    summary,
                )
            )

    candidates.sort(key=lambda item: item[0])
    selected = [row for _, row in candidates[: max(limit, 0)]]
    for rank, row in enumerate(selected, start=1):
        row["rank"] = str(rank)
    return selected


def route_debt_verdict(
    *,
    quality_status: str,
    latency_status: str,
    cpu_debt_status: str,
    route_debt_status: str,
) -> str:
    if quality_status == "regressed":
        return "route_debt_quality_regressed"
    if route_debt_status != "improved":
        return "route_debt_not_lower"
    if latency_status == "regressed" or cpu_debt_status == "regressed":
        return "route_debt_lower_cost_mixed"
    if quality_status == "improved":
        return "route_debt_quality_improved_lighter"
    if quality_status == "neutral":
        return "route_debt_quality_neutral_lighter"
    return "route_debt_inconclusive"


def route_debt_recommendations(
    payloads: list[tuple[str, dict[str, Any]]],
    *,
    limit: int,
) -> list[dict[str, str]]:
    candidates: list[tuple[tuple[float, ...], dict[str, str]]] = []
    verdict_rank = {
        "route_debt_quality_improved_lighter": 0,
        "route_debt_quality_neutral_lighter": 1,
    }
    for source, payload in payloads:
        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for row in source_rows(payload):
            wave_dilations = str(row.get("wave_dilations", "-"))
            route_debt = parse_number_cell(row.get("coherence_route_debt_mean"))
            if wave_dilations in {"", "-"} or route_debt is None:
                continue
            key = tuple(str(row.get(column, "-")) for column in ROUTE_DEBT_GROUP_COLUMNS)
            grouped.setdefault(key, []).append(row)

        for key, rows in sorted(grouped.items()):
            for candidate in rows:
                candidate_wave = str(candidate.get("wave_dilations", "-"))
                candidate_route_debt = parse_number_cell(
                    candidate.get("coherence_route_debt_mean")
                )
                if candidate_route_debt is None:
                    continue
                for baseline in rows:
                    baseline_wave = str(baseline.get("wave_dilations", "-"))
                    if baseline is candidate or baseline_wave == candidate_wave:
                        continue
                    baseline_route_debt = parse_number_cell(
                        baseline.get("coherence_route_debt_mean")
                    )
                    if baseline_route_debt is None:
                        continue

                    route_debt_delta = candidate_route_debt - baseline_route_debt
                    route_status_delta = classify_lower_is_better_delta(
                        route_debt_delta
                    )
                    if route_status_delta != "improved":
                        continue

                    candidate_final_nll = parse_number_cell(
                        candidate.get("final_nll_mean")
                    )
                    baseline_final_nll = parse_number_cell(
                        baseline.get("final_nll_mean")
                    )
                    candidate_best_nll = parse_number_cell(
                        candidate.get("best_nll_mean")
                    )
                    baseline_best_nll = parse_number_cell(baseline.get("best_nll_mean"))
                    candidate_bigram = parse_number_cell(
                        candidate.get("final_vs_bigram_mean")
                    )
                    baseline_bigram = parse_number_cell(
                        baseline.get("final_vs_bigram_mean")
                    )
                    candidate_step_ms = parse_number_cell(
                        candidate.get("trace_step_ms_mean_mean")
                    )
                    baseline_step_ms = parse_number_cell(
                        baseline.get("trace_step_ms_mean_mean")
                    )
                    candidate_cpu_debt = parse_number_cell(
                        candidate.get("cpu_debt_ops_mean")
                    )
                    baseline_cpu_debt = parse_number_cell(
                        baseline.get("cpu_debt_ops_mean")
                    )

                    final_nll_delta = (
                        candidate_final_nll - baseline_final_nll
                        if candidate_final_nll is not None
                        and baseline_final_nll is not None
                        else None
                    )
                    best_nll_delta = (
                        candidate_best_nll - baseline_best_nll
                        if candidate_best_nll is not None
                        and baseline_best_nll is not None
                        else None
                    )
                    bigram_delta = (
                        candidate_bigram - baseline_bigram
                        if candidate_bigram is not None and baseline_bigram is not None
                        else None
                    )
                    step_ms_delta = (
                        candidate_step_ms - baseline_step_ms
                        if candidate_step_ms is not None
                        and baseline_step_ms is not None
                        else None
                    )
                    cpu_debt_delta = (
                        candidate_cpu_debt - baseline_cpu_debt
                        if candidate_cpu_debt is not None
                        and baseline_cpu_debt is not None
                        else None
                    )

                    quality_status = combine_quality_status(
                        classify_lower_is_better_delta(final_nll_delta),
                        classify_lower_is_better_delta(best_nll_delta),
                        classify_lower_is_better_delta(bigram_delta),
                    )
                    latency_status = classify_lower_is_better_delta(step_ms_delta)
                    cpu_debt_status = classify_lower_is_better_delta(cpu_debt_delta)
                    verdict = route_debt_verdict(
                        quality_status=quality_status,
                        latency_status=latency_status,
                        cpu_debt_status=cpu_debt_status,
                        route_debt_status=route_status_delta,
                    )
                    if verdict not in RECOMMENDED_ROUTE_DEBT_VERDICTS:
                        continue

                    row = {
                        "source": source,
                        **dict(zip(ROUTE_DEBT_GROUP_COLUMNS, key, strict=True)),
                        "candidate_wave_dilations": candidate_wave,
                        "baseline_wave_dilations": baseline_wave,
                        "candidate_runs": str(candidate.get("runs", "-")),
                        "baseline_runs": str(baseline.get("runs", "-")),
                        "candidate_final_nll": str(
                            candidate.get("final_nll_mean", "-")
                        ),
                        "baseline_final_nll": str(baseline.get("final_nll_mean", "-")),
                        "final_nll_delta": fmt_delta(final_nll_delta),
                        "candidate_best_nll": str(candidate.get("best_nll_mean", "-")),
                        "baseline_best_nll": str(baseline.get("best_nll_mean", "-")),
                        "best_nll_delta": fmt_delta(best_nll_delta),
                        "candidate_final_vs_bigram": str(
                            candidate.get("final_vs_bigram_mean", "-")
                        ),
                        "baseline_final_vs_bigram": str(
                            baseline.get("final_vs_bigram_mean", "-")
                        ),
                        "final_vs_bigram_delta": fmt_delta(bigram_delta),
                        "candidate_trace_step_ms": str(
                            candidate.get("trace_step_ms_mean_mean", "-")
                        ),
                        "baseline_trace_step_ms": str(
                            baseline.get("trace_step_ms_mean_mean", "-")
                        ),
                        "trace_step_ms_delta": fmt_delta(step_ms_delta),
                        "trace_step_ms_ratio": fmt_ratio(
                            candidate_step_ms, baseline_step_ms
                        ),
                        "candidate_cpu_debt": str(
                            candidate.get("cpu_debt_ops_mean", "-")
                        ),
                        "baseline_cpu_debt": str(baseline.get("cpu_debt_ops_mean", "-")),
                        "cpu_debt_delta": fmt_delta(cpu_debt_delta),
                        "cpu_debt_ratio": fmt_ratio(
                            candidate_cpu_debt, baseline_cpu_debt
                        ),
                        "candidate_route_debt": str(
                            candidate.get("coherence_route_debt_mean", "-")
                        ),
                        "baseline_route_debt": str(
                            baseline.get("coherence_route_debt_mean", "-")
                        ),
                        "route_debt_delta": fmt_delta(route_debt_delta),
                        "route_debt_ratio": fmt_ratio(
                            candidate_route_debt, baseline_route_debt
                        ),
                        "quality_status": quality_status,
                        "latency_status": latency_status,
                        "cpu_debt_status": cpu_debt_status,
                        "route_debt_status": route_status_delta,
                        "route_debt_verdict": verdict,
                        "candidate_coherence_route_status": str(
                            candidate.get("coherence_route_status", "-")
                        ),
                        "baseline_coherence_route_status": str(
                            baseline.get("coherence_route_status", "-")
                        ),
                        "candidate_route_status": route_status(candidate),
                        "baseline_route_status": route_status(baseline),
                    }
                    candidates.append(
                        (
                            (
                                float(verdict_rank.get(verdict, 99)),
                                ratio_for_sort(
                                    candidate_route_debt, baseline_route_debt
                                ),
                                ratio_for_sort(
                                    candidate_cpu_debt, baseline_cpu_debt
                                ),
                                ratio_for_sort(candidate_step_ms, baseline_step_ms),
                                final_nll_delta
                                if final_nll_delta is not None
                                else float("inf"),
                                best_nll_delta
                                if best_nll_delta is not None
                                else float("inf"),
                                str(source),
                            ),
                            row,
                        )
                    )

    candidates.sort(key=lambda item: item[0])
    selected = [row for _, row in candidates[: max(limit, 0)]]
    for rank, row in enumerate(selected, start=1):
        row["rank"] = str(rank)
        verdict = str(row.get("route_debt_verdict", "-"))
        row["recommendation"] = RECOMMENDED_ROUTE_DEBT_VERDICTS.get(verdict, "-")
    return selected


def route_debt_recommendation_summary(
    recommendations: list[dict[str, str]],
    *,
    fail_on_decisions: list[str] | None = None,
) -> dict[str, str]:
    forbidden_decisions = fail_on_decisions or []
    if not recommendations:
        decision = "no_route_debt_recommendation"
        return {
            "decision": decision,
            "failed": str(decision in set(forbidden_decisions)).lower(),
            "fail_on_decisions": ",".join(forbidden_decisions),
            "recommendation_rows": "0",
            "top_recommendation": "-",
            "top_candidate_wave_dilations": "-",
            "top_baseline_wave_dilations": "-",
            "top_quality_status": "-",
            "top_final_nll_delta": "-",
            "top_route_debt_ratio": "-",
            "top_cpu_debt_ratio": "-",
            "top_trace_step_ms_ratio": "-",
        }
    top = recommendations[0]
    decision = "promote_lite_wave"
    return {
        "decision": decision,
        "failed": str(decision in set(forbidden_decisions)).lower(),
        "fail_on_decisions": ",".join(forbidden_decisions),
        "recommendation_rows": str(len(recommendations)),
        "top_recommendation": str(top.get("recommendation", "-")),
        "top_candidate_wave_dilations": str(
            top.get("candidate_wave_dilations", "-")
        ),
        "top_baseline_wave_dilations": str(top.get("baseline_wave_dilations", "-")),
        "top_quality_status": str(top.get("quality_status", "-")),
        "top_final_nll_delta": str(top.get("final_nll_delta", "-")),
        "top_route_debt_ratio": str(top.get("route_debt_ratio", "-")),
        "top_cpu_debt_ratio": str(top.get("cpu_debt_ratio", "-")),
        "top_trace_step_ms_ratio": str(top.get("trace_step_ms_ratio", "-")),
    }


def summarize_rows(
    payload: dict[str, Any],
    *,
    limit: int,
    route_clean_only: bool,
    prefer_clean_route: bool,
    sort_metric: str = DEFAULT_SORT_METRIC,
    source: str = "-",
) -> list[dict[str, str]]:
    candidates = summarize_candidates(
        payload,
        source=source,
        route_clean_only=route_clean_only,
        prefer_clean_route=prefer_clean_route,
        sort_metric=sort_metric,
    )
    return ranked_summary_rows(candidates, limit=limit)


def sort_metric_key(row: dict[str, Any], sort_metric: str) -> float:
    column = SORT_METRIC_COLUMNS.get(sort_metric, SORT_METRIC_COLUMNS[DEFAULT_SORT_METRIC])
    value = parse_number_cell(row.get(column))
    if value is None:
        return float("inf")
    if sort_metric in HIGHER_IS_BETTER_SORT_METRICS:
        return -value
    return value


def summarize_candidates(
    payload: dict[str, Any],
    *,
    source: str,
    route_clean_only: bool,
    prefer_clean_route: bool,
    sort_metric: str,
) -> list[tuple[tuple[float, ...], dict[str, str]]]:
    candidates: list[tuple[tuple[float, ...], dict[str, str]]] = []
    for row in source_rows(payload):
        status = route_status(row)
        if route_clean_only and status != "clean_route":
            continue
        final_nll = parse_number_cell(row.get("final_nll_mean"))
        best_nll = parse_number_cell(row.get("best_nll_mean"))
        cpu_debt = parse_number_cell(row.get("cpu_debt_ops_mean"))
        sort_key = (
            float(route_penalty_for_row(row, status)) if prefer_clean_route else 0.0,
            sort_metric_key(row, sort_metric),
            final_nll if final_nll is not None else float("inf"),
            best_nll if best_nll is not None else float("inf"),
            cpu_debt if cpu_debt is not None else float("inf"),
        )
        summary = {
            header: str(row.get(header, "-"))
            for header in SUMMARY_HEADERS
            if header not in {"rank", "source"}
        }
        summary["source"] = source
        summary["route_status"] = status
        candidates.append((sort_key, summary))
    return candidates


def ranked_summary_rows(
    candidates: list[tuple[tuple[float, ...], dict[str, str]]],
    *,
    limit: int,
) -> list[dict[str, str]]:

    candidates.sort(key=lambda item: item[0])
    selected = [row for _, row in candidates[: max(limit, 0)]]
    for idx, row in enumerate(selected, start=1):
        row["rank"] = str(idx)
    return selected


def summarize_compare_payloads(
    payloads: list[tuple[str, dict[str, Any]]],
    *,
    limit: int,
    route_clean_only: bool,
    prefer_clean_route: bool,
    sort_metric: str = DEFAULT_SORT_METRIC,
) -> list[dict[str, str]]:
    candidates: list[tuple[tuple[float, ...], dict[str, str]]] = []
    for source, payload in payloads:
        candidates.extend(
            summarize_candidates(
                payload,
                source=source,
                route_clean_only=route_clean_only,
                prefer_clean_route=prefer_clean_route,
                sort_metric=sort_metric,
            )
        )
    return ranked_summary_rows(candidates, limit=limit)


def route_status_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = {header: 0 for header in ROUTE_COUNT_HEADERS if header != "scope"}
    counts["rows"] = len(rows)
    for row in rows:
        status = row.get("route_status", "no_scan_route")
        if status not in counts:
            counts[status] = 0
        counts[status] += 1
    return counts


def route_status_counts_for_payloads(payloads: list[tuple[str, dict[str, Any]]]) -> dict[str, int]:
    counts = {header: 0 for header in ROUTE_COUNT_HEADERS if header != "scope"}
    for _, payload in payloads:
        for row in source_rows(payload):
            status = route_status(row)
            counts["rows"] += 1
            if status not in counts:
                counts[status] = 0
            counts[status] += 1
    return counts


def parse_csv_filters(raw_values: list[str] | None) -> list[str]:
    values: list[str] = []
    for raw in raw_values or []:
        for item in raw.split(","):
            value = item.strip()
            if value:
                values.append(value)
    return values


def parse_route_status_filters(raw_values: list[str] | None) -> list[str]:
    return parse_csv_filters(raw_values)


def route_status_gate_failures(
    counts: dict[str, int],
    forbidden_statuses: list[str],
) -> dict[str, int]:
    return {
        status: counts.get(status, 0)
        for status in forbidden_statuses
        if counts.get(status, 0) > 0
    }


def paired_recurrent_gate_failures(
    pairs: list[dict[str, Any]],
    forbidden_quality_statuses: list[str],
    forbidden_efficiency_verdicts: list[str],
) -> dict[str, Any]:
    quality_counts: dict[str, int] = {}
    verdict_counts: dict[str, int] = {}
    failed_pairs: list[dict[str, Any]] = []
    forbidden_quality = set(forbidden_quality_statuses)
    forbidden_verdicts = set(forbidden_efficiency_verdicts)
    summary_fields = [
        "source",
        "backend",
        "head_prior",
        "head_resid",
        "char_feature",
        "mode",
        "steps",
        "hidden",
        "embed_dim",
        "epochs",
        "batches",
        "batch",
        "eval_samples",
        "lr",
        "candidate_recurrent",
        "baseline_recurrent",
        "final_nll_delta",
        "final_vs_bigram_delta",
        "trace_step_ms_delta",
        "cpu_debt_delta",
        "quality_status",
        "efficiency_verdict",
    ]
    for pair in pairs:
        triggers: list[str] = []
        quality_status = str(pair.get("quality_status", ""))
        verdict = str(pair.get("efficiency_verdict", ""))
        if quality_status in forbidden_quality:
            quality_counts[quality_status] = quality_counts.get(quality_status, 0) + 1
            triggers.append(f"quality_status:{quality_status}")
        if verdict in forbidden_verdicts:
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            triggers.append(f"efficiency_verdict:{verdict}")
        if triggers:
            failed_pair = {field: pair.get(field, "-") for field in summary_fields}
            failed_pair["failures"] = triggers
            failed_pairs.append(failed_pair)
    return {
        "quality_statuses": quality_counts,
        "efficiency_verdicts": verdict_counts,
        "pairs": failed_pairs,
    }


def paired_recurrent_gate_failed(failures: dict[str, Any]) -> bool:
    pairs = failures.get("pairs")
    return isinstance(pairs, list) and bool(pairs)


def route_counts_table(counts_by_scope: list[tuple[str, dict[str, int]]]) -> str:
    lines = [
        "| " + " | ".join(ROUTE_COUNT_HEADERS) + " |",
        "| " + " | ".join("---" for _ in ROUTE_COUNT_HEADERS) + " |",
    ]
    for scope, counts in counts_by_scope:
        lines.append(
            "| "
            + " | ".join(
                md_cell(scope) if header == "scope" else str(counts.get(header, 0))
                for header in ROUTE_COUNT_HEADERS
            )
            + " |"
        )
    return "\n".join(lines)


def route_counts_report(
    rows: list[dict[str, str]],
    *,
    all_counts: dict[str, int] | None = None,
) -> list[tuple[str, dict[str, int]]]:
    selected_counts = route_status_counts(rows)
    if all_counts is None:
        return [("selected", selected_counts)]
    return [("all_candidates", all_counts), ("selected", selected_counts)]


def route_counts_table_for_rows(rows: list[dict[str, str]]) -> str:
    return route_counts_table(route_counts_report(rows))


def legacy_route_counts_table(counts: dict[str, int]) -> str:
    return "\n".join(
        [
            "| "
            + " | ".join(header for header in ROUTE_COUNT_HEADERS if header != "scope")
            + " |",
            "| "
            + " | ".join("---" for header in ROUTE_COUNT_HEADERS if header != "scope")
            + " |",
            "| "
            + " | ".join(
                str(counts.get(header, 0)) for header in ROUTE_COUNT_HEADERS if header != "scope"
            )
            + " |",
        ]
    )


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(SUMMARY_HEADERS) + " |",
        "| " + " | ".join("---" for _ in SUMMARY_HEADERS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(md_cell(row.get(header, "-")) for header in SUMMARY_HEADERS) + " |")
    return "\n".join(lines)


def paired_delta_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(PAIR_DELTA_HEADERS) + " |",
        "| " + " | ".join("---" for _ in PAIR_DELTA_HEADERS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(md_cell(row.get(header, "-")) for header in PAIR_DELTA_HEADERS)
            + " |"
        )
    return "\n".join(lines)


def paired_recommendation_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(PAIR_RECOMMENDATION_HEADERS) + " |",
        "| " + " | ".join("---" for _ in PAIR_RECOMMENDATION_HEADERS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_cell(row.get(header, "-")) for header in PAIR_RECOMMENDATION_HEADERS
            )
            + " |"
        )
    return "\n".join(lines)


def bigram_guard_delta_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(BIGRAM_GUARD_DELTA_HEADERS) + " |",
        "| " + " | ".join("---" for _ in BIGRAM_GUARD_DELTA_HEADERS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_cell(row.get(header, "-")) for header in BIGRAM_GUARD_DELTA_HEADERS
            )
            + " |"
        )
    return "\n".join(lines)


def bigram_guard_recommendation_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(BIGRAM_GUARD_RECOMMENDATION_HEADERS) + " |",
        "| " + " | ".join("---" for _ in BIGRAM_GUARD_RECOMMENDATION_HEADERS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_cell(row.get(header, "-"))
                for header in BIGRAM_GUARD_RECOMMENDATION_HEADERS
            )
            + " |"
        )
    return "\n".join(lines)


def bigram_rank_guard_delta_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(BIGRAM_RANK_GUARD_DELTA_HEADERS) + " |",
        "| " + " | ".join("---" for _ in BIGRAM_RANK_GUARD_DELTA_HEADERS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_cell(row.get(header, "-"))
                for header in BIGRAM_RANK_GUARD_DELTA_HEADERS
            )
            + " |"
        )
    return "\n".join(lines)


def bigram_rank_guard_recommendation_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(BIGRAM_RANK_GUARD_RECOMMENDATION_HEADERS) + " |",
        "| "
        + " | ".join("---" for _ in BIGRAM_RANK_GUARD_RECOMMENDATION_HEADERS)
        + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_cell(row.get(header, "-"))
                for header in BIGRAM_RANK_GUARD_RECOMMENDATION_HEADERS
            )
            + " |"
        )
    return "\n".join(lines)


def bigram_rank_guard_seed_delta_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(BIGRAM_RANK_GUARD_SEED_DELTA_HEADERS) + " |",
        "| " + " | ".join("---" for _ in BIGRAM_RANK_GUARD_SEED_DELTA_HEADERS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_cell(row.get(header, "-"))
                for header in BIGRAM_RANK_GUARD_SEED_DELTA_HEADERS
            )
            + " |"
        )
    return "\n".join(lines)


def bigram_rank_guard_stability_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(BIGRAM_RANK_GUARD_STABILITY_HEADERS) + " |",
        "| " + " | ".join("---" for _ in BIGRAM_RANK_GUARD_STABILITY_HEADERS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_cell(row.get(header, "-"))
                for header in BIGRAM_RANK_GUARD_STABILITY_HEADERS
            )
            + " |"
        )
    return "\n".join(lines)


def bigram_rank_band_delta_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(BIGRAM_RANK_BAND_DELTA_HEADERS) + " |",
        "| " + " | ".join("---" for _ in BIGRAM_RANK_BAND_DELTA_HEADERS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_cell(row.get(header, "-"))
                for header in BIGRAM_RANK_BAND_DELTA_HEADERS
            )
            + " |"
        )
    return "\n".join(lines)


def bigram_rank_band_recommendation_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(BIGRAM_RANK_BAND_RECOMMENDATION_HEADERS) + " |",
        "| "
        + " | ".join("---" for _ in BIGRAM_RANK_BAND_RECOMMENDATION_HEADERS)
        + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_cell(row.get(header, "-"))
                for header in BIGRAM_RANK_BAND_RECOMMENDATION_HEADERS
            )
            + " |"
        )
    return "\n".join(lines)


def bigram_rank_band_seed_delta_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(BIGRAM_RANK_BAND_SEED_DELTA_HEADERS) + " |",
        "| " + " | ".join("---" for _ in BIGRAM_RANK_BAND_SEED_DELTA_HEADERS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_cell(row.get(header, "-"))
                for header in BIGRAM_RANK_BAND_SEED_DELTA_HEADERS
            )
            + " |"
        )
    return "\n".join(lines)


def bigram_rank_band_stability_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(BIGRAM_RANK_BAND_STABILITY_HEADERS) + " |",
        "| " + " | ".join("---" for _ in BIGRAM_RANK_BAND_STABILITY_HEADERS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_cell(row.get(header, "-"))
                for header in BIGRAM_RANK_BAND_STABILITY_HEADERS
            )
            + " |"
        )
    return "\n".join(lines)


def bigram_rank_min_delta_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(BIGRAM_RANK_MIN_DELTA_HEADERS) + " |",
        "| " + " | ".join("---" for _ in BIGRAM_RANK_MIN_DELTA_HEADERS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_cell(row.get(header, "-"))
                for header in BIGRAM_RANK_MIN_DELTA_HEADERS
            )
            + " |"
        )
    return "\n".join(lines)


def bigram_rank_min_recommendation_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(BIGRAM_RANK_MIN_RECOMMENDATION_HEADERS) + " |",
        "| "
        + " | ".join("---" for _ in BIGRAM_RANK_MIN_RECOMMENDATION_HEADERS)
        + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_cell(row.get(header, "-"))
                for header in BIGRAM_RANK_MIN_RECOMMENDATION_HEADERS
            )
            + " |"
        )
    return "\n".join(lines)


def bigram_rank_min_seed_delta_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(BIGRAM_RANK_MIN_SEED_DELTA_HEADERS) + " |",
        "| " + " | ".join("---" for _ in BIGRAM_RANK_MIN_SEED_DELTA_HEADERS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_cell(row.get(header, "-"))
                for header in BIGRAM_RANK_MIN_SEED_DELTA_HEADERS
            )
            + " |"
        )
    return "\n".join(lines)


def bigram_rank_min_stability_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(BIGRAM_RANK_MIN_STABILITY_HEADERS) + " |",
        "| " + " | ".join("---" for _ in BIGRAM_RANK_MIN_STABILITY_HEADERS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_cell(row.get(header, "-"))
                for header in BIGRAM_RANK_MIN_STABILITY_HEADERS
            )
            + " |"
        )
    return "\n".join(lines)


def bigram_rank_min_stable_recommendation_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(BIGRAM_RANK_MIN_STABLE_RECOMMENDATION_HEADERS) + " |",
        "| "
        + " | ".join("---" for _ in BIGRAM_RANK_MIN_STABLE_RECOMMENDATION_HEADERS)
        + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_cell(row.get(header, "-"))
                for header in BIGRAM_RANK_MIN_STABLE_RECOMMENDATION_HEADERS
            )
            + " |"
        )
    return "\n".join(lines)


def bigram_rank_min_promotion_gate_table(gate: dict[str, str]) -> str:
    lines = [
        "| " + " | ".join(BIGRAM_RANK_MIN_PROMOTION_GATE_HEADERS) + " |",
        "| "
        + " | ".join("---" for _ in BIGRAM_RANK_MIN_PROMOTION_GATE_HEADERS)
        + " |",
        "| "
        + " | ".join(
            md_cell(gate.get(header, "-"))
            for header in BIGRAM_RANK_MIN_PROMOTION_GATE_HEADERS
        )
        + " |",
    ]
    return "\n".join(lines)


def bigram_soft_guard_delta_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(BIGRAM_SOFT_GUARD_DELTA_HEADERS) + " |",
        "| " + " | ".join("---" for _ in BIGRAM_SOFT_GUARD_DELTA_HEADERS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_cell(row.get(header, "-"))
                for header in BIGRAM_SOFT_GUARD_DELTA_HEADERS
            )
            + " |"
        )
    return "\n".join(lines)


def bigram_soft_guard_recommendation_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(BIGRAM_SOFT_GUARD_RECOMMENDATION_HEADERS) + " |",
        "| "
        + " | ".join("---" for _ in BIGRAM_SOFT_GUARD_RECOMMENDATION_HEADERS)
        + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_cell(row.get(header, "-"))
                for header in BIGRAM_SOFT_GUARD_RECOMMENDATION_HEADERS
            )
            + " |"
        )
    return "\n".join(lines)


def bigram_soft_guard_seed_delta_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(BIGRAM_SOFT_GUARD_SEED_DELTA_HEADERS) + " |",
        "| " + " | ".join("---" for _ in BIGRAM_SOFT_GUARD_SEED_DELTA_HEADERS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_cell(row.get(header, "-"))
                for header in BIGRAM_SOFT_GUARD_SEED_DELTA_HEADERS
            )
            + " |"
        )
    return "\n".join(lines)


def bigram_soft_guard_stability_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(BIGRAM_SOFT_GUARD_STABILITY_HEADERS) + " |",
        "| " + " | ".join("---" for _ in BIGRAM_SOFT_GUARD_STABILITY_HEADERS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_cell(row.get(header, "-"))
                for header in BIGRAM_SOFT_GUARD_STABILITY_HEADERS
            )
            + " |"
        )
    return "\n".join(lines)


def baseline_difficulty_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(BASELINE_DIFFICULTY_HEADERS) + " |",
        "| " + " | ".join("---" for _ in BASELINE_DIFFICULTY_HEADERS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_cell(row.get(header, "-")) for header in BASELINE_DIFFICULTY_HEADERS
            )
            + " |"
        )
    return "\n".join(lines)


def learning_scoreboard_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(LEARNING_SCOREBOARD_HEADERS) + " |",
        "| " + " | ".join("---" for _ in LEARNING_SCOREBOARD_HEADERS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_cell(row.get(header, "-")) for header in LEARNING_SCOREBOARD_HEADERS
            )
            + " |"
        )
    return "\n".join(lines)


def route_debt_recommendation_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(ROUTE_DEBT_RECOMMENDATION_HEADERS) + " |",
        "| "
        + " | ".join("---" for _ in ROUTE_DEBT_RECOMMENDATION_HEADERS)
        + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_cell(row.get(header, "-"))
                for header in ROUTE_DEBT_RECOMMENDATION_HEADERS
            )
            + " |"
        )
    return "\n".join(lines)


def route_debt_summary_table(summary: dict[str, str]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(ROUTE_DEBT_SUMMARY_HEADERS) + " |",
            "| " + " | ".join("---" for _ in ROUTE_DEBT_SUMMARY_HEADERS) + " |",
            "| "
            + " | ".join(
                md_cell(summary.get(header, "-"))
                for header in ROUTE_DEBT_SUMMARY_HEADERS
            )
            + " |",
        ]
    )


def markdown_report(
    rows: list[dict[str, str]],
    *,
    all_counts: dict[str, int] | None = None,
    paired_deltas: list[dict[str, str]] | None = None,
    paired_recommendations: list[dict[str, str]] | None = None,
    bigram_guard_deltas: list[dict[str, str]] | None = None,
    bigram_guard_recommendations: list[dict[str, str]] | None = None,
    bigram_rank_guard_deltas: list[dict[str, str]] | None = None,
    bigram_rank_guard_recommendations: list[dict[str, str]] | None = None,
    bigram_rank_guard_seed_deltas: list[dict[str, str]] | None = None,
    bigram_rank_guard_stability: list[dict[str, str]] | None = None,
    bigram_rank_band_deltas: list[dict[str, str]] | None = None,
    bigram_rank_band_recommendations: list[dict[str, str]] | None = None,
    bigram_rank_band_seed_deltas: list[dict[str, str]] | None = None,
    bigram_rank_band_stability: list[dict[str, str]] | None = None,
    bigram_rank_min_deltas: list[dict[str, str]] | None = None,
    bigram_rank_min_recommendations: list[dict[str, str]] | None = None,
    bigram_rank_min_seed_deltas: list[dict[str, str]] | None = None,
    bigram_rank_min_stability: list[dict[str, str]] | None = None,
    bigram_rank_min_stable_recommendations: list[dict[str, str]] | None = None,
    bigram_rank_min_promotion_gate: dict[str, str] | None = None,
    bigram_soft_guard_deltas: list[dict[str, str]] | None = None,
    bigram_soft_guard_recommendations: list[dict[str, str]] | None = None,
    bigram_soft_guard_seed_deltas: list[dict[str, str]] | None = None,
    bigram_soft_guard_stability: list[dict[str, str]] | None = None,
    baseline_difficulty: list[dict[str, str]] | None = None,
    learning_scoreboard: list[dict[str, str]] | None = None,
    route_debt_recommendations: list[dict[str, str]] | None = None,
    route_debt_summary: dict[str, str] | None = None,
) -> str:
    sections = [
        "## Route Status Counts",
        route_counts_table(route_counts_report(rows, all_counts=all_counts)),
    ]
    if learning_scoreboard:
        sections.extend(
            [
                "## Learning Scoreboard",
                learning_scoreboard_table(learning_scoreboard),
            ]
        )
    if route_debt_recommendations:
        sections.extend(
            [
                "## Route Debt Decision",
                route_debt_summary_table(
                    route_debt_summary
                    or route_debt_recommendation_summary(route_debt_recommendations)
                ),
                "## Route Debt Recommendations",
                route_debt_recommendation_table(route_debt_recommendations),
            ]
        )
    if paired_recommendations:
        sections.extend(
            [
                "## Paired Recurrent Recommendations",
                paired_recommendation_table(paired_recommendations),
            ]
        )
    if bigram_guard_recommendations:
        sections.extend(
            [
                "## Bigram Guard Recommendations",
                bigram_guard_recommendation_table(bigram_guard_recommendations),
            ]
        )
    if bigram_rank_guard_recommendations:
        sections.extend(
            [
                "## Bigram Rank Guard Recommendations",
                bigram_rank_guard_recommendation_table(
                    bigram_rank_guard_recommendations
                ),
            ]
        )
    if bigram_rank_band_recommendations:
        sections.extend(
            [
                "## Bigram Rank Band Recommendations",
                bigram_rank_band_recommendation_table(
                    bigram_rank_band_recommendations
                ),
            ]
        )
    if bigram_rank_min_recommendations:
        sections.extend(
            [
                "## Bigram Rank Min Recommendations",
                bigram_rank_min_recommendation_table(
                    bigram_rank_min_recommendations
                ),
            ]
        )
    if bigram_rank_min_stable_recommendations:
        sections.extend(
            [
                "## Bigram Rank Min Stable Recommendations",
                bigram_rank_min_stable_recommendation_table(
                    bigram_rank_min_stable_recommendations
                ),
            ]
        )
    if bigram_rank_min_promotion_gate:
        sections.extend(
            [
                "## Bigram Rank Min Promotion Gate",
                bigram_rank_min_promotion_gate_table(
                    bigram_rank_min_promotion_gate
                ),
            ]
        )
    if bigram_soft_guard_recommendations:
        sections.extend(
            [
                "## Bigram Soft Guard Recommendations",
                bigram_soft_guard_recommendation_table(
                    bigram_soft_guard_recommendations
                ),
            ]
        )
    if baseline_difficulty:
        sections.extend(
            [
                "## Baseline Difficulty Hotspots",
                baseline_difficulty_table(baseline_difficulty),
            ]
        )
    if paired_deltas:
        sections.extend(["## Paired Recurrent Deltas", paired_delta_table(paired_deltas)])
    if bigram_guard_deltas:
        sections.extend(
            ["## Bigram Guard Deltas", bigram_guard_delta_table(bigram_guard_deltas)]
        )
    if bigram_rank_guard_deltas:
        sections.extend(
            [
                "## Bigram Rank Guard Deltas",
                bigram_rank_guard_delta_table(bigram_rank_guard_deltas),
            ]
        )
    if bigram_rank_band_deltas:
        sections.extend(
            [
                "## Bigram Rank Band Deltas",
                bigram_rank_band_delta_table(bigram_rank_band_deltas),
            ]
        )
    if bigram_rank_min_deltas:
        sections.extend(
            [
                "## Bigram Rank Min Deltas",
                bigram_rank_min_delta_table(bigram_rank_min_deltas),
            ]
        )
    if bigram_soft_guard_deltas:
        sections.extend(
            [
                "## Bigram Soft Guard Deltas",
                bigram_soft_guard_delta_table(bigram_soft_guard_deltas),
            ]
        )
    if bigram_rank_guard_stability:
        sections.extend(
            [
                "## Bigram Rank Guard Stability",
                bigram_rank_guard_stability_table(bigram_rank_guard_stability),
            ]
        )
    if bigram_rank_band_stability:
        sections.extend(
            [
                "## Bigram Rank Band Stability",
                bigram_rank_band_stability_table(bigram_rank_band_stability),
            ]
        )
    if bigram_rank_min_stability:
        sections.extend(
            [
                "## Bigram Rank Min Stability",
                bigram_rank_min_stability_table(bigram_rank_min_stability),
            ]
        )
    if bigram_soft_guard_stability:
        sections.extend(
            [
                "## Bigram Soft Guard Stability",
                bigram_soft_guard_stability_table(bigram_soft_guard_stability),
            ]
        )
    if bigram_rank_guard_seed_deltas:
        sections.extend(
            [
                "## Bigram Rank Guard Seed Deltas",
                bigram_rank_guard_seed_delta_table(bigram_rank_guard_seed_deltas),
            ]
        )
    if bigram_rank_band_seed_deltas:
        sections.extend(
            [
                "## Bigram Rank Band Seed Deltas",
                bigram_rank_band_seed_delta_table(bigram_rank_band_seed_deltas),
            ]
        )
    if bigram_rank_min_seed_deltas:
        sections.extend(
            [
                "## Bigram Rank Min Seed Deltas",
                bigram_rank_min_seed_delta_table(bigram_rank_min_seed_deltas),
            ]
        )
    if bigram_soft_guard_seed_deltas:
        sections.extend(
            [
                "## Bigram Soft Guard Seed Deltas",
                bigram_soft_guard_seed_delta_table(bigram_soft_guard_seed_deltas),
            ]
        )
    sections.extend(["## Compare Summary", markdown_table(rows)])
    return "\n\n".join(sections)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "compare_json",
        type=Path,
        nargs="+",
        help="compare.json files or sweep directories containing compare.json",
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--route-clean-only", action="store_true")
    parser.add_argument("--prefer-clean-route", action="store_true")
    parser.add_argument(
        "--sort-metric",
        choices=sorted(SORT_METRIC_COLUMNS),
        default=DEFAULT_SORT_METRIC,
        help="primary metric used after optional route preference",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="search directories recursively for compare.json files",
    )
    parser.add_argument(
        "--fail-on-route-status",
        action="append",
        default=[],
        help=(
            "fail when all-candidate route counts include this status; "
            "may be repeated or comma-separated"
        ),
    )
    parser.add_argument(
        "--fail-on-paired-quality-status",
        action="append",
        default=[],
        help=(
            "fail when paired recurrent deltas include this quality status; "
            "may be repeated or comma-separated"
        ),
    )
    parser.add_argument(
        "--fail-on-efficiency-verdict",
        action="append",
        default=[],
        help=(
            "fail when paired recurrent deltas include this efficiency verdict; "
            "may be repeated or comma-separated"
        ),
    )
    parser.add_argument(
        "--merge-evidence-sources",
        action="store_true",
        help=(
            "merge compatible rank-min seed stability groups across compare inputs "
            "for promotion evidence"
        ),
    )
    parser.add_argument(
        "--fail-on-rank-min-promotion-decision",
        action="append",
        default=[],
        help=(
            "fail when the rank-min promotion gate has this decision; "
            "may be repeated or comma-separated"
        ),
    )
    parser.add_argument(
        "--fail-on-route-debt-decision",
        action="append",
        default=[],
        help=(
            "fail when the route-debt recommendation summary has this decision; "
            "may be repeated or comma-separated"
        ),
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    forbidden_statuses = parse_route_status_filters(args.fail_on_route_status)
    unknown_statuses = sorted(set(forbidden_statuses) - set(VALID_ROUTE_STATUSES))
    if unknown_statuses:
        parser.error(
            "unknown --fail-on-route-status value(s): "
            + ", ".join(unknown_statuses)
            + f" (expected one of {', '.join(VALID_ROUTE_STATUSES)})"
        )
    forbidden_pair_quality_statuses = parse_csv_filters(
        args.fail_on_paired_quality_status
    )
    unknown_quality_statuses = sorted(
        set(forbidden_pair_quality_statuses) - set(VALID_PAIR_QUALITY_STATUSES)
    )
    if unknown_quality_statuses:
        parser.error(
            "unknown --fail-on-paired-quality-status value(s): "
            + ", ".join(unknown_quality_statuses)
            + f" (expected one of {', '.join(VALID_PAIR_QUALITY_STATUSES)})"
        )
    forbidden_efficiency_verdicts = parse_csv_filters(args.fail_on_efficiency_verdict)
    unknown_efficiency_verdicts = sorted(
        set(forbidden_efficiency_verdicts) - set(VALID_EFFICIENCY_VERDICTS)
    )
    if unknown_efficiency_verdicts:
        parser.error(
            "unknown --fail-on-efficiency-verdict value(s): "
            + ", ".join(unknown_efficiency_verdicts)
            + f" (expected one of {', '.join(VALID_EFFICIENCY_VERDICTS)})"
        )
    forbidden_rank_min_promotion_decisions = parse_csv_filters(
        args.fail_on_rank_min_promotion_decision
    )
    unknown_rank_min_promotion_decisions = sorted(
        set(forbidden_rank_min_promotion_decisions)
        - set(VALID_RANK_MIN_PROMOTION_DECISIONS)
    )
    if unknown_rank_min_promotion_decisions:
        parser.error(
            "unknown --fail-on-rank-min-promotion-decision value(s): "
            + ", ".join(unknown_rank_min_promotion_decisions)
            + f" (expected one of {', '.join(VALID_RANK_MIN_PROMOTION_DECISIONS)})"
        )
    forbidden_route_debt_decisions = parse_csv_filters(
        args.fail_on_route_debt_decision
    )
    unknown_route_debt_decisions = sorted(
        set(forbidden_route_debt_decisions) - set(VALID_ROUTE_DEBT_DECISIONS)
    )
    if unknown_route_debt_decisions:
        parser.error(
            "unknown --fail-on-route-debt-decision value(s): "
            + ", ".join(unknown_route_debt_decisions)
            + f" (expected one of {', '.join(VALID_ROUTE_DEBT_DECISIONS)})"
        )

    compare_paths = resolve_compare_paths(args.compare_json, recursive=args.recursive)
    payloads = [(str(path), read_json(path)) for path in compare_paths]
    all_route_status_counts = route_status_counts_for_payloads(payloads)
    recurrent_deltas = paired_recurrent_deltas(payloads)
    recurrent_recommendations = paired_recurrent_recommendations(
        recurrent_deltas,
        limit=args.limit,
    )
    bigram_guard_deltas = paired_bigram_guard_deltas(payloads)
    bigram_guard_recommendations = paired_bigram_guard_recommendations(
        bigram_guard_deltas,
        limit=args.limit,
    )
    bigram_rank_guard_deltas = paired_bigram_rank_guard_deltas(payloads)
    bigram_rank_guard_recommendations = paired_bigram_rank_guard_recommendations(
        bigram_rank_guard_deltas,
        limit=args.limit,
    )
    bigram_rank_guard_seed_deltas = paired_bigram_rank_guard_seed_deltas(payloads)
    bigram_rank_guard_stability = bigram_rank_guard_stability_rows(
        bigram_rank_guard_seed_deltas
    )
    bigram_rank_band_deltas = paired_bigram_rank_band_deltas(payloads)
    bigram_rank_band_recommendations = paired_bigram_rank_band_recommendations(
        bigram_rank_band_deltas,
        limit=args.limit,
    )
    bigram_rank_band_seed_deltas = paired_bigram_rank_band_seed_deltas(payloads)
    bigram_rank_band_stability = bigram_rank_band_stability_rows(
        bigram_rank_band_seed_deltas
    )
    bigram_rank_min_deltas = paired_bigram_rank_min_deltas(payloads)
    bigram_rank_min_recommendations = paired_bigram_rank_min_recommendations(
        bigram_rank_min_deltas,
        limit=args.limit,
    )
    bigram_rank_min_seed_deltas = paired_bigram_rank_min_seed_deltas(payloads)
    bigram_rank_min_stability = bigram_rank_min_stability_rows(
        bigram_rank_min_seed_deltas,
        merge_sources=args.merge_evidence_sources,
    )
    bigram_rank_min_stable_recs = (
        bigram_rank_min_stable_recommendations(
            bigram_rank_min_stability,
            limit=args.limit,
        )
    )
    bigram_rank_min_gate = bigram_rank_min_promotion_gate(
        bigram_rank_min_stability,
        bigram_rank_min_stable_recs,
        fail_on_decisions=forbidden_rank_min_promotion_decisions,
    )
    bigram_soft_guard_deltas = paired_bigram_soft_guard_deltas(payloads)
    bigram_soft_guard_recommendations = paired_bigram_soft_guard_recommendations(
        bigram_soft_guard_deltas,
        limit=args.limit,
    )
    bigram_soft_guard_seed_deltas = paired_bigram_soft_guard_seed_deltas(payloads)
    bigram_soft_guard_stability = bigram_soft_guard_stability_rows(
        bigram_soft_guard_seed_deltas
    )
    learning_scoreboard = learning_scoreboard_rows(payloads, limit=args.limit)
    route_debt_recs = route_debt_recommendations(payloads, limit=args.limit)
    route_debt_summary = route_debt_recommendation_summary(
        route_debt_recs,
        fail_on_decisions=forbidden_route_debt_decisions,
    )
    baseline_difficulty = baseline_difficulty_rows(payloads, limit=args.limit)
    rows = summarize_compare_payloads(
        payloads,
        limit=args.limit,
        route_clean_only=args.route_clean_only,
        prefer_clean_route=args.prefer_clean_route,
        sort_metric=args.sort_metric,
    )
    gate_failures = route_status_gate_failures(
        all_route_status_counts,
        forbidden_statuses,
    )
    paired_gate_failures = paired_recurrent_gate_failures(
        recurrent_deltas,
        forbidden_pair_quality_statuses,
        forbidden_efficiency_verdicts,
    )
    output = {
        "schema": "st.char_lm.compare_summary.v1",
        "source": str(compare_paths[0]) if len(compare_paths) == 1 else "-",
        "sources": [str(path) for path in compare_paths],
        "merge_evidence_sources": bool(args.merge_evidence_sources),
        "sort_metric": args.sort_metric,
        "route_status_counts": all_route_status_counts,
        "selected_route_status_counts": route_status_counts(rows),
        "paired_recurrent_deltas": recurrent_deltas,
        "paired_recurrent_recommendations": recurrent_recommendations,
        "bigram_guard_deltas": bigram_guard_deltas,
        "bigram_guard_recommendations": bigram_guard_recommendations,
        "bigram_rank_guard_deltas": bigram_rank_guard_deltas,
        "bigram_rank_guard_recommendations": bigram_rank_guard_recommendations,
        "bigram_rank_guard_seed_deltas": bigram_rank_guard_seed_deltas,
        "bigram_rank_guard_stability": bigram_rank_guard_stability,
        "bigram_rank_band_deltas": bigram_rank_band_deltas,
        "bigram_rank_band_recommendations": bigram_rank_band_recommendations,
        "bigram_rank_band_seed_deltas": bigram_rank_band_seed_deltas,
        "bigram_rank_band_stability": bigram_rank_band_stability,
        "bigram_rank_min_deltas": bigram_rank_min_deltas,
        "bigram_rank_min_recommendations": bigram_rank_min_recommendations,
        "bigram_rank_min_seed_deltas": bigram_rank_min_seed_deltas,
        "bigram_rank_min_stability": bigram_rank_min_stability,
        "bigram_rank_min_stable_recommendations": bigram_rank_min_stable_recs,
        "bigram_rank_min_promotion_gate": bigram_rank_min_gate,
        "bigram_soft_guard_deltas": bigram_soft_guard_deltas,
        "bigram_soft_guard_recommendations": bigram_soft_guard_recommendations,
        "bigram_soft_guard_seed_deltas": bigram_soft_guard_seed_deltas,
        "bigram_soft_guard_stability": bigram_soft_guard_stability,
        "learning_scoreboard_rows": learning_scoreboard,
        "route_debt_recommendation_summary": route_debt_summary,
        "route_debt_recommendations": route_debt_recs,
        "baseline_difficulty_rows": baseline_difficulty,
        "route_status_gate": {
            "fail_on": forbidden_statuses,
            "failures": gate_failures,
            "failed": bool(gate_failures),
        },
        "paired_recurrent_gate": {
            "fail_on_quality_statuses": forbidden_pair_quality_statuses,
            "fail_on_efficiency_verdicts": forbidden_efficiency_verdicts,
            "failures": paired_gate_failures,
            "failed": paired_recurrent_gate_failed(paired_gate_failures),
        },
        "rows": rows,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        markdown_report(
            rows,
            all_counts=all_route_status_counts,
            paired_deltas=recurrent_deltas,
            paired_recommendations=recurrent_recommendations,
            bigram_guard_deltas=bigram_guard_deltas,
            bigram_guard_recommendations=bigram_guard_recommendations,
            bigram_rank_guard_deltas=bigram_rank_guard_deltas,
            bigram_rank_guard_recommendations=bigram_rank_guard_recommendations,
            bigram_rank_guard_seed_deltas=bigram_rank_guard_seed_deltas,
            bigram_rank_guard_stability=bigram_rank_guard_stability,
            bigram_rank_band_deltas=bigram_rank_band_deltas,
            bigram_rank_band_recommendations=bigram_rank_band_recommendations,
            bigram_rank_band_seed_deltas=bigram_rank_band_seed_deltas,
            bigram_rank_band_stability=bigram_rank_band_stability,
            bigram_rank_min_deltas=bigram_rank_min_deltas,
            bigram_rank_min_recommendations=bigram_rank_min_recommendations,
            bigram_rank_min_seed_deltas=bigram_rank_min_seed_deltas,
            bigram_rank_min_stability=bigram_rank_min_stability,
            bigram_rank_min_stable_recommendations=bigram_rank_min_stable_recs,
            bigram_rank_min_promotion_gate=bigram_rank_min_gate,
            bigram_soft_guard_deltas=bigram_soft_guard_deltas,
            bigram_soft_guard_recommendations=bigram_soft_guard_recommendations,
            bigram_soft_guard_seed_deltas=bigram_soft_guard_seed_deltas,
            bigram_soft_guard_stability=bigram_soft_guard_stability,
            baseline_difficulty=baseline_difficulty,
            learning_scoreboard=learning_scoreboard,
            route_debt_recommendations=route_debt_recs,
            route_debt_summary=route_debt_summary,
        )
    )
    exit_code = 0
    if gate_failures:
        failure_text = ", ".join(
            f"{status}={count}" for status, count in sorted(gate_failures.items())
        )
        print(f"route status gate failed: {failure_text}", file=sys.stderr)
        exit_code = 1
    if paired_recurrent_gate_failed(paired_gate_failures):
        quality_text = ", ".join(
            f"{status}={count}"
            for status, count in sorted(
                paired_gate_failures["quality_statuses"].items()
            )
        )
        verdict_text = ", ".join(
            f"{verdict}={count}"
            for verdict, count in sorted(
                paired_gate_failures["efficiency_verdicts"].items()
            )
        )
        parts = [part for part in [quality_text, verdict_text] if part]
        print(
            "paired recurrent gate failed: " + "; ".join(parts),
            file=sys.stderr,
        )
        exit_code = 1
    if bigram_rank_min_gate.get("failed") == "true":
        print(
            "rank-min promotion gate failed: "
            + str(bigram_rank_min_gate.get("decision", "-")),
            file=sys.stderr,
        )
        exit_code = 1
    if route_debt_summary.get("failed") == "true":
        print(
            "route-debt decision gate failed: "
            + str(route_debt_summary.get("decision", "-")),
            file=sys.stderr,
        )
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
