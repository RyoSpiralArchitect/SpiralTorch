#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
# Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

"""Compare SpiralTorch char-LM model-zoo run summaries."""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
from typing import Any

from backend_sweep_meta import BACKEND_RESIDUAL_HEADERS, backend_residual_columns

REPO_ROOT = Path(__file__).resolve().parents[1]

TRACE_REPAIR_COLUMNS = [
    "repair_steps",
    "repair_max",
    "repair_last",
    "repair_pre_max",
]

TRACE_TIMING_COLUMNS = [
    "trace_step_ms_last",
    "trace_step_ms_mean",
    "trace_step_ms_max",
]

TRACE_OPTIM_COLUMNS = [
    "trace_update_l2",
    "trace_update_ratio",
    "trace_update_ratio_max",
    "trace_update_max_l2",
    "trace_update_max_ratio",
    "trace_zero_param_ratio",
    "trace_lr",
    "trace_state_lr",
    "trace_adapter_energy",
    "trace_adapter_curv",
    "trace_adapter_spin",
    "trace_sync_world",
    "trace_sync_buffers",
    "trace_sync_values",
]

DATA_GROUP_COLUMNS = [
    "data_label",
]

DATA_MEAN_COLUMNS = [
    "data_files",
    "train_tokens",
    "validation_tokens",
    "vocab_size",
]

AGGREGATE_GROUP_COLUMNS = [
    "arch",
    "recurrent",
    "backend",
    *DATA_GROUP_COLUMNS,
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
    "wave_dilations",
    "steps",
    "hidden",
    "embed_dim",
    "epochs",
    "batches",
    "batch",
    "eval_samples",
    "val_start",
    "lr",
    "lr_schedule",
    "lr_warmup",
    "lr_final_scale",
    "restored_best",
]

AGGREGATE_MEAN_COLUMNS = [
    *DATA_MEAN_COLUMNS,
    "final_lr",
    "best_lr",
    "val_start_actual",
    "final_windows",
    "unigram_windows",
    "bigram_windows",
    "rank_cov_windows",
    "rank_cov_unbounded",
    "rank_cov_band",
    "rank_cov_min",
    "rank_cov_guarded",
    "rank_cov_effective_band",
    "rank_cov_adaptive_fill_ratio",
    "rank_cov_filled",
    "rank_cov_zero_ratio",
    "rank_cov_mass",
    "rank_cov_band_ratio",
    "rank_cov_topk_ratio",
    "delta_nll",
    "unigram_nll",
    "bigram_nll",
    "best_nll",
    "final_vs_unigram",
    "final_vs_bigram",
    "best_vs_unigram",
    "best_vs_bigram",
    "final_logprob_lift",
    "final_rank_lift",
    "final_unigram_target_rank",
    "final_unigram_rank_debt",
    "final_bigram_logprob_lift",
    "final_bigram_rank_lift",
    "final_bigram_target_rank",
    "final_bigram_rank_debt",
    "final_kl_bigram",
    "trace_step_ms_mean",
    "trace_update_ratio",
    "cpu_debt_ops",
    "lstm_est_cpu_debt_ops",
    "coherence_route_debt",
    "lstm_est_gate_wgpu_ops",
    "lstm_est_bptt_wgpu_ops",
]

AGGREGATE_RAW_MEAN_COLUMNS = [
    "final_rank_lift_raw",
    "final_unigram_target_rank_raw",
    "final_unigram_rank_debt_raw",
    "final_bigram_rank_lift_raw",
    "final_bigram_target_rank_raw",
    "final_bigram_rank_debt_raw",
    "final_top5_bigram_overlap_raw",
]

AGGREGATE_PERCENT_COLUMNS = [
    "final_top5_bigram_overlap",
]

AGGREGATE_COUNT_COLUMNS = [
    "route_status",
    "lstm_scan_backend_counts",
    "lstm_scan_fallback_counts",
    "coherence_route_status",
    "coherence_route_status_counts",
    "coherence_route_counts",
]

COHERENCE_ROUTE_COLUMNS = [
    "coherence_route_status",
    "coherence_route_counts",
    "coherence_route_debt",
]

TOP_AGGREGATE_COLUMNS = [
    *AGGREGATE_GROUP_COLUMNS,
    "runs",
    "data_files_mean",
    "train_tokens_mean",
    "validation_tokens_mean",
    "vocab_size_mean",
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
    "cpu_debt_ops_mean",
    "lstm_est_cpu_debt_ops_mean",
    "route_status",
    "lstm_scan_backend_counts",
    "lstm_scan_fallback_counts",
    "coherence_route_status",
    "coherence_route_status_counts",
    "coherence_route_counts",
    "coherence_route_debt_mean",
]

BACKEND_COLUMNS = [
    "tensor_ops",
    "tensor_wgpu",
    "tensor_wgpu_dense",
    "tensor_cpu",
    "tensor_cpu_simd",
    "tensor_f64_cpu",
    "tensor_fallbacks",
]

BACKEND_RESIDUAL_COLUMNS = BACKEND_RESIDUAL_HEADERS

RUN_BACKEND_AUDIT_COLUMNS = [
    "backend_status",
    "backend_kernels",
    "backend_feature",
    "hip_real",
    "tensor_policy_matmul",
    "tensor_policy_prepacked",
    "tensor_policy_softmax",
    "tensor_policy_util",
    "rt_wgpu_initialized",
    "rt_wgpu_compiled",
    "rt_wgpu_ctx",
    "rt_wgpu_ready",
    "rt_wgpu_statuses",
    "rt_wgpu_shapes",
]

LEARNING_OP_COLUMNS = [
    "matmul_wgpu",
    "matmul_faer",
    "matmul_naive",
    "prepacked_wgpu",
    "prepacked_faer",
    "prepacked_naive",
    "softmax_wgpu",
    "softmax_cpu",
    "softmax_bwd_wgpu",
    "softmax_bwd_cpu",
    "coherence_scan_fwd_wgpu",
    "coherence_scan_fwd_cpu",
    "coherence_scan_bwd_wgpu",
    "coherence_scan_bwd_cpu",
    "psi_heatmap_summary_cpu",
    "zspace_semantic_dist_cpu",
    "zspace_semantic_dist_infer_cpu",
    "zspace_semantic_window_cpu",
    "zspace_semantic_window_hybrid",
    "zspace_semantic_window_energy_wgpu",
    "zspace_semantic_window_energy_cpu",
    "zspace_semantic_window_scale_wgpu",
    "zspace_semantic_window_scale_cpu",
    "zspace_semantic_window_control_cpu",
    "zspace_maxwell_summary_cpu",
    "zspace_semantic_fusion_cpu",
    "zspace_semantic_fusion_hybrid",
    "zspace_semantic_fusion_accum_wgpu",
    "zspace_semantic_fusion_accum_cpu",
    "zspace_semantic_fusion_scale_wgpu",
    "zspace_semantic_fusion_scale_cpu",
    "lawvere_guard_control_cpu",
    "biome_absorb_topos_cpu",
    "biome_renorm_control_cpu",
    "biome_canopy_hybrid",
    "biome_canopy_accum_wgpu",
    "biome_canopy_accum_cpu",
    "biome_canopy_normalise_wgpu",
    "biome_canopy_normalise_cpu",
    "biome_canopy_rewrite_topos_cpu",
    "desire_auto_probability_cpu",
    "desire_auto_hybrid",
    "desire_auto_sanitize_wgpu",
    "desire_auto_sanitize_cpu",
    "desire_auto_scale_wgpu",
    "desire_auto_scale_cpu",
    "desire_softmax_probability_cpu",
    "desire_softmax_hybrid",
    "desire_softmax_row_wgpu",
    "desire_softmax_row_cpu",
    "desire_softmax_exp_wgpu",
    "desire_softmax_exp_cpu",
    "desire_softmax_scale_wgpu",
    "desire_softmax_scale_cpu",
    "desire_norm_probability_cpu",
    "desire_norm_hybrid",
    "desire_norm_sanitize_wgpu",
    "desire_norm_sanitize_cpu",
    "desire_norm_scale_wgpu",
    "desire_norm_scale_cpu",
    "concept_diffusion_probability_cpu",
    "concept_diffusion_f64_cpu",
    "concept_diffusion_sum_f64_cpu",
    "concept_diffusion_precision_f64_cpu",
    "concept_diffusion_scale_f64_cpu",
    "sparse_kernel_probability_cpu",
    "sparse_kernel_hybrid",
    "sparse_kernel_scan_cpu",
    "sparse_kernel_sum_wgpu",
    "sparse_kernel_sum_cpu",
    "sparse_kernel_scale_wgpu",
    "sparse_kernel_scale_cpu",
    "semantic_bridge_cpu",
    "semantic_bridge_hybrid",
    "semantic_bridge_accum_wgpu",
    "semantic_bridge_accum_cpu",
    "semantic_bridge_scale_wgpu",
    "semantic_bridge_scale_cpu",
    "concept_hint_cpu",
    "concept_hint_hybrid",
    "concept_hint_sanitize_wgpu",
    "concept_hint_sanitize_cpu",
    "concept_hint_infer_cpu",
    "concept_hint_scale_wgpu",
    "concept_hint_scale_cpu",
    "gw_marginal_probability_cpu",
    "gw_marginal_hybrid",
    "gw_marginal_scan_cpu",
    "gw_marginal_sum_wgpu",
    "gw_marginal_sum_cpu",
    "gw_marginal_scale_wgpu",
    "gw_marginal_scale_cpu",
    "gw_marginal_ip_probability_cpu",
    "gw_marginal_ip_hybrid",
    "gw_marginal_ip_scan_cpu",
    "gw_marginal_ip_sum_wgpu",
    "gw_marginal_ip_sum_cpu",
    "gw_marginal_ip_scale_wgpu",
    "gw_marginal_ip_scale_cpu",
    "spectral_lr_control_cpu",
    "zspace_optimizer_lr_control_cpu",
    "warmup_cosine_lr_control_cpu",
    "wave_scan_fwd_wgpu",
    "wave_scan_fwd_cpu",
    "wave_scan_bwd_wgpu",
    "wave_scan_bwd_cpu",
    "wave_scan_stack_fwd_composite",
    "wave_scan_stack_bwd_composite",
    "coherence_wave_fwd_composite",
    "coherence_wave_bwd_composite",
    "topos_res_fwd_composite",
    "topos_res_bwd_composite",
    "embedding_fwd_wgpu",
    "embedding_fwd_cpu",
    "embedding_bwd_wgpu",
    "embedding_bwd_cpu",
    "relu_fwd_wgpu",
    "relu_fwd_cpu",
    "relu_bwd_wgpu",
    "relu_bwd_cpu",
    "relu_util_wgpu",
    "relu_util_cpu",
    "dropout_fwd_cpu",
    "dropout_bwd_cpu",
    "dropout_fwd_composite",
    "dropout_bwd_composite",
    "scale_wgpu",
    "scale_cpu",
    "add_wgpu",
    "add_cpu",
    "hadamard_wgpu",
    "hadamard_cpu",
    "mul_row_wgpu",
    "mul_row_cpu",
    "row_affine_wgpu",
    "row_affine_cpu",
    "add_scaled_wgpu",
    "add_scaled_cpu",
    "sub_wgpu",
    "sub_cpu",
    "reduce_wgpu",
    "reduce_cpu",
    "l1_wgpu",
    "l1_cpu",
    "l2_wgpu",
    "l2_cpu",
    "hypergrad_accum_wgpu",
    "hypergrad_accum_cpu",
    "hypergrad_update_cpu",
    "realgrad_accum_cpu",
    "tensor_mse_composite",
    "loss_fwd_wgpu",
    "loss_fwd_cpu",
    "loss_bwd_wgpu",
    "loss_bwd_cpu",
    "gelu_bwd_wgpu",
    "gelu_bwd_cpu",
    "layer_norm_wgpu",
    "layer_norm_cpu",
    "layer_norm_bwd_cpu",
    "layer_norm_bwd_hybrid",
    "layer_norm_bwd_input_hybrid",
    "layer_norm_bwd_input_wgpu",
    "layer_norm_bwd_input_cpu",
    "layer_norm_bwd_input_reduce_wgpu",
    "layer_norm_bwd_input_reduce_cpu",
    "layer_norm_bwd_normalization_wgpu",
    "layer_norm_bwd_normalization_cpu",
    "batch_norm_bwd_cpu",
    "batch_norm_bwd_hybrid",
    "batch_norm_bwd_input_wgpu",
    "batch_norm_bwd_input_cpu",
    "batch_norm_bwd_input_reduce_wgpu",
    "batch_norm_bwd_input_reduce_cpu",
    "batch_norm_bwd_normalization_wgpu",
    "batch_norm_bwd_normalization_cpu",
    "attention_wgpu",
    "attention_cpu",
    "zrba_cov_cpu",
    "zrba_cov_hybrid",
    "zrba_cov_center_cpu",
    "zrba_cov_accum_wgpu",
    "zrba_cov_accum_cpu",
    "zrba_cov_low_rank_cpu_eigen",
    "zrba_cov_psd_cpu_eigen",
    "zrba_metric_control_cpu",
    "zrba_softmax_summary_cpu",
    "max_pool_fwd_wgpu",
    "max_pool_fwd_cpu",
    "max_pool_bwd_wgpu",
    "max_pool_bwd_cpu",
    "avg_pool_fwd_wgpu",
    "avg_pool_fwd_cpu",
    "avg_pool_bwd_wgpu",
    "avg_pool_bwd_cpu",
    "wavelet_fwd_cpu",
    "wavelet_bwd_cpu",
    "dynamic_field_fwd_wgpu",
    "dynamic_field_fwd_cpu",
    "dynamic_field_bwd_wgpu",
    "dynamic_field_bwd_cpu",
    "lstm_fwd_cpu",
    "lstm_bwd_cpu",
    "lstm_fwd_composite",
    "lstm_fwd_hybrid",
    "lstm_fwd_proj_wgpu",
    "lstm_fwd_recurrent_wgpu",
    "lstm_fwd_recurrent_cpu",
    "lstm_fwd_gate_cpu",
    "lstm_fwd_gate_wgpu",
    "lstm_bwd_hybrid",
    "lstm_bwd_recurrent_wgpu",
    "lstm_bwd_recurrent_cpu",
    "lstm_bwd_gate_cpu",
    "lstm_bwd_gate_wgpu",
    "lstm_bwd_bptt_cpu",
    "lstm_bwd_bptt_wgpu",
    "lstm_bwd_bptt_scan_cpu",
    "lstm_bwd_bptt_scan_wgpu",
    "lstm_bwd_bptt_gate_cpu",
    "lstm_bwd_bptt_gate_wgpu",
    "lstm_bwd_bptt_cell_cpu",
    "lstm_bwd_bptt_cell_wgpu",
    "lstm_bwd_bptt_state_cpu",
    "lstm_bwd_bptt_state_wgpu",
    "lstm_bwd_input_wgpu",
    "lstm_bwd_input_cpu",
    "lstm_bwd_param_hybrid",
    "lstm_bwd_param_reduce_wgpu",
    "lstm_bwd_param_reduce_cpu",
    "lstm_bwd_param_cpu",
    "lstm_bwd_bias_wgpu",
    "lstm_bwd_bias_cpu",
    "lstm_bwd_param_scale_wgpu",
    "lstm_est_cpu_debt_ops",
    "lstm_est_fwd_gate_cpu_debt_ops",
    "lstm_est_fwd_gate_wgpu_ops",
    "lstm_est_bwd_gate_cpu_debt_ops",
    "lstm_est_bwd_gate_wgpu_ops",
    "lstm_est_gate_cpu_debt_ops",
    "lstm_est_gate_wgpu_ops",
    "lstm_est_bptt_cpu_debt_ops",
    "lstm_est_bptt_wgpu_ops",
    "zrel_fwd_adapter",
    "zrel_bwd_adapter",
    "mixer_fwd_cpu",
    "mixer_bwd_cpu",
    "mixer_fwd_composite",
    "mixer_bwd_composite",
    "wave_gate_fwd_wgpu",
    "wave_gate_fwd_cpu",
    "wave_gate_bwd_wgpu",
    "wave_gate_bwd_cpu",
    "projector_fwd_cpu",
    "projector_bwd_cpu",
    "projector_fwd_composite",
    "scaler_fwd_cpu",
    "scaler_bwd_cpu",
    "scaler_fwd_composite",
    "scaler_bwd_composite",
    "non_liner_fwd_cpu",
    "non_liner_bwd_cpu",
    "non_liner_fwd_composite",
    "non_liner_bwd_composite",
]

LEARNING_OP_METRICS = {
    "matmul_wgpu": (
        "tensor_op_backend_matmul_wgpu",
        "tensor_op_backend_matmul_scaled_wgpu",
    ),
    "matmul_faer": (
        "tensor_op_backend_matmul_faer",
        "tensor_op_backend_matmul_scaled_faer",
    ),
    "matmul_naive": (
        "tensor_op_backend_matmul_naive",
        "tensor_op_backend_matmul_scaled_naive",
    ),
    "prepacked_wgpu": ("tensor_op_backend_matmul_prepacked_wgpu",),
    "prepacked_faer": ("tensor_op_backend_matmul_prepacked_faer",),
    "prepacked_naive": ("tensor_op_backend_matmul_prepacked_naive",),
    "softmax_wgpu": (
        "tensor_op_backend_row_softmax_wgpu",
        "tensor_op_backend_row_softmax_hardmax_wgpu",
        "tensor_op_backend_row_softmax_hardmax_spiral_wgpu",
    ),
    "softmax_cpu": (
        "tensor_op_backend_row_softmax_cpu",
        "tensor_op_backend_row_softmax_hardmax_cpu",
        "tensor_op_backend_row_softmax_hardmax_spiral_cpu",
    ),
    "softmax_bwd_wgpu": ("tensor_op_backend_zspace_softmax_backward_wgpu",),
    "softmax_bwd_cpu": ("tensor_op_backend_zspace_softmax_backward_cpu",),
    "coherence_scan_fwd_wgpu": ("tensor_op_backend_zspace_coherence_scan_forward_wgpu",),
    "coherence_scan_fwd_cpu": ("tensor_op_backend_zspace_coherence_scan_forward_cpu",),
    "coherence_scan_bwd_wgpu": ("tensor_op_backend_zspace_coherence_scan_backward_wgpu",),
    "coherence_scan_bwd_cpu": ("tensor_op_backend_zspace_coherence_scan_backward_cpu",),
    "psi_heatmap_summary_cpu": ("tensor_op_backend_psi_heatmap_distribution_summary_cpu",),
    "zspace_semantic_dist_cpu": (
        "tensor_op_backend_zspace_semantic_distribution_semantic_cpu",
    ),
    "zspace_semantic_dist_infer_cpu": (
        "tensor_op_backend_zspace_semantic_distribution_semantic_inference_semantic_cpu",
    ),
    "zspace_semantic_window_cpu": (
        "tensor_op_backend_zspace_semantic_window_semantic_cpu",
    ),
    "zspace_semantic_window_hybrid": (
        "tensor_op_backend_zspace_semantic_window_hybrid",
    ),
    "zspace_semantic_window_energy_wgpu": (
        "tensor_op_backend_zspace_semantic_window_window_energy_wgpu",
    ),
    "zspace_semantic_window_energy_cpu": (
        "tensor_op_backend_zspace_semantic_window_window_energy_cpu",
        "tensor_op_backend_zspace_semantic_window_window_energy_semantic_cpu",
    ),
    "zspace_semantic_window_scale_wgpu": (
        "tensor_op_backend_zspace_semantic_window_distribution_scale_wgpu",
    ),
    "zspace_semantic_window_scale_cpu": (
        "tensor_op_backend_zspace_semantic_window_distribution_scale_cpu",
    ),
    "zspace_semantic_window_control_cpu": (
        "tensor_op_backend_zspace_semantic_window_semantic_control_cpu",
    ),
    "zspace_maxwell_summary_cpu": (
        "tensor_op_backend_zspace_maxwell_pulse_summary_summary_cpu",
    ),
    "zspace_semantic_fusion_cpu": (
        "tensor_op_backend_zspace_semantic_distribution_fusion_semantic_cpu",
    ),
    "zspace_semantic_fusion_hybrid": (
        "tensor_op_backend_zspace_semantic_distribution_fusion_hybrid",
    ),
    "zspace_semantic_fusion_accum_wgpu": (
        "tensor_op_backend_zspace_semantic_distribution_fusion_fusion_accumulation_wgpu",
    ),
    "zspace_semantic_fusion_accum_cpu": (
        "tensor_op_backend_zspace_semantic_distribution_fusion_fusion_accumulation_cpu",
        "tensor_op_backend_zspace_semantic_distribution_fusion_fusion_accumulation_semantic_cpu",
    ),
    "zspace_semantic_fusion_scale_wgpu": (
        "tensor_op_backend_zspace_semantic_distribution_fusion_distribution_scale_wgpu",
    ),
    "zspace_semantic_fusion_scale_cpu": (
        "tensor_op_backend_zspace_semantic_distribution_fusion_distribution_scale_cpu",
    ),
    "lawvere_guard_control_cpu": (
        "tensor_op_backend_lawvere_guard_probability_slice_control_cpu",
    ),
    "biome_absorb_topos_cpu": (
        "tensor_op_backend_tensor_biome_absorb_weighted_topos_cpu",
    ),
    "biome_renorm_control_cpu": (
        "tensor_op_backend_tensor_biome_renormalise_weights_control_cpu",
    ),
    "biome_canopy_hybrid": ("tensor_op_backend_tensor_biome_canopy_hybrid",),
    "biome_canopy_accum_wgpu": (
        "tensor_op_backend_tensor_biome_canopy_accumulation_wgpu",
    ),
    "biome_canopy_accum_cpu": (
        "tensor_op_backend_tensor_biome_canopy_accumulation_cpu",
    ),
    "biome_canopy_normalise_wgpu": (
        "tensor_op_backend_tensor_biome_canopy_normalise_wgpu",
    ),
    "biome_canopy_normalise_cpu": (
        "tensor_op_backend_tensor_biome_canopy_normalise_cpu",
    ),
    "biome_canopy_rewrite_topos_cpu": (
        "tensor_op_backend_tensor_biome_canopy_rewrite_topos_cpu",
    ),
    "desire_auto_probability_cpu": (
        "tensor_op_backend_desire_automation_vector_normalise_probability_cpu",
    ),
    "desire_auto_hybrid": (
        "tensor_op_backend_desire_automation_vector_normalise_hybrid",
    ),
    "desire_auto_sanitize_wgpu": (
        "tensor_op_backend_desire_automation_vector_normalise_sanitize_wgpu",
    ),
    "desire_auto_sanitize_cpu": (
        "tensor_op_backend_desire_automation_vector_normalise_sanitize_cpu",
        "tensor_op_backend_desire_automation_vector_normalise_sanitize_probability_cpu",
    ),
    "desire_auto_scale_wgpu": (
        "tensor_op_backend_desire_automation_vector_normalise_distribution_scale_wgpu",
    ),
    "desire_auto_scale_cpu": (
        "tensor_op_backend_desire_automation_vector_normalise_distribution_scale_cpu",
    ),
    "desire_softmax_probability_cpu": ("tensor_op_backend_desire_softmax_probability_cpu",),
    "desire_softmax_hybrid": ("tensor_op_backend_desire_softmax_hybrid",),
    "desire_softmax_row_wgpu": (
        "tensor_op_backend_desire_softmax_softmax_wgpu",
    ),
    "desire_softmax_row_cpu": (
        "tensor_op_backend_desire_softmax_softmax_cpu",
    ),
    "desire_softmax_exp_wgpu": (
        "tensor_op_backend_desire_softmax_exp_wgpu",
    ),
    "desire_softmax_exp_cpu": (
        "tensor_op_backend_desire_softmax_exp_probability_cpu",
    ),
    "desire_softmax_scale_wgpu": (
        "tensor_op_backend_desire_softmax_distribution_scale_wgpu",
    ),
    "desire_softmax_scale_cpu": (
        "tensor_op_backend_desire_softmax_distribution_scale_cpu",
    ),
    "desire_norm_probability_cpu": ("tensor_op_backend_desire_normalise_probability_cpu",),
    "desire_norm_hybrid": ("tensor_op_backend_desire_normalise_hybrid",),
    "desire_norm_sanitize_wgpu": (
        "tensor_op_backend_desire_normalise_sanitize_wgpu",
    ),
    "desire_norm_sanitize_cpu": (
        "tensor_op_backend_desire_normalise_sanitize_cpu",
        "tensor_op_backend_desire_normalise_sanitize_probability_cpu",
    ),
    "desire_norm_scale_wgpu": (
        "tensor_op_backend_desire_normalise_distribution_scale_wgpu",
    ),
    "desire_norm_scale_cpu": (
        "tensor_op_backend_desire_normalise_distribution_scale_cpu",
    ),
    "concept_diffusion_probability_cpu": (
        "tensor_op_backend_concept_diffusion_state_normalise_probability_cpu",
    ),
    "concept_diffusion_f64_cpu": (
        "tensor_op_backend_concept_diffusion_state_normalise_f64_cpu",
    ),
    "concept_diffusion_sum_f64_cpu": (
        "tensor_op_backend_concept_diffusion_state_normalise_state_sum_f64_cpu",
    ),
    "concept_diffusion_precision_f64_cpu": (
        "tensor_op_backend_concept_diffusion_state_normalise_precision_f64_cpu",
    ),
    "concept_diffusion_scale_f64_cpu": (
        "tensor_op_backend_concept_diffusion_state_normalise_distribution_scale_f64_cpu",
    ),
    "sparse_kernel_probability_cpu": (
        "tensor_op_backend_sparse_kernel_probability_row_probability_cpu",
    ),
    "sparse_kernel_hybrid": ("tensor_op_backend_sparse_kernel_probability_row_hybrid",),
    "sparse_kernel_scan_cpu": (
        "tensor_op_backend_sparse_kernel_probability_row_row_scan_probability_cpu",
    ),
    "sparse_kernel_sum_wgpu": (
        "tensor_op_backend_sparse_kernel_probability_row_row_sum_wgpu",
    ),
    "sparse_kernel_sum_cpu": (
        "tensor_op_backend_sparse_kernel_probability_row_row_sum_cpu",
    ),
    "sparse_kernel_scale_wgpu": (
        "tensor_op_backend_sparse_kernel_probability_row_distribution_scale_wgpu",
    ),
    "sparse_kernel_scale_cpu": (
        "tensor_op_backend_sparse_kernel_probability_row_distribution_scale_cpu",
    ),
    "semantic_bridge_cpu": (
        "tensor_op_backend_semantic_bridge_window_distribution_semantic_cpu",
    ),
    "semantic_bridge_hybrid": (
        "tensor_op_backend_semantic_bridge_window_distribution_hybrid",
    ),
    "semantic_bridge_accum_wgpu": (
        "tensor_op_backend_semantic_bridge_window_distribution_semantic_accumulation_wgpu",
    ),
    "semantic_bridge_accum_cpu": (
        "tensor_op_backend_semantic_bridge_window_distribution_semantic_accumulation_cpu",
        "tensor_op_backend_semantic_bridge_window_distribution_semantic_accumulation_semantic_cpu",
    ),
    "semantic_bridge_scale_wgpu": (
        "tensor_op_backend_semantic_bridge_window_distribution_distribution_scale_wgpu",
    ),
    "semantic_bridge_scale_cpu": (
        "tensor_op_backend_semantic_bridge_window_distribution_distribution_scale_cpu",
    ),
    "concept_hint_cpu": ("tensor_op_backend_concept_hint_distribution_semantic_cpu",),
    "concept_hint_hybrid": ("tensor_op_backend_concept_hint_distribution_hybrid",),
    "concept_hint_sanitize_wgpu": (
        "tensor_op_backend_concept_hint_distribution_semantic_sanitize_wgpu",
    ),
    "concept_hint_sanitize_cpu": (
        "tensor_op_backend_concept_hint_distribution_semantic_sanitize_cpu",
        "tensor_op_backend_concept_hint_distribution_semantic_sanitize_semantic_cpu",
    ),
    "concept_hint_infer_cpu": (
        "tensor_op_backend_concept_hint_distribution_semantic_inference_semantic_cpu",
    ),
    "concept_hint_scale_wgpu": (
        "tensor_op_backend_concept_hint_distribution_distribution_scale_wgpu",
    ),
    "concept_hint_scale_cpu": (
        "tensor_op_backend_concept_hint_distribution_distribution_scale_cpu",
    ),
    "gw_marginal_probability_cpu": (
        "tensor_op_backend_gw_marginal_normalise_probability_cpu",
    ),
    "gw_marginal_hybrid": ("tensor_op_backend_gw_marginal_normalise_hybrid",),
    "gw_marginal_scan_cpu": (
        "tensor_op_backend_gw_marginal_normalise_marginal_scan_probability_cpu",
    ),
    "gw_marginal_sum_wgpu": (
        "tensor_op_backend_gw_marginal_normalise_marginal_sum_wgpu",
    ),
    "gw_marginal_sum_cpu": (
        "tensor_op_backend_gw_marginal_normalise_marginal_sum_cpu",
    ),
    "gw_marginal_scale_wgpu": (
        "tensor_op_backend_gw_marginal_normalise_distribution_scale_wgpu",
    ),
    "gw_marginal_scale_cpu": (
        "tensor_op_backend_gw_marginal_normalise_distribution_scale_cpu",
    ),
    "gw_marginal_ip_probability_cpu": (
        "tensor_op_backend_gw_marginal_normalise_in_place_probability_cpu",
    ),
    "gw_marginal_ip_hybrid": (
        "tensor_op_backend_gw_marginal_normalise_in_place_hybrid",
    ),
    "gw_marginal_ip_scan_cpu": (
        "tensor_op_backend_gw_marginal_normalise_in_place_marginal_scan_probability_cpu",
    ),
    "gw_marginal_ip_sum_wgpu": (
        "tensor_op_backend_gw_marginal_normalise_in_place_marginal_sum_wgpu",
    ),
    "gw_marginal_ip_sum_cpu": (
        "tensor_op_backend_gw_marginal_normalise_in_place_marginal_sum_cpu",
    ),
    "gw_marginal_ip_scale_wgpu": (
        "tensor_op_backend_gw_marginal_normalise_in_place_distribution_scale_wgpu",
    ),
    "gw_marginal_ip_scale_cpu": (
        "tensor_op_backend_gw_marginal_normalise_in_place_distribution_scale_cpu",
    ),
    "spectral_lr_control_cpu": (
        "tensor_op_backend_spectral_lr_scale_optimizer_control_cpu",
    ),
    "zspace_optimizer_lr_control_cpu": (
        "tensor_op_backend_zspace_optimizer_lr_scale_optimizer_control_cpu",
    ),
    "warmup_cosine_lr_control_cpu": (
        "tensor_op_backend_warmup_cosine_lr_step_optimizer_control_cpu",
    ),
    "wave_scan_fwd_wgpu": ("tensor_op_backend_wave_scan_forward_wgpu",),
    "wave_scan_fwd_cpu": ("tensor_op_backend_wave_scan_forward_cpu",),
    "wave_scan_bwd_wgpu": ("tensor_op_backend_wave_scan_backward_wgpu",),
    "wave_scan_bwd_cpu": ("tensor_op_backend_wave_scan_backward_cpu",),
    "wave_scan_stack_fwd_composite": ("tensor_op_backend_wave_scan_stack_forward_composite",),
    "wave_scan_stack_bwd_composite": ("tensor_op_backend_wave_scan_stack_backward_composite",),
    "coherence_wave_fwd_composite": ("tensor_op_backend_coherence_wave_forward_composite",),
    "coherence_wave_bwd_composite": ("tensor_op_backend_coherence_wave_backward_composite",),
    "topos_res_fwd_composite": ("tensor_op_backend_topos_resonator_forward_composite",),
    "topos_res_bwd_composite": ("tensor_op_backend_topos_resonator_backward_composite",),
    "embedding_fwd_wgpu": ("tensor_op_backend_embedding_forward_wgpu",),
    "embedding_fwd_cpu": ("tensor_op_backend_embedding_forward_cpu",),
    "embedding_bwd_wgpu": ("tensor_op_backend_embedding_backward_wgpu",),
    "embedding_bwd_cpu": ("tensor_op_backend_embedding_backward_cpu",),
    "relu_fwd_wgpu": ("tensor_op_backend_relu_forward_wgpu",),
    "relu_fwd_cpu": ("tensor_op_backend_relu_forward_cpu",),
    "relu_bwd_wgpu": ("tensor_op_backend_relu_backward_wgpu",),
    "relu_bwd_cpu": ("tensor_op_backend_relu_backward_cpu",),
    "relu_util_wgpu": ("tensor_op_backend_relu_wgpu",),
    "relu_util_cpu": ("tensor_op_backend_relu_cpu",),
    "dropout_fwd_cpu": ("tensor_op_backend_dropout_forward_cpu",),
    "dropout_bwd_cpu": ("tensor_op_backend_dropout_backward_cpu",),
    "dropout_fwd_composite": ("tensor_op_backend_dropout_forward_composite",),
    "dropout_bwd_composite": ("tensor_op_backend_dropout_backward_composite",),
    "scale_wgpu": ("tensor_op_backend_scale_wgpu",),
    "scale_cpu": ("tensor_op_backend_scale_cpu",),
    "add_wgpu": ("tensor_op_backend_add_wgpu",),
    "add_cpu": ("tensor_op_backend_add_cpu",),
    "hadamard_wgpu": ("tensor_op_backend_hadamard_wgpu",),
    "hadamard_cpu": ("tensor_op_backend_hadamard_cpu",),
    "mul_row_wgpu": ("tensor_op_backend_mul_row_wgpu",),
    "mul_row_cpu": ("tensor_op_backend_mul_row_cpu",),
    "row_affine_wgpu": ("tensor_op_backend_row_affine_wgpu",),
    "row_affine_cpu": ("tensor_op_backend_row_affine_cpu",),
    "add_scaled_wgpu": ("tensor_op_backend_add_scaled_wgpu",),
    "add_scaled_cpu": ("tensor_op_backend_add_scaled_cpu",),
    "sub_wgpu": ("tensor_op_backend_sub_wgpu",),
    "sub_cpu": ("tensor_op_backend_sub_cpu",),
    "reduce_wgpu": (
        "tensor_op_backend_sum_axis0_wgpu",
        "tensor_op_backend_sum_axis0_scaled_wgpu",
    ),
    "reduce_cpu": (
        "tensor_op_backend_sum_axis0_cpu",
        "tensor_op_backend_sum_axis0_scaled_cpu",
    ),
    "l1_wgpu": ("tensor_op_backend_sum_abs_wgpu",),
    "l1_cpu": ("tensor_op_backend_sum_abs_cpu",),
    "l2_wgpu": ("tensor_op_backend_squared_l2_norm_wgpu",),
    "l2_cpu": ("tensor_op_backend_squared_l2_norm_cpu",),
    "hypergrad_accum_wgpu": ("tensor_op_backend_hypergrad_accumulate_wave_wgpu",),
    "hypergrad_accum_cpu": (
        "tensor_op_backend_hypergrad_accumulate_wave_cpu",
        "tensor_op_backend_hypergrad_accumulate_pair_cpu",
    ),
    "hypergrad_update_cpu": ("tensor_op_backend_hypergrad_apply_update_cpu",),
    "realgrad_accum_cpu": (
        "tensor_op_backend_realgrad_accumulate_wave_cpu",
        "tensor_op_backend_realgrad_accumulate_pair_cpu",
    ),
    "tensor_mse_composite": ("tensor_op_backend_mean_squared_error_composite",),
    "loss_fwd_wgpu": (
        "tensor_op_backend_mse_loss_forward_wgpu",
        "tensor_op_backend_categorical_cross_entropy_forward_wgpu",
        "tensor_op_backend_focal_loss_forward_wgpu",
        "tensor_op_backend_hyperbolic_cross_entropy_forward_wgpu",
        "tensor_op_backend_contrastive_loss_forward_wgpu",
        "tensor_op_backend_triplet_loss_forward_wgpu",
    ),
    "loss_fwd_cpu": (
        "tensor_op_backend_mse_loss_forward_cpu",
        "tensor_op_backend_categorical_cross_entropy_forward_cpu",
        "tensor_op_backend_focal_loss_forward_cpu",
        "tensor_op_backend_hyperbolic_cross_entropy_forward_cpu",
        "tensor_op_backend_contrastive_loss_forward_cpu",
        "tensor_op_backend_triplet_loss_forward_cpu",
    ),
    "loss_bwd_wgpu": (
        "tensor_op_backend_mse_loss_backward_wgpu",
        "tensor_op_backend_categorical_cross_entropy_backward_wgpu",
        "tensor_op_backend_focal_loss_backward_wgpu",
        "tensor_op_backend_hyperbolic_cross_entropy_backward_wgpu",
        "tensor_op_backend_contrastive_loss_backward_wgpu",
        "tensor_op_backend_triplet_loss_backward_wgpu",
    ),
    "loss_bwd_cpu": (
        "tensor_op_backend_mse_loss_backward_cpu",
        "tensor_op_backend_categorical_cross_entropy_backward_cpu",
        "tensor_op_backend_focal_loss_backward_cpu",
        "tensor_op_backend_hyperbolic_cross_entropy_backward_cpu",
        "tensor_op_backend_contrastive_loss_backward_cpu",
        "tensor_op_backend_triplet_loss_backward_cpu",
    ),
    "gelu_bwd_wgpu": ("tensor_op_backend_gelu_backward_wgpu",),
    "gelu_bwd_cpu": ("tensor_op_backend_gelu_backward_cpu",),
    "layer_norm_wgpu": ("tensor_op_backend_layer_norm_wgpu",),
    "layer_norm_cpu": ("tensor_op_backend_layer_norm_cpu",),
    "layer_norm_bwd_cpu": (
        "tensor_op_backend_layer_norm_backward_cpu",
        "tensor_op_backend_zspace_layer_norm_backward_cpu",
    ),
    "layer_norm_bwd_hybrid": (
        "tensor_op_backend_layer_norm_backward_hybrid",
        "tensor_op_backend_zspace_layer_norm_backward_hybrid",
    ),
    "layer_norm_bwd_input_hybrid": (
        "tensor_op_backend_layer_norm_backward_input_gradient_hybrid",
        "tensor_op_backend_zspace_layer_norm_backward_input_gradient_hybrid",
    ),
    "layer_norm_bwd_input_wgpu": (
        "tensor_op_backend_layer_norm_backward_input_gradient_wgpu",
        "tensor_op_backend_zspace_layer_norm_backward_input_gradient_wgpu",
    ),
    "layer_norm_bwd_input_cpu": (
        "tensor_op_backend_layer_norm_backward_input_gradient_cpu",
        "tensor_op_backend_zspace_layer_norm_backward_input_gradient_cpu",
    ),
    "layer_norm_bwd_input_reduce_wgpu": (
        "tensor_op_backend_layer_norm_backward_input_gradient_reduction_wgpu",
        "tensor_op_backend_zspace_layer_norm_backward_input_gradient_reduction_wgpu",
    ),
    "layer_norm_bwd_input_reduce_cpu": (
        "tensor_op_backend_layer_norm_backward_input_gradient_reduction_cpu",
        "tensor_op_backend_zspace_layer_norm_backward_input_gradient_reduction_cpu",
    ),
    "layer_norm_bwd_normalization_wgpu": (
        "tensor_op_backend_layer_norm_backward_normalization_wgpu",
        "tensor_op_backend_zspace_layer_norm_backward_normalization_wgpu",
    ),
    "layer_norm_bwd_normalization_cpu": (
        "tensor_op_backend_layer_norm_backward_normalization_cpu",
        "tensor_op_backend_zspace_layer_norm_backward_normalization_cpu",
    ),
    "batch_norm_bwd_cpu": (
        "tensor_op_backend_batch_norm_backward_cpu",
        "tensor_op_backend_zspace_batch_norm_backward_cpu",
    ),
    "batch_norm_bwd_hybrid": (
        "tensor_op_backend_batch_norm_backward_hybrid",
        "tensor_op_backend_zspace_batch_norm_backward_hybrid",
    ),
    "batch_norm_bwd_input_wgpu": (
        "tensor_op_backend_batch_norm_backward_input_gradient_wgpu",
        "tensor_op_backend_zspace_batch_norm_backward_input_gradient_wgpu",
    ),
    "batch_norm_bwd_input_cpu": (
        "tensor_op_backend_batch_norm_backward_input_gradient_cpu",
        "tensor_op_backend_zspace_batch_norm_backward_input_gradient_cpu",
    ),
    "batch_norm_bwd_input_reduce_wgpu": (
        "tensor_op_backend_batch_norm_backward_input_gradient_reduction_wgpu",
        "tensor_op_backend_zspace_batch_norm_backward_input_gradient_reduction_wgpu",
    ),
    "batch_norm_bwd_input_reduce_cpu": (
        "tensor_op_backend_batch_norm_backward_input_gradient_reduction_cpu",
        "tensor_op_backend_zspace_batch_norm_backward_input_gradient_reduction_cpu",
    ),
    "batch_norm_bwd_normalization_wgpu": (
        "tensor_op_backend_batch_norm_backward_normalization_wgpu",
        "tensor_op_backend_zspace_batch_norm_backward_normalization_wgpu",
    ),
    "batch_norm_bwd_normalization_cpu": (
        "tensor_op_backend_batch_norm_backward_normalization_cpu",
        "tensor_op_backend_zspace_batch_norm_backward_normalization_cpu",
    ),
    "attention_wgpu": ("tensor_op_backend_scaled_dot_attention_wgpu",),
    "attention_cpu": ("tensor_op_backend_scaled_dot_attention_cpu",),
    "zrba_cov_cpu": ("tensor_op_backend_zrba_cov_head_forward_cpu",),
    "zrba_cov_hybrid": ("tensor_op_backend_zrba_cov_head_forward_hybrid",),
    "zrba_cov_center_cpu": (
        "tensor_op_backend_zrba_cov_head_forward_covariance_centering_cpu",
    ),
    "zrba_cov_accum_wgpu": (
        "tensor_op_backend_zrba_cov_head_forward_covariance_accumulation_wgpu",
    ),
    "zrba_cov_accum_cpu": (
        "tensor_op_backend_zrba_cov_head_forward_covariance_accumulation_cpu",
    ),
    "zrba_cov_low_rank_cpu_eigen": (
        "tensor_op_backend_zrba_cov_head_forward_low_rank_projection_cpu_eigen",
    ),
    "zrba_cov_psd_cpu_eigen": (
        "tensor_op_backend_zrba_cov_head_forward_psd_projection_cpu_eigen",
    ),
    "zrba_metric_control_cpu": ("tensor_op_backend_zrba_metric_weights_normalise_control_cpu",),
    "zrba_softmax_summary_cpu": (
        "tensor_op_backend_zrba_workspace_softmax_summary_summary_cpu",
    ),
    "max_pool_fwd_wgpu": ("tensor_op_backend_max_pool2d_forward_wgpu",),
    "max_pool_fwd_cpu": ("tensor_op_backend_max_pool2d_forward_cpu",),
    "max_pool_bwd_wgpu": ("tensor_op_backend_max_pool2d_backward_wgpu",),
    "max_pool_bwd_cpu": ("tensor_op_backend_max_pool2d_backward_cpu",),
    "avg_pool_fwd_wgpu": ("tensor_op_backend_avg_pool2d_forward_wgpu",),
    "avg_pool_fwd_cpu": ("tensor_op_backend_avg_pool2d_forward_cpu",),
    "avg_pool_bwd_wgpu": ("tensor_op_backend_avg_pool2d_backward_wgpu",),
    "avg_pool_bwd_cpu": ("tensor_op_backend_avg_pool2d_backward_cpu",),
    "wavelet_fwd_cpu": ("tensor_op_backend_continuous_wavelet_forward_cpu",),
    "wavelet_bwd_cpu": ("tensor_op_backend_continuous_wavelet_backward_cpu",),
    "dynamic_field_fwd_wgpu": (
        "tensor_op_backend_dynamic_field_klein_gordon_forward_wgpu",
        "tensor_op_backend_dynamic_field_hamilton_jacobi_forward_wgpu",
        "tensor_op_backend_dynamic_field_stochastic_schrodinger_forward_wgpu",
    ),
    "dynamic_field_fwd_cpu": (
        "tensor_op_backend_dynamic_field_klein_gordon_forward_cpu",
        "tensor_op_backend_dynamic_field_hamilton_jacobi_forward_cpu",
        "tensor_op_backend_dynamic_field_stochastic_schrodinger_forward_cpu",
    ),
    "dynamic_field_bwd_wgpu": (
        "tensor_op_backend_dynamic_field_klein_gordon_backward_wgpu",
        "tensor_op_backend_dynamic_field_hamilton_jacobi_backward_wgpu",
        "tensor_op_backend_dynamic_field_stochastic_schrodinger_backward_wgpu",
    ),
    "dynamic_field_bwd_cpu": (
        "tensor_op_backend_dynamic_field_klein_gordon_backward_cpu",
        "tensor_op_backend_dynamic_field_hamilton_jacobi_backward_cpu",
        "tensor_op_backend_dynamic_field_stochastic_schrodinger_backward_cpu",
    ),
    "lstm_fwd_cpu": ("tensor_op_backend_lstm_forward_cpu",),
    "lstm_bwd_cpu": ("tensor_op_backend_lstm_backward_cpu",),
    "lstm_fwd_composite": ("tensor_op_backend_lstm_forward_composite",),
    "lstm_fwd_hybrid": ("tensor_op_backend_lstm_forward_hybrid",),
    "lstm_fwd_proj_wgpu": ("tensor_op_backend_lstm_forward_input_projection_wgpu",),
    "lstm_fwd_recurrent_wgpu": ("tensor_op_backend_lstm_forward_recurrent_wgpu",),
    "lstm_fwd_recurrent_cpu": ("tensor_op_backend_lstm_forward_recurrent_cpu",),
    "lstm_fwd_gate_cpu": ("tensor_op_backend_lstm_forward_gate_activation_cpu",),
    "lstm_fwd_gate_wgpu": ("tensor_op_backend_lstm_forward_gate_activation_wgpu",),
    "lstm_bwd_hybrid": ("tensor_op_backend_lstm_backward_hybrid",),
    "lstm_bwd_recurrent_wgpu": ("tensor_op_backend_lstm_backward_recurrent_wgpu",),
    "lstm_bwd_recurrent_cpu": ("tensor_op_backend_lstm_backward_recurrent_cpu",),
    "lstm_bwd_gate_cpu": ("tensor_op_backend_lstm_backward_gate_activation_cpu",),
    "lstm_bwd_gate_wgpu": ("tensor_op_backend_lstm_backward_gate_activation_wgpu",),
    "lstm_bwd_bptt_cpu": ("tensor_op_backend_lstm_backward_bptt_cpu",),
    "lstm_bwd_bptt_wgpu": ("tensor_op_backend_lstm_backward_bptt_wgpu",),
    "lstm_bwd_bptt_scan_cpu": ("tensor_op_backend_lstm_backward_bptt_scan_cpu",),
    "lstm_bwd_bptt_scan_wgpu": ("tensor_op_backend_lstm_backward_bptt_scan_wgpu",),
    "lstm_bwd_bptt_gate_cpu": (
        "tensor_op_backend_lstm_backward_bptt_gate_derivative_cpu",
    ),
    "lstm_bwd_bptt_gate_wgpu": (
        "tensor_op_backend_lstm_backward_bptt_gate_derivative_wgpu",
    ),
    "lstm_bwd_bptt_cell_cpu": (
        "tensor_op_backend_lstm_backward_bptt_cell_recurrence_cpu",
    ),
    "lstm_bwd_bptt_cell_wgpu": (
        "tensor_op_backend_lstm_backward_bptt_cell_recurrence_wgpu",
    ),
    "lstm_bwd_bptt_state_cpu": (
        "tensor_op_backend_lstm_backward_bptt_state_carry_cpu",
    ),
    "lstm_bwd_bptt_state_wgpu": (
        "tensor_op_backend_lstm_backward_bptt_state_carry_wgpu",
    ),
    "lstm_bwd_input_wgpu": ("tensor_op_backend_lstm_backward_input_gradient_wgpu",),
    "lstm_bwd_input_cpu": ("tensor_op_backend_lstm_backward_input_gradient_cpu",),
    "lstm_bwd_param_hybrid": (
        "tensor_op_backend_lstm_backward_raw_parameter_gradient_hybrid",
    ),
    "lstm_bwd_param_reduce_wgpu": (
        "tensor_op_backend_lstm_backward_parameter_gradient_reduction_wgpu",
    ),
    "lstm_bwd_param_reduce_cpu": (
        "tensor_op_backend_lstm_backward_parameter_gradient_reduction_cpu",
    ),
    "lstm_bwd_param_cpu": ("tensor_op_backend_lstm_backward_raw_parameter_gradient_cpu",),
    "lstm_bwd_bias_wgpu": ("tensor_op_backend_lstm_backward_bias_gradient_wgpu",),
    "lstm_bwd_bias_cpu": ("tensor_op_backend_lstm_backward_bias_gradient_cpu",),
    "lstm_bwd_param_scale_wgpu": (
        "tensor_op_backend_lstm_backward_parameter_gradient_scale_wgpu",
    ),
    "lstm_est_cpu_debt_ops": ("lstm_estimated_cpu_debt_ops",),
    "lstm_est_fwd_gate_cpu_debt_ops": (
        "lstm_forward_estimated_gate_activation_cpu_debt_ops",
    ),
    "lstm_est_fwd_gate_wgpu_ops": ("lstm_forward_estimated_gate_activation_wgpu_ops",),
    "lstm_est_bwd_gate_cpu_debt_ops": (
        "lstm_backward_estimated_gate_activation_cpu_debt_ops",
    ),
    "lstm_est_bwd_gate_wgpu_ops": ("lstm_backward_estimated_gate_activation_wgpu_ops",),
    "lstm_est_gate_cpu_debt_ops": ("lstm_estimated_gate_activation_cpu_debt_ops",),
    "lstm_est_gate_wgpu_ops": ("lstm_estimated_gate_activation_wgpu_ops",),
    "lstm_est_bptt_cpu_debt_ops": ("lstm_estimated_bptt_cpu_debt_ops",),
    "lstm_est_bptt_wgpu_ops": ("lstm_estimated_bptt_wgpu_ops",),
    "zrel_fwd_adapter": ("tensor_op_backend_zrelativity_module_forward_parameter_adapter",),
    "zrel_bwd_adapter": ("tensor_op_backend_zrelativity_module_backward_parameter_adapter",),
    "mixer_fwd_cpu": ("tensor_op_backend_zspace_mixer_forward_cpu",),
    "mixer_bwd_cpu": ("tensor_op_backend_zspace_mixer_backward_cpu",),
    "mixer_fwd_composite": ("tensor_op_backend_zspace_mixer_forward_composite",),
    "mixer_bwd_composite": ("tensor_op_backend_zspace_mixer_backward_composite",),
    "wave_gate_fwd_wgpu": ("tensor_op_backend_wave_gate_project_wgpu",),
    "wave_gate_fwd_cpu": ("tensor_op_backend_wave_gate_project_cpu",),
    "wave_gate_bwd_wgpu": ("tensor_op_backend_wave_gate_backward_wgpu",),
    "wave_gate_bwd_cpu": ("tensor_op_backend_wave_gate_backward_cpu",),
    "projector_fwd_cpu": ("tensor_op_backend_zspace_projector_forward_cpu",),
    "projector_bwd_cpu": ("tensor_op_backend_zspace_projector_backward_cpu",),
    "projector_fwd_composite": ("tensor_op_backend_zspace_projector_forward_composite",),
    "scaler_fwd_cpu": ("tensor_op_backend_scaler_forward_cpu",),
    "scaler_bwd_cpu": ("tensor_op_backend_scaler_backward_cpu",),
    "scaler_fwd_composite": ("tensor_op_backend_scaler_forward_composite",),
    "scaler_bwd_composite": ("tensor_op_backend_scaler_backward_composite",),
    "non_liner_fwd_cpu": ("tensor_op_backend_non_liner_forward_cpu",),
    "non_liner_bwd_cpu": ("tensor_op_backend_non_liner_backward_cpu",),
    "non_liner_fwd_composite": ("tensor_op_backend_non_liner_forward_composite",),
    "non_liner_bwd_composite": ("tensor_op_backend_non_liner_backward_composite",),
}

LSTM_SCAN_ROUTE_COLUMNS = [
    "lstm_scan_backend",
    "lstm_scan_kernel",
    "lstm_scan_lowering",
    "lstm_scan_fallback",
]

TRACE_POLICY_COLUMNS = [
    "policy_events",
    "policy_wgpu",
    "policy_unison",
    "policy_kdsl_env",
    "policy_kv",
    "policy_kv_soft",
    "policy_tuner",
    "policy_tensor_util",
    "policy_tuner_status",
    "policy_tensor_util_status",
    "policy_kdsl_env_status",
    "policy_kv_status",
    "policy_wgpu_src",
    "policy_unison_src",
    "policy_wg",
    "policy_lanes",
    "policy_candidates",
    "policy_best_score",
    "policy_base_score",
    "policy_gen_score",
    "policy_gen_delta",
    "policy_gen_tie",
    "policy_util_values",
    "policy_util_threshold",
]

_TRAINER_TRACE_MODULE: Any | None = None
_TRAINER_TRACE_SUMMARIZER: Any | None = None


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def number_value(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def trainer_trace_module() -> Any | None:
    global _TRAINER_TRACE_MODULE
    if _TRAINER_TRACE_MODULE is not None:
        return _TRAINER_TRACE_MODULE
    helper_path = REPO_ROOT / "bindings" / "st-py" / "spiraltorch" / "trainer_trace.py"
    if not helper_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_spiraltorch_trainer_trace", helper_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _TRAINER_TRACE_MODULE = module
    return module


def trainer_trace_summarizer() -> Any | None:
    global _TRAINER_TRACE_SUMMARIZER
    if _TRAINER_TRACE_SUMMARIZER is not None:
        return _TRAINER_TRACE_SUMMARIZER

    module = trainer_trace_module()
    if module is None:
        return None
    summarizer = getattr(module, "summarize_trainer_trace_events", None)
    if callable(summarizer):
        _TRAINER_TRACE_SUMMARIZER = summarizer
        return summarizer
    return None


def trainer_trace_jsonl_path(run_dir: Path) -> Path | None:
    for name in (
        "trainer_trace.jsonl",
        "spiraltorch_trainer_trace.jsonl",
        "trainer_steps.jsonl",
    ):
        trace_path = run_dir / name
        if trace_path.exists():
            return trace_path
    return None


def load_trainer_trace_summary(run_dir: Path) -> dict[str, Any] | None:
    for name in (
        "trainer_trace_summary.json",
        "trainer_trace.summary.json",
        "spiraltorch_trainer_trace.summary.json",
    ):
        summary_path = run_dir / name
        if summary_path.exists():
            return read_json(summary_path)

    summarizer = trainer_trace_summarizer()
    if summarizer is None:
        return None
    trace_path = trainer_trace_jsonl_path(run_dir)
    if trace_path is not None:
        try:
            summary = summarizer(trace_path)
        except Exception:
            return None
        return summary if isinstance(summary, dict) else None
    return None


def trace_repair_columns(run_dir: Path) -> dict[str, str]:
    defaults = {key: "-" for key in TRACE_REPAIR_COLUMNS}
    summary = load_trainer_trace_summary(run_dir)
    if not isinstance(summary, dict):
        return defaults
    repairs = summary.get("coherence_repairs")
    if not isinstance(repairs, dict) or not repairs:
        return defaults

    repair_steps = repairs.get("total_nonzero_steps")
    if isinstance(repair_steps, int):
        defaults["repair_steps"] = str(repair_steps)
    defaults["repair_max"] = fmt_float(number_value(repairs.get("max_total")), digits=2)
    defaults["repair_last"] = fmt_float(number_value(repairs.get("last_total")), digits=2)
    defaults["repair_pre_max"] = fmt_float(
        number_value(repairs.get("max_pre_discard_total")),
        digits=2,
    )
    return defaults


def trace_metric_stat(summary: dict[str, Any], metric: str, stat: str) -> float | None:
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return None
    entry = metrics.get(metric)
    if not isinstance(entry, dict):
        return None
    return number_value(entry.get(stat))


def trace_timing_columns(run_dir: Path) -> dict[str, str]:
    defaults = {key: "-" for key in TRACE_TIMING_COLUMNS}
    summary = load_trainer_trace_summary(run_dir)
    if not isinstance(summary, dict):
        return defaults
    defaults["trace_step_ms_last"] = fmt_float(
        trace_metric_stat(summary, "step_time_ms", "last"),
        digits=3,
    )
    defaults["trace_step_ms_mean"] = fmt_float(
        trace_metric_stat(summary, "step_time_ms", "mean"),
        digits=3,
    )
    defaults["trace_step_ms_max"] = fmt_float(
        trace_metric_stat(summary, "step_time_ms", "max"),
        digits=3,
    )
    return defaults


def trace_optim_columns(run_dir: Path) -> dict[str, str]:
    defaults = {key: "-" for key in TRACE_OPTIM_COLUMNS}
    summary = load_trainer_trace_summary(run_dir)
    if not isinstance(summary, dict):
        return defaults
    defaults["trace_update_l2"] = fmt_float(
        trace_metric_stat(summary, "optim_param_update_l2", "last"),
        digits=6,
    )
    defaults["trace_update_ratio"] = fmt_float(
        trace_metric_stat(summary, "optim_param_update_ratio_l2", "last"),
        digits=6,
    )
    defaults["trace_update_ratio_max"] = fmt_float(
        trace_metric_stat(summary, "optim_param_update_ratio_l2", "max"),
        digits=6,
    )
    defaults["trace_update_max_l2"] = fmt_float(
        trace_metric_stat(summary, "optim_param_update_max_l2", "last"),
        digits=6,
    )
    defaults["trace_update_max_ratio"] = fmt_float(
        trace_metric_stat(summary, "optim_param_update_max_ratio_l2", "last"),
        digits=6,
    )
    defaults["trace_zero_param_ratio"] = fmt_float(
        trace_metric_stat(summary, "optim_param_update_zero_param_ratio", "last"),
        digits=4,
    )
    defaults["trace_lr"] = fmt_float(
        trace_metric_stat(summary, "optim_step_fallback_lr", "last"),
        digits=6,
    )
    defaults["trace_state_lr"] = fmt_float(
        trace_metric_stat(summary, "optim_state_fallback_lr", "last"),
        digits=6,
    )
    defaults["trace_adapter_energy"] = fmt_float(
        trace_metric_stat(summary, "optim_state_adapter_avg_energy", "last"),
        digits=6,
    )
    defaults["trace_adapter_curv"] = fmt_float(
        trace_metric_stat(summary, "optim_state_adapter_avg_curvature", "last"),
        digits=6,
    )
    defaults["trace_adapter_spin"] = fmt_float(
        trace_metric_stat(summary, "optim_state_adapter_avg_spin", "last"),
        digits=6,
    )
    defaults["trace_sync_world"] = fmt_float(
        trace_metric_stat(summary, "optim_accumulator_sync_world_size", "last"),
        digits=0,
    )
    defaults["trace_sync_buffers"] = fmt_float(
        trace_metric_stat(summary, "optim_accumulator_sync_buffers", "last"),
        digits=0,
    )
    defaults["trace_sync_values"] = fmt_float(
        trace_metric_stat(summary, "optim_accumulator_sync_values", "last"),
        digits=0,
    )
    return defaults


def trace_backend_columns(run_dir: Path) -> dict[str, str]:
    defaults = {key: "-" for key in BACKEND_COLUMNS}
    summary = load_trainer_trace_summary(run_dir)
    if not isinstance(summary, dict):
        return defaults
    defaults["tensor_ops"] = fmt_float(
        trace_metric_stat(summary, "tensor_ops_total", "last"),
        digits=0,
    )
    defaults["tensor_wgpu"] = fmt_float(
        trace_metric_stat(summary, "tensor_backend_wgpu", "last"),
        digits=0,
    )
    defaults["tensor_wgpu_dense"] = fmt_float(
        trace_metric_stat(summary, "tensor_kernel_backend_wgpu_dense", "last"),
        digits=0,
    )
    defaults["tensor_cpu"] = fmt_float(
        trace_metric_stat(summary, "tensor_backend_cpu", "last"),
        digits=0,
    )
    defaults["tensor_cpu_simd"] = fmt_float(
        trace_metric_stat(summary, "tensor_backend_cpu_simd", "last"),
        digits=0,
    )
    defaults["tensor_f64_cpu"] = fmt_float(
        trace_metric_stat(summary, "tensor_backend_f64_cpu", "last"),
        digits=0,
    )
    defaults["tensor_fallbacks"] = fmt_float(
        trace_metric_stat(summary, "tensor_backend_fallbacks", "last"),
        digits=0,
    )
    return defaults


def trace_learning_op_columns(run_dir: Path) -> dict[str, str]:
    defaults = {key: "-" for key in LEARNING_OP_COLUMNS}
    summary = load_trainer_trace_summary(run_dir)
    if not isinstance(summary, dict):
        return defaults
    for column, metric_keys in LEARNING_OP_METRICS.items():
        values = [
            value
            for metric_key in metric_keys
            for value in [trace_metric_stat(summary, metric_key, "last")]
            if value is not None
        ]
        if values:
            total = sum(values)
            defaults[column] = fmt_float(total, digits=0)
    return defaults


def latest_lstm_scan_route(run_dir: Path) -> dict[str, str | None]:
    route: dict[str, str | None] = {
        "backend": None,
        "kernel": None,
        "lowering": None,
        "fallback_reason": None,
    }
    trace_path = trainer_trace_jsonl_path(run_dir)
    module = trainer_trace_module()
    if trace_path is None or module is None:
        return route
    loader = getattr(module, "load_trainer_trace_events", None)
    if not callable(loader):
        return route
    try:
        events = loader(trace_path, event_type="TensorOpMeta")
    except Exception:
        return route
    for event in events:
        if not isinstance(event, dict) or event.get("op_name") != "lstm_backward":
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        for source, target in (
            ("bptt_scan_backend", "backend"),
            ("bptt_scan_kernel", "kernel"),
            ("bptt_scan_lowering", "lowering"),
            ("bptt_scan_fallback_reason", "fallback_reason"),
        ):
            value = data.get(source)
            if isinstance(value, str) and value:
                route[target] = value
    return route


def trace_lstm_scan_route_columns(run_dir: Path) -> dict[str, str]:
    defaults = {key: "-" for key in LSTM_SCAN_ROUTE_COLUMNS}
    route = latest_lstm_scan_route(run_dir)
    defaults["lstm_scan_backend"] = route.get("backend") or "-"
    defaults["lstm_scan_kernel"] = route.get("kernel") or "-"
    defaults["lstm_scan_lowering"] = route.get("lowering") or "-"
    defaults["lstm_scan_fallback"] = route.get("fallback_reason") or "-"
    return defaults


def coherence_route_columns(
    arch: str,
    requested_backend: str,
    learning_op_cols: dict[str, str],
) -> dict[str, str]:
    defaults = {key: "-" for key in COHERENCE_ROUTE_COLUMNS}
    arch_label = arch.lower()
    if "coherence_scan" not in arch_label and "coherence_wave" not in arch_label:
        return defaults

    scan_wgpu = sum(
        value
        for key in ("coherence_scan_fwd_wgpu", "coherence_scan_bwd_wgpu")
        for value in [parse_number_cell(learning_op_cols.get(key))]
        if value is not None
    )
    scan_cpu = sum(
        value
        for key in ("coherence_scan_fwd_cpu", "coherence_scan_bwd_cpu")
        for value in [parse_number_cell(learning_op_cols.get(key))]
        if value is not None
    )
    wave_wgpu = sum(
        value
        for key in ("wave_scan_fwd_wgpu", "wave_scan_bwd_wgpu")
        for value in [parse_number_cell(learning_op_cols.get(key))]
        if value is not None
    )
    wave_cpu = sum(
        value
        for key in ("wave_scan_fwd_cpu", "wave_scan_bwd_cpu")
        for value in [parse_number_cell(learning_op_cols.get(key))]
        if value is not None
    )
    composite = sum(
        value
        for key in ("coherence_wave_fwd_composite", "coherence_wave_bwd_composite")
        for value in [parse_number_cell(learning_op_cols.get(key))]
        if value is not None
    )
    wgpu_ops = scan_wgpu + wave_wgpu
    cpu_ops = scan_cpu + wave_cpu
    total_route_ops = wgpu_ops + cpu_ops
    if total_route_ops <= 0 and composite <= 0:
        return defaults

    defaults["coherence_route_counts"] = (
        f"wgpu:{fmt_float(wgpu_ops, digits=0)},"
        f"cpu:{fmt_float(cpu_ops, digits=0)},"
        f"composite:{fmt_float(composite, digits=0)}"
    )
    defaults["coherence_route_debt"] = fmt_float(cpu_ops, digits=0)
    requested = requested_backend.lower()
    if requested in {"wgpu", "cuda", "hip", "mps"}:
        if cpu_ops > 0 and wgpu_ops > 0:
            defaults["coherence_route_status"] = "coherence_route_mixed"
        elif cpu_ops > 0:
            defaults["coherence_route_status"] = "coherence_route_cpu"
        elif wgpu_ops > 0:
            defaults["coherence_route_status"] = "coherence_route_clean"
        else:
            defaults["coherence_route_status"] = "coherence_composite_only"
    elif cpu_ops > 0:
        defaults["coherence_route_status"] = "coherence_route_cpu_expected"
    elif wgpu_ops > 0:
        defaults["coherence_route_status"] = "coherence_route_wgpu_unrequested"
    else:
        defaults["coherence_route_status"] = "coherence_composite_only"
    return defaults


def trace_backend_residual_columns(run_dir: Path) -> dict[str, str]:
    defaults = {key: "-" for key in BACKEND_RESIDUAL_COLUMNS}
    summary = load_trainer_trace_summary(run_dir)
    if not isinstance(summary, dict):
        return defaults
    return backend_residual_columns(summary)


def _dominant_count_label(counts: dict[str, Any], prefix: str) -> str:
    candidates: list[tuple[float, str]] = []
    for key, value in counts.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        numeric = number_value(value)
        if numeric is not None and numeric > 0:
            candidates.append((numeric, key[len(prefix):]))
    if not candidates:
        return "-"
    candidates.sort(key=lambda item: (-item[0], item[1]))
    count, label = candidates[0]
    return f"{label}:{fmt_float(count, digits=0)}"


def trace_policy_columns(run_dir: Path) -> dict[str, str]:
    defaults = {key: "-" for key in TRACE_POLICY_COLUMNS}
    summary = load_trainer_trace_summary(run_dir)
    if not isinstance(summary, dict):
        return defaults
    policy = summary.get("backend_policy")
    if not isinstance(policy, dict) or not policy:
        return defaults
    counts = policy.get("counts")
    if not isinstance(counts, dict):
        counts = {}
    last = policy.get("last")
    if not isinstance(last, dict):
        last = {}
    source_counts = policy.get("source_counts")
    if not isinstance(source_counts, dict):
        source_counts = {}
    status_counts = policy.get("status_counts")
    if not isinstance(status_counts, dict):
        status_counts = {}

    defaults["policy_events"] = fmt_float(number_value(counts.get("events")), digits=0)
    defaults["policy_wgpu"] = fmt_float(number_value(counts.get("wgpu_choices")), digits=0)
    defaults["policy_unison"] = fmt_float(number_value(counts.get("unison_choices")), digits=0)
    defaults["policy_kdsl_env"] = fmt_float(number_value(counts.get("kdsl_env_events")), digits=0)
    defaults["policy_kv"] = fmt_float(number_value(counts.get("kdsl_kv_events")), digits=0)
    defaults["policy_kv_soft"] = fmt_float(number_value(counts.get("kv_soft_events")), digits=0)
    defaults["policy_tuner"] = fmt_float(
        number_value(counts.get("wasm_tuner_events")),
        digits=0,
    )
    defaults["policy_tensor_util"] = fmt_float(
        number_value(counts.get("tensor_util_routes")),
        digits=0,
    )
    defaults["policy_tuner_status"] = _dominant_count_label(
        status_counts,
        "wasm_tuner_choice_",
    )
    defaults["policy_tensor_util_status"] = _dominant_count_label(
        status_counts,
        "tensor_util_route_",
    )
    defaults["policy_kdsl_env_status"] = _dominant_count_label(
        status_counts,
        "kdsl_env_bridge_",
    )
    defaults["policy_kv_status"] = _dominant_count_label(
        status_counts,
        "kdsl_kv_bridge_",
    )
    defaults["policy_wgpu_src"] = _dominant_count_label(
        source_counts,
        "wgpu_heuristic_choice_",
    )
    defaults["policy_unison_src"] = _dominant_count_label(
        source_counts,
        "unison_rank_choice_",
    )
    defaults["policy_wg"] = fmt_float(number_value(last.get("wgpu_last_workgroup")), digits=0)
    defaults["policy_lanes"] = fmt_float(number_value(last.get("wgpu_last_lanes")), digits=0)
    defaults["policy_candidates"] = fmt_float(
        number_value(last.get("unison_last_candidate_count")),
        digits=0,
    )
    defaults["policy_best_score"] = fmt_float(
        number_value(last.get("unison_last_best_score")),
        digits=4,
    )
    defaults["policy_base_score"] = fmt_float(
        number_value(last.get("unison_last_baseline_score")),
        digits=4,
    )
    defaults["policy_gen_score"] = fmt_float(
        number_value(last.get("unison_last_wgpu_generated_score")),
        digits=4,
    )
    gen_delta = number_value(last.get("unison_last_wgpu_generated_score_delta"))
    if gen_delta is None:
        gen_score = number_value(last.get("unison_last_wgpu_generated_score"))
        base_score = number_value(last.get("unison_last_baseline_score"))
        if gen_score is not None and base_score is not None:
            gen_delta = gen_score - base_score
    defaults["policy_gen_delta"] = fmt_float(gen_delta, digits=4)
    if gen_delta is not None:
        defaults["policy_gen_tie"] = "yes" if abs(gen_delta) <= 1e-6 else "no"
    defaults["policy_util_values"] = fmt_float(
        number_value(last.get("tensor_util_last_values")),
        digits=0,
    )
    defaults["policy_util_threshold"] = fmt_float(
        number_value(last.get("tensor_util_last_threshold")),
        digits=0,
    )
    return defaults


def metric_backend_columns(run_dir: Path) -> dict[str, str]:
    defaults = {key: "-" for key in BACKEND_COLUMNS}
    rows = read_jsonl(run_dir / "metrics.jsonl")
    if not rows:
        return defaults
    backend = rows[-1].get("tensor_backend")
    if not isinstance(backend, dict):
        return defaults
    defaults["tensor_ops"] = fmt_float(number_value(backend.get("ops_total")), digits=0)
    defaults["tensor_wgpu"] = fmt_float(number_value(backend.get("backend_wgpu")), digits=0)
    defaults["tensor_wgpu_dense"] = fmt_float(
        number_value(backend.get("kernel_backend_wgpu_dense")),
        digits=0,
    )
    defaults["tensor_cpu"] = fmt_float(number_value(backend.get("backend_cpu")), digits=0)
    defaults["tensor_cpu_simd"] = fmt_float(
        number_value(backend.get("backend_cpu_simd")),
        digits=0,
    )
    defaults["tensor_f64_cpu"] = fmt_float(
        number_value(backend.get("backend_f64_cpu")),
        digits=0,
    )
    defaults["tensor_fallbacks"] = fmt_float(number_value(backend.get("fallbacks")), digits=0)
    return defaults


def backend_columns(run_dir: Path) -> dict[str, str]:
    trace_columns = trace_backend_columns(run_dir)
    if any(value != "-" for value in trace_columns.values()):
        return trace_columns
    return metric_backend_columns(run_dir)


def bool_label(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return "-"


def run_backend_audit_columns(run: dict[str, Any]) -> dict[str, str]:
    defaults = {key: "-" for key in RUN_BACKEND_AUDIT_COLUMNS}
    policy = run.get("tensor_policy")
    if isinstance(policy, dict):
        defaults["tensor_policy_matmul"] = str(policy.get("matmul_backend", "-"))
        defaults["tensor_policy_prepacked"] = str(
            policy.get("prepacked_matmul_backend", "-")
        )
        defaults["tensor_policy_softmax"] = str(policy.get("softmax_backend", "-"))
        defaults["tensor_policy_util"] = str(policy.get("tensor_util_backend", "-"))

    runtime = run.get("backend_runtime")
    if isinstance(runtime, dict):
        defaults["backend_status"] = str(runtime.get("requested_backend_status", "-"))
        defaults["backend_kernels"] = bool_label(
            runtime.get("requested_backend_kernels_wired")
        )
        defaults["backend_feature"] = bool_label(
            runtime.get("requested_backend_feature_enabled")
        )
        defaults["hip_real"] = bool_label(runtime.get("hip_real_compiled"))
        defaults["rt_wgpu_initialized"] = bool_label(
            runtime.get("wgpu_rank_runtime_initialized")
        )

    audit = run.get("roundtable_backend_audit")
    if not isinstance(audit, dict):
        return defaults
    defaults["rt_wgpu_compiled"] = bool_label(audit.get("wgpu_runtime_compiled"))
    defaults["rt_wgpu_ctx"] = bool_label(audit.get("wgpu_runtime_context_installed"))
    defaults["rt_wgpu_ready"] = bool_label(audit.get("any_wgpu_exact_runtime_ready"))

    bands = audit.get("bands")
    if not isinstance(bands, list):
        return defaults
    statuses: list[str] = []
    shapes: list[str] = []
    for band in bands:
        if not isinstance(band, dict):
            continue
        band_name = str(band.get("band", "?"))
        status = band.get("wgpu_exact_status")
        if isinstance(status, str) and status:
            statuses.append(f"{band_name}:{status}")
        shape = band.get("wgpu_exact_shape_supported")
        if isinstance(shape, bool):
            shapes.append(f"{band_name}:{'yes' if shape else 'no'}")
    if statuses:
        defaults["rt_wgpu_statuses"] = ",".join(statuses)
    if shapes:
        defaults["rt_wgpu_shapes"] = ",".join(shapes)
    return defaults


def run_paths(raw: str) -> tuple[Path, Path, Path | None]:
    path = Path(raw)
    if path.is_dir():
        summary_path = path / "summary.json"
        run_path = path / "run.json"
        return path, summary_path, run_path if run_path.exists() else None
    return path.parent, path, None


def metric_value(metric: dict[str, Any] | None, field: str) -> float | None:
    if not isinstance(metric, dict):
        return None
    value = metric.get(field)
    return float(value) if isinstance(value, (int, float)) else None


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def fmt_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100.0:.2f}%"


def fmt_raw(value: float | None) -> str:
    return fmt_float(value, digits=8)


def fmt_percent_points_raw(value: float | None) -> str:
    return fmt_raw(value * 100.0 if value is not None else None)


def metadata_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def data_label_for_run(run: dict[str, Any]) -> str:
    data_paths = run.get("data_paths")
    if not isinstance(data_paths, list):
        return "-"
    labels = []
    for item in data_paths:
        if not isinstance(item, str) or not item:
            continue
        labels.append(Path(item).name or item)
    if not labels:
        return "-"
    if len(labels) <= 3:
        return ",".join(labels)
    return ",".join([*labels[:3], f"+{len(labels) - 3}"])


def learnability_value(metric: dict[str, Any], field: str) -> float | None:
    learnability = metric.get("learnability")
    if not isinstance(learnability, dict):
        return None
    value = learnability.get(field)
    return float(value) if isinstance(value, (int, float)) else None


def backend_curve_columns(metric: dict[str, Any]) -> dict[str, str]:
    defaults = {key: "-" for key in BACKEND_COLUMNS}
    backend = metric.get("tensor_backend")
    if not isinstance(backend, dict):
        return defaults
    defaults["tensor_ops"] = fmt_float(number_value(backend.get("ops_total")), digits=0)
    defaults["tensor_wgpu"] = fmt_float(number_value(backend.get("backend_wgpu")), digits=0)
    defaults["tensor_wgpu_dense"] = fmt_float(
        number_value(backend.get("kernel_backend_wgpu_dense")),
        digits=0,
    )
    defaults["tensor_cpu"] = fmt_float(number_value(backend.get("backend_cpu")), digits=0)
    defaults["tensor_cpu_simd"] = fmt_float(
        number_value(backend.get("backend_cpu_simd")),
        digits=0,
    )
    defaults["tensor_f64_cpu"] = fmt_float(
        number_value(backend.get("backend_f64_cpu")),
        digits=0,
    )
    defaults["tensor_fallbacks"] = fmt_float(number_value(backend.get("fallbacks")), digits=0)
    return defaults


def row_for(raw: str) -> tuple[dict[str, str], Path]:
    run_dir, summary_path, run_path = run_paths(raw)
    summary = read_json(summary_path)
    run = read_json(run_path) if run_path is not None else {}
    initial = summary.get("initial_validation")
    final = summary.get("final_validation")
    unigram = summary.get("unigram_validation")
    bigram = summary.get("bigram_validation")
    if not isinstance(initial, dict):
        initial = None
    if not isinstance(final, dict):
        final = None
    if not isinstance(unigram, dict):
        unigram = None
    if not isinstance(bigram, dict):
        bigram = None

    label = run_dir.name or str(run_dir)
    arch = str(run.get("arch", "-"))
    backend = str(run.get("backend", "-"))
    recurrent = str(run.get("recurrent", "-"))
    seed = metadata_cell(run.get("seed"))
    data_label = data_label_for_run(run)
    data_files = metadata_cell(run.get("data_file_count"))
    train_tokens = metadata_cell(run.get("train_tokens"))
    validation_tokens = metadata_cell(run.get("validation_tokens"))
    vocab_size = metadata_cell(run.get("vocab_size"))
    head_prior = str(run.get("head_prior", "-"))
    head_residual_scale = run.get("head_residual_scale")
    bigram_topk_guard = run.get("bigram_topk_guard")
    bigram_topk_guard_k = run.get("bigram_topk_guard_k")
    bigram_rank_guard = run.get("bigram_rank_guard")
    bigram_rank_guard_margin = run.get("bigram_rank_guard_margin")
    bigram_rank_guard_band = run.get("bigram_rank_guard_band")
    bigram_rank_guard_min_candidates = run.get("bigram_rank_guard_min_candidates")
    rank_guard_coverage = run.get("bigram_rank_guard_coverage")
    if not isinstance(rank_guard_coverage, dict):
        rank_guard_coverage = {}
    bigram_soft_guard = run.get("bigram_soft_guard")
    char_feature = str(run.get("char_feature", "-"))
    mode = str(run.get("mode", "-"))
    steps = metadata_cell(run.get("steps"))
    hidden = metadata_cell(run.get("hidden"))
    embed_dim = metadata_cell(run.get("embed_dim"))
    epochs = metadata_cell(run.get("epochs"))
    batches = metadata_cell(run.get("batches_per_epoch"))
    batch = metadata_cell(run.get("batch"))
    eval_samples = metadata_cell(run.get("eval_samples"))
    validation_start_fraction = run.get("validation_start_fraction_requested")
    validation_start_fraction_actual = run.get("validation_start_fraction_actual")
    lr = metadata_cell(run.get("learning_rate"))
    lr_schedule = metadata_cell(run.get("learning_rate_schedule") or "constant")
    lr_warmup = metadata_cell(
        run.get("learning_rate_warmup_epochs")
        if run.get("learning_rate_warmup_epochs") is not None
        else 0
    )
    learning_rate_final_scale = run.get("learning_rate_final_scale")
    lr_final_scale = fmt_float(
        float(learning_rate_final_scale)
        if isinstance(learning_rate_final_scale, (int, float))
        else 1.0
    )
    final_learning_rate = summary.get("final_learning_rate")
    best_learning_rate = summary.get("best_learning_rate")
    context_scale = run.get("context_scale")
    self_score_scale = run.get("self_score_scale")
    query_residual_scale = run.get("query_residual_scale")
    wave_kernel = run.get("kernel")
    wave_dilations = run.get("dilations")
    init_nll = metric_value(initial, "mean_nll")
    final_nll = metric_value(final, "mean_nll")
    unigram_nll = metric_value(unigram, "mean_nll")
    bigram_nll = metric_value(bigram, "mean_nll")
    delta_nll = summary.get("validation_nll_delta")
    final_vs_unigram = summary.get("final_vs_unigram_nll_delta")
    final_vs_bigram = summary.get("final_vs_bigram_nll_delta")
    best_nll = summary.get("best_validation_mean_nll")
    best_vs_unigram = summary.get("best_vs_unigram_nll_delta")
    best_vs_bigram = summary.get("best_vs_bigram_nll_delta")
    final_minus_best = summary.get("final_minus_best_validation_nll")
    delta_acc = summary.get("validation_accuracy_delta")
    if not isinstance(delta_nll, (int, float)):
        delta_nll = None
    if not isinstance(final_vs_unigram, (int, float)):
        final_vs_unigram = None
    if not isinstance(final_vs_bigram, (int, float)):
        final_vs_bigram = None
    if not isinstance(best_nll, (int, float)):
        best_nll = None
    if not isinstance(best_vs_unigram, (int, float)):
        best_vs_unigram = None
    if not isinstance(best_vs_bigram, (int, float)):
        best_vs_bigram = None
    if not isinstance(final_minus_best, (int, float)):
        final_minus_best = None
    if not isinstance(delta_acc, (int, float)):
        delta_acc = None
    best_checkpoint_path = summary.get("best_checkpoint_path")
    best_checkpoint = "yes" if isinstance(best_checkpoint_path, str) and best_checkpoint_path else "-"
    restore_best_requested = summary.get("restore_best_at_end") is True
    restored_best_at_end = summary.get("restored_best_at_end") is True
    restored_best = (
        "yes" if restored_best_at_end else ("requested" if restore_best_requested else "-")
    )
    early_stopped_epoch = summary.get("early_stopped_epoch")
    repair_columns = trace_repair_columns(run_dir)
    timing_columns = trace_timing_columns(run_dir)
    optim_columns = trace_optim_columns(run_dir)
    backend_cols = backend_columns(run_dir)
    run_backend_audit_cols = run_backend_audit_columns(run)
    backend_residual_cols = trace_backend_residual_columns(run_dir)
    learning_op_cols = trace_learning_op_columns(run_dir)
    lstm_scan_route_cols = trace_lstm_scan_route_columns(run_dir)
    coherence_route_cols = coherence_route_columns(arch, backend, learning_op_cols)
    policy_cols = trace_policy_columns(run_dir)

    return (
        {
            "run": label,
            "arch": arch,
            "backend": backend,
            "recurrent": recurrent,
            "seed": seed,
            "data_label": data_label,
            "data_files": data_files,
            "train_tokens": train_tokens,
            "validation_tokens": validation_tokens,
            "vocab_size": vocab_size,
            "head_prior": head_prior,
            "head_resid": fmt_float(
                float(head_residual_scale)
                if isinstance(head_residual_scale, (int, float))
                else None
            ),
            "bigram_guard": fmt_float(
                float(bigram_topk_guard)
                if isinstance(bigram_topk_guard, (int, float))
                else None
            ),
            "bigram_guard_k": metadata_cell(bigram_topk_guard_k),
            "bigram_rank_guard": fmt_float(
                float(bigram_rank_guard)
                if isinstance(bigram_rank_guard, (int, float))
                else None
            ),
            "bigram_rank_margin": fmt_float(
                float(bigram_rank_guard_margin)
                if isinstance(bigram_rank_guard_margin, (int, float))
                else None
            ),
            "bigram_rank_band": fmt_float(
                float(bigram_rank_guard_band)
                if isinstance(bigram_rank_guard_band, (int, float))
                else None
            ),
            "bigram_rank_min": metadata_cell(bigram_rank_guard_min_candidates),
            "bigram_soft_guard": fmt_float(
                float(bigram_soft_guard)
                if isinstance(bigram_soft_guard, (int, float))
                else None
            ),
            "char_feature": char_feature,
            "mode": mode,
            "steps": steps,
            "hidden": hidden,
            "embed_dim": embed_dim,
            "epochs": epochs,
            "batches": batches,
            "batch": batch,
            "eval_samples": eval_samples,
            "val_start": fmt_float(
                float(validation_start_fraction)
                if isinstance(validation_start_fraction, (int, float))
                else None
            ),
            "val_start_actual": fmt_float(
                float(validation_start_fraction_actual)
                if isinstance(validation_start_fraction_actual, (int, float))
                else None
            ),
            "lr": lr,
            "lr_schedule": lr_schedule,
            "lr_warmup": lr_warmup,
            "lr_final_scale": lr_final_scale,
            "final_lr": fmt_float(
                float(final_learning_rate)
                if isinstance(final_learning_rate, (int, float))
                else None
            ),
            "best_lr": fmt_float(
                float(best_learning_rate)
                if isinstance(best_learning_rate, (int, float))
                else None
            ),
            "context_scale": fmt_float(
                float(context_scale) if isinstance(context_scale, (int, float)) else None
            ),
            "self_score": fmt_float(
                float(self_score_scale) if isinstance(self_score_scale, (int, float)) else None
            ),
            "query_resid": fmt_float(
                float(query_residual_scale)
                if isinstance(query_residual_scale, (int, float))
                else None
            ),
            "wave_kernel": metadata_cell(wave_kernel),
            "wave_dilations": (
                ",".join(str(value) for value in wave_dilations)
                if isinstance(wave_dilations, list)
                else metadata_cell(wave_dilations)
            ),
            "init_nll": fmt_float(init_nll),
            "final_windows": fmt_float(metric_value(final, "windows"), digits=0),
            "unigram_windows": fmt_float(metric_value(unigram, "windows"), digits=0),
            "bigram_windows": fmt_float(metric_value(bigram, "windows"), digits=0),
            "rank_cov_windows": fmt_float(
                metric_value(rank_guard_coverage, "windows"), digits=0
            ),
            "rank_cov_unbounded": fmt_float(
                metric_value(rank_guard_coverage, "mean_unbounded_candidates")
            ),
            "rank_cov_band": fmt_float(
                metric_value(rank_guard_coverage, "mean_band_candidates")
            ),
            "rank_cov_min": fmt_float(
                metric_value(rank_guard_coverage, "min_candidates"), digits=0
            ),
            "rank_cov_guarded": fmt_float(
                metric_value(rank_guard_coverage, "mean_guarded_candidates")
            ),
            "rank_cov_effective_band": fmt_float(
                metric_value(rank_guard_coverage, "mean_effective_rank_band")
            ),
            "rank_cov_adaptive_fill_ratio": fmt_float(
                metric_value(rank_guard_coverage, "adaptive_fill_ratio")
            ),
            "rank_cov_filled": fmt_float(
                metric_value(rank_guard_coverage, "mean_adaptive_filled_candidates")
            ),
            "rank_cov_zero_ratio": fmt_float(
                metric_value(rank_guard_coverage, "zero_guarded_candidate_ratio")
            ),
            "rank_cov_mass": fmt_float(
                metric_value(rank_guard_coverage, "mean_guarded_candidate_mass")
            ),
            "rank_cov_band_ratio": fmt_float(
                metric_value(
                    rank_guard_coverage,
                    "mean_band_to_unbounded_candidate_ratio",
                )
            ),
            "rank_cov_topk_ratio": fmt_float(
                metric_value(
                    rank_guard_coverage,
                    "mean_guarded_to_unbounded_topk_ratio",
                )
            ),
            "final_nll": fmt_float(final_nll),
            "delta_nll": fmt_float(float(delta_nll) if delta_nll is not None else None),
            "unigram_nll": fmt_float(unigram_nll),
            "bigram_nll": fmt_float(bigram_nll),
            "final_vs_unigram": fmt_float(
                float(final_vs_unigram) if final_vs_unigram is not None else None
            ),
            "final_vs_bigram": fmt_float(
                float(final_vs_bigram) if final_vs_bigram is not None else None
            ),
            "final_logprob_lift": fmt_float(
                metric_value(final, "mean_target_logprob_lift")
            ),
            "final_rank_lift": fmt_float(
                metric_value(final, "mean_target_rank_lift"), digits=2
            ),
            "final_rank_lift_raw": fmt_raw(
                metric_value(final, "mean_target_rank_lift")
            ),
            "final_unigram_target_rank": fmt_float(
                metric_value(final, "mean_unigram_target_rank"), digits=2
            ),
            "final_unigram_target_rank_raw": fmt_raw(
                metric_value(final, "mean_unigram_target_rank")
            ),
            "final_unigram_rank_debt": fmt_float(
                metric_value(final, "mean_target_rank_debt_vs_unigram"), digits=2
            ),
            "final_unigram_rank_debt_raw": fmt_raw(
                metric_value(final, "mean_target_rank_debt_vs_unigram")
            ),
            "final_kl_unigram": fmt_float(metric_value(final, "mean_kl_to_unigram")),
            "final_top5_overlap": fmt_percent(
                metric_value(final, "mean_top5_overlap_with_unigram")
            ),
            "final_bigram_logprob_lift": fmt_float(
                metric_value(final, "mean_target_logprob_lift_vs_bigram")
            ),
            "final_bigram_rank_lift": fmt_float(
                metric_value(final, "mean_target_rank_lift_vs_bigram"), digits=2
            ),
            "final_bigram_rank_lift_raw": fmt_raw(
                metric_value(final, "mean_target_rank_lift_vs_bigram")
            ),
            "final_bigram_target_rank": fmt_float(
                metric_value(final, "mean_bigram_target_rank"), digits=2
            ),
            "final_bigram_target_rank_raw": fmt_raw(
                metric_value(final, "mean_bigram_target_rank")
            ),
            "final_bigram_rank_debt": fmt_float(
                metric_value(final, "mean_target_rank_debt_vs_bigram"), digits=2
            ),
            "final_bigram_rank_debt_raw": fmt_raw(
                metric_value(final, "mean_target_rank_debt_vs_bigram")
            ),
            "final_kl_bigram": fmt_float(metric_value(final, "mean_kl_to_bigram")),
            "final_top5_bigram_overlap": fmt_percent(
                metric_value(final, "mean_top5_overlap_with_bigram")
            ),
            "final_top5_bigram_overlap_raw": fmt_percent_points_raw(
                metric_value(final, "mean_top5_overlap_with_bigram")
            ),
            "final_ppl": fmt_float(metric_value(final, "perplexity")),
            "final_acc": fmt_percent(metric_value(final, "accuracy")),
            "final_entropy": fmt_float(metric_value(final, "mean_entropy")),
            "final_rank": fmt_float(metric_value(final, "mean_target_rank"), digits=2),
            "delta_acc": fmt_percent(float(delta_acc) if delta_acc is not None else None),
            "best_epoch": str(summary.get("best_validation_epoch", "-")),
            "best_nll": fmt_float(float(best_nll) if best_nll is not None else None),
            "best_vs_unigram": fmt_float(
                float(best_vs_unigram) if best_vs_unigram is not None else None
            ),
            "best_vs_bigram": fmt_float(
                float(best_vs_bigram) if best_vs_bigram is not None else None
            ),
            "final_minus_best": fmt_float(
                float(final_minus_best) if final_minus_best is not None else None
            ),
            "restored_best": restored_best,
            "early_stop_epoch": str(early_stopped_epoch)
            if isinstance(early_stopped_epoch, int)
            else "-",
            "best_ckpt": best_checkpoint,
            **repair_columns,
            **timing_columns,
            **optim_columns,
            **run_backend_audit_cols,
            **backend_cols,
            **backend_residual_cols,
            **learning_op_cols,
            **lstm_scan_route_cols,
            **coherence_route_cols,
            **policy_cols,
        },
        run_dir,
    )


def markdown_table(rows: list[dict[str, str]]) -> str:
    headers = [
        "run",
        "arch",
        "backend",
        "recurrent",
        "seed",
        "data_label",
        "data_files",
        "train_tokens",
        "validation_tokens",
        "vocab_size",
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
        "steps",
        "hidden",
        "embed_dim",
        "epochs",
        "batches",
        "batch",
        "eval_samples",
        "val_start",
        "val_start_actual",
        "lr",
        "lr_schedule",
        "lr_warmup",
        "lr_final_scale",
        "final_lr",
        "best_lr",
        "context_scale",
        "self_score",
        "query_resid",
        "wave_kernel",
        "wave_dilations",
        "init_nll",
        "final_windows",
        "unigram_windows",
        "bigram_windows",
        "rank_cov_windows",
        "rank_cov_unbounded",
        "rank_cov_band",
        "rank_cov_min",
        "rank_cov_guarded",
        "rank_cov_effective_band",
        "rank_cov_adaptive_fill_ratio",
        "rank_cov_filled",
        "rank_cov_zero_ratio",
        "rank_cov_mass",
        "rank_cov_band_ratio",
        "rank_cov_topk_ratio",
        "final_nll",
        "delta_nll",
        "unigram_nll",
        "bigram_nll",
        "final_vs_unigram",
        "final_vs_bigram",
        "final_logprob_lift",
        "final_rank_lift",
        "final_unigram_target_rank",
        "final_unigram_rank_debt",
        "final_kl_unigram",
        "final_top5_overlap",
        "final_bigram_logprob_lift",
        "final_bigram_rank_lift",
        "final_bigram_target_rank",
        "final_bigram_rank_debt",
        "final_kl_bigram",
        "final_top5_bigram_overlap",
        "final_ppl",
        "final_acc",
        "final_entropy",
        "final_rank",
        "delta_acc",
        "best_epoch",
        "best_nll",
        "best_vs_unigram",
        "best_vs_bigram",
        "final_minus_best",
        "restored_best",
        "early_stop_epoch",
        "best_ckpt",
    ]
    if any(
        row.get(header, "-") != "-"
        for row in rows
        for header in TRACE_REPAIR_COLUMNS
    ):
        headers.extend(TRACE_REPAIR_COLUMNS)
    if any(row.get(header, "-") != "-" for row in rows for header in TRACE_TIMING_COLUMNS):
        headers.extend(TRACE_TIMING_COLUMNS)
    if any(
        row.get(header, "-") != "-"
        for row in rows
        for header in TRACE_OPTIM_COLUMNS
    ):
        headers.extend(TRACE_OPTIM_COLUMNS)
    if any(row.get(header, "-") != "-" for row in rows for header in BACKEND_COLUMNS):
        headers.extend(BACKEND_COLUMNS)
    if any(
        row.get(header, "-") != "-"
        for row in rows
        for header in BACKEND_RESIDUAL_COLUMNS
    ):
        headers.extend(BACKEND_RESIDUAL_COLUMNS)
    if any(
        row.get(header, "-") != "-"
        for row in rows
        for header in RUN_BACKEND_AUDIT_COLUMNS
    ):
        headers.extend(RUN_BACKEND_AUDIT_COLUMNS)
    if any(row.get(header, "-") != "-" for row in rows for header in LEARNING_OP_COLUMNS):
        headers.extend(LEARNING_OP_COLUMNS)
    if any(row.get(header, "-") != "-" for row in rows for header in LSTM_SCAN_ROUTE_COLUMNS):
        headers.extend(LSTM_SCAN_ROUTE_COLUMNS)
    if any(row.get(header, "-") != "-" for row in rows for header in COHERENCE_ROUTE_COLUMNS):
        headers.extend(COHERENCE_ROUTE_COLUMNS)
    if any(row.get(header, "-") != "-" for row in rows for header in TRACE_POLICY_COLUMNS):
        headers.extend(TRACE_POLICY_COLUMNS)
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(row[header] for header in headers) + " |")
    return "\n".join(out)


def parse_number_cell(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value == "-":
        return None
    if value.endswith("%"):
        value = value[:-1]
    try:
        return float(value)
    except ValueError:
        return None


def mean_value(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def std_value(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance ** 0.5


def count_label_cells(values: list[str | None], *, missing_label: str = "-") -> str:
    counts: dict[str, int] = {}
    for value in values:
        label = value if value and value != "-" else missing_label
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return "-"
    return ",".join(f"{label}:{count}" for label, count in sorted(counts.items()))


def parse_count_cell(value: str | None) -> dict[str, int]:
    if value is None or value == "-":
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


def sum_count_cells(values: list[str | None]) -> str:
    totals: dict[str, int] = {}
    for value in values:
        for label, count in parse_count_cell(value).items():
            totals[label] = totals.get(label, 0) + count
    if not totals:
        return "-"
    return ",".join(f"{label}:{count}" for label, count in sorted(totals.items()))


def aggregate_route_status(row: dict[str, str]) -> str:
    scan_backends = parse_count_cell(row.get("lstm_scan_backend_counts"))
    if not scan_backends:
        return "-"

    fallbacks = parse_count_cell(row.get("lstm_scan_fallback_counts"))
    active_fallbacks = {
        label: count for label, count in fallbacks.items() if label != "none" and count > 0
    }
    if active_fallbacks:
        return "scan_fallback"

    requested_backend = row.get("backend")
    if requested_backend in {"wgpu", "cuda", "hip", "mps"} and requested_backend not in scan_backends:
        return "scan_route_mismatch"
    if len(scan_backends) > 1:
        return "scan_route_mixed"
    return "clean_route"


def aggregate_coherence_route_status(row: dict[str, str]) -> str:
    statuses = parse_count_cell(row.get("coherence_route_status_counts"))
    if not statuses:
        return "-"
    if any(
        status in statuses
        for status in ("coherence_route_cpu", "coherence_route_mixed")
    ):
        return "coherence_route_debt"
    if "coherence_route_clean" in statuses:
        return "coherence_route_clean"
    if "coherence_route_cpu_expected" in statuses:
        return "coherence_route_cpu_expected"
    if "coherence_route_wgpu_unrequested" in statuses:
        return "coherence_route_wgpu_unrequested"
    if "coherence_composite_only" in statuses:
        return "coherence_composite_only"
    return "coherence_route_observed"


def aggregate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(column, "-") for column in AGGREGATE_GROUP_COLUMNS)
        grouped[key].append(row)

    output = []
    for key, group_rows in sorted(grouped.items()):
        row = dict(zip(AGGREGATE_GROUP_COLUMNS, key, strict=True))
        row["runs"] = str(len(group_rows))
        final_values = [
            value
            for value in (parse_number_cell(item.get("final_nll")) for item in group_rows)
            if value is not None
        ]
        row["final_nll_mean"] = fmt_float(mean_value(final_values))
        row["final_nll_std"] = fmt_float(std_value(final_values))
        for column in AGGREGATE_MEAN_COLUMNS:
            values = [
                value
                for value in (parse_number_cell(item.get(column)) for item in group_rows)
                if value is not None
            ]
            row[f"{column}_mean"] = fmt_float(mean_value(values))
        for column in AGGREGATE_RAW_MEAN_COLUMNS:
            values = [
                value
                for value in (parse_number_cell(item.get(column)) for item in group_rows)
                if value is not None
            ]
            row[f"{column}_mean"] = fmt_raw(mean_value(values))
        for column in AGGREGATE_PERCENT_COLUMNS:
            values = [
                value
                for value in (parse_number_cell(item.get(column)) for item in group_rows)
                if value is not None
            ]
            percent_mean = mean_value(values)
            row[f"{column}_mean"] = (
                fmt_percent(percent_mean / 100.0) if percent_mean is not None else "-"
            )
        has_lstm_scan_route = any(
            item.get("lstm_scan_backend") not in (None, "-") for item in group_rows
        )
        if has_lstm_scan_route:
            row["lstm_scan_backend_counts"] = count_label_cells(
                [item.get("lstm_scan_backend") for item in group_rows]
            )
            row["lstm_scan_fallback_counts"] = count_label_cells(
                [
                    (
                        "none"
                        if item.get("lstm_scan_backend") not in (None, "-")
                        and item.get("lstm_scan_fallback") in (None, "-")
                        else item.get("lstm_scan_fallback")
                    )
                    for item in group_rows
                ],
            )
        else:
            row["lstm_scan_backend_counts"] = "-"
            row["lstm_scan_fallback_counts"] = "-"
        row["route_status"] = aggregate_route_status(row)
        has_coherence_route = any(
            item.get("coherence_route_status") not in (None, "-")
            for item in group_rows
        )
        if has_coherence_route:
            row["coherence_route_status_counts"] = count_label_cells(
                [item.get("coherence_route_status") for item in group_rows]
            )
            row["coherence_route_counts"] = sum_count_cells(
                [item.get("coherence_route_counts") for item in group_rows]
            )
            row["coherence_route_status"] = aggregate_coherence_route_status(row)
        else:
            row["coherence_route_status_counts"] = "-"
            row["coherence_route_counts"] = "-"
            row["coherence_route_status"] = "-"
        output.append(row)
    return output


def ranked_aggregate_rows(
    rows: list[dict[str, str]], limit: int = 8
) -> list[dict[str, str]]:
    def metric_or_inf(row: dict[str, str], column: str) -> float:
        value = parse_number_cell(row.get(column))
        return value if value is not None else float("inf")

    ranked = [
        row
        for row in rows
        if parse_number_cell(row.get("final_nll_mean")) is not None
    ]
    ranked.sort(
        key=lambda row: (
            metric_or_inf(row, "final_nll_mean"),
            metric_or_inf(row, "best_nll_mean"),
            metric_or_inf(row, "trace_step_ms_mean_mean"),
            metric_or_inf(row, "cpu_debt_ops_mean"),
            row.get("arch", "-"),
            row.get("recurrent", "-"),
            row.get("steps", "-"),
            row.get("hidden", "-"),
        )
    )
    return ranked[: max(limit, 0)]


def top_aggregate_table(rows: list[dict[str, str]], limit: int = 8) -> str:
    ranked = ranked_aggregate_rows(rows, limit=limit)
    if not ranked:
        return ""
    headers = [
        header
        for header in TOP_AGGREGATE_COLUMNS
        if any(row.get(header, "-") != "-" for row in ranked)
    ]
    out = ["## Top Aggregate Runs", ""]
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in ranked:
        out.append("| " + " | ".join(row.get(header, "-") for header in headers) + " |")
    return "\n".join(out)


def aggregate_table(rows: list[dict[str, str]]) -> str:
    aggregate = aggregate_rows(rows)
    headers = [
        *AGGREGATE_GROUP_COLUMNS,
        "runs",
        "final_nll_mean",
        "final_nll_std",
        *(f"{column}_mean" for column in AGGREGATE_MEAN_COLUMNS),
        *(f"{column}_mean" for column in AGGREGATE_PERCENT_COLUMNS),
    ]
    if any(row.get(column, "-") != "-" for row in aggregate for column in AGGREGATE_COUNT_COLUMNS):
        headers.extend(AGGREGATE_COUNT_COLUMNS)
    out = ["## Aggregate Runs", ""]
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in aggregate:
        out.append("| " + " | ".join(row.get(header, "-") for header in headers) + " |")
    top_table = top_aggregate_table(aggregate)
    if top_table and len(aggregate) > 1:
        out.extend(["", top_table])
    return "\n".join(out)


def curve_rows_for(summary_row: dict[str, str], run_dir: Path) -> list[dict[str, str]]:
    rows = []
    for metric in read_jsonl(run_dir / "metrics.jsonl"):
        validation = metric.get("validation")
        if not isinstance(validation, dict):
            validation = None
        rows.append(
            {
                "run": summary_row["run"],
                "arch": summary_row["arch"],
                "backend": summary_row["backend"],
                "epoch": str(metric.get("epoch", "-")),
                "train_loss": fmt_float(
                    float(metric["average_loss"])
                    if isinstance(metric.get("average_loss"), (int, float))
                    else None
                ),
                "val_nll": fmt_float(metric_value(validation, "mean_nll")),
                "val_ppl": fmt_float(metric_value(validation, "perplexity")),
                "val_acc": fmt_percent(metric_value(validation, "accuracy")),
                "val_entropy": fmt_float(metric_value(validation, "mean_entropy")),
                "val_rank": fmt_float(metric_value(validation, "mean_target_rank"), digits=2),
                "logprob_lift": fmt_float(
                    metric_value(validation, "mean_target_logprob_lift")
                ),
                "rank_lift": fmt_float(
                    metric_value(validation, "mean_target_rank_lift"), digits=2
                ),
                "kl_unigram": fmt_float(metric_value(validation, "mean_kl_to_unigram")),
                "update_l2": fmt_float(learnability_value(metric, "total_update_l2"), digits=6),
                "update_ratio": fmt_float(
                    learnability_value(metric, "mean_update_to_value_l2"), digits=6
                ),
            }
            | backend_curve_columns(metric)
        )
    return rows


def curve_table(rows: list[dict[str, str]]) -> str:
    headers = [
        "run",
        "arch",
        "backend",
        "epoch",
        "train_loss",
        "val_nll",
        "val_ppl",
        "val_acc",
        "val_entropy",
        "val_rank",
        "logprob_lift",
        "rank_lift",
        "kl_unigram",
        "update_l2",
        "update_ratio",
    ]
    if any(row.get(header, "-") != "-" for row in rows for header in BACKEND_COLUMNS):
        headers.extend(BACKEND_COLUMNS)
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(row[header] for header in headers) + " |")
    return "\n".join(out)


def parameter_rows_for(
    summary_row: dict[str, str], run_dir: Path, limit: int
) -> list[dict[str, str]]:
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    if not metrics:
        return []
    metric = metrics[-1]
    learnability = metric.get("learnability")
    if not isinstance(learnability, dict):
        return []
    parameters = learnability.get("parameters")
    if not isinstance(parameters, list):
        return []

    rows: list[dict[str, str]] = []
    for param in parameters:
        if not isinstance(param, dict):
            continue
        update_l2 = param.get("update_l2")
        update_ratio = param.get("update_to_value_l2")
        rows.append(
            {
                "run": summary_row["run"],
                "arch": summary_row["arch"],
                "epoch": str(metric.get("epoch", "-")),
                "parameter": str(param.get("name", "-")),
                "shape": f"{param.get('rows', '-')}x{param.get('cols', '-')}",
                "value_l2": fmt_float(
                    float(param["value_l2"])
                    if isinstance(param.get("value_l2"), (int, float))
                    else None,
                    digits=6,
                ),
                "update_l2": fmt_float(
                    float(update_l2) if isinstance(update_l2, (int, float)) else None,
                    digits=6,
                ),
                "update_ratio": fmt_float(
                    float(update_ratio) if isinstance(update_ratio, (int, float)) else None,
                    digits=6,
                ),
            }
        )
    rows.sort(
        key=lambda row: float(row["update_l2"]) if row["update_l2"] != "-" else -1.0,
        reverse=True,
    )
    return rows[: max(limit, 0)]


def parameter_table(rows: list[dict[str, str]]) -> str:
    headers = [
        "run",
        "arch",
        "epoch",
        "parameter",
        "shape",
        "value_l2",
        "update_l2",
        "update_ratio",
    ]
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(row[header] for header in headers) + " |")
    return "\n".join(out)


def compare_payload(
    pairs: list[tuple[dict[str, str], Path]],
    *,
    include_aggregate: bool,
    include_curves: bool,
    params_limit: int,
) -> dict[str, Any]:
    rows = [row for row, _ in pairs]
    aggregate = aggregate_rows(rows) if include_aggregate else []
    curves: list[dict[str, str]] = []
    if include_curves:
        for row, run_dir in pairs:
            curves.extend(curve_rows_for(row, run_dir))
    parameters: list[dict[str, str]] = []
    if params_limit > 0:
        for row, run_dir in pairs:
            parameters.extend(parameter_rows_for(row, run_dir, params_limit))
    return {
        "schema": "st.char_lm.compare.v1",
        "runs": rows,
        "aggregate_runs": aggregate,
        "top_aggregate_runs": ranked_aggregate_rows(aggregate) if aggregate else [],
        "curves": curves,
        "parameters": parameters,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare summary.json artifacts from char-LM model-zoo runs."
    )
    parser.add_argument(
        "--curves",
        action="store_true",
        help="also print epoch-level validation curves from metrics.jsonl",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="also print grouped mean/std rows for multi-seed comparisons",
    )
    parser.add_argument(
        "--params",
        type=int,
        default=0,
        metavar="N",
        help="also print the top N parameter updates from the last metrics epoch",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="write machine-readable comparison rows to this JSON file",
    )
    parser.add_argument("runs", nargs="+", help="run directories or summary.json files")
    args = parser.parse_args()
    pairs = [row_for(raw) for raw in args.runs]
    rows = [row for row, _ in pairs]
    payload = compare_payload(
        pairs,
        include_aggregate=args.aggregate,
        include_curves=args.curves,
        params_limit=args.params,
    )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(markdown_table(rows))
    if args.aggregate:
        print()
        print(aggregate_table(rows))
    if args.curves:
        curve_rows: list[dict[str, str]] = []
        for row, run_dir in pairs:
            curve_rows.extend(curve_rows_for(row, run_dir))
        if curve_rows:
            print()
            print(curve_table(curve_rows))
    if args.params > 0:
        parameter_rows: list[dict[str, str]] = []
        for row, run_dir in pairs:
            parameter_rows.extend(parameter_rows_for(row, run_dir, args.params))
        if parameter_rows:
            print()
            print(parameter_table(parameter_rows))


if __name__ == "__main__":
    main()
