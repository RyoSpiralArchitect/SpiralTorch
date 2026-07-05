from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "load_trainer_trace_events",
    "summarize_trainer_trace_events",
    "write_trainer_trace_html",
]

COHERENCE_REPAIR_METRIC_KEYS = (
    "coherence_repairs_total",
    "coherence_repaired_detected",
    "coherence_repaired_weights_total",
    "coherence_repaired_non_finite_weights",
    "coherence_repaired_negative_weights",
    "coherence_pre_discard_repairs_total",
    "coherence_pre_discard_repaired_non_finite",
    "coherence_pre_discard_repaired_negative",
)

TRACE_SPOTLIGHT_KEYS = (
    "loss_weighted",
    "loss_weighted_base",
    "spectral_label",
    "spectral_turnover",
    "spectral_lr_scale",
    "softlogic_inertia",
    "softlogic_z",
    "curvature_value",
    "curvature_pressure_rel_var",
    "batch_input_rows",
    "batch_target_rows",
    "batch_prediction_rows",
    "batch_prediction_non_finite_values",
    "batch_loss_non_finite_values",
    "batch_grad_output_non_finite_values",
    "batch_grad_output_l2_finite",
    "optim_step_fallback_lr",
    "optim_step_hyper_lr",
    "optim_state_fallback_lr",
    "optim_state_hyper_lr",
    "optim_state_realgrad_enabled",
    "optim_state_adapter_avg_energy",
    "optim_state_adapter_avg_curvature",
    "optim_state_adapter_avg_spin",
    "optim_accumulator_sync_enabled",
    "optim_accumulator_sync_world_size",
    "optim_accumulator_sync_buffers",
    "optim_accumulator_sync_values",
    "optim_param_update_l2",
    "optim_param_update_ratio_l2",
    "optim_param_update_max_l2",
    "optim_param_update_max_ratio_l2",
    "optim_param_update_zero_param_ratio",
    "coherence_repairs_total",
    "coherence_repaired_detected",
    "coherence_pre_discard_repairs_total",
    "grad_values_non_finite",
    "tensor_meta_non_finite_sentinels",
    "tensor_ops_total",
    "tensor_backend_wgpu",
    "tensor_kernel_backend_wgpu_dense",
    "tensor_backend_cpu",
    "tensor_backend_cpu_simd",
    "tensor_backend_f64_cpu",
    "tensor_backend_fallbacks",
    "tensor_backend_requested_wgpu_hits",
    "tensor_backend_requested_wgpu_runtime_fallbacks",
    "tensor_backend_requested_wgpu_component_hits",
    "tensor_backend_requested_wgpu_component_fallbacks",
    "tensor_op_backend_matmul_scaled_wgpu",
    "tensor_op_backend_matmul_scaled_faer",
    "tensor_op_backend_matmul_scaled_naive",
    "tensor_op_backend_zspace_softmax_forward_auto",
    "tensor_op_backend_zspace_softmax_forward_cpu",
    "tensor_op_backend_zspace_softmax_forward_cpu_adaptive",
    "tensor_op_backend_zspace_softmax_forward_wgpu",
    "tensor_op_backend_zspace_softmax_backward_cpu",
    "tensor_op_backend_zspace_coherence_scan_forward_wgpu",
    "tensor_op_backend_zspace_coherence_scan_forward_cpu",
    "tensor_op_backend_zspace_coherence_scan_backward_wgpu",
    "tensor_op_backend_zspace_coherence_scan_backward_cpu",
    "tensor_op_backend_psi_heatmap_distribution_summary_cpu",
    "tensor_op_backend_zspace_semantic_distribution_semantic_cpu",
    "tensor_op_backend_zspace_semantic_distribution_semantic_inference_semantic_cpu",
    "tensor_op_backend_zspace_semantic_distribution_hybrid",
    "tensor_op_backend_zspace_semantic_distribution_semantic_sparse_scan_semantic_cpu",
    "tensor_op_backend_zspace_semantic_distribution_semantic_accumulation_wgpu",
    "tensor_op_backend_zspace_semantic_distribution_semantic_accumulation_cpu",
    "tensor_op_backend_zspace_semantic_distribution_distribution_scale_wgpu",
    "tensor_op_backend_zspace_semantic_distribution_distribution_scale_cpu",
    "tensor_op_backend_zspace_semantic_window_semantic_cpu",
    "tensor_op_backend_zspace_semantic_window_hybrid",
    "tensor_op_backend_zspace_semantic_window_window_energy_semantic_cpu",
    "tensor_op_backend_zspace_semantic_window_window_energy_wgpu",
    "tensor_op_backend_zspace_semantic_window_window_energy_cpu",
    "tensor_op_backend_zspace_semantic_window_distribution_scale_wgpu",
    "tensor_op_backend_zspace_semantic_window_distribution_scale_cpu",
    "tensor_op_backend_zspace_semantic_window_semantic_control_cpu",
    "tensor_op_backend_zspace_maxwell_pulse_summary_summary_cpu",
    "tensor_op_backend_zspace_semantic_distribution_fusion_semantic_cpu",
    "tensor_op_backend_zspace_semantic_distribution_fusion_hybrid",
    "tensor_op_backend_zspace_semantic_distribution_fusion_fusion_accumulation_semantic_cpu",
    "tensor_op_backend_zspace_semantic_distribution_fusion_fusion_accumulation_wgpu",
    "tensor_op_backend_zspace_semantic_distribution_fusion_fusion_accumulation_cpu",
    "tensor_op_backend_zspace_semantic_distribution_fusion_distribution_scale_wgpu",
    "tensor_op_backend_zspace_semantic_distribution_fusion_distribution_scale_cpu",
    "tensor_op_backend_lawvere_guard_probability_slice_control_cpu",
    "tensor_op_backend_tensor_biome_absorb_weighted_topos_cpu",
    "tensor_op_backend_tensor_biome_renormalise_weights_control_cpu",
    "tensor_op_backend_tensor_biome_canopy_hybrid",
    "tensor_op_backend_tensor_biome_canopy_accumulation_wgpu",
    "tensor_op_backend_tensor_biome_canopy_accumulation_cpu",
    "tensor_op_backend_tensor_biome_canopy_normalise_wgpu",
    "tensor_op_backend_tensor_biome_canopy_normalise_cpu",
    "tensor_op_backend_tensor_biome_canopy_rewrite_topos_cpu",
    "tensor_op_backend_desire_automation_vector_normalise_probability_cpu",
    "tensor_op_backend_desire_automation_vector_normalise_hybrid",
    "tensor_op_backend_desire_automation_vector_normalise_sanitize_wgpu",
    "tensor_op_backend_desire_automation_vector_normalise_sanitize_cpu",
    "tensor_op_backend_desire_automation_vector_normalise_sanitize_probability_cpu",
    "tensor_op_backend_desire_automation_vector_normalise_distribution_scale_wgpu",
    "tensor_op_backend_desire_automation_vector_normalise_distribution_scale_cpu",
    "tensor_op_backend_desire_softmax_probability_cpu",
    "tensor_op_backend_desire_softmax_hybrid",
    "tensor_op_backend_desire_softmax_softmax_wgpu",
    "tensor_op_backend_desire_softmax_softmax_cpu",
    "tensor_op_backend_desire_softmax_exp_wgpu",
    "tensor_op_backend_desire_softmax_exp_probability_cpu",
    "tensor_op_backend_desire_softmax_distribution_scale_wgpu",
    "tensor_op_backend_desire_softmax_distribution_scale_cpu",
    "tensor_op_backend_desire_normalise_probability_cpu",
    "tensor_op_backend_desire_normalise_hybrid",
    "tensor_op_backend_desire_normalise_sanitize_probability_cpu",
    "tensor_op_backend_desire_normalise_sanitize_wgpu",
    "tensor_op_backend_desire_normalise_sanitize_cpu",
    "tensor_op_backend_desire_normalise_distribution_scale_wgpu",
    "tensor_op_backend_desire_normalise_distribution_scale_cpu",
    "tensor_op_backend_concept_diffusion_state_normalise_probability_cpu",
    "tensor_op_backend_concept_diffusion_state_normalise_f64_cpu",
    "tensor_op_backend_concept_diffusion_state_normalise_state_sum_f64_cpu",
    "tensor_op_backend_concept_diffusion_state_normalise_precision_f64_cpu",
    "tensor_op_backend_concept_diffusion_state_normalise_distribution_scale_f64_cpu",
    "tensor_op_backend_sparse_kernel_probability_row_probability_cpu",
    "tensor_op_backend_sparse_kernel_probability_row_hybrid",
    "tensor_op_backend_sparse_kernel_probability_row_row_scan_probability_cpu",
    "tensor_op_backend_sparse_kernel_probability_row_row_sum_wgpu",
    "tensor_op_backend_sparse_kernel_probability_row_row_sum_cpu",
    "tensor_op_backend_sparse_kernel_probability_row_distribution_scale_wgpu",
    "tensor_op_backend_sparse_kernel_probability_row_distribution_scale_cpu",
    "tensor_op_backend_semantic_bridge_window_distribution_semantic_cpu",
    "tensor_op_backend_semantic_bridge_window_distribution_hybrid",
    "tensor_op_backend_semantic_bridge_window_distribution_semantic_sparse_scan_semantic_cpu",
    "tensor_op_backend_semantic_bridge_window_distribution_semantic_accumulation_semantic_cpu",
    "tensor_op_backend_semantic_bridge_window_distribution_semantic_accumulation_wgpu",
    "tensor_op_backend_semantic_bridge_window_distribution_semantic_accumulation_cpu",
    "tensor_op_backend_semantic_bridge_window_distribution_distribution_scale_wgpu",
    "tensor_op_backend_semantic_bridge_window_distribution_distribution_scale_cpu",
    "tensor_op_backend_concept_hint_distribution_semantic_cpu",
    "tensor_op_backend_concept_hint_distribution_hybrid",
    "tensor_op_backend_concept_hint_distribution_semantic_sanitize_semantic_cpu",
    "tensor_op_backend_concept_hint_distribution_semantic_sanitize_wgpu",
    "tensor_op_backend_concept_hint_distribution_semantic_sanitize_cpu",
    "tensor_op_backend_concept_hint_distribution_semantic_inference_semantic_cpu",
    "tensor_op_backend_concept_hint_distribution_semantic_inference_semantic_bridge_window_distribution",
    "tensor_op_backend_concept_hint_distribution_semantic_sparse_scan_semantic_cpu",
    "tensor_op_backend_concept_hint_distribution_distribution_scale_wgpu",
    "tensor_op_backend_concept_hint_distribution_distribution_scale_cpu",
    "tensor_op_backend_gw_marginal_normalise_probability_cpu",
    "tensor_op_backend_gw_marginal_normalise_hybrid",
    "tensor_op_backend_gw_marginal_normalise_marginal_scan_probability_cpu",
    "tensor_op_backend_gw_marginal_normalise_marginal_sum_wgpu",
    "tensor_op_backend_gw_marginal_normalise_marginal_sum_cpu",
    "tensor_op_backend_gw_marginal_normalise_distribution_scale_wgpu",
    "tensor_op_backend_gw_marginal_normalise_distribution_scale_cpu",
    "tensor_op_backend_gw_marginal_normalise_in_place_probability_cpu",
    "tensor_op_backend_gw_marginal_normalise_in_place_hybrid",
    "tensor_op_backend_gw_marginal_normalise_in_place_marginal_scan_probability_cpu",
    "tensor_op_backend_gw_marginal_normalise_in_place_marginal_sum_wgpu",
    "tensor_op_backend_gw_marginal_normalise_in_place_marginal_sum_cpu",
    "tensor_op_backend_gw_marginal_normalise_in_place_distribution_scale_wgpu",
    "tensor_op_backend_gw_marginal_normalise_in_place_distribution_scale_cpu",
    "tensor_op_backend_spectral_lr_scale_optimizer_control_cpu",
    "tensor_op_backend_zspace_optimizer_lr_scale_optimizer_control_cpu",
    "tensor_op_backend_warmup_cosine_lr_step_optimizer_control_cpu",
    "tensor_op_backend_wave_scan_forward_wgpu",
    "tensor_op_backend_wave_scan_forward_cpu",
    "tensor_op_backend_wave_scan_backward_wgpu",
    "tensor_op_backend_wave_scan_backward_cpu",
    "tensor_op_backend_wave_scan_stack_forward_composite",
    "tensor_op_backend_wave_scan_stack_backward_composite",
    "tensor_op_backend_wave_scan_stack_forward_merge_wgpu",
    "tensor_op_backend_wave_scan_stack_forward_merge_cpu",
    "tensor_op_backend_requested_wgpu_component_hit_wave_scan_stack_forward_merge_wgpu",
    "tensor_op_backend_requested_wgpu_component_fallback_wave_scan_stack_forward_merge_cpu",
    "tensor_op_backend_wave_scan_stack_backward_merge_wgpu",
    "tensor_op_backend_wave_scan_stack_backward_merge_cpu",
    "tensor_op_backend_requested_wgpu_component_hit_wave_scan_stack_backward_merge_wgpu",
    "tensor_op_backend_requested_wgpu_component_fallback_wave_scan_stack_backward_merge_cpu",
    "tensor_op_backend_coherence_wave_forward_composite",
    "tensor_op_backend_coherence_wave_backward_composite",
    "tensor_op_backend_coherence_wave_forward_merge_wgpu",
    "tensor_op_backend_coherence_wave_forward_merge_cpu",
    "tensor_op_backend_coherence_wave_backward_merge_wgpu",
    "tensor_op_backend_coherence_wave_backward_merge_cpu",
    "tensor_op_backend_topos_resonator_forward_composite",
    "tensor_op_backend_topos_resonator_backward_composite",
    "tensor_op_backend_embedding_forward_cpu",
    "tensor_op_backend_embedding_backward_cpu",
    "tensor_op_backend_relu_forward_cpu",
    "tensor_op_backend_relu_backward_cpu",
    "tensor_op_backend_relu_wgpu",
    "tensor_op_backend_relu_cpu",
    "tensor_op_backend_layer_norm_backward_cpu",
    "tensor_op_backend_zspace_layer_norm_backward_cpu",
    "tensor_op_backend_layer_norm_backward_hybrid",
    "tensor_op_backend_zspace_layer_norm_backward_hybrid",
    "tensor_op_backend_layer_norm_backward_input_gradient_hybrid",
    "tensor_op_backend_zspace_layer_norm_backward_input_gradient_hybrid",
    "tensor_op_backend_layer_norm_backward_input_gradient_wgpu",
    "tensor_op_backend_zspace_layer_norm_backward_input_gradient_wgpu",
    "tensor_op_backend_layer_norm_backward_input_gradient_cpu",
    "tensor_op_backend_zspace_layer_norm_backward_input_gradient_cpu",
    "tensor_op_backend_layer_norm_backward_input_gradient_reduction_wgpu",
    "tensor_op_backend_zspace_layer_norm_backward_input_gradient_reduction_wgpu",
    "tensor_op_backend_layer_norm_backward_input_gradient_reduction_cpu",
    "tensor_op_backend_zspace_layer_norm_backward_input_gradient_reduction_cpu",
    "tensor_op_backend_layer_norm_backward_normalization_wgpu",
    "tensor_op_backend_zspace_layer_norm_backward_normalization_wgpu",
    "tensor_op_backend_layer_norm_backward_normalization_cpu",
    "tensor_op_backend_zspace_layer_norm_backward_normalization_cpu",
    "tensor_op_backend_batch_norm_backward_cpu",
    "tensor_op_backend_zspace_batch_norm_backward_cpu",
    "tensor_op_backend_batch_norm_backward_hybrid",
    "tensor_op_backend_zspace_batch_norm_backward_hybrid",
    "tensor_op_backend_batch_norm_backward_input_gradient_wgpu",
    "tensor_op_backend_zspace_batch_norm_backward_input_gradient_wgpu",
    "tensor_op_backend_batch_norm_backward_input_gradient_cpu",
    "tensor_op_backend_zspace_batch_norm_backward_input_gradient_cpu",
    "tensor_op_backend_batch_norm_backward_input_gradient_reduction_wgpu",
    "tensor_op_backend_zspace_batch_norm_backward_input_gradient_reduction_wgpu",
    "tensor_op_backend_batch_norm_backward_input_gradient_reduction_cpu",
    "tensor_op_backend_zspace_batch_norm_backward_input_gradient_reduction_cpu",
    "tensor_op_backend_batch_norm_backward_normalization_wgpu",
    "tensor_op_backend_zspace_batch_norm_backward_normalization_wgpu",
    "tensor_op_backend_batch_norm_backward_normalization_cpu",
    "tensor_op_backend_zspace_batch_norm_backward_normalization_cpu",
    "tensor_op_backend_dropout_forward_cpu",
    "tensor_op_backend_dropout_backward_cpu",
    "tensor_op_backend_dropout_forward_composite",
    "tensor_op_backend_dropout_backward_composite",
    "tensor_op_backend_dropout_forward_mask_wgpu",
    "tensor_op_backend_dropout_forward_mask_cpu",
    "tensor_op_backend_dropout_forward_rng_cpu",
    "tensor_op_backend_requested_wgpu_component_hit_dropout_forward_mask_wgpu",
    "tensor_op_backend_requested_wgpu_component_fallback_dropout_forward_mask_cpu",
    "tensor_op_backend_dropout_backward_mask_wgpu",
    "tensor_op_backend_dropout_backward_mask_cpu",
    "tensor_op_backend_dropout_backward_rng_cpu",
    "tensor_op_backend_requested_wgpu_component_hit_dropout_backward_mask_wgpu",
    "tensor_op_backend_requested_wgpu_component_fallback_dropout_backward_mask_cpu",
    "tensor_op_backend_scale_wgpu",
    "tensor_op_backend_scale_cpu",
    "tensor_op_backend_add_wgpu",
    "tensor_op_backend_add_cpu",
    "tensor_op_backend_hadamard_wgpu",
    "tensor_op_backend_hadamard_cpu",
    "tensor_op_backend_mul_row_wgpu",
    "tensor_op_backend_mul_row_cpu",
    "tensor_op_backend_row_affine_wgpu",
    "tensor_op_backend_row_affine_cpu",
    "tensor_op_backend_add_scaled_wgpu",
    "tensor_op_backend_add_scaled_cpu",
    "tensor_op_backend_sub_wgpu",
    "tensor_op_backend_sub_cpu",
    "tensor_op_backend_sum_axis0_wgpu",
    "tensor_op_backend_sum_axis0_cpu",
    "tensor_op_backend_sum_axis0_scaled_wgpu",
    "tensor_op_backend_sum_axis0_scaled_cpu",
    "tensor_op_backend_sum_abs_wgpu",
    "tensor_op_backend_sum_abs_cpu",
    "tensor_op_backend_hypergrad_accumulate_wave_cpu",
    "tensor_op_backend_hypergrad_accumulate_pair_cpu",
    "tensor_op_backend_hypergrad_apply_update_cpu",
    "tensor_op_backend_realgrad_accumulate_wave_cpu",
    "tensor_op_backend_realgrad_accumulate_pair_cpu",
    "tensor_op_backend_mean_squared_error_composite",
    "tensor_op_backend_mse_loss_backward_cpu",
    "tensor_op_backend_categorical_cross_entropy_backward_cpu",
    "tensor_op_backend_focal_loss_backward_cpu",
    "tensor_op_backend_hyperbolic_cross_entropy_backward_cpu",
    "tensor_op_backend_contrastive_loss_backward_cpu",
    "tensor_op_backend_triplet_loss_backward_cpu",
    "tensor_op_backend_zrba_cov_head_forward_cpu",
    "tensor_op_backend_zrba_cov_head_forward_hybrid",
    "tensor_op_backend_zrba_cov_head_forward_covariance_centering_cpu",
    "tensor_op_backend_zrba_cov_head_forward_covariance_accumulation_wgpu",
    "tensor_op_backend_zrba_cov_head_forward_covariance_accumulation_cpu",
    "tensor_op_backend_zrba_cov_head_forward_low_rank_projection_cpu_eigen",
    "tensor_op_backend_zrba_cov_head_forward_psd_projection_cpu_eigen",
    "tensor_op_backend_zrba_metric_weights_normalise_control_cpu",
    "tensor_op_backend_zrba_workspace_softmax_summary_summary_cpu",
    "tensor_op_backend_max_pool2d_forward_wgpu",
    "tensor_op_backend_max_pool2d_forward_cpu",
    "tensor_op_backend_max_pool2d_backward_wgpu",
    "tensor_op_backend_max_pool2d_backward_cpu",
    "tensor_op_backend_avg_pool2d_forward_wgpu",
    "tensor_op_backend_avg_pool2d_forward_cpu",
    "tensor_op_backend_avg_pool2d_backward_wgpu",
    "tensor_op_backend_avg_pool2d_backward_cpu",
    "tensor_op_backend_continuous_wavelet_forward_cpu",
    "tensor_op_backend_continuous_wavelet_backward_cpu",
    "tensor_op_backend_dynamic_field_klein_gordon_forward_wgpu",
    "tensor_op_backend_dynamic_field_klein_gordon_backward_wgpu",
    "tensor_op_backend_dynamic_field_klein_gordon_forward_cpu",
    "tensor_op_backend_dynamic_field_klein_gordon_backward_cpu",
    "tensor_op_backend_dynamic_field_hamilton_jacobi_forward_wgpu",
    "tensor_op_backend_dynamic_field_hamilton_jacobi_backward_wgpu",
    "tensor_op_backend_dynamic_field_hamilton_jacobi_forward_cpu",
    "tensor_op_backend_dynamic_field_hamilton_jacobi_backward_cpu",
    "tensor_op_backend_dynamic_field_stochastic_schrodinger_forward_wgpu",
    "tensor_op_backend_dynamic_field_stochastic_schrodinger_backward_wgpu",
    "tensor_op_backend_dynamic_field_stochastic_schrodinger_forward_cpu",
    "tensor_op_backend_dynamic_field_stochastic_schrodinger_backward_cpu",
    "tensor_op_backend_dynamic_field_stochastic_schrodinger_forward_deterministic_wgpu",
    "tensor_op_backend_dynamic_field_stochastic_schrodinger_forward_deterministic_cpu",
    "tensor_op_backend_dynamic_field_stochastic_schrodinger_forward_rng_cpu",
    "tensor_op_backend_requested_wgpu_component_hit_dynamic_field_stochastic_schrodinger_forward_deterministic_wgpu",
    "tensor_op_backend_requested_wgpu_component_fallback_dynamic_field_stochastic_schrodinger_forward_deterministic_cpu",
    "tensor_op_backend_dynamic_field_stochastic_schrodinger_backward_gradient_scale_wgpu",
    "tensor_op_backend_dynamic_field_stochastic_schrodinger_backward_gradient_scale_cpu",
    "tensor_op_backend_requested_wgpu_component_hit_dynamic_field_stochastic_schrodinger_backward_gradient_scale_wgpu",
    "tensor_op_backend_requested_wgpu_component_fallback_dynamic_field_stochastic_schrodinger_backward_gradient_scale_cpu",
    "tensor_op_backend_lstm_forward_cpu",
    "tensor_op_backend_lstm_backward_cpu",
    "tensor_op_backend_lstm_forward_composite",
    "tensor_op_backend_lstm_forward_hybrid",
    "tensor_op_backend_lstm_forward_input_projection_wgpu",
    "tensor_op_backend_lstm_forward_bias_wgpu",
    "tensor_op_backend_lstm_forward_recurrent_wgpu",
    "tensor_op_backend_lstm_forward_recurrent_cpu",
    "tensor_op_backend_lstm_forward_gate_activation_cpu",
    "tensor_op_backend_lstm_forward_gate_activation_wgpu",
    "tensor_op_backend_lstm_backward_hybrid",
    "tensor_op_backend_lstm_backward_recurrent_wgpu",
    "tensor_op_backend_lstm_backward_recurrent_cpu",
    "tensor_op_backend_lstm_backward_gate_activation_cpu",
    "tensor_op_backend_lstm_backward_gate_activation_wgpu",
    "tensor_op_backend_lstm_backward_bptt_cpu",
    "tensor_op_backend_lstm_backward_bptt_wgpu",
    "tensor_op_backend_lstm_backward_bptt_scan_cpu",
    "tensor_op_backend_lstm_backward_bptt_scan_wgpu",
    "tensor_op_backend_lstm_backward_bptt_gate_derivative_cpu",
    "tensor_op_backend_lstm_backward_bptt_gate_derivative_wgpu",
    "tensor_op_backend_lstm_backward_bptt_cell_recurrence_cpu",
    "tensor_op_backend_lstm_backward_bptt_cell_recurrence_wgpu",
    "tensor_op_backend_lstm_backward_bptt_state_carry_cpu",
    "tensor_op_backend_lstm_backward_bptt_state_carry_wgpu",
    "tensor_op_backend_lstm_backward_input_gradient_wgpu",
    "tensor_op_backend_lstm_backward_input_gradient_cpu",
    "tensor_op_backend_lstm_backward_raw_parameter_gradient_hybrid",
    "tensor_op_backend_lstm_backward_raw_parameter_gradient_cpu",
    "tensor_op_backend_lstm_backward_parameter_gradient_reduction_wgpu",
    "tensor_op_backend_lstm_backward_parameter_gradient_reduction_cpu",
    "tensor_op_backend_lstm_backward_bias_gradient_wgpu",
    "tensor_op_backend_lstm_backward_bias_gradient_cpu",
    "tensor_op_backend_lstm_backward_parameter_gradient_scale_wgpu",
    "tensor_op_backend_lstm_backward_parameter_gradient_scale_cpu",
    "lstm_estimated_cpu_debt_ops",
    "lstm_estimated_bptt_cpu_debt_ops",
    "lstm_estimated_bptt_wgpu_ops",
    "lstm_estimated_gate_activation_ops",
    "lstm_estimated_gate_activation_cpu_debt_ops",
    "lstm_estimated_gate_activation_wgpu_ops",
    "lstm_backward_bptt_scan_shape_supported",
    "lstm_backward_bptt_scan_runtime_requested",
    "lstm_backward_bptt_scan_runtime_available",
    "lstm_backward_bptt_scan_runtime_unavailable",
    "lstm_backward_bptt_scan_elapsed_us",
    "lstm_backward_bptt_scan_hidden_values",
    "lstm_backward_bptt_scan_gate_values",
    "lstm_backward_bptt_scan_cell_values",
    "lstm_backward_bptt_scan_recurrent_weight_values",
    "lstm_backward_bptt_scan_scratch_values",
    "lstm_backward_bptt_scan_kernel_dispatches",
    "lstm_backward_bptt_scan_serial_steps",
    "lstm_backward_bptt_scan_workgroup_size",
    "lstm_backward_bptt_scan_parallel_lanes",
    "lstm_backward_estimated_bptt_ops_per_scan_step",
    "lstm_forward_estimated_gate_activation_ops",
    "lstm_forward_estimated_gate_activation_cpu_debt_ops",
    "lstm_forward_estimated_gate_activation_wgpu_ops",
    "lstm_backward_estimated_gate_activation_ops",
    "lstm_backward_estimated_gate_activation_cpu_debt_ops",
    "lstm_backward_estimated_gate_activation_wgpu_ops",
    "lstm_backward_estimated_bptt_ops",
    "lstm_backward_estimated_bptt_cpu_debt_ops",
    "lstm_backward_estimated_bptt_wgpu_ops",
    "lstm_backward_estimated_bptt_gate_derivative_ops",
    "lstm_backward_estimated_bptt_cell_recurrence_ops",
    "lstm_backward_estimated_bptt_state_carry_ops",
    "lstm_backward_estimated_bptt_scan_steps",
    "tensor_op_backend_zrelativity_module_forward_parameter_adapter",
    "tensor_op_backend_zrelativity_module_backward_parameter_adapter",
    "tensor_op_backend_zspace_mixer_forward_cpu",
    "tensor_op_backend_zspace_mixer_backward_cpu",
    "tensor_op_backend_zspace_mixer_forward_composite",
    "tensor_op_backend_zspace_mixer_backward_composite",
    "tensor_op_backend_zspace_mixer_forward_broadcast_wgpu",
    "tensor_op_backend_zspace_mixer_forward_broadcast_cpu",
    "tensor_op_backend_zspace_mixer_backward_broadcast_wgpu",
    "tensor_op_backend_zspace_mixer_backward_broadcast_cpu",
    "tensor_op_backend_zspace_mixer_backward_gradient_reduction_wgpu",
    "tensor_op_backend_zspace_mixer_backward_gradient_reduction_cpu",
    "tensor_op_backend_wave_gate_forward_cpu",
    "tensor_op_backend_wave_gate_forward_wgpu",
    "tensor_op_backend_wave_gate_project_wgpu",
    "tensor_op_backend_wave_gate_project_cpu",
    "tensor_op_backend_wave_gate_backward_wgpu",
    "tensor_op_backend_wave_gate_backward_cpu",
    "tensor_op_backend_zspace_projector_forward_cpu",
    "tensor_op_backend_zspace_projector_backward_cpu",
    "tensor_op_backend_zspace_projector_forward_composite",
    "tensor_op_backend_zspace_projector_forward_projection_wgpu",
    "tensor_op_backend_zspace_projector_forward_projection_cpu",
    "tensor_op_backend_zspace_projector_backward_projection_cpu",
    "tensor_op_backend_zspace_projector_backward_projection_gradient_cpu",
    "tensor_op_backend_zspace_projector_backward_saturation_gradient_cpu",
    "tensor_op_backend_scaler_forward_cpu",
    "tensor_op_backend_scaler_backward_cpu",
    "tensor_op_backend_scaler_forward_composite",
    "tensor_op_backend_scaler_backward_composite",
    "tensor_op_backend_non_liner_forward_cpu",
    "tensor_op_backend_non_liner_backward_cpu",
    "tensor_op_backend_non_liner_forward_composite",
    "tensor_op_backend_non_liner_backward_composite",
    "tensor_op_backend_non_liner_forward_preactivation_wgpu",
    "tensor_op_backend_non_liner_forward_preactivation_cpu",
    "tensor_op_backend_requested_wgpu_component_hit_non_liner_forward_preactivation_wgpu",
    "tensor_op_backend_requested_wgpu_component_fallback_non_liner_forward_preactivation_cpu",
    "tensor_op_backend_non_liner_forward_activation_cpu",
    "tensor_op_backend_requested_wgpu_component_fallback_non_liner_forward_activation_cpu",
    "tensor_op_backend_non_liner_forward_geometry_cpu",
    "tensor_op_backend_requested_wgpu_component_fallback_non_liner_forward_geometry_cpu",
    "tensor_op_backend_non_liner_forward_broadcast_wgpu",
    "tensor_op_backend_non_liner_forward_broadcast_cpu",
    "tensor_op_backend_requested_wgpu_component_hit_non_liner_forward_broadcast_wgpu",
    "tensor_op_backend_requested_wgpu_component_fallback_non_liner_forward_broadcast_cpu",
    "tensor_op_backend_non_liner_backward_preactivation_wgpu",
    "tensor_op_backend_non_liner_backward_preactivation_cpu",
    "tensor_op_backend_non_liner_backward_activation_cpu",
    "tensor_op_backend_non_liner_backward_geometry_cpu",
    "tensor_op_backend_non_liner_backward_broadcast_wgpu",
    "tensor_op_backend_non_liner_backward_broadcast_cpu",
    "tensor_embedding_token_repairs_total",
    "tensor_embedding_unique_token_indices",
    "tensor_embedding_repeated_token_indices",
    "tensor_embedding_non_finite_tokens",
    "tensor_embedding_clamped_high_tokens",
    "backend_policy_events",
    "backend_policy_wgpu_choices",
    "backend_policy_unison_choices",
    "backend_policy_kdsl_env_events",
    "backend_policy_kdsl_kv_events",
    "backend_policy_kv_soft_events",
    "backend_policy_wasm_tuner_events",
    "backend_policy_tensor_util_routes",
    "backend_policy_wgpu_last_workgroup",
    "backend_policy_wgpu_last_lanes",
    "backend_policy_wgpu_last_compaction_tile",
    "backend_policy_wgpu_last_fft_radix",
    "backend_policy_unison_last_candidate_count",
    "backend_policy_unison_last_best_score",
    "backend_policy_tensor_util_last_values",
    "backend_policy_tensor_util_last_threshold",
)

BACKEND_POLICY_COUNT_KEYS = (
    "backend_policy_events",
    "backend_policy_wgpu_choices",
    "backend_policy_unison_choices",
    "backend_policy_kdsl_env_events",
    "backend_policy_kdsl_kv_events",
    "backend_policy_kv_soft_events",
    "backend_policy_wasm_tuner_events",
    "backend_policy_tensor_util_routes",
)

BACKEND_POLICY_LAST_KEYS = (
    "backend_policy_wgpu_last_workgroup",
    "backend_policy_wgpu_last_lanes",
    "backend_policy_wgpu_last_compaction_tile",
    "backend_policy_wgpu_last_fft_radix",
    "backend_policy_wgpu_last_fft_segments",
    "backend_policy_wgpu_last_override_count",
    "backend_policy_unison_last_candidate_count",
    "backend_policy_unison_last_best_score",
    "backend_policy_unison_last_baseline_score",
    "backend_policy_unison_last_wgpu_generated_score",
    "backend_policy_unison_last_wgpu_generated_score_delta",
    "backend_policy_tensor_util_last_values",
    "backend_policy_tensor_util_last_threshold",
)

BACKEND_POLICY_STATUS_PREFIX = "backend_policy_status_"
BACKEND_POLICY_SOURCE_PREFIX = "backend_policy_source_"
BACKEND_POLICY_OPS = (
    "wgpu_heuristic_choice",
    "unison_rank_choice",
    "kdsl_env_bridge",
    "kdsl_kv_bridge",
    "kv_consensus_soft_rules",
    "wasm_tuner_choice",
    "tensor_util_route",
)

CPU_RUNTIME_BACKENDS = {
    "cpu",
    "cpu_eigen",
    "cpu_simd",
    "f64_cpu",
    "faer",
    "naive",
    "probability_cpu",
    "semantic_cpu",
    "topos_cpu",
}

METADATA_ONLY_BACKENDS = {
    "composite",
    "hybrid",
    "view",
    "semantic_bridge_window_distribution",
}

TENSOR_META_SUB_BACKEND_FIELD_SPECS = (
    ("input_gradient_backend", True),
    ("input_gradient_reduction_backend", True),
    ("gradient_reduction_backend", True),
    ("affine_gradient_backend", True),
    ("normalization_backend", True),
    ("input_projection_backend", True),
    ("bias_backend", True),
    ("recurrent_backend", True),
    ("gate_activation_backend", True),
    ("bptt_backend", True),
    ("bptt_scan_backend", False),
    ("bptt_gate_derivative_backend", True),
    ("bptt_cell_recurrence_backend", True),
    ("bptt_state_carry_backend", True),
    ("raw_parameter_gradient_backend", True),
    ("parameter_gradient_reduction_backend", True),
    ("bias_gradient_backend", True),
    ("parameter_gradient_scale_backend", True),
    ("broadcast_backend", True),
    ("activation_backend", True),
    ("preactivation_backend", True),
    ("geometry_backend", True),
    ("mask_backend", True),
    ("rng_backend", False),
    ("deterministic_backend", True),
    ("gradient_scale_backend", True),
    ("merge_backend", True),
    ("accumulation_backend", True),
    ("normalise_backend", True),
    ("rewrite_backend", True),
    ("projection_backend", True),
    ("projection_gradient_backend", True),
    ("saturation_gradient_backend", True),
    ("softmax_backend", True),
    ("exp_backend", True),
    ("sanitize_backend", True),
    ("distribution_scale_backend", True),
    ("semantic_inference_backend", True),
    ("semantic_sparse_scan_backend", True),
    ("semantic_accumulation_backend", True),
    ("semantic_sanitize_backend", True),
    ("window_energy_backend", True),
    ("fusion_accumulation_backend", True),
    ("marginal_scan_backend", True),
    ("marginal_sum_backend", True),
    ("row_scan_backend", True),
    ("row_sum_backend", True),
    ("state_sum_backend", True),
    ("precision_backend", True),
    ("reduction_backend", True),
    ("covariance_centering_backend", True),
    ("covariance_accumulation_backend", True),
    ("low_rank_projection_backend", True),
    ("psd_projection_backend", True),
)

TENSOR_META_SUB_BACKEND_FIELDS = tuple(
    field for field, _count_fallback in TENSOR_META_SUB_BACKEND_FIELD_SPECS
)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def _extract_payload(record: dict[str, Any], *, event_type: str) -> Any | None:
    record_type = record.get("event_type") or record.get("type")
    if record_type == event_type and "payload" in record:
        return record.get("payload")
    if "payload" in record and record_type is None:
        return record.get("payload")
    event = record.get("event")
    if isinstance(event, dict) and event.get("kind") == "Custom":
        data = event.get("data")
        if isinstance(data, dict) and data.get("event_type") == event_type:
            return data.get("data")
    if len(record) == 1:
        return record
    return None


def _extract_value(sample: dict[str, Any], key: str) -> Any | None:
    if not key:
        return None
    metrics = sample.get("metrics")
    if isinstance(metrics, dict):
        extra = metrics.get("extra")
        if isinstance(extra, dict) and key in extra:
            return extra.get(key)
        if key in metrics:
            return metrics.get(key)
    return sample.get(key)


def _numeric_keys(events: Iterable[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for event in events:
        metrics = event.get("metrics")
        if isinstance(metrics, dict):
            extra = metrics.get("extra")
            if isinstance(extra, dict):
                for key, value in extra.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if math.isfinite(float(value)):
                            keys.add(key)
            for key, value in metrics.items():
                if key == "extra":
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if math.isfinite(float(value)):
                        keys.add(key)
        for key, value in event.items():
            if key == "metrics":
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isfinite(float(value)):
                    keys.add(key)
    return sorted(keys)


def _metric_stats(values: list[float]) -> dict[str, Any]:
    return {
        "first": values[0],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "sum": sum(values),
        "samples": len(values),
        "nonzero": sum(1 for value in values if value != 0.0),
    }


def _coherence_repair_summary(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    present = {
        key: value
        for key, value in metrics.items()
        if key in COHERENCE_REPAIR_METRIC_KEYS
    }
    if not present:
        return {}

    total = present.get("coherence_repairs_total")
    detected = present.get("coherence_repaired_detected")
    pre_discard = present.get("coherence_pre_discard_repairs_total")
    aggregate = present.get("coherence_repaired_weights_total")
    return {
        "keys": sorted(present),
        "total_nonzero_steps": int(total.get("nonzero", 0)) if total else 0,
        "max_total": float(total.get("max", 0.0)) if total else 0.0,
        "last_total": float(total.get("last", 0.0)) if total else 0.0,
        "detected_steps": int(detected.get("nonzero", 0)) if detected else 0,
        "max_pre_discard_total": float(pre_discard.get("max", 0.0)) if pre_discard else 0.0,
        "max_aggregate_total": float(aggregate.get("max", 0.0)) if aggregate else 0.0,
    }


def _metric_sum(metrics: dict[str, dict[str, Any]], key: str) -> float:
    entry = metrics.get(key)
    if not isinstance(entry, dict):
        return 0.0
    value = entry.get("sum")
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else 0.0


def _metric_last(metrics: dict[str, dict[str, Any]], key: str) -> float | None:
    entry = metrics.get(key)
    if not isinstance(entry, dict):
        return None
    value = entry.get("last")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _prefixed_metric_sums(
    metrics: dict[str, dict[str, Any]],
    prefix: str,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in sorted(metrics):
        if not key.startswith(prefix):
            continue
        total = _metric_sum(metrics, key)
        if total:
            out[key[len(prefix):]] = total
    return out


def _metric_fragment(value: str) -> str:
    out = []
    for char in value.lower():
        if char.isalnum():
            out.append(char)
        elif char in {"_", "-", ".", ":", "/"}:
            out.append("_")
    fragment = "".join(out).strip("_")
    while "__" in fragment:
        fragment = fragment.replace("__", "_")
    return fragment or "unknown"


def _backend_metric_fragment(value: str) -> str:
    fragment = _metric_fragment(value)
    if fragment == "wgpu_dense":
        return "wgpu"
    if fragment == "simd":
        return "cpu_simd"
    return fragment


def _is_metadata_only_backend(value: str) -> bool:
    return value in METADATA_ONLY_BACKENDS


def _finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    return None


def _insert_metric_value(
    metrics: dict[str, dict[str, Any]],
    key: str,
    value: float,
) -> None:
    metrics[key] = _metric_stats([value])


def _tensor_backend_metrics_from_tensor_meta(
    events: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    counts: dict[str, float] = {}
    fallback_count = 0.0

    def inc(key: str, amount: float = 1.0) -> None:
        counts[key] = counts.get(key, 0.0) + amount

    def record_sub_backend(
        op_fragment: str,
        requested_backend: str | None,
        field: str,
        backend: str,
        *,
        count_fallback: bool = True,
    ) -> None:
        nonlocal fallback_count
        backend_fragment = _backend_metric_fragment(backend)
        if backend_fragment == "auto":
            return
        component = field.removesuffix("_backend")
        inc(f"tensor_op_backend_{op_fragment}_{component}_{backend_fragment}")
        if count_fallback and requested_backend == "wgpu":
            if backend_fragment == "wgpu":
                inc("tensor_backend_requested_wgpu_component_hits")
                inc(
                    "tensor_op_backend_requested_wgpu_component_hit_"
                    f"{op_fragment}_{component}_{backend_fragment}"
                )
            elif not _is_metadata_only_backend(backend_fragment):
                inc("tensor_backend_requested_wgpu_component_fallbacks")
                inc(
                    "tensor_op_backend_requested_wgpu_component_fallback_"
                    f"{op_fragment}_{component}_{backend_fragment}"
                )
        if (
            count_fallback
            and requested_backend is not None
            and requested_backend != "auto"
            and requested_backend != backend_fragment
            and not _is_metadata_only_backend(backend_fragment)
        ):
            fallback_count += 1.0

    for event in events:
        op_name = event.get("op_name")
        data = event.get("data")
        if not isinstance(op_name, str) or not isinstance(data, dict):
            continue
        backend = data.get("backend")
        if not isinstance(backend, str) or not backend:
            continue

        op_fragment = _metric_fragment(op_name)
        backend_fragment = _backend_metric_fragment(backend)
        kernel_backend_fragment = _metric_fragment(backend)
        inc("tensor_ops_total")
        inc(f"tensor_backend_{backend_fragment}")
        inc(f"tensor_kernel_backend_{kernel_backend_fragment}")
        inc(f"tensor_op_{op_fragment}")
        inc(f"tensor_op_backend_{op_fragment}_{backend_fragment}")
        inc(f"tensor_op_kernel_backend_{op_fragment}_{kernel_backend_fragment}")

        requested_backend_value = data.get("requested_backend")
        requested_backend = (
            _backend_metric_fragment(requested_backend_value)
            if isinstance(requested_backend_value, str)
            else None
        )
        if (
            requested_backend is not None
            and requested_backend != "auto"
            and requested_backend != backend_fragment
            and not _is_metadata_only_backend(backend_fragment)
        ):
            fallback_count += 1.0

        for field, count_fallback in TENSOR_META_SUB_BACKEND_FIELD_SPECS:
            sub_backend = data.get(field)
            if not isinstance(sub_backend, str) or not sub_backend:
                continue
            record_sub_backend(
                op_fragment,
                requested_backend,
                field,
                sub_backend,
                count_fallback=count_fallback,
            )

    if counts:
        counts["tensor_backend_fallbacks"] = fallback_count

    metrics: dict[str, dict[str, Any]] = {}
    for key, value in counts.items():
        _insert_metric_value(metrics, key, value)
    return metrics


def _backend_policy_metrics_from_tensor_meta(
    events: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    counts: dict[str, float] = {}
    last_values: dict[str, float] = {}

    def inc(key: str, amount: float = 1.0) -> None:
        counts[key] = counts.get(key, 0.0) + amount

    for event in events:
        op_name = event.get("op_name")
        data = event.get("data")
        if not isinstance(op_name, str) or op_name not in BACKEND_POLICY_OPS:
            continue
        if not isinstance(data, dict):
            data = {}
        inc("backend_policy_events")
        if op_name == "wgpu_heuristic_choice":
            inc("backend_policy_wgpu_choices")
            field_map = {
                "workgroup": "backend_policy_wgpu_last_workgroup",
                "lanes": "backend_policy_wgpu_last_lanes",
                "compaction_tile": "backend_policy_wgpu_last_compaction_tile",
                "fft_radix": "backend_policy_wgpu_last_fft_radix",
                "fft_segments": "backend_policy_wgpu_last_fft_segments",
                "override_count": "backend_policy_wgpu_last_override_count",
            }
        elif op_name == "unison_rank_choice":
            inc("backend_policy_unison_choices")
            field_map = {
                "candidate_count": "backend_policy_unison_last_candidate_count",
                "best_score": "backend_policy_unison_last_best_score",
                "baseline_score": "backend_policy_unison_last_baseline_score",
                "wgpu_generated_score": "backend_policy_unison_last_wgpu_generated_score",
                "wgpu_generated_score_delta": "backend_policy_unison_last_wgpu_generated_score_delta",
            }
        elif op_name == "kdsl_env_bridge":
            inc("backend_policy_kdsl_env_events")
            field_map = {}
        elif op_name == "kdsl_kv_bridge":
            inc("backend_policy_kdsl_kv_events")
            field_map = {}
        elif op_name == "kv_consensus_soft_rules":
            inc("backend_policy_kv_soft_events")
            field_map = {}
        elif op_name == "wasm_tuner_choice":
            inc("backend_policy_wasm_tuner_events")
            field_map = {}
        elif op_name == "tensor_util_route":
            inc("backend_policy_tensor_util_routes")
            field_map = {
                "values": "backend_policy_tensor_util_last_values",
                "threshold": "backend_policy_tensor_util_last_threshold",
            }
        else:
            field_map = {}

        status = data.get("status")
        if isinstance(status, str) and status:
            inc(f"{BACKEND_POLICY_STATUS_PREFIX}{_metric_fragment(op_name)}_{_metric_fragment(status)}")
        source = data.get("choice_source")
        if isinstance(source, str) and source:
            inc(f"{BACKEND_POLICY_SOURCE_PREFIX}{_metric_fragment(op_name)}_{_metric_fragment(source)}")
        for source_field, metric_key in field_map.items():
            numeric = _finite_number(data.get(source_field))
            if numeric is not None:
                last_values[metric_key] = numeric

    metrics: dict[str, dict[str, Any]] = {}
    for key, value in counts.items():
        _insert_metric_value(metrics, key, value)
    for key, value in last_values.items():
        _insert_metric_value(metrics, key, value)
    return metrics


def _backend_request_metrics_from_tensor_meta(
    events: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    counts: dict[str, float] = {}

    def inc(key: str, amount: float = 1.0) -> None:
        counts[key] = counts.get(key, 0.0) + amount

    for event in events:
        op_name = event.get("op_name")
        data = event.get("data")
        if not isinstance(op_name, str) or not isinstance(data, dict):
            continue
        requested = data.get("requested_backend")
        backend = data.get("backend")
        if not isinstance(requested, str) or not isinstance(backend, str):
            continue
        if requested != "wgpu":
            continue
        backend_fragment = _backend_metric_fragment(backend)
        op_fragment = _metric_fragment(op_name)
        if backend == "wgpu" or backend == "wgpu_dense":
            inc("tensor_backend_requested_wgpu_hits")
            inc(f"tensor_op_backend_requested_wgpu_hit_{op_fragment}_{backend_fragment}")
        elif backend in CPU_RUNTIME_BACKENDS and _is_wgpu_runtime_fallback(data):
            inc("tensor_backend_requested_wgpu_runtime_fallbacks")
            inc(f"tensor_op_backend_wgpu_runtime_fallback_{op_fragment}_{backend_fragment}")

    metrics: dict[str, dict[str, Any]] = {}
    for key, value in counts.items():
        _insert_metric_value(metrics, key, value)
    return metrics


def _is_wgpu_runtime_fallback(data: dict[str, Any]) -> bool:
    fallback = data.get("fallback")
    if not isinstance(fallback, dict):
        return False
    if fallback.get("from") != "wgpu":
        return False
    if fallback.get("reason") == "runtime_unavailable":
        return True
    message = fallback.get("message")
    if not isinstance(message, str):
        return False
    return (
        "no suitable WGPU adapter" in message
        or "failed to initialize WGPU" in message
        or "WGPU backend not available" in message
    )


LSTM_BACKEND_FIELDS = {
    "lstm_forward": (
        ("backend", ""),
        ("input_projection_backend", "input_projection"),
        ("bias_backend", "bias"),
        ("recurrent_backend", "recurrent"),
        ("gate_activation_backend", "gate_activation"),
    ),
    "lstm_backward": (
        ("backend", ""),
        ("recurrent_backend", "recurrent"),
        ("gate_activation_backend", "gate_activation"),
        ("bptt_backend", "bptt"),
        ("bptt_scan_backend", "bptt_scan"),
        ("bptt_gate_derivative_backend", "bptt_gate_derivative"),
        ("bptt_cell_recurrence_backend", "bptt_cell_recurrence"),
        ("bptt_state_carry_backend", "bptt_state_carry"),
        ("input_gradient_backend", "input_gradient"),
        ("raw_parameter_gradient_backend", "raw_parameter_gradient"),
        ("parameter_gradient_reduction_backend", "parameter_gradient_reduction"),
        ("bias_gradient_backend", "bias_gradient"),
        ("parameter_gradient_scale_backend", "parameter_gradient_scale"),
    ),
}


def _lstm_backend_metrics_from_tensor_meta(
    events: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    counts: dict[str, float] = {}

    def inc(key: str, amount: float = 1.0) -> None:
        counts[key] = counts.get(key, 0.0) + amount

    for event in events:
        op_name = event.get("op_name")
        data = event.get("data")
        if not isinstance(op_name, str) or not isinstance(data, dict):
            continue
        field_specs = LSTM_BACKEND_FIELDS.get(op_name)
        if field_specs is None:
            continue
        op_fragment = _metric_fragment(op_name)
        for field, component in field_specs:
            backend = data.get(field)
            if not isinstance(backend, str) or not backend:
                continue
            backend_fragment = _metric_fragment(backend)
            component_fragment = _metric_fragment(component) if component else ""
            if component_fragment:
                key = f"tensor_op_backend_{op_fragment}_{component_fragment}_{backend_fragment}"
            else:
                key = f"tensor_op_backend_{op_fragment}_{backend_fragment}"
            inc(key)

    metrics: dict[str, dict[str, Any]] = {}
    for key, value in counts.items():
        _insert_metric_value(metrics, key, value)
    return metrics


def _lstm_estimated_metrics_from_tensor_meta(
    events: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    values = {
        "lstm_forward_estimated_gate_activation_ops": 0.0,
        "lstm_forward_estimated_gate_activation_cpu_debt_ops": 0.0,
        "lstm_forward_estimated_gate_activation_wgpu_ops": 0.0,
        "lstm_backward_estimated_gate_activation_ops": 0.0,
        "lstm_backward_estimated_gate_activation_cpu_debt_ops": 0.0,
        "lstm_backward_estimated_gate_activation_wgpu_ops": 0.0,
        "lstm_backward_estimated_bptt_ops": 0.0,
        "lstm_backward_estimated_bptt_cpu_debt_ops": 0.0,
        "lstm_backward_estimated_bptt_wgpu_ops": 0.0,
        "lstm_backward_estimated_bptt_gate_derivative_ops": 0.0,
        "lstm_backward_estimated_bptt_cell_recurrence_ops": 0.0,
        "lstm_backward_estimated_bptt_state_carry_ops": 0.0,
        "lstm_backward_estimated_bptt_scan_steps": 0.0,
        "lstm_backward_bptt_scan_shape_supported": 0.0,
        "lstm_backward_bptt_scan_runtime_requested": 0.0,
        "lstm_backward_bptt_scan_runtime_available": 0.0,
        "lstm_backward_bptt_scan_runtime_unavailable": 0.0,
        "lstm_backward_bptt_scan_elapsed_us": 0.0,
        "lstm_backward_bptt_scan_hidden_values": 0.0,
        "lstm_backward_bptt_scan_gate_values": 0.0,
        "lstm_backward_bptt_scan_cell_values": 0.0,
        "lstm_backward_bptt_scan_recurrent_weight_values": 0.0,
        "lstm_backward_bptt_scan_scratch_values": 0.0,
        "lstm_backward_bptt_scan_kernel_dispatches": 0.0,
        "lstm_backward_bptt_scan_serial_steps": 0.0,
        "lstm_backward_bptt_scan_workgroup_size": 0.0,
        "lstm_backward_bptt_scan_parallel_lanes": 0.0,
        "lstm_backward_estimated_bptt_ops_per_scan_step": 0.0,
    }
    saw_lstm_event = False
    for event in events:
        op_name = event.get("op_name")
        data = event.get("data")
        if not isinstance(op_name, str) or not isinstance(data, dict):
            continue
        if op_name == "lstm_forward":
            saw_lstm_event = True
            value = _finite_number(data.get("estimated_gate_activation_ops"))
            if value is not None:
                values["lstm_forward_estimated_gate_activation_ops"] += value
                if _metric_fragment(str(data.get("gate_activation_backend") or "cpu")) == "wgpu":
                    values["lstm_forward_estimated_gate_activation_wgpu_ops"] += value
                else:
                    values["lstm_forward_estimated_gate_activation_cpu_debt_ops"] += value
        elif op_name == "lstm_backward":
            saw_lstm_event = True
            field_map = {
                "estimated_bptt_ops": "lstm_backward_estimated_bptt_ops",
                "estimated_bptt_cpu_debt_ops": "lstm_backward_estimated_bptt_cpu_debt_ops",
                "estimated_bptt_wgpu_ops": "lstm_backward_estimated_bptt_wgpu_ops",
                "estimated_bptt_gate_derivative_ops": (
                    "lstm_backward_estimated_bptt_gate_derivative_ops"
                ),
                "estimated_bptt_cell_recurrence_ops": (
                    "lstm_backward_estimated_bptt_cell_recurrence_ops"
                ),
                "estimated_bptt_state_carry_ops": "lstm_backward_estimated_bptt_state_carry_ops",
                "estimated_bptt_scan_steps": "lstm_backward_estimated_bptt_scan_steps",
                "bptt_scan_elapsed_us": "lstm_backward_bptt_scan_elapsed_us",
                "bptt_scan_hidden_values": "lstm_backward_bptt_scan_hidden_values",
                "bptt_scan_gate_values": "lstm_backward_bptt_scan_gate_values",
                "bptt_scan_cell_values": "lstm_backward_bptt_scan_cell_values",
                "bptt_scan_recurrent_weight_values": (
                    "lstm_backward_bptt_scan_recurrent_weight_values"
                ),
                "bptt_scan_scratch_values": "lstm_backward_bptt_scan_scratch_values",
                "bptt_scan_kernel_dispatches": "lstm_backward_bptt_scan_kernel_dispatches",
                "bptt_scan_serial_steps": "lstm_backward_bptt_scan_serial_steps",
                "estimated_bptt_ops_per_scan_step": (
                    "lstm_backward_estimated_bptt_ops_per_scan_step"
                ),
            }
            for source_key, metric_key in field_map.items():
                value = _finite_number(data.get(source_key))
                if value is not None:
                    values[metric_key] += value
            last_value_field_map = {
                "bptt_scan_workgroup_size": "lstm_backward_bptt_scan_workgroup_size",
                "bptt_scan_parallel_lanes": "lstm_backward_bptt_scan_parallel_lanes",
            }
            for source_key, metric_key in last_value_field_map.items():
                value = _finite_number(data.get(source_key))
                if value is not None:
                    values[metric_key] = value
            gate_activation_ops = _finite_number(data.get("estimated_gate_activation_ops"))
            if gate_activation_ops is not None:
                values["lstm_backward_estimated_gate_activation_ops"] += gate_activation_ops
                if _metric_fragment(str(data.get("gate_activation_backend") or "cpu")) == "wgpu":
                    values["lstm_backward_estimated_gate_activation_wgpu_ops"] += (
                        gate_activation_ops
                    )
                else:
                    values["lstm_backward_estimated_gate_activation_cpu_debt_ops"] += (
                        gate_activation_ops
                    )
            bptt_ops = _finite_number(data.get("estimated_bptt_ops"))
            bptt_backend = data.get("bptt_backend") or data.get("bptt_scan_backend")
            if bptt_ops is not None:
                if bptt_backend == "wgpu" and data.get("estimated_bptt_wgpu_ops") is None:
                    values["lstm_backward_estimated_bptt_wgpu_ops"] += bptt_ops
                elif bptt_backend == "cpu" and data.get("estimated_bptt_cpu_debt_ops") is None:
                    values["lstm_backward_estimated_bptt_cpu_debt_ops"] += bptt_ops
            shape_supported = data.get("bptt_scan_shape_supported")
            runtime_requested = data.get("bptt_scan_runtime_requested")
            runtime_available = data.get("bptt_scan_runtime_available")
            if shape_supported is True:
                values["lstm_backward_bptt_scan_shape_supported"] += 1.0
            if runtime_requested is True:
                values["lstm_backward_bptt_scan_runtime_requested"] += 1.0
            if runtime_available is True:
                values["lstm_backward_bptt_scan_runtime_available"] += 1.0
            if runtime_requested is True and runtime_available is False:
                values["lstm_backward_bptt_scan_runtime_unavailable"] += 1.0

    gate_activation_ops = (
        values["lstm_forward_estimated_gate_activation_ops"]
        + values["lstm_backward_estimated_gate_activation_ops"]
    )
    gate_activation_cpu_debt_ops = (
        values["lstm_forward_estimated_gate_activation_cpu_debt_ops"]
        + values["lstm_backward_estimated_gate_activation_cpu_debt_ops"]
    )
    gate_activation_wgpu_ops = (
        values["lstm_forward_estimated_gate_activation_wgpu_ops"]
        + values["lstm_backward_estimated_gate_activation_wgpu_ops"]
    )
    values["lstm_estimated_gate_activation_ops"] = gate_activation_ops
    values["lstm_estimated_gate_activation_cpu_debt_ops"] = gate_activation_cpu_debt_ops
    values["lstm_estimated_gate_activation_wgpu_ops"] = gate_activation_wgpu_ops
    values["lstm_estimated_bptt_cpu_debt_ops"] = values[
        "lstm_backward_estimated_bptt_cpu_debt_ops"
    ]
    values["lstm_estimated_bptt_wgpu_ops"] = values["lstm_backward_estimated_bptt_wgpu_ops"]
    values["lstm_estimated_cpu_debt_ops"] = (
        gate_activation_cpu_debt_ops + values["lstm_estimated_bptt_cpu_debt_ops"]
    )

    metrics: dict[str, dict[str, Any]] = {}
    for key, value in values.items():
        if value or saw_lstm_event:
            _insert_metric_value(metrics, key, value)
    return metrics


def _backend_policy_summary(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not any(
        key in metrics
        or any(metric_key.startswith(key) for metric_key in metrics)
        for key in (
            "backend_policy_",
            BACKEND_POLICY_STATUS_PREFIX,
            BACKEND_POLICY_SOURCE_PREFIX,
        )
    ):
        return {}

    counts = {
        key.removeprefix("backend_policy_"): _metric_sum(metrics, key)
        for key in BACKEND_POLICY_COUNT_KEYS
        if key in metrics
    }
    last = {
        key.removeprefix("backend_policy_"): value
        for key in BACKEND_POLICY_LAST_KEYS
        for value in [_metric_last(metrics, key)]
        if value is not None
    }
    status_counts = _prefixed_metric_sums(metrics, BACKEND_POLICY_STATUS_PREFIX)
    source_counts = _prefixed_metric_sums(metrics, BACKEND_POLICY_SOURCE_PREFIX)
    if not counts and not last and not status_counts and not source_counts:
        return {}
    return {
        "counts": counts,
        "status_counts": status_counts,
        "source_counts": source_counts,
        "last": last,
    }


def load_trainer_trace_events(
    path: str | Path,
    *,
    event_type: str = "TrainerStep",
) -> list[dict[str, Any]]:
    """Load a trainer trace JSONL file recorded via `spiraltorch.plugin.record(...)`."""

    trace_path = Path(path)
    events: list[dict[str, Any]] = []
    for record in _iter_jsonl(trace_path):
        payload = _extract_payload(record, event_type=event_type)
        if not isinstance(payload, dict):
            continue
        event = dict(payload)
        if "ts" in record and "ts" not in event:
            event["ts"] = record["ts"]
        events.append(event)
    return events


def _tensor_meta_derived_metrics(path: str | Path) -> dict[str, dict[str, Any]]:
    tensor_meta_events = load_trainer_trace_events(path, event_type="TensorOpMeta")
    metrics: dict[str, dict[str, Any]] = {}
    metrics.update(_tensor_backend_metrics_from_tensor_meta(tensor_meta_events))
    metrics.update(_backend_policy_metrics_from_tensor_meta(tensor_meta_events))
    metrics.update(_backend_request_metrics_from_tensor_meta(tensor_meta_events))
    metrics.update(_lstm_backend_metrics_from_tensor_meta(tensor_meta_events))
    metrics.update(_lstm_estimated_metrics_from_tensor_meta(tensor_meta_events))
    return metrics


def summarize_trainer_trace_events(
    path: str | Path,
    *,
    event_type: str = "TrainerStep",
    keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compute simple aggregates for numeric values in a trainer trace JSONL file."""

    events = load_trainer_trace_events(path, event_type=event_type)
    tensor_meta_metrics = _tensor_meta_derived_metrics(path)
    if not events:
        return {
            "event_type": event_type,
            "count": 0,
            "first_step": None,
            "last_step": None,
            "metrics": tensor_meta_metrics,
            "coherence_repairs": {},
            "backend_policy": _backend_policy_summary(tensor_meta_metrics),
        }

    selected_keys = list(keys) if keys is not None else _numeric_keys(events)
    metrics: dict[str, dict[str, Any]] = {}
    for key in selected_keys:
        values: list[float] = []
        for event in events:
            value = _extract_value(event, key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric = float(value)
                if math.isfinite(numeric):
                    values.append(numeric)
        if not values:
            continue
        metrics[key] = _metric_stats(values)

    for key, value in tensor_meta_metrics.items():
        metrics.setdefault(key, value)

    first_step = next(
        (
            int(step)
            for event in events
            for step in [event.get("step")]
            if isinstance(step, (int, float)) and not isinstance(step, bool)
        ),
        None,
    )
    last_step = next(
        (
            int(step)
            for event in reversed(events)
            for step in [event.get("step")]
            if isinstance(step, (int, float)) and not isinstance(step, bool)
        ),
        None,
    )
    return {
        "event_type": event_type,
        "count": len(events),
        "first_step": first_step,
        "last_step": last_step,
        "metrics": metrics,
        "coherence_repairs": _coherence_repair_summary(metrics),
        "backend_policy": _backend_policy_summary(metrics),
    }


def write_trainer_trace_html(
    trace_jsonl: str | Path,
    html_path: str | Path | None = None,
    *,
    title: str = "SpiralTorch Trainer Trace",
    event_type: str = "TrainerStep",
    marker_event_type: str | None = "TrainerPhase",
) -> str:
    """Render a self-contained HTML viewer for a trainer trace JSONL file."""

    trace_jsonl = Path(trace_jsonl)
    events = load_trainer_trace_events(trace_jsonl, event_type=event_type)
    markers: list[dict[str, Any]] = []
    if marker_event_type:
        markers = load_trainer_trace_events(trace_jsonl, event_type=marker_event_type)
    html_path = Path(html_path) if html_path is not None else trace_jsonl.with_suffix(".html")
    payload = json.dumps(events, ensure_ascii=True)
    marker_payload = json.dumps(markers, ensure_ascii=True)
    spotlight_payload = json.dumps(TRACE_SPOTLIGHT_KEYS, ensure_ascii=True)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b0f14;
      --panel: #121826;
      --text: #e8eefc;
      --muted: #9ab0d0;
      --accent: #6ee7ff;
      --border: rgba(255,255,255,.08);
      --danger: #fb7185;
      --phase0: rgba(148,163,184,.18);
      --phase1: rgba(110,231,255,.14);
      --phase2: rgba(167,139,250,.14);
      --phase3: rgba(251,113,133,.14);
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      padding: 18px 20px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(110,231,255,.08), rgba(0,0,0,0));
    }}
    header h1 {{
      margin: 0;
      font-size: 16px;
      letter-spacing: .2px;
      color: var(--text);
    }}
    header p {{
      margin: 6px 0 0;
      font-size: 12px;
      color: var(--muted);
    }}
    main {{
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 14px;
      padding: 14px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
      overflow: hidden;
    }}
    .row {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .row label {{
      font-size: 12px;
      color: var(--muted);
    }}
    select {{
      width: 100%;
      padding: 8px 10px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: rgba(0,0,0,.25);
      color: var(--text);
      outline: none;
    }}
    input[type="range"] {{
      width: 100%;
    }}
    canvas {{
      width: 100%;
      height: auto;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: rgba(0,0,0,.25);
    }}
    pre {{
      margin: 0;
      padding: 10px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: rgba(0,0,0,.25);
      overflow: auto;
      max-height: 360px;
      font-size: 11px;
      color: #cfe0ff;
    }}
    .kv {{
      margin-top: 10px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      font-size: 12px;
      color: var(--muted);
    }}
    .kv div strong {{
      color: var(--text);
      font-weight: 600;
    }}
    .badge {{
      display: inline-flex;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(110,231,255,.08);
      color: var(--accent);
      font-size: 11px;
    }}
    .badge.danger {{
      background: rgba(251,113,133,.10);
      color: var(--danger);
    }}
    .legend {{
      margin-top: 10px;
      display: grid;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .swatch {{
      width: 14px;
      height: 10px;
      border-radius: 999px;
      border: 1px solid var(--border);
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p>event_type: <span class="badge" id="event-type">—</span> · steps: <span id="count">0</span> · markers: <span id="marker-count">0</span></p>
  </header>
  <main>
    <section class="panel" style="grid-column: 1;">
      <div class="row" style="justify-content: space-between;">
        <span class="badge" id="epoch">epoch —</span>
        <span class="badge">step <span id="step">—</span></span>
      </div>
      <div class="row" style="justify-content: space-between; margin-top: 10px;">
        <span class="badge" id="phase">phase —</span>
        <span class="badge" id="turnover">turnover —</span>
      </div>
      <div style="margin-top: 12px;">
        <label for="idx">sample index</label>
        <input id="idx" type="range" min="0" max="0" value="0" step="1"/>
      </div>
      <div style="margin-top: 12px;">
        <label for="key">plot key</label>
        <select id="key"></select>
      </div>
      <div class="kv" id="stats"></div>
      <div class="legend" id="phase-legend">
        <div class="legend-item"><span class="swatch" style="background: var(--phase0);"></span><span>0 · Background</span></div>
        <div class="legend-item"><span class="swatch" style="background: var(--phase1);"></span><span>1 · SymmetricPulse</span></div>
        <div class="legend-item"><span class="swatch" style="background: var(--phase2);"></span><span>2 · CascadeImbalance</span></div>
        <div class="legend-item"><span class="swatch" style="background: var(--phase3);"></span><span>3 · DiffuseDrift</span></div>
      </div>
      <div style="margin-top: 12px;">
        <span class="badge">raw sample</span>
        <pre id="raw"></pre>
      </div>
    </section>
    <section class="panel" style="grid-column: 2;">
      <div class="row" style="justify-content: space-between; margin-bottom: 10px;">
        <div class="row">
          <span class="badge">timeseries</span>
          <span style="font-size: 12px; color: var(--muted);">single-key line chart</span>
        </div>
      </div>
      <canvas id="plot" width="960" height="360"></canvas>
    </section>
  </main>

  <script id="trace-meta" type="application/json">{json.dumps({"event_type": event_type, "marker_event_type": marker_event_type}, ensure_ascii=True)}</script>
  <script id="trace-data" type="application/json">{payload}</script>
  <script id="trace-markers" type="application/json">{marker_payload}</script>
  <script id="trace-spotlight-keys" type="application/json">{spotlight_payload}</script>
  <script>
    const meta = JSON.parse(document.getElementById("trace-meta").textContent || "{{}}");
    const samples = JSON.parse(document.getElementById("trace-data").textContent || "[]");
    const markers = JSON.parse(document.getElementById("trace-markers").textContent || "[]");
    const spotlightKeys = JSON.parse(document.getElementById("trace-spotlight-keys").textContent || "[]");
    const idx = document.getElementById("idx");
    const count = document.getElementById("count");
    const markerCount = document.getElementById("marker-count");
    const stepEl = document.getElementById("step");
    const epochEl = document.getElementById("epoch");
    const keyEl = document.getElementById("key");
    const rawEl = document.getElementById("raw");
    const statsEl = document.getElementById("stats");
    const plot = document.getElementById("plot");
    const eventTypeEl = document.getElementById("event-type");
    const phaseEl = document.getElementById("phase");
    const turnoverEl = document.getElementById("turnover");

    eventTypeEl.textContent = meta.event_type || "TrainerStep";
    count.textContent = String(samples.length);
    markerCount.textContent = String(markers.length);
    idx.max = Math.max(0, samples.length - 1);

    const phaseMap = new Map([
      [0, {{ name: "Background", css: "var(--phase0)" }}],
      [1, {{ name: "SymmetricPulse", css: "var(--phase1)" }}],
      [2, {{ name: "CascadeImbalance", css: "var(--phase2)" }}],
      [3, {{ name: "DiffuseDrift", css: "var(--phase3)" }}],
    ]);
    const phaseLineColors = new Map([
      [0, "rgba(148,163,184,.55)"],
      [1, "rgba(110,231,255,.68)"],
      [2, "rgba(167,139,250,.68)"],
      [3, "rgba(251,113,133,.68)"],
    ]);

    function isFiniteNumber(v) {{
      return typeof v === "number" && Number.isFinite(v);
    }}

    function extractValue(sample, key) {{
      if (!sample || !key) return null;
      const metrics = sample.metrics || {{}};
      const extra = metrics.extra || {{}};
      if (isFiniteNumber(extra[key])) return extra[key];
      if (isFiniteNumber(metrics[key])) return metrics[key];
      if (isFiniteNumber(sample[key])) return sample[key];
      return null;
    }}

    function phaseFor(sample) {{
      const code = extractValue(sample, "spectral_label");
      if (!isFiniteNumber(code)) return null;
      const rounded = Math.round(code);
      if (!phaseMap.has(rounded)) return null;
      return {{ code: rounded, ...phaseMap.get(rounded) }};
    }}

    function turnoverFor(sample) {{
      const v = extractValue(sample, "spectral_turnover");
      return isFiniteNumber(v) ? v : null;
    }}

    function collectKeys() {{
      const keys = new Set();
      for (const s of samples) {{
        const metrics = (s && s.metrics) ? s.metrics : {{}};
        const extra = (metrics && metrics.extra) ? metrics.extra : {{}};
        for (const k of Object.keys(extra)) keys.add(k);
      }}
      for (const k of spotlightKeys) {{
        if (samples.some(s => extractValue(s, k) !== null)) keys.add(k);
      }}
      return Array.from(keys).sort();
    }}

    const keys = collectKeys();
    const defaultKey = (keys.includes("loss_weighted") ? "loss_weighted"
      : keys.includes("spectral_label") ? "spectral_label"
      : (keys[0] || ""));

    for (const k of keys) {{
      const opt = document.createElement("option");
      opt.value = k;
      opt.textContent = k;
      keyEl.appendChild(opt);
    }}
    keyEl.value = defaultKey;

    function computeSeries(key) {{
      const ys = [];
      const xs = [];
      for (let i = 0; i < samples.length; i++) {{
        const s = samples[i];
        xs.push(isFiniteNumber(s.step) ? s.step : i);
        ys.push(extractValue(s, key));
      }}
      return {{ xs, ys }};
    }}

    function drawLine(ctx, pts, color) {{
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      let started = false;
      for (const p of pts) {{
        if (p === null) {{
          started = false;
          continue;
        }}
        if (!started) {{
          ctx.moveTo(p.x, p.y);
          started = true;
        }} else {{
          ctx.lineTo(p.x, p.y);
        }}
      }}
      ctx.stroke();
    }}

    function renderPlot() {{
      const key = keyEl.value;
      const {{ xs, ys }} = computeSeries(key);
      const w = plot.width;
      const h = plot.height;
      const padL = 46, padR = 16, padT = 16, padB = 28;
      const ctx = plot.getContext("2d");
      ctx.clearRect(0, 0, w, h);

      const numeric = ys.filter(v => isFiniteNumber(v));
      let min = numeric.length ? Math.min(...numeric) : 0;
      let max = numeric.length ? Math.max(...numeric) : 1;
      if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {{
        const base = Number.isFinite(min) ? min : 0;
        min = base - 1;
        max = base + 1;
      }}
      const span = max - min;
      min = min - span * 0.05;
      max = max + span * 0.05;

      const x0 = xs[0] ?? 0;
      const x1 = xs[xs.length - 1] ?? Math.max(1, xs.length - 1);
      const xSpan = (x1 - x0) || 1;

      const toX = (x) => padL + ((x - x0) / xSpan) * (w - padL - padR);
      const toY = (v) => padT + ((max - v) / (max - min)) * (h - padT - padB);

      // phase background stripes (based on spectral_label)
      let lastPhase = null;
      let segStart = 0;
      for (let i = 0; i <= samples.length; i++) {{
        const p = (i < samples.length) ? phaseFor(samples[i]) : null;
        const code = p ? p.code : null;
        if (i === 0) {{
          lastPhase = code;
          segStart = 0;
          continue;
        }}
        if (code === lastPhase && i < samples.length) continue;
        if (lastPhase !== null) {{
          const left = toX(xs[segStart] ?? segStart);
          const right = (i < xs.length) ? toX(xs[i] ?? i) : (w - padR);
          const width = Math.max(0, right - left);
          const phaseInfo = phaseMap.get(lastPhase);
          if (phaseInfo) {{
            ctx.fillStyle = phaseInfo.css;
            ctx.fillRect(left, padT, width, h - padT - padB);
          }}
        }}
        lastPhase = code;
        segStart = i;
      }}

      // axes
      ctx.strokeStyle = "rgba(255,255,255,.14)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, padT);
      ctx.lineTo(padL, h - padB);
      ctx.lineTo(w - padR, h - padB);
      ctx.stroke();

      // y labels
      ctx.fillStyle = "rgba(154,176,208,.95)";
      ctx.font = "12px ui-sans-serif, system-ui";
      const ticks = 4;
      for (let i = 0; i <= ticks; i++) {{
        const t = i / ticks;
        const v = max - t * (max - min);
        const y = padT + t * (h - padT - padB);
        ctx.strokeStyle = "rgba(255,255,255,.06)";
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(w - padR, y);
        ctx.stroke();
        ctx.fillText(v.toFixed(4), 6, y + 4);
      }}

      // series
      const pts = ys.map((v, i) => {{
        if (!isFiniteNumber(v)) return null;
        return {{ x: toX(xs[i]), y: toY(v) }};
      }});
      drawLine(ctx, pts, "rgba(110,231,255,.95)");

      // marker events (TrainerPhase)
      for (const m of markers) {{
        if (!m || !isFiniteNumber(m.step)) continue;
        const x = toX(m.step);
        const kind = String(m.kind || "");
        let color = "rgba(110,231,255,.25)";
        let width = 1.0;
        if (kind === "turnover_spike") {{
          color = "rgba(251,113,133,.9)";
          width = 2.0;
        }} else if (kind === "label_change") {{
          let code = null;
          if (m.to && isFiniteNumber(m.to.code)) code = Math.round(m.to.code);
          if (code === null && isFiniteNumber(m.label_code)) code = Math.round(m.label_code);
          if (code !== null && phaseLineColors.has(code)) {{
            color = phaseLineColors.get(code);
            width = 1.75;
          }}
        }} else if (kind === "loss_spike") {{
          color = "rgba(250,204,21,.88)";
          width = 2.0;
        }} else if (kind === "drift_spike") {{
          color = "rgba(251,146,60,.88)";
          width = 2.0;
        }} else if (kind === "band_shift") {{
          color = "rgba(110,231,255,.55)";
          width = 1.5;
        }}
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.beginPath();
        ctx.moveTo(x, padT);
        ctx.lineTo(x, h - padB);
        ctx.stroke();
      }}

      // marker for current idx
      const i = Number(idx.value) || 0;
      const cx = toX(xs[i] ?? i);
      ctx.strokeStyle = "rgba(251,113,133,.9)";
      ctx.beginPath();
      ctx.moveTo(cx, padT);
      ctx.lineTo(cx, h - padB);
      ctx.stroke();

      ctx.fillStyle = "rgba(110,231,255,.95)";
      ctx.fillText(key, padL + 6, padT + 14);
    }}

    function renderSample() {{
      const i = Math.max(0, Math.min(samples.length - 1, Number(idx.value) || 0));
      const s = samples[i] || {{}};
      stepEl.textContent = String(isFiniteNumber(s.step) ? s.step : i);
      epochEl.textContent = "epoch " + String(isFiniteNumber(s.epoch) ? s.epoch : "—");
      const phase = phaseFor(s);
      phaseEl.textContent = phase ? (`phase ${{phase.code}} · ${{phase.name}}`) : "phase —";
      const turnover = turnoverFor(s);
      turnoverEl.textContent = turnover !== null ? (`turnover ${{turnover.toFixed(4)}}`) : "turnover —";
      rawEl.textContent = JSON.stringify(s, null, 2);

      const key = keyEl.value;
      const v = extractValue(s, key);
      const series = computeSeries(key).ys.filter(isFiniteNumber);
      const last = series.length ? series[series.length - 1] : null;
      const min = series.length ? Math.min(...series) : null;
      const max = series.length ? Math.max(...series) : null;
      const mean = series.length ? (series.reduce((a,b)=>a+b,0) / series.length) : null;

      statsEl.innerHTML = "";
      const entries = [
        ["value", v],
        ["last", last],
        ["min", min],
        ["max", max],
        ["mean", mean],
      ];
      for (const [label, value] of entries) {{
        const div = document.createElement("div");
        const formatted = isFiniteNumber(value) ? value.toFixed(6) : "—";
        div.innerHTML = `<strong>${{label}}</strong><br/>${{formatted}}`;
        statsEl.appendChild(div);
      }}
    }}

    idx.addEventListener("input", () => {{
      renderSample();
      renderPlot();
    }});
    keyEl.addEventListener("change", () => {{
      renderSample();
      renderPlot();
    }});

    renderSample();
    renderPlot();
  </script>
</body>
</html>
"""

    html_path.write_text(html, encoding="utf-8")
    return str(html_path)
