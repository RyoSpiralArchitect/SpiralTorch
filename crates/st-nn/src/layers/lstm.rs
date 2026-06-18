// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::execution::{
    current_backend_policy, current_matmul_backend, current_tensor_util_backend_for_values,
};
use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor};
use st_tensor::{
    emit_tensor_op, emit_tensor_op_meta, MatmulBackend, TensorError, TensorUtilBackend,
};
use std::cell::RefCell;
use std::time::Instant;

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn validate_finite_value(label: &'static str, value: f32) -> PureResult<()> {
    if !value.is_finite() {
        return Err(TensorError::NonFiniteValue { label, value });
    }
    Ok(())
}

fn checked_value(label: &'static str, value: f32) -> PureResult<f32> {
    validate_finite_value(label, value)?;
    Ok(value)
}

fn validate_finite_slice(label: &'static str, values: &[f32]) -> PureResult<()> {
    for &value in values {
        validate_finite_value(label, value)?;
    }
    Ok(())
}

fn validate_finite_tensor(label: &'static str, tensor: &Tensor) -> PureResult<()> {
    validate_finite_slice(label, tensor.data())
}

fn relabel_non_finite<T>(result: PureResult<T>, label: &'static str) -> PureResult<T> {
    match result {
        Err(TensorError::NonFiniteValue { value, .. }) => {
            Err(TensorError::NonFiniteValue { label, value })
        }
        other => other,
    }
}

fn matmul_backend_label(backend: MatmulBackend) -> &'static str {
    match backend {
        MatmulBackend::Auto => "auto",
        MatmulBackend::CpuFaer => "faer",
        MatmulBackend::CpuSimd => "cpu_simd",
        MatmulBackend::CpuNaive => "naive",
        #[cfg(feature = "wgpu")]
        MatmulBackend::GpuWgpu => "wgpu",
        #[cfg(feature = "hip")]
        MatmulBackend::GpuHip => "hip",
        #[allow(unreachable_patterns)]
        _ => "gpu",
    }
}

fn effective_matmul_backend_label(backend: MatmulBackend) -> &'static str {
    #[cfg(feature = "wgpu")]
    {
        if matches!(backend, MatmulBackend::GpuWgpu) && !st_tensor::wgpu_dense::is_available() {
            return "naive";
        }
    }
    matmul_backend_label(backend)
}

fn tensor_util_backend_label(backend: TensorUtilBackend) -> &'static str {
    match backend {
        TensorUtilBackend::Auto => "auto",
        TensorUtilBackend::Cpu => "cpu",
        TensorUtilBackend::GpuWgpu => "wgpu",
    }
}

fn effective_tensor_util_backend_label(backend: TensorUtilBackend) -> &'static str {
    #[cfg(feature = "wgpu")]
    {
        if matches!(backend, TensorUtilBackend::GpuWgpu) && !st_tensor::wgpu_dense::is_available() {
            return "cpu";
        }
    }
    tensor_util_backend_label(backend)
}

fn current_lstm_requested_label() -> &'static str {
    current_backend_policy()
        .map(|policy| policy.device_backend_label())
        .unwrap_or("auto")
}

fn summarize_backend_labels(labels: &[&'static str]) -> String {
    if let Some(first) = labels.first().copied() {
        if labels.iter().all(|label| *label == first) {
            return first.to_string();
        }
    }
    "mixed".to_string()
}

fn elapsed_us_since(start: Instant) -> u64 {
    let micros = start.elapsed().as_micros().min(u64::MAX as u128) as u64;
    micros.max(1)
}

fn emit_lstm_meta(
    op_name: &'static str,
    timesteps: usize,
    input_dim: usize,
    hidden_dim: usize,
    backward: bool,
    requested_backend: &'static str,
    layer_backend: &'static str,
    input_projection_backend: Option<&'static str>,
    bias_backend: Option<String>,
    recurrent_backend: Option<&'static str>,
    gate_activation_backend: Option<String>,
    gate_activation_fallback_reason: Option<String>,
    input_gradient_backend: Option<&'static str>,
    raw_parameter_gradient_backend: Option<String>,
    parameter_gradient_reduction_backend: Option<&'static str>,
    bias_gradient_backend: Option<&'static str>,
    parameter_gradient_scale_backend: Option<String>,
    bptt_scan_backend: Option<&'static str>,
    bptt_scan_kernel: Option<&'static str>,
    bptt_scan_lowering: Option<&'static str>,
    bptt_scan_fallback_reason: Option<String>,
    bptt_scan_shape_supported: Option<bool>,
    bptt_scan_runtime_requested: Option<bool>,
    bptt_scan_runtime_available: Option<bool>,
    bptt_scan_elapsed_us: Option<u64>,
    bptt_scan_workgroup_size: Option<usize>,
    bptt_scan_parallel_lanes: Option<usize>,
    bptt_scan_parallel_axis: Option<&'static str>,
) {
    emit_tensor_op(op_name, &[timesteps, input_dim], &[timesteps, hidden_dim]);
    emit_tensor_op_meta(op_name, || {
        let bptt_backend = if backward {
            bptt_scan_backend.or(Some("cpu"))
        } else {
            None
        };
        let gate_activation_backend =
            gate_activation_backend.unwrap_or_else(|| bptt_backend.unwrap_or("cpu").to_string());
        let gradient_scale = if backward && timesteps > 0 {
            Some(1.0 / (timesteps as f32))
        } else {
            None
        };
        let gate_width = hidden_dim.saturating_mul(4);
        let hidden_values = timesteps.saturating_mul(hidden_dim);
        let bptt_scan_gate_values = timesteps.saturating_mul(gate_width);
        let bptt_scan_cell_values = timesteps.saturating_add(1).saturating_mul(hidden_dim);
        let bptt_scan_recurrent_weight_values = gate_width.saturating_mul(hidden_dim);
        let bptt_scan_scratch_values = hidden_dim.saturating_mul(2);
        let non_empty_dispatches = usize::from(timesteps > 0 && input_dim > 0 && hidden_dim > 0);
        let input_projection_ops = timesteps
            .saturating_mul(input_dim)
            .saturating_mul(gate_width);
        let recurrent_projection_ops = timesteps
            .saturating_mul(hidden_dim)
            .saturating_mul(gate_width);
        let gate_activation_ops = timesteps.saturating_mul(hidden_dim).saturating_mul(8);
        let bptt_gate_derivative_ops = if backward {
            timesteps.saturating_mul(hidden_dim).saturating_mul(12)
        } else {
            0
        };
        let bptt_cell_recurrence_ops = if backward {
            timesteps.saturating_mul(hidden_dim).saturating_mul(4)
        } else {
            0
        };
        let bptt_state_carry_ops = if backward {
            timesteps.saturating_mul(hidden_dim).saturating_mul(2)
        } else {
            0
        };
        let bptt_ops = if backward {
            timesteps.saturating_mul(hidden_dim).saturating_mul(
                gate_width
                    .saturating_add(input_dim)
                    .saturating_add(hidden_dim),
            )
        } else {
            0
        };
        let bptt_cpu_debt_ops = if backward && bptt_backend == Some("cpu") {
            bptt_ops
        } else {
            0
        };
        let bptt_wgpu_ops = if backward && bptt_backend == Some("wgpu") {
            bptt_ops
        } else {
            0
        };
        let bptt_ops_per_scan_step = if backward && timesteps > 0 {
            Some(bptt_ops / timesteps)
        } else {
            None
        };
        let mut object = serde_json::Map::new();
        macro_rules! insert_meta {
            ($key:literal, $value:expr) => {
                object.insert($key.to_string(), serde_json::json!($value));
            };
        }
        insert_meta!("backend", layer_backend);
        insert_meta!("requested_backend", requested_backend);
        insert_meta!("kernel", "lstm.hybrid");
        insert_meta!(
            "kind",
            if backward {
                "recurrent_backward"
            } else {
                "recurrent_forward"
            }
        );
        insert_meta!("rows", timesteps);
        insert_meta!("cols", input_dim);
        insert_meta!("values", timesteps.saturating_mul(input_dim));
        insert_meta!("output_rows", timesteps);
        insert_meta!("output_cols", hidden_dim);
        insert_meta!("output_values", timesteps.saturating_mul(hidden_dim));
        insert_meta!("timesteps", timesteps);
        insert_meta!("input_dim", input_dim);
        insert_meta!("hidden_dim", hidden_dim);
        insert_meta!("gate_width", gate_width);
        insert_meta!("input_projection_backend", input_projection_backend);
        insert_meta!("bias_backend", bias_backend);
        insert_meta!("recurrent_backend", recurrent_backend);
        insert_meta!("gate_activation_backend", gate_activation_backend);
        insert_meta!(
            "gate_activation_fallback_reason",
            gate_activation_fallback_reason
        );
        insert_meta!("bptt_backend", bptt_backend);
        insert_meta!("bptt_scan_backend", bptt_backend);
        insert_meta!(
            "bptt_scan_kernel",
            if backward {
                bptt_scan_kernel.or(Some("lstm_backward_scan.cpu_fused_loop"))
            } else {
                None
            }
        );
        insert_meta!(
            "bptt_scan_lowering",
            if backward {
                bptt_scan_lowering.or(Some("host_reverse_recurrent_scan"))
            } else {
                None
            }
        );
        insert_meta!("bptt_scan_fallback_reason", bptt_scan_fallback_reason);
        insert_meta!(
            "bptt_scan_shape_supported",
            if backward {
                bptt_scan_shape_supported
            } else {
                None
            }
        );
        insert_meta!(
            "bptt_scan_runtime_requested",
            if backward {
                bptt_scan_runtime_requested
            } else {
                None
            }
        );
        insert_meta!(
            "bptt_scan_runtime_available",
            if backward {
                bptt_scan_runtime_available
            } else {
                None
            }
        );
        insert_meta!(
            "bptt_scan_elapsed_us",
            if backward { bptt_scan_elapsed_us } else { None }
        );
        insert_meta!(
            "bptt_scan_workgroup_size",
            if backward {
                bptt_scan_workgroup_size
            } else {
                None
            }
        );
        insert_meta!(
            "bptt_scan_parallel_lanes",
            if backward {
                bptt_scan_parallel_lanes
            } else {
                None
            }
        );
        insert_meta!(
            "bptt_scan_parallel_axis",
            if backward {
                bptt_scan_parallel_axis
            } else {
                None
            }
        );
        insert_meta!(
            "bptt_scan_hidden_values",
            if backward { Some(hidden_values) } else { None }
        );
        insert_meta!(
            "bptt_scan_gate_values",
            if backward {
                Some(bptt_scan_gate_values)
            } else {
                None
            }
        );
        insert_meta!(
            "bptt_scan_cell_values",
            if backward {
                Some(bptt_scan_cell_values)
            } else {
                None
            }
        );
        insert_meta!(
            "bptt_scan_recurrent_weight_values",
            if backward {
                Some(bptt_scan_recurrent_weight_values)
            } else {
                None
            }
        );
        insert_meta!(
            "bptt_scan_scratch_values",
            if backward {
                Some(bptt_scan_scratch_values)
            } else {
                None
            }
        );
        insert_meta!(
            "bptt_scan_kernel_dispatches",
            if backward {
                Some(usize::from(bptt_backend == Some("wgpu")))
            } else {
                None
            }
        );
        insert_meta!(
            "bptt_scan_serial_steps",
            if backward { Some(timesteps) } else { None }
        );
        insert_meta!("estimated_bptt_ops_per_scan_step", bptt_ops_per_scan_step);
        insert_meta!("input_gradient_backend", input_gradient_backend);
        insert_meta!(
            "input_gradient_mode",
            if backward {
                Some("batched_sequence_matmul")
            } else {
                None
            }
        );
        insert_meta!(
            "recurrent_gradient_mode",
            if backward {
                Some("reverse_time_step_matmul")
            } else {
                None
            }
        );
        insert_meta!(
            "raw_parameter_gradient_backend",
            raw_parameter_gradient_backend
        );
        insert_meta!(
            "parameter_gradient_reduction_backend",
            parameter_gradient_reduction_backend
        );
        insert_meta!("bias_gradient_backend", bias_gradient_backend);
        insert_meta!(
            "parameter_gradient_scale_backend",
            parameter_gradient_scale_backend
        );
        insert_meta!(
            "trainable_parameters",
            input_dim
                .saturating_mul(gate_width)
                .saturating_add(hidden_dim.saturating_mul(gate_width))
                .saturating_add(gate_width.saturating_mul(2))
        );
        insert_meta!("estimated_input_projection_ops", input_projection_ops);
        insert_meta!(
            "estimated_recurrent_projection_ops",
            recurrent_projection_ops
        );
        insert_meta!("estimated_gate_activation_ops", gate_activation_ops);
        insert_meta!("estimated_bptt_ops", bptt_ops);
        insert_meta!("estimated_bptt_cpu_debt_ops", bptt_cpu_debt_ops);
        insert_meta!("estimated_bptt_wgpu_ops", bptt_wgpu_ops);
        insert_meta!(
            "estimated_total_ops",
            input_projection_ops
                .saturating_add(recurrent_projection_ops)
                .saturating_add(gate_activation_ops)
                .saturating_add(bptt_ops)
        );
        insert_meta!("gradient_scale", gradient_scale);
        insert_meta!("parameter_gradient_scale", gradient_scale);
        insert_meta!(
            "input_gradient_scale",
            if backward { Some(1.0) } else { None }
        );
        insert_meta!("empty", timesteps == 0 || input_dim == 0 || hidden_dim == 0);
        let mut data = serde_json::Value::Object(object);
        if let Some(object) = data.as_object_mut() {
            object.insert(
                "bptt_gate_derivative_backend".to_string(),
                serde_json::json!(bptt_backend),
            );
            object.insert(
                "bptt_cell_recurrence_backend".to_string(),
                serde_json::json!(bptt_backend),
            );
            object.insert(
                "bptt_state_carry_backend".to_string(),
                serde_json::json!(bptt_backend),
            );
            object.insert(
                "bptt_sequence_dependency".to_string(),
                serde_json::json!(if backward {
                    Some("reverse_time_recurrence")
                } else {
                    None
                }),
            );
            object.insert(
                "bptt_fusion_candidate".to_string(),
                serde_json::json!(if backward {
                    Some("fused_lstm_backward_scan")
                } else {
                    None
                }),
            );
            object.insert(
                "bptt_scan_fusion_blocker".to_string(),
                serde_json::json!(if backward {
                    Some("grad_h_next_and_grad_c_next_recurrence")
                } else {
                    None
                }),
            );
            object.insert(
                "estimated_bptt_gate_derivative_ops".to_string(),
                serde_json::json!(bptt_gate_derivative_ops),
            );
            object.insert(
                "estimated_bptt_cell_recurrence_ops".to_string(),
                serde_json::json!(bptt_cell_recurrence_ops),
            );
            object.insert(
                "estimated_bptt_state_carry_ops".to_string(),
                serde_json::json!(bptt_state_carry_ops),
            );
            object.insert(
                "estimated_input_gradient_ops".to_string(),
                serde_json::json!(if backward {
                    timesteps
                        .saturating_mul(gate_width)
                        .saturating_mul(input_dim)
                } else {
                    0
                }),
            );
            object.insert(
                "estimated_input_projection_dispatches".to_string(),
                serde_json::json!(if backward { 0 } else { non_empty_dispatches }),
            );
            object.insert(
                "estimated_recurrent_projection_dispatches".to_string(),
                serde_json::json!(if backward { 0 } else { timesteps }),
            );
            object.insert(
                "estimated_input_gradient_dispatches".to_string(),
                serde_json::json!(if backward { non_empty_dispatches } else { 0 }),
            );
            object.insert(
                "estimated_recurrent_gradient_dispatches".to_string(),
                serde_json::json!(if backward { timesteps } else { 0 }),
            );
            object.insert(
                "estimated_parameter_gradient_matmul_dispatches".to_string(),
                serde_json::json!(if backward {
                    2 * non_empty_dispatches
                } else {
                    0
                }),
            );
            object.insert(
                "estimated_bptt_scan_steps".to_string(),
                serde_json::json!(if backward { timesteps } else { 0 }),
            );
        }
        data
    });
}

fn scale_with_current_tensor_util_backend(
    tensor: &Tensor,
    scale: f32,
    label: &'static str,
) -> PureResult<(Tensor, &'static str)> {
    validate_finite_value(label, scale)?;
    let backend = current_tensor_util_backend_for_values(tensor.data().len());
    let scaled = relabel_non_finite(tensor.scale_with_backend(scale, backend), label)?;
    validate_finite_tensor(label, &scaled)?;
    Ok((scaled, effective_tensor_util_backend_label(backend)))
}

fn transpose_with_current_tensor_util_backend(
    tensor: &Tensor,
    label: &'static str,
) -> PureResult<(Tensor, &'static str)> {
    let backend = current_tensor_util_backend_for_values(tensor.data().len());
    let transposed = relabel_non_finite(tensor.transpose_with_backend(backend), label)?;
    validate_finite_tensor(label, &transposed)?;
    Ok((transposed, effective_tensor_util_backend_label(backend)))
}

fn sum_axis0_with_current_tensor_util_backend(
    tensor: &Tensor,
    label: &'static str,
) -> PureResult<(Vec<f32>, &'static str)> {
    let backend = current_tensor_util_backend_for_values(tensor.data().len());
    let reduced = relabel_non_finite(tensor.try_sum_axis0_with_backend(backend), label)?;
    validate_finite_slice(label, &reduced)?;
    Ok((reduced, effective_tensor_util_backend_label(backend)))
}

/// Single-layer LSTM operating on sequences laid out along the batch axis.
#[derive(Debug)]
pub struct Lstm {
    input_dim: usize,
    hidden_dim: usize,
    weight_ih: Parameter,
    weight_hh: Parameter,
    bias_ih: Parameter,
    bias_hh: Parameter,
    hidden_state: RefCell<Tensor>,
    cell_state: RefCell<Tensor>,
    cache: RefCell<Option<LstmCache>>,
}

#[derive(Debug, Clone)]
struct LstmCache {
    inputs: Vec<f32>,
    gates_i: Vec<f32>,
    gates_f: Vec<f32>,
    gates_g: Vec<f32>,
    gates_o: Vec<f32>,
    hidden_states: Vec<f32>,
    cell_states: Vec<f32>,
    timesteps: usize,
    input_dim: usize,
    hidden_dim: usize,
}

impl LstmCache {
    fn new(timesteps: usize, input_dim: usize, hidden_dim: usize, h0: &[f32], c0: &[f32]) -> Self {
        let mut hidden_states = vec![0.0f32; (timesteps + 1) * hidden_dim];
        hidden_states[..hidden_dim].copy_from_slice(h0);
        let mut cell_states = vec![0.0f32; (timesteps + 1) * hidden_dim];
        cell_states[..hidden_dim].copy_from_slice(c0);
        Self {
            inputs: vec![0.0f32; timesteps * input_dim],
            gates_i: vec![0.0f32; timesteps * hidden_dim],
            gates_f: vec![0.0f32; timesteps * hidden_dim],
            gates_g: vec![0.0f32; timesteps * hidden_dim],
            gates_o: vec![0.0f32; timesteps * hidden_dim],
            hidden_states,
            cell_states,
            timesteps,
            input_dim,
            hidden_dim,
        }
    }

    fn validate(&self) -> PureResult<()> {
        validate_finite_slice("lstm_cache_inputs", &self.inputs)?;
        validate_finite_slice("lstm_cache_gate_i", &self.gates_i)?;
        validate_finite_slice("lstm_cache_gate_f", &self.gates_f)?;
        validate_finite_slice("lstm_cache_gate_g", &self.gates_g)?;
        validate_finite_slice("lstm_cache_gate_o", &self.gates_o)?;
        validate_finite_slice("lstm_cache_hidden_states", &self.hidden_states)?;
        validate_finite_slice("lstm_cache_cell_states", &self.cell_states)
    }
}

#[derive(Debug)]
struct LstmForwardGateStep {
    gates_i: Vec<f32>,
    gates_f: Vec<f32>,
    gates_g: Vec<f32>,
    gates_o: Vec<f32>,
    cell: Vec<f32>,
    hidden: Vec<f32>,
    backend: &'static str,
    fallback_reason: Option<String>,
}

#[cfg(feature = "wgpu")]
fn parse_lstm_forward_gate_step(
    buffer: Vec<f32>,
    hidden_dim: usize,
    backend: &'static str,
) -> PureResult<LstmForwardGateStep> {
    let expected = hidden_dim.saturating_mul(6);
    if buffer.len() != expected {
        return Err(TensorError::DataLength {
            expected,
            got: buffer.len(),
        });
    }
    validate_finite_slice("lstm_gate_activation_output", &buffer)?;
    Ok(LstmForwardGateStep {
        gates_i: buffer[..hidden_dim].to_vec(),
        gates_f: buffer[hidden_dim..2 * hidden_dim].to_vec(),
        gates_g: buffer[2 * hidden_dim..3 * hidden_dim].to_vec(),
        gates_o: buffer[3 * hidden_dim..4 * hidden_dim].to_vec(),
        cell: buffer[4 * hidden_dim..5 * hidden_dim].to_vec(),
        hidden: buffer[5 * hidden_dim..6 * hidden_dim].to_vec(),
        backend,
        fallback_reason: None,
    })
}

fn lstm_forward_gate_step_cpu(
    gates: &[f32],
    cell_prev: &[f32],
    hidden_dim: usize,
) -> PureResult<LstmForwardGateStep> {
    let gate_width = hidden_dim.saturating_mul(4);
    if gates.len() != gate_width {
        return Err(TensorError::DataLength {
            expected: gate_width,
            got: gates.len(),
        });
    }
    if cell_prev.len() != hidden_dim {
        return Err(TensorError::DataLength {
            expected: hidden_dim,
            got: cell_prev.len(),
        });
    }
    let mut gates_i = vec![0.0; hidden_dim];
    let mut gates_f = vec![0.0; hidden_dim];
    let mut gates_g = vec![0.0; hidden_dim];
    let mut gates_o = vec![0.0; hidden_dim];
    let mut cell = vec![0.0; hidden_dim];
    let mut hidden = vec![0.0; hidden_dim];
    for unit in 0..hidden_dim {
        let gi = checked_value("lstm_gate_i", sigmoid(gates[unit]))?;
        let gf = checked_value("lstm_gate_f", sigmoid(gates[hidden_dim + unit]))?;
        let gg = checked_value("lstm_gate_g", gates[2 * hidden_dim + unit].tanh())?;
        let go = checked_value("lstm_gate_o", sigmoid(gates[3 * hidden_dim + unit]))?;
        let forget = checked_value("lstm_cell_state", gf * cell_prev[unit])?;
        let input_update = checked_value("lstm_cell_state", gi * gg)?;
        let cell_value = checked_value("lstm_cell_state", forget + input_update)?;
        let tanh_cell = checked_value("lstm_hidden_state", cell_value.tanh())?;
        let hidden_value = checked_value("lstm_hidden_state", go * tanh_cell)?;
        gates_i[unit] = gi;
        gates_f[unit] = gf;
        gates_g[unit] = gg;
        gates_o[unit] = go;
        cell[unit] = cell_value;
        hidden[unit] = hidden_value;
    }
    Ok(LstmForwardGateStep {
        gates_i,
        gates_f,
        gates_g,
        gates_o,
        cell,
        hidden,
        backend: "cpu",
        fallback_reason: None,
    })
}

fn lstm_forward_gate_step_with_backend(
    gates: &[f32],
    cell_prev: &[f32],
    hidden_dim: usize,
    _backend: TensorUtilBackend,
) -> PureResult<LstmForwardGateStep> {
    #[cfg(feature = "wgpu")]
    {
        if matches!(_backend, TensorUtilBackend::GpuWgpu)
            && hidden_dim > 0
            && st_tensor::wgpu_dense::is_available()
        {
            match st_tensor::wgpu_dense::lstm_forward_gate_step(gates, cell_prev, hidden_dim) {
                Ok(buffer) => return parse_lstm_forward_gate_step(buffer, hidden_dim, "wgpu"),
                Err(message) => {
                    let mut step = lstm_forward_gate_step_cpu(gates, cell_prev, hidden_dim)?;
                    step.fallback_reason = Some(message);
                    return Ok(step);
                }
            }
        }
    }
    lstm_forward_gate_step_cpu(gates, cell_prev, hidden_dim)
}

#[derive(Debug, Clone)]
struct LstmBackwardScanRoute {
    backend: &'static str,
    kernel: &'static str,
    lowering: &'static str,
    recurrent_gradient_backend: &'static str,
    fallback_reason: Option<String>,
    shape_supported: bool,
    runtime_requested: bool,
    runtime_available: bool,
    elapsed_us: u64,
    workgroup_size: usize,
    parallel_lanes: usize,
    parallel_axis: &'static str,
}

#[derive(Debug)]
struct LstmBackwardScanResult {
    gate_gradients: Vec<f32>,
    route: LstmBackwardScanRoute,
}

fn lstm_backward_scan_cpu(
    cache: &LstmCache,
    grad_output: &Tensor,
    weight_hh_t: &Tensor,
    matmul_backend: MatmulBackend,
) -> PureResult<LstmBackwardScanResult> {
    let scan_start = Instant::now();
    let timesteps = cache.timesteps;
    let hidden_dim = cache.hidden_dim;
    let gate_width = 4 * hidden_dim;
    let mut gate_gradients = vec![0.0f32; timesteps * gate_width];
    let mut grad_h_next = vec![0.0f32; hidden_dim];
    let mut grad_c_next = vec![0.0f32; hidden_dim];
    for step in (0..timesteps).rev() {
        let grad_hidden_slice = &grad_output.data()[step * hidden_dim..(step + 1) * hidden_dim];
        let prev_cell = &cache.cell_states[step * hidden_dim..(step + 1) * hidden_dim];
        let curr_cell = &cache.cell_states[(step + 1) * hidden_dim..(step + 2) * hidden_dim];
        let mut gate_grad = vec![0.0f32; gate_width];
        for unit in 0..hidden_dim {
            let dh = checked_value(
                "lstm_backward_hidden_grad",
                grad_hidden_slice[unit] + grad_h_next[unit],
            )?;
            let o = cache.gates_o[step * hidden_dim + unit];
            let i = cache.gates_i[step * hidden_dim + unit];
            let f = cache.gates_f[step * hidden_dim + unit];
            let g = cache.gates_g[step * hidden_dim + unit];
            let tanh_c = checked_value("lstm_backward_tanh_cell", curr_cell[unit].tanh())?;
            let do_gate = checked_value("lstm_backward_gate_grad", dh * tanh_c * o * (1.0 - o))?;
            let dc_recurrent =
                checked_value("lstm_backward_cell_grad", dh * o * (1.0 - tanh_c * tanh_c))?;
            let dc = checked_value("lstm_backward_cell_grad", dc_recurrent + grad_c_next[unit])?;
            let di = checked_value("lstm_backward_gate_grad", dc * g * i * (1.0 - i))?;
            let dg = checked_value("lstm_backward_gate_grad", dc * i * (1.0 - g * g))?;
            let df = checked_value(
                "lstm_backward_gate_grad",
                dc * prev_cell[unit] * f * (1.0 - f),
            )?;
            grad_c_next[unit] = checked_value("lstm_backward_cell_grad", dc * f)?;
            gate_grad[unit] = di;
            gate_grad[hidden_dim + unit] = df;
            gate_grad[2 * hidden_dim + unit] = dg;
            gate_grad[3 * hidden_dim + unit] = do_gate;
        }
        gate_gradients[step * gate_width..(step + 1) * gate_width].copy_from_slice(&gate_grad);
        let gate_tensor = Tensor::from_vec(1, gate_width, gate_grad)?;
        let recurrent_projection = relabel_non_finite(
            gate_tensor.matmul_with_backend(weight_hh_t, matmul_backend),
            "lstm_backward_recurrent_grad",
        )?;
        validate_finite_tensor("lstm_backward_recurrent_grad", &recurrent_projection)?;
        let next_h = recurrent_projection.data().to_vec();
        validate_finite_slice("lstm_backward_recurrent_grad", &next_h)?;
        grad_h_next = next_h;
    }
    validate_finite_slice("lstm_gate_gradients", &gate_gradients)?;
    Ok(LstmBackwardScanResult {
        gate_gradients,
        route: LstmBackwardScanRoute {
            backend: "cpu",
            kernel: "lstm_backward_scan.cpu_fused_loop",
            lowering: "host_reverse_recurrent_scan",
            recurrent_gradient_backend: effective_matmul_backend_label(matmul_backend),
            fallback_reason: None,
            shape_supported: true,
            runtime_requested: false,
            runtime_available: false,
            elapsed_us: elapsed_us_since(scan_start),
            workgroup_size: 1,
            parallel_lanes: 1,
            parallel_axis: "none",
        },
    })
}

fn lstm_backward_scan(
    cache: &LstmCache,
    grad_output: &Tensor,
    weight_hh_t: &Tensor,
    matmul_backend: MatmulBackend,
) -> PureResult<LstmBackwardScanResult> {
    #[cfg(feature = "wgpu")]
    {
        if matches!(matmul_backend, MatmulBackend::GpuWgpu) {
            let shape_supported = st_tensor::wgpu_dense::lstm_backward_scan_shape_supported(
                cache.timesteps,
                cache.hidden_dim,
            );
            if !shape_supported {
                let mut scan =
                    lstm_backward_scan_cpu(cache, grad_output, weight_hh_t, matmul_backend)?;
                scan.route.shape_supported = false;
                scan.route.fallback_reason =
                    Some("lstm_backward_scan shape unsupported by WGPU helper".to_string());
                return Ok(scan);
            }
            let scan_start = Instant::now();
            match st_tensor::wgpu_dense::lstm_backward_scan(
                &cache.gates_i,
                &cache.gates_f,
                &cache.gates_g,
                &cache.gates_o,
                &cache.cell_states,
                grad_output.data(),
                weight_hh_t.data(),
                cache.timesteps,
                cache.hidden_dim,
            ) {
                Ok(gate_gradients) => {
                    validate_finite_slice("lstm_gate_gradients", &gate_gradients)?;
                    return Ok(LstmBackwardScanResult {
                        gate_gradients,
                        route: LstmBackwardScanRoute {
                            backend: "wgpu",
                            kernel: "lstm_backward_scan.wgsl",
                            lowering: "wgpu_single_workgroup_hidden_parallel_recurrence",
                            recurrent_gradient_backend: "wgpu",
                            fallback_reason: None,
                            shape_supported: true,
                            runtime_requested: true,
                            runtime_available: true,
                            elapsed_us: elapsed_us_since(scan_start),
                            workgroup_size:
                                st_tensor::wgpu_dense::lstm_backward_scan_workgroup_size(),
                            parallel_lanes:
                                st_tensor::wgpu_dense::lstm_backward_scan_workgroup_size(),
                            parallel_axis: "hidden",
                        },
                    });
                }
                Err(err) => {
                    let mut scan =
                        lstm_backward_scan_cpu(cache, grad_output, weight_hh_t, matmul_backend)?;
                    scan.route.fallback_reason = Some(err);
                    scan.route.shape_supported = true;
                    scan.route.runtime_requested = true;
                    scan.route.runtime_available = false;
                    return Ok(scan);
                }
            }
        }
    }
    lstm_backward_scan_cpu(cache, grad_output, weight_hh_t, matmul_backend)
}

impl Lstm {
    /// Creates a new LSTM layer with small deterministic parameters.
    pub fn new(name: impl Into<String>, input_dim: usize, hidden_dim: usize) -> PureResult<Self> {
        if input_dim == 0 || hidden_dim == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: input_dim,
                cols: hidden_dim,
            });
        }
        let name = name.into();
        let weight_ih = Tensor::from_fn(input_dim, 4 * hidden_dim, |row, col| {
            (((row * 13 + col * 7) % 17) as f32 + 1.0) * 0.01
        })?;
        let weight_hh = Tensor::from_fn(hidden_dim, 4 * hidden_dim, |row, col| {
            (((row * 11 + col * 5) % 23) as f32 + 1.0) * 0.01
        })?;
        let bias_ih = Tensor::zeros(1, 4 * hidden_dim)?;
        let bias_hh = Tensor::zeros(1, 4 * hidden_dim)?;
        let hidden_state = Tensor::zeros(1, hidden_dim)?;
        let cell_state = Tensor::zeros(1, hidden_dim)?;
        Ok(Self {
            input_dim,
            hidden_dim,
            weight_ih: Parameter::new(format!("{name}::weight_ih"), weight_ih),
            weight_hh: Parameter::new(format!("{name}::weight_hh"), weight_hh),
            bias_ih: Parameter::new(format!("{name}::bias_ih"), bias_ih),
            bias_hh: Parameter::new(format!("{name}::bias_hh"), bias_hh),
            hidden_state: RefCell::new(hidden_state),
            cell_state: RefCell::new(cell_state),
            cache: RefCell::new(None),
        })
    }

    /// Resets the hidden and cell state to zero.
    pub fn reset_state(&self) -> PureResult<()> {
        *self.hidden_state.borrow_mut() = Tensor::zeros(1, self.hidden_dim)?;
        *self.cell_state.borrow_mut() = Tensor::zeros(1, self.hidden_dim)?;
        Ok(())
    }

    /// Loads an explicit initial hidden and cell state.
    pub fn set_state(&self, hidden: &Tensor, cell: &Tensor) -> PureResult<()> {
        if hidden.shape() != (1, self.hidden_dim) || cell.shape() != (1, self.hidden_dim) {
            return Err(TensorError::ShapeMismatch {
                left: hidden.shape(),
                right: (1, self.hidden_dim),
            });
        }
        validate_finite_tensor("lstm_hidden_state", hidden)?;
        validate_finite_tensor("lstm_cell_state", cell)?;
        *self.hidden_state.borrow_mut() = hidden.clone();
        *self.cell_state.borrow_mut() = cell.clone();
        Ok(())
    }

    fn validate_parameters(&self) -> PureResult<()> {
        validate_finite_tensor("lstm_weight_ih", self.weight_ih.value())?;
        validate_finite_tensor("lstm_weight_hh", self.weight_hh.value())?;
        validate_finite_tensor("lstm_bias_ih", self.bias_ih.value())?;
        validate_finite_tensor("lstm_bias_hh", self.bias_hh.value())
    }

    fn guard_input(&self, input: &Tensor) -> PureResult<()> {
        let (rows, cols) = input.shape();
        if cols != self.input_dim {
            return Err(TensorError::ShapeMismatch {
                left: (rows, cols),
                right: (rows, self.input_dim),
            });
        }
        if rows == 0 {
            return Err(TensorError::EmptyInput("lstm_forward"));
        }
        Ok(())
    }
}

impl Module for Lstm {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        self.guard_input(input)?;
        self.cache.borrow_mut().take();
        validate_finite_tensor("lstm_input", input)?;
        self.validate_parameters()?;
        let (timesteps, _) = input.shape();
        let hidden_dim = self.hidden_dim;
        let input_dim = self.input_dim;
        let mut output = vec![0.0f32; timesteps * hidden_dim];
        let mut hidden_prev = self.hidden_state.borrow().data().to_vec();
        let mut cell_prev = self.cell_state.borrow().data().to_vec();
        validate_finite_slice("lstm_hidden_state", &hidden_prev)?;
        validate_finite_slice("lstm_cell_state", &cell_prev)?;
        let cache_template =
            LstmCache::new(timesteps, input_dim, hidden_dim, &hidden_prev, &cell_prev);
        let mut cache = cache_template;
        let weight_ih = self.weight_ih.value();
        let weight_hh = self.weight_hh.value();
        let bias_ih = self.bias_ih.value();
        let bias_hh = self.bias_hh.value();
        let matmul_backend = current_matmul_backend();
        let mut input_projection = input.matmul_with_backend(weight_ih, matmul_backend)?;
        let bias_backend = current_tensor_util_backend_for_values(input_projection.data().len());
        input_projection.add_row_inplace_with_backend(bias_ih.data(), bias_backend)?;
        validate_finite_tensor("lstm_input_projection", &input_projection)?;
        let gate_activation_backend =
            current_tensor_util_backend_for_values(hidden_dim.saturating_mul(6));
        let mut gate_activation_backends = Vec::with_capacity(timesteps);
        let mut gate_activation_fallback_reasons = Vec::new();
        for t in 0..timesteps {
            let input_slice = &input.data()[t * input_dim..(t + 1) * input_dim];
            cache.inputs[t * input_dim..(t + 1) * input_dim].copy_from_slice(input_slice);
            let gate_start = t * 4 * hidden_dim;
            let mut gates =
                input_projection.data()[gate_start..gate_start + 4 * hidden_dim].to_vec();
            let hidden_tensor = Tensor::from_vec(1, hidden_dim, hidden_prev.clone())?;
            let recurrent_projection =
                hidden_tensor.matmul_with_backend(weight_hh, matmul_backend)?;
            validate_finite_tensor("lstm_recurrent_projection", &recurrent_projection)?;
            for (gate, slot) in gates.iter_mut().enumerate() {
                let value = checked_value(
                    "lstm_gate_pre_activation",
                    *slot + bias_hh.data()[gate] + recurrent_projection.data()[gate],
                )?;
                validate_finite_value("lstm_gate_pre_activation", value)?;
                *slot = value;
            }
            let gate_step = lstm_forward_gate_step_with_backend(
                &gates,
                &cell_prev,
                hidden_dim,
                gate_activation_backend,
            )?;
            gate_activation_backends.push(gate_step.backend);
            if let Some(reason) = gate_step.fallback_reason.as_deref() {
                if !gate_activation_fallback_reasons
                    .iter()
                    .any(|existing| existing == reason)
                {
                    gate_activation_fallback_reasons.push(reason.to_string());
                }
            }
            for unit in 0..hidden_dim {
                let cell = gate_step.cell[unit];
                let hidden = gate_step.hidden[unit];
                cache.gates_i[t * hidden_dim + unit] = gate_step.gates_i[unit];
                cache.gates_f[t * hidden_dim + unit] = gate_step.gates_f[unit];
                cache.gates_g[t * hidden_dim + unit] = gate_step.gates_g[unit];
                cache.gates_o[t * hidden_dim + unit] = gate_step.gates_o[unit];
                cache.cell_states[(t + 1) * hidden_dim + unit] = cell;
                cache.hidden_states[(t + 1) * hidden_dim + unit] = hidden;
                cell_prev[unit] = cell;
                hidden_prev[unit] = hidden;
                output[t * hidden_dim + unit] = hidden;
            }
        }
        validate_finite_slice("lstm_output", &output)?;
        cache.validate()?;
        let next_hidden = Tensor::from_vec(1, hidden_dim, hidden_prev.clone())?;
        let next_cell = Tensor::from_vec(1, hidden_dim, cell_prev.clone())?;
        validate_finite_tensor("lstm_hidden_state", &next_hidden)?;
        validate_finite_tensor("lstm_cell_state", &next_cell)?;
        let output = Tensor::from_vec(timesteps, hidden_dim, output)?;
        *self.hidden_state.borrow_mut() = next_hidden;
        *self.cell_state.borrow_mut() = next_cell;
        *self.cache.borrow_mut() = Some(cache);
        emit_lstm_meta(
            "lstm_forward",
            timesteps,
            input_dim,
            hidden_dim,
            false,
            current_lstm_requested_label(),
            "hybrid",
            Some(effective_matmul_backend_label(matmul_backend)),
            Some(effective_tensor_util_backend_label(bias_backend).to_string()),
            Some(effective_matmul_backend_label(matmul_backend)),
            Some(summarize_backend_labels(&gate_activation_backends)),
            if gate_activation_fallback_reasons.is_empty() {
                None
            } else {
                Some(gate_activation_fallback_reasons.join("; "))
            },
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        Ok(output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        self.guard_input(input)?;
        if grad_output.shape().0 != input.shape().0 || grad_output.shape().1 != self.hidden_dim {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (input.shape().0, self.hidden_dim),
            });
        }
        validate_finite_tensor("lstm_backward_input", input)?;
        validate_finite_tensor("lstm_backward_grad_output", grad_output)?;
        self.validate_parameters()?;
        let cache = self
            .cache
            .borrow()
            .as_ref()
            .cloned()
            .ok_or(TensorError::InvalidValue {
                label: "lstm_cache_missing",
            })?;
        cache.validate()?;
        let timesteps = cache.timesteps;
        let input_dim = cache.input_dim;
        let hidden_dim = cache.hidden_dim;
        let weight_ih = self.weight_ih.value();
        let weight_hh = self.weight_hh.value();
        let gate_width = 4 * hidden_dim;
        let matmul_backend = current_matmul_backend();
        let matmul_backend_name = effective_matmul_backend_label(matmul_backend);
        let (weight_ih_t, _) =
            transpose_with_current_tensor_util_backend(weight_ih, "lstm_weight_ih_transpose")?;
        let (weight_hh_t, _) =
            transpose_with_current_tensor_util_backend(weight_hh, "lstm_weight_hh_transpose")?;
        let scan = lstm_backward_scan(&cache, grad_output, &weight_hh_t, matmul_backend)?;
        let gate_gradients = Tensor::from_vec(timesteps, gate_width, scan.gate_gradients)?;
        validate_finite_tensor("lstm_gate_gradients", &gate_gradients)?;
        let grad_input = relabel_non_finite(
            gate_gradients.matmul_with_backend(&weight_ih_t, matmul_backend),
            "lstm_backward_input_grad",
        )?;
        validate_finite_tensor("lstm_backward_input_grad", &grad_input)?;
        let input_history = Tensor::from_vec(timesteps, input_dim, cache.inputs.clone())?;
        let (input_history_t, _) = transpose_with_current_tensor_util_backend(
            &input_history,
            "lstm_backward_input_history_transpose",
        )?;
        let hidden_history = Tensor::from_vec(
            timesteps,
            hidden_dim,
            cache.hidden_states[..timesteps * hidden_dim].to_vec(),
        )?;
        let (hidden_history_t, _) = transpose_with_current_tensor_util_backend(
            &hidden_history,
            "lstm_backward_hidden_history_transpose",
        )?;
        let grad_w_ih_raw = relabel_non_finite(
            input_history_t.matmul_with_backend(&gate_gradients, matmul_backend),
            "lstm_weight_ih_grad",
        )?;
        validate_finite_tensor("lstm_weight_ih_grad", &grad_w_ih_raw)?;
        let grad_w_hh_raw = relabel_non_finite(
            hidden_history_t.matmul_with_backend(&gate_gradients, matmul_backend),
            "lstm_weight_hh_grad",
        )?;
        validate_finite_tensor("lstm_weight_hh_grad", &grad_w_hh_raw)?;
        let (bias_grad_raw, bias_gradient_backend) =
            sum_axis0_with_current_tensor_util_backend(&gate_gradients, "lstm_bias_grad")?;
        let grad_b_ih_raw = Tensor::from_vec(1, gate_width, bias_grad_raw.clone())?;
        let grad_b_hh_raw = Tensor::from_vec(1, gate_width, bias_grad_raw)?;
        let gradient_scale = 1.0 / (timesteps as f32);
        let (grad_w_ih, grad_w_ih_scale_backend) = scale_with_current_tensor_util_backend(
            &grad_w_ih_raw,
            gradient_scale,
            "lstm_weight_ih_grad",
        )?;
        let (grad_w_hh, grad_w_hh_scale_backend) = scale_with_current_tensor_util_backend(
            &grad_w_hh_raw,
            gradient_scale,
            "lstm_weight_hh_grad",
        )?;
        let (grad_b_ih, grad_b_ih_scale_backend) = scale_with_current_tensor_util_backend(
            &grad_b_ih_raw,
            gradient_scale,
            "lstm_bias_ih_grad",
        )?;
        let (grad_b_hh, grad_b_hh_scale_backend) = scale_with_current_tensor_util_backend(
            &grad_b_hh_raw,
            gradient_scale,
            "lstm_bias_hh_grad",
        )?;
        let parameter_gradient_scale_backend = summarize_backend_labels(&[
            grad_w_ih_scale_backend,
            grad_w_hh_scale_backend,
            grad_b_ih_scale_backend,
            grad_b_hh_scale_backend,
        ]);
        self.weight_ih.accumulate_euclidean(&grad_w_ih)?;
        self.weight_hh.accumulate_euclidean(&grad_w_hh)?;
        self.bias_ih.accumulate_euclidean(&grad_b_ih)?;
        self.bias_hh.accumulate_euclidean(&grad_b_hh)?;
        self.cache.borrow_mut().take();
        emit_lstm_meta(
            "lstm_backward",
            timesteps,
            input_dim,
            hidden_dim,
            true,
            current_lstm_requested_label(),
            "hybrid",
            None,
            None,
            Some(scan.route.recurrent_gradient_backend),
            Some(scan.route.backend.to_string()),
            scan.route.fallback_reason.clone(),
            Some(matmul_backend_name),
            Some("hybrid".to_string()),
            Some(matmul_backend_name),
            Some(bias_gradient_backend),
            Some(parameter_gradient_scale_backend),
            Some(scan.route.backend),
            Some(scan.route.kernel),
            Some(scan.route.lowering),
            scan.route.fallback_reason,
            Some(scan.route.shape_supported),
            Some(scan.route.runtime_requested),
            Some(scan.route.runtime_available),
            Some(scan.route.elapsed_us),
            Some(scan.route.workgroup_size),
            Some(scan.route.parallel_lanes),
            Some(scan.route.parallel_axis),
        );
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.weight_ih)?;
        visitor(&self.weight_hh)?;
        visitor(&self.bias_ih)?;
        visitor(&self.bias_hh)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.weight_ih)?;
        visitor(&mut self.weight_hh)?;
        visitor(&mut self.bias_ih)?;
        visitor(&mut self.bias_hh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "wgpu")]
    use crate::execution::{push_backend_policy, BackendPolicy};
    #[cfg(feature = "wgpu")]
    use st_core::backend::device_caps::DeviceCaps;
    #[cfg(feature = "wgpu")]
    use st_tensor::backend::wgpu_dense;
    use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

    fn observer_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    fn copy_params(
        lstm: &mut Lstm,
        weight_ih: &Tensor,
        weight_hh: &Tensor,
        bias_ih: &Tensor,
        bias_hh: &Tensor,
    ) {
        *lstm.weight_ih.value_mut() = weight_ih.clone();
        *lstm.weight_hh.value_mut() = weight_hh.clone();
        *lstm.bias_ih.value_mut() = bias_ih.clone();
        *lstm.bias_hh.value_mut() = bias_hh.clone();
    }

    fn sequence_loss(output: &Tensor, grad_out: &Tensor, scale: f32) -> f32 {
        output
            .data()
            .iter()
            .zip(grad_out.data())
            .map(|(value, grad)| *value * *grad)
            .sum::<f32>()
            * scale
    }

    fn lstm_loss_with_weight_ih_delta(
        weight_ih: &Tensor,
        weight_hh: &Tensor,
        bias_ih: &Tensor,
        bias_hh: &Tensor,
        input: &Tensor,
        grad_out: &Tensor,
        parameter_index: usize,
        delta: f32,
        scale: f32,
    ) -> PureResult<f32> {
        let mut lstm = Lstm::new("probe", input.shape().1, grad_out.shape().1)?;
        copy_params(&mut lstm, weight_ih, weight_hh, bias_ih, bias_hh);
        lstm.weight_ih.value_mut().data_mut()[parameter_index] += delta;
        let output = lstm.forward(input)?;
        Ok(sequence_loss(&output, grad_out, scale))
    }

    fn lstm_loss_with_input_delta(
        weight_ih: &Tensor,
        weight_hh: &Tensor,
        bias_ih: &Tensor,
        bias_hh: &Tensor,
        input: &Tensor,
        grad_out: &Tensor,
        input_index: usize,
        delta: f32,
        scale: f32,
    ) -> PureResult<f32> {
        let mut lstm = Lstm::new("probe", input.shape().1, grad_out.shape().1)?;
        copy_params(&mut lstm, weight_ih, weight_hh, bias_ih, bias_hh);
        let mut perturbed = input.clone();
        perturbed.data_mut()[input_index] += delta;
        let output = lstm.forward(&perturbed)?;
        Ok(sequence_loss(&output, grad_out, scale))
    }

    fn max_abs_entry(data: &[f32]) -> (usize, f32) {
        let mut best = (0, data[0]);
        for (index, &value) in data.iter().enumerate().skip(1) {
            if value.abs() > best.1.abs() {
                best = (index, value);
            }
        }
        best
    }

    fn assert_close(label: &str, actual: f32, expected: f32, tolerance: f32) {
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "{label}: actual={actual}, expected={expected}, error={error}, tolerance={tolerance}"
        );
    }

    fn assert_non_finite_label<T>(result: PureResult<T>, expected: &'static str) {
        match result {
            Err(TensorError::NonFiniteValue { label, .. }) => assert_eq!(label, expected),
            Err(error) => panic!("expected non-finite label {expected}, got {error:?}"),
            Ok(_) => panic!("expected non-finite label {expected}"),
        }
    }

    #[cfg(feature = "wgpu")]
    fn run_wgpu_runtime_tests() -> bool {
        std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS")
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
            .unwrap_or(false)
    }

    fn reference_lstm_gate_gradients(
        cache: &LstmCache,
        grad_output: &Tensor,
        weight_hh_t: &Tensor,
    ) -> Vec<f32> {
        let timesteps = cache.timesteps;
        let hidden_dim = cache.hidden_dim;
        let gate_width = 4 * hidden_dim;
        let mut gate_gradients = vec![0.0f32; timesteps * gate_width];
        let mut grad_h_next = vec![0.0f32; hidden_dim];
        let mut grad_c_next = vec![0.0f32; hidden_dim];

        for step in (0..timesteps).rev() {
            let hidden_base = step * hidden_dim;
            let prev_cell = &cache.cell_states[hidden_base..hidden_base + hidden_dim];
            let curr_cell = &cache.cell_states[(step + 1) * hidden_dim..(step + 2) * hidden_dim];
            let gate_base = step * gate_width;
            for unit in 0..hidden_dim {
                let value_idx = hidden_base + unit;
                let dh = grad_output.data()[value_idx] + grad_h_next[unit];
                let o = cache.gates_o[value_idx];
                let i = cache.gates_i[value_idx];
                let f = cache.gates_f[value_idx];
                let g = cache.gates_g[value_idx];
                let tanh_c = curr_cell[unit].tanh();
                let do_gate = dh * tanh_c * o * (1.0 - o);
                let dc_recurrent = dh * o * (1.0 - tanh_c * tanh_c);
                let dc = dc_recurrent + grad_c_next[unit];
                let di = dc * g * i * (1.0 - i);
                let dg = dc * i * (1.0 - g * g);
                let df = dc * prev_cell[unit] * f * (1.0 - f);
                grad_c_next[unit] = dc * f;
                gate_gradients[gate_base + unit] = di;
                gate_gradients[gate_base + hidden_dim + unit] = df;
                gate_gradients[gate_base + 2 * hidden_dim + unit] = dg;
                gate_gradients[gate_base + 3 * hidden_dim + unit] = do_gate;
            }

            for unit in 0..hidden_dim {
                let mut sum = 0.0f32;
                for gate in 0..gate_width {
                    sum += gate_gradients[gate_base + gate]
                        * weight_hh_t.data()[gate * hidden_dim + unit];
                }
                grad_h_next[unit] = sum;
            }
        }

        gate_gradients
    }

    #[test]
    fn lstm_forward_produces_hidden_sequence() {
        let lstm = Lstm::new("lstm", 2, 3).unwrap();
        let input = Tensor::from_vec(4, 2, vec![0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8]).unwrap();
        let output = lstm.forward(&input).unwrap();
        assert_eq!(output.shape(), (4, 3));
        for value in output.data() {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn lstm_set_state_rejects_non_finite_without_mutating_state() {
        let lstm = Lstm::new("lstm", 2, 2).unwrap();
        let hidden_before = lstm.hidden_state.borrow().clone();
        let cell_before = lstm.cell_state.borrow().clone();
        let bad_hidden = Tensor::from_vec(1, 2, vec![0.0, f32::NAN]).unwrap();
        let cell = Tensor::zeros(1, 2).unwrap();

        assert_non_finite_label(lstm.set_state(&bad_hidden, &cell), "lstm_hidden_state");
        assert_eq!(*lstm.hidden_state.borrow(), hidden_before);
        assert_eq!(*lstm.cell_state.borrow(), cell_before);
    }

    #[test]
    fn lstm_forward_rejects_non_finite_input_without_mutating_state_and_clears_cache() {
        let lstm = Lstm::new("lstm", 2, 2).unwrap();
        let input = Tensor::from_vec(2, 2, vec![0.2, -0.1, 0.4, 0.3]).unwrap();
        let _ = lstm.forward(&input).unwrap();
        assert!(lstm.cache.borrow().is_some());
        let hidden_before = lstm.hidden_state.borrow().clone();
        let cell_before = lstm.cell_state.borrow().clone();
        let mut bad_input = input.clone();
        bad_input.data_mut()[2] = f32::INFINITY;

        assert_non_finite_label(lstm.forward(&bad_input), "lstm_input");
        assert_eq!(*lstm.hidden_state.borrow(), hidden_before);
        assert_eq!(*lstm.cell_state.borrow(), cell_before);
        assert!(lstm.cache.borrow().is_none());
    }

    #[test]
    fn lstm_forward_rejects_non_finite_parameter_without_mutating_state() {
        let mut lstm = Lstm::new("lstm", 2, 2).unwrap();
        let hidden_before = lstm.hidden_state.borrow().clone();
        let cell_before = lstm.cell_state.borrow().clone();
        lstm.weight_ih.value_mut().data_mut()[0] = f32::NAN;
        let input = Tensor::from_vec(1, 2, vec![0.2, -0.1]).unwrap();

        assert_non_finite_label(lstm.forward(&input), "lstm_weight_ih");
        assert_eq!(*lstm.hidden_state.borrow(), hidden_before);
        assert_eq!(*lstm.cell_state.borrow(), cell_before);
        assert!(lstm.cache.borrow().is_none());
    }

    #[test]
    fn lstm_backward_accumulates_gradients() {
        let mut lstm = Lstm::new("lstm", 3, 2).unwrap();
        let input =
            Tensor::from_vec(3, 3, vec![0.2, -0.1, 0.3, 0.4, -0.5, 0.6, -0.2, 0.1, 0.7]).unwrap();
        let grad_out = Tensor::from_vec(3, 2, vec![0.1, -0.2, 0.3, 0.2, -0.4, 0.5]).unwrap();
        let _ = lstm.forward(&input).unwrap();
        let grad_input = lstm.backward(&input, &grad_out).unwrap();
        assert_eq!(grad_input.shape(), (3, 3));
        assert!(lstm.weight_ih.gradient().is_some());
        assert!(lstm.bias_hh.gradient().is_some());
    }

    #[test]
    fn lstm_backward_rejects_non_finite_grad_without_consuming_cache_or_accumulating() {
        let mut lstm = Lstm::new("lstm", 2, 2).unwrap();
        let input = Tensor::from_vec(2, 2, vec![0.2, -0.1, 0.4, 0.3]).unwrap();
        let _ = lstm.forward(&input).unwrap();
        let mut grad_out = Tensor::from_vec(2, 2, vec![0.3, -0.2, 0.1, 0.4]).unwrap();
        grad_out.data_mut()[1] = f32::NAN;

        assert_non_finite_label(
            lstm.backward(&input, &grad_out),
            "lstm_backward_grad_output",
        );
        assert!(lstm.cache.borrow().is_some());
        assert!(lstm.weight_ih.gradient().is_none());
        assert!(lstm.weight_hh.gradient().is_none());
        assert!(lstm.bias_ih.gradient().is_none());
        assert!(lstm.bias_hh.gradient().is_none());
    }

    #[test]
    fn lstm_backward_rejects_overflowing_input_grad_without_accumulating() {
        let mut lstm = Lstm::new("lstm", 1, 1).unwrap();
        lstm.weight_ih.value_mut().data_mut()[2] = f32::MAX;
        let input = Tensor::from_vec(1, 1, vec![0.0]).unwrap();
        let _ = lstm.forward(&input).unwrap();
        let grad_out = Tensor::from_vec(1, 1, vec![8.0]).unwrap();

        assert_non_finite_label(lstm.backward(&input, &grad_out), "lstm_backward_input_grad");
        assert!(lstm.cache.borrow().is_some());
        assert!(lstm.weight_ih.gradient().is_none());
        assert!(lstm.weight_hh.gradient().is_none());
        assert!(lstm.bias_ih.gradient().is_none());
        assert!(lstm.bias_hh.gradient().is_none());
    }

    #[test]
    fn lstm_parameter_gradient_matches_timestep_averaged_loss() {
        let mut lstm = Lstm::new("lstm", 2, 2).unwrap();
        let input = Tensor::from_vec(3, 2, vec![0.2, -0.1, 0.4, 0.3, -0.3, 0.5]).unwrap();
        let grad_out = Tensor::from_vec(3, 2, vec![0.3, -0.2, 0.1, 0.4, -0.5, 0.2]).unwrap();
        let weight_ih = lstm.weight_ih.value().clone();
        let weight_hh = lstm.weight_hh.value().clone();
        let bias_ih = lstm.bias_ih.value().clone();
        let bias_hh = lstm.bias_hh.value().clone();

        let _ = lstm.forward(&input).unwrap();
        let _ = lstm.backward(&input, &grad_out).unwrap();

        let gradient = lstm.weight_ih.gradient().expect("weight_ih gradient");
        let (parameter_index, analytic) = max_abs_entry(gradient.data());
        assert!(analytic.abs() > 1.0e-6);

        let eps = 1.0e-3;
        let loss_scale = 1.0 / (input.shape().0 as f32);
        let plus = lstm_loss_with_weight_ih_delta(
            &weight_ih,
            &weight_hh,
            &bias_ih,
            &bias_hh,
            &input,
            &grad_out,
            parameter_index,
            eps,
            loss_scale,
        )
        .unwrap();
        let minus = lstm_loss_with_weight_ih_delta(
            &weight_ih,
            &weight_hh,
            &bias_ih,
            &bias_hh,
            &input,
            &grad_out,
            parameter_index,
            -eps,
            loss_scale,
        )
        .unwrap();
        let numerical = (plus - minus) / (2.0 * eps);
        let tolerance = 5.0e-4_f32.max(numerical.abs() * 0.08);
        assert_close(
            "weight_ih averaged gradient",
            analytic,
            numerical,
            tolerance,
        );
    }

    #[test]
    fn lstm_input_gradient_matches_unaveraged_loss() {
        let mut lstm = Lstm::new("lstm", 2, 2).unwrap();
        let input = Tensor::from_vec(3, 2, vec![0.2, -0.1, 0.4, 0.3, -0.3, 0.5]).unwrap();
        let grad_out = Tensor::from_vec(3, 2, vec![0.3, -0.2, 0.1, 0.4, -0.5, 0.2]).unwrap();
        let weight_ih = lstm.weight_ih.value().clone();
        let weight_hh = lstm.weight_hh.value().clone();
        let bias_ih = lstm.bias_ih.value().clone();
        let bias_hh = lstm.bias_hh.value().clone();

        let _ = lstm.forward(&input).unwrap();
        let grad_input = lstm.backward(&input, &grad_out).unwrap();

        let (input_index, analytic) = max_abs_entry(grad_input.data());
        assert!(analytic.abs() > 1.0e-6);

        let eps = 1.0e-3;
        let plus = lstm_loss_with_input_delta(
            &weight_ih,
            &weight_hh,
            &bias_ih,
            &bias_hh,
            &input,
            &grad_out,
            input_index,
            eps,
            1.0,
        )
        .unwrap();
        let minus = lstm_loss_with_input_delta(
            &weight_ih,
            &weight_hh,
            &bias_ih,
            &bias_hh,
            &input,
            &grad_out,
            input_index,
            -eps,
            1.0,
        )
        .unwrap();
        let numerical = (plus - minus) / (2.0 * eps);
        let tolerance = 5.0e-4_f32.max(numerical.abs() * 0.08);
        assert_close("input unaveraged gradient", analytic, numerical, tolerance);
    }

    #[test]
    fn lstm_backward_scan_cpu_matches_reference_gate_gradients() {
        let lstm = Lstm::new("lstm", 2, 2).unwrap();
        let input =
            Tensor::from_vec(4, 2, vec![0.2, -0.1, 0.4, 0.3, -0.3, 0.5, 0.1, -0.4]).unwrap();
        let grad_out =
            Tensor::from_vec(4, 2, vec![0.3, -0.2, 0.1, 0.4, -0.5, 0.2, 0.6, -0.1]).unwrap();

        let _ = lstm.forward(&input).unwrap();
        let cache = lstm.cache.borrow().as_ref().cloned().unwrap();
        let weight_hh_t = lstm
            .weight_hh
            .value()
            .transpose_with_backend(TensorUtilBackend::Cpu)
            .unwrap();
        let expected = reference_lstm_gate_gradients(&cache, &grad_out, &weight_hh_t);
        let scan = lstm_backward_scan_cpu(&cache, &grad_out, &weight_hh_t, MatmulBackend::CpuNaive)
            .unwrap();

        assert_eq!(scan.route.backend, "cpu");
        assert_eq!(scan.route.kernel, "lstm_backward_scan.cpu_fused_loop");
        assert_eq!(scan.route.lowering, "host_reverse_recurrent_scan");
        assert_eq!(scan.route.recurrent_gradient_backend, "naive");
        assert!(scan.route.fallback_reason.is_none());
        assert!(scan.route.shape_supported);
        assert!(!scan.route.runtime_requested);
        assert!(!scan.route.runtime_available);
        assert!(scan.route.elapsed_us > 0);
        assert_eq!(scan.gate_gradients.len(), expected.len());
        for (idx, (&actual, &expected)) in
            scan.gate_gradients.iter().zip(expected.iter()).enumerate()
        {
            assert_close(&format!("gate gradient {idx}"), actual, expected, 1.0e-6);
        }
    }

    #[test]
    fn lstm_forward_backward_emit_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut lstm = Lstm::new("lstm", 3, 2).unwrap();
        let input =
            Tensor::from_vec(3, 3, vec![0.2, -0.1, 0.3, 0.4, -0.5, 0.6, -0.2, 0.1, 0.7]).unwrap();
        let grad_out = Tensor::from_vec(3, 2, vec![0.1, -0.2, 0.3, 0.2, -0.4, 0.5]).unwrap();
        let _ = lstm.forward(&input).unwrap();
        let _ = lstm.backward(&input, &grad_out).unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        let forward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "lstm_forward"
                    && data["timesteps"] == 3
                    && data["input_dim"] == 3
                    && data["hidden_dim"] == 2
            })
            .expect("lstm forward metadata event");
        assert_eq!(forward.1["backend"], "hybrid");
        assert_eq!(forward.1["kind"], "recurrent_forward");
        assert_eq!(forward.1["kernel"], "lstm.hybrid");
        assert_eq!(forward.1["gate_width"], 8);
        assert_eq!(forward.1["input_projection_backend"], "auto");
        assert_eq!(forward.1["bias_backend"], "auto");
        assert_eq!(forward.1["recurrent_backend"], "auto");
        assert_eq!(forward.1["gate_activation_backend"], "cpu");
        assert_eq!(forward.1["estimated_input_projection_dispatches"], 1);
        assert_eq!(forward.1["estimated_recurrent_projection_dispatches"], 3);
        assert!(forward.1["estimated_total_ops"].as_u64().unwrap_or(0) > 0);

        let backward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "lstm_backward"
                    && data["timesteps"] == 3
                    && data["input_dim"] == 3
                    && data["hidden_dim"] == 2
            })
            .expect("lstm backward metadata event");
        assert_eq!(backward.1["backend"], "hybrid");
        assert_eq!(backward.1["kind"], "recurrent_backward");
        assert_eq!(backward.1["recurrent_backend"], "auto");
        assert_eq!(backward.1["gate_activation_backend"], "cpu");
        assert_eq!(backward.1["bptt_backend"], "cpu");
        assert_eq!(backward.1["bptt_scan_backend"], "cpu");
        assert_eq!(
            backward.1["bptt_scan_kernel"],
            "lstm_backward_scan.cpu_fused_loop"
        );
        assert_eq!(
            backward.1["bptt_scan_lowering"],
            "host_reverse_recurrent_scan"
        );
        assert_eq!(backward.1["bptt_scan_shape_supported"], true);
        assert_eq!(backward.1["bptt_scan_runtime_requested"], false);
        assert_eq!(backward.1["bptt_scan_runtime_available"], false);
        assert!(backward.1["bptt_scan_elapsed_us"].as_u64().unwrap_or(0) > 0);
        assert_eq!(backward.1["bptt_scan_hidden_values"], 3 * 2);
        assert_eq!(backward.1["bptt_scan_gate_values"], 3 * 8);
        assert_eq!(backward.1["bptt_scan_cell_values"], 4 * 2);
        assert_eq!(backward.1["bptt_scan_recurrent_weight_values"], 8 * 2);
        assert_eq!(backward.1["bptt_scan_scratch_values"], 2 * 2);
        assert_eq!(backward.1["bptt_scan_kernel_dispatches"], 0);
        assert_eq!(backward.1["bptt_scan_serial_steps"], 3);
        assert_eq!(backward.1["bptt_scan_workgroup_size"], 1);
        assert_eq!(backward.1["bptt_scan_parallel_lanes"], 1);
        assert_eq!(backward.1["bptt_scan_parallel_axis"], "none");
        assert_eq!(backward.1["bptt_gate_derivative_backend"], "cpu");
        assert_eq!(backward.1["bptt_cell_recurrence_backend"], "cpu");
        assert_eq!(backward.1["bptt_state_carry_backend"], "cpu");
        assert_eq!(
            backward.1["bptt_sequence_dependency"],
            "reverse_time_recurrence"
        );
        assert_eq!(
            backward.1["bptt_fusion_candidate"],
            "fused_lstm_backward_scan"
        );
        assert_eq!(
            backward.1["bptt_scan_fusion_blocker"],
            "grad_h_next_and_grad_c_next_recurrence"
        );
        assert_eq!(backward.1["input_gradient_backend"], "auto");
        assert_eq!(backward.1["input_gradient_mode"], "batched_sequence_matmul");
        assert_eq!(
            backward.1["recurrent_gradient_mode"],
            "reverse_time_step_matmul"
        );
        assert_eq!(backward.1["raw_parameter_gradient_backend"], "hybrid");
        assert_eq!(backward.1["parameter_gradient_reduction_backend"], "auto");
        assert_eq!(backward.1["bias_gradient_backend"], "auto");
        assert_eq!(backward.1["parameter_gradient_scale_backend"], "auto");
        assert!((backward.1["gradient_scale"].as_f64().unwrap() - (1.0 / 3.0)).abs() < 1.0e-6);
        assert!(
            (backward.1["parameter_gradient_scale"].as_f64().unwrap() - (1.0 / 3.0)).abs() < 1.0e-6
        );
        assert_eq!(backward.1["input_gradient_scale"], 1.0);
        assert_eq!(backward.1["estimated_input_gradient_ops"], 3 * 8 * 3);
        assert_eq!(backward.1["estimated_input_gradient_dispatches"], 1);
        assert_eq!(backward.1["estimated_recurrent_gradient_dispatches"], 3);
        assert_eq!(
            backward.1["estimated_parameter_gradient_matmul_dispatches"],
            2
        );
        assert_eq!(backward.1["estimated_bptt_scan_steps"], 3);
        assert_eq!(
            backward.1["estimated_bptt_ops_per_scan_step"],
            backward.1["estimated_bptt_ops"].as_u64().unwrap() / 3
        );
        assert!(
            backward.1["estimated_bptt_ops"].as_u64().unwrap_or(0)
                > forward.1["estimated_bptt_ops"].as_u64().unwrap_or(0)
        );
        assert_eq!(
            backward.1["estimated_bptt_cpu_debt_ops"],
            backward.1["estimated_bptt_ops"]
        );
        assert_eq!(backward.1["estimated_bptt_wgpu_ops"], 0);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn lstm_forced_wgpu_routes_input_projection_and_matches_cpu_reference() {
        if !run_wgpu_runtime_tests() {
            eprintln!(
                "skipping runtime WGPU LSTM parity test; set SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1"
            );
            return;
        }
        if !wgpu_dense::is_available() {
            return;
        }
        let _lock = observer_lock();
        let previous_threshold = std::env::var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES").ok();
        std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", "1");

        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let input = Tensor::from_fn(4, 3, |row, col| {
            ((row * 17 + col * 7) % 19) as f32 * 0.031 - 0.22
        })
        .unwrap();
        let grad_out = Tensor::from_fn(4, 2, |row, col| {
            ((row * 13 + col * 5) % 17) as f32 * 0.027 - 0.18
        })
        .unwrap();

        let cpu_policy = BackendPolicy::from_device_caps(DeviceCaps::cpu());
        let mut cpu_lstm = Lstm::new("lstm_cpu", 3, 2).unwrap();
        let cpu_forward = {
            let _guard = push_backend_policy(cpu_policy);
            cpu_lstm.forward(&input).unwrap()
        };
        let cpu_grad_input = {
            let _guard = push_backend_policy(cpu_policy);
            cpu_lstm.backward(&input, &grad_out).unwrap()
        };

        let wgpu_policy = BackendPolicy::from_device_caps(DeviceCaps::wgpu(32, true, 256));
        let mut wgpu_lstm = Lstm::new("lstm_wgpu", 3, 2).unwrap();
        let (wgpu_forward, wgpu_grad_input) = {
            let _guard = push_backend_policy(wgpu_policy);
            (
                wgpu_lstm.forward(&input).unwrap(),
                wgpu_lstm.backward(&input, &grad_out).unwrap(),
            )
        };

        st_tensor::set_tensor_op_meta_observer(previous);
        match previous_threshold {
            Some(value) => std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", value),
            None => std::env::remove_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES"),
        }

        for (idx, (&cpu, &wgpu)) in cpu_forward
            .data()
            .iter()
            .zip(wgpu_forward.data().iter())
            .enumerate()
        {
            let delta = (cpu - wgpu).abs();
            assert!(
                delta <= 1e-5,
                "lstm forward mismatch at {idx}: cpu={cpu} wgpu={wgpu} delta={delta}"
            );
        }
        for (idx, (&cpu, &wgpu)) in cpu_grad_input
            .data()
            .iter()
            .zip(wgpu_grad_input.data().iter())
            .enumerate()
        {
            let delta = (cpu - wgpu).abs();
            assert!(
                delta <= 1e-5,
                "lstm grad input mismatch at {idx}: cpu={cpu} wgpu={wgpu} delta={delta}"
            );
        }
        for (cpu_param, wgpu_param) in [
            (
                cpu_lstm.weight_ih.gradient().unwrap(),
                wgpu_lstm.weight_ih.gradient().unwrap(),
            ),
            (
                cpu_lstm.weight_hh.gradient().unwrap(),
                wgpu_lstm.weight_hh.gradient().unwrap(),
            ),
            (
                cpu_lstm.bias_ih.gradient().unwrap(),
                wgpu_lstm.bias_ih.gradient().unwrap(),
            ),
            (
                cpu_lstm.bias_hh.gradient().unwrap(),
                wgpu_lstm.bias_hh.gradient().unwrap(),
            ),
        ] {
            for (idx, (&cpu, &wgpu)) in cpu_param
                .data()
                .iter()
                .zip(wgpu_param.data().iter())
                .enumerate()
            {
                let delta = (cpu - wgpu).abs();
                assert!(
                    delta <= 1e-5,
                    "lstm parameter gradient mismatch at {idx}: cpu={cpu} wgpu={wgpu} delta={delta}"
                );
            }
        }

        let events = events.lock().unwrap();
        let wgpu_runtime_available = st_tensor::wgpu_dense::is_available();
        let expected_matmul_backend = if wgpu_runtime_available {
            "wgpu"
        } else {
            "naive"
        };
        let expected_tensor_util_backend = if wgpu_runtime_available {
            "wgpu"
        } else {
            "cpu"
        };
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "matmul"
                && data["backend"] == expected_matmul_backend
                && data["requested_backend"] == "wgpu"
        }));
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "lstm_forward"
                && data["backend"] == "hybrid"
                && data["input_projection_backend"] == expected_matmul_backend
                && data["bias_backend"] == expected_tensor_util_backend
                && data["recurrent_backend"] == expected_matmul_backend
                && data["gate_activation_backend"] == expected_tensor_util_backend
        }));
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "lstm_backward"
                && data["backend"] == "hybrid"
                && data["input_gradient_backend"] == expected_matmul_backend
                && data["recurrent_backend"] == expected_matmul_backend
                && data["raw_parameter_gradient_backend"] == "hybrid"
                && data["parameter_gradient_reduction_backend"] == expected_matmul_backend
                && data["bias_gradient_backend"] == expected_tensor_util_backend
                && data["parameter_gradient_scale_backend"] == expected_tensor_util_backend
        }));
        let backward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "lstm_backward"
                    && data["backend"] == "hybrid"
                    && data["input_gradient_backend"] == expected_matmul_backend
                    && data["recurrent_backend"] == expected_matmul_backend
            })
            .expect("forced WGPU LSTM backward metadata");
        if backward.1["bptt_scan_backend"] == "wgpu" {
            assert_eq!(backward.1["bptt_backend"], "wgpu");
            assert_eq!(backward.1["bptt_scan_kernel"], "lstm_backward_scan.wgsl");
            assert_eq!(
                backward.1["bptt_scan_lowering"],
                "wgpu_single_workgroup_hidden_parallel_recurrence"
            );
            assert_eq!(backward.1["bptt_gate_derivative_backend"], "wgpu");
            assert_eq!(backward.1["bptt_cell_recurrence_backend"], "wgpu");
            assert_eq!(backward.1["bptt_state_carry_backend"], "wgpu");
            assert_eq!(backward.1["gate_activation_backend"], "wgpu");
            assert!(backward.1["bptt_scan_fallback_reason"].is_null());
            assert_eq!(backward.1["bptt_scan_shape_supported"], true);
            assert_eq!(backward.1["bptt_scan_runtime_requested"], true);
            assert_eq!(backward.1["bptt_scan_runtime_available"], true);
            assert!(backward.1["bptt_scan_elapsed_us"].as_u64().unwrap_or(0) > 0);
            assert_eq!(backward.1["bptt_scan_kernel_dispatches"], 1);
            assert_eq!(
                backward.1["bptt_scan_workgroup_size"],
                st_tensor::wgpu_dense::lstm_backward_scan_workgroup_size()
            );
            assert_eq!(
                backward.1["bptt_scan_parallel_lanes"],
                st_tensor::wgpu_dense::lstm_backward_scan_workgroup_size()
            );
            assert_eq!(backward.1["bptt_scan_parallel_axis"], "hidden");
            assert_eq!(
                backward.1["bptt_scan_serial_steps"],
                backward.1["estimated_bptt_scan_steps"]
            );
            assert_eq!(backward.1["estimated_bptt_cpu_debt_ops"], 0);
            assert_eq!(
                backward.1["estimated_bptt_wgpu_ops"],
                backward.1["estimated_bptt_ops"]
            );
        } else {
            assert_eq!(backward.1["bptt_backend"], "cpu");
            assert_eq!(backward.1["bptt_scan_backend"], "cpu");
            assert_eq!(
                backward.1["bptt_scan_kernel"],
                "lstm_backward_scan.cpu_fused_loop"
            );
            assert_eq!(
                backward.1["bptt_scan_lowering"],
                "host_reverse_recurrent_scan"
            );
            assert_eq!(backward.1["bptt_gate_derivative_backend"], "cpu");
            assert_eq!(backward.1["bptt_cell_recurrence_backend"], "cpu");
            assert_eq!(backward.1["bptt_state_carry_backend"], "cpu");
            assert_eq!(backward.1["gate_activation_backend"], "cpu");
            assert!(backward.1["bptt_scan_fallback_reason"].as_str().is_some());
            assert_eq!(backward.1["bptt_scan_shape_supported"], true);
            assert_eq!(backward.1["bptt_scan_runtime_requested"], true);
            assert_eq!(backward.1["bptt_scan_runtime_available"], false);
            assert!(backward.1["bptt_scan_elapsed_us"].as_u64().unwrap_or(0) > 0);
            assert_eq!(backward.1["bptt_scan_kernel_dispatches"], 0);
            assert_eq!(backward.1["bptt_scan_workgroup_size"], 1);
            assert_eq!(backward.1["bptt_scan_parallel_lanes"], 1);
            assert_eq!(backward.1["bptt_scan_parallel_axis"], "none");
            assert_eq!(
                backward.1["bptt_scan_serial_steps"],
                backward.1["estimated_bptt_scan_steps"]
            );
            assert_eq!(
                backward.1["estimated_bptt_cpu_debt_ops"],
                backward.1["estimated_bptt_ops"]
            );
            assert_eq!(backward.1["estimated_bptt_wgpu_ops"], 0);
        }
    }
}
