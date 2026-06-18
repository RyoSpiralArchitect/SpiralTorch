// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::conv::Conv1d;
use super::wave_gate::WaveGate;
use crate::execution::current_tensor_util_backend_for_values;
use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
#[cfg(feature = "wgpu")]
use st_tensor::wgpu_dense;
use st_tensor::{emit_tensor_op, emit_tensor_op_meta, TensorUtilBackend};
use std::cell::RefCell;

#[derive(Clone, Debug)]
struct WaveScanCache {
    input: Tensor,
    gating_in: Tensor,
    batch: usize,
    input_steps: usize,
    out_steps: usize,
    features: usize,
}

fn tensor_util_backend_label(backend: TensorUtilBackend) -> &'static str {
    match backend {
        TensorUtilBackend::Auto => "auto",
        TensorUtilBackend::Cpu => "cpu",
        TensorUtilBackend::GpuWgpu => "wgpu",
    }
}

fn validate_finite_slice(label: &'static str, values: &[f32]) -> PureResult<()> {
    for &value in values {
        if !value.is_finite() {
            return Err(TensorError::NonFiniteValue { label, value });
        }
    }
    Ok(())
}

#[cfg(feature = "wgpu")]
fn strict_gpu_path() -> bool {
    std::env::var("SPIRALTORCH_STRICT_GPU")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

#[cfg(feature = "wgpu")]
fn wave_scan_wgpu_error(op_name: &'static str, message: String) -> TensorError {
    TensorError::BackendFailure {
        backend: "wgpu",
        message: format!("{op_name} wgpu path failed ({message}); fallback disabled"),
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_wave_scan_meta(
    op_name: &'static str,
    kind: &'static str,
    batch: usize,
    input_steps: usize,
    in_channels: usize,
    out_steps: usize,
    features: usize,
    backward: bool,
    route_backend: TensorUtilBackend,
    actual_backend: &'static str,
    kernel: &'static str,
    fallback: Option<&str>,
) {
    let input_values = batch
        .saturating_mul(input_steps)
        .saturating_mul(in_channels);
    let gated_values = batch.saturating_mul(out_steps).saturating_mul(features);
    let context_values = batch.saturating_mul(features);
    emit_tensor_op(
        op_name,
        &[batch, input_steps, in_channels],
        &[batch, features],
    );
    emit_tensor_op_meta(op_name, || {
        let mut data = serde_json::json!({
            "backend": actual_backend,
            "requested_backend": tensor_util_backend_label(route_backend),
            "kernel": kernel,
            "kind": kind,
            "batch": batch,
            "input_steps": input_steps,
            "in_channels": in_channels,
            "out_steps": out_steps,
            "features": features,
            "input_values": input_values,
            "gated_values": gated_values,
            "context_values": context_values,
            "final_step_offset": out_steps.saturating_sub(1),
            "estimated_gather_values": if backward { 0 } else { context_values },
            "estimated_scatter_values": if backward { context_values } else { 0 },
            "backward": backward,
            "empty": input_values == 0 || context_values == 0,
        });
        if let Some(message) = fallback {
            if let Some(map) = data.as_object_mut() {
                map.insert(
                    "fallback".to_string(),
                    serde_json::json!({"from": "wgpu", "message": message}),
                );
            }
        }
        data
    });
}

#[allow(clippy::too_many_arguments)]
fn emit_wave_scan_stack_meta(
    op_name: &'static str,
    kind: &'static str,
    batch: usize,
    input_cols: usize,
    features: usize,
    branch_count: usize,
    merge_backend: TensorUtilBackend,
    backward: bool,
) {
    let input_values = batch.saturating_mul(input_cols);
    let context_values = batch.saturating_mul(features);
    let branch_values = branch_count.saturating_mul(context_values);
    emit_tensor_op(op_name, &[batch, input_cols], &[batch, features]);
    emit_tensor_op_meta(op_name, || {
        serde_json::json!({
            "backend": "composite",
            "requested_backend": tensor_util_backend_label(merge_backend),
            "merge_backend": tensor_util_backend_label(merge_backend),
            "kernel": "wave_scan_stack.composite",
            "kind": kind,
            "batch": batch,
            "input_cols": input_cols,
            "features": features,
            "branch_count": branch_count,
            "input_values": input_values,
            "context_values": context_values,
            "branch_values": branch_values,
            "estimated_branch_adds": branch_values,
            "estimated_average_scales": context_values,
            "estimated_backward_adds": if backward { input_values.saturating_mul(branch_count) } else { 0 },
            "backward": backward,
            "empty": input_values == 0 || context_values == 0,
        })
    });
}

/// Convolutional scan module that reads a flattened (channels×steps) sequence and emits
/// a single context vector from the final step after Z-space WaveGate projection.
#[derive(Debug)]
pub struct WaveScan {
    conv: Conv1d,
    gate: WaveGate,
    in_channels: usize,
    features: usize,
    cache: RefCell<Option<WaveScanCache>>,
}

impl WaveScan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        in_channels: usize,
        features: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        curvature: f32,
        temperature: f32,
    ) -> PureResult<Self> {
        if in_channels == 0 || features == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: in_channels.max(1),
                cols: features.max(1),
            });
        }
        let name = name.into();
        let conv = Conv1d::new(
            format!("{name}::conv"),
            in_channels,
            features,
            kernel_size,
            stride,
            padding,
            dilation,
        )?;
        let gate = WaveGate::new(format!("{name}::gate"), features, curvature, temperature)?;
        Ok(Self {
            conv,
            gate,
            in_channels,
            features,
            cache: RefCell::new(None),
        })
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    pub fn features(&self) -> usize {
        self.features
    }

    pub fn gate(&self) -> &WaveGate {
        &self.gate
    }

    pub fn gate_mut(&mut self) -> &mut WaveGate {
        &mut self.gate
    }

    fn infer_steps(&self, cols: usize) -> PureResult<usize> {
        if !cols.is_multiple_of(self.in_channels) {
            return Err(TensorError::ShapeMismatch {
                left: (1, cols),
                right: (1, self.in_channels),
            });
        }
        Ok(cols / self.in_channels)
    }
}

impl Module for WaveScan {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (batch, cols) = input.shape();
        let steps = self.infer_steps(cols)?;
        if steps == 0 {
            return Err(TensorError::InvalidDimensions { rows: batch, cols });
        }

        let conv_out = self.conv.forward(input)?;
        let (rows, conv_cols) = conv_out.shape();
        if rows != batch || conv_cols % self.features != 0 {
            return Err(TensorError::ShapeMismatch {
                left: conv_out.shape(),
                right: (batch, self.features),
            });
        }
        let out_steps = conv_cols / self.features;
        if out_steps == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: batch,
                cols: conv_cols,
            });
        }

        let gating_in = conv_out.reshape(batch * out_steps, self.features)?;
        let gating_out = self.gate.forward(&gating_in)?;
        let gating_reshaped = gating_out.reshape(batch, self.features * out_steps)?;
        let gather_backend =
            current_tensor_util_backend_for_values(batch.saturating_mul(self.features));
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(gather_backend, TensorUtilBackend::GpuWgpu)
                && batch > 0
                && out_steps > 0
                && self.features > 0
                && wgpu_dense::is_available()
            {
                match wgpu_dense::sequence_last_step_gather(
                    gating_reshaped.data(),
                    batch,
                    out_steps,
                    self.features,
                ) {
                    Ok(buffer) => {
                        validate_finite_slice("wave_scan_final_hidden", &buffer)?;
                        let final_hidden = Tensor::from_vec(batch, self.features, buffer)?;
                        *self.cache.borrow_mut() = Some(WaveScanCache {
                            input: input.clone(),
                            gating_in,
                            batch,
                            input_steps: steps,
                            out_steps,
                            features: self.features,
                        });
                        emit_wave_scan_meta(
                            "wave_scan_forward",
                            "wave_scan_final_step_gather",
                            batch,
                            steps,
                            self.in_channels,
                            out_steps,
                            self.features,
                            false,
                            gather_backend,
                            "wgpu_dense",
                            "tensor_util.sequence_last_step_gather",
                            None,
                        );
                        return Ok(final_hidden);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(wave_scan_wgpu_error("wave_scan_forward", message));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut final_hidden = Tensor::zeros(batch, self.features)?;
        {
            let source = gating_reshaped.data();
            let dest = final_hidden.data_mut();
            for b in 0..batch {
                let src_offset = b * self.features * out_steps + (out_steps - 1) * self.features;
                let dst_offset = b * self.features;
                dest[dst_offset..dst_offset + self.features]
                    .copy_from_slice(&source[src_offset..src_offset + self.features]);
            }
        }
        validate_finite_slice("wave_scan_final_hidden", final_hidden.data())?;

        *self.cache.borrow_mut() = Some(WaveScanCache {
            input: input.clone(),
            gating_in,
            batch,
            input_steps: steps,
            out_steps,
            features: self.features,
        });
        emit_wave_scan_meta(
            "wave_scan_forward",
            "wave_scan_final_step_gather",
            batch,
            steps,
            self.in_channels,
            out_steps,
            self.features,
            false,
            gather_backend,
            "cpu",
            "wave_scan.final_step_cpu_gather",
            {
                #[cfg(feature = "wgpu")]
                {
                    wgpu_failure.as_deref()
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    None
                }
            },
        );
        Ok(final_hidden)
    }

    fn backward(&mut self, _input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let cache = self
            .cache
            .borrow_mut()
            .take()
            .ok_or(TensorError::EmptyInput("wave_scan_cache"))?;
        if grad_output.shape() != (cache.batch, cache.features) {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (cache.batch, cache.features),
            });
        }
        validate_finite_slice("wave_scan_grad_output", grad_output.data())?;
        let scatter_backend =
            current_tensor_util_backend_for_values(cache.batch.saturating_mul(cache.features));
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        let maybe_grad_gate_out = if matches!(scatter_backend, TensorUtilBackend::GpuWgpu)
            && cache.batch > 0
            && cache.out_steps > 0
            && cache.features > 0
            && wgpu_dense::is_available()
        {
            match wgpu_dense::sequence_last_step_scatter(
                grad_output.data(),
                cache.batch,
                cache.out_steps,
                cache.features,
            ) {
                Ok(buffer) => {
                    validate_finite_slice("wave_scan_grad_gate_out", &buffer)?;
                    Some(Tensor::from_vec(
                        cache.batch,
                        cache.features * cache.out_steps,
                        buffer,
                    )?)
                }
                Err(message) if strict_gpu_path() => {
                    return Err(wave_scan_wgpu_error("wave_scan_backward", message));
                }
                Err(message) => {
                    wgpu_failure = Some(message);
                    None
                }
            }
        } else {
            None
        };
        #[cfg(not(feature = "wgpu"))]
        let maybe_grad_gate_out: Option<Tensor> = None;

        let (grad_gate_out, actual_backend, kernel) = if let Some(grad_gate_out) =
            maybe_grad_gate_out
        {
            (
                grad_gate_out,
                "wgpu_dense",
                "tensor_util.sequence_last_step_scatter",
            )
        } else {
            let mut grad_gate_out = Tensor::zeros(cache.batch, cache.features * cache.out_steps)?;
            {
                let src = grad_output.data();
                let dst = grad_gate_out.data_mut();
                for b in 0..cache.batch {
                    let src_offset = b * cache.features;
                    let dst_offset = b * cache.features * cache.out_steps
                        + (cache.out_steps - 1) * cache.features;
                    dst[dst_offset..dst_offset + cache.features]
                        .copy_from_slice(&src[src_offset..src_offset + cache.features]);
                }
            }
            validate_finite_slice("wave_scan_grad_gate_out", grad_gate_out.data())?;
            (grad_gate_out, "cpu", "wave_scan.final_step_cpu_scatter")
        };

        let grad_gate_out = grad_gate_out.reshape(cache.batch * cache.out_steps, cache.features)?;
        let grad_gate_in = self.gate.backward(&cache.gating_in, &grad_gate_out)?;
        let grad_conv_out = grad_gate_in.reshape(cache.batch, cache.features * cache.out_steps)?;
        let grad_input = self.conv.backward(&cache.input, &grad_conv_out)?;
        emit_wave_scan_meta(
            "wave_scan_backward",
            "wave_scan_final_step_scatter",
            cache.batch,
            cache.input_steps,
            self.in_channels,
            cache.out_steps,
            cache.features,
            true,
            scatter_backend,
            actual_backend,
            kernel,
            {
                #[cfg(feature = "wgpu")]
                {
                    wgpu_failure.as_deref()
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    None
                }
            },
        );
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.conv.visit_parameters(visitor)?;
        self.gate.visit_parameters(visitor)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.conv.visit_parameters_mut(visitor)?;
        self.gate.visit_parameters_mut(visitor)?;
        Ok(())
    }

    fn infuse_text(&mut self, text: &str) -> PureResult<()> {
        self.gate.infuse_text(text)
    }
}

/// Multi-dilation WaveScan stack that averages the per-dilation summaries.
#[derive(Debug)]
pub struct WaveScanStack {
    scans: Vec<WaveScan>,
    features: usize,
}

impl WaveScanStack {
    pub fn new(scans: Vec<WaveScan>) -> PureResult<Self> {
        if scans.is_empty() {
            return Err(TensorError::EmptyInput("wave_scan_stack"));
        }
        let features = scans[0].features();
        if scans.iter().any(|scan| scan.features() != features) {
            return Err(TensorError::InvalidDimensions {
                rows: features,
                cols: 0,
            });
        }
        Ok(Self { scans, features })
    }

    pub fn features(&self) -> usize {
        self.features
    }

    pub fn scans(&self) -> &[WaveScan] {
        &self.scans
    }

    pub fn scans_mut(&mut self) -> &mut [WaveScan] {
        &mut self.scans
    }
}

impl Module for WaveScanStack {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (batch, input_cols) = input.shape();
        let mut sum = Tensor::zeros(batch, self.features)?;
        let merge_backend = current_tensor_util_backend_for_values(sum.data().len());
        for scan in &self.scans {
            let out = scan.forward(input)?;
            sum.add_scaled_with_backend(&out, 1.0, merge_backend)?;
        }
        let inv = 1.0 / (self.scans.len() as f32).max(1.0);
        let output = sum.scale_with_backend(inv, merge_backend)?;
        emit_wave_scan_stack_meta(
            "wave_scan_stack_forward",
            "wave_scan_stack_average",
            batch,
            input_cols,
            self.features,
            self.scans.len(),
            merge_backend,
            false,
        );
        Ok(output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        if self.scans.is_empty() {
            return Ok(grad_output.clone());
        }
        let inv = 1.0 / (self.scans.len() as f32).max(1.0);
        let scale_backend = current_tensor_util_backend_for_values(grad_output.data().len());
        let grad_scaled = grad_output.scale_with_backend(inv, scale_backend)?;
        let (batch, cols) = input.shape();
        let mut total = Tensor::zeros(batch, cols)?;
        let accum_backend = current_tensor_util_backend_for_values(total.data().len());
        for scan in &mut self.scans {
            let grad_in = scan.backward(input, &grad_scaled)?;
            total.add_scaled_with_backend(&grad_in, 1.0, accum_backend)?;
        }
        emit_wave_scan_stack_meta(
            "wave_scan_stack_backward",
            "wave_scan_stack_grad_average",
            batch,
            cols,
            self.features,
            self.scans.len(),
            accum_backend,
            true,
        );
        Ok(total)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        for scan in &self.scans {
            scan.visit_parameters(visitor)?;
        }
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        for scan in &mut self.scans {
            scan.visit_parameters_mut(visitor)?;
        }
        Ok(())
    }

    fn infuse_text(&mut self, text: &str) -> PureResult<()> {
        for scan in &mut self.scans {
            scan.infuse_text(text)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "wgpu")]
    use crate::execution::{push_backend_policy, BackendPolicy};
    #[cfg(feature = "wgpu")]
    use st_core::backend::device_caps::DeviceCaps;
    use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

    fn observer_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    #[cfg(feature = "wgpu")]
    fn approx_eq(lhs: &[f32], rhs: &[f32]) {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            assert!(
                (*a - *b).abs() < 1e-4,
                "mismatch at {idx}: left={a}, right={b}"
            );
        }
    }

    #[cfg(feature = "wgpu")]
    fn collect_gradients(scan: &WaveScan) -> Vec<Vec<f32>> {
        let mut gradients = Vec::new();
        scan.visit_parameters(&mut |param| {
            if let Some(gradient) = param.gradient() {
                gradients.push(gradient.data().to_vec());
            }
            Ok(())
        })
        .unwrap();
        gradients
    }

    #[test]
    fn wave_scan_forward_backward_matches_shapes() {
        let mut scan = WaveScan::new("ws", 4, 4, 3, 1, 1, 2, -1.0, 0.7).unwrap();
        let input = Tensor::from_vec(2, 16, vec![0.1; 32]).unwrap(); // in_channels=4, steps=4
        let out = scan.forward(&input).unwrap();
        assert_eq!(out.shape(), (2, 4));
        let grad_out = Tensor::from_vec(2, 4, vec![0.05; 8]).unwrap();
        let grad_in = scan.backward(&input, &grad_out).unwrap();
        assert_eq!(grad_in.shape(), input.shape());
    }

    #[test]
    fn wave_scan_and_stack_emit_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let input = Tensor::from_vec(1, 12, vec![0.1; 12]).unwrap();
        let mut scan = WaveScan::new("meta_scan", 3, 3, 3, 1, 1, 1, -1.0, 1.0).unwrap();
        let out = scan.forward(&input).unwrap();
        assert_eq!(out.shape(), (1, 3));
        let grad_out = Tensor::from_vec(1, 3, vec![0.05; 3]).unwrap();
        let grad_in = scan.backward(&input, &grad_out).unwrap();
        assert_eq!(grad_in.shape(), input.shape());

        let scan_a = WaveScan::new("meta_stack_a", 3, 3, 3, 1, 1, 1, -1.0, 1.0).unwrap();
        let scan_b = WaveScan::new("meta_stack_b", 3, 3, 3, 1, 1, 2, -1.0, 1.0).unwrap();
        let mut stack = WaveScanStack::new(vec![scan_a, scan_b]).unwrap();
        let stack_out = stack.forward(&input).unwrap();
        assert_eq!(stack_out.shape(), (1, 3));
        let stack_grad_in = stack.backward(&input, &grad_out).unwrap();
        assert_eq!(stack_grad_in.shape(), input.shape());
        st_tensor::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        let wave_forward = events
            .iter()
            .find(|(name, data)| {
                *name == "wave_scan_forward"
                    && data["kind"] == "wave_scan_final_step_gather"
                    && data["input_steps"] == 4
                    && data["in_channels"] == 3
                    && data["features"] == 3
            })
            .unwrap_or_else(|| panic!("wave_scan_forward metadata event"));
        assert_eq!(wave_forward.1["backend"], "cpu");
        assert_eq!(wave_forward.1["requested_backend"], "auto");
        assert_eq!(wave_forward.1["gated_values"], 12);
        assert_eq!(wave_forward.1["context_values"], 3);
        assert_eq!(wave_forward.1["estimated_gather_values"], 3);

        let wave_backward = events
            .iter()
            .find(|(name, data)| {
                *name == "wave_scan_backward"
                    && data["kind"] == "wave_scan_final_step_scatter"
                    && data["input_steps"] == 4
                    && data["in_channels"] == 3
                    && data["features"] == 3
            })
            .unwrap_or_else(|| panic!("wave_scan_backward metadata event"));
        assert_eq!(wave_backward.1["backend"], "cpu");
        assert_eq!(wave_backward.1["requested_backend"], "auto");
        assert_eq!(wave_backward.1["estimated_scatter_values"], 3);
        assert_eq!(wave_backward.1["backward"], true);

        for (op_name, kind, backward) in [
            ("wave_scan_stack_forward", "wave_scan_stack_average", false),
            (
                "wave_scan_stack_backward",
                "wave_scan_stack_grad_average",
                true,
            ),
        ] {
            let event = events
                .iter()
                .find(|(name, data)| {
                    *name == op_name
                        && data["kind"] == kind
                        && data["input_cols"] == 12
                        && data["features"] == 3
                        && data["branch_count"] == 2
                })
                .unwrap_or_else(|| panic!("{op_name} metadata event"));
            assert_eq!(event.1["backend"], "composite");
            assert_eq!(event.1["context_values"], 3);
            assert_eq!(event.1["branch_values"], 6);
            assert_eq!(event.1["backward"], backward);
        }
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn wave_scan_forced_wgpu_uses_sequence_last_step_and_matches_cpu_reference() {
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

        let input = Tensor::from_fn(2, 12, |row, col| {
            ((row * 13 + col * 5) % 23) as f32 * 0.027 - 0.21
        })
        .unwrap();
        let grad_out = Tensor::from_fn(2, 3, |row, col| {
            ((row * 7 + col * 11) % 17) as f32 * 0.019 - 0.12
        })
        .unwrap();

        let cpu_policy = BackendPolicy::from_device_caps(DeviceCaps::cpu());
        let mut cpu_scan = WaveScan::new("scan_cpu", 3, 3, 3, 1, 1, 1, -1.0, 1.0).unwrap();
        let cpu_forward = {
            let _guard = push_backend_policy(cpu_policy);
            cpu_scan.forward(&input).unwrap()
        };
        let cpu_grad_input = {
            let _guard = push_backend_policy(cpu_policy);
            cpu_scan.backward(&input, &grad_out).unwrap()
        };

        let wgpu_policy = BackendPolicy::from_device_caps(DeviceCaps::wgpu(32, true, 256));
        let mut wgpu_scan = WaveScan::new("scan_wgpu", 3, 3, 3, 1, 1, 1, -1.0, 1.0).unwrap();
        let (wgpu_forward, wgpu_grad_input) = {
            let _guard = push_backend_policy(wgpu_policy);
            (
                wgpu_scan.forward(&input).unwrap(),
                wgpu_scan.backward(&input, &grad_out).unwrap(),
            )
        };

        st_tensor::set_tensor_op_meta_observer(previous);
        match previous_threshold {
            Some(value) => std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", value),
            None => std::env::remove_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES"),
        }

        approx_eq(cpu_forward.data(), wgpu_forward.data());
        approx_eq(cpu_grad_input.data(), wgpu_grad_input.data());
        let cpu_gradients = collect_gradients(&cpu_scan);
        let wgpu_gradients = collect_gradients(&wgpu_scan);
        assert_eq!(cpu_gradients.len(), wgpu_gradients.len());
        for (cpu_gradient, wgpu_gradient) in cpu_gradients.iter().zip(wgpu_gradients.iter()) {
            approx_eq(cpu_gradient, wgpu_gradient);
        }

        let events = events.lock().unwrap();
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "wave_scan_forward"
                && data["backend"] == "wgpu_dense"
                && data["requested_backend"] == "wgpu"
                && data["kernel"] == "tensor_util.sequence_last_step_gather"
        }));
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "wave_scan_backward"
                && data["backend"] == "wgpu_dense"
                && data["requested_backend"] == "wgpu"
                && data["kernel"] == "tensor_util.sequence_last_step_scatter"
        }));
    }

    #[test]
    fn wave_scan_stack_averages_branches() {
        let scan_a = WaveScan::new("a", 2, 2, 3, 1, 1, 1, -1.0, 1.0).unwrap();
        let scan_b = WaveScan::new("b", 2, 2, 3, 1, 1, 2, -1.0, 1.0).unwrap();
        let mut stack = WaveScanStack::new(vec![scan_a, scan_b]).unwrap();
        let input = Tensor::from_vec(1, 8, vec![0.2; 8]).unwrap(); // in_channels=2, steps=4
        let out = stack.forward(&input).unwrap();
        assert_eq!(out.shape(), (1, 2));
        let grad_out = Tensor::from_vec(1, 2, vec![0.1, -0.2]).unwrap();
        let grad_in = stack.backward(&input, &grad_out).unwrap();
        assert_eq!(grad_in.shape(), input.shape());
    }
}
