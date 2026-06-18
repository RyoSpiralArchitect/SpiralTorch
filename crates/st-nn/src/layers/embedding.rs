// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::execution::current_tensor_util_backend_for_values;
use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
#[cfg(feature = "wgpu")]
use st_tensor::wgpu_dense;
use st_tensor::{emit_tensor_op, emit_tensor_op_meta, TensorUtilBackend};

fn validate_finite_value(label: &'static str, value: f32) -> PureResult<()> {
    if !value.is_finite() {
        return Err(TensorError::NonFiniteValue { label, value });
    }
    Ok(())
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

#[derive(Clone, Copy, Debug, Default)]
struct TokenIndexStats {
    total: usize,
    unique_indices: usize,
    repeated_indices: usize,
    non_finite: usize,
    rounded: usize,
    clamped_low: usize,
    clamped_high: usize,
}

fn token_to_index_with_stats(
    value: f32,
    vocab_size: usize,
    mut stats: Option<&mut TokenIndexStats>,
) -> usize {
    if let Some(stats) = stats.as_deref_mut() {
        stats.total = stats.total.saturating_add(1);
    }
    if vocab_size == 0 {
        return 0;
    }
    if !value.is_finite() {
        if let Some(stats) = stats.as_deref_mut() {
            stats.non_finite = stats.non_finite.saturating_add(1);
        }
        return 0;
    }
    let rounded = value.round();
    if !rounded.is_finite() {
        if let Some(stats) = stats.as_deref_mut() {
            stats.non_finite = stats.non_finite.saturating_add(1);
        }
        return 0;
    }
    if (value - rounded).abs() > f32::EPSILON {
        if let Some(stats) = stats.as_deref_mut() {
            stats.rounded = stats.rounded.saturating_add(1);
        }
    }
    let idx = rounded as isize;
    if idx < 0 {
        if let Some(stats) = stats.as_deref_mut() {
            stats.clamped_low = stats.clamped_low.saturating_add(1);
        }
        return 0;
    }
    if idx == 0 {
        return 0;
    }
    let max = vocab_size.saturating_sub(1) as isize;
    if idx > max {
        if let Some(stats) = stats.as_deref_mut() {
            stats.clamped_high = stats.clamped_high.saturating_add(1);
        }
        return max as usize;
    }
    idx as usize
}

fn observe_token_index(stats: &mut TokenIndexStats, seen: &mut [bool], idx: usize) {
    let Some(slot) = seen.get_mut(idx) else {
        return;
    };
    if *slot {
        stats.repeated_indices = stats.repeated_indices.saturating_add(1);
    } else {
        *slot = true;
        stats.unique_indices = stats.unique_indices.saturating_add(1);
    }
}

fn collect_token_indices(
    input_data: &[f32],
    batch: usize,
    steps: usize,
    vocab_size: usize,
) -> (Vec<f32>, TokenIndexStats) {
    let mut indices = Vec::with_capacity(batch.saturating_mul(steps));
    let mut stats = TokenIndexStats::default();
    let mut seen = vec![false; vocab_size];
    for b in 0..batch {
        let row_offset = b * steps;
        for t in 0..steps {
            let idx =
                token_to_index_with_stats(input_data[row_offset + t], vocab_size, Some(&mut stats));
            observe_token_index(&mut stats, &mut seen, idx);
            indices.push(idx as f32);
        }
    }
    (indices, stats)
}

fn tensor_util_backend_label(backend: TensorUtilBackend) -> &'static str {
    match backend {
        TensorUtilBackend::Auto => "auto",
        TensorUtilBackend::Cpu => "cpu",
        TensorUtilBackend::GpuWgpu => "wgpu",
    }
}

#[cfg(feature = "wgpu")]
fn strict_gpu_path() -> bool {
    std::env::var("SPIRALTORCH_STRICT_GPU")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

#[cfg(feature = "wgpu")]
fn embedding_wgpu_error(op_name: &'static str, message: String) -> TensorError {
    TensorError::BackendFailure {
        backend: "wgpu",
        message: format!("{op_name} wgpu path failed ({message}); fallback disabled"),
    }
}

fn emit_embedding_meta(
    op_name: &'static str,
    input_shape: (usize, usize),
    output_shape: (usize, usize),
    vocab_size: usize,
    embed_dim: usize,
    stats: TokenIndexStats,
    requested_backend: &'static str,
    backend: &'static str,
    kernel: &'static str,
    gradient_reduction_backend: Option<String>,
    gradient_scale: Option<f32>,
    fallback_from: Option<&'static str>,
    fallback_message: Option<&str>,
) {
    emit_tensor_op(
        op_name,
        &[input_shape.0, input_shape.1, vocab_size, embed_dim],
        &[output_shape.0, output_shape.1],
    );
    emit_tensor_op_meta(op_name, || {
        let backward = op_name == "embedding_backward";
        let token_embedding_values = stats.total.saturating_mul(embed_dim);
        let grad_weight_values = if backward {
            vocab_size.saturating_mul(embed_dim)
        } else {
            0
        };
        serde_json::json!({
            "backend": backend,
            "requested_backend": requested_backend,
            "kernel": kernel,
            "kind": if backward { "embedding_backward_scatter" } else { "embedding_forward_lookup" },
            "batch": input_shape.0,
            "steps": input_shape.1,
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "output_rows": output_shape.0,
            "output_cols": output_shape.1,
            "output_values": output_shape.0.saturating_mul(output_shape.1),
            "trainable_parameters": vocab_size.saturating_mul(embed_dim),
            "tokens": stats.total,
            "unique_token_indices": stats.unique_indices,
            "repeated_token_indices": stats.repeated_indices,
            "non_finite_tokens": stats.non_finite,
            "rounded_tokens": stats.rounded,
            "clamped_low_tokens": stats.clamped_low,
            "clamped_high_tokens": stats.clamped_high,
            "gradient_reduction_backend": gradient_reduction_backend,
            "gradient_scale": gradient_scale,
            "grad_weight_rows": if backward { vocab_size } else { 0 },
            "grad_weight_cols": if backward { embed_dim } else { 0 },
            "grad_weight_values": grad_weight_values,
            "estimated_lookup_values": if backward { 0 } else { token_embedding_values },
            "estimated_scatter_adds": if backward { token_embedding_values } else { 0 },
            "empty": input_shape.0 == 0 || input_shape.1 == 0 || embed_dim == 0,
            "fallback": {
                "from": fallback_from,
                "message": fallback_message,
            },
        })
    });
}

/// Simple embedding lookup table.
///
/// Inputs are expected to be integer token IDs stored as floats in a tensor
/// shaped `(batch, steps)`. Outputs are flattened embeddings shaped
/// `(batch, steps * embed_dim)` so they compose with sequence modules that
/// consume flattened steps.
#[derive(Debug)]
pub struct Embedding {
    weight: Parameter,
    vocab_size: usize,
    embed_dim: usize,
}

impl Embedding {
    pub fn new(name: impl Into<String>, vocab_size: usize, embed_dim: usize) -> PureResult<Self> {
        if vocab_size == 0 || embed_dim == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: vocab_size.max(1),
                cols: embed_dim.max(1),
            });
        }
        let name = name.into();
        let mut scale = 0.01f32;
        let weight = Tensor::from_fn(vocab_size, embed_dim, |_r, _c| {
            let value = scale;
            scale = (scale + 0.013).rem_euclid(0.05).max(1e-4);
            value
        })?;
        Ok(Self {
            weight: Parameter::new(format!("{name}::weight"), weight),
            vocab_size,
            embed_dim,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    pub fn weight(&self) -> &Parameter {
        &self.weight
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (batch, steps) = input.shape();
        let output_cols = steps * self.embed_dim;
        if steps == 0 {
            let output = Tensor::zeros(batch, 0)?;
            emit_embedding_meta(
                "embedding_forward",
                input.shape(),
                output.shape(),
                self.vocab_size,
                self.embed_dim,
                TokenIndexStats::default(),
                "auto",
                "cpu",
                "embedding.cpu_gather",
                None,
                None,
                None,
                None,
            );
            return Ok(output);
        }
        let weights = self.weight.value().data();
        validate_finite_slice("embedding_weight", weights)?;
        let input_data = input.data();
        let (indices, stats) = collect_token_indices(input_data, batch, steps, self.vocab_size);
        let route_backend =
            current_tensor_util_backend_for_values(batch.saturating_mul(output_cols));
        let requested_backend = tensor_util_backend_label(route_backend);
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(route_backend, TensorUtilBackend::GpuWgpu) && wgpu_dense::is_available() {
                match wgpu_dense::embedding_gather(&indices, weights, indices.len(), self.embed_dim)
                {
                    Ok(out) => {
                        validate_finite_slice("embedding_output", &out)?;
                        let output = Tensor::from_vec(batch, output_cols, out)?;
                        emit_embedding_meta(
                            "embedding_forward",
                            input.shape(),
                            output.shape(),
                            self.vocab_size,
                            self.embed_dim,
                            stats,
                            requested_backend,
                            "wgpu_dense",
                            "embedding.wgpu_gather",
                            None,
                            None,
                            None,
                            None,
                        );
                        return Ok(output);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(embedding_wgpu_error("embedding_forward", message));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut out = Vec::with_capacity(batch * output_cols);
        for &idx in &indices {
            let start = idx as usize * self.embed_dim;
            out.extend_from_slice(&weights[start..start + self.embed_dim]);
        }
        validate_finite_slice("embedding_output", &out)?;
        let output = Tensor::from_vec(batch, output_cols, out)?;
        emit_embedding_meta(
            "embedding_forward",
            input.shape(),
            output.shape(),
            self.vocab_size,
            self.embed_dim,
            stats,
            requested_backend,
            "cpu",
            "embedding.cpu_gather",
            None,
            None,
            #[cfg(feature = "wgpu")]
            wgpu_failure.as_ref().map(|_| "wgpu"),
            #[cfg(not(feature = "wgpu"))]
            None,
            #[cfg(feature = "wgpu")]
            wgpu_failure.as_deref(),
            #[cfg(not(feature = "wgpu"))]
            None,
        );
        Ok(output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let (batch, steps) = input.shape();
        let output_cols = steps * self.embed_dim;
        if grad_output.shape() != (batch, output_cols) {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (batch, output_cols),
            });
        }
        if batch == 0 || steps == 0 {
            let grad_input = Tensor::zeros(batch, steps)?;
            emit_embedding_meta(
                "embedding_backward",
                input.shape(),
                grad_input.shape(),
                self.vocab_size,
                self.embed_dim,
                TokenIndexStats::default(),
                "auto",
                "cpu",
                "embedding.cpu_scatter_add",
                None,
                None,
                None,
                None,
            );
            return Ok(grad_input);
        }

        let input_data = input.data();
        let grad_data = grad_output.data();
        validate_finite_tensor("embedding_grad_output", grad_output)?;
        let (indices, stats) = collect_token_indices(input_data, batch, steps, self.vocab_size);
        let scatter_backend =
            current_tensor_util_backend_for_values(self.vocab_size.saturating_mul(self.embed_dim));
        let requested_backend = tensor_util_backend_label(scatter_backend);
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        let maybe_grad_weight = if matches!(scatter_backend, TensorUtilBackend::GpuWgpu)
            && wgpu_dense::is_available()
        {
            match wgpu_dense::embedding_scatter_add(
                &indices,
                grad_data,
                indices.len(),
                self.vocab_size,
                self.embed_dim,
            ) {
                Ok(grad_weight) => Some(grad_weight),
                Err(message) if strict_gpu_path() => {
                    return Err(embedding_wgpu_error("embedding_backward", message));
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
        let maybe_grad_weight: Option<Vec<f32>> = None;

        let (grad_weight, scatter_backend_label, scatter_kernel) =
            if let Some(grad_weight) = maybe_grad_weight {
                validate_finite_slice("embedding_grad_weight", &grad_weight)?;
                (grad_weight, "wgpu_dense", "embedding.wgpu_scatter_add")
            } else {
                let mut grad_weight = vec![0.0f32; self.vocab_size * self.embed_dim];
                for b in 0..batch {
                    let grad_row = b * output_cols;
                    for t in 0..steps {
                        let idx = indices[b * steps + t] as usize;
                        let gw_base = idx * self.embed_dim;
                        let go_base = grad_row + t * self.embed_dim;
                        for c in 0..self.embed_dim {
                            let contribution = grad_data[go_base + c];
                            validate_finite_value("embedding_grad_contribution", contribution)?;
                            grad_weight[gw_base + c] += contribution;
                            validate_finite_value(
                                "embedding_grad_weight_sum",
                                grad_weight[gw_base + c],
                            )?;
                        }
                    }
                }
                (grad_weight, "cpu", "embedding.cpu_scatter_add")
            };
        validate_finite_slice("embedding_grad_weight", &grad_weight)?;
        let grad_w = Tensor::from_vec(self.vocab_size, self.embed_dim, grad_weight)?;
        let gradient_scale = 1.0 / batch as f32;
        let gradient_backend = current_tensor_util_backend_for_values(grad_w.data().len());
        let grad_w = grad_w.scale_with_backend(gradient_scale, gradient_backend)?;
        validate_finite_tensor("embedding_grad_weight", &grad_w)?;
        let grad_input = Tensor::zeros(batch, steps)?;
        self.weight.accumulate_euclidean(&grad_w)?;

        emit_embedding_meta(
            "embedding_backward",
            input.shape(),
            grad_input.shape(),
            self.vocab_size,
            self.embed_dim,
            stats,
            requested_backend,
            scatter_backend_label,
            scatter_kernel,
            Some(gradient_backend.to_string()),
            Some(gradient_scale),
            #[cfg(feature = "wgpu")]
            wgpu_failure.as_ref().map(|_| "wgpu"),
            #[cfg(not(feature = "wgpu"))]
            None,
            #[cfg(feature = "wgpu")]
            wgpu_failure.as_deref(),
            #[cfg(not(feature = "wgpu"))]
            None,
        );
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.weight)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.weight)?;
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

    #[test]
    fn embedding_forward_picks_rows() {
        let layer = Embedding::new("emb", 4, 3).unwrap();
        let input = Tensor::from_vec(2, 3, vec![0.0, 1.0, 3.0, 2.0, 1.0, 0.0]).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), (2, 9));

        let weights = layer.weight().value().data();
        let expect_row = |idx: usize| -> Vec<f32> {
            let start = idx * 3;
            weights[start..start + 3].to_vec()
        };
        let out = output.data();
        assert_eq!(out[0..3], expect_row(0));
        assert_eq!(out[3..6], expect_row(1));
        assert_eq!(out[6..9], expect_row(3));
        assert_eq!(out[9..12], expect_row(2));
        assert_eq!(out[12..15], expect_row(1));
        assert_eq!(out[15..18], expect_row(0));
    }

    #[test]
    fn embedding_forward_rejects_non_finite_weight() {
        let mut layer = Embedding::new("emb", 4, 2).unwrap();
        layer.weight.value_mut().data_mut()[0] = f32::NAN;
        let input = Tensor::from_vec(1, 2, vec![0.0, 1.0]).unwrap();

        let err = layer.forward(&input).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "embedding_weight",
                value,
            } if value.is_nan()
        ));
    }

    #[test]
    fn embedding_backward_updates_weight() {
        let mut layer = Embedding::new("emb", 5, 2).unwrap();
        layer.attach_hypergrad(-1.0, 0.05).unwrap();
        let input = Tensor::from_vec(
            3,
            4,
            vec![0.0, 1.0, 2.0, 3.0, 1.0, 1.0, 4.0, 0.0, 2.0, 2.0, 2.0, 2.0],
        )
        .unwrap();
        let output = layer.forward(&input).unwrap();
        let grad_output = Tensor::from_vec(
            output.shape().0,
            output.shape().1,
            vec![1.0; output.data().len()],
        )
        .unwrap();
        let grad_in = layer.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_in.shape(), input.shape());
        assert!(grad_in.data().iter().all(|v| *v == 0.0));

        let before = layer.weight().value().clone();
        layer.apply_step(0.01).unwrap();
        let after = layer.weight().value();
        assert_ne!(before, *after);
    }

    #[test]
    fn embedding_backward_rejects_non_finite_grad_without_accumulating() {
        let mut layer = Embedding::new("emb", 4, 2).unwrap();
        let input = Tensor::from_vec(1, 2, vec![0.0, 1.0]).unwrap();
        let grad_output = Tensor::from_vec(1, 4, vec![0.5, f32::NAN, 0.25, 0.75]).unwrap();

        let err = layer.backward(&input, &grad_output).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "embedding_grad_output",
                value,
            } if value.is_nan()
        ));
        assert!(layer.weight().gradient().is_none());
    }

    #[test]
    fn embedding_backward_rejects_overflowing_scatter_without_accumulating() {
        let mut layer = Embedding::new("emb", 2, 1).unwrap();
        let input = Tensor::from_vec(1, 2, vec![0.0, 0.0]).unwrap();
        let grad_output = Tensor::from_vec(1, 2, vec![f32::MAX, f32::MAX]).unwrap();

        let err = layer.backward(&input, &grad_output).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "embedding_grad_weight_sum",
                value,
            } if value.is_infinite()
        ));
        assert!(layer.weight().gradient().is_none());
    }

    #[test]
    fn embedding_empty_steps_emit_meta_without_weight_update() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut layer = Embedding::new("emb_empty", 4, 2).unwrap();
        let input = Tensor::from_vec(3, 0, Vec::new()).unwrap();
        let output = layer.forward(&input).unwrap();
        let grad_input = layer.backward(&input, &output).unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        assert_eq!(output.shape(), (3, 0));
        assert_eq!(grad_input.shape(), input.shape());
        assert!(layer.weight().gradient().is_none());

        let events = events.lock().unwrap();
        let forward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "embedding_forward"
                    && data["batch"] == 3
                    && data["steps"] == 0
                    && data["tokens"] == 0
            })
            .expect("empty embedding forward metadata event");
        assert_eq!(forward.1["backend"], "cpu");

        let backward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "embedding_backward"
                    && data["batch"] == 3
                    && data["steps"] == 0
                    && data["tokens"] == 0
            })
            .expect("empty embedding backward metadata event");
        assert_eq!(backward.1["backend"], "cpu");
    }

    #[test]
    fn embedding_forward_backward_emit_backend_meta_and_token_repairs() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut layer = Embedding::new("emb_meta", 4, 2).unwrap();
        let input = Tensor::from_vec(1, 5, vec![0.0, 1.25, f32::NAN, 9.0, -2.0]).unwrap();
        let output = layer.forward(&input).unwrap();
        let grad_output = Tensor::from_vec(1, 10, vec![0.5; 10]).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        assert_eq!(output.shape(), (1, 10));
        assert_eq!(grad_input.shape(), input.shape());
        let events = events.lock().unwrap();
        let forward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "embedding_forward"
                    && data["batch"] == 1
                    && data["steps"] == 5
                    && data["embed_dim"] == 2
            })
            .expect("embedding forward metadata event");
        assert_eq!(forward.1["backend"], "cpu");
        assert_eq!(forward.1["tokens"], 5);
        assert_eq!(forward.1["unique_token_indices"], 3);
        assert_eq!(forward.1["repeated_token_indices"], 2);
        assert_eq!(forward.1["non_finite_tokens"], 1);
        assert_eq!(forward.1["rounded_tokens"], 1);
        assert_eq!(forward.1["clamped_low_tokens"], 1);
        assert_eq!(forward.1["clamped_high_tokens"], 1);

        let backward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "embedding_backward"
                    && data["batch"] == 1
                    && data["steps"] == 5
                    && data["embed_dim"] == 2
            })
            .expect("embedding backward metadata event");
        assert_eq!(backward.1["backend"], "cpu");
        assert_eq!(backward.1["tokens"], 5);
        assert_eq!(backward.1["unique_token_indices"], 3);
        assert_eq!(backward.1["repeated_token_indices"], 2);
        assert_eq!(backward.1["non_finite_tokens"], 1);
        assert_eq!(backward.1["rounded_tokens"], 1);
        assert_eq!(backward.1["clamped_low_tokens"], 1);
        assert_eq!(backward.1["clamped_high_tokens"], 1);
        assert_eq!(backward.1["gradient_reduction_backend"], "auto");
        assert_eq!(backward.1["gradient_scale"], 1.0);
        assert_eq!(backward.1["grad_weight_values"], 8);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn embedding_forward_backward_can_run_wgpu_when_policy_forced() {
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

        let policy = BackendPolicy::from_device_caps(DeviceCaps::wgpu(32, true, 256));
        let guard = push_backend_policy(policy);
        let mut layer = Embedding::new("emb_wgpu", 5, 3).unwrap();
        let input = Tensor::from_vec(2, 4, vec![0.0, 1.0, 2.0, 1.0, 4.0, 3.0, 2.0, 4.0]).unwrap();
        let output = layer.forward(&input).unwrap();
        let grad_output = Tensor::from_vec(
            output.shape().0,
            output.shape().1,
            (0..output.data().len())
                .map(|idx| idx as f32 * 0.01 - 0.05)
                .collect(),
        )
        .unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();
        drop(guard);
        st_tensor::set_tensor_op_meta_observer(previous);
        match previous_threshold {
            Some(value) => std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", value),
            None => std::env::remove_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES"),
        }

        assert_eq!(output.shape(), (2, 12));
        assert_eq!(grad_input.shape(), input.shape());
        let events = events.lock().unwrap();
        let forward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "embedding_forward"
                    && data["backend"] == "wgpu_dense"
                    && data["kernel"] == "embedding.wgpu_gather"
            })
            .expect("wgpu embedding forward metadata event");
        assert_eq!(forward.1["requested_backend"], "wgpu");
        assert_eq!(forward.1["tokens"], 8);
        assert_eq!(forward.1["repeated_token_indices"], 3);

        let backward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "embedding_backward"
                    && data["backend"] == "wgpu_dense"
                    && data["kernel"] == "embedding.wgpu_scatter_add"
            })
            .expect("wgpu embedding backward metadata event");
        assert_eq!(backward.1["requested_backend"], "wgpu");
        assert_eq!(backward.1["gradient_reduction_backend"], "wgpu");
        assert_eq!(backward.1["tokens"], 8);
    }
}
