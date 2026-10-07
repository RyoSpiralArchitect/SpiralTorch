// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::execution::current_tensor_util_backend_for_values;
use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
#[cfg(feature = "wgpu")]
use st_core::dynamics::topos_resonator::{
    audit_topos_resonator, audit_topos_resonator_backward, backward_topos_resonator,
    ToposResonatorAuditRequest, ToposResonatorBackwardAuditRequest, ToposResonatorBackwardRequest,
};
use st_core::dynamics::topos_resonator::{
    validate_topos_resonator_state, ToposResonatorLearningBatch, ToposResonatorOperator,
    ToposResonatorRequest, TOPOS_RESONATOR_BACKWARD, TOPOS_RESONATOR_CONTRACT_VERSION,
    TOPOS_RESONATOR_EQUATION, TOPOS_RESONATOR_REWRITE, TOPOS_RESONATOR_SCHEME,
    TOPOS_RESONATOR_SEMANTIC_BACKEND, TOPOS_RESONATOR_SEMANTIC_OWNER, TOPOS_RESONATOR_STABILITY,
    TOPOS_RESONATOR_STATE,
};
pub use st_core::dynamics::topos_resonator::{
    ToposResonatorAudit, ToposResonatorBackwardAudit, ToposResonatorConfig,
};
#[cfg(feature = "wgpu")]
use st_tensor::backend::wgpu_dense;
use st_tensor::topos::OpenCartesianTopos;
use st_tensor::{
    emit_tensor_op, emit_tensor_op_meta, LanguageWaveEncoder, Layout, TensorUtilBackend,
};
use std::borrow::Cow;
use std::cell::RefCell;
use std::sync::Arc;

const DEFAULT_CURVATURE: f32 = -1.0;
const DEFAULT_TOLERANCE: f32 = 1e-6;
const DEFAULT_SATURATION: f32 = 1e6;
const DEFAULT_MAX_DEPTH: usize = 64;

/// Whether each input element owns a gate or rows share one gate per feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToposGateLayout {
    Elementwise,
    SharedRows,
}

fn topos_resonator_error(error: impl std::fmt::Display) -> TensorError {
    TensorError::Generic(format!("Topos resonator contract failed: {error}"))
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
    crate::execution::current_accelerator_fallback().is_strict()
}

#[cfg(feature = "wgpu")]
fn topos_resonator_wgpu_error(op_name: &'static str, message: String) -> TensorError {
    TensorError::BackendFailure {
        backend: "wgpu",
        message: format!("{op_name} wgpu path failed ({message}); fallback disabled"),
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_topos_resonator_meta(
    op_name: &'static str,
    rows: usize,
    cols: usize,
    backend: &'static str,
    requested_backend: TensorUtilBackend,
    kernel: &'static str,
    config: ToposResonatorConfig,
    topos: &OpenCartesianTopos,
    gate_layout: ToposGateLayout,
    encoder_attached: bool,
    backward: bool,
    captured: bool,
    audit: ToposResonatorAudit,
    backward_audit: Option<ToposResonatorBackwardAudit>,
    fallback: Option<String>,
) {
    emit_tensor_op(op_name, &[rows, cols, rows, cols], &[rows, cols]);
    emit_tensor_op_meta(op_name, || {
        let values = rows.saturating_mul(cols);
        let gate_rows = if gate_layout == ToposGateLayout::SharedRows {
            1
        } else {
            rows
        };
        let estimated_ops_per_value = if backward && captured {
            3
        } else {
            config
                .iterations()
                .saturating_mul(if backward || captured { 13 } else { 6 })
        };
        let mut data = serde_json::json!({
            "backend": backend,
            "requested_backend": tensor_util_backend_label(requested_backend),
            "delegate_backend": backend,
            "kernel": kernel,
            "kind": if backward { "topos_resonator_backward" } else { "topos_resonator_forward" },
            "rows": rows,
            "cols": cols,
            "values": values,
            "output_rows": rows,
            "output_cols": cols,
            "output_values": values,
            "gate_rows": gate_rows,
            "gate_cols": cols,
            "gate_values": gate_rows.saturating_mul(cols),
            "trainable_parameters": gate_rows.saturating_mul(cols),
            "encoder_attached": encoder_attached,
            "backward": backward,
            "contract_version": TOPOS_RESONATOR_CONTRACT_VERSION,
            "semantic_owner": TOPOS_RESONATOR_SEMANTIC_OWNER,
            "semantic_backend": TOPOS_RESONATOR_SEMANTIC_BACKEND,
            "equation": TOPOS_RESONATOR_EQUATION,
            "rewrite": TOPOS_RESONATOR_REWRITE,
            "scheme": TOPOS_RESONATOR_SCHEME,
            "resonance_state": TOPOS_RESONATOR_STATE,
            "stability_contract": TOPOS_RESONATOR_STABILITY,
            "backward_contract": TOPOS_RESONATOR_BACKWARD,
            "coupling": config.coupling(),
            "iterations": config.iterations(),
            "contraction_bound": config.contraction_bound(),
            "amplification_bound": config.amplification_bound(),
            "curvature": topos.curvature(),
            "saturation": topos.saturation(),
            "porosity": topos.porosity(),
            "tolerance": topos.tolerance(),
            "max_depth": topos.max_depth(),
            "max_volume": topos.max_volume(),
            "audit": audit,
            "backward_audit": backward_audit,
            "estimated_ops_per_value": estimated_ops_per_value,
            "estimated_total_ops": values.saturating_mul(estimated_ops_per_value),
            "empty": values == 0,
        });
        data["captured_sensitivity"] = serde_json::json!(captured);
        data["gate_layout"] = serde_json::json!(match gate_layout {
            ToposGateLayout::Elementwise => "elementwise",
            ToposGateLayout::SharedRows => "shared_rows",
        });
        data["gate_gradient_reduction"] =
            serde_json::json!(if backward && gate_layout == ToposGateLayout::SharedRows {
                "sum_axis0_no_additional_mean"
            } else {
                "none"
            });
        data["backward_audit_scope"] = serde_json::json!("expanded_elementwise_vjps");
        data["capture_reused"] = serde_json::json!(backward && captured);
        data["estimated_ops_scope"] =
            serde_json::json!("pointwise_recurrence_or_vjp_excludes_audit_and_transport");
        data["finite_amplification_bound"] = serde_json::json!(config.finite_amplification_bound());
        if let Some(message) = fallback {
            data["fallback"] = serde_json::json!({"from": "wgpu", "message": message});
        }
        data
    });
}

#[derive(Clone, Debug)]
enum ToposResonatorSaved {
    Captured(Arc<ToposResonatorLearningBatch>),
    #[cfg(feature = "wgpu")]
    Recomputed {
        input: Tensor,
        gate: Tensor,
        output: Tensor,
    },
}

impl ToposResonatorSaved {
    fn input(&self) -> &[f32] {
        match self {
            Self::Captured(batch) => batch.input(),
            #[cfg(feature = "wgpu")]
            Self::Recomputed { input, .. } => input.data(),
        }
    }

    #[cfg(feature = "wgpu")]
    fn gate(&self) -> &[f32] {
        match self {
            Self::Captured(batch) => batch.gate(),
            #[cfg(feature = "wgpu")]
            Self::Recomputed { gate, .. } => gate.data(),
        }
    }

    fn output(&self) -> &[f32] {
        match self {
            Self::Captured(batch) => batch.output(),
            #[cfg(feature = "wgpu")]
            Self::Recomputed { output, .. } => output.data(),
        }
    }
}

#[derive(Clone, Debug)]
struct ToposResonatorStepCache {
    saved: ToposResonatorSaved,
    parameter_gate: Tensor,
    audit: ToposResonatorAudit,
    backward_audit: Option<ToposResonatorBackwardAudit>,
}

/// One audited open-topos resonance transition exposed as a tensor.
#[derive(Clone, Debug)]
pub struct ToposResonatorTensorStep {
    pub output: Tensor,
    pub audit: ToposResonatorAudit,
}

/// Exact unrolled gradient for a cached open-topos resonance transition.
#[derive(Clone, Debug)]
pub struct ToposResonatorTensorBackward {
    pub grad_input: Tensor,
    pub audit: ToposResonatorBackwardAudit,
}

/// Trainable finite Picard resonator toward a unique open-topos guarded fixed point.
///
/// `st-core` owns the recurrence, contraction proof boundary, porous rewrite,
/// exact unrolled derivative, and audits. This layer owns only the trainable
/// gate, backend routing, forward-cache integrity, and module integration.
#[derive(Debug)]
pub struct ToposResonator {
    gate: Parameter,
    gate_layout: ToposGateLayout,
    encoder: Option<LanguageWaveEncoder>,
    topos: OpenCartesianTopos,
    config: ToposResonatorConfig,
    last_step: RefCell<Option<ToposResonatorStepCache>>,
}

impl ToposResonator {
    /// Creates a resonator with an identity drive gate and a bounded default topos.
    pub fn new(name: impl Into<String>, rows: usize, cols: usize) -> PureResult<Self> {
        let volume = checked_layer_volume(rows, cols)?;
        let topos = OpenCartesianTopos::new(
            DEFAULT_CURVATURE,
            DEFAULT_TOLERANCE,
            DEFAULT_SATURATION,
            DEFAULT_MAX_DEPTH,
            volume,
        )?;
        Self::with_config_and_topos(name, rows, cols, ToposResonatorConfig::default(), topos)
    }

    /// Creates a resonator with an inferred topos and an explicit core contract.
    pub fn with_config(
        name: impl Into<String>,
        rows: usize,
        cols: usize,
        config: ToposResonatorConfig,
    ) -> PureResult<Self> {
        let volume = checked_layer_volume(rows, cols)?;
        let max_depth = config.iterations().saturating_add(1).max(DEFAULT_MAX_DEPTH);
        let topos = OpenCartesianTopos::new(
            DEFAULT_CURVATURE,
            DEFAULT_TOLERANCE,
            DEFAULT_SATURATION,
            max_depth,
            volume,
        )?;
        Self::with_config_and_topos(name, rows, cols, config, topos)
    }

    /// Creates a resonator with explicit versioned dynamics and guard topology.
    pub fn with_config_and_topos(
        name: impl Into<String>,
        rows: usize,
        cols: usize,
        config: ToposResonatorConfig,
        topos: OpenCartesianTopos,
    ) -> PureResult<Self> {
        let volume = checked_layer_volume(rows, cols)?;
        validate_layer_topos(volume, config, &topos)?;
        let weights = Tensor::from_vec(rows, cols, vec![1.0; volume])?;
        Ok(Self {
            gate: Parameter::new(name, weights),
            gate_layout: ToposGateLayout::Elementwise,
            encoder: None,
            topos,
            config,
            last_step: RefCell::new(None),
        })
    }

    /// Learns one gate per feature across variable-sized row batches.
    ///
    /// The supplied topos bounds each expanded input, not just the parameter.
    /// Backward sums row contributions without averaging them; loss reduction
    /// is already represented in the caller's upstream gradient. Core audits
    /// describe the expanded elementwise VJPs, before this finite-checked sum.
    pub fn with_shared_gate(
        name: impl Into<String>,
        features: usize,
        config: ToposResonatorConfig,
        topos: OpenCartesianTopos,
    ) -> PureResult<Self> {
        let mut layer = Self::with_config_and_topos(name, 1, features, config, topos)?;
        layer.gate_layout = ToposGateLayout::SharedRows;
        Ok(layer)
    }

    pub fn gate_layout(&self) -> ToposGateLayout {
        self.gate_layout
    }

    pub fn config(&self) -> ToposResonatorConfig {
        self.config
    }

    pub fn topos(&self) -> &OpenCartesianTopos {
        &self.topos
    }

    pub fn coupling(&self) -> f32 {
        self.config.coupling()
    }

    pub fn iterations(&self) -> usize {
        self.config.iterations()
    }

    pub fn with_coupling(mut self, coupling: f32) -> PureResult<Self> {
        let config = self
            .config
            .with_coupling(coupling)
            .map_err(topos_resonator_error)?;
        validate_layer_topos(self.gate.value().data().len(), config, &self.topos)?;
        self.config = config;
        self.last_step.get_mut().take();
        Ok(self)
    }

    pub fn with_iterations(mut self, iterations: usize) -> PureResult<Self> {
        let config = self
            .config
            .with_iterations(iterations)
            .map_err(topos_resonator_error)?;
        validate_layer_topos(self.gate.value().data().len(), config, &self.topos)?;
        self.config = config;
        self.last_step.get_mut().take();
        Ok(self)
    }

    /// Provides immutable access to the trainable drive gate.
    pub fn parameter(&self) -> &Parameter {
        &self.gate
    }

    /// Provides mutable gate access and invalidates any cached transition.
    pub fn parameter_mut(&mut self) -> &mut Parameter {
        self.last_step.get_mut().take();
        &mut self.gate
    }

    /// Attaches a text encoder used only to accumulate gate updates.
    pub fn with_encoder(mut self, encoder: LanguageWaveEncoder) -> Self {
        self.encoder = Some(encoder);
        self
    }

    /// Streams raw text into the gate accumulator when an encoder is attached.
    pub fn infuse_text(&mut self, text: &str) -> PureResult<()> {
        let encoder = self
            .encoder
            .as_ref()
            .ok_or(TensorError::EmptyInput("topos resonator encoder"))?;
        self.gate.absorb_text(encoder, text)
    }

    /// Makes a caller-supplied open topos authoritative for both forward and optimisation.
    pub fn attach_open_topos(
        &mut self,
        curvature: f32,
        learning_rate: f32,
        topos: OpenCartesianTopos,
    ) -> PureResult<()> {
        if (curvature - topos.curvature()).abs() > topos.tolerance() {
            return Err(TensorError::CurvatureMismatch {
                expected: topos.curvature(),
                got: curvature,
            });
        }
        validate_layer_topos(self.gate.value().data().len(), self.config, &topos)?;
        self.gate
            .attach_hypergrad_with_topos(curvature, learning_rate, topos.clone())?;
        self.topos = topos;
        self.last_step.get_mut().take();
        Ok(())
    }

    pub fn latest_output(&self) -> Option<Tensor> {
        self.last_step.borrow().as_ref().map(|step| {
            Tensor::from_vec(
                step.audit.rows,
                step.audit.features,
                step.saved.output().to_vec(),
            )
            .expect("cached Topos shape was validated at capture")
        })
    }

    pub fn latest_audit(&self) -> Option<ToposResonatorAudit> {
        self.last_step.borrow().as_ref().map(|step| step.audit)
    }

    pub fn latest_backward_audit(&self) -> Option<ToposResonatorBackwardAudit> {
        self.last_step
            .borrow()
            .as_ref()
            .and_then(|step| step.backward_audit)
    }

    fn core_request<'a>(&'a self, input: &'a Tensor, gate: &'a [f32]) -> ToposResonatorRequest<'a> {
        let (rows, features) = input.shape();
        ToposResonatorRequest {
            input: input.data(),
            gate,
            rows,
            features,
            config: self.config,
            topos: &self.topos,
        }
    }

    fn validate_optimizer_topos_alignment(&self) -> PureResult<()> {
        let Some(hypergrad) = self.gate.hypergrad() else {
            return Ok(());
        };
        let optimizer_topos = hypergrad.topos();
        let curvature_aligned =
            (hypergrad.curvature() - self.topos.curvature()).abs() <= self.topos.tolerance();
        let guard_aligned = optimizer_topos.curvature().to_bits()
            == self.topos.curvature().to_bits()
            && optimizer_topos.tolerance().to_bits() == self.topos.tolerance().to_bits()
            && optimizer_topos.saturation().to_bits() == self.topos.saturation().to_bits()
            && optimizer_topos.porosity().to_bits() == self.topos.porosity().to_bits()
            && optimizer_topos.max_depth() == self.topos.max_depth()
            && optimizer_topos.max_volume() == self.topos.max_volume();
        if curvature_aligned && guard_aligned {
            Ok(())
        } else {
            Err(TensorError::InvalidValue {
                label: "topos_resonator_optimizer_topos_mismatch",
            })
        }
    }

    fn topos_with_curvature(&self, curvature: f32) -> PureResult<OpenCartesianTopos> {
        OpenCartesianTopos::new(
            curvature,
            self.topos.tolerance(),
            self.topos.saturation(),
            self.topos.max_depth(),
            self.topos.max_volume(),
        )?
        .with_porosity(self.topos.porosity())
    }

    #[cfg(feature = "wgpu")]
    fn commit_step(
        &self,
        input: &Tensor,
        gate: &[f32],
        output: Vec<f32>,
        audit: ToposResonatorAudit,
    ) -> PureResult<ToposResonatorTensorStep> {
        let (rows, cols) = input.shape();
        let output = Tensor::from_vec(rows, cols, output)?;
        let saved_gate = match self.gate_layout {
            ToposGateLayout::Elementwise => self.gate.value().to_layout(Layout::RowMajor)?,
            ToposGateLayout::SharedRows => Tensor::from_vec(rows, cols, gate.to_vec())?,
        };
        self.last_step
            .borrow_mut()
            .replace(ToposResonatorStepCache {
                saved: ToposResonatorSaved::Recomputed {
                    input: input.clone(),
                    gate: saved_gate,
                    output: output.clone(),
                },
                parameter_gate: self.gate.value().clone(),
                audit,
                backward_audit: None,
            });
        Ok(ToposResonatorTensorStep { output, audit })
    }

    fn commit_capture(
        &self,
        batch: ToposResonatorLearningBatch,
    ) -> PureResult<ToposResonatorTensorStep> {
        let audit = batch.step().audit;
        let output = Tensor::from_vec(audit.rows, audit.features, batch.output().to_vec())?;
        self.last_step
            .borrow_mut()
            .replace(ToposResonatorStepCache {
                saved: ToposResonatorSaved::Captured(Arc::new(batch)),
                parameter_gate: self.gate.value().clone(),
                audit,
                backward_audit: None,
            });
        Ok(ToposResonatorTensorStep { output, audit })
    }

    fn commit_backward(&self, audit: ToposResonatorBackwardAudit) {
        if let Some(step) = self.last_step.borrow_mut().as_mut() {
            step.backward_audit = Some(audit);
        }
    }

    /// Evaluates one audited resonance transition through the selected executor.
    pub fn step_resonance(&self, input: &Tensor) -> PureResult<ToposResonatorTensorStep> {
        let (rows, cols) = input.shape();
        let expected_gate_shape = match self.gate_layout {
            ToposGateLayout::Elementwise => (rows, cols),
            ToposGateLayout::SharedRows => (1, cols),
        };
        if self.gate.value().shape() != expected_gate_shape {
            return Err(TensorError::ShapeMismatch {
                left: self.gate.value().shape(),
                right: expected_gate_shape,
            });
        }
        validate_layer_topos(input.data().len(), self.config, &self.topos)?;
        self.validate_optimizer_topos_alignment()?;
        let input = input.to_layout(Layout::RowMajor)?;
        let parameter_gate = self.gate.value().to_layout(Layout::RowMajor)?;
        let gate = match self.gate_layout {
            ToposGateLayout::Elementwise => Cow::Borrowed(parameter_gate.data()),
            ToposGateLayout::SharedRows => Cow::Owned(parameter_gate.data().repeat(rows)),
        };
        let request = self.core_request(&input, &gate);
        validate_topos_resonator_state(request).map_err(topos_resonator_error)?;
        let (rows, cols) = input.shape();
        let route_backend = current_tensor_util_backend_for_values(input.data().len());
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        if matches!(route_backend, TensorUtilBackend::GpuWgpu) {
            if wgpu_dense::is_available() {
                let wgpu_step = wgpu_dense::topos_resonator_forward(
                    input.data(),
                    &gate,
                    rows,
                    cols,
                    self.config.coupling(),
                    self.topos.saturation(),
                    self.topos.porosity(),
                    self.config.iterations(),
                )
                .and_then(|output| {
                    audit_topos_resonator(ToposResonatorAuditRequest {
                        request,
                        output: &output,
                    })
                    .map(|audit| (output, audit))
                    .map_err(|error| format!("Rust semantic audit failed: {error}"))
                });
                match wgpu_step {
                    Ok((output, audit)) => {
                        let step = self.commit_step(&input, &gate, output, audit)?;
                        emit_topos_resonator_meta(
                            "topos_resonator_forward",
                            rows,
                            cols,
                            "wgpu_dense",
                            route_backend,
                            "tensor_util.topos_resonator_forward",
                            self.config,
                            &self.topos,
                            self.gate_layout,
                            self.encoder.is_some(),
                            false,
                            false,
                            audit,
                            None,
                            None,
                        );
                        return Ok(step);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(topos_resonator_wgpu_error(
                            "topos_resonator_forward",
                            message,
                        ));
                    }
                    Err(message) => wgpu_failure = Some(message),
                }
            } else if strict_gpu_path() {
                return Err(topos_resonator_wgpu_error(
                    "topos_resonator_forward",
                    "WGPU backend not available".to_string(),
                ));
            } else {
                wgpu_failure = Some("WGPU backend not available".to_string());
            }
        }

        let operator = ToposResonatorOperator::new(self.config, self.topos.clone())
            .map_err(topos_resonator_error)?;
        let batch = match gate {
            Cow::Borrowed(gate) => operator.capture(input.data(), gate, rows, cols),
            Cow::Owned(gate) => operator.capture_owned(input.data().to_vec(), gate, rows, cols),
        }
        .map_err(topos_resonator_error)?;
        let step = self.commit_capture(batch)?;
        let audit = step.audit;
        emit_topos_resonator_meta(
            "topos_resonator_forward",
            rows,
            cols,
            "cpu",
            route_backend,
            "st_core.ToposResonatorOperator.capture",
            self.config,
            &self.topos,
            self.gate_layout,
            self.encoder.is_some(),
            false,
            true,
            audit,
            None,
            {
                #[cfg(feature = "wgpu")]
                {
                    wgpu_failure
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    None
                }
            },
        );
        Ok(step)
    }

    fn accumulate_gate_vjp(
        &mut self,
        grad_gate: Tensor,
        backend: TensorUtilBackend,
    ) -> PureResult<()> {
        let grad_gate = match self.gate_layout {
            ToposGateLayout::Elementwise => grad_gate,
            ToposGateLayout::SharedRows => Tensor::from_vec(
                1,
                grad_gate.shape().1,
                grad_gate.try_sum_axis0_with_backend(backend)?,
            )?,
        };
        self.gate.accumulate_euclidean(&grad_gate)
    }

    /// Differentiates the exact cached transition and accumulates its gate gradient.
    pub fn backward_resonance(
        &mut self,
        input: &Tensor,
        grad_output: &Tensor,
    ) -> PureResult<ToposResonatorTensorBackward> {
        let cache = self
            .last_step
            .borrow()
            .as_ref()
            .cloned()
            .ok_or(TensorError::InvalidValue {
                label: "topos_resonator_forward_cache",
            })?;
        let cached_shape = (cache.audit.rows, cache.audit.features);
        if input.shape() != cached_shape {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: cached_shape,
            });
        }
        if grad_output.shape() != cached_shape {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: cached_shape,
            });
        }
        let input = input.to_layout(Layout::RowMajor)?;
        let grad_output = grad_output.to_layout(Layout::RowMajor)?;
        if input.data() != cache.saved.input() {
            return Err(TensorError::InvalidValue {
                label: "topos_resonator_forward_input_mismatch",
            });
        }
        if self.gate.value().shape() != cache.parameter_gate.shape()
            || self.gate.value().data() != cache.parameter_gate.data()
        {
            return Err(TensorError::InvalidValue {
                label: "topos_resonator_forward_gate_mismatch",
            });
        }
        let (rows, cols) = input.shape();
        #[cfg(feature = "wgpu")]
        let backward_request = ToposResonatorBackwardRequest {
            request: ToposResonatorRequest {
                input: input.data(),
                gate: cache.saved.gate(),
                rows,
                features: cols,
                config: self.config,
                topos: &self.topos,
            },
            grad_output: grad_output.data(),
        };
        let route_backend = current_tensor_util_backend_for_values(input.data().len());
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        if matches!(route_backend, TensorUtilBackend::GpuWgpu) {
            if wgpu_dense::is_available() {
                let wgpu_backward = wgpu_dense::topos_resonator_backward(
                    input.data(),
                    cache.saved.gate(),
                    grad_output.data(),
                    rows,
                    cols,
                    self.config.coupling(),
                    self.topos.saturation(),
                    self.topos.porosity(),
                    self.config.iterations(),
                )
                .and_then(|(grad_input, grad_gate)| {
                    audit_topos_resonator_backward(ToposResonatorBackwardAuditRequest {
                        request: backward_request,
                        grad_input: &grad_input,
                        grad_gate: &grad_gate,
                    })
                    .map(|audit| (grad_input, grad_gate, audit))
                    .map_err(|error| format!("Rust semantic backward audit failed: {error}"))
                });
                match wgpu_backward {
                    Ok((grad_input, grad_gate, backward_audit)) => {
                        let grad_input = Tensor::from_vec(rows, cols, grad_input)?;
                        let grad_gate = Tensor::from_vec(rows, cols, grad_gate)?;
                        self.accumulate_gate_vjp(grad_gate, TensorUtilBackend::GpuWgpu)?;
                        self.commit_backward(backward_audit);
                        emit_topos_resonator_meta(
                            "topos_resonator_backward",
                            rows,
                            cols,
                            "wgpu_dense",
                            route_backend,
                            "tensor_util.topos_resonator_backward",
                            self.config,
                            &self.topos,
                            self.gate_layout,
                            self.encoder.is_some(),
                            true,
                            false,
                            cache.audit,
                            Some(backward_audit),
                            None,
                        );
                        return Ok(ToposResonatorTensorBackward {
                            grad_input,
                            audit: backward_audit,
                        });
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(topos_resonator_wgpu_error(
                            "topos_resonator_backward",
                            message,
                        ));
                    }
                    Err(message) => wgpu_failure = Some(message),
                }
            } else if strict_gpu_path() {
                return Err(topos_resonator_wgpu_error(
                    "topos_resonator_backward",
                    "WGPU backend not available".to_string(),
                ));
            } else {
                wgpu_failure = Some("WGPU backend not available".to_string());
            }
        }

        let (backward, backward_audit, captured) = match &cache.saved {
            ToposResonatorSaved::Captured(batch) => {
                let (backward, audit) = batch
                    .vjp_audited(grad_output.data())
                    .map_err(topos_resonator_error)?;
                (backward, audit, true)
            }
            #[cfg(feature = "wgpu")]
            ToposResonatorSaved::Recomputed { .. } => {
                let backward =
                    backward_topos_resonator(backward_request).map_err(topos_resonator_error)?;
                let audit = audit_topos_resonator_backward(ToposResonatorBackwardAuditRequest {
                    request: backward_request,
                    grad_input: &backward.grad_input,
                    grad_gate: &backward.grad_gate,
                })
                .map_err(topos_resonator_error)?;
                (backward, audit, false)
            }
        };
        let grad_input = Tensor::from_vec(rows, cols, backward.grad_input)?;
        let grad_gate = Tensor::from_vec(rows, cols, backward.grad_gate)?;
        self.accumulate_gate_vjp(grad_gate, TensorUtilBackend::Cpu)?;
        self.commit_backward(backward_audit);
        emit_topos_resonator_meta(
            "topos_resonator_backward",
            rows,
            cols,
            "cpu",
            route_backend,
            if captured {
                "st_core.ToposResonatorLearningBatch.vjp_audited"
            } else {
                "st_core.backward_topos_resonator"
            },
            self.config,
            &self.topos,
            self.gate_layout,
            self.encoder.is_some(),
            true,
            captured,
            cache.audit,
            Some(backward_audit),
            {
                #[cfg(feature = "wgpu")]
                {
                    wgpu_failure
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    None
                }
            },
        );
        Ok(ToposResonatorTensorBackward {
            grad_input,
            audit: backward_audit,
        })
    }
}

fn checked_layer_volume(rows: usize, cols: usize) -> PureResult<usize> {
    if rows == 0 || cols == 0 {
        return Err(TensorError::InvalidDimensions { rows, cols });
    }
    rows.checked_mul(cols)
        .ok_or(TensorError::InvalidDimensions { rows, cols })
}

fn validate_layer_topos(
    volume: usize,
    config: ToposResonatorConfig,
    topos: &OpenCartesianTopos,
) -> PureResult<()> {
    config.validate().map_err(topos_resonator_error)?;
    if volume > topos.max_volume() {
        return Err(TensorError::TensorVolumeExceeded {
            label: "topos_resonator_gate",
            volume,
            max_volume: topos.max_volume(),
        });
    }
    topos.ensure_loop_free(config.iterations())
}

impl Module for ToposResonator {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        Ok(self.step_resonance(input)?.output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        Ok(self.backward_resonance(input, grad_output)?.grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.gate)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.last_step.get_mut().take();
        visitor(&mut self.gate)
    }

    fn infuse_text(&mut self, text: &str) -> PureResult<()> {
        let Some(encoder) = self.encoder.as_ref() else {
            return Ok(());
        };
        self.gate.absorb_text(encoder, text)
    }

    fn attach_hypergrad(&mut self, curvature: f32, learning_rate: f32) -> PureResult<()> {
        let topos = self.topos_with_curvature(curvature)?;
        self.attach_open_topos(curvature, learning_rate, topos)
    }

    fn attach_hypergrad_with_topos(
        &mut self,
        curvature: f32,
        learning_rate: f32,
        topos: OpenCartesianTopos,
    ) -> PureResult<()> {
        self.attach_open_topos(curvature, learning_rate, topos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "wgpu")]
    use crate::execution::push_backend_policy;
    use st_core::backend::device_caps::DeviceCaps;
    use st_core::dynamics::topos_resonator::{
        apply_topos_resonator, audit_topos_resonator_backward, backward_topos_resonator,
        ToposResonatorBackwardAuditRequest, ToposResonatorBackwardRequest,
    };
    use std::sync::{Arc, Mutex};

    fn observer_lock() -> std::sync::MutexGuard<'static, ()> {
        crate::test_global_state_lock()
    }

    fn shared_layer(features: usize, max_values: usize) -> ToposResonator {
        ToposResonator::with_shared_gate(
            "shared_gate",
            features,
            ToposResonatorConfig::new(0.2, 5).unwrap(),
            OpenCartesianTopos::new(-1.0, 1e-6, 1.0, 16, max_values)
                .unwrap()
                .with_porosity(0.3)
                .unwrap(),
        )
        .unwrap()
    }

    fn sum_gate_rows(values: &[f32], features: usize) -> Vec<f32> {
        let mut sums = vec![0.0_f64; features];
        for row in values.chunks_exact(features) {
            for (sum, value) in sums.iter_mut().zip(row) {
                *sum += f64::from(*value);
            }
        }
        sums.into_iter().map(|sum| sum as f32).collect()
    }

    #[test]
    fn shared_gate_matches_expanded_vjps_across_batches_layouts_and_replays() {
        let _policy = crate::execution::push_backend_policy(
            crate::execution::BackendPolicy::from_device_caps(DeviceCaps::cpu()),
        );
        let mut layer = shared_layer(3, 12);
        *layer.parameter_mut().value_mut() = Tensor::from_vec(1, 3, vec![0.7, -0.8, 0.4]).unwrap();
        assert_eq!(layer.gate_layout(), ToposGateLayout::SharedRows);
        for rows in [0, 1, 4, 2] {
            for layout in [Layout::RowMajor, Layout::ColMajor] {
                layer.parameter_mut().zero_gradient();
                let input =
                    Tensor::from_fn(rows, 3, |r, c| (r * 3 + c) as f32 * 0.41 - 1.3).unwrap();
                let dy = Tensor::from_fn(rows, 3, |r, c| (r * 2 + c) as f32 * 0.09 - 0.25).unwrap();
                let expanded = layer.parameter().value().data().repeat(rows);
                let request = ToposResonatorRequest {
                    input: input.data(),
                    gate: &expanded,
                    rows,
                    features: 3,
                    config: layer.config(),
                    topos: layer.topos(),
                };
                let expected = apply_topos_resonator(request).unwrap();
                let backward_request = ToposResonatorBackwardRequest {
                    request,
                    grad_output: dy.data(),
                };
                let gradients = backward_topos_resonator(backward_request).unwrap();
                let audit = audit_topos_resonator_backward(ToposResonatorBackwardAuditRequest {
                    request: backward_request,
                    grad_input: &gradients.grad_input,
                    grad_gate: &gradients.grad_gate,
                })
                .unwrap();
                let shared = sum_gate_rows(&gradients.grad_gate, 3);
                assert_eq!(
                    layer
                        .forward(&input.to_layout(layout).unwrap())
                        .unwrap()
                        .data(),
                    expected.output
                );
                for replay in 1..=2 {
                    let dx = layer
                        .backward(&input, &dy.to_layout(layout).unwrap())
                        .unwrap();
                    assert_eq!(dx.data(), gradients.grad_input);
                    assert_eq!(layer.latest_backward_audit(), Some(audit));
                    let accumulated = layer.parameter().gradient().unwrap();
                    assert_eq!(accumulated.shape(), (1, 3));
                    for (actual, expected) in accumulated.data().iter().zip(&shared) {
                        assert!((actual - expected * replay as f32).abs() <= 1e-6);
                    }
                }
            }
        }
    }

    #[test]
    fn elementwise_gate_respects_logical_tensor_layout() {
        let _policy = crate::execution::push_backend_policy(
            crate::execution::BackendPolicy::from_device_caps(DeviceCaps::cpu()),
        );
        let input = Tensor::from_vec(2, 3, vec![0.3, 2.1, -1.2, 0.4, -0.8, 0.5]).unwrap();
        let gate = Tensor::from_vec(2, 3, vec![0.7, -0.8, 0.4, 0.6, 0.5, -0.7]).unwrap();
        let dy = Tensor::from_vec(2, 3, vec![0.5, -0.1, 0.4, 0.3, -0.7, 0.2]).unwrap();
        let mut layer = ToposResonator::new("gate", 2, 3).unwrap();
        *layer.parameter_mut().value_mut() = gate.to_layout(Layout::ColMajor).unwrap();
        let operator = ToposResonatorOperator::new(layer.config(), layer.topos().clone()).unwrap();
        let expected = operator.forward(input.data(), gate.data(), 2, 3).unwrap();
        let gradients = operator
            .backward(input.data(), gate.data(), dy.data(), 2, 3)
            .unwrap();
        assert_eq!(
            layer
                .forward(&input.to_layout(Layout::ColMajor).unwrap())
                .unwrap()
                .data(),
            expected.output
        );
        let dx = layer
            .backward(&input, &dy.to_layout(Layout::ColMajor).unwrap())
            .unwrap();
        assert_eq!(dx.data(), gradients.grad_input);
        assert_eq!(
            layer.parameter().gradient().unwrap().data(),
            gradients.grad_gate
        );
    }

    #[test]
    fn shared_gate_vjp_matches_finite_differences_without_batch_averaging() {
        let _policy = crate::execution::push_backend_policy(
            crate::execution::BackendPolicy::from_device_caps(DeviceCaps::cpu()),
        );
        let mut layer = shared_layer(2, 6);
        let gate = [0.7, -0.8];
        *layer.parameter_mut().value_mut() = Tensor::from_vec(1, 2, gate.to_vec()).unwrap();
        let input = Tensor::from_vec(3, 2, vec![-1.3, 0.4, 2.2, -0.6, 0.2, -2.0]).unwrap();
        let dy = Tensor::from_vec(3, 2, vec![0.3, -0.2, 0.4, 0.7, -0.1, 0.2]).unwrap();
        layer.forward(&input).unwrap();
        layer.backward(&input, &dy).unwrap();
        let operator = ToposResonatorOperator::new(layer.config(), layer.topos().clone()).unwrap();
        let objective = |g: &[f32]| {
            operator
                .forward(input.data(), &g.repeat(3), 3, 2)
                .unwrap()
                .output
                .iter()
                .zip(dy.data())
                .map(|(value, upstream)| f64::from(*value) * f64::from(*upstream))
                .sum::<f64>()
        };
        for col in 0..2 {
            let mut plus = gate;
            let mut minus = gate;
            plus[col] += 1e-3;
            minus[col] -= 1e-3;
            let numerical = (objective(&plus) - objective(&minus)) / 2e-3;
            let analytic = f64::from(layer.parameter().gradient().unwrap().data()[col]);
            assert!(
                (analytic - numerical).abs() < 3e-4,
                "{analytic} != {numerical}"
            );
        }
    }

    #[test]
    fn shared_gate_rejects_expanded_volume_and_invalidates_parameter_changes() {
        let mut layer = shared_layer(2, 6);
        let input = Tensor::from_fn(2, 2, |_, _| 1.0).unwrap();
        layer.forward(&input).unwrap();
        assert!(matches!(
            layer.forward(&Tensor::from_fn(4, 2, |_, _| 1.0).unwrap()),
            Err(TensorError::TensorVolumeExceeded { .. })
        ));
        assert!(layer
            .forward(&Tensor::from_fn(2, 3, |_, _| 1.0).unwrap())
            .is_err());
        layer.backward(&input, &input).unwrap();
        layer.parameter_mut().zero_gradient();
        assert!(matches!(
            layer.backward(&input, &input),
            Err(TensorError::InvalidValue {
                label: "topos_resonator_forward_cache"
            })
        ));
        *layer.parameter_mut().value_mut() = Tensor::from_fn(2, 2, |_, _| 1.0).unwrap();
        assert!(layer.forward(&input).is_err());
    }

    #[test]
    fn shared_gate_reduction_overflow_does_not_commit_a_gradient_or_audit() {
        let _policy = crate::execution::push_backend_policy(
            crate::execution::BackendPolicy::from_device_caps(DeviceCaps::cpu()),
        );
        let mut layer = ToposResonator::with_shared_gate(
            "gate",
            1,
            ToposResonatorConfig::new(0.0, 1).unwrap(),
            OpenCartesianTopos::new(-1.0, 1e-6, f32::MAX, 4, 2).unwrap(),
        )
        .unwrap();
        *layer.parameter_mut().value_mut() = Tensor::zeros(1, 1).unwrap();
        layer
            .parameter_mut()
            .accumulate_euclidean(&Tensor::zeros(1, 1).unwrap())
            .unwrap();
        let input = Tensor::from_vec(2, 1, vec![f32::MAX * 0.75; 2]).unwrap();
        layer.forward(&input).unwrap();
        assert!(layer
            .backward(&input, &Tensor::from_fn(2, 1, |_, _| 1.0).unwrap())
            .is_err());
        assert_eq!(layer.parameter().gradient().unwrap().data(), &[0.0]);
        assert!(layer.latest_backward_audit().is_none());
        layer
            .backward(&input, &Tensor::from_vec(2, 1, vec![0.25; 2]).unwrap())
            .unwrap();
        assert!(layer.parameter().gradient().unwrap().data()[0].is_finite());
        assert!(layer.latest_backward_audit().is_some());
    }

    #[test]
    fn shared_gate_sequential_learning_matches_recomputation_with_variable_rows() {
        let _policy = crate::execution::push_backend_policy(
            crate::execution::BackendPolicy::from_device_caps(DeviceCaps::cpu()),
        );
        let layer = shared_layer(2, 6);
        let operator = ToposResonatorOperator::new(layer.config(), layer.topos().clone()).unwrap();
        let mut model = crate::Sequential::new();
        model.push(crate::layers::Identity);
        model.push(layer);
        model.push(crate::layers::Identity);
        let mut gate = vec![1.0_f32; 2];
        for step in 0..100 {
            let rows = 1 + step % 3;
            let input = Tensor::from_fn(rows, 2, |r, c| 0.1 + (r * 2 + c) as f32 * 0.07).unwrap();
            let target = Tensor::from_fn(rows, 2, |r, c| -0.15 + (r + c) as f32 * 0.02).unwrap();
            let expanded = gate.repeat(rows);
            let expected = operator.forward(input.data(), &expanded, rows, 2).unwrap();
            let output = model.forward(&input).unwrap();
            assert_eq!(output.data(), expected.output);
            let dy = Tensor::from_vec(
                rows,
                2,
                output
                    .data()
                    .iter()
                    .zip(target.data())
                    .map(|(a, b)| 2.0 * (a - b) / (rows * 2) as f32)
                    .collect(),
            )
            .unwrap();
            let expected = operator
                .backward(input.data(), &expanded, dy.data(), rows, 2)
                .unwrap();
            assert_eq!(
                model.backward(&input, &dy).unwrap().data(),
                expected.grad_input
            );
            let dg = sum_gate_rows(&expected.grad_gate, 2);
            model
                .visit_parameters_mut(&mut |parameter| {
                    assert_eq!(parameter.value().shape(), (1, 2));
                    assert_eq!(parameter.gradient().unwrap().data(), dg);
                    for (g, d) in gate.iter_mut().zip(&dg) {
                        *g -= 0.03 * d;
                    }
                    *parameter.value_mut() = Tensor::from_vec(1, 2, gate.clone())?;
                    parameter.zero_gradient();
                    Ok(())
                })
                .unwrap();
        }
        assert!(gate.iter().all(|value| value.is_finite() && *value < 1.0));
    }

    #[test]
    fn shared_gate_metadata_separates_parameter_shape_and_expanded_audits() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            if event.op_name == "topos_resonator_backward" {
                captured.lock().unwrap().push(event.data.clone());
            }
        })));
        let mut layer = shared_layer(2, 6);
        let input = Tensor::from_fn(3, 2, |_, _| 1.0).unwrap();
        layer.forward(&input).unwrap();
        layer.backward(&input, &input).unwrap();
        st_tensor::set_thread_meta_observer(previous);
        let events = events.lock().unwrap();
        let event = &events[0];
        assert_eq!(event["rows"], 3);
        assert_eq!(event["gate_rows"], 1);
        assert_eq!(event["trainable_parameters"], 2);
        assert_eq!(event["gate_layout"], "shared_rows");
        assert_eq!(
            event["gate_gradient_reduction"],
            "sum_axis0_no_additional_mean"
        );
        assert_eq!(event["backward_audit_scope"], "expanded_elementwise_vjps");
    }

    #[test]
    fn cpu_capture_replays_exact_gradients_and_audits_without_cloning_the_tape() {
        #[cfg(feature = "wgpu")]
        let _policy =
            crate::execution::push_backend_policy(crate::test_backend_policy(DeviceCaps::cpu(), 1));
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let input = Tensor::from_vec(2, 3, vec![0.0, -0.0, 0.2, -0.4, 4.0, -3.0]).unwrap();
        let gate = Tensor::from_vec(2, 3, vec![1.0, -1.0, 0.3, -0.2, 0.7, 0.5]).unwrap();
        for (coupling, iterations) in [(0.0, 1), (0.35, 6), (0.9, 16)] {
            for porosity in [0.0, 0.2, 1.0] {
                let topos = OpenCartesianTopos::new(-1.0, 1e-6, 1.0, 32, 6)
                    .unwrap()
                    .with_porosity(porosity)
                    .unwrap();
                let config = ToposResonatorConfig::new(coupling, iterations).unwrap();
                let mut layer =
                    ToposResonator::with_config_and_topos("gate", 2, 3, config, topos.clone())
                        .unwrap();
                *layer.parameter_mut().value_mut() = gate.clone();
                let request = ToposResonatorRequest {
                    input: input.data(),
                    gate: gate.data(),
                    rows: 2,
                    features: 3,
                    config,
                    topos: &topos,
                };
                let expected = apply_topos_resonator(request).unwrap();
                let actual = layer.step_resonance(&input).unwrap();
                assert_eq!(bits(actual.output.data()), bits(&expected.output));
                assert_eq!(actual.audit, expected.audit);
                assert_eq!(
                    bits(layer.latest_output().unwrap().data()),
                    bits(&expected.output)
                );
                let saved = layer.last_step.borrow().as_ref().unwrap().saved.clone();
                #[allow(irrefutable_let_patterns)]
                let ToposResonatorSaved::Captured(tape) = saved
                else {
                    panic!("CPU forward must retain its sensitivity");
                };
                assert_eq!(Arc::strong_count(&tape), 2);
                let mut expected_gate = Vec::new();
                for (pass, scale) in [0.3, -0.2, 0.0].into_iter().enumerate() {
                    let dy = Tensor::from_vec(2, 3, vec![scale; 6]).unwrap();
                    let backward_request = ToposResonatorBackwardRequest {
                        request,
                        grad_output: dy.data(),
                    };
                    let expected = backward_topos_resonator(backward_request).unwrap();
                    let audit =
                        audit_topos_resonator_backward(ToposResonatorBackwardAuditRequest {
                            request: backward_request,
                            grad_input: &expected.grad_input,
                            grad_gate: &expected.grad_gate,
                        })
                        .unwrap();
                    let actual = layer.backward_resonance(&input, &dy).unwrap();
                    assert_eq!(bits(actual.grad_input.data()), bits(&expected.grad_input));
                    assert_eq!(actual.audit, audit);
                    assert_eq!(layer.latest_backward_audit(), Some(audit));
                    if pass == 0 {
                        expected_gate = expected.grad_gate.clone();
                    } else {
                        for (sum, value) in expected_gate.iter_mut().zip(&expected.grad_gate) {
                            *sum += value;
                        }
                    }
                    assert_eq!(
                        bits(layer.parameter().gradient().unwrap().data()),
                        bits(&expected_gate)
                    );
                    assert_eq!(Arc::strong_count(&tape), 2);
                }
            }
        }
    }

    #[test]
    fn captured_module_errors_leave_gradients_audit_and_prior_tape_intact() {
        #[cfg(feature = "wgpu")]
        let _policy =
            crate::execution::push_backend_policy(crate::test_backend_policy(DeviceCaps::cpu(), 1));
        let mut layer = ToposResonator::new("gate", 1, 2).unwrap();
        let input = Tensor::from_vec(1, 2, vec![0.1, 0.2]).unwrap();
        let dy = Tensor::from_vec(1, 2, vec![0.2, 0.3]).unwrap();
        layer.forward(&input).unwrap();
        layer.backward(&input, &dy).unwrap();
        let gradient = layer.parameter().gradient().unwrap().clone();
        let audit = layer.latest_backward_audit();
        let invalid = Tensor::from_vec(1, 2, vec![f32::MAX; 2]).unwrap();
        assert!(layer.backward(&input, &invalid).is_err());
        assert!(layer.backward(&dy, &dy).is_err());
        assert_eq!(
            layer.parameter().gradient().unwrap().data(),
            gradient.data()
        );
        assert_eq!(layer.latest_backward_audit(), audit);
        let bad_shape = Tensor::from_vec(2, 1, vec![0.1, 0.2]).unwrap();
        assert!(layer.forward(&bad_shape).is_err());
        assert_eq!(layer.latest_backward_audit(), audit);
        layer.backward(&input, &dy).unwrap();
        layer.parameter_mut().zero_gradient();
        assert!(layer.latest_output().is_none());
        assert!(layer.backward(&input, &dy).is_err());
        layer.forward(&input).unwrap();
        assert!(layer.latest_backward_audit().is_none());
        layer.backward(&input, &dy).unwrap();
    }

    #[test]
    fn captured_module_sgd_matches_recomputed_learning_for_one_hundred_updates() {
        #[cfg(feature = "wgpu")]
        let _policy =
            crate::execution::push_backend_policy(crate::test_backend_policy(DeviceCaps::cpu(), 1));
        let config = ToposResonatorConfig::new(0.35, 6).unwrap();
        let topos = OpenCartesianTopos::new(-1.0, 1e-6, 1.0, 16, 4)
            .unwrap()
            .with_porosity(0.2)
            .unwrap();
        let mut layer =
            ToposResonator::with_config_and_topos("gate", 2, 2, config, topos.clone()).unwrap();
        let input = Tensor::from_vec(2, 2, vec![0.2, -0.3, 0.4, -0.5]).unwrap();
        let target = [0.05, -0.075, 0.1, -0.125];
        let mut reference_gate = Parameter::new("reference", layer.parameter().value().clone());
        let mut losses = Vec::new();
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for _ in 0..100 {
            let request = ToposResonatorRequest {
                input: input.data(),
                gate: reference_gate.value().data(),
                rows: 2,
                features: 2,
                config,
                topos: &topos,
            };
            let expected = apply_topos_resonator(request).unwrap();
            let actual = layer.forward(&input).unwrap();
            assert_eq!(bits(actual.data()), bits(&expected.output));
            let residual: Vec<_> = actual
                .data()
                .iter()
                .zip(target)
                .map(|(x, y)| x - y)
                .collect();
            losses.push(residual.iter().map(|x| x * x).sum::<f32>() / 4.0);
            let dy = Tensor::from_vec(2, 2, residual.iter().map(|x| x * 0.5).collect()).unwrap();
            let expected = backward_topos_resonator(ToposResonatorBackwardRequest {
                request,
                grad_output: dy.data(),
            })
            .unwrap();
            let actual = layer.backward(&input, &dy).unwrap();
            assert_eq!(bits(actual.data()), bits(&expected.grad_input));
            assert_eq!(
                bits(layer.parameter().gradient().unwrap().data()),
                bits(&expected.grad_gate)
            );
            reference_gate
                .accumulate_euclidean(&Tensor::from_vec(2, 2, expected.grad_gate).unwrap())
                .unwrap();
            reference_gate.apply_step(0.2).unwrap();
            reference_gate.zero_gradient();
            layer.parameter_mut().apply_step(0.2).unwrap();
            layer.parameter_mut().zero_gradient();
            assert_eq!(
                bits(layer.parameter().value().data()),
                bits(reference_gate.value().data())
            );
        }
        assert!(losses[99] < losses[0] * 0.1);
    }

    #[test]
    fn default_resonator_has_a_bounded_nontrivial_response() {
        let resonator = ToposResonator::new("gate", 1, 2).unwrap();
        let input = Tensor::from_vec(1, 2, vec![1.0, -2.0]).unwrap();
        let out = resonator.forward(&input).unwrap();
        let expected_gain = 1.0 + 0.25 + 0.25f32.powi(2) + 0.25f32.powi(3);
        assert!((out.data()[0] - expected_gain).abs() < 1e-6);
        assert!((out.data()[1] + 2.0 * expected_gain).abs() < 1e-6);
        let audit = resonator.latest_audit().expect("forward audit");
        assert_eq!(audit.iterations, 4);
        assert_eq!(audit.coupling, 0.25);
    }

    #[test]
    fn backward_requires_and_matches_the_cached_forward() {
        let mut resonator = ToposResonator::new("gate", 1, 2).unwrap();
        let input = Tensor::from_vec(1, 2, vec![1.0, 2.0]).unwrap();
        let grad_output = Tensor::from_vec(1, 2, vec![0.5, 0.25]).unwrap();
        assert!(matches!(
            resonator.backward(&input, &grad_output),
            Err(TensorError::InvalidValue {
                label: "topos_resonator_forward_cache"
            })
        ));
        resonator.forward(&input).unwrap();
        let grad_input = resonator.backward(&input, &grad_output).unwrap();
        let sensitivity = 1.0 + 0.25 + 0.25f32.powi(2) + 0.25f32.powi(3);
        assert!((grad_input.data()[0] - 0.5 * sensitivity).abs() < 1e-6);
        assert!((grad_input.data()[1] - 0.25 * sensitivity).abs() < 1e-6);
        let gradient = resonator.parameter().gradient().expect("gate gradient");
        assert!((gradient.data()[0] - 0.5 * sensitivity).abs() < 1e-6);
        assert!((gradient.data()[1] - 0.5 * sensitivity).abs() < 1e-6);
        assert!(resonator.latest_backward_audit().is_some());
    }

    #[test]
    fn replacing_topos_controls_forward_and_optimizer_together() {
        let mut resonator = ToposResonator::new("gate", 1, 1).unwrap();
        let topos = OpenCartesianTopos::new(-0.7, 1e-6, 0.5, 16, 1)
            .unwrap()
            .with_porosity(0.0)
            .unwrap();
        resonator.attach_open_topos(-0.7, 0.01, topos).unwrap();
        let input = Tensor::from_vec(1, 1, vec![10.0]).unwrap();
        let output = resonator.forward(&input).unwrap();
        assert_eq!(output.data(), &[0.5]);
        assert_eq!(resonator.topos().curvature(), -0.7);
        assert_eq!(
            resonator
                .parameter()
                .hypergrad()
                .unwrap()
                .topos()
                .curvature(),
            -0.7
        );
    }

    #[test]
    fn mismatched_topos_curvature_is_transactional() {
        let mut resonator = ToposResonator::new("gate", 1, 1).unwrap();
        let original_curvature = resonator.topos().curvature();
        let topos = OpenCartesianTopos::new(-0.7, 1e-6, 1.0, 16, 1).unwrap();
        let error = resonator.attach_open_topos(-0.9, 0.01, topos).unwrap_err();
        assert!(matches!(error, TensorError::CurvatureMismatch { .. }));
        assert_eq!(resonator.topos().curvature(), original_curvature);
        assert!(resonator.parameter().hypergrad().is_none());
    }

    #[test]
    fn module_hypergrad_routes_keep_forward_and_optimizer_topoi_aligned() {
        let mut resonator = ToposResonator::new("gate", 1, 1).unwrap();
        let supplied = OpenCartesianTopos::new(-0.7, 1e-5, 0.75, 16, 1)
            .unwrap()
            .with_porosity(0.4)
            .unwrap();
        let trainer = crate::ModuleTrainer::new(DeviceCaps::cpu(), -0.7, 0.01, 0.01);
        trainer
            .prepare_with_topos(&mut resonator, supplied)
            .unwrap();
        assert_eq!(resonator.topos().curvature(), -0.7);
        assert_eq!(resonator.topos().saturation(), 0.75);
        assert_eq!(resonator.topos().porosity(), 0.4);
        assert_eq!(
            resonator
                .parameter()
                .hypergrad()
                .unwrap()
                .topos()
                .saturation(),
            resonator.topos().saturation()
        );

        <ToposResonator as Module>::attach_hypergrad(&mut resonator, -0.55, 0.02).unwrap();
        assert_eq!(resonator.topos().curvature(), -0.55);
        assert_eq!(resonator.topos().saturation(), 0.75);
        assert_eq!(
            resonator
                .parameter()
                .hypergrad()
                .unwrap()
                .topos()
                .curvature(),
            -0.55
        );
    }

    #[test]
    fn direct_parameter_escape_hatch_cannot_silently_split_topoi() {
        let mut resonator = ToposResonator::new("gate", 1, 1).unwrap();
        let optimizer_only = OpenCartesianTopos::new(-1.0, 1e-6, 0.25, 16, 1).unwrap();
        resonator
            .parameter_mut()
            .attach_hypergrad_with_topos(-1.0, 0.01, optimizer_only)
            .unwrap();
        let input = Tensor::from_vec(1, 1, vec![1.0]).unwrap();
        assert!(matches!(
            resonator.forward(&input),
            Err(TensorError::InvalidValue {
                label: "topos_resonator_optimizer_topos_mismatch"
            })
        ));
    }

    #[test]
    fn forward_backward_emit_versioned_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut resonator = ToposResonator::new("gate", 1, 2).unwrap();
        let input = Tensor::from_vec(1, 2, vec![1.0, 2.0]).unwrap();
        resonator.forward(&input).unwrap();
        let grad_output = Tensor::from_vec(1, 2, vec![0.5, 0.25]).unwrap();
        resonator.backward(&input, &grad_output).unwrap();
        st_tensor::set_thread_meta_observer(previous);

        let events = events.lock().unwrap();
        for (op_name, backward) in [
            ("topos_resonator_forward", false),
            ("topos_resonator_backward", true),
        ] {
            let event = events
                .iter()
                .find(|(name, data)| {
                    *name == op_name
                        && data["backend"] == "cpu"
                        && data["semantic_owner"] == TOPOS_RESONATOR_SEMANTIC_OWNER
                })
                .unwrap_or_else(|| panic!("{op_name} metadata event"));
            assert_eq!(
                event.1["contract_version"],
                TOPOS_RESONATOR_CONTRACT_VERSION
            );
            assert_eq!(event.1["requested_backend"], "auto");
            assert_eq!(event.1["coupling"], 0.25);
            assert_eq!(event.1["iterations"], 4);
            assert_eq!(event.1["backward"], backward);
            assert_eq!(event.1["captured_sensitivity"], true);
            assert_eq!(event.1["capture_reused"], backward);
            assert_eq!(
                event.1["estimated_ops_per_value"],
                if backward { 3 } else { 52 }
            );
            assert!(event.1["audit"].is_object());
        }
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn forced_wgpu_matches_cpu_contract_and_audits() {
        if !wgpu_dense::is_available() {
            eprintln!("Topos mixed-route test skipped: WGPU is unavailable");
            return;
        }
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let rows = 257;
        let cols = 5;
        let input = Tensor::from_fn(rows, cols, |row, col| {
            ((row * 13 + col * 7) % 29) as f32 * 0.071 - 0.95
        })
        .unwrap();
        let grad_output = Tensor::from_fn(rows, cols, |row, col| {
            ((row * 5 + col * 11) % 23) as f32 * 0.031 - 0.3
        })
        .unwrap();
        let config = ToposResonatorConfig::new(0.4, 7).unwrap();
        let make_layer = |name: &str| {
            let topos = OpenCartesianTopos::new(-0.8, 1e-6, 0.75, 16, rows * cols)
                .unwrap()
                .with_porosity(0.35)
                .unwrap();
            ToposResonator::with_config_and_topos(name, rows, cols, config, topos).unwrap()
        };

        let cpu_policy = crate::test_backend_policy(DeviceCaps::cpu(), 1);
        let mut cpu_layer = make_layer("cpu");
        let (cpu_output, cpu_grad_input) = {
            let _guard = push_backend_policy(cpu_policy);
            (
                cpu_layer.forward(&input).unwrap(),
                cpu_layer.backward(&input, &grad_output).unwrap(),
            )
        };

        let wgpu_policy = crate::test_backend_policy(DeviceCaps::wgpu(32, true, 256), 1);
        let mut wgpu_layer = make_layer("wgpu");
        let (wgpu_output, wgpu_grad_input) = {
            let _guard = push_backend_policy(wgpu_policy);
            (
                wgpu_layer.forward(&input).unwrap(),
                wgpu_layer.backward(&input, &grad_output).unwrap(),
            )
        };

        for (cpu, wgpu) in cpu_output.data().iter().zip(wgpu_output.data()) {
            assert!((cpu - wgpu).abs() < 1e-5);
        }
        for (cpu, wgpu) in cpu_grad_input.data().iter().zip(wgpu_grad_input.data()) {
            assert!((cpu - wgpu).abs() < 1e-5);
        }
        let cpu_grad_gate = cpu_layer.parameter().gradient().unwrap().clone();
        let wgpu_grad_gate = wgpu_layer.parameter().gradient().unwrap().clone();
        for (cpu, wgpu) in cpu_grad_gate.data().iter().zip(wgpu_grad_gate.data()) {
            assert!((cpu - wgpu).abs() < 1e-5);
        }
        assert!(wgpu_layer.latest_audit().unwrap().max_output_error <= 1e-5);
        assert!(
            wgpu_layer
                .latest_backward_audit()
                .unwrap()
                .max_grad_input_error
                <= 1e-5
        );
        assert!(
            wgpu_layer
                .latest_backward_audit()
                .unwrap()
                .max_grad_gate_error
                <= 1e-5
        );

        // Compare each executor before replay can mix their gradients or
        // overwrite its audit, then check cross-route accumulation separately.
        {
            let _guard = push_backend_policy(crate::test_backend_policy(DeviceCaps::cpu(), 1));
            let replay = wgpu_layer.backward(&input, &grad_output).unwrap();
            for (expected, actual) in cpu_grad_input.data().iter().zip(replay.data()) {
                assert_eq!(expected.to_bits(), actual.to_bits());
            }
        }
        {
            let _guard = push_backend_policy(crate::test_backend_policy(
                DeviceCaps::wgpu(32, true, 256),
                1,
            ));
            let replay = cpu_layer.backward(&input, &grad_output).unwrap();
            for (expected, actual) in wgpu_grad_input.data().iter().zip(replay.data()) {
                assert!((expected - actual).abs() < 1e-5);
            }
        }
        for layer in [&cpu_layer, &wgpu_layer] {
            for ((cpu, wgpu), accumulated) in cpu_grad_gate
                .data()
                .iter()
                .zip(wgpu_grad_gate.data())
                .zip(layer.parameter().gradient().unwrap().data())
            {
                assert!((cpu + wgpu - accumulated).abs() < 1e-5);
            }
        }
        st_tensor::set_thread_meta_observer(previous);
        let events = events.lock().unwrap();
        assert!(events.iter().any(|(name, data)| {
            *name == "topos_resonator_forward"
                && data["backend"] == "wgpu_dense"
                && data["requested_backend"] == "wgpu"
        }));
        assert!(events.iter().any(|(name, data)| {
            *name == "topos_resonator_backward"
                && data["backend"] == "wgpu_dense"
                && data["requested_backend"] == "wgpu"
        }));
        eprintln!("Topos mixed-route test executed CPU/WGPU forward and both backward routes");
    }
}
