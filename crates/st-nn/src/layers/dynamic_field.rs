// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::execution::current_tensor_util_backend_for_values;
use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use spiral_config::determinism;
use st_core::dynamics::klein_gordon::{
    apply_klein_gordon_step, audit_klein_gordon_backward, backward_klein_gordon_step,
    validate_klein_gordon_state, KleinGordonBackwardAuditRequest, KleinGordonBackwardRequest,
    KLEIN_GORDON_BACKWARD, KLEIN_GORDON_BOUNDARY, KLEIN_GORDON_CONTRACT_VERSION,
    KLEIN_GORDON_EQUATION, KLEIN_GORDON_INTEGRATOR, KLEIN_GORDON_SEMANTIC_BACKEND,
    KLEIN_GORDON_SEMANTIC_OWNER, KLEIN_GORDON_SOURCE_MODEL, KLEIN_GORDON_STATE,
};
#[cfg(feature = "wgpu")]
use st_core::dynamics::klein_gordon::{audit_klein_gordon_step, KleinGordonAuditRequest};
pub use st_core::dynamics::klein_gordon::{
    KleinGordonAudit, KleinGordonBackwardAudit, KleinGordonConfig,
};
use st_core::dynamics::stochastic_schrodinger::{
    apply_stochastic_schrodinger_step, audit_stochastic_schrodinger_backward,
    backward_stochastic_schrodinger_step, validate_stochastic_schrodinger_state,
    StochasticSchrodingerBackwardAuditRequest, STOCHASTIC_SCHRODINGER_CALCULUS,
    STOCHASTIC_SCHRODINGER_CONTRACT_VERSION, STOCHASTIC_SCHRODINGER_HAMILTONIAN,
    STOCHASTIC_SCHRODINGER_INTEGRATOR, STOCHASTIC_SCHRODINGER_NOISE_MODEL,
    STOCHASTIC_SCHRODINGER_OPEN_SYSTEM_MODE, STOCHASTIC_SCHRODINGER_OUTPUT_OBSERVABLE,
    STOCHASTIC_SCHRODINGER_SEMANTIC_BACKEND, STOCHASTIC_SCHRODINGER_SEMANTIC_OWNER,
};
#[cfg(feature = "wgpu")]
use st_core::dynamics::stochastic_schrodinger::{
    audit_stochastic_schrodinger_step, StochasticSchrodingerAuditRequest,
};
pub use st_core::dynamics::stochastic_schrodinger::{
    StochasticSchrodingerAudit, StochasticSchrodingerBackwardAudit, StochasticSchrodingerConfig,
};
#[cfg(feature = "wgpu")]
use st_tensor::wgpu_dense;
use st_tensor::{emit_tensor_op, emit_tensor_op_meta, TensorUtilBackend};
use std::cell::RefCell;

fn tensor_util_backend_label(backend: TensorUtilBackend) -> &'static str {
    match backend {
        TensorUtilBackend::Auto => "auto",
        TensorUtilBackend::Cpu => "cpu",
        TensorUtilBackend::GpuWgpu => "wgpu",
    }
}

#[cfg(feature = "wgpu")]
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
    crate::execution::current_accelerator_fallback().is_strict()
}

#[cfg(feature = "wgpu")]
fn dynamic_field_wgpu_error(op_name: &'static str, message: String) -> TensorError {
    TensorError::BackendFailure {
        backend: "wgpu",
        message: format!("{op_name} wgpu path failed ({message}); fallback disabled"),
    }
}

#[derive(Clone, Copy)]
struct StochasticSchrodingerRouteMeta {
    config: StochasticSchrodingerConfig,
    audit: Option<StochasticSchrodingerAudit>,
    backward_audit: Option<StochasticSchrodingerBackwardAudit>,
}

#[derive(Clone, Copy)]
struct KleinGordonRouteMeta {
    config: KleinGordonConfig,
    audit: Option<KleinGordonAudit>,
    backward_audit: Option<KleinGordonBackwardAudit>,
}

#[derive(Clone, Copy)]
enum DynamicFieldContractMeta {
    KleinGordon(KleinGordonRouteMeta),
    StochasticSchrodinger(StochasticSchrodingerRouteMeta),
}

#[allow(clippy::too_many_arguments)]
fn emit_dynamic_field_route_meta(
    op_name: &'static str,
    field_model: &'static str,
    rows: usize,
    cols: usize,
    trainable_parameters: usize,
    estimated_ops_per_value: usize,
    backward: bool,
    gradient_scale: Option<f32>,
    backend: &'static str,
    requested_backend: &'static str,
    kernel: &'static str,
    input_gradient_backend: Option<&'static str>,
    gradient_reduction_backend: Option<&'static str>,
    gradient_scale_backend: Option<&'static str>,
    fallback: Option<String>,
) {
    emit_dynamic_field_route_meta_inner(
        op_name,
        field_model,
        rows,
        cols,
        trainable_parameters,
        estimated_ops_per_value,
        backward,
        gradient_scale,
        backend,
        requested_backend,
        kernel,
        input_gradient_backend,
        gradient_reduction_backend,
        gradient_scale_backend,
        fallback,
        None,
    );
}

#[allow(clippy::too_many_arguments)]
fn emit_stochastic_schrodinger_route_meta(
    op_name: &'static str,
    rows: usize,
    cols: usize,
    estimated_ops_per_value: usize,
    backward: bool,
    gradient_scale: Option<f32>,
    backend: &'static str,
    requested_backend: &'static str,
    kernel: &'static str,
    input_gradient_backend: Option<&'static str>,
    gradient_reduction_backend: Option<&'static str>,
    gradient_scale_backend: Option<&'static str>,
    fallback: Option<String>,
    config: StochasticSchrodingerConfig,
    audit: Option<StochasticSchrodingerAudit>,
    backward_audit: Option<StochasticSchrodingerBackwardAudit>,
) {
    emit_dynamic_field_route_meta_inner(
        op_name,
        "stochastic_schrodinger",
        rows,
        cols,
        1,
        estimated_ops_per_value,
        backward,
        gradient_scale,
        backend,
        requested_backend,
        kernel,
        input_gradient_backend,
        gradient_reduction_backend,
        gradient_scale_backend,
        fallback,
        Some(DynamicFieldContractMeta::StochasticSchrodinger(
            StochasticSchrodingerRouteMeta {
                config,
                audit,
                backward_audit,
            },
        )),
    );
}

#[allow(clippy::too_many_arguments)]
fn emit_klein_gordon_route_meta(
    op_name: &'static str,
    rows: usize,
    cols: usize,
    estimated_ops_per_value: usize,
    backward: bool,
    gradient_scale: Option<f32>,
    backend: &'static str,
    requested_backend: &'static str,
    kernel: &'static str,
    input_gradient_backend: Option<&'static str>,
    gradient_reduction_backend: Option<&'static str>,
    gradient_scale_backend: Option<&'static str>,
    fallback: Option<String>,
    config: KleinGordonConfig,
    audit: Option<KleinGordonAudit>,
    backward_audit: Option<KleinGordonBackwardAudit>,
) {
    emit_dynamic_field_route_meta_inner(
        op_name,
        "klein_gordon",
        rows,
        cols,
        2,
        estimated_ops_per_value,
        backward,
        gradient_scale,
        backend,
        requested_backend,
        kernel,
        input_gradient_backend,
        gradient_reduction_backend,
        gradient_scale_backend,
        fallback,
        Some(DynamicFieldContractMeta::KleinGordon(
            KleinGordonRouteMeta {
                config,
                audit,
                backward_audit,
            },
        )),
    );
}

#[allow(clippy::too_many_arguments)]
fn emit_dynamic_field_route_meta_inner(
    op_name: &'static str,
    field_model: &'static str,
    rows: usize,
    cols: usize,
    trainable_parameters: usize,
    estimated_ops_per_value: usize,
    backward: bool,
    gradient_scale: Option<f32>,
    backend: &'static str,
    requested_backend: &'static str,
    kernel: &'static str,
    input_gradient_backend: Option<&'static str>,
    gradient_reduction_backend: Option<&'static str>,
    gradient_scale_backend: Option<&'static str>,
    fallback: Option<String>,
    contract: Option<DynamicFieldContractMeta>,
) {
    emit_tensor_op(op_name, &[rows, cols], &[rows, cols]);
    emit_tensor_op_meta(op_name, || {
        let values = rows.saturating_mul(cols);
        let deterministic_backend = if field_model == "stochastic_schrodinger" && !backward {
            Some(if backend == "wgpu_dense" {
                "wgpu"
            } else {
                "cpu"
            })
        } else {
            None
        };
        let rng_backend = if field_model == "stochastic_schrodinger" && !backward {
            Some("cpu")
        } else {
            None
        };
        let mut data = serde_json::json!({
            "backend": backend,
            "requested_backend": requested_backend,
            "kernel": kernel,
            "deterministic_backend": deterministic_backend,
            "rng_backend": rng_backend,
            "kind": if backward { "dynamic_field_backward" } else { "dynamic_field_forward" },
            "backward": backward,
            "field_model": field_model,
            "rows": rows,
            "cols": cols,
            "values": values,
            "output_rows": rows,
            "output_cols": cols,
            "output_values": values,
            "trainable_parameters": trainable_parameters,
            "gradient_scale": gradient_scale,
            "parameter_gradient_scale": gradient_scale,
            "input_gradient_backend": input_gradient_backend,
            "gradient_reduction_backend": gradient_reduction_backend,
            "gradient_scale_backend": gradient_scale_backend,
            "input_gradient_scale": if backward && gradient_scale.is_some() {
                Some(1.0f32)
            } else {
                None
            },
            "estimated_ops_per_value": estimated_ops_per_value,
            "estimated_total_ops": values.saturating_mul(estimated_ops_per_value),
            "empty": rows == 0 || cols == 0,
        });
        if let Some(message) = fallback {
            if let Some(map) = data.as_object_mut() {
                map.insert(
                    "fallback".to_string(),
                    serde_json::json!({"from": "wgpu", "message": message}),
                );
            }
        }
        if let (Some(map), Some(contract)) = (data.as_object_mut(), contract) {
            match contract {
                DynamicFieldContractMeta::KleinGordon(klein_gordon) => {
                    let config = klein_gordon.config;
                    map.insert(
                        "contract_version".to_string(),
                        serde_json::json!(KLEIN_GORDON_CONTRACT_VERSION),
                    );
                    map.insert(
                        "semantic_owner".to_string(),
                        serde_json::json!(KLEIN_GORDON_SEMANTIC_OWNER),
                    );
                    map.insert(
                        "semantic_backend".to_string(),
                        serde_json::json!(KLEIN_GORDON_SEMANTIC_BACKEND),
                    );
                    map.insert(
                        "equation".to_string(),
                        serde_json::json!(KLEIN_GORDON_EQUATION),
                    );
                    map.insert(
                        "integrator".to_string(),
                        serde_json::json!(KLEIN_GORDON_INTEGRATOR),
                    );
                    map.insert(
                        "boundary".to_string(),
                        serde_json::json!(KLEIN_GORDON_BOUNDARY),
                    );
                    map.insert(
                        "phase_space_state".to_string(),
                        serde_json::json!(KLEIN_GORDON_STATE),
                    );
                    map.insert(
                        "source_model".to_string(),
                        serde_json::json!(KLEIN_GORDON_SOURCE_MODEL),
                    );
                    map.insert(
                        "backward_contract".to_string(),
                        serde_json::json!(KLEIN_GORDON_BACKWARD),
                    );
                    map.insert(
                        "time_step".to_string(),
                        serde_json::json!(config.time_step()),
                    );
                    map.insert("damping".to_string(), serde_json::json!(config.damping()));
                    map.insert(
                        "wave_speed".to_string(),
                        serde_json::json!(config.wave_speed()),
                    );
                    map.insert(
                        "lattice_spacing".to_string(),
                        serde_json::json!(config.lattice_spacing()),
                    );
                    map.insert(
                        "self_coupling".to_string(),
                        serde_json::json!(config.self_coupling()),
                    );
                    map.insert(
                        "phase_space_output_values".to_string(),
                        serde_json::json!(values.saturating_mul(2)),
                    );
                    map.insert("audit".to_string(), serde_json::json!(klein_gordon.audit));
                    map.insert(
                        "backward_audit".to_string(),
                        serde_json::json!(klein_gordon.backward_audit),
                    );
                }
                DynamicFieldContractMeta::StochasticSchrodinger(stochastic) => {
                    let config = stochastic.config;
                    map.insert(
                        "contract_version".to_string(),
                        serde_json::json!(STOCHASTIC_SCHRODINGER_CONTRACT_VERSION),
                    );
                    map.insert(
                        "semantic_owner".to_string(),
                        serde_json::json!(STOCHASTIC_SCHRODINGER_SEMANTIC_OWNER),
                    );
                    map.insert(
                        "semantic_backend".to_string(),
                        serde_json::json!(STOCHASTIC_SCHRODINGER_SEMANTIC_BACKEND),
                    );
                    map.insert(
                        "integrator".to_string(),
                        serde_json::json!(STOCHASTIC_SCHRODINGER_INTEGRATOR),
                    );
                    map.insert(
                        "stochastic_calculus".to_string(),
                        serde_json::json!(STOCHASTIC_SCHRODINGER_CALCULUS),
                    );
                    map.insert(
                        "hamiltonian".to_string(),
                        serde_json::json!(STOCHASTIC_SCHRODINGER_HAMILTONIAN),
                    );
                    map.insert(
                        "noise_model".to_string(),
                        serde_json::json!(STOCHASTIC_SCHRODINGER_NOISE_MODEL),
                    );
                    map.insert(
                        "open_system_mode".to_string(),
                        serde_json::json!(STOCHASTIC_SCHRODINGER_OPEN_SYSTEM_MODE),
                    );
                    map.insert(
                        "output_observable".to_string(),
                        serde_json::json!(STOCHASTIC_SCHRODINGER_OUTPUT_OBSERVABLE),
                    );
                    map.insert(
                        "time_step".to_string(),
                        serde_json::json!(config.time_step()),
                    );
                    map.insert(
                        "hopping_rate".to_string(),
                        serde_json::json!(config.hopping_rate()),
                    );
                    map.insert(
                        "loss_rate".to_string(),
                        serde_json::json!(config.loss_rate()),
                    );
                    map.insert(
                        "decoherence_rate".to_string(),
                        serde_json::json!(config.loss_rate()),
                    );
                    map.insert(
                        "noise_scale".to_string(),
                        serde_json::json!(config.noise_scale()),
                    );
                    map.insert("audit".to_string(), serde_json::json!(stochastic.audit));
                    map.insert(
                        "backward_audit".to_string(),
                        serde_json::json!(stochastic.backward_audit),
                    );
                }
            }
        }
        data
    });
}

fn klein_gordon_error(error: impl std::fmt::Display) -> TensorError {
    TensorError::Generic(format!("Klein-Gordon contract failed: {error}"))
}

#[derive(Clone, Debug)]
struct KleinGordonStepCache {
    input_field: Tensor,
    input_momentum: Tensor,
    mass_squared: Tensor,
    source: Tensor,
    output_field: Tensor,
    output_momentum: Tensor,
    audit: KleinGordonAudit,
    backward_audit: Option<KleinGordonBackwardAudit>,
    grad_input_momentum: Option<Tensor>,
}

/// One canonical Klein-Gordon phase-space transition exposed as tensors.
#[derive(Clone, Debug)]
pub struct KleinGordonTensorStep {
    pub field: Tensor,
    pub momentum: Tensor,
    pub audit: KleinGordonAudit,
}

/// Discrete-adjoint gradients for both components of the phase-space input.
#[derive(Clone, Debug)]
pub struct KleinGordonTensorBackward {
    pub grad_field: Tensor,
    pub grad_momentum: Tensor,
    pub audit: KleinGordonBackwardAudit,
}

/// Audited damped Klein-Gordon propagation on a periodic feature lattice.
///
/// Rust core owns the second-order field equation, exact damping split,
/// velocity-Verlet integration, stability gate, energy observation, and
/// analytic discrete adjoint. The ordinary [`Module`] path starts each sample
/// with zero momentum. [`KleinGordonPropagation::step_phase_space`] exposes the
/// complete `(field, momentum)` transition for explicit recurrent evolution.
#[derive(Debug)]
pub struct KleinGordonPropagation {
    // Compatibility keys preserve existing state dictionaries. Canonically
    // these are signed mass-squared and a static scalar-condensate source.
    mass: Parameter,
    spinor: Parameter,
    config: KleinGordonConfig,
    last_step: RefCell<Option<KleinGordonStepCache>>,
}

impl KleinGordonPropagation {
    /// Creates a propagation layer with deterministic initial parameters.
    pub fn new(
        name: impl Into<String>,
        features: usize,
        time_step: f32,
        damping: f32,
    ) -> PureResult<Self> {
        let config = KleinGordonConfig::new(time_step, damping).map_err(klein_gordon_error)?;
        Self::with_config(name, features, config)
    }

    /// Creates a layer from the versioned Rust-core phase-space contract.
    pub fn with_config(
        name: impl Into<String>,
        features: usize,
        config: KleinGordonConfig,
    ) -> PureResult<Self> {
        if features == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: 1,
                cols: features,
            });
        }
        config.validate().map_err(klein_gordon_error)?;
        let name = name.into();
        let mass_squared = Tensor::from_fn(1, features, |_r, c| 0.05 + c as f32 * 0.01)?;
        let source = Tensor::from_fn(1, features, |_r, c| ((c as f32 + 1.0) * 0.37).sin() * 0.1)?;
        Ok(Self {
            mass: Parameter::new(format!("{name}::mass"), mass_squared),
            spinor: Parameter::new(format!("{name}::spinor"), source),
            config,
            last_step: RefCell::new(None),
        })
    }

    pub fn with_wave_speed(mut self, wave_speed: f32) -> PureResult<Self> {
        self.config = self
            .config
            .with_wave_speed(wave_speed)
            .map_err(klein_gordon_error)?;
        Ok(self)
    }

    pub fn with_lattice_spacing(mut self, lattice_spacing: f32) -> PureResult<Self> {
        self.config = self
            .config
            .with_lattice_spacing(lattice_spacing)
            .map_err(klein_gordon_error)?;
        Ok(self)
    }

    pub fn with_self_coupling(mut self, self_coupling: f32) -> PureResult<Self> {
        self.config = self
            .config
            .with_self_coupling(self_coupling)
            .map_err(klein_gordon_error)?;
        Ok(self)
    }

    pub fn config(&self) -> KleinGordonConfig {
        self.config
    }

    pub fn time_step(&self) -> f32 {
        self.config.time_step()
    }

    pub fn damping(&self) -> f32 {
        self.config.damping()
    }

    pub fn wave_speed(&self) -> f32 {
        self.config.wave_speed()
    }

    pub fn lattice_spacing(&self) -> f32 {
        self.config.lattice_spacing()
    }

    pub fn self_coupling(&self) -> f32 {
        self.config.self_coupling()
    }

    /// Canonical accessor for the learned signed mass-squared coefficient.
    pub fn mass_squared(&self) -> &Parameter {
        &self.mass
    }

    /// Compatibility alias for [`KleinGordonPropagation::mass_squared`].
    pub fn mass(&self) -> &Parameter {
        self.mass_squared()
    }

    /// Canonical accessor for the learned static scalar-condensate source.
    pub fn source(&self) -> &Parameter {
        &self.spinor
    }

    /// Compatibility alias for [`KleinGordonPropagation::source`].
    pub fn spinor(&self) -> &Parameter {
        self.source()
    }

    pub fn latest_momentum(&self) -> Option<Tensor> {
        self.last_step
            .borrow()
            .as_ref()
            .map(|step| step.output_momentum.clone())
    }

    pub fn latest_audit(&self) -> Option<KleinGordonAudit> {
        self.last_step.borrow().as_ref().map(|step| step.audit)
    }

    pub fn latest_backward_audit(&self) -> Option<KleinGordonBackwardAudit> {
        self.last_step
            .borrow()
            .as_ref()
            .and_then(|step| step.backward_audit)
    }

    pub fn latest_input_momentum_gradient(&self) -> Option<Tensor> {
        self.last_step
            .borrow()
            .as_ref()
            .and_then(|step| step.grad_input_momentum.clone())
    }

    fn commit_step(
        &self,
        input_field: &Tensor,
        input_momentum: &Tensor,
        output_field: Vec<f32>,
        output_momentum: Vec<f32>,
        audit: KleinGordonAudit,
    ) -> PureResult<KleinGordonTensorStep> {
        let (rows, cols) = input_field.shape();
        let output_field = Tensor::from_vec(rows, cols, output_field)?;
        let output_momentum = Tensor::from_vec(rows, cols, output_momentum)?;
        self.last_step.borrow_mut().replace(KleinGordonStepCache {
            input_field: input_field.clone(),
            input_momentum: input_momentum.clone(),
            mass_squared: self.mass.value().clone(),
            source: self.spinor.value().clone(),
            output_field: output_field.clone(),
            output_momentum: output_momentum.clone(),
            audit,
            backward_audit: None,
            grad_input_momentum: None,
        });
        Ok(KleinGordonTensorStep {
            field: output_field,
            momentum: output_momentum,
            audit,
        })
    }

    fn commit_backward(&self, audit: KleinGordonBackwardAudit, grad_input_momentum: &Tensor) {
        if let Some(step) = self.last_step.borrow_mut().as_mut() {
            step.backward_audit = Some(audit);
            step.grad_input_momentum = Some(grad_input_momentum.clone());
        }
    }

    /// Advances an explicit canonical `(field, momentum)` state by one step.
    pub fn step_phase_space(
        &self,
        field: &Tensor,
        momentum: &Tensor,
    ) -> PureResult<KleinGordonTensorStep> {
        let (rows, cols) = field.shape();
        if momentum.shape() != field.shape() {
            return Err(TensorError::ShapeMismatch {
                left: momentum.shape(),
                right: field.shape(),
            });
        }
        if cols != self.mass.value().shape().1 {
            return Err(TensorError::ShapeMismatch {
                left: field.shape(),
                right: self.mass.value().shape(),
            });
        }
        let volume = rows.checked_mul(cols).ok_or_else(|| {
            TensorError::Generic("Klein-Gordon tensor volume overflow".to_string())
        })?;
        let mass_squared = self.mass.value().data();
        let source = self.spinor.value().data();
        validate_klein_gordon_state(
            field.data(),
            momentum.data(),
            mass_squared,
            source,
            rows,
            cols,
            self.config,
        )
        .map_err(klein_gordon_error)?;
        let route_backend = current_tensor_util_backend_for_values(volume);
        let requested_backend = tensor_util_backend_label(route_backend);
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(route_backend, TensorUtilBackend::GpuWgpu)
                && rows > 0
                && cols > 0
                && wgpu_dense::is_available()
            {
                let wgpu_step = wgpu_dense::dynamic_klein_gordon_forward(
                    field.data(),
                    momentum.data(),
                    mass_squared,
                    source,
                    rows,
                    cols,
                    self.config.time_step(),
                    self.config.damping_half_factor(),
                    self.config.laplacian_scale(),
                    self.config.self_coupling(),
                )
                .and_then(|(output_field, output_momentum)| {
                    audit_klein_gordon_step(KleinGordonAuditRequest {
                        field: field.data(),
                        momentum: momentum.data(),
                        mass_squared,
                        source,
                        output_field: &output_field,
                        output_momentum: &output_momentum,
                        rows,
                        features: cols,
                        config: self.config,
                    })
                    .map(|audit| (output_field, output_momentum, audit))
                    .map_err(|error| format!("Rust semantic audit failed: {error}"))
                });
                match wgpu_step {
                    Ok((output_field, output_momentum, audit)) => {
                        validate_finite_slice("dynamic_klein_gordon_forward_field", &output_field)?;
                        validate_finite_slice(
                            "dynamic_klein_gordon_forward_momentum",
                            &output_momentum,
                        )?;
                        let step = self.commit_step(
                            field,
                            momentum,
                            output_field,
                            output_momentum,
                            audit,
                        )?;
                        emit_klein_gordon_route_meta(
                            "dynamic_field_klein_gordon_forward",
                            rows,
                            cols,
                            52,
                            false,
                            None,
                            "wgpu_dense",
                            requested_backend,
                            "tensor_util.dynamic_klein_gordon_forward",
                            None,
                            None,
                            None,
                            None,
                            self.config,
                            Some(audit),
                            None,
                        );
                        return Ok(step);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(dynamic_field_wgpu_error(
                            "dynamic_field_klein_gordon_forward",
                            message,
                        ));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let step = apply_klein_gordon_step(
            field.data(),
            momentum.data(),
            mass_squared,
            source,
            rows,
            cols,
            self.config,
        )
        .map_err(klein_gordon_error)?;
        let audit = step.audit;
        let tensor_step =
            self.commit_step(field, momentum, step.field, step.momentum, step.audit)?;
        emit_klein_gordon_route_meta(
            "dynamic_field_klein_gordon_forward",
            rows,
            cols,
            52,
            false,
            None,
            "cpu",
            requested_backend,
            "st_core.apply_klein_gordon_step",
            None,
            None,
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
            self.config,
            Some(audit),
            None,
        );
        Ok(tensor_step)
    }

    /// Runs the complete discrete adjoint for a cached phase-space transition.
    pub fn backward_phase_space(
        &mut self,
        field: &Tensor,
        momentum: &Tensor,
        grad_output_field: &Tensor,
        grad_output_momentum: &Tensor,
    ) -> PureResult<KleinGordonTensorBackward> {
        let cache = self
            .last_step
            .borrow()
            .as_ref()
            .cloned()
            .ok_or(TensorError::InvalidValue {
                label: "klein_gordon_forward_cache",
            })?;
        if field.shape() != cache.input_field.shape() {
            return Err(TensorError::ShapeMismatch {
                left: field.shape(),
                right: cache.input_field.shape(),
            });
        }
        if momentum.shape() != cache.input_momentum.shape() {
            return Err(TensorError::ShapeMismatch {
                left: momentum.shape(),
                right: cache.input_momentum.shape(),
            });
        }
        if grad_output_field.shape() != cache.output_field.shape() {
            return Err(TensorError::ShapeMismatch {
                left: grad_output_field.shape(),
                right: cache.output_field.shape(),
            });
        }
        if grad_output_momentum.shape() != cache.output_momentum.shape() {
            return Err(TensorError::ShapeMismatch {
                left: grad_output_momentum.shape(),
                right: cache.output_momentum.shape(),
            });
        }
        if field.data() != cache.input_field.data() {
            return Err(TensorError::InvalidValue {
                label: "klein_gordon_forward_field_mismatch",
            });
        }
        if momentum.data() != cache.input_momentum.data() {
            return Err(TensorError::InvalidValue {
                label: "klein_gordon_forward_momentum_mismatch",
            });
        }
        let (rows, cols) = field.shape();
        let route_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        let requested_backend = tensor_util_backend_label(route_backend);
        let scale_backend = current_tensor_util_backend_for_values(cols);
        let scale_backend_label = tensor_util_backend_label(scale_backend);
        // The discrete adjoint belongs to the exact parameters used by the
        // cached forward, even if an external caller mutated live parameters.
        let mass_squared = cache.mass_squared.data();
        let source = cache.source.data();
        let core_request = KleinGordonBackwardRequest {
            field: field.data(),
            momentum: momentum.data(),
            mass_squared,
            source,
            grad_output_field: grad_output_field.data(),
            grad_output_momentum: grad_output_momentum.data(),
            rows,
            features: cols,
            config: self.config,
        };
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(route_backend, TensorUtilBackend::GpuWgpu)
                && rows > 0
                && cols > 0
                && wgpu_dense::is_available()
            {
                let wgpu_backward = wgpu_dense::dynamic_klein_gordon_backward(
                    field.data(),
                    momentum.data(),
                    cache.output_field.data(),
                    grad_output_field.data(),
                    grad_output_momentum.data(),
                    mass_squared,
                    source,
                    rows,
                    cols,
                    self.config.time_step(),
                    self.config.damping_half_factor(),
                    self.config.laplacian_scale(),
                    self.config.self_coupling(),
                )
                .and_then(
                    |(grad_field, grad_momentum, grad_mass_squared, grad_source)| {
                        audit_klein_gordon_backward(KleinGordonBackwardAuditRequest {
                            request: core_request,
                            grad_field: &grad_field,
                            grad_momentum: &grad_momentum,
                            grad_mass_squared: &grad_mass_squared,
                            grad_source: &grad_source,
                        })
                        .map(|audit| {
                            (
                                grad_field,
                                grad_momentum,
                                grad_mass_squared,
                                grad_source,
                                audit,
                            )
                        })
                        .map_err(|error| format!("Rust semantic backward audit failed: {error}"))
                    },
                );
                match wgpu_backward {
                    Ok((grad_field, grad_momentum, grad_mass, grad_source, backward_audit)) => {
                        validate_finite_slice("dynamic_klein_gordon_grad_field", &grad_field)?;
                        validate_finite_slice(
                            "dynamic_klein_gordon_grad_momentum",
                            &grad_momentum,
                        )?;
                        validate_finite_slice("dynamic_klein_gordon_grad_mass", &grad_mass)?;
                        validate_finite_slice("dynamic_klein_gordon_grad_source", &grad_source)?;
                        let grad_field = Tensor::from_vec(rows, cols, grad_field)?;
                        let grad_momentum = Tensor::from_vec(rows, cols, grad_momentum)?;
                        let grad_mass = Tensor::from_vec(1, cols, grad_mass)?;
                        let grad_source = Tensor::from_vec(1, cols, grad_source)?;
                        let gradient_scale = if rows > 0 {
                            let scale = 1.0 / rows as f32;
                            self.mass.accumulate_euclidean(
                                &grad_mass.scale_with_backend(scale, scale_backend)?,
                            )?;
                            self.spinor.accumulate_euclidean(
                                &grad_source.scale_with_backend(scale, scale_backend)?,
                            )?;
                            Some(scale)
                        } else {
                            None
                        };
                        self.commit_backward(backward_audit, &grad_momentum);
                        emit_klein_gordon_route_meta(
                            "dynamic_field_klein_gordon_backward",
                            rows,
                            cols,
                            88,
                            true,
                            gradient_scale,
                            "wgpu_dense",
                            requested_backend,
                            "tensor_util.dynamic_klein_gordon_backward",
                            Some("wgpu"),
                            Some("wgpu"),
                            Some(scale_backend_label),
                            None,
                            self.config,
                            Some(cache.audit),
                            Some(backward_audit),
                        );
                        return Ok(KleinGordonTensorBackward {
                            grad_field,
                            grad_momentum,
                            audit: backward_audit,
                        });
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(dynamic_field_wgpu_error(
                            "dynamic_field_klein_gordon_backward",
                            message,
                        ));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let backward = backward_klein_gordon_step(core_request).map_err(klein_gordon_error)?;
        let backward_audit = audit_klein_gordon_backward(KleinGordonBackwardAuditRequest {
            request: core_request,
            grad_field: &backward.grad_field,
            grad_momentum: &backward.grad_momentum,
            grad_mass_squared: &backward.grad_mass_squared,
            grad_source: &backward.grad_source,
        })
        .map_err(klein_gordon_error)?;
        let grad_field = Tensor::from_vec(rows, cols, backward.grad_field)?;
        let grad_momentum = Tensor::from_vec(rows, cols, backward.grad_momentum)?;
        let grad_mass = Tensor::from_vec(1, cols, backward.grad_mass_squared)?;
        let grad_source = Tensor::from_vec(1, cols, backward.grad_source)?;
        let gradient_scale = if rows > 0 {
            let scale = 1.0 / rows as f32;
            self.mass
                .accumulate_euclidean(&grad_mass.scale_with_backend(scale, scale_backend)?)?;
            self.spinor
                .accumulate_euclidean(&grad_source.scale_with_backend(scale, scale_backend)?)?;
            Some(scale)
        } else {
            None
        };
        self.commit_backward(backward_audit, &grad_momentum);
        emit_klein_gordon_route_meta(
            "dynamic_field_klein_gordon_backward",
            rows,
            cols,
            88,
            true,
            gradient_scale,
            "cpu",
            requested_backend,
            "st_core.backward_klein_gordon_step",
            Some("cpu"),
            Some("cpu"),
            if gradient_scale.is_some() {
                Some(scale_backend_label)
            } else {
                None
            },
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
            self.config,
            Some(cache.audit),
            Some(backward_audit),
        );
        Ok(KleinGordonTensorBackward {
            grad_field,
            grad_momentum,
            audit: backward_audit,
        })
    }
}

impl Module for KleinGordonPropagation {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        let zero_momentum = Tensor::zeros(rows, cols)?;
        Ok(self.step_phase_space(input, &zero_momentum)?.field)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let momentum = self
            .last_step
            .borrow()
            .as_ref()
            .map(|step| step.input_momentum.clone())
            .ok_or(TensorError::InvalidValue {
                label: "klein_gordon_forward_cache",
            })?;
        let (rows, cols) = grad_output.shape();
        let zero_grad_momentum = Tensor::zeros(rows, cols)?;
        Ok(self
            .backward_phase_space(input, &momentum, grad_output, &zero_grad_momentum)?
            .grad_field)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.mass)?;
        visitor(&self.spinor)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.mass)?;
        visitor(&mut self.spinor)?;
        Ok(())
    }
}

/// Hamilton–Jacobi style flow that nudges timelines towards minimal-action
/// trajectories. Each column receives an independent potential term that
/// encourages smoothness across the temporal dimension.
#[derive(Debug)]
pub struct HamiltonJacobiFlow {
    potential: Parameter,
    step_size: f32,
}

impl HamiltonJacobiFlow {
    /// Creates a flow model for the provided feature count.
    pub fn new(name: impl Into<String>, features: usize, step_size: f32) -> PureResult<Self> {
        if features == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: 1,
                cols: features,
            });
        }
        if !step_size.is_finite() || step_size <= 0.0 {
            return Err(TensorError::InvalidValue {
                label: "hamilton_jacobi_step",
            });
        }
        let name = name.into();
        let potential = Tensor::from_fn(1, features, |_r, c| {
            ((c as f32 + 1.0) * 0.13).sin().abs() + 0.1
        })?;
        Ok(Self {
            potential: Parameter::new(format!("{name}::potential"), potential),
            step_size,
        })
    }

    /// Returns the internal potential parameter.
    pub fn potential(&self) -> &Parameter {
        &self.potential
    }

    /// Returns the configured gradient descent step size.
    pub fn step_size(&self) -> f32 {
        self.step_size
    }
}

impl Module for HamiltonJacobiFlow {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if cols != self.potential.value().shape().1 {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: self.potential.value().shape(),
            });
        }
        let route_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        let requested_backend = tensor_util_backend_label(route_backend);
        let step = self.step_size;
        let potential = self.potential.value().data();
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(route_backend, TensorUtilBackend::GpuWgpu)
                && rows > 0
                && cols > 0
                && wgpu_dense::is_available()
            {
                match wgpu_dense::dynamic_hamilton_jacobi_forward(
                    input.data(),
                    potential,
                    rows,
                    cols,
                    step,
                ) {
                    Ok(buffer) => {
                        validate_finite_slice("dynamic_hamilton_jacobi_forward_output", &buffer)?;
                        let output = Tensor::from_vec(rows, cols, buffer)?;
                        emit_dynamic_field_route_meta(
                            "dynamic_field_hamilton_jacobi_forward",
                            "hamilton_jacobi",
                            rows,
                            cols,
                            1,
                            8,
                            false,
                            None,
                            "wgpu_dense",
                            requested_backend,
                            "tensor_util.dynamic_hamilton_jacobi_forward",
                            None,
                            None,
                            None,
                            None,
                        );
                        return Ok(output);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(dynamic_field_wgpu_error(
                            "dynamic_field_hamilton_jacobi_forward",
                            message,
                        ));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut output = Tensor::zeros(rows, cols)?;
        {
            let input_buf = input.data();
            let out_buf = output.data_mut();
            for r in 0..rows {
                let offset = r * cols;
                let input_row = &input_buf[offset..offset + cols];
                let prev_row = if r > 0 {
                    &input_buf[offset - cols..offset]
                } else {
                    input_row
                };
                let next_row = if r + 1 < rows {
                    &input_buf[offset + cols..offset + 2 * cols]
                } else {
                    input_row
                };
                let out_row = &mut out_buf[offset..offset + cols];

                for (((out, &current), &potential), (&prev, &next)) in out_row
                    .iter_mut()
                    .zip(input_row.iter())
                    .zip(potential.iter())
                    .zip(prev_row.iter().zip(next_row.iter()))
                {
                    let grad = (2.0 * current - prev - next) + potential * current;
                    *out = current - step * grad;
                }
            }
        }
        emit_dynamic_field_route_meta(
            "dynamic_field_hamilton_jacobi_forward",
            "hamilton_jacobi",
            rows,
            cols,
            1,
            8,
            false,
            None,
            "cpu",
            requested_backend,
            "dynamic_field.scalar",
            None,
            None,
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
        Ok(output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if grad_output.shape() != (rows, cols) {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (rows, cols),
            });
        }
        let route_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        let requested_backend = tensor_util_backend_label(route_backend);
        let scale_backend = current_tensor_util_backend_for_values(cols);
        let scale_backend_label = tensor_util_backend_label(scale_backend);
        let step = self.step_size;
        let potential = self.potential.value().data();
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(route_backend, TensorUtilBackend::GpuWgpu)
                && rows > 0
                && cols > 0
                && wgpu_dense::is_available()
            {
                match wgpu_dense::dynamic_hamilton_jacobi_backward(
                    input.data(),
                    grad_output.data(),
                    potential,
                    rows,
                    cols,
                    step,
                ) {
                    Ok((grad_input_values, grad_potential_values)) => {
                        validate_finite_slice(
                            "dynamic_hamilton_jacobi_backward_grad_input",
                            &grad_input_values,
                        )?;
                        validate_finite_slice(
                            "dynamic_hamilton_jacobi_backward_grad_potential",
                            &grad_potential_values,
                        )?;
                        let grad_input = Tensor::from_vec(rows, cols, grad_input_values)?;
                        let grad_potential = Tensor::from_vec(1, cols, grad_potential_values)?;
                        let gradient_scale = if rows > 0 {
                            let scale = 1.0 / rows as f32;
                            self.potential.accumulate_euclidean(
                                &grad_potential.scale_with_backend(scale, scale_backend)?,
                            )?;
                            Some(scale)
                        } else {
                            None
                        };
                        emit_dynamic_field_route_meta(
                            "dynamic_field_hamilton_jacobi_backward",
                            "hamilton_jacobi",
                            rows,
                            cols,
                            1,
                            12,
                            true,
                            gradient_scale,
                            "wgpu_dense",
                            requested_backend,
                            "tensor_util.dynamic_hamilton_jacobi_backward",
                            Some("wgpu"),
                            Some("wgpu"),
                            Some(scale_backend_label),
                            None,
                        );
                        return Ok(grad_input);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(dynamic_field_wgpu_error(
                            "dynamic_field_hamilton_jacobi_backward",
                            message,
                        ));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut grad_input = Tensor::zeros(rows, cols)?;
        let mut grad_potential = Tensor::zeros(1, cols)?;
        {
            let input_buf = input.data();
            let grad_output_buf = grad_output.data();
            let grad_input_buf = grad_input.data_mut();
            let grad_potential_buf = grad_potential.data_mut();
            for r in 0..rows {
                let offset = r * cols;
                for c in 0..cols {
                    let idx = offset + c;
                    let current = input_buf[idx];
                    let go = grad_output_buf[idx];
                    if rows == 1 {
                        grad_input_buf[idx] += go * (1.0 - step * potential[c]);
                    } else if r == 0 {
                        grad_input_buf[idx] += go * (1.0 - step * (1.0 + potential[c]));
                        grad_input_buf[idx + cols] += go * step;
                    } else if r + 1 == rows {
                        grad_input_buf[idx] += go * (1.0 - step * (1.0 + potential[c]));
                        grad_input_buf[idx - cols] += go * step;
                    } else {
                        grad_input_buf[idx] += go * (1.0 - step * (2.0 + potential[c]));
                        grad_input_buf[idx - cols] += go * step;
                        grad_input_buf[idx + cols] += go * step;
                    }
                    grad_potential_buf[c] += -step * current * go;
                }
            }
        }
        let gradient_scale = if rows > 0 {
            let scale = 1.0 / rows as f32;
            self.potential
                .accumulate_euclidean(&grad_potential.scale_with_backend(scale, scale_backend)?)?;
            Some(scale)
        } else {
            None
        };
        emit_dynamic_field_route_meta(
            "dynamic_field_hamilton_jacobi_backward",
            "hamilton_jacobi",
            rows,
            cols,
            1,
            12,
            true,
            gradient_scale,
            "cpu",
            requested_backend,
            "dynamic_field.scalar",
            Some("cpu"),
            Some("cpu"),
            if gradient_scale.is_some() {
                Some(scale_backend_label)
            } else {
                None
            },
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
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.potential)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.potential)
    }
}

fn stochastic_schrodinger_error(error: impl std::fmt::Display) -> TensorError {
    TensorError::Generic(format!("stochastic Schrodinger contract failed: {error}"))
}

#[derive(Clone, Debug)]
struct StochasticSchrodingerStepCache {
    input: Tensor,
    imaginary: Tensor,
    phase: Tensor,
    audit: StochasticSchrodingerAudit,
    backward_audit: Option<StochasticSchrodingerBackwardAudit>,
}

/// Audited real-time open-system Schrödinger step for real-valued tensor clients.
///
/// Each row enters as the real quadrature of a complex wavefunction. The Rust
/// core owns the learned diagonal Hamiltonian, exact split subflows, second-order
/// Strang composition, Gaussian Stratonovich phase diffusion, no-jump damping,
/// analytic backward, and numerical audits. WGPU executes the same contract and
/// is audited by the core.
#[derive(Debug)]
pub struct StochasticSchrodingerLayer {
    // The compatibility name keeps existing state dictionaries loadable. Its
    // canonical meaning is the learned diagonal Hamiltonian potential.
    coherence: Parameter,
    config: StochasticSchrodingerConfig,
    rng: RefCell<StdRng>,
    last_step: RefCell<Option<StochasticSchrodingerStepCache>>,
}

impl StochasticSchrodingerLayer {
    /// Creates a stochastic layer using configured Gaussian phase noise.
    pub fn new(
        name: impl Into<String>,
        features: usize,
        loss_rate: f32,
        noise_scale: f32,
    ) -> PureResult<Self> {
        Self::with_seed(name, features, loss_rate, noise_scale, None)
    }

    /// Same as [`StochasticSchrodingerLayer::new`] with an optional deterministic
    /// seed for pathwise CPU/WGPU replay and reproducible experiments.
    pub fn with_seed(
        name: impl Into<String>,
        features: usize,
        loss_rate: f32,
        noise_scale: f32,
        seed: Option<u64>,
    ) -> PureResult<Self> {
        let config = StochasticSchrodingerConfig::new(loss_rate, noise_scale)
            .map_err(stochastic_schrodinger_error)?;
        Self::with_seed_and_config(name, features, config, seed)
    }

    /// Creates a layer from the versioned Rust-core evolution contract.
    pub fn with_seed_and_config(
        name: impl Into<String>,
        features: usize,
        config: StochasticSchrodingerConfig,
        seed: Option<u64>,
    ) -> PureResult<Self> {
        if features == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: 1,
                cols: features,
            });
        }
        config.validate().map_err(stochastic_schrodinger_error)?;
        let name = name.into();
        let potential = Tensor::from_fn(1, features, |_r, c| (0.85 - (c as f32 * 0.03)).max(0.1))?;
        let rng = determinism::rng_from_optional(seed, "st-nn/layers/stochastic_schrodinger");
        Ok(Self {
            coherence: Parameter::new(format!("{name}::coherence"), potential),
            config,
            rng: RefCell::new(rng),
            last_step: RefCell::new(None),
        })
    }

    /// Sets the real-time integration step.
    pub fn with_time_step(mut self, time_step: f32) -> PureResult<Self> {
        self.config = self
            .config
            .with_time_step(time_step)
            .map_err(stochastic_schrodinger_error)?;
        Ok(self)
    }

    /// Sets the Hermitian adjacent-pair hopping rate.
    pub fn with_hopping_rate(mut self, hopping_rate: f32) -> PureResult<Self> {
        self.config = self
            .config
            .with_hopping_rate(hopping_rate)
            .map_err(stochastic_schrodinger_error)?;
        Ok(self)
    }

    /// Returns the complete versioned evolution configuration.
    pub fn config(&self) -> StochasticSchrodingerConfig {
        self.config
    }

    pub fn time_step(&self) -> f32 {
        self.config.time_step()
    }

    pub fn hopping_rate(&self) -> f32 {
        self.config.hopping_rate()
    }

    pub fn loss_rate(&self) -> f32 {
        self.config.loss_rate()
    }

    /// Compatibility alias for [`StochasticSchrodingerLayer::loss_rate`].
    pub fn decoherence_rate(&self) -> f32 {
        self.loss_rate()
    }

    pub fn noise_scale(&self) -> f32 {
        self.config.noise_scale()
    }

    /// Canonical accessor for the learned diagonal Hamiltonian potential.
    pub fn potential(&self) -> &Parameter {
        &self.coherence
    }

    /// Compatibility alias for [`StochasticSchrodingerLayer::potential`].
    pub fn coherence(&self) -> &Parameter {
        self.potential()
    }

    /// Returns the imaginary quadrature retained from the latest forward step.
    pub fn latest_imaginary_quadrature(&self) -> Option<Tensor> {
        self.last_step
            .borrow()
            .as_ref()
            .map(|step| step.imaginary.clone())
    }

    /// Returns the sampled diagonal phase from the latest forward step.
    pub fn latest_phase(&self) -> Option<Tensor> {
        self.last_step
            .borrow()
            .as_ref()
            .map(|step| step.phase.clone())
    }

    /// Returns the fail-closed norm audit from the latest forward step.
    pub fn latest_audit(&self) -> Option<StochasticSchrodingerAudit> {
        self.last_step.borrow().as_ref().map(|step| step.audit)
    }

    /// Returns the Rust-core audit of the latest completed backward pass.
    pub fn latest_backward_audit(&self) -> Option<StochasticSchrodingerBackwardAudit> {
        self.last_step
            .borrow()
            .as_ref()
            .and_then(|step| step.backward_audit)
    }

    fn sample_standard_normal(&self, volume: usize) -> Vec<f32> {
        let mut rng = self.rng.borrow_mut();
        (0..volume)
            .map(|_| {
                let sample: f32 = StandardNormal.sample(&mut *rng);
                sample
            })
            .collect()
    }

    fn commit_step(
        &self,
        input: &Tensor,
        output_real: Vec<f32>,
        output_imaginary: Vec<f32>,
        phase: Vec<f32>,
        audit: StochasticSchrodingerAudit,
    ) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        let output = Tensor::from_vec(rows, cols, output_real)?;
        let imaginary = Tensor::from_vec(rows, cols, output_imaginary)?;
        let phase = Tensor::from_vec(rows, cols, phase)?;
        self.last_step
            .borrow_mut()
            .replace(StochasticSchrodingerStepCache {
                input: input.clone(),
                imaginary,
                phase,
                audit,
                backward_audit: None,
            });
        Ok(output)
    }

    fn commit_backward_audit(&self, audit: StochasticSchrodingerBackwardAudit) {
        if let Some(step) = self.last_step.borrow_mut().as_mut() {
            step.backward_audit = Some(audit);
        }
    }
}

impl Module for StochasticSchrodingerLayer {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if cols != self.coherence.value().shape().1 {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: self.coherence.value().shape(),
            });
        }
        let volume = rows.checked_mul(cols).ok_or_else(|| {
            TensorError::Generic("stochastic Schrodinger tensor volume overflow".to_string())
        })?;
        let potential = self.coherence.value().data();
        validate_stochastic_schrodinger_state(input.data(), potential, rows, cols, self.config)
            .map_err(stochastic_schrodinger_error)?;
        let standard_normal = self.sample_standard_normal(volume);
        let route_backend = current_tensor_util_backend_for_values(volume);
        let requested_backend = tensor_util_backend_label(route_backend);
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(route_backend, TensorUtilBackend::GpuWgpu)
                && rows > 0
                && cols > 0
                && wgpu_dense::is_available()
            {
                let wgpu_step = wgpu_dense::dynamic_schrodinger_forward(
                    input.data(),
                    potential,
                    &standard_normal,
                    rows,
                    cols,
                    self.config.time_step(),
                    self.config.hopping_rate(),
                    self.config.loss_rate(),
                    self.config.noise_scale(),
                )
                .and_then(|(output_real, output_imaginary, phase)| {
                    audit_stochastic_schrodinger_step(StochasticSchrodingerAuditRequest {
                        input: input.data(),
                        potential,
                        output_real: &output_real,
                        output_imaginary: &output_imaginary,
                        phase: &phase,
                        standard_normal: &standard_normal,
                        rows,
                        features: cols,
                        config: self.config,
                    })
                    .map(|audit| (output_real, output_imaginary, phase, audit))
                    .map_err(|error| format!("Rust semantic audit failed: {error}"))
                });
                match wgpu_step {
                    Ok((output_real, output_imaginary, phase, audit)) => {
                        let output =
                            self.commit_step(input, output_real, output_imaginary, phase, audit)?;
                        emit_stochastic_schrodinger_route_meta(
                            "dynamic_field_stochastic_schrodinger_forward",
                            rows,
                            cols,
                            36,
                            false,
                            None,
                            "wgpu_dense",
                            requested_backend,
                            "tensor_util.dynamic_schrodinger_forward",
                            None,
                            None,
                            None,
                            None,
                            self.config,
                            Some(audit),
                            None,
                        );
                        return Ok(output);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(dynamic_field_wgpu_error(
                            "dynamic_field_stochastic_schrodinger_forward",
                            message,
                        ));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let step = apply_stochastic_schrodinger_step(
            input.data(),
            potential,
            &standard_normal,
            rows,
            cols,
            self.config,
        )
        .map_err(stochastic_schrodinger_error)?;
        let audit = step.audit;
        let output = self.commit_step(
            input,
            step.output_real,
            step.output_imaginary,
            step.phase,
            audit,
        )?;
        emit_stochastic_schrodinger_route_meta(
            "dynamic_field_stochastic_schrodinger_forward",
            rows,
            cols,
            36,
            false,
            None,
            "cpu",
            requested_backend,
            "st_core.apply_stochastic_schrodinger_step",
            None,
            None,
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
            self.config,
            Some(audit),
            None,
        );
        Ok(output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let (cached_input, cached_phase, audit) = {
            let cache = self.last_step.borrow();
            let Some(cache) = cache.as_ref() else {
                return Err(TensorError::InvalidValue {
                    label: "schrodinger_forward_cache",
                });
            };
            if input.shape() != cache.input.shape() {
                return Err(TensorError::ShapeMismatch {
                    left: input.shape(),
                    right: cache.input.shape(),
                });
            }
            if grad_output.shape() != cache.input.shape() {
                return Err(TensorError::ShapeMismatch {
                    left: grad_output.shape(),
                    right: cache.input.shape(),
                });
            }
            if input.data() != cache.input.data() {
                return Err(TensorError::InvalidValue {
                    label: "schrodinger_forward_input_mismatch",
                });
            }
            (cache.input.clone(), cache.phase.clone(), cache.audit)
        };
        let (rows, cols) = cached_input.shape();
        let route_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        let requested_backend = tensor_util_backend_label(route_backend);
        let scale_backend = current_tensor_util_backend_for_values(cols);
        let scale_backend_label = tensor_util_backend_label(scale_backend);
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(route_backend, TensorUtilBackend::GpuWgpu)
                && rows > 0
                && cols > 0
                && wgpu_dense::is_available()
            {
                let wgpu_backward = wgpu_dense::dynamic_schrodinger_backward(
                    cached_input.data(),
                    cached_phase.data(),
                    grad_output.data(),
                    rows,
                    cols,
                    self.config.time_step(),
                    self.config.hopping_rate(),
                    self.config.loss_rate(),
                )
                .and_then(|(grad_input, grad_potential)| {
                    audit_stochastic_schrodinger_backward(
                        StochasticSchrodingerBackwardAuditRequest {
                            input: cached_input.data(),
                            phase: cached_phase.data(),
                            grad_output: grad_output.data(),
                            grad_input: &grad_input,
                            grad_potential: &grad_potential,
                            rows,
                            features: cols,
                            config: self.config,
                        },
                    )
                    .map(|audit| (grad_input, grad_potential, audit))
                    .map_err(|error| format!("Rust semantic backward audit failed: {error}"))
                });
                match wgpu_backward {
                    Ok((grad_input_values, grad_potential_values, backward_audit)) => {
                        validate_finite_slice(
                            "dynamic_schrodinger_backward_grad_input",
                            &grad_input_values,
                        )?;
                        validate_finite_slice(
                            "dynamic_schrodinger_backward_grad_potential",
                            &grad_potential_values,
                        )?;
                        let grad_input = Tensor::from_vec(rows, cols, grad_input_values)?;
                        let grad_potential = Tensor::from_vec(1, cols, grad_potential_values)?;
                        let gradient_scale = if rows > 0 {
                            let scale = 1.0 / rows as f32;
                            self.coherence.accumulate_euclidean(
                                &grad_potential.scale_with_backend(scale, scale_backend)?,
                            )?;
                            Some(scale)
                        } else {
                            None
                        };
                        self.commit_backward_audit(backward_audit);
                        emit_stochastic_schrodinger_route_meta(
                            "dynamic_field_stochastic_schrodinger_backward",
                            rows,
                            cols,
                            32,
                            true,
                            gradient_scale,
                            "wgpu_dense",
                            requested_backend,
                            "tensor_util.dynamic_schrodinger_backward",
                            Some("wgpu"),
                            Some("wgpu"),
                            Some(scale_backend_label),
                            None,
                            self.config,
                            Some(audit),
                            Some(backward_audit),
                        );
                        return Ok(grad_input);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(dynamic_field_wgpu_error(
                            "dynamic_field_stochastic_schrodinger_backward",
                            message,
                        ));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let backward = backward_stochastic_schrodinger_step(
            cached_input.data(),
            cached_phase.data(),
            grad_output.data(),
            rows,
            cols,
            self.config,
        )
        .map_err(stochastic_schrodinger_error)?;
        let backward_audit =
            audit_stochastic_schrodinger_backward(StochasticSchrodingerBackwardAuditRequest {
                input: cached_input.data(),
                phase: cached_phase.data(),
                grad_output: grad_output.data(),
                grad_input: &backward.grad_input,
                grad_potential: &backward.grad_potential,
                rows,
                features: cols,
                config: self.config,
            })
            .map_err(stochastic_schrodinger_error)?;
        let grad_input = Tensor::from_vec(rows, cols, backward.grad_input)?;
        let grad_potential = Tensor::from_vec(1, cols, backward.grad_potential)?;
        let gradient_scale = if rows > 0 {
            let scale = 1.0 / rows as f32;
            self.coherence
                .accumulate_euclidean(&grad_potential.scale_with_backend(scale, scale_backend)?)?;
            Some(scale)
        } else {
            None
        };
        self.commit_backward_audit(backward_audit);
        emit_stochastic_schrodinger_route_meta(
            "dynamic_field_stochastic_schrodinger_backward",
            rows,
            cols,
            32,
            true,
            gradient_scale,
            "cpu",
            requested_backend,
            "st_core.backward_stochastic_schrodinger_step",
            Some("cpu"),
            Some("cpu"),
            if gradient_scale.is_some() {
                Some(scale_backend_label)
            } else {
                None
            },
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
            self.config,
            Some(audit),
            Some(backward_audit),
        );
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.coherence)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.coherence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "wgpu")]
    use crate::execution::{push_backend_policy, BackendPolicy};
    #[cfg(feature = "wgpu")]
    use st_core::backend::device_caps::DeviceCaps;
    use std::sync::{Arc, Mutex};

    fn observer_lock() -> std::sync::MutexGuard<'static, ()> {
        crate::test_global_state_lock()
    }

    fn assert_close(left: f32, right: f32) {
        let diff = (left - right).abs();
        assert!(diff < 1e-6, "expected {left} ≈ {right} (diff={diff})");
    }

    #[cfg(feature = "wgpu")]
    fn approx_eq(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        for (idx, (l, r)) in left.iter().zip(right.iter()).enumerate() {
            let diff = (l - r).abs();
            assert!(diff < 1e-5, "idx={idx} left={l} right={r} diff={diff}");
        }
    }

    #[test]
    fn klein_gordon_shapes_match() {
        let layer = KleinGordonPropagation::new("kg", 4, 0.2, 0.1).unwrap();
        let input = Tensor::from_vec(3, 4, vec![0.1; 12]).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), input.shape());
        assert_eq!(layer.latest_momentum().unwrap().shape(), input.shape());
        let audit = layer.latest_audit().expect("Klein-Gordon audit");
        assert!(audit.stability_margin > 0.0);
        assert_eq!(audit.max_field_error, 0.0);
    }

    #[test]
    fn klein_gordon_explicit_phase_space_and_adjoint_match_core() {
        let input = Tensor::from_vec(2, 3, vec![0.2, -0.3, 0.1, -0.15, 0.25, 0.4]).unwrap();
        let momentum = Tensor::from_vec(2, 3, vec![0.05, -0.02, 0.08, -0.04, 0.03, 0.01]).unwrap();
        let grad_field = Tensor::from_vec(2, 3, vec![0.4, -0.2, 0.1, 0.3, 0.05, -0.25]).unwrap();
        let grad_momentum =
            Tensor::from_vec(2, 3, vec![0.03, -0.04, 0.02, 0.01, 0.05, -0.02]).unwrap();
        let mut layer = KleinGordonPropagation::new("kg", 3, 0.08, 0.06)
            .unwrap()
            .with_wave_speed(0.7)
            .unwrap()
            .with_self_coupling(0.03)
            .unwrap();
        let step = layer.step_phase_space(&input, &momentum).unwrap();
        assert_eq!(step.field.shape(), input.shape());
        assert_eq!(step.momentum.shape(), input.shape());
        let expected = backward_klein_gordon_step(KleinGordonBackwardRequest {
            field: input.data(),
            momentum: momentum.data(),
            mass_squared: layer.mass_squared().value().data(),
            source: layer.source().value().data(),
            grad_output_field: grad_field.data(),
            grad_output_momentum: grad_momentum.data(),
            rows: 2,
            features: 3,
            config: layer.config(),
        })
        .unwrap();
        let backward = layer
            .backward_phase_space(&input, &momentum, &grad_field, &grad_momentum)
            .unwrap();
        for (observed, expected) in backward.grad_field.data().iter().zip(&expected.grad_field) {
            assert_close(*observed, *expected);
        }
        for (observed, expected) in backward
            .grad_momentum
            .data()
            .iter()
            .zip(&expected.grad_momentum)
        {
            assert_close(*observed, *expected);
        }
        assert_eq!(backward.audit.max_grad_field_error, 0.0);
        assert!(layer.latest_input_momentum_gradient().is_some());
    }

    #[test]
    fn klein_gordon_backward_rejects_missing_or_stale_phase_state() {
        let input = Tensor::from_vec(1, 2, vec![0.2, -0.1]).unwrap();
        let grad = Tensor::from_vec(1, 2, vec![0.3, 0.4]).unwrap();
        let mut layer = KleinGordonPropagation::new("kg", 2, 0.1, 0.0).unwrap();
        assert!(matches!(
            layer.backward(&input, &grad),
            Err(TensorError::InvalidValue {
                label: "klein_gordon_forward_cache"
            })
        ));
        let _ = layer.forward(&input).unwrap();
        let stale = Tensor::from_vec(1, 2, vec![0.2, -0.11]).unwrap();
        assert!(matches!(
            layer.backward(&stale, &grad),
            Err(TensorError::InvalidValue {
                label: "klein_gordon_forward_field_mismatch"
            })
        ));
    }

    #[test]
    fn klein_gordon_backward_uses_forward_parameter_snapshot() {
        let input = Tensor::from_vec(1, 3, vec![0.2, -0.1, 0.35]).unwrap();
        let grad = Tensor::from_vec(1, 3, vec![0.4, -0.25, 0.1]).unwrap();
        let mut layer = KleinGordonPropagation::new("kg", 3, 0.1, 0.02).unwrap();
        let mass = layer.mass_squared().value().data().to_vec();
        let source = layer.source().value().data().to_vec();
        let _ = layer.forward(&input).unwrap();
        let expected = backward_klein_gordon_step(KleinGordonBackwardRequest {
            field: input.data(),
            momentum: &[0.0; 3],
            mass_squared: &mass,
            source: &source,
            grad_output_field: grad.data(),
            grad_output_momentum: &[0.0; 3],
            rows: 1,
            features: 3,
            config: layer.config(),
        })
        .unwrap();
        layer
            .visit_parameters_mut(&mut |parameter| {
                parameter.value_mut().data_mut()[0] += 0.75;
                Ok(())
            })
            .unwrap();
        let observed = layer.backward(&input, &grad).unwrap();
        for (observed, expected) in observed.data().iter().zip(&expected.grad_field) {
            assert_close(*observed, *expected);
        }
        for (observed, expected) in layer
            .mass_squared()
            .gradient()
            .unwrap()
            .data()
            .iter()
            .zip(&expected.grad_mass_squared)
        {
            assert_close(*observed, *expected);
        }
        for (observed, expected) in layer
            .source()
            .gradient()
            .unwrap()
            .data()
            .iter()
            .zip(&expected.grad_source)
        {
            assert_close(*observed, *expected);
        }
    }

    #[test]
    fn hamilton_jacobi_smooths_edges() {
        let mut layer = HamiltonJacobiFlow::new("hj", 3, 0.1).unwrap();
        let input = Tensor::from_vec(
            4,
            3,
            vec![
                1.0, 0.0, -1.0, 1.0, 0.5, -0.5, 1.0, 0.0, -1.0, 1.0, 0.5, -0.5,
            ],
        )
        .unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), input.shape());
        let grad = Tensor::from_vec(4, 3, vec![0.2; 12]).unwrap();
        let back = layer.backward(&input, &grad).unwrap();
        assert_eq!(back.shape(), input.shape());
    }

    #[test]
    fn dynamic_field_parameter_gradients_are_batch_normalized() {
        let input = Tensor::from_vec(2, 2, vec![0.5, -0.3, 0.7, 0.2]).unwrap();
        let grad = Tensor::from_vec(2, 2, vec![0.4, -0.1, 0.2, 0.5]).unwrap();

        let mut kg = KleinGordonPropagation::new("kg", 2, 0.2, 0.1).unwrap();
        let _ = kg.forward(&input).unwrap();
        let zero_momentum = vec![0.0; 4];
        let zero_grad_momentum = vec![0.0; 4];
        let expected_backward = backward_klein_gordon_step(KleinGordonBackwardRequest {
            field: input.data(),
            momentum: &zero_momentum,
            mass_squared: kg.mass_squared().value().data(),
            source: kg.source().value().data(),
            grad_output_field: grad.data(),
            grad_output_momentum: &zero_grad_momentum,
            rows: 2,
            features: 2,
            config: kg.config(),
        })
        .unwrap();
        let grad_input = kg.backward(&input, &grad).unwrap();
        assert_eq!(grad_input.shape(), input.shape());
        let mass_grad = kg.mass().gradient().expect("mass gradient");
        let spin_grad = kg.spinor().gradient().expect("spinor gradient");
        for col in 0..2 {
            assert_close(
                mass_grad.data()[col],
                expected_backward.grad_mass_squared[col] * 0.5,
            );
            assert_close(
                spin_grad.data()[col],
                expected_backward.grad_source[col] * 0.5,
            );
        }

        let mut hj = HamiltonJacobiFlow::new("hj", 2, 0.1).unwrap();
        let grad_input = hj.backward(&input, &grad).unwrap();
        assert_eq!(grad_input.shape(), input.shape());
        let potential_grad = hj.potential().gradient().expect("potential gradient");
        for col in 0..2 {
            let mut expected = 0.0f32;
            for row in 0..2 {
                let idx = row * 2 + col;
                expected += -0.1 * input.data()[idx] * grad.data()[idx];
            }
            assert_close(potential_grad.data()[col], expected * 0.5);
        }

        let mut qs = StochasticSchrodingerLayer::with_seed("qs", 2, 0.4, 0.0, Some(7)).unwrap();
        let _ = qs.forward(&input).unwrap();
        let phase = qs.latest_phase().expect("forward phase");
        let expected_backward = backward_stochastic_schrodinger_step(
            input.data(),
            phase.data(),
            grad.data(),
            2,
            2,
            qs.config(),
        )
        .expect("core backward");
        let grad_input = qs.backward(&input, &grad).unwrap();
        assert_eq!(grad_input.shape(), input.shape());
        let coherence_grad = qs.coherence().gradient().expect("coherence gradient");
        for col in 0..2 {
            assert_close(
                coherence_grad.data()[col],
                expected_backward.grad_potential[col] * 0.5,
            );
        }
    }

    #[test]
    fn dynamic_field_empty_batches_skip_parameter_updates() {
        let input = Tensor::from_vec(0, 2, Vec::new()).unwrap();
        let grad = Tensor::from_vec(0, 2, Vec::new()).unwrap();

        let mut kg = KleinGordonPropagation::new("kg", 2, 0.2, 0.1).unwrap();
        let _ = kg.forward(&input).unwrap();
        let grad_input = kg.backward(&input, &grad).unwrap();
        assert_eq!(grad_input.shape(), input.shape());
        assert!(grad_input.data().is_empty());
        assert!(kg.mass().gradient().is_none());
        assert!(kg.spinor().gradient().is_none());

        let mut hj = HamiltonJacobiFlow::new("hj", 2, 0.1).unwrap();
        let _ = hj.forward(&input).unwrap();
        let grad_input = hj.backward(&input, &grad).unwrap();
        assert_eq!(grad_input.shape(), input.shape());
        assert!(grad_input.data().is_empty());
        assert!(hj.potential().gradient().is_none());

        let mut qs = StochasticSchrodingerLayer::with_seed("qs", 2, 0.4, 0.0, Some(7)).unwrap();
        let _ = qs.forward(&input).unwrap();
        let grad_input = qs.backward(&input, &grad).unwrap();
        assert_eq!(grad_input.shape(), input.shape());
        assert!(grad_input.data().is_empty());
        assert!(qs.coherence().gradient().is_none());
    }

    #[test]
    fn dynamic_field_layers_emit_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let input = Tensor::from_vec(2, 3, vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6]).unwrap();
        let grad = Tensor::from_vec(2, 3, vec![0.05; 6]).unwrap();

        let mut kg = KleinGordonPropagation::new("kg", 3, 0.2, 0.1).unwrap();
        let _ = kg.forward(&input).unwrap();
        let _ = kg.backward(&input, &grad).unwrap();

        let mut hj = HamiltonJacobiFlow::new("hj", 3, 0.1).unwrap();
        let _ = hj.forward(&input).unwrap();
        let _ = hj.backward(&input, &grad).unwrap();

        let mut qs = StochasticSchrodingerLayer::with_seed("qs", 3, 0.4, 0.05, Some(7)).unwrap();
        let _ = qs.forward(&input).unwrap();
        let _ = qs.backward(&input, &grad).unwrap();
        st_tensor::set_thread_meta_observer(previous);

        let events = events.lock().unwrap();
        for (op_name, field_model, kind) in [
            (
                "dynamic_field_klein_gordon_forward",
                "klein_gordon",
                "dynamic_field_forward",
            ),
            (
                "dynamic_field_klein_gordon_backward",
                "klein_gordon",
                "dynamic_field_backward",
            ),
            (
                "dynamic_field_hamilton_jacobi_forward",
                "hamilton_jacobi",
                "dynamic_field_forward",
            ),
            (
                "dynamic_field_hamilton_jacobi_backward",
                "hamilton_jacobi",
                "dynamic_field_backward",
            ),
            (
                "dynamic_field_stochastic_schrodinger_forward",
                "stochastic_schrodinger",
                "dynamic_field_forward",
            ),
            (
                "dynamic_field_stochastic_schrodinger_backward",
                "stochastic_schrodinger",
                "dynamic_field_backward",
            ),
        ] {
            let event = events
                .iter()
                .find(|(observed, data)| *observed == op_name && data["rows"] == 2)
                .unwrap_or_else(|| panic!("{op_name} metadata event"));
            assert_eq!(event.1["backend"], "cpu");
            assert_eq!(event.1["kind"], kind);
            assert_eq!(event.1["backward"], kind == "dynamic_field_backward");
            assert_eq!(event.1["field_model"], field_model);
            if kind == "dynamic_field_backward" {
                assert_eq!(event.1["gradient_scale"], 0.5);
                assert_eq!(event.1["parameter_gradient_scale"], 0.5);
                assert_eq!(event.1["input_gradient_scale"], 1.0);
            }
            if field_model == "klein_gordon" {
                assert_eq!(event.1["contract_version"], KLEIN_GORDON_CONTRACT_VERSION);
                assert_eq!(event.1["semantic_owner"], KLEIN_GORDON_SEMANTIC_OWNER);
                assert_eq!(event.1["semantic_backend"], "rust");
                assert_eq!(event.1["equation"], KLEIN_GORDON_EQUATION);
                assert_eq!(event.1["integrator"], KLEIN_GORDON_INTEGRATOR);
                assert_eq!(event.1["boundary"], KLEIN_GORDON_BOUNDARY);
                assert_eq!(event.1["phase_space_state"], KLEIN_GORDON_STATE);
                assert_eq!(event.1["phase_space_output_values"], 12);
                assert!(event.1["audit"]["stability_margin"].is_number());
                assert!(event.1["audit"]["output_energy"].is_number());
                if kind == "dynamic_field_backward" {
                    assert!(event.1["backward_audit"]["max_grad_field_error"].is_number());
                    assert!(event.1["backward_audit"]["max_grad_mass_squared_error"].is_number());
                }
            }
            if field_model == "stochastic_schrodinger" {
                assert_eq!(
                    event.1["contract_version"],
                    STOCHASTIC_SCHRODINGER_CONTRACT_VERSION
                );
                assert_eq!(
                    event.1["semantic_owner"],
                    STOCHASTIC_SCHRODINGER_SEMANTIC_OWNER
                );
                assert_eq!(event.1["semantic_backend"], "rust");
                assert_eq!(event.1["stochastic_calculus"], "stratonovich");
                assert_eq!(event.1["output_observable"], "real_quadrature");
                assert_eq!(event.1["noise_model"], STOCHASTIC_SCHRODINGER_NOISE_MODEL);
                assert!(event.1["audit"]["max_row_norm_error"].is_number());
                if kind == "dynamic_field_backward" {
                    assert!(event.1["backward_audit"]["max_grad_input_error"].is_number());
                    assert!(event.1["backward_audit"]["max_grad_potential_error"].is_number());
                }
            }
            assert!(event.1["estimated_total_ops"].as_u64().unwrap_or(0) > 0);
        }
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn klein_gordon_forced_wgpu_matches_cpu_reference_and_emits_backend() {
        if !wgpu_dense::is_available() {
            return;
        }
        let _lock = observer_lock();
        let previous_threshold = std::env::var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES").ok();
        std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", "1");

        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let input = Tensor::from_fn(257, 5, |row, col| {
            ((row * 11 + col * 5) % 29) as f32 * 0.021 - 0.27
        })
        .unwrap();
        let momentum = Tensor::from_fn(257, 5, |row, col| {
            ((row * 3 + col * 7) % 23) as f32 * 0.011 - 0.12
        })
        .unwrap();
        let grad_field = Tensor::from_fn(257, 5, |row, col| {
            ((row * 7 + col * 3) % 19) as f32 * 0.017 - 0.15
        })
        .unwrap();
        let grad_momentum = Tensor::from_fn(257, 5, |row, col| {
            ((row * 13 + col * 2) % 17) as f32 * 0.013 - 0.09
        })
        .unwrap();
        let cpu_policy = BackendPolicy::from_device_caps(DeviceCaps::cpu());
        let mut cpu_layer = KleinGordonPropagation::new("kg_cpu", 5, 0.12, 0.1)
            .unwrap()
            .with_wave_speed(0.8)
            .unwrap()
            .with_self_coupling(0.04)
            .unwrap();
        let cpu_step = {
            let _guard = push_backend_policy(cpu_policy);
            cpu_layer.step_phase_space(&input, &momentum).unwrap()
        };
        let cpu_backward = {
            let _guard = push_backend_policy(cpu_policy);
            cpu_layer
                .backward_phase_space(&input, &momentum, &grad_field, &grad_momentum)
                .unwrap()
        };

        let wgpu_policy = BackendPolicy::from_device_caps(DeviceCaps::wgpu(32, true, 256));
        let mut wgpu_layer = KleinGordonPropagation::new("kg_wgpu", 5, 0.12, 0.1)
            .unwrap()
            .with_wave_speed(0.8)
            .unwrap()
            .with_self_coupling(0.04)
            .unwrap();
        let (wgpu_step, wgpu_backward) = {
            let _guard = push_backend_policy(wgpu_policy);
            (
                wgpu_layer.step_phase_space(&input, &momentum).unwrap(),
                wgpu_layer
                    .backward_phase_space(&input, &momentum, &grad_field, &grad_momentum)
                    .unwrap(),
            )
        };

        st_tensor::set_thread_meta_observer(previous);
        match previous_threshold {
            Some(value) => std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", value),
            None => std::env::remove_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES"),
        }

        approx_eq(cpu_step.field.data(), wgpu_step.field.data());
        approx_eq(cpu_step.momentum.data(), wgpu_step.momentum.data());
        approx_eq(
            cpu_backward.grad_field.data(),
            wgpu_backward.grad_field.data(),
        );
        approx_eq(
            cpu_backward.grad_momentum.data(),
            wgpu_backward.grad_momentum.data(),
        );
        approx_eq(
            cpu_layer.mass().gradient().unwrap().data(),
            wgpu_layer.mass().gradient().unwrap().data(),
        );
        approx_eq(
            cpu_layer.spinor().gradient().unwrap().data(),
            wgpu_layer.spinor().gradient().unwrap().data(),
        );
        assert!(wgpu_step.audit.max_field_error <= 1.0e-5);
        assert!(wgpu_step.audit.max_momentum_error <= 1.0e-5);
        assert!(wgpu_backward.audit.max_grad_field_error <= 1.0e-5);
        assert!(wgpu_backward.audit.max_grad_momentum_error <= 1.0e-5);

        let events = events.lock().unwrap();
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "dynamic_field_klein_gordon_forward"
                && data["backend"] == "wgpu_dense"
                && data["requested_backend"] == "wgpu"
        }));
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "dynamic_field_klein_gordon_backward"
                && data["backend"] == "wgpu_dense"
                && data["input_gradient_backend"] == "wgpu"
                && data["gradient_reduction_backend"] == "wgpu"
        }));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn hamilton_jacobi_forced_wgpu_matches_cpu_reference_and_emits_backend() {
        if !wgpu_dense::is_available() {
            return;
        }
        let _lock = observer_lock();
        let previous_threshold = std::env::var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES").ok();
        std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", "1");

        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let input = Tensor::from_fn(5, 3, |row, col| {
            ((row * 13 + col * 7) % 19) as f32 * 0.037 - 0.29
        })
        .unwrap();
        let grad = Tensor::from_fn(5, 3, |row, col| {
            ((row * 5 + col * 11) % 17) as f32 * 0.023 - 0.15
        })
        .unwrap();
        let cpu_policy = BackendPolicy::from_device_caps(DeviceCaps::cpu());
        let mut cpu_layer = HamiltonJacobiFlow::new("hj_cpu", 3, 0.1).unwrap();
        let cpu_forward = {
            let _guard = push_backend_policy(cpu_policy);
            cpu_layer.forward(&input).unwrap()
        };
        let cpu_grad_input = {
            let _guard = push_backend_policy(cpu_policy);
            cpu_layer.backward(&input, &grad).unwrap()
        };

        let wgpu_policy = BackendPolicy::from_device_caps(DeviceCaps::wgpu(32, true, 256));
        let mut wgpu_layer = HamiltonJacobiFlow::new("hj_wgpu", 3, 0.1).unwrap();
        let (wgpu_forward, wgpu_grad_input) = {
            let _guard = push_backend_policy(wgpu_policy);
            (
                wgpu_layer.forward(&input).unwrap(),
                wgpu_layer.backward(&input, &grad).unwrap(),
            )
        };

        st_tensor::set_thread_meta_observer(previous);
        match previous_threshold {
            Some(value) => std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", value),
            None => std::env::remove_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES"),
        }

        approx_eq(cpu_forward.data(), wgpu_forward.data());
        approx_eq(cpu_grad_input.data(), wgpu_grad_input.data());
        approx_eq(
            cpu_layer.potential().gradient().unwrap().data(),
            wgpu_layer.potential().gradient().unwrap().data(),
        );

        let events = events.lock().unwrap();
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "dynamic_field_hamilton_jacobi_forward"
                && data["backend"] == "wgpu_dense"
                && data["requested_backend"] == "wgpu"
        }));
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "dynamic_field_hamilton_jacobi_backward"
                && data["backend"] == "wgpu_dense"
                && data["input_gradient_backend"] == "wgpu"
                && data["gradient_reduction_backend"] == "wgpu"
        }));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn stochastic_schrodinger_forced_wgpu_matches_cpu_reference_and_emits_backend() {
        if !wgpu_dense::is_available() {
            return;
        }
        let _lock = observer_lock();
        let previous_threshold = std::env::var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES").ok();
        std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", "1");

        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let input = Tensor::from_fn(257, 3, |row, col| {
            ((row * 17 + col * 5) % 23) as f32 * 0.031 - 0.33
        })
        .unwrap();
        let grad = Tensor::from_fn(257, 3, |row, col| {
            ((row * 7 + col * 13) % 19) as f32 * 0.021 - 0.16
        })
        .unwrap();
        let cpu_policy = BackendPolicy::from_device_caps(DeviceCaps::cpu());
        let mut cpu_layer =
            StochasticSchrodingerLayer::with_seed("qs_cpu", 3, 0.4, 0.05, Some(11)).unwrap();
        let cpu_forward = {
            let _guard = push_backend_policy(cpu_policy);
            cpu_layer.forward(&input).unwrap()
        };
        let cpu_grad_input = {
            let _guard = push_backend_policy(cpu_policy);
            cpu_layer.backward(&input, &grad).unwrap()
        };
        let cpu_imaginary = cpu_layer
            .latest_imaginary_quadrature()
            .expect("CPU imaginary quadrature");
        let cpu_phase = cpu_layer.latest_phase().expect("CPU phase");
        let cpu_audit = cpu_layer.latest_audit().expect("CPU audit");
        let cpu_backward_audit = cpu_layer
            .latest_backward_audit()
            .expect("CPU backward audit");

        let wgpu_policy = BackendPolicy::from_device_caps(DeviceCaps::wgpu(32, true, 256));
        let mut wgpu_layer =
            StochasticSchrodingerLayer::with_seed("qs_wgpu", 3, 0.4, 0.05, Some(11)).unwrap();
        let (wgpu_forward, wgpu_grad_input) = {
            let _guard = push_backend_policy(wgpu_policy);
            (
                wgpu_layer.forward(&input).unwrap(),
                wgpu_layer.backward(&input, &grad).unwrap(),
            )
        };
        let wgpu_imaginary = wgpu_layer
            .latest_imaginary_quadrature()
            .expect("WGPU imaginary quadrature");
        let wgpu_phase = wgpu_layer.latest_phase().expect("WGPU phase");
        let wgpu_audit = wgpu_layer.latest_audit().expect("WGPU audit");
        let wgpu_backward_audit = wgpu_layer
            .latest_backward_audit()
            .expect("WGPU backward audit");

        st_tensor::set_thread_meta_observer(previous);
        match previous_threshold {
            Some(value) => std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", value),
            None => std::env::remove_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES"),
        }

        approx_eq(cpu_forward.data(), wgpu_forward.data());
        approx_eq(cpu_imaginary.data(), wgpu_imaginary.data());
        approx_eq(cpu_phase.data(), wgpu_phase.data());
        approx_eq(cpu_grad_input.data(), wgpu_grad_input.data());
        approx_eq(
            cpu_layer.coherence().gradient().unwrap().data(),
            wgpu_layer.coherence().gradient().unwrap().data(),
        );
        assert!((cpu_audit.final_norm_squared - wgpu_audit.final_norm_squared).abs() < 1e-5);
        assert!(wgpu_audit.max_row_norm_tolerance_ratio <= 1.0);
        assert!(wgpu_audit.max_formula_tolerance_ratio <= 1.0);
        assert_eq!(cpu_backward_audit.max_formula_tolerance_ratio, 0.0);
        assert!(wgpu_backward_audit.max_formula_tolerance_ratio <= 1.0);

        let events = events.lock().unwrap();
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "dynamic_field_stochastic_schrodinger_forward"
                && data["backend"] == "wgpu_dense"
                && data["requested_backend"] == "wgpu"
                && data["deterministic_backend"] == "wgpu"
                && data["rng_backend"] == "cpu"
        }));
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "dynamic_field_stochastic_schrodinger_backward"
                && data["backend"] == "wgpu_dense"
                && data["input_gradient_backend"] == "wgpu"
                && data["gradient_reduction_backend"] == "wgpu"
                && data["backward_audit"]["max_formula_tolerance_ratio"].is_number()
        }));
    }

    #[test]
    fn stochastic_schrodinger_is_reproducible() {
        let mut first =
            StochasticSchrodingerLayer::with_seed("qs_first", 2, 0.4, 0.05, Some(7)).unwrap();
        let replay =
            StochasticSchrodingerLayer::with_seed("qs_replay", 2, 0.4, 0.05, Some(7)).unwrap();
        let input = Tensor::from_vec(1, 2, vec![0.3, -0.7]).unwrap();
        let first_a = first.forward(&input).unwrap();
        let replay_a = replay.forward(&input).unwrap();
        let first_b = first.forward(&input).unwrap();
        let replay_b = replay.forward(&input).unwrap();
        assert_eq!(first_a.data(), replay_a.data());
        assert_eq!(first_b.data(), replay_b.data());
        assert_ne!(first_a.data(), first_b.data());
        let grad = Tensor::from_vec(1, 2, vec![0.1, -0.2]).unwrap();
        let back = first.backward(&input, &grad).unwrap();
        assert_eq!(back.shape(), input.shape());
    }

    #[test]
    fn stochastic_schrodinger_zero_time_is_exact_identity() {
        let layer = StochasticSchrodingerLayer::with_seed("qs", 3, 0.4, 2.0, Some(7))
            .unwrap()
            .with_time_step(0.0)
            .unwrap()
            .with_hopping_rate(4.0)
            .unwrap();
        let input = Tensor::from_vec(2, 3, vec![0.3, -0.7, 1.1, -0.2, 0.4, 0.8]).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output, input);
        assert!(layer
            .latest_imaginary_quadrature()
            .unwrap()
            .data()
            .iter()
            .all(|value| *value == 0.0));
        let audit = layer.latest_audit().unwrap();
        assert_eq!(audit.expected_norm_ratio, 1.0);
        assert_eq!(audit.max_row_norm_error, 0.0);
    }

    #[test]
    fn stochastic_schrodinger_backward_rejects_stale_or_mismatched_input() {
        let input = Tensor::from_vec(1, 2, vec![0.3, -0.7]).unwrap();
        let grad = Tensor::from_vec(1, 2, vec![0.1, -0.2]).unwrap();
        let mut layer = StochasticSchrodingerLayer::with_seed("qs", 2, 0.4, 0.05, Some(7)).unwrap();
        assert!(matches!(
            layer.backward(&input, &grad),
            Err(TensorError::InvalidValue {
                label: "schrodinger_forward_cache"
            })
        ));

        let _ = layer.forward(&input).unwrap();
        let different = Tensor::from_vec(1, 2, vec![0.3, -0.6]).unwrap();
        assert!(matches!(
            layer.backward(&different, &grad),
            Err(TensorError::InvalidValue {
                label: "schrodinger_forward_input_mismatch"
            })
        ));
    }

    #[test]
    fn stochastic_schrodinger_failed_forward_keeps_last_valid_audit() {
        let layer = StochasticSchrodingerLayer::with_seed("qs", 2, 0.4, 0.05, Some(7)).unwrap();
        let input = Tensor::from_vec(1, 2, vec![0.3, -0.7]).unwrap();
        let _ = layer.forward(&input).unwrap();
        let audit = layer.latest_audit().unwrap();

        let non_finite = Tensor::from_vec(1, 2, vec![0.3, f32::NAN]).unwrap();
        assert!(layer.forward(&non_finite).is_err());
        assert_eq!(layer.latest_audit(), Some(audit));
    }
}
