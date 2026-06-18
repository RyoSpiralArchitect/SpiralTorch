// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Lightweight renormalisation-group flow DSL that lives on the Mellin log lattice.
//!
//! The model is intentionally narrative-friendly: the log-scale parameter can be
//! interpreted either as an energy scale or as a story depth, and the running
//! couplings are propagated by explicitly sampling Mellin resonances on the
//! Z-space lattice. The API is geared towards experimentation – researchers can
//! register operators with scaling dimensions, attach resonant Mellin poles, and
//! integrate the flow forward with a single call.

use num_complex::Complex32;
use st_frac::mellin::MellinLogGrid;
use st_frac::mellin_types::{ComplexScalar, MellinError, MellinResult};
use st_tensor::{emit_tensor_op, emit_tensor_op_meta};
use std::sync::Arc;
use thiserror::Error;

/// Error emitted by the [`RGFlowModel`] helpers.
#[derive(Debug, Error)]
pub enum RGFlowError {
    /// The log lattice must contain at least one point.
    #[error("log lattice must contain at least one point")]
    EmptyLattice,
    /// Invalid (non-finite or non-positive) log step supplied by the caller.
    #[error("log step must be finite and positive")]
    InvalidLogStep,
    /// Number of operators does not match the provided initial couplings.
    #[error("initial couplings mismatch: expected {expected}, got {got}")]
    InitialCouplings { expected: usize, got: usize },
    /// A Mellin evaluation failed while computing resonance feedback.
    #[error(transparent)]
    Mellin(#[from] MellinError),
}

/// Convenience result alias for RG flow helpers.
pub type RGFlowResult<T> = Result<T, RGFlowError>;

/// Resonant contribution attached to a running coupling.
#[derive(Clone)]
pub struct ResonanceProfile {
    pole: ComplexScalar,
    residue: ComplexScalar,
    grid: Arc<MellinLogGrid>,
}

impl core::fmt::Debug for ResonanceProfile {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ResonanceProfile")
            .field("pole", &self.pole)
            .field("residue", &self.residue)
            .field("lattice_len", &self.grid_len())
            .finish()
    }
}

impl ResonanceProfile {
    /// Creates a resonance profile from an owned Mellin grid.
    pub fn new(grid: MellinLogGrid, pole: ComplexScalar, residue: ComplexScalar) -> Self {
        Self {
            pole,
            residue,
            grid: Arc::new(grid),
        }
    }

    /// Builds a resonance profile from a shared Mellin grid.
    pub fn with_shared_grid(
        grid: Arc<MellinLogGrid>,
        pole: ComplexScalar,
        residue: ComplexScalar,
    ) -> Self {
        Self {
            pole,
            residue,
            grid,
        }
    }

    fn grid_len(&self) -> usize {
        self.grid.len()
    }

    /// Evaluates the resonance on the supplied log-scale position.
    pub fn evaluate(&self, log_scale: f32) -> MellinResult<ComplexScalar> {
        // The resonance is sampled along a vertical line passing through `pole`.
        // The log-scale drives the imaginary component, matching the Mellin
        // convention used throughout SpiralTorch.
        let s = Complex32::new(self.pole.re, self.pole.im + log_scale);
        self.grid.evaluate(s).map(|value| value * self.residue)
    }
}

/// Operator registered with the RG flow model.
#[derive(Clone, Debug)]
pub struct RGOperator {
    /// Human-readable name used when exporting the flow.
    pub name: String,
    /// Canonical scaling dimension (Δ) controlling the linear beta term.
    pub scaling_dimension: f32,
    /// Base coupling value at the lowest lattice site.
    pub initial_coupling: f32,
    /// Optional resonant contribution sampled from Mellin space.
    pub resonance: Option<ResonanceProfile>,
    /// Additional nonlinear feedback coefficient multiplying `g^2`.
    pub nonlinear_feedback: f32,
}

impl RGOperator {
    /// Creates an operator with the provided metadata.
    pub fn new(name: impl Into<String>, scaling_dimension: f32, initial_coupling: f32) -> Self {
        Self {
            name: name.into(),
            scaling_dimension,
            initial_coupling,
            resonance: None,
            nonlinear_feedback: 0.0,
        }
    }

    /// Attaches a resonance profile to the operator.
    pub fn with_resonance(mut self, resonance: ResonanceProfile) -> Self {
        self.resonance = Some(resonance);
        self
    }

    /// Configures the nonlinear `g^2` feedback coefficient.
    pub fn with_nonlinear_feedback(mut self, coefficient: f32) -> Self {
        self.nonlinear_feedback = coefficient;
        self
    }
}

/// Propagated coupling trajectory returned by the RG flow integrator.
#[derive(Clone, Debug)]
pub struct RGFlowTrajectory {
    /// Operator name.
    pub name: String,
    /// Coupling values at each lattice point.
    pub values: Vec<f32>,
}

impl RGFlowTrajectory {
    /// Returns the coupling sampled at the provided lattice index.
    pub fn at(&self, index: usize) -> Option<f32> {
        self.values.get(index).copied()
    }
}

/// Result of propagating all registered operators across the lattice.
#[derive(Clone, Debug)]
pub struct RGFlowSolution {
    lattice: Vec<f32>,
    trajectories: Vec<RGFlowTrajectory>,
}

impl RGFlowSolution {
    /// Returns the log-scale lattice used for the propagation.
    pub fn lattice(&self) -> &[f32] {
        &self.lattice
    }

    /// Returns the coupling trajectory for the named operator, if present.
    pub fn trajectory(&self, name: &str) -> Option<&RGFlowTrajectory> {
        self.trajectories.iter().find(|traj| traj.name == name)
    }

    /// Iterates over the coupling trajectories in registration order.
    pub fn iter(&self) -> impl Iterator<Item = &RGFlowTrajectory> {
        self.trajectories.iter()
    }
}

/// Minimal renormalisation-group flow engine operating on a log-uniform lattice.
#[derive(Clone, Debug)]
pub struct RGFlowModel {
    log_step: f32,
    lattice: Vec<f32>,
    operators: Vec<RGOperator>,
    /// Controls the damping injected into the beta function to keep stiff flows stable.
    pub damping: f32,
}

impl RGFlowModel {
    /// Builds a model from a log-uniform lattice.
    pub fn new(log_start: f32, log_step: f32, depth: usize) -> RGFlowResult<Self> {
        if depth == 0 {
            return Err(RGFlowError::EmptyLattice);
        }
        if !(log_step.is_finite() && log_step > 0.0) {
            return Err(RGFlowError::InvalidLogStep);
        }
        let mut lattice = Vec::with_capacity(depth);
        for i in 0..depth {
            lattice.push(log_start + log_step * i as f32);
        }
        Ok(Self {
            log_step,
            lattice,
            operators: Vec::new(),
            damping: 0.0,
        })
    }

    /// Builds a model from an arbitrary lattice supplied by the caller.
    pub fn from_lattice(lattice: Vec<f32>) -> RGFlowResult<Self> {
        if lattice.is_empty() {
            return Err(RGFlowError::EmptyLattice);
        }
        // Estimate step size from the first two entries when possible. When the lattice
        // is not uniform this value is only used as a stability hint.
        let log_step = if lattice.len() > 1 {
            lattice[1] - lattice[0]
        } else {
            1.0
        };
        Ok(Self {
            log_step,
            lattice,
            operators: Vec::new(),
            damping: 0.0,
        })
    }

    /// Returns a reference to the underlying log lattice.
    pub fn lattice(&self) -> &[f32] {
        &self.lattice
    }

    /// Adds an operator to the flow model.
    pub fn register_operator(&mut self, operator: RGOperator) {
        self.operators.push(operator);
    }

    /// Returns the registered operators.
    pub fn operators(&self) -> &[RGOperator] {
        &self.operators
    }

    /// Integrates the RG flow across the lattice using a forward Euler step.
    pub fn propagate(&self) -> RGFlowResult<RGFlowSolution> {
        if self.operators.is_empty() {
            let solution = RGFlowSolution {
                lattice: self.lattice.clone(),
                trajectories: Vec::new(),
            };
            emit_rg_mellin_flow_meta(self, &solution);
            return Ok(solution);
        }

        let depth = self.lattice.len();
        let mut trajectories = Vec::with_capacity(self.operators.len());

        for operator in &self.operators {
            let mut values = vec![0.0f32; depth];
            values[0] = operator.initial_coupling;
            let mut current = operator.initial_coupling;

            for (idx, &log_scale) in self.lattice.iter().enumerate().skip(1) {
                let beta = self.beta(operator, current, log_scale)?;
                current += self.log_step * beta;
                values[idx] = current;
            }

            trajectories.push(RGFlowTrajectory {
                name: operator.name.clone(),
                values,
            });
        }

        let solution = RGFlowSolution {
            lattice: self.lattice.clone(),
            trajectories,
        };
        emit_rg_mellin_flow_meta(self, &solution);
        Ok(solution)
    }

    /// Evaluates the beta function for the operator at the provided coupling.
    fn beta(&self, operator: &RGOperator, coupling: f32, log_scale: f32) -> RGFlowResult<f32> {
        // Canonical linear term: (Δ - d) * g. We interpret the ambient dimension as 4.
        let linear = (operator.scaling_dimension - 4.0) * coupling;
        let nonlinear = operator.nonlinear_feedback * coupling * coupling;
        let resonance = if let Some(res) = &operator.resonance {
            res.evaluate(log_scale)?.re
        } else {
            0.0
        };
        let damping = if self.damping > 0.0 {
            -self.damping * coupling
        } else {
            0.0
        };
        Ok(linear + nonlinear + resonance + damping)
    }

    /// Utility constructor emitting a lattice that mimics the "narrative depth" scale.
    pub fn narrative_depth(depth: usize, start: f32) -> RGFlowResult<Self> {
        // We pick log_step based on a golden-angle inspired spacing so successive
        // depths feel quasi-logarithmic but remain numerically stable.
        let golden = (5.0f32.sqrt() - 1.0) / 2.0;
        let log_step = golden / (depth.max(1) as f32);
        Self::new(start, log_step.max(1e-3), depth)
    }

    /// Exports the beta coefficients for external analysis.
    pub fn sample_beta(
        &self,
        coupling: f32,
        operator_index: usize,
        log_scale: f32,
    ) -> RGFlowResult<f32> {
        let operator = self
            .operators
            .get(operator_index)
            .ok_or(RGFlowError::InitialCouplings {
                expected: self.operators.len(),
                got: operator_index,
            })?;
        self.beta(operator, coupling, log_scale)
    }

    /// Synthesises a Mellin resonance matching a Gaussian spectral line.
    pub fn gaussian_resonance(
        log_start: f32,
        log_step: f32,
        len: usize,
        center: f32,
        bandwidth: f32,
        amplitude: f32,
    ) -> MellinResult<ResonanceProfile> {
        let sigma = bandwidth.max(1e-3);
        let grid = MellinLogGrid::from_function(log_start, log_step, len, |t| {
            let x = t - center;
            let envelope = (-0.5 * (x / sigma).powi(2)).exp();
            Complex32::new(amplitude * envelope, 0.0)
        })?;
        let pole = Complex32::new(2.0, 0.0);
        Ok(ResonanceProfile::new(grid, pole, Complex32::new(1.0, 0.0)))
    }

    /// Generates a simple breathing resonance anchored to the supplied Mellin pole.
    pub fn breathing_resonance(
        grid: Arc<MellinLogGrid>,
        pole: ComplexScalar,
        phase: f32,
        strength: f32,
    ) -> ResonanceProfile {
        let residue = Complex32::from_polar(strength, phase);
        ResonanceProfile::with_shared_grid(grid, pole, residue)
    }
}

fn finite_meta_f32(value: f32) -> f64 {
    if value.is_finite() {
        value as f64
    } else {
        0.0
    }
}

fn finite_meta_f64(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn emit_rg_mellin_flow_meta(model: &RGFlowModel, solution: &RGFlowSolution) {
    let operator_count = model.operators.len();
    let resonance_count = model
        .operators
        .iter()
        .filter(|operator| operator.resonance.is_some())
        .count();
    let nonlinear_count = model
        .operators
        .iter()
        .filter(|operator| operator.nonlinear_feedback.abs() > f32::EPSILON)
        .count();

    let mut scaling_min = f32::INFINITY;
    let mut scaling_max = f32::NEG_INFINITY;
    let mut scaling_sum = 0.0f32;
    let mut initial_abs_sum = 0.0f32;
    let mut nonlinear_abs_sum = 0.0f32;
    for operator in &model.operators {
        scaling_min = scaling_min.min(operator.scaling_dimension);
        scaling_max = scaling_max.max(operator.scaling_dimension);
        scaling_sum += operator.scaling_dimension;
        initial_abs_sum += operator.initial_coupling.abs();
        nonlinear_abs_sum += operator.nonlinear_feedback.abs();
    }
    let operator_den = operator_count.max(1) as f32;

    let mut final_abs_sum = 0.0f32;
    let mut final_abs_max = 0.0f32;
    let mut max_abs_coupling = 0.0f32;
    let mut drift_abs_sum = 0.0f32;
    let mut unstable_trajectory_count = 0usize;
    for trajectory in &solution.trajectories {
        let first = trajectory.values.first().copied().unwrap_or(0.0);
        let final_value = trajectory.values.last().copied().unwrap_or(0.0);
        if !trajectory.values.iter().all(|value| value.is_finite()) {
            unstable_trajectory_count += 1;
        }
        final_abs_sum += final_value.abs();
        final_abs_max = final_abs_max.max(final_value.abs());
        drift_abs_sum += (final_value - first).abs();
        for value in &trajectory.values {
            if value.is_finite() {
                max_abs_coupling = max_abs_coupling.max(value.abs());
            }
        }
    }
    let trajectory_den = solution.trajectories.len().max(1) as f32;

    emit_tensor_op(
        "rg_mellin_flow_propagate",
        &[solution.lattice.len(), operator_count.max(1)],
        &[
            solution.trajectories.len().max(1),
            solution.lattice.len().max(1),
        ],
    );
    emit_tensor_op_meta("rg_mellin_flow_propagate", || {
        let mut payload = serde_json::Map::new();
        payload.insert("backend".into(), "cpu".into());
        payload.insert("requested_backend".into(), "auto".into());
        payload.insert("kind".into(), "st_core_rg_mellin_flow_propagate".into());
        payload.insert("lattice_len".into(), solution.lattice.len().into());
        payload.insert("operator_count".into(), operator_count.into());
        payload.insert(
            "trajectory_count".into(),
            solution.trajectories.len().into(),
        );
        payload.insert("empty_operators".into(), model.operators.is_empty().into());
        payload.insert("log_step".into(), finite_meta_f32(model.log_step).into());
        payload.insert("damping".into(), finite_meta_f32(model.damping).into());
        payload.insert("resonance_count".into(), resonance_count.into());
        payload.insert("nonlinear_count".into(), nonlinear_count.into());
        payload.insert(
            "scaling_dimension_min".into(),
            finite_meta_f32(if operator_count == 0 {
                0.0
            } else {
                scaling_min
            })
            .into(),
        );
        payload.insert(
            "scaling_dimension_max".into(),
            finite_meta_f32(if operator_count == 0 {
                0.0
            } else {
                scaling_max
            })
            .into(),
        );
        payload.insert(
            "scaling_dimension_mean".into(),
            finite_meta_f32(scaling_sum / operator_den).into(),
        );
        payload.insert(
            "initial_abs_mean".into(),
            finite_meta_f32(initial_abs_sum / operator_den).into(),
        );
        payload.insert(
            "nonlinear_abs_mean".into(),
            finite_meta_f32(nonlinear_abs_sum / operator_den).into(),
        );
        payload.insert(
            "final_abs_mean".into(),
            finite_meta_f32(final_abs_sum / trajectory_den).into(),
        );
        payload.insert(
            "final_abs_max".into(),
            finite_meta_f32(final_abs_max).into(),
        );
        payload.insert(
            "max_abs_coupling".into(),
            finite_meta_f32(max_abs_coupling).into(),
        );
        payload.insert(
            "drift_abs_mean".into(),
            finite_meta_f32(drift_abs_sum / trajectory_den).into(),
        );
        payload.insert(
            "unstable_trajectory_count".into(),
            unstable_trajectory_count.into(),
        );
        payload.insert(
            "lattice_start".into(),
            finite_meta_f32(solution.lattice.first().copied().unwrap_or(0.0)).into(),
        );
        payload.insert(
            "lattice_end".into(),
            finite_meta_f32(solution.lattice.last().copied().unwrap_or(0.0)).into(),
        );
        payload.insert(
            "lattice_span".into(),
            finite_meta_f64(
                solution
                    .lattice
                    .last()
                    .zip(solution.lattice.first())
                    .map(|(last, first)| (*last - *first) as f64)
                    .unwrap_or(0.0),
            )
            .into(),
        );
        payload.into()
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;
    use std::sync::{Arc, Mutex};

    fn sample_model() -> RGFlowModel {
        let mut model = RGFlowModel::new(0.0, 0.5, 6).unwrap();
        model.register_operator(RGOperator::new("scalar", 3.0, 1.0));
        model
    }

    #[test]
    fn integrates_linear_flow() {
        let model = sample_model();
        let solution = model.propagate().unwrap();
        let traj = solution.trajectory("scalar").unwrap();
        // With Δ=3, the beta function is -(4-3) g = -g. Forward Euler yields a
        // geometric decay with ratio (1 - log_step).
        let expected = [1.0, 0.5, 0.25, 0.125];
        for (idx, &value) in expected.iter().enumerate() {
            let actual = traj.at(idx).unwrap();
            assert!(
                (actual - value).abs() < 1e-6,
                "idx={idx} actual={actual} expected={value}"
            );
        }
    }

    #[test]
    fn resonance_feedback_injects_energy() {
        let log_start = 0.0;
        let log_step = 0.25;
        let len = 8;
        let grid = Arc::new(
            MellinLogGrid::from_function(log_start, log_step, len, |t| {
                Complex32::new((t * PI).cos(), (t * PI).sin())
            })
            .unwrap(),
        );
        let resonance = ResonanceProfile::with_shared_grid(
            grid.clone(),
            Complex32::new(1.5, 0.0),
            Complex32::new(0.25, 0.0),
        );

        let mut model = RGFlowModel::new(log_start, log_step, len).unwrap();
        model.register_operator(RGOperator::new("flow", 4.0, 0.1).with_resonance(resonance));
        let solution = model.propagate().unwrap();
        let traj = solution.trajectory("flow").unwrap();
        // Resonance pushes the coupling upwards at intermediate scales.
        assert!(traj.values.iter().any(|&v| v > 0.1));
    }

    #[test]
    fn propagate_emits_backend_meta() {
        let _lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut model = RGFlowModel::new(0.0, 0.25, 5).unwrap();
        model.damping = 0.1;
        model.register_operator(RGOperator::new("scalar", 3.0, 1.0));
        model
            .register_operator(RGOperator::new("nonlinear", 4.2, 0.2).with_nonlinear_feedback(0.5));
        let solution = model.propagate().unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        assert_eq!(solution.lattice().len(), 5);
        assert_eq!(solution.iter().count(), 2);

        let events = events.lock().unwrap();
        let meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "rg_mellin_flow_propagate"
                    && data["kind"] == "st_core_rg_mellin_flow_propagate"
                    && data["lattice_len"] == 5
                    && data["operator_count"] == 2
            })
            .expect("rg_mellin_flow_propagate metadata event");
        assert_eq!(meta.1["backend"], "cpu");
        assert_eq!(meta.1["requested_backend"], "auto");
        assert_eq!(meta.1["trajectory_count"], 2);
        assert_eq!(meta.1["empty_operators"], false);
        assert_eq!(meta.1["nonlinear_count"], 1);
        assert_eq!(meta.1["resonance_count"], 0);
        assert_eq!(meta.1["unstable_trajectory_count"], 0);
        assert!(meta.1["max_abs_coupling"].as_f64().unwrap_or(0.0) > 0.0);
        assert!(meta.1["drift_abs_mean"].as_f64().unwrap_or(0.0) > 0.0);
        assert!(meta.1["lattice_span"].as_f64().unwrap_or(0.0) > 0.0);
    }
}
