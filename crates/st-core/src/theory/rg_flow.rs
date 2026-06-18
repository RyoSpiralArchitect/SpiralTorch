// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Renormalisation-group (RG) flow DSLs that live natively on the Z-space log lattice.
//!
//! The primitives exposed here wire Mellin log grids into narrative “scale depth”
//! coordinates.  Each RG step is represented as a node on the same logarithmic
//! lattice used by `st-frac::mellin`, which means the resulting flows can be projected
//! back into Z-space without lossy conversions.  Narrative depth becomes the scale
//! parameter: follow the flow forward to coarse grain a story, or backwards to recover
//! the ultraviolet details.

use st_frac::mellin::MellinLogGrid;
use st_frac::mellin_types::{ComplexScalar, MellinError};
use st_frac::zspace::{
    evaluate_weighted_series, evaluate_weighted_series_many, mellin_log_lattice_prefactor,
};
use st_tensor::{emit_tensor_op, emit_tensor_op_meta};
use thiserror::Error;

/// Result type produced by the RG flow helpers.
pub type RgFlowResult<T> = Result<T, RgFlowError>;

/// Error produced while building or interrogating an RG flow lattice.
#[derive(Debug, Error)]
pub enum RgFlowError {
    #[error("log_step must be positive and finite")]
    InvalidLogStep,
    #[error("log_start must be finite")]
    InvalidLogStart,
    #[error("RG flow requires at least one lattice step")]
    EmptyLattice,
    #[error("at least one coupling is required")]
    EmptyCouplings,
    #[error("beta function returned {got} entries, expected {expected}")]
    BetaDimension { expected: usize, got: usize },
    #[error("Mellin projection unavailable for this flow")]
    MissingMellinSeries,
    #[error(transparent)]
    Mellin(#[from] MellinError),
}

/// Node living on the RG lattice.
#[derive(Clone, Debug)]
pub struct RgFlowNode {
    log_scale: f32,
    scale: f32,
    couplings: Vec<f32>,
    beta: Vec<f32>,
}

impl RgFlowNode {
    /// Returns the logarithmic scale associated with this node.
    pub fn log_scale(&self) -> f32 {
        self.log_scale
    }

    /// Returns the physical scale (exp of the log scale).
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Alias exposing the same quantity as the “narrative depth”.
    pub fn narrative_depth(&self) -> f32 {
        self.scale
    }

    /// Returns the couplings attached to this lattice location.
    pub fn couplings(&self) -> &[f32] {
        &self.couplings
    }

    /// Returns the beta vector evaluated at this location.
    pub fn beta(&self) -> &[f32] {
        &self.beta
    }

    /// Magnitude of the beta function which is useful to detect fixed points.
    pub fn beta_norm(&self) -> f32 {
        self.beta.iter().map(|v| v * v).sum::<f32>().sqrt()
    }
}

/// Fixed-point summary extracted from a lattice.
#[derive(Clone, Debug)]
pub struct ScaleFixedPoint {
    log_scale: f32,
    scale: f32,
    couplings: Vec<f32>,
}

impl ScaleFixedPoint {
    /// Logarithmic scale where the beta vanished.
    pub fn log_scale(&self) -> f32 {
        self.log_scale
    }

    /// Physical scale of the fixed point.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Narrative depth (identical to the physical scale in this DSL).
    pub fn narrative_depth(&self) -> f32 {
        self.scale
    }

    /// Couplings frozen at this fixed point.
    pub fn couplings(&self) -> &[f32] {
        &self.couplings
    }
}

/// Log-uniform lattice representing the RG flow of a set of couplings.
#[derive(Clone, Debug)]
pub struct RgFlowLattice {
    log_start: f32,
    log_step: f32,
    nodes: Vec<RgFlowNode>,
    mellin_series: Option<Vec<ComplexScalar>>,
}

impl RgFlowLattice {
    /// Construct a lattice directly from scalar parameters.
    pub fn new_with_beta<F>(
        log_start: f32,
        log_step: f32,
        steps: usize,
        root_couplings: Vec<f32>,
        mut beta: F,
    ) -> RgFlowResult<Self>
    where
        F: FnMut(f32, &[f32]) -> Vec<f32>,
    {
        if !log_start.is_finite() {
            return Err(RgFlowError::InvalidLogStart);
        }
        if !(log_step.is_finite() && log_step > 0.0) {
            return Err(RgFlowError::InvalidLogStep);
        }
        if steps == 0 {
            return Err(RgFlowError::EmptyLattice);
        }
        if root_couplings.is_empty() {
            return Err(RgFlowError::EmptyCouplings);
        }

        let mut nodes = Vec::with_capacity(steps);
        let mut current = root_couplings;
        let mut log_scale = log_start;

        for step in 0..steps {
            let beta_vec = beta(log_scale, &current);
            if beta_vec.len() != current.len() {
                return Err(RgFlowError::BetaDimension {
                    expected: current.len(),
                    got: beta_vec.len(),
                });
            }
            let scale = log_scale.exp();
            nodes.push(RgFlowNode {
                log_scale,
                scale,
                couplings: current.clone(),
                beta: beta_vec.clone(),
            });

            if step + 1 < steps {
                for (value, delta) in current.iter_mut().zip(beta_vec.iter()) {
                    *value += log_step * *delta;
                }
                log_scale += log_step;
            }
        }

        Ok(Self {
            log_start,
            log_step,
            nodes,
            mellin_series: None,
        })
    }

    /// Bind an existing Mellin log grid to the RG lattice.
    pub fn from_mellin_grid<F>(
        grid: &MellinLogGrid,
        root_couplings: Vec<f32>,
        beta: F,
    ) -> RgFlowResult<Self>
    where
        F: FnMut(f32, &[f32]) -> Vec<f32>,
    {
        let mut lattice = Self::new_with_beta(
            grid.log_start(),
            grid.log_step(),
            grid.len(),
            root_couplings,
            beta,
        )?;
        lattice.mellin_series = Some(grid.weighted_series()?);
        Ok(lattice)
    }

    /// Returns the nodes hosted by this lattice.
    pub fn nodes(&self) -> &[RgFlowNode] {
        &self.nodes
    }

    /// Exposes the stored Mellin weighted series when available.
    pub fn mellin_series(&self) -> Option<&[ComplexScalar]> {
        self.mellin_series.as_deref()
    }

    /// Evaluate the Mellin transform of the flow on the bound lattice.
    pub fn evaluate_mellin(&self, s: ComplexScalar) -> RgFlowResult<ComplexScalar> {
        let weighted = self
            .mellin_series
            .as_ref()
            .ok_or(RgFlowError::MissingMellinSeries)?;
        let (prefactor, z) = mellin_log_lattice_prefactor(self.log_start, self.log_step, s)?;
        let series = evaluate_weighted_series(weighted, z)?;
        Ok(prefactor * series)
    }

    /// Batch variant of [`evaluate_mellin`].
    pub fn evaluate_mellin_many(
        &self,
        s_values: &[ComplexScalar],
    ) -> RgFlowResult<Vec<ComplexScalar>> {
        if s_values.is_empty() {
            return Ok(Vec::new());
        }
        let weighted = self
            .mellin_series
            .as_ref()
            .ok_or(RgFlowError::MissingMellinSeries)?;
        let mut prefactors = Vec::with_capacity(s_values.len());
        let mut z_values = Vec::with_capacity(s_values.len());
        for &s in s_values {
            let (prefactor, z) = mellin_log_lattice_prefactor(self.log_start, self.log_step, s)?;
            prefactors.push(prefactor);
            z_values.push(z);
        }
        let series = evaluate_weighted_series_many(weighted, &z_values)?;
        Ok(series
            .into_iter()
            .zip(prefactors)
            .map(|(series, prefactor)| prefactor * series)
            .collect())
    }

    /// Returns the lattice nodes that behave as approximate fixed points.
    pub fn fixed_points(&self, tolerance: f32) -> Vec<ScaleFixedPoint> {
        let fixed_points: Vec<ScaleFixedPoint> = self
            .nodes
            .iter()
            .filter(|node| node.beta_norm() <= tolerance)
            .map(|node| ScaleFixedPoint {
                log_scale: node.log_scale,
                scale: node.scale,
                couplings: node.couplings.clone(),
            })
            .collect();
        emit_fixed_points_meta(self, tolerance, &fixed_points);
        fixed_points
    }

    /// Returns the native log-step used by the lattice.
    pub fn log_step(&self) -> f32 {
        self.log_step
    }

    /// Returns the starting log-coordinate.
    pub fn log_start(&self) -> f32 {
        self.log_start
    }

    /// Number of nodes stored in the lattice.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True when no nodes are present.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

fn emit_fixed_points_meta(
    lattice: &RgFlowLattice,
    tolerance: f32,
    fixed_points: &[ScaleFixedPoint],
) {
    let coupling_dim = lattice
        .nodes
        .first()
        .map(|node| node.couplings.len())
        .unwrap_or(0);
    let mut beta_total = 0.0f64;
    let mut beta_min = f32::INFINITY;
    let mut beta_max = 0.0f32;
    for norm in lattice.nodes.iter().map(RgFlowNode::beta_norm) {
        if norm.is_finite() {
            beta_total += norm as f64;
            beta_min = beta_min.min(norm);
            beta_max = beta_max.max(norm);
        }
    }
    let mut scale_min = f32::INFINITY;
    let mut scale_max = 0.0f32;
    let mut coupling_abs_total = 0.0f64;
    let mut coupling_values = 0usize;
    for point in fixed_points {
        scale_min = scale_min.min(point.scale);
        scale_max = scale_max.max(point.scale);
        for coupling in &point.couplings {
            if coupling.is_finite() {
                coupling_abs_total += coupling.abs() as f64;
                coupling_values += 1;
            }
        }
    }
    emit_tensor_op(
        "rg_flow_fixed_points",
        &[lattice.nodes.len(), coupling_dim],
        &[fixed_points.len(), coupling_dim.max(1)],
    );
    emit_tensor_op_meta("rg_flow_fixed_points", || {
        serde_json::json!({
            "backend": "cpu",
            "requested_backend": "auto",
            "kind": "st_core_rg_flow_fixed_points",
            "nodes": lattice.nodes.len(),
            "coupling_dim": coupling_dim,
            "fixed_points": fixed_points.len(),
            "tolerance": if tolerance.is_finite() { tolerance } else { 0.0 },
            "log_start": lattice.log_start,
            "log_step": lattice.log_step,
            "has_mellin_series": lattice.mellin_series.is_some(),
            "beta_norm_min": if beta_min.is_finite() { beta_min } else { 0.0 },
            "beta_norm_mean": if lattice.nodes.is_empty() {
                0.0
            } else {
                beta_total / lattice.nodes.len() as f64
            },
            "beta_norm_max": beta_max,
            "fixed_scale_min": if scale_min.is_finite() { scale_min } else { 0.0 },
            "fixed_scale_max": scale_max,
            "mean_fixed_coupling_abs": if coupling_values == 0 {
                0.0
            } else {
                coupling_abs_total / coupling_values as f64
            },
        })
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    fn beta_profile(log_scale: f32, couplings: &[f32]) -> Vec<f32> {
        let target = log_scale.tanh();
        couplings.iter().map(|&g| -0.35 * (g - target)).collect()
    }

    #[test]
    fn builds_flow_from_mellin_grid() {
        let grid = MellinLogGrid::from_function(-2.0, 0.5, 8, |x| {
            let amp = x.exp();
            ComplexScalar::new(amp, amp.sin())
        })
        .unwrap();
        let flow = RgFlowLattice::from_mellin_grid(&grid, vec![0.25, -0.1], beta_profile).unwrap();
        assert_eq!(flow.len(), grid.len());
        assert!(flow.mellin_series().is_some());
        let nodes = flow.nodes();
        assert_eq!(nodes.first().unwrap().couplings().len(), 2);
        assert!(nodes[0].beta_norm() > 0.0);
    }

    #[test]
    fn detects_fixed_point_and_projects() {
        let grid = MellinLogGrid::from_function(-1.5, 0.3, 10, |x| {
            let amp = (x * 1.3).sin();
            ComplexScalar::new(amp, amp.cos())
        })
        .unwrap();
        let flow = RgFlowLattice::from_mellin_grid(&grid, vec![0.1, 0.05], beta_profile).unwrap();
        let min_norm = flow
            .nodes()
            .iter()
            .map(RgFlowNode::beta_norm)
            .fold(f32::INFINITY, f32::min);
        assert!(min_norm < 0.1, "min_norm={min_norm}");
        let s = ComplexScalar::new(1.1, 0.4);
        let value = flow.evaluate_mellin(s).unwrap();
        assert!(value.norm() > 0.0);
        let values = flow
            .evaluate_mellin_many(&[s, ComplexScalar::new(0.8, -0.2)])
            .unwrap();
        assert_eq!(values.len(), 2);
        let diff = (values[0] - value).norm();
        assert!(diff < 1e-5, "diff={diff}");
    }

    #[test]
    fn fixed_points_emit_backend_meta() {
        let _lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let flow = RgFlowLattice::new_with_beta(0.0, 0.25, 4, vec![0.0, 0.0], |_log, couplings| {
            vec![0.0; couplings.len()]
        })
        .unwrap();
        let fixed = flow.fixed_points(1.0e-4);
        st_tensor::set_tensor_op_meta_observer(previous);

        assert_eq!(fixed.len(), 4);
        let events = events.lock().unwrap();
        let meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "rg_flow_fixed_points"
                    && data["nodes"] == 4
                    && data["fixed_points"] == 4
            })
            .expect("rg_flow_fixed_points metadata event");
        assert_eq!(meta.1["backend"], "cpu");
        assert_eq!(meta.1["kind"], "st_core_rg_flow_fixed_points");
        assert_eq!(meta.1["coupling_dim"], 2);
        assert_eq!(meta.1["has_mellin_series"], false);
        assert_eq!(meta.1["beta_norm_max"], 0.0);
    }
}
