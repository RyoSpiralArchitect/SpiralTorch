// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![allow(clippy::needless_update)]

//! Microlocal boundary gauges and the Z-space conductor used to fuse their
//! output.  The previous revision shipped in a partially duplicated state that
//! could no longer compile.  This module rebuilds the tooling from first
//! principles with an explicit focus on the behaviours exercised by the public
//! unit tests: interface detection, orientation reconstruction, Z-lift
//! projection, and policy-controlled fusion.

use crate::telemetry::hub::{SoftlogicEllipticSample, SoftlogicZFeedback};
use crate::theory::microlocal_bank::GaugeBank;
use crate::theory::zpulse::{
    ZConductor, ZConductorCfg, ZEmitter, ZPulse, ZRegistry, ZScale, ZSource, ZSupport,
};
use crate::util::math::LeechProjector;
use ndarray::{indices, ArrayD, ArrayViewD, Dimension, IxDyn};
use rustc_hash::FxHashMap;
use statrs::function::gamma::gamma;
use std::collections::VecDeque;
use std::f32::consts::{PI, TAU};
use std::fmt;
use std::sync::{Arc, Mutex};

/// Result of running an [`InterfaceGauge`] on a binary phase field.
#[derive(Debug, Clone)]
pub struct InterfaceSignature {
    pub r_machine: ArrayD<f32>,
    pub raw_density: ArrayD<f32>,
    pub perimeter_density: ArrayD<f32>,
    pub mean_curvature: ArrayD<f32>,
    pub signed_mean_curvature: Option<ArrayD<f32>>,
    pub orientation: Option<ArrayD<f32>>,
    pub kappa_d: f32,
    pub radius: isize,
    pub physical_radius: f32,
}

impl InterfaceSignature {
    /// Returns `true` when any interface cell was detected.
    pub fn has_interface(&self) -> bool {
        self.r_machine.iter().any(|v| *v > 0.0)
    }
}

/// Microlocal interface detector approximating the BV blow-up.
#[derive(Debug, Clone)]
pub struct InterfaceGauge {
    grid_spacing: f32,
    physical_radius: f32,
    threshold: f32,
}

impl InterfaceGauge {
    pub fn new(grid_spacing: f32, physical_radius: f32) -> Self {
        Self {
            grid_spacing: grid_spacing.max(f32::EPSILON),
            physical_radius: physical_radius.max(f32::EPSILON),
            threshold: 0.25,
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold.max(0.0);
        self
    }

    pub fn physical_radius(&self) -> f32 {
        self.physical_radius
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    pub fn scale_threshold(&mut self, scale: f32) {
        if scale > 0.0 {
            self.threshold = (self.threshold * scale).max(0.0);
        }
    }

    pub fn analyze(&self, mask: &ArrayD<f32>) -> InterfaceSignature {
        self.analyze_with_radius(mask, None, self.physical_radius)
    }

    pub fn analyze_with_label(
        &self,
        mask: &ArrayD<f32>,
        c_prime: Option<&ArrayD<f32>>,
    ) -> InterfaceSignature {
        self.analyze_with_radius(mask, c_prime, self.physical_radius)
    }

    pub fn analyze_with_radius(
        &self,
        mask: &ArrayD<f32>,
        c_prime: Option<&ArrayD<f32>>,
        physical_radius: f32,
    ) -> InterfaceSignature {
        let radius_steps = radius_in_steps(physical_radius, self.grid_spacing);
        self.analyze_with_steps(mask, c_prime, radius_steps, physical_radius)
    }

    pub fn analyze_multiradius(
        &self,
        mask: &ArrayD<f32>,
        c_prime: Option<&ArrayD<f32>>,
        radii: &[f32],
    ) -> Vec<InterfaceSignature> {
        radii
            .iter()
            .map(|&radius| self.analyze_with_radius(mask, c_prime, radius))
            .collect()
    }

    fn analyze_with_steps(
        &self,
        mask: &ArrayD<f32>,
        c_prime: Option<&ArrayD<f32>>,
        radius_steps: isize,
        physical_radius: f32,
    ) -> InterfaceSignature {
        assert!(mask.ndim() > 0, "mask must have positive dimension");
        if let Some(label) = c_prime {
            assert_eq!(label.shape(), mask.shape(), "c′ must match mask shape");
        }

        let radius_steps = radius_steps.max(1);
        let dim = mask.ndim();
        let raw_dim = mask.raw_dim();
        let mut r_machine = ArrayD::<f32>::zeros(raw_dim.clone());
        let mut raw_density = ArrayD::<f32>::zeros(raw_dim.clone());
        let mut perimeter_density = ArrayD::<f32>::zeros(raw_dim.clone());
        let mut mean_curvature = ArrayD::<f32>::zeros(raw_dim.clone());
        let mut signed_mean_curvature = c_prime.map(|_| ArrayD::<f32>::zeros(raw_dim.clone()));
        let mut orientation = c_prime.map(|_| {
            let mut orient_shape = Vec::with_capacity(dim + 1);
            orient_shape.push(dim);
            orient_shape.extend_from_slice(mask.shape());
            ArrayD::<f32>::zeros(IxDyn(&orient_shape))
        });

        let kappa_d = unit_sphere_area(dim);
        let threshold = self.threshold;

        for idx in indices(raw_dim.clone()) {
            let idx_dyn = IxDyn(idx.slice());
            let center = mask[&idx_dyn];
            let mut gradient = vec![0.0f32; dim];
            let mut variation = 0.0f32;
            let mut interface = false;

            for axis in 0..dim {
                // forward difference
                if idx[axis] + 1 < mask.shape()[axis] {
                    let mut forward_idx = idx.slice().to_vec();
                    forward_idx[axis] += 1;
                    let forward = mask[IxDyn(&forward_idx)];
                    let delta = forward - center;
                    variation = variation.max(delta.abs());
                    gradient[axis] += delta;
                    if delta.abs() >= threshold {
                        interface = true;
                    }
                }
                // backward difference
                if idx[axis] > 0 {
                    let mut back_idx = idx.slice().to_vec();
                    back_idx[axis] -= 1;
                    let backward = mask[IxDyn(&back_idx)];
                    let delta = center - backward;
                    variation = variation.max(delta.abs());
                    gradient[axis] -= delta;
                    if delta.abs() >= threshold {
                        interface = true;
                    }
                }
            }

            raw_density[&idx_dyn] = variation;
            if interface {
                r_machine[&idx_dyn] = 1.0;
                perimeter_density[&idx_dyn] = kappa_d;
            }

            let laplacian = discrete_laplacian(mask.view(), &idx_dyn);
            mean_curvature[&idx_dyn] = (laplacian.abs() * 0.25).min(kappa_d);

            if let Some(target) = signed_mean_curvature.as_mut() {
                let sign = c_prime.map(|label| label[&idx_dyn].signum()).unwrap_or(0.0);
                target[&idx_dyn] = laplacian * sign;
            }

            if let Some(orient) = orientation.as_mut() {
                let mut magnitude = 0.0f32;
                for (axis, component) in gradient.iter().enumerate() {
                    magnitude += component * component;
                    let mut orient_idx = Vec::with_capacity(dim + 1);
                    orient_idx.push(axis);
                    orient_idx.extend_from_slice(idx.slice());
                    orient[IxDyn(&orient_idx)] = *component;
                }
                if magnitude > 0.0 {
                    let scale = magnitude.sqrt().max(1e-6);
                    for axis in 0..dim {
                        let mut orient_idx = Vec::with_capacity(dim + 1);
                        orient_idx.push(axis);
                        orient_idx.extend_from_slice(idx.slice());
                        let value = orient[IxDyn(&orient_idx)] / scale;
                        orient[IxDyn(&orient_idx)] = value;
                    }
                }
            }
        }

        InterfaceSignature {
            r_machine,
            raw_density,
            perimeter_density,
            mean_curvature,
            signed_mean_curvature,
            orientation,
            kappa_d,
            radius: radius_steps,
            physical_radius,
        }
    }
}

/// Registry holding named [`InterfaceGauge`] instances.
#[derive(Clone, Debug)]
pub struct MicrolocalGaugeBank {
    inner: GaugeBank<InterfaceGauge>,
}

impl Default for MicrolocalGaugeBank {
    fn default() -> Self {
        Self {
            inner: GaugeBank::new(),
        }
    }
}

impl MicrolocalGaugeBank {
    /// Creates an empty gauge registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a gauge with the provided identifier. Returns `false` if an
    /// entry with the same identifier already exists.
    pub fn register(&mut self, id: impl Into<String>, gauge: InterfaceGauge) -> bool {
        self.inner.register(id, gauge)
    }

    /// Convenience for registering and returning `self` in a builder-style
    /// workflow.
    pub fn with_registered(mut self, id: impl Into<String>, gauge: InterfaceGauge) -> Self {
        let _ = self.register(id, gauge);
        self
    }

    /// Returns an immutable reference to the named gauge.
    pub fn get(&self, id: &str) -> Option<&InterfaceGauge> {
        self.inner.get(id)
    }

    /// Returns a mutable reference to the named gauge.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut InterfaceGauge> {
        self.inner.get_mut(id)
    }

    /// Removes a gauge from the registry and returns it if present.
    pub fn remove(&mut self, id: &str) -> Option<InterfaceGauge> {
        self.inner.remove(id)
    }

    /// Iterates over registered identifiers and gauges in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &InterfaceGauge)> {
        self.inner.iter()
    }

    /// Iterates over registered identifiers and mutable gauges in insertion order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&str, &mut InterfaceGauge)> {
        self.inner.iter_mut()
    }

    /// Returns the registered identifiers.
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.inner.ids()
    }

    /// Number of registered gauges.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` when no gauges are registered.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Clones the registered gauges into a vector preserving insertion order.
    pub fn to_vec(&self) -> Vec<InterfaceGauge> {
        self.inner.to_vec()
    }

    /// Consumes the bank and returns the gauges in insertion order.
    pub fn into_vec(self) -> Vec<InterfaceGauge> {
        self.inner.into_vec()
    }

    /// Consumes the bank and returns identifier-gauge pairs in insertion order.
    pub fn into_entries(self) -> Vec<(Arc<str>, InterfaceGauge)> {
        self.inner.into_entries()
    }

    /// Runs every registered gauge against the provided mask and optional
    /// orientation labels, returning the resulting signatures keyed by id.
    pub fn analyze_all(
        &self,
        mask: &ArrayD<f32>,
        c_prime: Option<&ArrayD<f32>>,
    ) -> FxHashMap<Arc<str>, InterfaceSignature> {
        let mut results = FxHashMap::default();
        for (id, gauge) in self.inner.entries() {
            let signature = gauge.analyze_with_label(mask, c_prime);
            results.insert(Arc::clone(id), signature);
        }
        results
    }
}

impl IntoIterator for MicrolocalGaugeBank {
    type Item = InterfaceGauge;
    type IntoIter = <GaugeBank<InterfaceGauge> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

fn discrete_laplacian(mask: ArrayViewD<f32>, idx: &IxDyn) -> f32 {
    let mut lap = 0.0f32;
    let dim = mask.ndim();
    let shape = mask.shape().to_vec();
    let coords = idx.slice();
    let center = mask[idx.clone()];
    for axis in 0..dim {
        let mut forward = center;
        let mut backward = center;
        let mut has_forward = false;
        let mut has_backward = false;
        if coords[axis] + 1 < shape[axis] {
            let mut f_idx = coords.to_vec();
            f_idx[axis] += 1;
            forward = mask[IxDyn(&f_idx)];
            has_forward = true;
        }
        if coords[axis] > 0 {
            let mut b_idx = coords.to_vec();
            b_idx[axis] -= 1;
            backward = mask[IxDyn(&b_idx)];
            has_backward = true;
        }
        if has_forward && has_backward {
            lap += forward + backward - 2.0 * center;
        }
    }
    lap
}

fn radius_in_steps(radius: f32, spacing: f32) -> isize {
    (radius / spacing).round() as isize
}

fn unit_sphere_area(dim: usize) -> f32 {
    if dim == 0 {
        return 0.0;
    }
    let dim_f = dim as f64;
    let numerator = 2.0 * std::f64::consts::PI.powf(dim_f / 2.0);
    let denominator = gamma(dim_f / 2.0);
    (numerator / denominator) as f32
}

/// Positive-curvature warp that remaps microlocal orientations onto an elliptic
/// Z-frame.
#[derive(Clone, Debug)]
pub struct EllipticWarp {
    curvature_radius: f32,
    sheet_count: usize,
    spin_harmonics: usize,
}

impl EllipticWarp {
    /// Creates a warp anchored to the provided curvature radius.
    pub fn new(curvature_radius: f32) -> Self {
        let radius = curvature_radius.max(1e-6);
        Self {
            curvature_radius: radius,
            sheet_count: 2,
            spin_harmonics: 1,
        }
    }

    /// Configures the number of discrete sheets representing the χ axis.
    pub fn with_sheet_count(mut self, sheet_count: usize) -> Self {
        self.sheet_count = sheet_count.max(1);
        self
    }

    /// Configures the number of spin harmonics applied while computing ν.
    pub fn with_spin_harmonics(mut self, harmonics: usize) -> Self {
        self.spin_harmonics = harmonics.max(1);
        self
    }

    /// Returns the curvature radius associated with the warp.
    pub fn curvature_radius(&self) -> f32 {
        self.curvature_radius
    }

    /// Returns the number of χ sheets encoded by the warp.
    pub fn sheet_count(&self) -> usize {
        self.sheet_count
    }

    /// Maximum geodesic radius reachable on the warp.
    pub fn max_geodesic(&self) -> f32 {
        self.curvature_radius * PI
    }

    /// Maps an orientation vector to elliptic telemetry describing the warped
    /// coordinates. Returns `None` when the orientation is degenerate.
    pub fn map_orientation(&self, orientation: &[f32]) -> Option<EllipticTelemetry> {
        if orientation.is_empty() {
            return None;
        }

        let mut dir = [0.0f32; 3];
        for (dst, &value) in dir.iter_mut().zip(orientation.iter()).take(3) {
            *dst = value;
        }
        let norm = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        if !norm.is_finite() || norm <= 1e-6 {
            return None;
        }
        for component in dir.iter_mut() {
            *component /= norm;
        }

        let polar = dir[2].clamp(-1.0, 1.0).acos();
        let geodesic_radius = polar * self.curvature_radius;
        let azimuth = dir[1].atan2(dir[0]);
        let mut spin_alignment = (azimuth / PI).clamp(-1.0, 1.0);
        if self.spin_harmonics > 1 {
            spin_alignment = (spin_alignment * self.spin_harmonics as f32).sin();
        }

        let normalized = ((azimuth + PI) / TAU).rem_euclid(1.0);
        let sheet_f = normalized * self.sheet_count as f32;
        let mut sheet_index = sheet_f.floor() as usize;
        if sheet_index >= self.sheet_count {
            sheet_index = self.sheet_count - 1;
        }
        let sheet_position = if self.sheet_count <= 1 {
            0.0
        } else {
            (sheet_f / self.sheet_count as f32).clamp(0.0, 1.0)
        };

        Some(EllipticTelemetry {
            curvature_radius: self.curvature_radius,
            geodesic_radius,
            spin_alignment,
            sheet_index,
            sheet_position,
            normal_bias: dir[2].clamp(-1.0, 1.0),
            sheet_count: self.sheet_count,
        })
    }
}

/// Telemetry describing an elliptic Z-frame projection.
#[derive(Clone, Debug, Default)]
pub struct EllipticTelemetry {
    pub curvature_radius: f32,
    pub geodesic_radius: f32,
    pub spin_alignment: f32,
    pub sheet_index: usize,
    pub sheet_position: f32,
    pub normal_bias: f32,
    pub sheet_count: usize,
}

impl EllipticTelemetry {
    /// Normalised geodesic radius within \([0, 1]\).
    pub fn normalized_radius(&self) -> f32 {
        if self.curvature_radius <= 0.0 {
            0.0
        } else {
            (self.geodesic_radius / (self.curvature_radius * PI)).clamp(0.0, 1.0)
        }
    }

    /// Interpolates two telemetry samples.
    pub fn lerp(&self, other: &EllipticTelemetry, t: f32) -> EllipticTelemetry {
        let clamped = t.clamp(0.0, 1.0);
        let sheet_count = self.sheet_count.max(other.sheet_count).max(1);
        let sheet_position =
            (lerp(self.sheet_position, other.sheet_position, clamped)).clamp(0.0, 1.0);
        let sheet_index = ((sheet_position * sheet_count as f32).round() as usize)
            .min(sheet_count.saturating_sub(1));
        EllipticTelemetry {
            curvature_radius: lerp(self.curvature_radius, other.curvature_radius, clamped)
                .max(1e-6),
            geodesic_radius: lerp(self.geodesic_radius, other.geodesic_radius, clamped).max(0.0),
            spin_alignment: lerp(self.spin_alignment, other.spin_alignment, clamped)
                .clamp(-1.0, 1.0),
            sheet_index,
            sheet_position,
            normal_bias: lerp(self.normal_bias, other.normal_bias, clamped).clamp(-1.0, 1.0),
            sheet_count,
        }
    }

    /// Returns event tags that summarise the elliptic telemetry.
    pub fn event_tags(&self) -> [String; 3] {
        [
            format!("elliptic.sheet:{:02}", self.sheet_index),
            format!("elliptic.radius:{:.4}", self.normalized_radius()),
            format!("elliptic.spin:{:.3}", self.spin_alignment),
        ]
    }
}

impl From<&EllipticTelemetry> for SoftlogicEllipticSample {
    fn from(telemetry: &EllipticTelemetry) -> Self {
        SoftlogicEllipticSample {
            curvature_radius: telemetry.curvature_radius,
            geodesic_radius: telemetry.geodesic_radius,
            normalized_radius: telemetry.normalized_radius(),
            spin_alignment: telemetry.spin_alignment,
            sheet_index: telemetry.sheet_index as u32,
            sheet_position: telemetry.sheet_position,
            normal_bias: telemetry.normal_bias,
            sheet_count: telemetry.sheet_count as u32,
        }
    }
}

#[derive(Default)]
struct EllipticAccumulator {
    curvature_sum: f32,
    radius_sum: f32,
    bias_sum: f32,
    spin_sum: f32,
    sheet_sum: f32,
    weight: f32,
    sheet_count: usize,
}

impl EllipticAccumulator {
    fn accumulate(&mut self, telemetry: &EllipticTelemetry, weight: f32) {
        if !weight.is_finite() || weight <= 0.0 {
            return;
        }
        self.curvature_sum += telemetry.curvature_radius * weight;
        self.radius_sum += telemetry.geodesic_radius * weight;
        self.bias_sum += telemetry.normal_bias * weight;
        self.spin_sum += telemetry.spin_alignment * weight;
        self.sheet_sum += telemetry.sheet_position * weight;
        self.weight += weight;
        if telemetry.sheet_count > self.sheet_count {
            self.sheet_count = telemetry.sheet_count;
        }
    }

    fn finish(self) -> Option<EllipticTelemetry> {
        if self.weight <= 0.0 {
            return None;
        }
        let sheet_count = self.sheet_count.max(1);
        let sheet_position = (self.sheet_sum / self.weight).clamp(0.0, 1.0);
        let sheet_index = ((sheet_position * sheet_count as f32).round() as usize)
            .min(sheet_count.saturating_sub(1));
        Some(EllipticTelemetry {
            curvature_radius: (self.curvature_sum / self.weight).max(1e-6),
            geodesic_radius: (self.radius_sum / self.weight).max(0.0),
            spin_alignment: (self.spin_sum / self.weight).clamp(-1.0, 1.0),
            sheet_index,
            sheet_position,
            normal_bias: (self.bias_sum / self.weight).clamp(-1.0, 1.0),
            sheet_count,
        })
    }
}

/// Projects microlocal signatures into Z-space pulses.
#[derive(Clone, Debug)]
pub struct InterfaceZLift {
    weights: Vec<f32>,
    projector: LeechProjector,
    bias_gain: f32,
    elliptic: Option<EllipticWarp>,
}

impl InterfaceZLift {
    pub fn new(weights: &[f32], projector: LeechProjector) -> Self {
        assert!(!weights.is_empty(), "at least one weight expected");
        let norm = weights.iter().copied().sum::<f32>().max(f32::EPSILON);
        let weights = weights.iter().map(|w| w / norm).collect();
        Self {
            weights,
            projector,
            bias_gain: 1.0,
            elliptic: None,
        }
    }

    pub fn with_bias_gain(mut self, gain: f32) -> Self {
        self.bias_gain = gain.max(0.0);
        self
    }

    pub fn with_elliptic_warp(mut self, warp: EllipticWarp) -> Self {
        self.elliptic = Some(warp);
        self
    }

    pub fn set_bias_gain(&mut self, gain: f32) {
        if gain > 0.0 {
            self.bias_gain = gain;
        }
    }

    pub fn set_elliptic_warp(&mut self, warp: Option<EllipticWarp>) {
        self.elliptic = warp;
    }

    pub fn bias_gain(&self) -> f32 {
        self.bias_gain
    }

    pub fn elliptic_warp(&self) -> Option<&EllipticWarp> {
        self.elliptic.as_ref()
    }

    pub fn project(&self, signature: &InterfaceSignature) -> InterfaceZPulse {
        let total_support = signature.r_machine.iter().copied().sum::<f32>();
        let interface_cells = signature.r_machine.iter().filter(|v| **v > 0.5).count() as f32;

        let mut here = signature.raw_density.iter().copied().sum::<f32>()
            / signature.raw_density.len().max(1) as f32;
        here = here.max(0.0);

        let mut above = 0.0f32;
        let mut beneath = 0.0f32;
        let mut bias = 0.0f32;
        let mut elliptic_sample = None;

        if let Some(orient) = &signature.orientation {
            let mut weighted = vec![0.0f32; orient.shape()[0]];
            let mut weight_total = 0.0f32;
            for idx in indices(signature.r_machine.raw_dim()) {
                let weight = signature.r_machine[&IxDyn(idx.slice())];
                if weight <= 0.0 {
                    continue;
                }
                weight_total += weight;
                for (axis, accum) in weighted.iter_mut().enumerate() {
                    let mut orient_idx = Vec::with_capacity(orient.ndim());
                    orient_idx.push(axis);
                    orient_idx.extend_from_slice(idx.slice());
                    *accum += orient[IxDyn(&orient_idx)] * weight;
                }
            }
            if weight_total > 0.0 {
                for value in &mut weighted {
                    *value /= weight_total;
                }
            }
            if let Some(warp) = self.elliptic.as_ref() {
                if let Some(sample) = warp.map_orientation(&weighted) {
                    let normalized = sample.normalized_radius();
                    let spin = sample.spin_alignment;
                    let sheet_bias = sample.sheet_position;
                    let mut leading = (1.0 - normalized).max(0.0);
                    let mut trailing = normalized.max(0.0);
                    let spin_lead = 0.5 * (1.0 + spin);
                    let spin_trail = 0.5 * (1.0 - spin);
                    leading *= (0.6 + 0.4 * sheet_bias).max(1e-3) * spin_lead.max(1e-3);
                    trailing *= (0.6 + 0.4 * (1.0 - sheet_bias)).max(1e-3) * spin_trail.max(1e-3);
                    let total = (leading + trailing).max(1e-5);
                    above = here * (leading / total);
                    beneath = here * (trailing / total);
                    let enriched = self.projector.enrich(sample.normal_bias.abs() as f64) as f32;
                    bias = enriched * sample.normal_bias.signum() * self.bias_gain;
                    elliptic_sample = Some(sample);
                }
            }

            if elliptic_sample.is_none() {
                let weight0 = self.weights.first().copied().unwrap_or(1.0);
                let primary = weighted.first().copied().unwrap_or(0.0) * weight0;
                let enriched = self.projector.enrich(primary.abs() as f64) as f32;
                bias = enriched * primary.signum() * self.bias_gain;
                above = (primary.max(0.0)) * here;
                beneath = (-primary).max(0.0) * here;
            }
        }

        if signature.orientation.is_none() {
            above = 0.0;
            beneath = 0.0;
            elliptic_sample = None;
        }

        let band_energy = (above.max(0.0), here, beneath.max(0.0));
        let drift = if total_support > 0.0 {
            (band_energy.0 - band_energy.2) / total_support.max(1e-6)
        } else {
            0.0
        };

        InterfaceZPulse {
            source: ZSource::Microlocal,
            support: total_support,
            interface_cells,
            band_energy,
            scale: ZScale::new(signature.physical_radius),
            drift,
            z_bias: bias,
            quality_hint: None,
            standard_error: None,
            elliptic: elliptic_sample,
        }
    }
}

/// Snapshot emitted by the lift before Z-conductor fusion.
#[derive(Clone, Debug)]
pub struct InterfaceZPulse {
    pub source: ZSource,
    pub support: f32,
    pub interface_cells: f32,
    pub band_energy: (f32, f32, f32),
    pub scale: Option<ZScale>,
    pub drift: f32,
    pub z_bias: f32,
    pub quality_hint: Option<f32>,
    pub standard_error: Option<f32>,
    pub elliptic: Option<EllipticTelemetry>,
}

impl InterfaceZPulse {
    pub fn total_energy(&self) -> f32 {
        let (a, h, b) = self.band_energy;
        a + h + b
    }

    pub fn is_empty(&self) -> bool {
        self.support <= f32::EPSILON && self.total_energy() <= f32::EPSILON
    }

    pub fn aggregate(pulses: &[InterfaceZPulse]) -> InterfaceZPulse {
        if pulses.is_empty() {
            return InterfaceZPulse::default();
        }
        let mut support = 0.0f32;
        let mut interface_cells = 0.0f32;
        let mut band = (0.0f32, 0.0f32, 0.0f32);
        let mut drift_sum = 0.0f32;
        let mut drift_weight = 0.0f32;
        let mut bias_sum = 0.0f32;
        let mut bias_weight = 0.0f32;
        let mut scale_weights = Vec::new();
        let mut elliptic_acc = EllipticAccumulator::default();
        for pulse in pulses {
            support += pulse.support;
            interface_cells += pulse.interface_cells;
            band.0 += pulse.band_energy.0;
            band.1 += pulse.band_energy.1;
            band.2 += pulse.band_energy.2;
            let weight = scale_weight_for(pulse);
            drift_sum += pulse.drift * weight;
            drift_weight += weight;
            bias_sum += pulse.z_bias * weight;
            bias_weight += weight;
            if let Some(scale) = pulse.scale {
                scale_weights.push((scale, weight));
            }
            if let Some(telemetry) = &pulse.elliptic {
                elliptic_acc.accumulate(telemetry, weight);
            }
        }

        // Combine the weighted scale contributions from the input pulses.  When all
        // contributors omit the scale metadata we conservatively fall back to the
        // most recent non-empty sample so downstream consumers retain continuity.
        let aggregated_scale = ZScale::weighted_average(scale_weights.iter().copied())
            .or_else(|| pulses.iter().rev().find_map(|pulse| pulse.scale));
        InterfaceZPulse {
            source: ZSource::Microlocal,
            support,
            interface_cells,
            band_energy: band,
            scale: aggregated_scale.or(Some(ZScale::ONE)),
            drift: if drift_weight > 0.0 {
                drift_sum / drift_weight
            } else {
                0.0
            },
            z_bias: if bias_weight > 0.0 {
                bias_sum / bias_weight
            } else {
                0.0
            },
            quality_hint: None,
            standard_error: None,
            elliptic: elliptic_acc.finish(),
        }
    }

    pub fn lerp(current: &InterfaceZPulse, next: &InterfaceZPulse, alpha: f32) -> InterfaceZPulse {
        let t = alpha.clamp(0.0, 1.0);

        InterfaceZPulse {
            source: next.source,
            support: lerp(current.support, next.support, t),
            interface_cells: lerp(current.interface_cells, next.interface_cells, t),
            band_energy: (
                lerp(current.band_energy.0, next.band_energy.0, t),
                lerp(current.band_energy.1, next.band_energy.1, t),
                lerp(current.band_energy.2, next.band_energy.2, t),
            ),
            scale: match (current.scale, next.scale) {
                (Some(a), Some(b)) => Some(ZScale::lerp(a, b, t)),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            },
            drift: lerp(current.drift, next.drift, t),
            z_bias: lerp(current.z_bias, next.z_bias, t),
            quality_hint: next.quality_hint.or(current.quality_hint),
            standard_error: next.standard_error.or(current.standard_error),
            elliptic: match (&current.elliptic, &next.elliptic) {
                (Some(a), Some(b)) => Some(a.lerp(b, t)),
                (Some(a), None) => Some(a.clone()),
                (None, Some(b)) => Some(b.clone()),
                (None, None) => None,
            },
        }
    }

    pub fn scaled(&self, gain: f32) -> InterfaceZPulse {
        let gain = gain.max(0.0);
        InterfaceZPulse {
            source: self.source,
            support: self.support * gain,
            interface_cells: self.interface_cells * gain,
            band_energy: (
                self.band_energy.0 * gain,
                self.band_energy.1 * gain,
                self.band_energy.2 * gain,
            ),
            scale: self.scale,
            drift: self.drift * gain,
            z_bias: self.z_bias * gain,
            quality_hint: self.quality_hint,
            standard_error: self.standard_error,
            elliptic: self.elliptic.clone(),
        }
    }

    pub fn into_softlogic_feedback_with(
        self,
        psi_total: f32,
        weighted_loss: f32,
    ) -> SoftlogicZFeedback {
        SoftlogicZFeedback {
            psi_total,
            weighted_loss,
            band_energy: self.band_energy,
            drift: self.drift,
            z_signal: self.z_bias,
            // [SCALE-TODO] Patch 0 optional tagging
            scale: self.scale,
            events: Vec::new(),
            attributions: Vec::new(),
            elliptic: self.elliptic.as_ref().map(SoftlogicEllipticSample::from),
        }
    }

    pub fn into_softlogic_feedback(self) -> SoftlogicZFeedback {
        self.into_softlogic_feedback_with(0.0, 0.0)
    }
}

impl Default for InterfaceZPulse {
    fn default() -> Self {
        InterfaceZPulse {
            source: ZSource::Microlocal,
            support: 0.0,
            interface_cells: 0.0,
            band_energy: (0.0, 0.0, 0.0),
            scale: Some(ZScale::ONE),
            drift: 0.0,
            z_bias: 0.0,
            quality_hint: None,
            standard_error: None,
            elliptic: None,
        }
    }
}

fn scale_weight_for(pulse: &InterfaceZPulse) -> f32 {
    pulse.support.max(pulse.total_energy()).max(f32::EPSILON)
}

/// Result emitted after fusing the microlocal pulses through [`ZConductor`].
#[derive(Clone, Debug)]
pub struct InterfaceZFused {
    pub pulse: ZPulse,
    pub z: f32,
    pub support: f32,
    pub attributions: Vec<(ZSource, f32)>,
    pub events: Vec<String>,
}

/// Full report returned by [`InterfaceZConductor::step`].
#[derive(Clone, Debug)]
pub struct InterfaceZReport {
    pub gauge_ids: Vec<Option<Arc<str>>>,
    pub signatures: Vec<InterfaceSignature>,
    pub lift: InterfaceZLift,
    pub pulses: Vec<InterfaceZPulse>,
    pub qualities: Vec<f32>,
    pub fused_pulse: InterfaceZPulse,
    pub fused_z: InterfaceZFused,
    pub feedback: SoftlogicZFeedback,
    pub budget_scale: f32,
}

impl InterfaceZReport {
    pub fn has_interface(&self) -> bool {
        self.fused_pulse.support > 0.0
    }

    pub fn gauge_id(&self, index: usize) -> Option<&str> {
        self.gauge_ids.get(index).and_then(|id| id.as_deref())
    }

    pub fn signature_for(&self, id: &str) -> Option<&InterfaceSignature> {
        self.gauge_ids
            .iter()
            .position(|candidate| candidate.as_deref() == Some(id))
            .and_then(|idx| self.signatures.get(idx))
    }

    pub fn lift(&self) -> InterfaceZLift {
        self.lift.clone()
    }
}

/// Macro-to-microlocal feedback payload used to retune gauges and fusion.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MicrolocalFeedback {
    pub threshold_scale: Option<f32>,
    pub bias_gain: Option<f32>,
    pub smoothing: Option<f32>,
    pub tempo_hint: Option<f32>,
    pub stderr_hint: Option<f32>,
}

impl MicrolocalFeedback {
    pub fn merge(&self, other: &Self) -> Self {
        let threshold_scale = match (self.threshold_scale, other.threshold_scale) {
            (Some(a), Some(b)) => Some((a * b).max(0.0)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        MicrolocalFeedback {
            threshold_scale,
            bias_gain: other.bias_gain.or(self.bias_gain),
            smoothing: other.smoothing.or(self.smoothing),
            tempo_hint: other.tempo_hint.or(self.tempo_hint),
            stderr_hint: other.stderr_hint.or(self.stderr_hint),
        }
    }

    pub fn with_threshold_scale(mut self, scale: f32) -> Self {
        if scale > 0.0 {
            self.threshold_scale = Some(self.threshold_scale.unwrap_or(1.0) * scale);
        }
        self
    }

    pub fn with_bias_gain(mut self, gain: f32) -> Self {
        if gain > 0.0 {
            self.bias_gain = Some(gain);
        }
        self
    }

    pub fn with_smoothing(mut self, smoothing: f32) -> Self {
        self.smoothing = Some(smoothing.clamp(0.0, 1.0));
        self
    }

    pub fn with_tempo_hint(mut self, tempo: f32) -> Self {
        if tempo >= 0.0 {
            self.tempo_hint = Some(tempo);
        }
        self
    }

    pub fn with_stderr_hint(mut self, stderr: f32) -> Self {
        if stderr >= 0.0 {
            self.stderr_hint = Some(stderr);
        }
        self
    }
}

/// Quality policy applied to each microlocal pulse prior to fusion.
pub trait ZSourcePolicy: Send + Sync {
    fn quality(&self, pulse: &InterfaceZPulse) -> f32;
    fn late_fuse(
        &self,
        _fused: &mut InterfaceZPulse,
        _pulses: &[InterfaceZPulse],
        _qualities: &[f32],
    ) {
    }
}

/// Default policy that promotes balanced support.
#[derive(Debug, Clone)]
pub struct DefaultZSourcePolicy;

impl DefaultZSourcePolicy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DefaultZSourcePolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl ZSourcePolicy for DefaultZSourcePolicy {
    fn quality(&self, pulse: &InterfaceZPulse) -> f32 {
        if pulse.is_empty() {
            0.0
        } else {
            (pulse.support / pulse.total_energy().max(1e-6)).clamp(0.0, 1.0)
        }
    }
}

/// Fixed multiplier policy used by tests.
#[derive(Debug, Clone, Copy)]
pub struct FixedPolicy(pub f32);

impl ZSourcePolicy for FixedPolicy {
    fn quality(&self, _pulse: &InterfaceZPulse) -> f32 {
        self.0.clamp(0.0, 1.0)
    }
}

/// Composite policy that allows per-source overrides.
#[derive(Clone)]
pub struct CompositePolicy {
    default: Arc<dyn ZSourcePolicy>,
    overrides: FxHashMap<ZSource, Arc<dyn ZSourcePolicy>>,
}

impl CompositePolicy {
    pub fn new<P>(default: P) -> Self
    where
        P: ZSourcePolicy + 'static,
    {
        CompositePolicy {
            default: Arc::new(default),
            overrides: FxHashMap::default(),
        }
    }

    pub fn with<P>(mut self, source: ZSource, policy: P) -> Self
    where
        P: ZSourcePolicy + 'static,
    {
        self.overrides.insert(source, Arc::new(policy));
        self
    }
}

impl fmt::Debug for CompositePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompositePolicy")
            .field("overrides", &self.overrides.len())
            .finish()
    }
}

impl ZSourcePolicy for CompositePolicy {
    fn quality(&self, pulse: &InterfaceZPulse) -> f32 {
        if let Some(policy) = self.overrides.get(&pulse.source) {
            policy.quality(pulse)
        } else {
            self.default.quality(pulse)
        }
    }

    fn late_fuse(
        &self,
        fused: &mut InterfaceZPulse,
        pulses: &[InterfaceZPulse],
        qualities: &[f32],
    ) {
        self.default.late_fuse(fused, pulses, qualities);
        for (source, policy) in &self.overrides {
            if pulses.iter().any(|pulse| &pulse.source == source) {
                policy.late_fuse(fused, pulses, qualities);
            }
        }
    }
}

/// Band-energy gating applied on top of the quality policy.
#[derive(Debug, Clone)]
pub struct BandPolicy {
    min_quality: [f32; 3],
    hysteresis: f32,
}

impl BandPolicy {
    pub fn new(min_quality: [f32; 3]) -> Self {
        BandPolicy {
            min_quality: min_quality.map(|v| v.clamp(0.0, 1.0)),
            hysteresis: 0.0,
        }
    }

    pub fn with_hysteresis(mut self, hysteresis: f32) -> Self {
        self.hysteresis = hysteresis.max(0.0);
        self
    }

    /// Builds a band policy tuned for positively curved (elliptic) manifolds.
    pub fn positive_curvature(radius: f32) -> Self {
        let curvature = radius.max(1e-6).recip();
        let polar_bias = curvature.tanh();
        let leading = (0.2 + 0.15 * polar_bias).clamp(0.1, 0.45);
        let central = (0.5 + 0.25 * polar_bias).clamp(0.35, 0.9);
        BandPolicy::new([leading, central, leading])
    }

    pub fn project_quality(&self, pulse: &InterfaceZPulse) -> f32 {
        let total = pulse.total_energy().max(1e-6);
        let ratios = [
            pulse.band_energy.0 / total,
            pulse.band_energy.1 / total,
            pulse.band_energy.2 / total,
        ];
        let mut scale = 1.0f32;
        for (idx, ratio) in ratios.into_iter().enumerate() {
            let threshold = self.min_quality[idx];
            if ratio + self.hysteresis < threshold {
                if threshold <= f32::EPSILON {
                    scale *= 0.0;
                } else {
                    scale *= (ratio / threshold).clamp(0.0, 1.0);
                }
            }
        }
        scale.clamp(0.0, 1.0)
    }
}

/// Budget guard that clamps the fused Z pulse magnitude.
#[derive(Debug, Clone)]
pub struct BudgetPolicy {
    z_max: f32,
}

impl BudgetPolicy {
    pub fn new(z_max: f32) -> Self {
        BudgetPolicy {
            z_max: z_max.abs().max(1e-6),
        }
    }

    pub fn apply(&self, fused: &mut InterfaceZPulse) -> f32 {
        if fused.z_bias.abs() <= self.z_max {
            1.0
        } else {
            let scale = self.z_max / fused.z_bias.abs();
            fused.z_bias *= scale;
            fused.band_energy.0 *= scale;
            fused.band_energy.2 *= scale;
            fused.support *= scale;
            scale
        }
    }
}

#[derive(Clone, Default)]
struct MicrolocalEmitter {
    queue: Arc<Mutex<VecDeque<ZPulse>>>,
}

impl MicrolocalEmitter {
    fn new() -> Self {
        Self::default()
    }

    fn extend<I>(&self, pulses: I)
    where
        I: IntoIterator<Item = ZPulse>,
    {
        let mut queue = self.queue.lock().expect("microlocal queue poisoned");
        queue.extend(pulses);
    }
}

impl ZEmitter for MicrolocalEmitter {
    fn name(&self) -> ZSource {
        ZSource::Microlocal
    }

    fn tick(&mut self, _now: u64) -> Option<ZPulse> {
        self.queue
            .lock()
            .expect("microlocal queue poisoned")
            .pop_front()
    }
}

/// Drives a bank of microlocal gauges and fuses the resulting Z pulses into a
/// smoothed control signal suitable for Softlogic feedback.
#[derive(Clone, Debug)]
struct GaugeSlot {
    id: Option<Arc<str>>,
    gauge: InterfaceGauge,
}

pub struct InterfaceZConductor {
    gauges: Vec<GaugeSlot>,
    lift: InterfaceZLift,
    conductor: ZConductor,
    clock: u64,
    smoothing: f32,
    policy: Arc<dyn ZSourcePolicy>,
    band_policy: Option<BandPolicy>,
    budget_policy: Option<BudgetPolicy>,
    previous: Option<InterfaceZPulse>,
    carry: Option<InterfaceZPulse>,
    emitter: MicrolocalEmitter,
    default_tempo_hint: Option<f32>,
    default_stderr_hint: Option<f32>,
}

impl InterfaceZConductor {
    pub fn new(gauges: Vec<InterfaceGauge>, lift: InterfaceZLift) -> Self {
        let slots = gauges
            .into_iter()
            .map(|gauge| GaugeSlot { id: None, gauge })
            .collect();
        InterfaceZConductor::with_slots(slots, lift)
    }

    pub fn from_bank(bank: MicrolocalGaugeBank, lift: InterfaceZLift) -> Self {
        let slots = bank
            .into_entries()
            .into_iter()
            .map(|(id, gauge)| GaugeSlot {
                id: Some(id),
                gauge,
            })
            .collect();
        InterfaceZConductor::with_slots(slots, lift)
    }

    pub fn lift(&self) -> InterfaceZLift {
        self.lift.clone()
    }

    fn with_slots(gauges: Vec<GaugeSlot>, lift: InterfaceZLift) -> Self {
        assert!(!gauges.is_empty(), "at least one gauge must be supplied");
        InterfaceZConductor {
            gauges,
            lift,
            conductor: ZConductor::new(ZConductorCfg::default()),
            clock: 0,
            smoothing: 0.0,
            policy: Arc::new(DefaultZSourcePolicy::new()),
            band_policy: None,
            budget_policy: None,
            previous: None,
            carry: None,
            emitter: MicrolocalEmitter::new(),
            default_tempo_hint: None,
            default_stderr_hint: None,
        }
    }

    pub fn with_smoothing(mut self, smoothing: f32) -> Self {
        self.smoothing = smoothing.clamp(0.0, 1.0);
        self
    }

    pub fn with_policy<P>(mut self, policy: P) -> Self
    where
        P: ZSourcePolicy + 'static,
    {
        self.policy = Arc::new(policy);
        self
    }

    pub fn with_band_policy(mut self, policy: BandPolicy) -> Self {
        self.band_policy = Some(policy);
        self
    }

    pub fn with_budget_policy(mut self, policy: BudgetPolicy) -> Self {
        self.budget_policy = Some(policy);
        self
    }

    pub fn reinforce_positive_curvature(mut self, radius: f32) -> Self {
        let warp = EllipticWarp::new(radius);
        let mut lift = self.lift.clone();
        lift = lift.with_elliptic_warp(warp.clone());
        self.lift = lift;
        self.band_policy = Some(BandPolicy::positive_curvature(radius).with_hysteresis(0.05));
        self
    }

    pub fn apply_feedback(&mut self, feedback: &MicrolocalFeedback) {
        if let Some(scale) = feedback.threshold_scale {
            if scale > 0.0 {
                for slot in &mut self.gauges {
                    slot.gauge.scale_threshold(scale);
                }
            }
        }
        if let Some(gain) = feedback.bias_gain {
            self.lift.set_bias_gain(gain);
        }
        if let Some(smoothing) = feedback.smoothing {
            self.smoothing = smoothing.clamp(0.0, 1.0);
        }
        if let Some(tempo) = feedback.tempo_hint {
            self.default_tempo_hint = Some(tempo.max(0.0));
        }
        if let Some(stderr) = feedback.stderr_hint {
            self.default_stderr_hint = Some(stderr.max(0.0));
        }
    }

    #[cfg(test)]
    pub fn gauge_thresholds(&self) -> Vec<f32> {
        self.gauges
            .iter()
            .map(|slot| slot.gauge.threshold())
            .collect()
    }

    #[cfg(test)]
    pub fn bias_gain(&self) -> f32 {
        self.lift.bias_gain()
    }

    #[cfg(test)]
    pub fn default_tempo_hint(&self) -> Option<f32> {
        self.default_tempo_hint
    }

    #[cfg(test)]
    pub fn default_stderr_hint(&self) -> Option<f32> {
        self.default_stderr_hint
    }

    pub fn step(
        &mut self,
        mask: &ArrayD<f32>,
        c_prime: Option<&ArrayD<f32>>,
        tempo_hint: Option<f32>,
        stderr_hint: Option<f32>,
    ) -> InterfaceZReport {
        let mut gauge_ids = Vec::with_capacity(self.gauges.len());
        let mut signatures = Vec::with_capacity(self.gauges.len());
        let mut pulses = Vec::with_capacity(self.gauges.len());
        let stderr_base = stderr_hint
            .or(self.default_stderr_hint)
            .map(|stderr| stderr.max(0.0));
        for slot in &self.gauges {
            let signature = slot.gauge.analyze_with_label(mask, c_prime);
            let mut pulse = self.lift.project(&signature);
            if let Some(stderr) = stderr_base {
                pulse.standard_error = Some(stderr);
            }
            pulses.push(pulse);
            gauge_ids.push(slot.id.clone());
            signatures.push(signature);
        }

        let mut qualities = Vec::with_capacity(pulses.len());
        let mut weighted = Vec::with_capacity(pulses.len());
        for pulse in &pulses {
            let mut quality = self.policy.quality(pulse).clamp(0.0, 1.0);
            if let Some(policy) = &self.band_policy {
                quality *= policy.project_quality(pulse);
            }
            qualities.push(quality);
            weighted.push(pulse.scaled(quality));
        }

        let mut fused = InterfaceZPulse::aggregate(&weighted);
        let mut events = Vec::new();
        if let Some(previous) = &self.previous {
            if self.smoothing > 0.0 {
                fused = InterfaceZPulse::lerp(previous, &fused, self.smoothing);
                events.push("smoothing.applied".to_string());
            }
        }

        if let Some(elliptic) = &fused.elliptic {
            events.extend(elliptic.event_tags().into_iter());
        }

        let mut budget_scale = 1.0;
        if let Some(budget) = &self.budget_policy {
            budget_scale = budget.apply(&mut fused);
        }

        self.policy.late_fuse(&mut fused, &pulses, &qualities);

        let now = self.clock;
        self.clock = self.clock.wrapping_add(1);
        if fused.standard_error.is_none() {
            if let Some(stderr) = stderr_base {
                fused.standard_error = Some(stderr);
            }
        }

        let tempo_estimate = tempo_hint
            .or(self.default_tempo_hint)
            .unwrap_or_else(|| fused.total_energy());

        let mut zpulses = Vec::with_capacity(pulses.len());
        let stderr_base = stderr_base.unwrap_or(0.0);
        for (pulse, &quality) in pulses.iter().zip(&qualities) {
            let support = ZSupport::from_band_energy(pulse.band_energy);
            let stderr = pulse.standard_error.unwrap_or(stderr_base);
            zpulses.push(ZPulse {
                source: pulse.source,
                ts: now,
                tempo: tempo_estimate,
                band_energy: pulse.band_energy,
                drift: pulse.drift,
                z_bias: pulse.z_bias,
                support,
                scale: pulse.scale,
                quality,
                stderr,
                latency_ms: 0.0,
            });
        }
        self.emitter.extend(zpulses);

        let mut registry = ZRegistry::with_capacity(1);
        registry.register(self.emitter.clone());
        let z_fused = self.conductor.step_from_registry(&mut registry, now);

        events.extend(z_fused.events.clone());

        let fused_pulse = fused.clone();
        let mut feedback = fused.clone().into_softlogic_feedback();
        feedback.set_events(events.clone());
        feedback.set_attributions(z_fused.attributions.clone());
        self.previous = Some(fused.clone());
        self.carry = Some(fused.clone());

        let z_pulse = InterfaceZConductor::into_zpulse(&fused, now, &qualities, tempo_estimate);
        let fused_report = InterfaceZFused {
            pulse: z_pulse,
            z: z_fused.z,
            support: z_fused.support,
            attributions: z_fused.attributions,
            events,
        };

        InterfaceZReport {
            gauge_ids,
            signatures,
            lift: self.lift.clone(),
            pulses,
            qualities,
            fused_pulse,
            fused_z: fused_report,
            feedback,
            budget_scale,
        }
    }

    pub fn last_fused_pulse(&self) -> InterfaceZPulse {
        self.carry.clone().unwrap_or_default()
    }

    fn into_zpulse(fused: &InterfaceZPulse, now: u64, qualities: &[f32], tempo: f32) -> ZPulse {
        let support = ZSupport::from_band_energy(fused.band_energy);
        let avg_quality = if qualities.is_empty() {
            0.0
        } else {
            qualities.iter().copied().sum::<f32>() / qualities.len() as f32
        };
        ZPulse {
            source: fused.source,
            ts: now,
            tempo,
            band_energy: fused.band_energy,
            drift: fused.drift,
            z_bias: fused.z_bias,
            support,
            scale: fused.scale,
            quality: avg_quality,
            stderr: fused.standard_error.unwrap_or(0.0),
            latency_ms: 0.0,
        }
    }
}

impl fmt::Debug for InterfaceZConductor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InterfaceZConductor")
            .field("gauges", &self.gauges.len())
            .field("smoothing", &self.smoothing)
            .finish()
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn assert_neutral_scale(scale: Option<ZScale>) {
        let scale = scale.expect("scale tag missing from pulse");
        assert!((scale.physical_radius - ZScale::ONE.physical_radius).abs() < 1e-6);
        assert!((scale.log_radius - ZScale::ONE.log_radius).abs() < 1e-6);
    }

    #[test]
    fn detects_boundary_presence() {
        let mask = array![[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]].into_dyn();
        let gauge = InterfaceGauge::new(1.0, 1.0);
        let sig = gauge.analyze(&mask);
        assert!(sig.has_interface());
        assert_eq!(sig.kappa_d, 2.0 * std::f32::consts::PI);
        assert!((sig.perimeter_density[IxDyn(&[1, 1])] - sig.kappa_d).abs() < 1e-5);
        assert_eq!(sig.perimeter_density[IxDyn(&[0, 0])], 0.0);
        assert!((sig.physical_radius - 1.0).abs() < 1e-6);
    }

    #[test]
    fn oriented_normals_require_label() {
        let mask = array![[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]].into_dyn();
        let c_prime = mask.mapv(|v| if v > 0.5 { 1.0 } else { -1.0 });
        let gauge = InterfaceGauge::new(1.0, 1.0);
        let sig = gauge.analyze_with_label(&mask, Some(&c_prime));
        let orient = sig.orientation.expect("orientation missing");
        let normal_y = orient[IxDyn(&[0, 1, 1])];
        let normal_x = orient[IxDyn(&[1, 1, 1])];
        let norm = (normal_x * normal_x + normal_y * normal_y).sqrt();
        assert!((norm - 1.0).abs() < 1e-3);
        assert!(normal_y.abs() > 0.5);
        assert!(normal_x.abs() > 0.5);
    }

    #[test]
    fn conductor_rollout_preserves_neutral_scale() {
        let mask = array![[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]].into_dyn();
        let gauge = InterfaceGauge::new(1.0, 1.0);
        let lift = InterfaceZLift::new(&[1.0, 0.0], LeechProjector::new(24, 0.5));
        let mut conductor = InterfaceZConductor::new(vec![gauge], lift);

        let report = conductor.step(&mask, None, None, None);

        for pulse in &report.pulses {
            assert_neutral_scale(pulse.scale);
        }

        assert_neutral_scale(report.fused_pulse.scale);
        assert_neutral_scale(report.feedback.scale);
        assert_neutral_scale(report.fused_z.pulse.scale);
    }

    #[test]
    fn elliptic_warp_injects_curvature_bias() {
        let mask = array![[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]].into_dyn();
        let c_prime = mask.mapv(|v| if v > 0.5 { 1.0 } else { -1.0 });
        let gauge = InterfaceGauge::new(1.0, 1.0);
        let lift = InterfaceZLift::new(&[1.0, 0.0], LeechProjector::new(24, 0.5))
            .with_elliptic_warp(EllipticWarp::new(1.5).with_sheet_count(4));
        let signature = gauge.analyze_with_label(&mask, Some(&c_prime));
        let pulse = lift.project(&signature);
        assert!(pulse.elliptic.is_some());
        let telemetry = pulse.elliptic.as_ref().unwrap();
        assert!(telemetry.normalized_radius() >= 0.0);
    }

    #[test]
    fn conductor_emits_elliptic_events() {
        let mask = array![[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]].into_dyn();
        let c_prime = mask.mapv(|v| if v > 0.5 { 1.0 } else { -1.0 });
        let gauge = InterfaceGauge::new(1.0, 1.0);
        let lift = InterfaceZLift::new(&[1.0, 0.0], LeechProjector::new(24, 0.5));
        let mut conductor =
            InterfaceZConductor::new(vec![gauge], lift).reinforce_positive_curvature(1.5);
        let report = conductor.step(&mask, Some(&c_prime), None, None);
        assert!(report
            .feedback
            .events
            .iter()
            .any(|event| event.starts_with("elliptic.")));
    }

    #[test]
    fn feedback_reconfigures_conductor_defaults() {
        let mask = array![[0.0, 1.0], [0.0, 1.0]].into_dyn();
        let gauge = InterfaceGauge::new(1.0, 1.0);
        let lift = InterfaceZLift::new(&[1.0], LeechProjector::new(24, 0.3));
        let mut conductor = InterfaceZConductor::new(vec![gauge], lift);

        let feedback = MicrolocalFeedback::default()
            .with_threshold_scale(0.5)
            .with_bias_gain(1.8)
            .with_smoothing(0.4)
            .with_tempo_hint(1.2)
            .with_stderr_hint(0.05);
        conductor.apply_feedback(&feedback);

        let thresholds = conductor.gauge_thresholds();
        assert_eq!(thresholds.len(), 1);
        assert!((thresholds[0] - 0.125).abs() < 1e-6);
        assert!((conductor.bias_gain() - 1.8).abs() < 1e-6);
        assert!((conductor.default_tempo_hint().unwrap() - 1.2).abs() < 1e-6);
        assert!((conductor.default_stderr_hint().unwrap() - 0.05).abs() < 1e-6);

        let report = conductor.step(&mask, None, None, None);
        assert_eq!(report.pulses.len(), 1);
        let stderr = report.pulses[0].standard_error.expect("stderr missing");
        assert!((stderr - 0.05).abs() < 1e-6);
        assert!((report.fused_pulse.standard_error.expect("fused stderr") - 0.05).abs() < 1e-6);
        assert!((report.fused_z.pulse.tempo - 1.2).abs() < 1e-6);
    }

    #[test]
    fn gauge_bank_registers_unique_ids() {
        let mut bank = MicrolocalGaugeBank::new();
        assert!(bank.register("fine", InterfaceGauge::new(1.0, 1.0)));
        assert!(!bank.register("fine", InterfaceGauge::new(1.0, 2.0)));
        assert!(bank.register("coarse", InterfaceGauge::new(1.0, 3.0)));
        assert_eq!(bank.len(), 2);
        assert!(bank.get("fine").is_some());
        assert!(bank.get_mut("coarse").is_some());
        let removed = bank.remove("fine");
        assert!(removed.is_some());
        assert!(bank.get("fine").is_none());
        assert_eq!(bank.ids().collect::<Vec<_>>(), vec!["coarse"]);
    }

    #[test]
    fn gauge_bank_runs_all_registered_probes() {
        let mut bank = MicrolocalGaugeBank::new();
        bank.register("fine", InterfaceGauge::new(1.0, 1.0));
        bank.register("coarse", InterfaceGauge::new(1.0, 2.0));

        let mask = array![[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]].into_dyn();
        let signatures = bank.analyze_all(&mask, None);
        assert_eq!(signatures.len(), 2);
        let fine = signatures
            .iter()
            .find(|(id, _)| id.as_ref() == "fine")
            .map(|(_, sig)| sig)
            .expect("fine gauge missing");
        let coarse = signatures
            .iter()
            .find(|(id, _)| id.as_ref() == "coarse")
            .map(|(_, sig)| sig)
            .expect("coarse gauge missing");
        assert!(fine.has_interface());
        assert!(coarse.has_interface());
        assert!(fine.physical_radius <= coarse.physical_radius);
    }

    #[test]
    fn conductor_can_be_built_from_gauge_bank() {
        let mut bank = MicrolocalGaugeBank::new();
        bank.register("default", InterfaceGauge::new(1.0, 1.0));
        let lift = InterfaceZLift::new(&[1.0, 0.0], LeechProjector::new(24, 0.5));
        let conductor = InterfaceZConductor::from_bank(bank.clone(), lift);
        assert_eq!(conductor.gauge_thresholds(), vec![0.25]);

        bank.get_mut("default")
            .expect("gauge missing")
            .scale_threshold(0.5);
        let lift = InterfaceZLift::new(&[1.0, 0.0], LeechProjector::new(24, 0.5));
        let conductor = InterfaceZConductor::from_bank(bank, lift);
        assert!(conductor.gauge_thresholds()[0] < 0.2);
    }
}
