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

use crate::telemetry::hub::SoftlogicZFeedback;
use crate::theory::zpulse::{
    ZConductor, ZConductorCfg, ZEmitter, ZPulse, ZRegistry, ZScale, ZSource, ZSupport,
};
use crate::util::math::LeechProjector;
use ndarray::{indices, ArrayD, ArrayViewD, Dimension, IxDyn};
use rustc_hash::FxHashMap;
use statrs::function::gamma::gamma;
use std::collections::VecDeque;
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

/// Projects microlocal signatures into Z-space pulses.
#[derive(Clone)]
pub struct InterfaceZLift {
    weights: Vec<f32>,
    projector: LeechProjector,
    bias_gain: f32,
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
        }
    }

    pub fn with_bias_gain(mut self, gain: f32) -> Self {
        self.bias_gain = gain.max(0.0);
        self
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
            let weight0 = self.weights.first().copied().unwrap_or(1.0);
            let primary = weighted.first().copied().unwrap_or(0.0) * weight0;
            let enriched = self.projector.enrich(primary.abs() as f64) as f32;
            bias = enriched * primary.signum() * self.bias_gain;
            above = (primary.max(0.0)) * here;
            beneath = (-primary).max(0.0) * here;
        }

        if signature.orientation.is_none() {
            above = 0.0;
            beneath = 0.0;
        }

        let band_energy = (above.max(0.0), here, beneath.max(0.0));
        let drift = if total_support > 0.0 {
            (band_energy.0 - band_energy.2) / total_support.max(1e-6)
        } else {
            0.0
        };

        // ここで物理半径からスケールを設定
        let scale = ZScale::new(signature.physical_radius).unwrap_or(ZScale::ONE);

        InterfaceZPulse {
            source: ZSource::Microlocal,
            support: total_support,
            interface_cells,
            band_energy,
            scale: Some(scale),
            drift,
            z_bias: bias,
            quality_hint: None,
            standard_error: None,
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
        let mut scale_phys = 0.0f32;
        let mut scale_log = 0.0f32;
        let mut scale_weight = 0.0f32;
        for pulse in pulses {
            support += pulse.support;
            interface_cells += pulse.interface_cells;
            band.0 += pulse.band_energy.0;
            band.1 += pulse.band_energy.1;
            band.2 += pulse.band_energy.2;
            let weight = pulse.support.max(1e-6);
            drift_sum += pulse.drift * weight;
            drift_weight += weight;
            bias_sum += pulse.z_bias * weight;
            bias_weight += weight;
            if let Some(scale) = pulse.scale {
                let w = scale_weight_for(pulse);
                scale_phys += scale.physical_radius * w;
                scale_log += scale.log_radius * w;
                scale_weight += w;
            }
        }

        // 集約スケールを計算して利用する（unused 警告の解消）
        let scale = if scale_weight > 0.0 {
            ZScale::from_components(scale_phys / scale_weight, scale_log / scale_weight)
        } else {
            None
        };

        InterfaceZPulse {
            source: ZSource::Microlocal,
            support,
            interface_cells,
            band_energy: band,
            scale,
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
        }
    }

    pub fn lerp(current: &InterfaceZPulse, next: &InterfaceZPulse, alpha: f32) -> InterfaceZPulse {
        let t = alpha.clamp(0.0, 1.0);

        // スケールの補間（両方 Some の場合のみ補間、それ以外は次を優先）
        let scale = match (current.scale, next.scale) {
            (Some(a), Some(b)) => Some(ZScale::lerp(a, b, t)),
            (_, s @ Some(_)) => s,
            (s @ Some(_), None) => s,
            (None, None) => None,
        };

        InterfaceZPulse {
            source: next.source,
            support: lerp(current.support, next.support, t),
            interface_cells: lerp(current.interface_cells, next.interface_cells, t),
            band_energy: (
                lerp(current.band_energy.0, next.band_energy.0, t),
                lerp(current.band_energy.1, next.band_energy.1, t),
                lerp(current.band_energy.2, next.band_energy.2, t),
            ),
            scale,
            drift: lerp(current.drift, next.drift, t),
            z_bias: lerp(current.z_bias, next.z_bias, t),
            quality_hint: next.quality_hint.or(current.quality_hint),
            standard_error: next.standard_error.or(current.standard_error),
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
            scale: self.scale,
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
            scale: None,
            drift: 0.0,
            z_bias: 0.0,
            quality_hint: None,
            standard_error: None,
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
#[derive(Clone)]
pub struct InterfaceZConductor {
    gauges: Vec<InterfaceGauge>,
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
}

impl InterfaceZConductor {
    pub fn new(gauges: Vec<InterfaceGauge>, lift: InterfaceZLift) -> Self {
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

    pub fn step(
        &mut self,
        mask: &ArrayD<f32>,
        c_prime: Option<&ArrayD<f32>>,
        tempo_hint: Option<f32>,
        stderr_hint: Option<f32>,
    ) -> InterfaceZReport {
        let mut pulses = Vec::with_capacity(self.gauges.len());
        for gauge in &self.gauges {
            let signature = gauge.analyze_with_label(mask, c_prime);
            let mut pulse = self.lift.project(&signature);
            if let Some(stderr) = stderr_hint {
                pulse.standard_error = Some(stderr.max(0.0));
            }
            pulses.push(pulse);
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

        let mut budget_scale = 1.0;
        if let Some(budget) = &self.budget_policy {
            budget_scale = budget.apply(&mut fused);
        }

        self.policy.late_fuse(&mut fused, &pulses, &qualities);

        let now = self.clock;
        self.clock = self.clock.wrapping_add(1);
        let tempo_estimate = tempo_hint.unwrap_or_else(|| fused.total_energy());

        let mut zpulses = Vec::with_capacity(pulses.len());
        for (pulse, &quality) in pulses.iter().zip(&qualities) {
            let support = ZSupport::from_band_energy(pulse.band_energy);
            let stderr = pulse
                .standard_error
                .unwrap_or_else(|| stderr_hint.unwrap_or(0.0));
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
        let feedback = fused.clone().into_softlogic_feedback();
        self.previous = Some(fused.clone());
        self.carry = Some(fused.clone());

        let z_pulse = InterfaceZConductor::into_zpulse(&fused, now, &qualities);
        let fused_report = InterfaceZFused {
            pulse: z_pulse,
            z: z_fused.z,
            support: z_fused.support,
            attributions: z_fused.attributions,
            events,
        };

        InterfaceZReport {
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

    fn into_zpulse(fused: &InterfaceZPulse, now: u64, qualities: &[f32]) -> ZPulse {
        let support = ZSupport::from_band_energy(fused.band_energy);
        let avg_quality = if qualities.is_empty() {
            0.0
        } else {
            qualities.iter().copied().sum::<f32>() / qualities.len() as f32
        };
        ZPulse {
            source: fused.source,
            ts: now,
            tempo: fused.total_energy(),
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
}
