// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// ============================================================================
//  SpiralReality Proprietary
//  Copyright (c) 2025 SpiralReality. All Rights Reserved.
//
//  NOTICE: This file contains confidential and proprietary information of
//  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
//  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
//  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
//
//  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
//  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
//  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
//  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
//  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
//  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
//  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ============================================================================

//! Microlocal boundary gauges used to detect interfaces that survive the
//! blow-up limit without committing to a global label.
//!
//! The constructions in this module mirror the BV/varifold correspondence
//! outlined in the SpiralTorch sheaf notes: only the presence of an interface
//! (the `R` machine) remains gauge invariant in the microlocal limit, while
//! oriented data such as co-normals requires an external label `c′` to fix a
//! sign.  `InterfaceGauge` provides a practical discretisation that measures
//! the local total-variation density inside shrinking metric balls, emits the
//! stable boundary indicator, estimates the mean-curvature magnitude that
//! survives the gauge quotient, and optionally reconstructs the oriented normal
//! field together with signed curvature when a phase label is supplied.

use crate::coop::ai::{CoopAgent, CoopProposal};
use crate::telemetry::hub::{self, SoftlogicZFeedback};
use crate::theory::zpulse::{
    ZAdaptiveGainCfg, ZConductor, ZEmitter, ZFrequencyConfig, ZFused, ZPulse, ZRegistry, ZSource,
};
use crate::util::math::LeechProjector;
use ndarray::{indices, ArrayD, Dimension, IxDyn};
use statrs::function::gamma::gamma;
use std::f64::consts::PI;
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

#[derive(Clone, Default, Debug)]
pub struct MicrolocalEmitter {
    queue: Arc<Mutex<VecDeque<ZPulse>>>,
}

impl MicrolocalEmitter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enqueue(&self, pulse: ZPulse) {
        let mut queue = self
            .queue
            .lock()
            .expect("microlocal emitter queue poisoned");
        queue.push_back(pulse);
    }

    pub fn extend<I>(&self, pulses: I)
    where
        I: IntoIterator<Item = ZPulse>,
    {
        let mut queue = self
            .queue
            .lock()
            .expect("microlocal emitter queue poisoned");
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
            .expect("microlocal emitter queue poisoned")
            .pop_front()
    }
}

/// Result of running an [`InterfaceGauge`] on a binary phase field.
#[derive(Debug, Clone)]
pub struct InterfaceSignature {
    /// Gauge invariant boundary detector (1 = interface present).
    pub r_machine: ArrayD<f32>,
    /// Raw (0–1) estimate of the perimeter density relative to the flat lattice.
    pub raw_density: ArrayD<f32>,
    /// Perimeter density normalised by \(|D\chi_E|(B_\varepsilon)/\varepsilon^{d-1}|\),
    /// mapped to \{0, \kappa_d\} to match the finite perimeter blow-up.
    pub perimeter_density: ArrayD<f32>,
    /// Gauge invariant mean curvature magnitude estimated from the unlabeled mask.
    pub mean_curvature: ArrayD<f32>,
    /// Signed mean curvature reconstructed only when an oriented label `c′` is supplied.
    pub signed_mean_curvature: Option<ArrayD<f32>>,
    /// Optional unit normal field (only populated when `c_prime` was provided).
    pub orientation: Option<ArrayD<f32>>,
    /// Surface measure of the unit \((d-1)\)-sphere used for the normalisation.
    pub kappa_d: f32,
    /// Radius in lattice steps used to accumulate the blow-up statistics.
    pub radius: isize,
    /// Physical radius (same units as the lattice spacing) backing `radius`.
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
    /// Constructs a new interface gauge.
    ///
    /// * `grid_spacing` – physical size of one lattice step (used for gradients).
    /// * `physical_radius` – radius of the metric ball used in the blow-up probe.
    pub fn new(grid_spacing: f32, physical_radius: f32) -> Self {
        Self {
            grid_spacing: grid_spacing.max(f32::EPSILON),
            physical_radius: physical_radius.max(f32::EPSILON),
            threshold: 0.25,
        }
    }

    /// Adjusts the contrast threshold used when deciding if two samples belong
    /// to different phases. Defaults to `0.25` which is robust for \{0,1\} masks.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold.max(0.0);
        self
    }

    /// Evaluates the gauge on a binary mask, returning the interface signature.
    pub fn analyze(&self, mask: &ArrayD<f32>) -> InterfaceSignature {
        self.analyze_with_radius(mask, None, self.physical_radius)
    }

    /// Evaluates the gauge on a binary mask using an optional signed label `c′`
    /// to reconstruct oriented normals.
    pub fn analyze_with_label(
        &self,
        mask: &ArrayD<f32>,
        c_prime: Option<&ArrayD<f32>>,
    ) -> InterfaceSignature {
        self.analyze_with_radius(mask, c_prime, self.physical_radius)
    }

    /// Evaluates the gauge at a custom physical radius, returning the interface
    /// signature extracted at that scale.
    pub fn analyze_with_radius(
        &self,
        mask: &ArrayD<f32>,
        c_prime: Option<&ArrayD<f32>>,
        physical_radius: f32,
    ) -> InterfaceSignature {
        let dim = mask.ndim();
        assert!(dim > 0, "mask must have positive dimension");
        if let Some(label) = c_prime {
            assert_eq!(label.shape(), mask.shape(), "c′ must match mask shape");
        }

        let radius = radius_in_steps(physical_radius, self.grid_spacing);
        self.analyze_with_steps(mask, c_prime, radius, physical_radius.max(f32::EPSILON))
    }

    /// Evaluates the gauge across multiple radii and returns the resulting
    /// signatures ordered according to the supplied radii slice.
    pub fn analyze_multiradius(
        &self,
        mask: &ArrayD<f32>,
        c_prime: Option<&ArrayD<f32>>,
        radii: &[f32],
    ) -> Vec<InterfaceSignature> {
        assert!(!radii.is_empty(), "at least one radius must be provided");
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
        let dim = mask.ndim();
        let radius_steps = radius_steps.max(1);
        let offsets = generate_offsets(dim, radius_steps);
        let shape = mask.shape().to_vec();
        let raw_dim = mask.raw_dim();
        let mut r_machine = ArrayD::<f32>::zeros(raw_dim.clone());
        let mut raw_density = ArrayD::<f32>::zeros(raw_dim.clone());
        let mut perimeter_density = ArrayD::<f32>::zeros(raw_dim.clone());
        let mut mean_curvature = ArrayD::<f32>::zeros(raw_dim.clone());
        let mut signed_mean_curvature = c_prime.map(|_| ArrayD::<f32>::zeros(raw_dim.clone()));
        let mut orientation = c_prime.map(|_| {
            let mut full_shape = Vec::with_capacity(dim + 1);
            full_shape.push(dim);
            full_shape.extend_from_slice(&shape);
            ArrayD::<f32>::zeros(IxDyn(&full_shape))
        });

        let threshold = self.threshold;
        let kappa_d = unit_sphere_area(dim);

        for idx in indices(raw_dim.clone()) {
            let idx_slice = idx.slice();
            let idx_dyn = IxDyn(idx_slice);
            let center = mask[&idx_dyn];
            let mut min_val = center;
            let mut max_val = center;

            for off in offsets.iter().filter(|off| off.iter().any(|&o| o != 0)) {
                if let Some(neigh) = neighbor_sample(mask, idx_slice, &shape, off) {
                    min_val = min_val.min(neigh);
                    max_val = max_val.max(neigh);
                }
            }

            let has_interface = (max_val - min_val) >= threshold;
            let r_val = if has_interface { 1.0 } else { 0.0 };
            r_machine[&idx_dyn] = r_val;

            let (raw_val, discrete_jump) =
                local_raw_density(mask, idx_slice, &shape, threshold, center);
            raw_density[&idx_dyn] = raw_val;
            perimeter_density[&idx_dyn] = if discrete_jump { kappa_d } else { 0.0 };

            if has_interface {
                if let Some((abs_curv, _)) =
                    mean_curvature_from_field(mask, idx_slice, &shape, self.grid_spacing, threshold)
                {
                    mean_curvature[&idx_dyn] = abs_curv;
                }
                if let (Some(label_field), Some(curv_field)) =
                    (c_prime, signed_mean_curvature.as_mut())
                {
                    if let Some((_, signed_curv)) = mean_curvature_from_field(
                        label_field,
                        idx_slice,
                        &shape,
                        self.grid_spacing,
                        threshold,
                    ) {
                        curv_field[&idx_dyn] = signed_curv;
                    }
                }
            }

            if let (Some(label_field), Some(ref mut orient)) = (c_prime, orientation.as_mut()) {
                if has_interface {
                    let normal = oriented_normal(
                        label_field,
                        idx_slice,
                        &shape,
                        self.grid_spacing,
                        threshold,
                    );
                    if let Some(n) = normal {
                        for axis in 0..dim {
                            let mut coord = Vec::with_capacity(dim + 1);
                            coord.push(axis);
                            coord.extend_from_slice(idx_slice);
                            orient[IxDyn(&coord)] = n[axis];
                        }
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

    /// Returns the discretised sampling radius expressed in lattice steps.
    pub fn radius_in_steps(&self) -> isize {
        let steps = (self.physical_radius / self.grid_spacing).ceil() as isize;
        steps.max(1)
    }
}

/// Couples microlocal interface signatures into a Z-space control pulse.
///
/// The lift projects oriented perimeter density onto a preferred Z-axis,
/// accumulates Above/Here/Beneath energy, and enriches the signed drift with a
/// [`LeechProjector`] so downstream runtimes can bias their Z control signal
/// using interface geometry alone. When orientations are absent the lift
/// reduces to a neutral pulse (`z_bias = 0`) that still reports boundary
/// support through the Here band.
#[derive(Debug, Clone)]
pub struct InterfaceZLift {
    axis: Vec<f32>,
    projector: LeechProjector,
    bias_gain: f32,
    min_support: f32,
    orientation_floor: f32,
}

impl InterfaceZLift {
    /// Builds a new lift for the provided preferred Z-axis and projector.
    ///
    /// The axis is normalised internally so callers can provide any non-zero
    /// vector aligned with the desired Above direction in the microlocal
    /// lattice.
    pub fn new(axis: &[f32], projector: LeechProjector) -> Self {
        assert!(!axis.is_empty(), "axis must not be empty");
        let mut axis_vec = axis.to_vec();
        let norm = axis_vec
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(norm > f64::EPSILON, "axis must have non-zero length");
        let inv_norm = (norm as f32).recip();
        for value in &mut axis_vec {
            *value *= inv_norm;
        }
        Self {
            axis: axis_vec,
            projector,
            bias_gain: 1.0,
            min_support: 1.0,
            orientation_floor: 0.05,
        }
    }

    /// Scales the enriched Z bias produced by the lift.
    pub fn with_bias_gain(mut self, gain: f32) -> Self {
        self.bias_gain = gain;
        self
    }

    /// Requires a minimum number of interface cells before emitting bias.
    pub fn with_min_support(mut self, min_support: f32) -> Self {
        self.min_support = min_support.max(0.0);
        self
    }

    /// Floor on the oriented drift before the bias becomes active.
    pub fn with_orientation_floor(mut self, floor: f32) -> Self {
        self.orientation_floor = floor.max(0.0);
        self
    }

    /// Returns the current bias gain multiplier applied to the enriched drift.
    pub fn bias_gain(&self) -> f32 {
        self.bias_gain
    }

    /// Overrides the bias gain used during projection.
    pub fn set_bias_gain(&mut self, gain: f32) {
        self.bias_gain = gain.max(0.0);
    }

    /// Returns the minimum oriented drift required before emitting bias.
    pub fn orientation_floor(&self) -> f32 {
        self.orientation_floor
    }

    /// Adjusts the oriented drift floor that guards the bias emission.
    pub fn set_orientation_floor(&mut self, floor: f32) {
        self.orientation_floor = floor.clamp(0.0, 1.0);
    }

    /// Returns the normalised axis used during projection.
    pub fn axis(&self) -> &[f32] {
        &self.axis
    }

    /// Projects the provided microlocal signature into a Z-space pulse.
    pub fn project(&self, signature: &InterfaceSignature) -> InterfaceZPulse {
        let dim = signature.r_machine.ndim();
        assert_eq!(dim, self.axis.len(), "axis dimension mismatch");

        let mut interface_cells = 0.0f32;
        let mut boundary_mass = 0.0f32;
        let mut above = 0.0f32;
        let mut here = 0.0f32;
        let mut beneath = 0.0f32;

        let orientation = signature.orientation.as_ref();
        let mut coord = vec![0usize; dim + 1];

        for idx in indices(signature.r_machine.raw_dim()) {
            let idx_slice = idx.slice();
            let idx_dyn = IxDyn(idx_slice);
            let r_val = signature.r_machine[&idx_dyn];
            if r_val <= 0.0 {
                continue;
            }
            interface_cells += r_val;

            let weight = signature.perimeter_density[&idx_dyn];
            if weight <= f32::EPSILON {
                continue;
            }
            boundary_mass += weight;

            if let Some(field) = orientation {
                coord[1..].clone_from_slice(idx_slice);
                let mut projection = 0.0f32;
                for axis_idx in 0..dim {
                    coord[0] = axis_idx;
                    projection += field[IxDyn(&coord)] * self.axis[axis_idx];
                }
                let projection = projection.clamp(-1.0, 1.0);
                let aligned = projection.max(0.0);
                let anti = (-projection).max(0.0);
                let neutral = (1.0 - aligned - anti).max(0.0);
                above += weight * aligned;
                beneath += weight * anti;
                here += weight * neutral;
            } else {
                here += weight;
            }
        }

        let drift = above - beneath;
        let mut z_bias = 0.0f32;
        if interface_cells >= self.min_support && drift.abs() >= self.orientation_floor {
            let magnitude = f64::from(drift.abs());
            let enriched = self.projector.enrich(magnitude) as f32;
            if enriched > f32::EPSILON && self.bias_gain.abs() > f32::EPSILON {
                z_bias = drift.signum() * enriched * self.bias_gain;
            }
        }

        InterfaceZPulse {
            support: boundary_mass,
            interface_cells,
            band_energy: (above, here, beneath),
            drift,
            z_bias,
        }
    }
}

/// Aggregated Z-space pulse produced by [`InterfaceZLift::project`].
#[derive(Debug, Clone)]
pub struct InterfaceZPulse {
    /// Total perimeter mass supporting the pulse.
    pub support: f32,
    /// Number of interface cells participating in the pulse.
    pub interface_cells: f32,
    /// Above/Here/Beneath energy split produced during projection.
    pub band_energy: (f32, f32, f32),
    /// Signed drift between Above and Beneath energy.
    pub drift: f32,
    /// Signed Z bias generated after enriching the drift.
    pub z_bias: f32,
}

impl InterfaceZPulse {
    /// Returns the total band energy.
    pub fn total_energy(&self) -> f32 {
        let (above, here, beneath) = self.band_energy;
        above + here + beneath
    }

    /// Returns `true` when the pulse carries no support and therefore no
    /// actionable Z-bias signal.
    pub fn is_empty(&self) -> bool {
        self.support <= f32::EPSILON && self.total_energy() <= f32::EPSILON
    }

    /// Aggregates a batch of pulses into a single pulse whose band energies are
    /// the sum of the individual contributions while the Z bias is support
    /// weighted.
    pub fn aggregate(pulses: &[InterfaceZPulse]) -> InterfaceZPulse {
        if pulses.is_empty() {
            return InterfaceZPulse::default();
        }

        let mut support = 0.0f32;
        let mut interface_cells = 0.0f32;
        let mut above = 0.0f32;
        let mut here = 0.0f32;
        let mut beneath = 0.0f32;
        let mut weighted_bias = 0.0f32;

        for pulse in pulses {
            let (p_above, p_here, p_beneath) = pulse.band_energy;
            above += p_above;
            here += p_here;
            beneath += p_beneath;
            interface_cells += pulse.interface_cells;
            support += pulse.support;
            weighted_bias += pulse.z_bias * pulse.support;
        }

        let drift = above - beneath;
        let z_bias = if support > f32::EPSILON {
            weighted_bias / support
        } else {
            0.0
        };

        InterfaceZPulse {
            support,
            interface_cells,
            band_energy: (above, here, beneath),
            drift,
            z_bias,
        }
    }

    /// Blends two pulses using the weight `alpha` for the `next` pulse.
    pub fn lerp(current: &InterfaceZPulse, next: &InterfaceZPulse, alpha: f32) -> InterfaceZPulse {
        let alpha = alpha.clamp(0.0, 1.0);
        if alpha <= f32::EPSILON {
            return current.clone();
        }
        if (1.0 - alpha) <= f32::EPSILON {
            return next.clone();
        }
        let beta = 1.0 - alpha;
        let (cur_above, cur_here, cur_beneath) = current.band_energy;
        let (next_above, next_here, next_beneath) = next.band_energy;
        InterfaceZPulse {
            support: current.support * beta + next.support * alpha,
            interface_cells: current.interface_cells * beta + next.interface_cells * alpha,
            band_energy: (
                cur_above * beta + next_above * alpha,
                cur_here * beta + next_here * alpha,
                cur_beneath * beta + next_beneath * alpha,
            ),
            drift: current.drift * beta + next.drift * alpha,
            z_bias: current.z_bias * beta + next.z_bias * alpha,
        }
    }

    /// Converts the pulse into a [`SoftlogicZFeedback`] record using explicit
    /// ψ and weighted loss totals supplied by the caller.
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
        }
    }

    /// Converts the pulse into a [`SoftlogicZFeedback`] record using the
    /// perimeter mass both as ψ total and weighted loss surrogate.
    pub fn into_softlogic_feedback(self) -> SoftlogicZFeedback {
        let total = self.total_energy();
        SoftlogicZFeedback {
            psi_total: self.support,
            weighted_loss: total,
            band_energy: self.band_energy,
            drift: self.drift,
            z_signal: self.z_bias,
        }
    }

    /// Scales the pulse by `gain`, attenuating the support, band energy, and
    /// derived signals uniformly. Negative gains flip the drift and Z bias but
    /// keep the magnitude scaling consistent.
    pub fn scaled(&self, gain: f32) -> InterfaceZPulse {
        let (above, here, beneath) = self.band_energy;
        InterfaceZPulse {
            support: self.support * gain,
            interface_cells: self.interface_cells * gain,
            band_energy: (above * gain, here * gain, beneath * gain),
            drift: self.drift * gain,
            z_bias: self.z_bias * gain,
        }
    }
}

impl Default for InterfaceZPulse {
    fn default() -> Self {
        InterfaceZPulse {
            support: 0.0,
            interface_cells: 0.0,
            band_energy: (0.0, 0.0, 0.0),
            drift: 0.0,
            z_bias: 0.0,
        }
    }
}

/// Policy hook controlling per-source quality weights and fused adjustments.
pub trait ZSourcePolicy: Send + Sync {
    /// Computes a quality score in `[0, 1]` used to weight the corresponding
    /// pulse during fusion.
    fn quality(&self, pulse: &InterfaceZPulse) -> f32 {
        let total = pulse.total_energy().max(1e-6);
        let drift = (pulse.drift.abs() / total).tanh();
        let support = pulse.support.tanh();
        (drift * support).clamp(0.0, 1.0)
    }

    /// Provides an opportunity to amend the fused pulse once all sources have
    /// been combined.
    fn late_fuse(
        &self,
        _fused: &mut InterfaceZPulse,
        _pulses: &[InterfaceZPulse],
        _qualities: &[f32],
    ) {
    }
}

/// Default quality policy mirroring the stock microlocal heuristics.
#[derive(Debug, Default)]
pub struct DefaultZSourcePolicy;

impl DefaultZSourcePolicy {
    pub fn new() -> Self {
        Self
    }
}

impl ZSourcePolicy for DefaultZSourcePolicy {}

/// Band-energy gating applied on top of the quality policy.
#[derive(Debug, Clone)]
pub struct BandPolicy {
    min_quality: [f32; 3],
    hysteresis: f32,
}

impl BandPolicy {
    /// Creates a new policy with per-band minimum quality thresholds.
    pub fn new(min_quality: [f32; 3]) -> Self {
        BandPolicy {
            min_quality: min_quality.map(|v| v.clamp(0.0, 1.0)),
            hysteresis: 0.0,
        }
    }

    /// Configures a hysteresis margin applied when demoting bands.
    pub fn with_hysteresis(mut self, hysteresis: f32) -> Self {
        self.hysteresis = hysteresis.max(0.0);
        self
    }

    fn gate(&self, band: usize, quality: f32) -> f32 {
        let threshold = self.min_quality[band];
        if quality + self.hysteresis < threshold {
            if threshold <= f32::EPSILON {
                0.0
            } else {
                (quality / threshold).clamp(0.0, 1.0)
            }
        } else {
            1.0
        }
    }

    /// Projects an overall quality multiplier derived from the band energies.
    pub fn project_quality(&self, pulse: &InterfaceZPulse) -> f32 {
        let total = pulse.total_energy().max(1e-6);
        let ratios = [
            pulse.band_energy.0 / total,
            pulse.band_energy.1 / total,
            pulse.band_energy.2 / total,
        ];
        let mut scale = 1.0f32;
        for (idx, ratio) in ratios.into_iter().enumerate() {
            scale *= self.gate(idx, ratio.clamp(0.0, 1.0));
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
    /// Creates a new budget policy bounding the fused Z-bias magnitude.
    pub fn new(z_max: f32) -> Self {
        BudgetPolicy { z_max: z_max.abs() }
    }

    /// Applies the budget to the fused pulse, returning the applied scale.
    pub fn apply(&self, fused: &mut InterfaceZPulse) -> f32 {
        let limit = self.z_max.max(f32::EPSILON);
        let magnitude = fused.z_bias.abs();
        if magnitude <= limit {
            return 1.0;
        }
        let scale = (limit / magnitude).clamp(0.0, 1.0);
        let scaled = fused.scaled(scale);
        *fused = scaled;
        scale
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
    emitter: MicrolocalEmitter,
}

impl InterfaceZConductor {
    /// Creates a new conductor from the provided gauges and lift. The
    /// `smoothing` factor defaults to `1.0`, meaning the fused pulse mirrors the
    /// latest measurement unless [`with_smoothing`] is invoked.
    pub fn new(gauges: Vec<InterfaceGauge>, lift: InterfaceZLift) -> Self {
        assert!(!gauges.is_empty(), "at least one gauge must be supplied");
        let emitter = MicrolocalEmitter::new();
        InterfaceZConductor {
            gauges,
            lift,
            conductor: ZConductor::default(),
            clock: 0,
            emitter,
        }
    }

    /// Enables frequency-domain fusion with the provided configuration.
    pub fn with_frequency(mut self, cfg: ZFrequencyConfig) -> Self {
        self.conductor.set_frequency_config(Some(cfg));
        self
    }

    /// Enables adaptive gain tuning with the supplied configuration.
    pub fn with_adaptive_gain(mut self, cfg: ZAdaptiveGainCfg) -> Self {
        self.conductor.set_adaptive_gain_config(Some(cfg));
        self
    }

    /// Disables frequency-domain fusion.
    pub fn without_frequency(mut self) -> Self {
        self.conductor.set_frequency_config(None);
        self
    }

    /// Configures the exponential smoothing factor `alpha` applied when fusing
    /// subsequent pulses. Values in `[0,1]` blend the previous fused pulse with
    /// the latest measurement; `1` disables smoothing while `0` keeps the
    /// previous fused pulse unchanged.
    pub fn with_smoothing(mut self, alpha: f32) -> Self {
        let alpha = alpha.clamp(0.0, 1.0);
        let cfg = self.conductor.cfg_mut();
        cfg.alpha_slow = alpha;
        cfg.alpha_fast = alpha.max(0.4);
        self
    }

    /// Returns the gauges driven by the conductor.
    pub fn gauges(&self) -> &[InterfaceGauge] {
        &self.gauges
    }

    /// Configures the conductor with a custom policy controlling per-source
    /// quality weighting and post-fusion adjustments.
    pub fn with_policy<P>(mut self, policy: P) -> Self
    where
        P: ZSourcePolicy + 'static,
    {
        self.policy = Arc::new(policy);
        self
    }

    /// Applies a band policy to modulate quality weights using band-energy
    /// heuristics.
    pub fn with_band_policy(mut self, policy: BandPolicy) -> Self {
        self.band_policy = Some(policy);
        self
    }

    /// Enforces a budget over the fused Z signal to prevent runaway drift.
    pub fn with_budget_policy(mut self, policy: BudgetPolicy) -> Self {
        self.budget_policy = Some(policy);
        self
    }

    /// Processes a binary mask through each gauge, lifts the resulting
    /// signatures into Z pulses, and fuses them into a Softlogic feedback
    /// record. When `psi_total` or `weighted_loss` are omitted they default to
    /// the fused support and total energy respectively.
    pub fn step(
        &mut self,
        mask: &ArrayD<f32>,
        c_prime: Option<&ArrayD<f32>>,
        psi_total: Option<f32>,
        weighted_loss: Option<f32>,
    ) -> InterfaceZReport {
        let mut signatures = Vec::with_capacity(self.gauges.len());
        let mut pulses = Vec::with_capacity(self.gauges.len());

        for gauge in &self.gauges {
            let signature = gauge.analyze_with_label(mask, c_prime);
            let pulse = self.lift.project(&signature);
            signatures.push(signature);
            pulses.push(pulse);
        }

        let mut qualities = Vec::with_capacity(pulses.len());
        let mut weighted = Vec::with_capacity(pulses.len());
        for pulse in &pulses {
            let mut quality = self.policy.quality(pulse).clamp(0.0, 1.0);
            if let Some(band_policy) = &self.band_policy {
                quality *= band_policy.project_quality(pulse);
            }
            qualities.push(quality);
            weighted.push(pulse.scaled(quality));
        }

        let fused_raw = InterfaceZPulse::aggregate(&weighted);
        let mut fused = if let Some(prev) = &self.carry {
            InterfaceZPulse::lerp(prev, &fused_raw, self.smoothing)
        } else {
            fused_raw.clone()
        };
        if fused_raw.z_bias.abs() > f32::EPSILON
            && fused.z_bias.signum() != fused_raw.z_bias.signum()
        {
            fused.z_bias = fused_raw.z_bias * self.smoothing;
        }
        self.emitter.extend(z_pulses);
        let mut registry = ZRegistry::with_capacity(1);
        registry.register(self.emitter.clone());
        let fused_z = self.conductor.step_from_registry(&mut registry, now);

        self.policy.late_fuse(&mut fused, &pulses, &qualities);

        let mut budget_scale = 1.0;
        if let Some(budget) = &self.budget_policy {
            budget_scale = budget.apply(&mut fused);
        }

        self.carry = Some(fused.clone());

        let psi = psi_total.unwrap_or(fused.support);
        let loss = weighted_loss.unwrap_or(fused.total_energy());
        let feedback = fused.clone().into_softlogic_feedback_with(psi, loss);

        InterfaceZReport {
            signatures,
            pulses,
            fused_pulse: fused,
            fused_z,
            feedback,
            qualities,
            budget_scale,
        }
    }

    /// Returns the most recent fused pulse emitted by the conductor.
    pub fn last_fused_pulse(&self) -> InterfaceZPulse {
        self.last_pulse
    }
}

impl fmt::Debug for InterfaceZConductor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InterfaceZConductor")
            .field("gauges", &self.gauges.len())
            .field("smoothing", &self.smoothing)
            .field("carry", &self.carry)
            .field("band_policy", &self.band_policy)
            .field("budget_policy", &self.budget_policy)
            .finish()
    }
}

/// Result of a single [`InterfaceZConductor::step`] call, containing both the
/// per-gauge signatures and the fused Z-space feedback.
#[derive(Debug, Clone)]
pub struct InterfaceZReport {
    /// Signatures returned by each gauge, ordered as supplied to the conductor.
    pub signatures: Vec<InterfaceSignature>,
    /// Pulses generated from each signature prior to fusion.
    pub pulses: Vec<InterfaceZPulse>,
    /// Smoothed aggregate pulse after applying fusion and smoothing.
    pub fused_pulse: InterfaceZPulse,
    /// Canonical fused pulse shared with the global Z conductor.
    pub fused_z: ZFused,
    /// Ready-to-store Softlogic feedback record.
    pub feedback: SoftlogicZFeedback,
    /// Quality weights assigned to each pulse prior to fusion.
    pub qualities: Vec<f32>,
    /// Scale factor applied by the budget policy (1.0 when unclamped).
    pub budget_scale: f32,
}

impl CoopAgent for InterfaceZConductor {
    fn propose(&mut self) -> CoopProposal {
        let fused = self.last_fused_pulse();
        let (above, here, beneath) = fused.band_energy;
        let there = (above + beneath).max(1e-6);
        let here_ratio = here / there;
        if fused.total_energy() <= f32::EPSILON {
            return CoopProposal::neutral();
        }
        let weight = (fused.support + there).max(1e-3) * here_ratio.max(0.0);
        CoopProposal::new(fused.z_bias, weight)
    }

    fn observe(&mut self, team_reward: f32, credit: f32) {
        let fused = self.last_fused_pulse();
        let (above, here, beneath) = fused.band_energy;
        let there = (above + beneath).max(1e-3);
        let curvature_ratio = here / there;
        let imbalance = (above - beneath) / there;

        let mut bias_gain = self.lift.bias_gain();
        let credit_push = (credit * (1.0 - curvature_ratio)).clamp(-1.0, 1.0);
        bias_gain = (bias_gain + 0.08 * credit_push).clamp(0.05, 8.0);
        self.lift.set_bias_gain(bias_gain);

        let mut floor = self.lift.orientation_floor();
        let reward_push = team_reward.tanh();
        floor = (floor * (1.0 - 0.05 * reward_push)).clamp(0.01, 0.6);
        let drift_adjust = (imbalance * credit).clamp(-1.0, 1.0);
        floor = (floor - 0.015 * drift_adjust).clamp(0.01, 0.6);
        self.lift.set_orientation_floor(floor);
    }
}

impl InterfaceZReport {
    /// Returns `true` when any gauge detected an interface.
    pub fn has_interface(&self) -> bool {
        self.signatures
            .iter()
            .any(InterfaceSignature::has_interface)
    }
}

fn unit_sphere_area(dim: usize) -> f32 {
    let d = dim as f64;
    let area = 2.0 * PI.powf(d / 2.0) / gamma(d / 2.0);
    area as f32
}

fn radius_in_steps(physical_radius: f32, grid_spacing: f32) -> isize {
    let radius = physical_radius.max(f32::EPSILON);
    let spacing = grid_spacing.max(f32::EPSILON);
    let steps = (radius / spacing).ceil() as isize;
    steps.max(1)
}

fn generate_offsets(dim: usize, radius: isize) -> Vec<Vec<isize>> {
    fn recurse(acc: &mut Vec<Vec<isize>>, cur: &mut Vec<isize>, dim: usize, radius: isize) {
        if cur.len() == dim {
            acc.push(cur.clone());
            return;
        }
        for step in -radius..=radius {
            cur.push(step);
            recurse(acc, cur, dim, radius);
            cur.pop();
        }
    }
    let mut acc = Vec::new();
    let mut scratch = Vec::with_capacity(dim);
    recurse(&mut acc, &mut scratch, dim, radius);
    acc
}

fn neighbor_sample(
    field: &ArrayD<f32>,
    idx: &[usize],
    shape: &[usize],
    offset: &[isize],
) -> Option<f32> {
    let mut coord = Vec::with_capacity(idx.len());
    for (axis, &delta) in offset.iter().enumerate() {
        let pos = idx[axis] as isize + delta;
        if pos < 0 || pos >= shape[axis] as isize {
            return None;
        }
        coord.push(pos as usize);
    }
    Some(field[IxDyn(&coord)])
}

fn local_raw_density(
    mask: &ArrayD<f32>,
    idx: &[usize],
    shape: &[usize],
    threshold: f32,
    center: f32,
) -> (f32, bool) {
    let dim = idx.len();
    let mut diff_count = 0.0f32;
    let mut neighbor_total = 0.0f32;
    for axis in 0..dim {
        if idx[axis] + 1 < shape[axis] {
            neighbor_total += 1.0;
            let mut coord = idx.to_vec();
            coord[axis] += 1;
            let neigh = mask[IxDyn(&coord)];
            if (neigh - center).abs() >= threshold {
                diff_count += 1.0;
            }
        }
        if idx[axis] > 0 {
            neighbor_total += 1.0;
            let mut coord = idx.to_vec();
            coord[axis] -= 1;
            let neigh = mask[IxDyn(&coord)];
            if (neigh - center).abs() >= threshold {
                diff_count += 1.0;
            }
        }
    }
    if neighbor_total > 0.0 {
        let raw = diff_count / neighbor_total;
        (raw, diff_count > 0.0)
    } else {
        (0.0, false)
    }
}

fn oriented_normal(
    label: &ArrayD<f32>,
    idx: &[usize],
    shape: &[usize],
    h: f32,
    threshold: f32,
) -> Option<Vec<f32>> {
    let dim = idx.len();
    let mut grad = vec![0.0f32; dim];
    let mut magnitudes = vec![0.0f32; dim];
    let center = label[IxDyn(idx)];
    for axis in 0..dim {
        let forward = if idx[axis] + 1 < shape[axis] {
            let mut coord = idx.to_vec();
            coord[axis] += 1;
            label[IxDyn(&coord)]
        } else {
            center
        };
        let backward = if idx[axis] > 0 {
            let mut coord = idx.to_vec();
            coord[axis] -= 1;
            label[IxDyn(&coord)]
        } else {
            center
        };
        let forward_delta = forward - center;
        let backward_delta = center - backward;
        if forward_delta.abs() >= backward_delta.abs() {
            grad[axis] = forward_delta / h;
            magnitudes[axis] = forward_delta.abs();
        } else {
            grad[axis] = backward_delta / h;
            magnitudes[axis] = backward_delta.abs();
        }
    }
    let max_mag = magnitudes.iter().copied().fold(0.0f32, f32::max);
    if max_mag < threshold {
        return None;
    }
    let mut first_axis = None;
    for axis in 0..dim {
        if magnitudes[axis] + f32::EPSILON < max_mag {
            grad[axis] = 0.0;
        } else if (magnitudes[axis] - max_mag).abs() <= f32::EPSILON {
            if let Some(primary) = first_axis {
                if axis != primary {
                    grad[axis] = 0.0;
                }
            } else {
                first_axis = Some(axis);
            }
        }
    }
    let norm_sq = grad.iter().map(|g| (*g as f64).powi(2)).sum::<f64>() as f32;
    if norm_sq.sqrt() < threshold {
        return None;
    }
    let norm = norm_sq.sqrt().max(f32::EPSILON);
    for g in &mut grad {
        *g /= norm;
    }
    Some(grad)
}

fn mean_curvature_from_field(
    field: &ArrayD<f32>,
    idx: &[usize],
    shape: &[usize],
    h: f32,
    threshold: f32,
) -> Option<(f32, f32)> {
    let dim = idx.len();
    let center_normal = normalized_gradient(field, idx, shape, h, threshold)?;
    let mut divergence = 0.0f32;
    for axis in 0..dim {
        let forward = normalized_gradient_offset(field, idx, shape, axis, 1, h, threshold)
            .unwrap_or_else(|| center_normal.clone());
        let backward = normalized_gradient_offset(field, idx, shape, axis, -1, h, threshold)
            .unwrap_or_else(|| center_normal.clone());
        let derivative = (forward[axis] - backward[axis]) / (2.0 * h);
        if derivative.is_finite() {
            divergence += derivative;
        }
    }
    let signed = divergence;
    let magnitude = divergence.abs();
    Some((magnitude, signed))
}

fn normalized_gradient(
    field: &ArrayD<f32>,
    idx: &[usize],
    shape: &[usize],
    h: f32,
    threshold: f32,
) -> Option<Vec<f32>> {
    let dim = idx.len();
    let mut grad = vec![0.0f32; dim];
    let center = field[IxDyn(idx)];
    for axis in 0..dim {
        let forward = if idx[axis] + 1 < shape[axis] {
            let mut coord = idx.to_vec();
            coord[axis] += 1;
            field[IxDyn(&coord)]
        } else {
            center
        };
        let backward = if idx[axis] > 0 {
            let mut coord = idx.to_vec();
            coord[axis] -= 1;
            field[IxDyn(&coord)]
        } else {
            center
        };
        grad[axis] = match (idx[axis] > 0, idx[axis] + 1 < shape[axis]) {
            (true, true) => (forward - backward) / (2.0 * h),
            (false, true) => (forward - center) / h,
            (true, false) => (center - backward) / h,
            (false, false) => 0.0,
        };
    }
    let norm_sq = grad.iter().map(|g| (*g as f64).powi(2)).sum::<f64>() as f32;
    if norm_sq.sqrt() < threshold {
        return None;
    }
    let norm = norm_sq.sqrt().max(f32::EPSILON);
    for g in &mut grad {
        *g /= norm;
    }
    Some(grad)
}

fn normalized_gradient_offset(
    field: &ArrayD<f32>,
    idx: &[usize],
    shape: &[usize],
    axis: usize,
    delta: isize,
    h: f32,
    threshold: f32,
) -> Option<Vec<f32>> {
    let pos = idx[axis] as isize + delta;
    if pos < 0 || pos >= shape[axis] as isize {
        return None;
    }
    let mut coord = idx.to_vec();
    coord[axis] = pos as usize;
    normalized_gradient(field, &coord, shape, h, threshold)
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
        assert!(normal_y.abs() > 0.5);
        assert!(normal_x.abs() < 1e-3);
    }

    #[test]
    fn z_lift_produces_oriented_bias() {
        let mask = array![[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]].into_dyn();
        let c_prime = mask.mapv(|v| if v > 0.5 { 1.0 } else { -1.0 });
        let gauge = InterfaceGauge::new(1.0, 1.0);
        let sig = gauge.analyze_with_label(&mask, Some(&c_prime));
        let projector = LeechProjector::new(24, 0.5);
        let lift = InterfaceZLift::new(&[1.0, 0.0], projector);
        let pulse = lift.project(&sig);
        let (above, here, beneath) = pulse.band_energy;
        assert!(above > beneath);
        assert!(here >= 0.0);
        assert!(pulse.z_bias > 0.0);
        let feedback = pulse.clone().into_softlogic_feedback();
        assert_eq!(feedback.band_energy, pulse.band_energy);
        assert_eq!(feedback.z_signal, pulse.z_bias);
    }

    #[test]
    fn z_lift_remains_neutral_without_orientation() {
        let mask = array![[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]].into_dyn();
        let gauge = InterfaceGauge::new(1.0, 1.0);
        let sig = gauge.analyze(&mask);
        let projector = LeechProjector::new(24, 0.5);
        let lift = InterfaceZLift::new(&[0.0, 1.0], projector);
        let pulse = lift.project(&sig);
        let (above, here, beneath) = pulse.band_energy;
        assert!(above <= f32::EPSILON);
        assert!(beneath <= f32::EPSILON);
        assert!(here > 0.0);
        assert_eq!(pulse.z_bias, 0.0);
    }

    #[test]
    fn curvature_detects_flat_and_curved_interfaces() {
        let flat = array![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]].into_dyn();
        let gauge = InterfaceGauge::new(1.0, 1.0);
        let flat_sig = gauge.analyze(&flat);
        let flat_curvature = flat_sig.mean_curvature[IxDyn(&[1, 1])];
        assert!(
            flat_curvature.abs() < 1e-3,
            "flat interface should be near-zero curvature"
        );

        let curved = array![
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]
        .mapv(|v| v as f32)
        .into_dyn();
        let curved_sig = gauge.analyze(&curved);
        let max_curvature = curved_sig
            .mean_curvature
            .iter()
            .fold(0.0f32, |acc, v| acc.max(*v));
        assert!(
            max_curvature > 0.05,
            "curved interface should register positive curvature"
        );
    }

    #[test]
    fn multiradius_analysis_returns_distinct_signatures() {
        let mask = array![
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
        ]
        .into_dyn();
        let gauge = InterfaceGauge::new(1.0, 1.5);
        let c_prime = mask.mapv(|v| if v > 0.5 { 1.0 } else { -1.0 });
        let signatures = gauge.analyze_multiradius(&mask, Some(&c_prime), &[0.75, 1.5, 2.5]);
        assert_eq!(signatures.len(), 3);
        assert!(signatures[0].radius <= signatures[1].radius);
        assert!(signatures[1].radius <= signatures[2].radius);
        assert!((signatures[1].physical_radius - 1.5).abs() < 1e-6);
        assert!(signatures.iter().all(|sig| sig.orientation.is_some()));
    }

    #[test]
    fn conductor_fuses_multiscale_pulses_with_smoothing() {
        let mask = array![
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
        ]
        .into_dyn();
        let mut flipped = mask.clone();
        flipped[[1, 2]] = 0.0;
        flipped[[2, 2]] = 0.0;
        let c_prime = mask.mapv(|v| if v > 0.5 { 1.0 } else { -1.0 });
        let c_prime_neg = c_prime.mapv(|v| -v);

        let gauge_fine = InterfaceGauge::new(1.0, 1.0);
        let gauge_coarse = InterfaceGauge::new(1.0, 2.5);
        let projector = LeechProjector::new(24, 0.5);
        let lift = InterfaceZLift::new(&[1.0, 0.0], projector).with_bias_gain(0.5);
        let mut conductor =
            InterfaceZConductor::new(vec![gauge_fine, gauge_coarse], lift).with_smoothing(0.5);

        let first = conductor.step(&mask, Some(&c_prime), None, None);
        assert!(first.has_interface());
        assert!(first.fused_pulse.z_bias > 0.0);
        assert_eq!(first.qualities.len(), first.pulses.len());
        assert!(first.budget_scale > 0.0);

        let second = conductor.step(&flipped, Some(&c_prime_neg), None, None);
        let raw_second = InterfaceZPulse::aggregate(&second.pulses);
        assert!(raw_second.z_bias < 0.0);
        assert!(second.fused_z.events.iter().any(|e| e == "flip-held"));
        assert!(second.fused_z.z > raw_second.z_bias);
        assert_eq!(second.feedback.band_energy, second.fused_pulse.band_energy);
        assert_eq!(second.qualities.len(), second.pulses.len());
        assert!(second.budget_scale > 0.0);
    }

    #[derive(Debug)]
    struct HalfPolicy;

    impl ZSourcePolicy for HalfPolicy {
        fn quality(&self, _: &InterfaceZPulse) -> f32 {
            0.5
        }
    }

    #[test]
    fn custom_policy_scales_fused_pulse() {
        let mask = array![[0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0],].into_dyn();
        let c_prime = mask.mapv(|v| if v > 0.5 { 1.0 } else { -1.0 });
        let gauge = InterfaceGauge::new(1.0, 1.0);
        let projector = LeechProjector::new(16, 0.5);
        let lift = InterfaceZLift::new(&[1.0, 0.0], projector);
        let mut conductor = InterfaceZConductor::new(vec![gauge], lift).with_policy(HalfPolicy);

        let report = conductor.step(&mask, Some(&c_prime), None, None);
        let raw = InterfaceZPulse::aggregate(&report.pulses);
        assert_eq!(report.qualities.len(), 1);
        let quality = report.qualities[0];
        assert!((quality - 0.5).abs() < 1e-6);
        assert!((report.fused_pulse.support - raw.support * 0.5).abs() < 1e-6);
        assert!((report.fused_pulse.z_bias - raw.z_bias * 0.5).abs() < 1e-6);
    }

    #[test]
    fn budget_policy_clamps_bias() {
        let mask = array![[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0],].into_dyn();
        let c_prime = mask.mapv(|v| if v > 0.5 { 1.0 } else { -1.0 });
        let gauge = InterfaceGauge::new(1.0, 1.0);
        let projector = LeechProjector::new(16, 0.5);
        let lift = InterfaceZLift::new(&[1.0, 0.0], projector);
        let mut conductor =
            InterfaceZConductor::new(vec![gauge], lift).with_budget_policy(BudgetPolicy::new(0.02));

        let report = conductor.step(&mask, Some(&c_prime), None, None);
        assert!(report.budget_scale <= 1.0);
        assert!(report.fused_pulse.z_bias.abs() <= 0.02 + 1e-6);
    }

    #[test]
    fn band_policy_demotes_unbalanced_energy() {
        let pulse = InterfaceZPulse {
            support: 1.0,
            interface_cells: 1.0,
            band_energy: (0.9, 0.05, 0.05),
            drift: 0.4,
            z_bias: 0.3,
        };
        let policy = BandPolicy::new([0.2, 0.2, 0.2]);
        let quality = policy.project_quality(&pulse);
        assert!(quality < 1.0);
    }
}
