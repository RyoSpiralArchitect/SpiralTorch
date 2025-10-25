// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// =============================================================================
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
// =============================================================================

//! Maxwell-coded envelope utilities for injecting sequential evidence into
//! Z-space detectors.
//!
//! This module turns the memo contained in
//! `docs/coded_envelope_maxwell_model.md` into executable primitives that the
//! rest of SpiralTorch can call.  The goal is to instrument experiments that
//! vary shielding, distance, and polarisation, estimate the physical gain, and
//! accumulate block-level matched-filter scores while emitting online Z-stats.

use core::f64::consts::PI;

#[cfg(feature = "psi")]
use crate::telemetry::{
    hub,
    psi::{PsiComponent, PsiEvent, PsiReading},
};
use crate::{
    coop::ai::{CoopAgent, CoopProposal},
    telemetry::hub::SoftlogicZFeedback,
    theory::zpulse::{ZEmitter, ZPulse, ZSource, ZSupport},
    util::math::LeechProjector,
};
use std::cell::RefCell;
#[cfg(feature = "psi")]
use std::collections::HashMap;
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

#[derive(Clone, Default, Debug)]
pub struct MaxwellEmitter {
    queue: Arc<Mutex<VecDeque<ZPulse>>>,
}

impl MaxwellEmitter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enqueue(&self, pulse: ZPulse) {
        let mut queue = self.queue.lock().expect("maxwell emitter queue poisoned");
        queue.push_back(pulse);
    }

    pub fn enqueue_maxwell(&self, pulse: MaxwellZPulse) {
        self.enqueue(pulse.into());
    }

    pub fn extend<I>(&self, pulses: I)
    where
        I: IntoIterator<Item = ZPulse>,
    {
        let mut queue = self.queue.lock().expect("maxwell emitter queue poisoned");
        queue.extend(pulses);
    }
}

impl ZEmitter for MaxwellEmitter {
    fn name(&self) -> ZSource {
        ZSource::Maxwell
    }

    fn tick(&mut self, _now: u64) -> Option<ZPulse> {
        self.queue
            .lock()
            .expect("maxwell emitter queue poisoned")
            .pop_front()
    }
}

/// Consolidated Maxwell gain for a single coded envelope channel.
///
/// The gain aggregates all physical dependencies described in the technical
/// note:
///
/// ```text
/// λ = γ α |H_tis(ω_c)| S G pol / r
/// ```
///
/// where `S = 10^{-S_dB/20}` is the linear shielding factor,
/// `pol = |cos(θ)|` the polarisation alignment, and `r` the separation in
/// metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaxwellFingerprint {
    /// Non-linear demodulation efficiency γ.
    pub gamma: f64,
    /// Modulation depth α injected at the transmitter.
    pub modulation_depth: f64,
    /// Magnitude of the tissue transfer function at the carrier |H_tis(ω_c)|.
    pub tissue_response: f64,
    /// Shielding in decibels. Positive values attenuate the signal.
    pub shielding_db: f64,
    /// Transmit gain consolidating antenna/directivity terms.
    pub transmit_gain: f64,
    /// Polarisation angle θ between the transmitter and the tissue response
    /// (radians).
    pub polarization_angle: f64,
    /// Separation between transmitter and receiver (metres).
    pub distance_m: f64,
}

impl MaxwellFingerprint {
    /// Creates a new fingerprint. Input parameters are clamped to their
    /// physically meaningful ranges so that downstream computations never
    /// produce NaNs due to negative distances or modulation depths.
    pub fn new(
        gamma: f64,
        modulation_depth: f64,
        tissue_response: f64,
        shielding_db: f64,
        transmit_gain: f64,
        polarization_angle: f64,
        distance_m: f64,
    ) -> Self {
        Self {
            gamma: gamma.max(0.0),
            modulation_depth: modulation_depth.max(0.0),
            tissue_response: tissue_response.max(0.0),
            shielding_db,
            transmit_gain: transmit_gain.max(0.0),
            polarization_angle,
            distance_m: distance_m.max(f64::EPSILON),
        }
    }

    /// Returns the shielding attenuation factor \(\mathcal S = 10^{-S/20}\).
    pub fn shielding_factor(&self) -> f64 {
        10f64.powf(-self.shielding_db / 20.0)
    }

    /// Returns the absolute polarisation alignment `|cos θ|`.
    pub fn polarization_alignment(&self) -> f64 {
        self.polarization_angle.cos().abs()
    }

    /// Computes the consolidated Maxwell gain λ.
    pub fn lambda(&self) -> f64 {
        self.gamma
            * self.modulation_depth
            * self.tissue_response
            * self.shielding_factor()
            * self.transmit_gain
            * self.polarization_alignment()
            / self.distance_m
    }

    /// Returns the expected matched-filter mean for a block when the detector
    /// kernel contributes `κ`.
    pub fn expected_block_mean(&self, kappa: f64) -> f64 {
        kappa * self.lambda()
    }
}

/// Semantic gating applied on top of the physical Maxwell fingerprint.
///
/// The gate implements the minimal extension described in the memo:
///
/// ```text
/// u_tot(t) = [λ + μ ρ(t)] c(t)
/// ```
///
/// where `ρ` is clamped to the admissible interval [-1, 1].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeaningGate {
    /// Physical Maxwell gain λ.
    pub physical_gain: f64,
    /// Semantic coupling strength μ.
    pub semantic_gain: f64,
}

impl MeaningGate {
    /// Constructs a new meaning gate with the provided physical and semantic
    /// components.
    pub fn new(physical_gain: f64, semantic_gain: f64) -> Self {
        Self {
            physical_gain,
            semantic_gain: semantic_gain.max(0.0),
        }
    }

    /// Returns the instantaneous envelope coefficient `(λ + μ ρ)` for the
    /// supplied semantic alignment ρ.
    pub fn envelope(&self, rho: f64) -> f64 {
        let rho = rho.clamp(-1.0, 1.0);
        self.physical_gain + self.semantic_gain * rho
    }
}

/// Online estimator for sequential Z statistics built from matched-filter
/// scores.
#[derive(Clone, Debug, Default)]
pub struct SequentialZ {
    count: u64,
    mean: f64,
    m2: f64,
}

impl SequentialZ {
    /// Creates a new sequential Z tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of accumulated blocks.
    pub fn len(&self) -> u64 {
        self.count
    }

    /// Returns `true` when no samples have been observed.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Pushes a new matched-filter score and returns the current Z-statistic if
    /// it is defined. Z is only defined for `count >= 2` because we rely on the
    /// sample variance.
    pub fn push(&mut self, sample: f64) -> Option<f64> {
        self.count += 1;
        let delta = sample - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = sample - self.mean;
        self.m2 += delta * delta2;
        self.z_stat()
    }

    /// Returns the sample mean.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Returns the unbiased sample variance when available.
    pub fn variance(&self) -> Option<f64> {
        if self.count < 2 {
            None
        } else {
            Some(self.m2 / (self.count as f64 - 1.0))
        }
    }

    /// Returns the estimated standard error `σ̂/√N` when enough samples have
    /// been observed.
    pub fn standard_error(&self) -> Option<f64> {
        self.variance()
            .map(|var| (var / self.count as f64).sqrt())
            .filter(|stderr| stderr.is_finite() && *stderr > 0.0)
    }

    /// Returns the Z statistic `(\bar s) / (σ̂/√N)` when defined.
    pub fn z_stat(&self) -> Option<f64> {
        self.standard_error().map(|stderr| self.mean / stderr)
    }
}

/// Projects sequential Maxwell evidence into a Z-space control pulse.
#[derive(Clone, Debug)]
pub struct MaxwellZProjector {
    projector: LeechProjector,
    bias_gain: f32,
    min_blocks: u64,
    min_z_magnitude: f64,
    last_pulse: RefCell<Option<MaxwellZPulse>>,
}

impl MaxwellZProjector {
    /// Creates a projector with the provided Leech rank and weighting factor.
    pub fn new(rank: usize, weight: f64) -> Self {
        Self {
            projector: LeechProjector::new(rank, weight),
            bias_gain: 1.0,
            min_blocks: 2,
            min_z_magnitude: 0.0,
            last_pulse: RefCell::new(None),
        }
    }

    /// Scales the Z bias emitted once the matched filter shows a clear drift.
    pub fn with_bias_gain(mut self, bias_gain: f32) -> Self {
        self.bias_gain = bias_gain;
        self
    }

    /// Requires a minimum block count before emitting a Z-space pulse.
    pub fn with_min_blocks(mut self, min_blocks: u64) -> Self {
        self.min_blocks = min_blocks.max(1);
        self
    }

    /// Requires a minimum |Z| magnitude before emitting a Z-space pulse.
    pub fn with_min_z(mut self, min_z_magnitude: f64) -> Self {
        self.min_z_magnitude = min_z_magnitude.max(0.0);
        self
    }

    /// Projects the accumulated sequential statistic into a control pulse.
    pub fn project(&self, tracker: &SequentialZ) -> Option<MaxwellZPulse> {
        if tracker.len() < self.min_blocks {
            return None;
        }
        let z_score = tracker.z_stat()?;
        if z_score.abs() < self.min_z_magnitude {
            return None;
        }
        let stderr = tracker.standard_error()?;
        let mean = tracker.mean();

        let enriched = self.projector.enrich(z_score.abs()) as f32;
        let z_bias = if enriched > f32::EPSILON && self.bias_gain.abs() > f32::EPSILON {
            z_score.signum() as f32 * enriched * self.bias_gain
        } else {
            0.0
        };

        let above = mean.max(0.0) as f32;
        let beneath = (-mean).max(0.0) as f32;
        let here = stderr.max(0.0) as f32;

        let pulse = MaxwellZPulse {
            blocks: tracker.len(),
            mean,
            standard_error: stderr,
            z_score,
            band_energy: (above, here, beneath),
            z_bias,
        };
        self.last_pulse.replace(Some(pulse.clone()));
        Some(pulse)
    }

    /// Returns the most recent pulse emitted after enrichment.
    pub fn last_pulse(&self) -> Option<MaxwellZPulse> {
        self.last_pulse.borrow().clone()
    }
}

/// Z-space pulse generated from a sequential Maxwell tracker.
#[derive(Clone, Debug, PartialEq)]
pub struct MaxwellZPulse {
    /// Number of matched-filter blocks accumulated so far.
    pub blocks: u64,
    /// Sample mean of the matched-filter statistic.
    pub mean: f64,
    /// Estimated standard error of the mean.
    pub standard_error: f64,
    /// Z score produced by the sequential tracker.
    pub z_score: f64,
    /// Above/Here/Beneath energy tuple summarising the drift direction.
    pub band_energy: (f32, f32, f32),
    /// Signed Z-space bias emitted after enrichment.
    pub z_bias: f32,
}

impl MaxwellZPulse {
    /// Returns the absolute Z score driving the projection.
    pub fn magnitude(&self) -> f64 {
        self.z_score.abs()
    }

    /// Converts the pulse into a [`SoftlogicZFeedback`] packet with the provided context.
    pub fn into_softlogic_feedback(self, psi_total: f32, weighted_loss: f32) -> SoftlogicZFeedback {
        SoftlogicZFeedback {
            psi_total,
            weighted_loss,
            band_energy: self.band_energy,
            drift: self.mean as f32,
            z_signal: self.z_bias,
            scale: None,
            events: Vec::new(),
            attributions: Vec::new(),
            elliptic: None,
        }
    }
}

impl From<MaxwellZPulse> for ZPulse {
    fn from(pulse: MaxwellZPulse) -> Self {
        let support = ZSupport::new(
            pulse.band_energy.0,
            pulse.band_energy.1,
            pulse.band_energy.2,
        );
        let drift = pulse.mean as f32;
        let latency_ms = pulse.blocks as f32;
        let stderr = pulse.standard_error.max(0.0) as f32;
        let quality = {
            let snr = if stderr > 0.0 {
                (1.0 / stderr).min(1.0)
            } else {
                1.0
            };
            let z = pulse.z_score.abs() as f32;
            z.tanh() * snr
        };
        ZPulse {
            source: ZSource::Maxwell,
            ts: pulse.blocks,
            tempo: pulse.mean as f32,
            band_energy: pulse.band_energy,
            drift,
            z_bias: pulse.z_bias,
            support,
            scale: None,
            quality,
            stderr,
            latency_ms,
        }
    }
}

impl CoopAgent for MaxwellZProjector {
    fn propose(&mut self) -> CoopProposal {
        match self.last_pulse() {
            Some(pulse) => {
                let magnitude = pulse.magnitude() as f32;
                let weight = (magnitude.max(1e-3) + pulse.band_energy.1).max(1e-3);
                CoopProposal::new(pulse.z_bias, weight)
            }
            None => CoopProposal::neutral(),
        }
    }

    fn observe(&mut self, team_reward: f32, credit: f32) {
        let magnitude = self
            .last_pulse()
            .map(|pulse| pulse.magnitude() as f32)
            .unwrap_or(0.0);
        let credit_push = (credit / (1.0 + magnitude)).clamp(-1.0, 1.0);
        self.bias_gain = (self.bias_gain + 0.15 * credit_push).clamp(0.1, 10.0);

        let reward_push = team_reward.tanh() as f64;
        self.min_z_magnitude = (self.min_z_magnitude * (1.0 - 0.05 * reward_push)).clamp(0.0, 6.0);
    }
}

#[cfg(feature = "psi")]
/// Publishes Maxwell pulses into the PSI telemetry bridge so the desire
/// lagrangian can observe coded-envelope bias without bespoke glue code.
#[cfg(feature = "psi")]
#[derive(Clone, Debug)]
pub struct MaxwellPsiTelemetryBridge {
    psi_gain: f32,
    loss_gain: f32,
    band_threshold: f32,
}

#[cfg(feature = "psi")]
impl MaxwellPsiTelemetryBridge {
    /// Creates a new bridge with unit PSI and loss gains.
    pub fn new() -> Self {
        Self {
            psi_gain: 1.0,
            loss_gain: 1.0,
            band_threshold: 0.0,
        }
    }

    /// Scales the PSI total injected into the telemetry hub.
    pub fn with_psi_gain(mut self, psi_gain: f32) -> Self {
        self.psi_gain = psi_gain.max(0.0);
        self
    }

    /// Scales the weighted loss recorded alongside the feedback pulse.
    pub fn with_loss_gain(mut self, loss_gain: f32) -> Self {
        self.loss_gain = loss_gain.max(0.0);
        self
    }

    /// Emits a band-energy threshold crossing event when the accumulated energy
    /// meets or exceeds the provided value. Set to zero to disable.
    pub fn with_band_threshold(mut self, threshold: f32) -> Self {
        self.band_threshold = threshold.max(0.0);
        self
    }

    /// Converts the Maxwell pulse into PSI/SoftLogic telemetry and stores it in
    /// the global hub. The returned feedback can be reused immediately by the
    /// caller when additional processing is required.
    pub fn publish(&self, pulse: &MaxwellZPulse, step: u64) -> SoftlogicZFeedback {
        self.publish_with_reading(pulse, step).into_feedback()
    }

    /// Publishes the telemetry and returns the generated PSI reading alongside any events.
    pub fn publish_with_reading(&self, pulse: &MaxwellZPulse, step: u64) -> PublishedPsiTelemetry {
        let (reading, events, feedback) = self.synthesise_feedback(pulse, step);
        hub::set_last_psi(&reading);
        hub::set_last_psi_events(&events);
        hub::set_softlogic_z(feedback.clone());
        PublishedPsiTelemetry::new(feedback, reading, events)
    }

    fn synthesise_feedback(
        &self,
        pulse: &MaxwellZPulse,
        step: u64,
    ) -> (PsiReading, Vec<PsiEvent>, SoftlogicZFeedback) {
        let sanitise_energy = |value: f32| -> f32 {
            if value.is_finite() {
                value.max(0.0)
            } else {
                0.0
            }
        };

        let (raw_above, raw_here, raw_beneath) = pulse.band_energy;
        let above = sanitise_energy(raw_above);
        let here = sanitise_energy(raw_here);
        let beneath = sanitise_energy(raw_beneath);
        let band_total = sanitise_energy(above + here + beneath);
        let psi_total = sanitise_energy(band_total * self.psi_gain);

        let weighted_loss = {
            let magnitude = pulse.magnitude() as f32;
            let scaled = (if magnitude.is_finite() {
                magnitude.max(0.0)
            } else {
                0.0
            }) * self.loss_gain;
            if scaled.is_finite() {
                scaled.max(0.0)
            } else {
                0.0
            }
        };

        let mut breakdown = HashMap::new();
        breakdown.insert(PsiComponent::BAND_ENERGY, band_total);

        let reading = PsiReading {
            total: psi_total,
            breakdown,
            step,
        };

        let mut events = Vec::new();
        if self.band_threshold > 0.0 && band_total >= self.band_threshold {
            events.push(PsiEvent::ThresholdCross {
                component: PsiComponent::BAND_ENERGY,
                value: band_total,
                threshold: self.band_threshold,
                up: pulse.z_score >= 0.0,
                step,
            });
        }

        let mut feedback = pulse
            .clone()
            .into_softlogic_feedback(psi_total, weighted_loss);
        feedback.set_events(events.iter().map(|event| event.to_string()));
        feedback.set_attributions([(ZSource::Maxwell, band_total)]);

        (reading, events, feedback)
    }
}

#[cfg(feature = "psi")]
/// Bundled telemetry emitted by [`MaxwellPsiTelemetryBridge::publish_with_reading`].
#[derive(Clone, Debug)]
pub struct PublishedPsiTelemetry {
    feedback: SoftlogicZFeedback,
    reading: PsiReading,
    events: Vec<PsiEvent>,
}

#[cfg(feature = "psi")]
impl PublishedPsiTelemetry {
    /// Creates a new telemetry bundle with the provided components.
    pub fn new(feedback: SoftlogicZFeedback, reading: PsiReading, events: Vec<PsiEvent>) -> Self {
        Self {
            feedback,
            reading,
            events,
        }
    }

    /// Returns the stored [`SoftlogicZFeedback`] sample.
    pub fn feedback(&self) -> &SoftlogicZFeedback {
        &self.feedback
    }

    /// Returns the stored [`PsiReading`].
    pub fn reading(&self) -> &PsiReading {
        &self.reading
    }

    /// Returns the PSI events captured during publication.
    pub fn events(&self) -> &[PsiEvent] {
        self.events.as_slice()
    }

    /// Consumes the bundle and returns just the feedback packet.
    pub fn into_feedback(self) -> SoftlogicZFeedback {
        self.feedback
    }

    /// Consumes the bundle and exposes all components.
    pub fn into_parts(self) -> (SoftlogicZFeedback, PsiReading, Vec<PsiEvent>) {
        (self.feedback, self.reading, self.events)
    }
}

/// Hint that can be injected into SpiralK so coded-envelope experiments can
/// steer the runtime once a Maxwell pulse has been detected.
#[derive(Clone, Debug, PartialEq)]
pub struct MaxwellSpiralKHint {
    /// Sanitised channel label used inside the KDSl snippet.
    pub channel: String,
    /// Number of matched-filter blocks that produced the pulse.
    pub blocks: u64,
    /// Z score reported by the sequential detector.
    pub z_score: f64,
    /// Signed Z-space bias that should be promoted to SpiralK.
    pub z_bias: f32,
    /// Heuristic weight attached to the hint (0–1 range).
    pub weight: f32,
}

impl MaxwellSpiralKHint {
    /// Returns the SpiralK `soft(...)` line encoded by the hint.
    pub fn script_line(&self) -> String {
        format!(
            "soft (maxwell.bias, {:.6}, {:.3}, channel == \"{}\" && blocks >= {});",
            self.z_bias, self.weight, self.channel, self.blocks
        )
    }
}

/// Aggregates Maxwell pulses and converts them into SpiralK snippets.
#[derive(Clone, Debug, Default)]
pub struct MaxwellSpiralKBridge {
    base_program: Option<String>,
    min_weight: f32,
    max_weight: f32,
    hints: Vec<MaxwellSpiralKHint>,
}

impl MaxwellSpiralKBridge {
    /// Creates a bridge that maps Maxwell pulses into SpiralK hints.
    pub fn new() -> Self {
        Self {
            base_program: None,
            min_weight: 0.55,
            max_weight: 0.95,
            hints: Vec::new(),
        }
    }

    /// Sets the KDSl snippet that should be prepended to the generated hints.
    pub fn with_base_program(mut self, program: impl Into<String>) -> Self {
        let script = program.into();
        self.base_program = if script.trim().is_empty() {
            None
        } else {
            Some(script)
        };
        self
    }

    /// Overrides the weight interval assigned to SpiralK hints.
    pub fn with_weight_bounds(mut self, min_weight: f32, max_weight: f32) -> Self {
        let (lo, hi) = if min_weight <= max_weight {
            (min_weight, max_weight)
        } else {
            (max_weight, min_weight)
        };
        self.min_weight = lo.clamp(0.0, 1.0);
        self.max_weight = hi.clamp(0.0, 1.0);
        self
    }

    /// Returns the hints that have been accumulated so far.
    pub fn hints(&self) -> &[MaxwellSpiralKHint] {
        &self.hints
    }

    /// Returns `true` when no pulses have been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.hints.is_empty()
    }

    /// Registers a Maxwell pulse for the provided channel and returns the generated hint.
    pub fn push_pulse(
        &mut self,
        channel: impl AsRef<str>,
        pulse: &MaxwellZPulse,
    ) -> MaxwellSpiralKHint {
        let channel = sanitize_channel(channel.as_ref());
        let weight = self.weight_from_pulse(pulse);
        let hint = MaxwellSpiralKHint {
            channel,
            blocks: pulse.blocks,
            z_score: pulse.z_score,
            z_bias: pulse.z_bias,
            weight,
        };
        self.hints.push(hint.clone());
        hint
    }

    /// Builds the SpiralK script snippet that encodes every recorded hint.
    pub fn script(&self) -> Option<String> {
        let mut lines = Vec::new();
        if let Some(base) = &self.base_program {
            let trimmed = base.trim();
            if !trimmed.is_empty() {
                let mut line = trimmed.to_string();
                if !trimmed.ends_with(';') {
                    line.push(';');
                }
                lines.push(line);
            }
        }
        for hint in &self.hints {
            lines.push(hint.script_line());
        }
        if lines.is_empty() {
            None
        } else {
            Some(lines.join("\n"))
        }
    }

    fn weight_from_pulse(&self, pulse: &MaxwellZPulse) -> f32 {
        let magnitude = pulse.magnitude() as f32;
        let spread = (magnitude / (1.0 + magnitude)).clamp(0.0, 1.0);
        if self.max_weight <= self.min_weight {
            self.min_weight
        } else {
            self.min_weight + (self.max_weight - self.min_weight) * spread
        }
    }
}

fn sanitize_channel(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
        } else if matches!(ch, '-' | '_' | ' ' | ':' | '/') {
            out.push('_');
        }
    }
    if out.is_empty() {
        "maxwell".to_string()
    } else {
        out
    }
}

/// Computes the block count required to hit the target Z threshold under the
/// Gaussian approximation described in the memo.
///
/// Returns `None` when the provided parameters make detection impossible (e.g.
/// when either `κ` or `λ` is zero).
pub fn required_blocks(target_z: f64, sigma: f64, kappa: f64, lambda: f64) -> Option<f64> {
    let numerator = target_z.abs() * sigma.abs();
    let denominator = kappa.abs() * lambda.abs();
    if denominator <= f64::EPSILON {
        None
    } else {
        Some((numerator / denominator).powi(2))
    }
}

/// Estimates the polarisation sweep slope by regressing the expected amplitude
/// against the polarisation angle. The slope acts as a diagnostic — when the
/// returned value deviates significantly from `|cos θ|`, the experiment may be
/// dominated by leakage instead of the coded Maxwell channel.
pub fn polarisation_slope(fingerprint: &MaxwellFingerprint, samples: usize) -> f64 {
    let samples = samples.max(2);
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for k in 0..samples {
        let theta = (k as f64 / (samples - 1) as f64) * PI;
        let alignment = theta.cos().abs();
        numerator += alignment.powi(2) * fingerprint.lambda();
        denominator += alignment.powi(2);
    }
    if denominator <= f64::EPSILON {
        0.0
    } else {
        numerator / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};

    #[test]
    fn maxwell_fingerprint_computes_lambda() {
        let fingerprint = MaxwellFingerprint::new(0.8, 0.5, 1.2, 6.0, 3.0, 0.25 * PI, 2.0);
        let shielding = fingerprint.shielding_factor();
        assert_abs_diff_eq!(shielding, 10f64.powf(-6.0 / 20.0));
        let lambda = fingerprint.lambda();
        let expected = 0.8 * 0.5 * 1.2 * shielding * 3.0 * (0.25 * PI).cos().abs() / 2.0;
        assert_abs_diff_eq!(lambda, expected);
    }

    #[test]
    fn meaning_gate_limits_rho() {
        let gate = MeaningGate::new(1.0, 0.4);
        assert_abs_diff_eq!(gate.envelope(0.5), 1.0 + 0.4 * 0.5);
        assert_abs_diff_eq!(gate.envelope(10.0), 1.0 + 0.4);
        assert_abs_diff_eq!(gate.envelope(-10.0), 1.0 - 0.4);
    }

    #[test]
    fn sequential_z_tracks_statistics() {
        let mut tracker = SequentialZ::new();
        assert!(tracker.z_stat().is_none());
        let samples = [0.8, 1.2, 1.0, 1.4, 0.6];
        for sample in samples {
            tracker.push(sample);
        }
        assert_eq!(tracker.len(), 5);
        assert!(tracker.variance().unwrap() > 0.0);
        let z = tracker.z_stat().unwrap();
        assert!(z.is_finite());
    }

    #[test]
    fn required_blocks_returns_none_on_zero_gain() {
        assert!(required_blocks(3.0, 1.0, 0.0, 1.0).is_none());
        assert!(required_blocks(3.0, 1.0, 1.0, 0.0).is_none());
        let blocks = required_blocks(3.0, 1.0, 1.5, 0.8).unwrap();
        assert!(blocks > 0.0);
    }

    #[test]
    fn polarisation_slope_behaves_reasonably() {
        let fingerprint = MaxwellFingerprint::new(1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let slope = polarisation_slope(&fingerprint, 8);
        assert!(slope > 0.0);
        assert_relative_eq!(slope, fingerprint.lambda(), epsilon = 1e-6);
    }

    #[test]
    fn maxwell_projector_emits_pulse() {
        let mut tracker = SequentialZ::new();
        for sample in [1.1, 0.9, 1.3, 0.8, 1.4, 1.2] {
            tracker.push(sample);
        }

        let projector = MaxwellZProjector::new(24, 0.5)
            .with_bias_gain(2.0)
            .with_min_blocks(3)
            .with_min_z(0.5);

        let pulse = projector
            .project(&tracker)
            .expect("pulse should be emitted");
        assert!(pulse.magnitude() >= 0.5);
        assert!(pulse.z_bias.abs() > 0.0);

        let feedback = pulse.into_softlogic_feedback(12.0, 4.0);
        assert_eq!(feedback.psi_total, 12.0);
        assert_eq!(feedback.weighted_loss, 4.0);
        assert!(feedback.z_signal.abs() > 0.0);
    }

    #[test]
    fn maxwell_projector_coop_agent_adjusts_bias_gain() {
        let mut tracker = SequentialZ::new();
        for sample in [1.2, 0.95, 1.1, 0.88, 1.3, 1.05] {
            tracker.push(sample);
        }

        let mut projector = MaxwellZProjector::new(16, 0.4)
            .with_bias_gain(1.5)
            .with_min_blocks(3)
            .with_min_z(0.4);

        assert!(projector.project(&tracker).is_some());
        let proposal = CoopAgent::propose(&mut projector);
        assert!(proposal.weight > 0.0);

        let before_gain = projector.bias_gain;
        CoopAgent::observe(&mut projector, 0.25, 0.75);
        assert!(projector.bias_gain >= before_gain);
    }

    #[test]
    fn spiralk_bridge_emits_script() {
        let pulse = MaxwellZPulse {
            blocks: 12,
            mean: 0.42,
            standard_error: 0.08,
            z_score: 5.25,
            band_energy: (0.8, 0.6, 0.2),
            z_bias: 0.73,
        };

        let mut bridge = MaxwellSpiralKBridge::new().with_base_program("z: 1;");
        let hint = bridge.push_pulse("Ch-α/β", &pulse);

        assert_eq!(hint.channel, "Ch__");
        assert!(hint.weight >= 0.55 && hint.weight <= 0.95);

        let script = bridge.script().expect("script should exist");
        assert!(script.contains("z: 1;"));
        assert!(script.contains("maxwell.bias"));
        assert!(script.contains("blocks >= 12"));
    }

    #[cfg(feature = "psi")]
    #[test]
    fn psi_bridge_streams_into_hub() {
        use crate::telemetry::hub;

        let pulse = MaxwellZPulse {
            blocks: 9,
            mean: 0.18,
            standard_error: 0.05,
            z_score: 4.2,
            band_energy: (0.6, 0.4, 0.2),
            z_bias: 0.48,
        };

        let bridge = MaxwellPsiTelemetryBridge::new()
            .with_psi_gain(1.7)
            .with_loss_gain(0.3)
            .with_band_threshold(1.0);

        let _guard = hub::psi_telemetry_guard();
        hub::clear_last_psi();
        hub::clear_last_psi_events();
        hub::clear_softlogic_z();

        let (feedback, reading, events) = bridge.publish_with_reading(&pulse, 42).into_parts();

        let stored_feedback = hub::get_softlogic_z().expect("softlogic feedback");
        assert_eq!(stored_feedback.z_signal, feedback.z_signal);
        assert!((stored_feedback.psi_total - feedback.psi_total).abs() <= f32::EPSILON);

        assert_eq!(reading.step, 42);
        assert!(reading.total > 0.0);
        assert_eq!(reading.breakdown.len(), 1);

        if (pulse.band_energy.0 + pulse.band_energy.1 + pulse.band_energy.2) >= 1.0 {
            assert_eq!(events.len(), 1);
        } else {
            assert!(events.is_empty());
        }

        let stored_reading = hub::get_last_psi().expect("psi reading");
        assert_eq!(stored_reading.step, reading.step);
        assert_eq!(stored_reading.total, reading.total);

        let stored_events = hub::get_last_psi_events();
        assert_eq!(stored_events.len(), events.len());

        let legacy_feedback = bridge.publish(&pulse, 42);
        assert_eq!(legacy_feedback.psi_total, feedback.psi_total);
        assert_eq!(legacy_feedback.weighted_loss, feedback.weighted_loss);
    }

    #[test]
    fn sanitize_channel_falls_back_on_default() {
        assert_eq!(sanitize_channel(""), "maxwell");
        assert_eq!(sanitize_channel("A:B C"), "A_B_C");
    }
}
