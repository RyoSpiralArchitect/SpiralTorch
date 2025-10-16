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
        numerator += alignment * fingerprint.lambda();
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
}
