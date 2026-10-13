// SPDX-License-Identifier: AGPL-3.0-or-later
//! One-time, softmax-gauge-invariant scale matching of causal score biases.
//! This is an initialization transform, not differentiable normalization.
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum CausalBiasScaleError {
    #[error("bias scores need nonempty, portable [batch,heads,time,time]")]
    Shape,
    #[error("score length differs from its shape")]
    Length,
    #[error("bias scores or resulting moments are non-finite")]
    NonFinite,
    #[error("head/time shapes or accumulated causal-pair counts differ")]
    Coverage,
    #[error("accumulated causal-pair count overflows")]
    Overflow,
    #[error("each head needs nonzero centered signal in both biases")]
    NoSignal,
    #[error("one finite raw softplus gain per head is required")]
    Gain,
    #[error("relative calibration tolerance must be finite in [0,1)")]
    Tolerance,
    #[error("the requested scale cannot be represented by an f32 raw gain within tolerance")]
    Unrepresentable,
}

fn add(sum: &mut f64, correction: &mut f64, value: f64) {
    let adjusted = value - *correction;
    let total = *sum + adjusted;
    *correction = (total - *sum) - adjusted;
    *sum = total;
}

/// Sum of squared deviations from each causal row's mean, one value per head.
/// The denominator counts all k<=q pairs, including the zero-variance first
/// row. Future positions are finite-checked but never enter the moments.
#[derive(Clone, Debug)]
pub struct CausalBiasMoments {
    heads: usize,
    steps: usize,
    valid_pairs: u64,
    squared: Vec<f64>,
}

impl CausalBiasMoments {
    pub fn from_scores(shape: [usize; 4], scores: &[f32]) -> Result<Self, CausalBiasScaleError> {
        let [batch, heads, steps, keys] = shape;
        let length = shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .filter(|&n| n > 0 && u32::try_from(n).is_ok())
            .ok_or(CausalBiasScaleError::Shape)?;
        if steps != keys {
            return Err(CausalBiasScaleError::Shape);
        }
        if scores.len() != length {
            return Err(CausalBiasScaleError::Length);
        }
        if scores.iter().any(|v| !v.is_finite()) {
            return Err(CausalBiasScaleError::NonFinite);
        }
        let valid_pairs = (batch as u64) * (steps as u64) * (steps as u64 + 1) / 2;
        let mut squared = vec![0.; heads];
        for (h, total) in squared.iter_mut().enumerate() {
            let mut correction = 0.;
            for b in 0..batch {
                for q in 0..steps {
                    let start = ((b * heads + h) * steps + q) * steps;
                    let row = &scores[start..start + q + 1];
                    // Subtract an anchor before taking the mean to preserve
                    // small representable spreads around a large row offset.
                    let origin = f64::from(row[0]);
                    let (mut sum, mut c) = (0., 0.);
                    for &x in row {
                        add(&mut sum, &mut c, f64::from(x) - origin);
                    }
                    let mean_delta = sum / row.len() as f64;
                    for &x in row {
                        let delta = (f64::from(x) - origin) - mean_delta;
                        add(total, &mut correction, delta * delta);
                    }
                }
            }
        }
        if squared.iter().any(|v| !v.is_finite()) {
            return Err(CausalBiasScaleError::NonFinite);
        }
        Ok(Self {
            heads,
            steps,
            valid_pairs,
            squared,
        })
    }

    /// Combine disjoint observations without equally weighting unequal batches.
    /// Same counts do not prove same windows: the caller owns data identity and
    /// must select calibration inputs from training data, not held-out labels.
    pub fn merge(&mut self, other: &Self) -> Result<(), CausalBiasScaleError> {
        if (self.heads, self.steps) != (other.heads, other.steps) {
            return Err(CausalBiasScaleError::Coverage);
        }
        let count = self
            .valid_pairs
            .checked_add(other.valid_pairs)
            .ok_or(CausalBiasScaleError::Overflow)?;
        let squared: Vec<_> = self
            .squared
            .iter()
            .zip(&other.squared)
            .map(|(a, b)| a + b)
            .collect();
        if squared.iter().any(|v| !v.is_finite()) {
            return Err(CausalBiasScaleError::NonFinite);
        }
        self.valid_pairs = count;
        self.squared = squared;
        Ok(())
    }
    pub fn heads(&self) -> usize {
        self.heads
    }
    pub fn steps(&self) -> usize {
        self.steps
    }
    pub fn valid_pairs_per_head(&self) -> u64 {
        self.valid_pairs
    }
    pub fn rms(&self) -> Vec<f64> {
        self.squared
            .iter()
            .map(|v| (v / self.valid_pairs as f64).sqrt())
            .collect()
    }
}

fn log_softplus(raw: f64) -> f64 {
    if raw < -40. {
        // The relative correction is below f64 precision here; no exp underflow.
        raw
    } else {
        (raw.max(0.) + (-raw.abs()).exp().ln_1p()).ln()
    }
}

fn inverse_log_softplus(log_gain: f64) -> f64 {
    if log_gain < -40. {
        log_gain
    } else {
        let gain = log_gain.exp();
        gain + (-(-gain).exp_m1()).ln()
    }
}

/// Match candidate centered RMS to reference by changing only its softplus
/// raw gain. Candidate moments must include the supplied old gain already.
/// The error limit covers host inversion/f32 quantization, not device rounding;
/// measure the resulting device biases again before claiming a matched run.
#[derive(Clone, Debug)]
pub struct CausalBiasScaleMatch {
    raw_gains: Vec<f32>,
    requested_scales: Vec<f64>,
    realized_scales: Vec<f64>,
    relative_errors: Vec<f64>,
}

impl CausalBiasScaleMatch {
    pub fn new(
        reference: &CausalBiasMoments,
        candidate: &CausalBiasMoments,
        raw_gains: &[f32],
        max_relative_error: f64,
    ) -> Result<Self, CausalBiasScaleError> {
        if !max_relative_error.is_finite() || !(0. ..1.).contains(&max_relative_error) {
            return Err(CausalBiasScaleError::Tolerance);
        }
        if (reference.heads, reference.steps, reference.valid_pairs)
            != (candidate.heads, candidate.steps, candidate.valid_pairs)
        {
            return Err(CausalBiasScaleError::Coverage);
        }
        if raw_gains.len() != reference.heads || raw_gains.iter().any(|v| !v.is_finite()) {
            return Err(CausalBiasScaleError::Gain);
        }
        let (target, observed) = (reference.rms(), candidate.rms());
        let mut result = Self {
            raw_gains: Vec::new(),
            requested_scales: Vec::new(),
            realized_scales: Vec::new(),
            relative_errors: Vec::new(),
        };
        for ((target, observed), &raw) in target.iter().zip(&observed).zip(raw_gains) {
            if *target <= 0. || *observed <= 0. {
                return Err(CausalBiasScaleError::NoSignal);
            }
            let scale = target / observed;
            let desired_log_scale = scale.ln();
            let old_log_gain = log_softplus(f64::from(raw));
            let log_gain = old_log_gain + desired_log_scale;
            if log_gain > f64::from(f32::MAX).ln() {
                return Err(CausalBiasScaleError::Unrepresentable);
            }
            let calibrated = if scale == 1. {
                raw
            } else {
                inverse_log_softplus(log_gain) as f32
            };
            let realized_log_scale = log_softplus(f64::from(calibrated)) - old_log_gain;
            let error = (realized_log_scale - desired_log_scale).exp_m1().abs();
            if !calibrated.is_finite() || !error.is_finite() || error > max_relative_error {
                return Err(CausalBiasScaleError::Unrepresentable);
            }
            result.raw_gains.push(calibrated);
            result.requested_scales.push(scale);
            result.realized_scales.push(realized_log_scale.exp());
            result.relative_errors.push(error);
        }
        Ok(result)
    }
    pub fn raw_gains(&self) -> &[f32] {
        &self.raw_gains
    }
    pub fn requested_scales(&self) -> &[f64] {
        &self.requested_scales
    }
    pub fn realized_scales(&self) -> &[f64] {
        &self.realized_scales
    }
    pub fn relative_errors(&self) -> &[f64] {
        &self.relative_errors
    }
}

#[cfg(test)]
mod tests;
