// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use thiserror::Error;

/// Aggregate band energy used by the free-energy scoring proxy.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BandEnergy {
    pub above: f32,
    pub here: f32,
    pub beneath: f32,
}

impl BandEnergy {
    /// Returns the L1 magnitude across all bands.
    pub fn l1(&self) -> f32 {
        self.above.abs() + self.here.abs() + self.beneath.abs()
    }

    /// Normalises the band energies so they sum to one. When every component is
    /// zero the method returns an even split.
    pub fn norm(self) -> Self {
        self.try_norm().unwrap_or_else(|_| Self::uniform())
    }

    /// Strictly normalises finite, non-negative band energies.
    pub fn try_norm(self) -> Result<Self, FreeEnergyError> {
        for (field, value) in [
            ("band.above", self.above),
            ("band.here", self.here),
            ("band.beneath", self.beneath),
        ] {
            validate_non_negative(field, value)?;
        }
        let sum = f64::from(self.above) + f64::from(self.here) + f64::from(self.beneath);
        if sum <= f64::from(f32::EPSILON) {
            return Ok(Self::uniform());
        }
        Ok(Self {
            above: (f64::from(self.above) / sum) as f32,
            here: (f64::from(self.here) / sum) as f32,
            beneath: (f64::from(self.beneath) / sum) as f32,
        })
    }

    fn uniform() -> Self {
        Self {
            above: 1.0 / 3.0,
            here: 1.0 / 3.0,
            beneath: 1.0 / 3.0,
        }
    }
}

/// Context passed into [`score_with_free_energy`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FeCtx {
    pub loss_before: f32,
    pub loss_after: f32,
    pub step_ms: f32,
    pub mem_mb: f32,
    pub retry: f32,
    pub band: BandEnergy,
    pub entropy: f32,
}

/// Computes a free-energy inspired utility score used to rank plans.
pub fn score_with_free_energy(ctx: &FeCtx, beta: f32) -> f32 {
    try_score_with_free_energy(ctx, beta).unwrap_or(f32::NEG_INFINITY)
}

/// Strictly computes a finite free-energy utility score.
pub fn try_score_with_free_energy(ctx: &FeCtx, beta: f32) -> Result<f32, FreeEnergyError> {
    for (field, value) in [
        ("loss_before", ctx.loss_before),
        ("loss_after", ctx.loss_after),
        ("step_ms", ctx.step_ms),
        ("mem_mb", ctx.mem_mb),
        ("retry", ctx.retry),
        ("entropy", ctx.entropy),
        ("beta", beta),
    ] {
        validate_finite(field, value)?;
    }

    let beta = f64::from(beta.max(0.0));
    let delta_l = f64::from(ctx.loss_after) - f64::from(ctx.loss_before);
    let entropy = f64::from(ctx.entropy.max(0.0));
    let free_energy = delta_l + beta * entropy;

    let norm = ctx.band.try_norm()?;
    let novelty = f64::from(norm.above - norm.beneath);
    let stability = f64::from((1.0 - norm.here).clamp(0.0, 1.0));

    let step_penalty = 0.0025 * f64::from(ctx.step_ms.max(0.0));
    let mem_penalty = 0.001 * f64::from(ctx.mem_mb.max(0.0));
    let retry_penalty = 0.5 * f64::from(ctx.retry.max(0.0));

    let score =
        -free_energy - step_penalty - mem_penalty - retry_penalty + 0.2 * novelty - 0.1 * stability;
    if !score.is_finite() || score.abs() > f64::from(f32::MAX) {
        return Err(FreeEnergyError::ArithmeticOverflow);
    }
    Ok(score as f32)
}

fn validate_finite(field: &'static str, value: f32) -> Result<(), FreeEnergyError> {
    if !value.is_finite() {
        return Err(FreeEnergyError::NonFinite { field, value });
    }
    Ok(())
}

fn validate_non_negative(field: &'static str, value: f32) -> Result<(), FreeEnergyError> {
    validate_finite(field, value)?;
    if value < 0.0 {
        return Err(FreeEnergyError::Negative { field, value });
    }
    Ok(())
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum FreeEnergyError {
    #[error("free-energy field `{field}` must be finite, got {value}")]
    NonFinite { field: &'static str, value: f32 },
    #[error("free-energy field `{field}` must be non-negative, got {value}")]
    Negative { field: &'static str, value: f32 },
    #[error("free-energy score overflowed")]
    ArithmeticOverflow,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn higher_entropy_is_penalised() {
        let ctx = FeCtx {
            loss_before: 0.5,
            loss_after: 0.25,
            entropy: 1.2,
            step_ms: 10.0,
            mem_mb: 128.0,
            retry: 0.0,
            band: BandEnergy {
                above: 0.6,
                here: 0.3,
                beneath: 0.1,
            },
        };
        let tight = score_with_free_energy(&ctx, 0.0);
        let loose = score_with_free_energy(&ctx, 0.8);
        assert!(loose < tight);
    }

    #[test]
    fn novelty_bias_pushes_above_band() {
        let base = FeCtx {
            loss_before: 0.4,
            loss_after: 0.3,
            entropy: 0.4,
            step_ms: 12.0,
            mem_mb: 64.0,
            retry: 0.0,
            band: BandEnergy {
                above: 0.2,
                here: 0.6,
                beneath: 0.2,
            },
        };
        let novel = FeCtx {
            band: BandEnergy {
                above: 0.6,
                here: 0.3,
                beneath: 0.1,
            },
            ..base
        };
        assert!(score_with_free_energy(&novel, 0.3) > score_with_free_energy(&base, 0.3));
    }

    #[test]
    fn strict_score_rejects_non_finite_inputs() {
        let ctx = FeCtx {
            loss_after: f32::NAN,
            ..FeCtx::default()
        };
        let error =
            try_score_with_free_energy(&ctx, 0.3).expect_err("non-finite loss must be rejected");
        assert!(matches!(error, FreeEnergyError::NonFinite { .. }));
        assert_eq!(score_with_free_energy(&ctx, 0.3), f32::NEG_INFINITY);
    }

    #[test]
    fn strict_norm_rejects_negative_band_energy() {
        let band = BandEnergy {
            above: -0.1,
            here: 0.5,
            beneath: 0.6,
        };
        assert!(matches!(
            band.try_norm(),
            Err(FreeEnergyError::Negative { .. })
        ));
        assert_eq!(band.norm(), BandEnergy::uniform());
    }
}
