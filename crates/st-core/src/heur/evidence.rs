// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Guarded statistical evidence primitives shared by Rust runtime policies.

use serde::Serialize;
use thiserror::Error;

pub const WILSON_INTERVAL_CONTRACT: &str = "spiraltorch.evidence.wilson_interval";
pub const WILSON_INTERVAL_CONTRACT_VERSION: u32 = 1;
pub const WILSON_INTERVAL_FORMULA: &str =
    "center=(p+z^2/(2n))/(1+z^2/n);radius=z*sqrt((p*(1-p)+z^2/(4n))/n)/(1+z^2/n)";
/// Largest integer represented exactly by JSON, JavaScript, and `f64`.
pub const WILSON_MAX_EXACT_TRIALS: u64 = 9_007_199_254_740_991;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum WilsonError {
    #[error("Wilson evidence requires at least one trial")]
    NoTrials,
    #[error("Wilson successes {successes} exceed trials {trials}")]
    SuccessesExceedTrials { successes: u64, trials: u64 },
    #[error("Wilson trials {trials} exceed the exact cross-client limit {maximum}")]
    TrialLimit { trials: u64, maximum: u64 },
    #[error("Wilson confidence z must be finite and positive, got {z}")]
    InvalidConfidence { z: f64 },
    #[error("Wilson derived field '{field}' is not finite")]
    DerivedNonFinite { field: &'static str },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct WilsonInterval {
    pub contract: &'static str,
    pub contract_version: u32,
    pub formula: &'static str,
    pub successes: u64,
    pub trials: u64,
    pub confidence_z: f64,
    pub estimate: f64,
    pub lower: f64,
    pub upper: f64,
}

/// Evaluate a two-sided Wilson score interval for Bernoulli evidence.
pub fn try_wilson_interval(
    successes: u64,
    trials: u64,
    confidence_z: f64,
) -> Result<WilsonInterval, WilsonError> {
    if trials == 0 {
        return Err(WilsonError::NoTrials);
    }
    if successes > trials {
        return Err(WilsonError::SuccessesExceedTrials { successes, trials });
    }
    if trials > WILSON_MAX_EXACT_TRIALS {
        return Err(WilsonError::TrialLimit {
            trials,
            maximum: WILSON_MAX_EXACT_TRIALS,
        });
    }
    if !confidence_z.is_finite() || confidence_z <= 0.0 {
        return Err(WilsonError::InvalidConfidence { z: confidence_z });
    }

    let n = trials as f64;
    let estimate = successes as f64 / n;
    let z_squared = finite("z_squared", confidence_z * confidence_z)?;
    let denominator = finite("denominator", 1.0 + z_squared / n)?;
    let center = finite("center", (estimate + z_squared / (2.0 * n)) / denominator)?;
    let variance = finite(
        "variance",
        (estimate * (1.0 - estimate) + z_squared / (4.0 * n)) / n,
    )?;
    let radius = finite(
        "radius",
        confidence_z * variance.max(0.0).sqrt() / denominator,
    )?;
    let lower = finite("lower", (center - radius).clamp(0.0, 1.0))?;
    let upper = finite("upper", (center + radius).clamp(0.0, 1.0))?;

    Ok(WilsonInterval {
        contract: WILSON_INTERVAL_CONTRACT,
        contract_version: WILSON_INTERVAL_CONTRACT_VERSION,
        formula: WILSON_INTERVAL_FORMULA,
        successes,
        trials,
        confidence_z,
        estimate,
        lower,
        upper,
    })
}

pub fn try_wilson_lower(
    successes: u64,
    trials: u64,
    confidence_z: f64,
) -> Result<f64, WilsonError> {
    try_wilson_interval(successes, trials, confidence_z).map(|interval| interval.lower)
}

fn finite(field: &'static str, value: f64) -> Result<f64, WilsonError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(WilsonError::DerivedNonFinite { field })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_success_is_not_high_confidence_but_four_are_above_even_odds() {
        let one = try_wilson_interval(1, 1, 1.96).expect("one trial");
        let four = try_wilson_interval(4, 4, 1.96).expect("four trials");
        assert!(one.lower < 0.5);
        assert!(four.lower > 0.5);
        assert_eq!(one.contract, WILSON_INTERVAL_CONTRACT);
        assert_eq!(one.contract_version, WILSON_INTERVAL_CONTRACT_VERSION);
    }

    #[test]
    fn complementary_evidence_has_complementary_bounds() {
        let left = try_wilson_interval(7, 10, 1.96).expect("left interval");
        let right = try_wilson_interval(3, 10, 1.96).expect("right interval");
        assert!((left.lower - (1.0 - right.upper)).abs() < 1.0e-12);
        assert!((left.upper - (1.0 - right.lower)).abs() < 1.0e-12);
    }

    #[test]
    fn malformed_or_inexact_evidence_fails_closed() {
        assert_eq!(try_wilson_interval(0, 0, 1.96), Err(WilsonError::NoTrials));
        assert!(matches!(
            try_wilson_interval(2, 1, 1.96),
            Err(WilsonError::SuccessesExceedTrials { .. })
        ));
        assert!(matches!(
            try_wilson_interval(1, 1, f64::NAN),
            Err(WilsonError::InvalidConfidence { .. })
        ));
        assert!(matches!(
            try_wilson_interval(
                WILSON_MAX_EXACT_TRIALS + 1,
                WILSON_MAX_EXACT_TRIALS + 1,
                1.96
            ),
            Err(WilsonError::TrialLimit { .. })
        ));
    }
}
