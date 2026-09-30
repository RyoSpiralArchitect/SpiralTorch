//! Plain SGD candidate arithmetic shared by resident training paths.
use thiserror::Error;

pub const SGD_INVALID_GRADIENT: u32 = 1 << 12;
pub const SGD_INVALID_CHANGE: u32 = 1 << 13;
pub const SGD_INVALID_CANDIDATE: u32 = 1 << 14;

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum SgdError {
    #[error("learning rate must be finite and nonnegative")]
    LearningRate,
    #[error("non-finite SGD candidate arithmetic, mask {flags:#x}")]
    NonFinite { flags: u32 },
}

/// Validated plain-SGD rate. Loss reduction, clipping, and momentum are applied
/// before this rule; acceptance and all-parameter commit belong to the caller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SgdStep {
    rate: f32,
}

impl SgdStep {
    /// Zero is valid for derivative probes, but never masks an invalid gradient.
    pub fn new(rate: f32) -> Result<Self, SgdError> {
        if !rate.is_finite() || rate < 0.0 {
            return Err(SgdError::LearningRate);
        }
        Ok(Self { rate })
    }

    pub fn rate(self) -> f32 {
        self.rate
    }

    /// Validate the multiply and subtract separately, including overflow that
    /// fused arithmetic could cancel. A zero-rate transaction must still retain
    /// its original parameters rather than committing this diagnostic candidate.
    pub fn candidate(self, parameter: f32, gradient: f32) -> Result<f32, SgdError> {
        let change = self.rate * gradient;
        let candidate = parameter - change;
        let mut flags = 0;
        if !gradient.is_finite() {
            flags |= SGD_INVALID_GRADIENT;
        }
        if !change.is_finite() {
            flags |= SGD_INVALID_CHANGE;
        }
        if !candidate.is_finite() {
            flags |= SGD_INVALID_CANDIDATE;
        }
        if flags != 0 {
            Err(SgdError::NonFinite { flags })
        } else {
            Ok(candidate)
        }
    }
}

/// Shared arithmetic for a host-validated rate. Callers retain their gradient
/// diagnostic mask (plain gradient or EMA history) and whole-update decision.
pub fn sgd_candidate_wgsl() -> String {
    r#"
struct SgdCandidate {
    value: f32,
    flags: u32,
};

fn sgd_candidate(parameter: f32, gradient: f32, rate: f32, gradient_flag: u32) -> SgdCandidate {
    let change = rate * gradient;
    let candidate = parameter - change;
    var flags = 0u;
    if ((bitcast<u32>(gradient) & 0x7f800000u) == 0x7f800000u) {
        flags = flags | gradient_flag;
    }
    if ((bitcast<u32>(change) & 0x7f800000u) == 0x7f800000u) {
        flags = flags | SGD_CHANGE_MASK;
    }
    if ((bitcast<u32>(candidate) & 0x7f800000u) == 0x7f800000u) {
        flags = flags | SGD_CANDIDATE_MASK;
    }
    return SgdCandidate(candidate, flags);
}
"#
    .replace("SGD_CHANGE_MASK", &format!("{SGD_INVALID_CHANGE}u"))
    .replace("SGD_CANDIDATE_MASK", &format!("{SGD_INVALID_CANDIDATE}u"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_sgd_has_no_implicit_reduction_or_state() {
        let step = SgdStep::new(0.25).unwrap();
        assert_eq!(step.rate(), 0.25);
        assert_eq!(step.candidate(4.0, 8.0).unwrap(), 2.0);
        assert_eq!(step.candidate(4.0, -8.0).unwrap(), 6.0);
        assert_eq!(step.candidate(4.0, 0.0).unwrap(), 4.0);
        for rate in [-1.0, f32::NEG_INFINITY, f32::INFINITY, f32::NAN] {
            assert_eq!(SgdStep::new(rate), Err(SgdError::LearningRate));
        }
    }

    #[test]
    fn zero_rate_and_cancellation_cannot_hide_non_finite_arithmetic() {
        for rate in [0.0, -0.0, 1.0] {
            let step = SgdStep::new(rate).unwrap();
            for gradient in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let Err(SgdError::NonFinite { flags }) = step.candidate(1.0, gradient) else {
                    panic!("invalid gradient was accepted");
                };
                assert_ne!(flags & SGD_INVALID_GRADIENT, 0);
            }
            assert!(step.candidate(f32::INFINITY, 0.0).is_err());
        }
        assert_eq!(
            SgdStep::new(2.0).unwrap().candidate(f32::MAX, f32::MAX),
            Err(SgdError::NonFinite {
                flags: SGD_INVALID_CHANGE | SGD_INVALID_CANDIDATE,
            })
        );
        assert_eq!(
            SgdStep::new(1.0).unwrap().candidate(f32::MAX, -f32::MAX),
            Err(SgdError::NonFinite {
                flags: SGD_INVALID_CANDIDATE,
            })
        );
        assert_eq!(
            SgdStep::new(0.0).unwrap().candidate(f32::MAX, f32::MAX),
            Ok(f32::MAX)
        );
    }
}
