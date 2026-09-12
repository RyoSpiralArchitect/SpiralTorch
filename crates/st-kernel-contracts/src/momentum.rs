//! Topos' exponential-moving-average gradient state, independent of execution.
use thiserror::Error;

pub const EMA_MOMENTUM_RULE: &str = "m_t=damping*m_(t-1)+(1-damping)*g_clipped";
pub const MAX_MOMENTUM_DAMPING: f32 = 0.85;

#[derive(Debug, Error, Clone, Copy, PartialEq)]
pub enum MomentumError {
    #[error("momentum damping must be finite and in [0, 0.85]")]
    Damping,
    #[error("momentum transition requires finite values, got {0}")]
    NonFinite(f32),
}

/// This is a zero-initialized EMA, not heavy-ball or Nesterov momentum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmaMomentum {
    damping: f32,
}

impl EmaMomentum {
    pub fn new(damping: f32) -> Result<Self, MomentumError> {
        if !damping.is_finite() || !(0.0..=MAX_MOMENTUM_DAMPING).contains(&damping) {
            return Err(MomentumError::Damping);
        }
        Ok(Self { damping })
    }

    pub fn damping(self) -> f32 {
        self.damping
    }

    pub fn transition(self, gradient: f32, previous: f32) -> Result<f32, MomentumError> {
        for value in [gradient, previous] {
            if !value.is_finite() {
                return Err(MomentumError::NonFinite(value));
            }
        }
        let retained = self.damping * previous;
        let incoming = (1. - self.damping) * gradient;
        let next = retained + incoming;
        if !next.is_finite() {
            return Err(MomentumError::NonFinite(next));
        }
        Ok(next)
    }
}

/// The backend checks the inputs/result and commits the returned candidate only
/// after its whole-update decision. Kept here with the CPU semantic oracle.
pub const EMA_MOMENTUM_WGSL: &str = r#"
fn ema_momentum(gradient: f32, previous: f32, damping: f32) -> f32 {
    let retained = damping * previous;
    let incoming = (1.0 - damping) * gradient;
    return retained + incoming;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ema_is_zero_initialized_and_checks_masked_inputs() {
        for d in [-0.1, 0.86, f32::INFINITY, f32::NAN] {
            assert!(EmaMomentum::new(d).is_err());
        }
        let ema = EmaMomentum::new(0.5).unwrap();
        assert_eq!(ema.transition(2., 0.).unwrap(), 1.);
        assert_eq!(ema.transition(0., 1.).unwrap(), 0.5);
        assert_eq!(ema.transition(-1., 0.5).unwrap(), -0.25);
        for d in [0., 0.5, 0.85] {
            let ema = EmaMomentum::new(d).unwrap();
            assert!(ema.transition(f32::NAN, 0.).is_err());
            assert!(ema.transition(1., f32::INFINITY).is_err());
            assert!(ema.transition(f32::MAX, f32::MAX).unwrap().is_finite());
        }
    }
}
