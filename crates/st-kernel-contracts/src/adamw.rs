//! Decoupled AdamW arithmetic. Acceptance and atomic state commit belong to the owner.
use thiserror::Error;

pub const ADAMW_INVALID_INPUT: u32 = 1 << 15;
pub const ADAMW_INVALID_ARITHMETIC: u32 = 1 << 16;
pub const ADAMW_EXHAUSTED_CLOCK: u32 = 1 << 17;

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum AdamWError {
    #[error("AdamW betas must be zero or normal in [0,1), epsilon positive and normal")]
    Config,
    #[error("AdamW learning rate and weight decay must be zero or positive normal values")]
    Step,
    #[error("invalid AdamW state: {0}")]
    State(&'static str),
    #[error("invalid AdamW candidate, mask {flags:#x}")]
    Candidate { flags: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamWConfig {
    beta1: f32,
    beta2: f32,
    epsilon: f32,
}

impl AdamWConfig {
    pub fn new(beta1: f32, beta2: f32, epsilon: f32) -> Result<Self, AdamWError> {
        if ![beta1, beta2]
            .iter()
            .all(|b| (b.is_normal() || *b == 0.0) && (0.0..1.0).contains(b))
            || !epsilon.is_normal()
            || epsilon <= 0.0
        {
            return Err(AdamWError::Config);
        }
        Ok(Self {
            beta1,
            beta2,
            epsilon,
        })
    }
    pub fn beta1(self) -> f32 {
        self.beta1
    }
    pub fn beta2(self) -> f32 {
        self.beta2
    }
    pub fn epsilon(self) -> f32 {
        self.epsilon
    }
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

/// One tensor's caller-owned rate/decay policy. Zero rate explicitly freezes
/// values, moments AND its accepted-update clock, unlike merely setting lr=0
/// on a PyTorch parameter with a non-None gradient. Derivatives still validate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamWStep {
    rate: f32,
    weight_decay: f32,
}

impl AdamWStep {
    pub fn new(rate: f32, weight_decay: f32) -> Result<Self, AdamWError> {
        if !(rate.is_normal() || rate == 0.0)
            || rate < 0.0
            || !(weight_decay.is_normal() || weight_decay == 0.0)
            || weight_decay < 0.0
        {
            return Err(AdamWError::Step);
        }
        Ok(Self { rate, weight_decay })
    }
    pub fn rate(self) -> f32 {
        self.rate
    }
    pub fn weight_decay(self) -> f32 {
        self.weight_decay
    }

    pub fn candidate(
        self,
        config: AdamWConfig,
        value: f32,
        gradient: f32,
        first: f32,
        second: f32,
        steps: u32,
    ) -> Result<AdamWCandidate, AdamWError> {
        if ![value, gradient, first, second]
            .iter()
            .all(|x| x.is_finite())
            || second < 0.0
        {
            return Err(AdamWError::Candidate {
                flags: ADAMW_INVALID_INPUT,
            });
        }
        if self.rate == 0.0 {
            return Ok(AdamWCandidate {
                value,
                first,
                second,
                steps,
            });
        }
        let steps = steps.checked_add(1).ok_or(AdamWError::Candidate {
            flags: ADAMW_EXHAUSTED_CLOCK,
        })?;
        let square = checked_mul(gradient, gradient)?;
        let first = checked_add(
            checked_mul(config.beta1, first)?,
            checked_mul(1.0 - config.beta1, gradient)?,
        )?;
        let second = checked_add(
            checked_mul(config.beta2, second)?,
            checked_mul(1.0 - config.beta2, square)?,
        )?;
        let m_hat = checked_div(first, bias_correction(config.beta1, steps))?;
        let v_hat = checked_div(second, bias_correction(config.beta2, steps))?;
        let denominator = checked_add(v_hat.sqrt(), config.epsilon)?;
        let decay = checked_mul(self.rate, self.weight_decay)?;
        let decayed = checked_mul(value, checked_add(1.0, -decay)?)?;
        let change = checked_mul(self.rate, checked_div(m_hat, denominator)?)?;
        let value = checked_add(decayed, -change)?;
        Ok(AdamWCandidate {
            value,
            first,
            second,
            steps,
        })
    }
}

// WGSL overflow may produce a finite indeterminate value, so checking only the
// result is insufficient. Reserve 15 ulps below MAX for preflight rounding and
// mirror these conservative bounds in WGSL. Extreme finite candidates can reject.
const SAFE_MAX: f32 = f32::from_bits(0x7f7ffff0);
const DIVISOR_MAX: f32 = f32::from_bits(0x7e800000); // WGSL's 2^126 accuracy limit.
fn arithmetic_error() -> AdamWError {
    AdamWError::Candidate {
        flags: ADAMW_INVALID_ARITHMETIC,
    }
}
fn checked_mul(a: f32, b: f32) -> Result<f32, AdamWError> {
    let smaller = a.abs().min(b.abs());
    if smaller > 1.0 && (smaller > DIVISOR_MAX || a.abs().max(b.abs()) >= SAFE_MAX / smaller) {
        return Err(arithmetic_error());
    }
    let out = a * b;
    if out.is_finite() {
        Ok(out)
    } else {
        Err(arithmetic_error())
    }
}
fn checked_add(a: f32, b: f32) -> Result<f32, AdamWError> {
    if a != 0.0 && b != 0.0 && (a > 0.0) == (b > 0.0) && a.abs() >= SAFE_MAX - b.abs() {
        return Err(arithmetic_error());
    }
    let out = a + b;
    if out.is_finite() {
        Ok(out)
    } else {
        Err(arithmetic_error())
    }
}
fn checked_div(a: f32, b: f32) -> Result<f32, AdamWError> {
    if !b.is_normal()
        || b <= 0.0
        || b > DIVISOR_MAX
        || a.abs() >= SAFE_MAX
        || (b < 1.0 && a.abs() >= SAFE_MAX * b)
    {
        return Err(arithmetic_error());
    }
    let out = a / b;
    if out.is_finite() {
        Ok(out)
    } else {
        Err(arithmetic_error())
    }
}
// Compose complements: 1 - (1-a)(1-b) = a + (1-a)b. Computing 1-beta^n
// directly avoids cancellation and repeated squaring of a rounded near-one beta.
// The integer clock stays exact, including beyond float32's consecutive integers.
fn bias_correction(beta: f32, mut n: u32) -> f32 {
    let mut power_complement = 1.0 - beta;
    let mut value = 0.0;
    while n != 0 {
        if n & 1 != 0 {
            value += (1.0 - value) * power_complement;
        }
        n >>= 1;
        if n != 0 {
            power_complement += (1.0 - power_complement) * power_complement;
        }
    }
    value
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamWCandidate {
    pub value: f32,
    pub first: f32,
    pub second: f32,
    pub steps: u32,
}

pub const ADAMW_CANDIDATE_WGSL: &str = include_str!("adamw.wgsl");

/// Portable slots in the same order and shapes as their model's parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct AdamWSlot {
    pub shape: Vec<usize>,
    pub first: Vec<f32>,
    pub second: Vec<f32>,
    pub steps: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdamWState {
    pub config: AdamWConfig,
    pub slots: Vec<AdamWSlot>,
}

impl AdamWState {
    /// Validate before GPU allocation. A fresh optimizer may begin at a later
    /// model revision, but a slot's accepted clock cannot be ahead of that model.
    pub fn validate(&self, shapes: &[Vec<usize>], revision: u64) -> Result<(), AdamWError> {
        if self.slots.len() != shapes.len() || shapes.is_empty() {
            return Err(AdamWError::State("parameter count"));
        }
        for (slot, shape) in self.slots.iter().zip(shapes) {
            let len = shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or(AdamWError::State("shape overflow"))?;
            if len == 0
                || shape.is_empty()
                || slot.shape != *shape
                || slot.first.len() != len
                || slot.second.len() != len
            {
                return Err(AdamWError::State("parameter shape"));
            }
            if u64::from(slot.steps) > revision
                || !slot.first.iter().all(|x| x.is_finite())
                || !slot.second.iter().all(|x| x.is_finite() && *x >= 0.0)
                || (slot.steps == 0
                    && slot
                        .first
                        .iter()
                        .chain(&slot.second)
                        .any(|x| x.to_bits() != 0))
            {
                return Err(AdamWError::State("moments or accepted clock"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bias_correction_tracks_f64_across_u32_clocks() {
        let mut clocks: Vec<u32> = (0..=4096).collect();
        for bit in 0..32 {
            let power = 1u32 << bit;
            clocks.extend([power - 1, power, power + 1]);
        }
        clocks.push(u32::MAX);
        for beta in [
            0.0,
            f32::MIN_POSITIVE,
            0.01,
            0.1,
            0.5,
            0.9,
            0.99,
            0.999,
            0.9999,
            f32::from_bits(0x3f7fffff),
        ] {
            for &steps in &clocks {
                let expected = if steps == 0 {
                    0.0
                } else if beta == 0.0 {
                    1.0
                } else {
                    // Independent f64 oracle avoids subtracting nearly equal numbers.
                    -(f64::from(steps) * f64::from(beta).ln()).exp_m1() as f32
                };
                let actual = bias_correction(beta, steps);
                assert!((0.0..=1.0).contains(&actual));
                assert!(
                    actual.to_bits().abs_diff(expected.to_bits()) <= 4,
                    "beta={beta:?}, steps={steps}: {actual:?} versus {expected:?}"
                );
            }
        }
    }

    #[test]
    fn decoupled_decay_bias_correction_and_freeze() {
        let c = AdamWConfig::new(0.5, 0.75, 0.25).unwrap();
        let s = AdamWStep::new(0.1, 0.2).unwrap();
        let a = s.candidate(c, 2.0, 3.0, 0.0, 0.0, 0).unwrap();
        assert_eq!(a.first, 1.5);
        assert_eq!(a.second, 2.25);
        assert_eq!(a.steps, 1);
        assert!((a.value - (1.96 - 0.1 * 3.0 / 3.25)).abs() < 2e-7);
        let frozen = AdamWStep::new(0.0, f32::MAX)
            .unwrap()
            .candidate(c, -0.0, f32::MAX, -0.0, 0.0, u32::MAX)
            .unwrap();
        assert_eq!(frozen.value.to_bits(), (-0.0f32).to_bits());
        assert_eq!(frozen.first.to_bits(), (-0.0f32).to_bits());
        assert_eq!(frozen.steps, u32::MAX);
        assert!(s.candidate(c, 1.0, 0.0, 0.0, 0.0, u32::MAX).is_err());
    }

    #[test]
    fn invalid_inputs_never_hide_behind_freezing() {
        let c = AdamWConfig::default();
        for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(AdamWConfig::new(x, 0.9, 1e-8).is_err());
            assert!(AdamWConfig::new(0.9, x, 1e-8).is_err());
            assert!(AdamWConfig::new(0.9, 0.9, x).is_err());
            assert!(AdamWStep::new(x, 0.0).is_err());
            assert!(AdamWStep::new(0.0, x).is_err());
            assert!(AdamWStep::new(0.0, 0.0)
                .unwrap()
                .candidate(c, 1.0, x, 0.0, 0.0, 0)
                .is_err());
        }
        assert!(AdamWConfig::new(1.0, 0.9, 1e-8).is_err());
        assert!(AdamWConfig::new(0.9, -0.1, 1e-8).is_err());
        assert!(AdamWConfig::new(0.9, 0.9, 0.0).is_err());
        let s = AdamWStep::new(0.1, 0.0).unwrap();
        assert!(s.candidate(c, 1.0, f32::MAX, 0.0, 0.0, 0).is_err());
        assert!(s.candidate(c, 1.0, 1.0, 0.0, -1.0, 0).is_err());
    }

    #[test]
    fn portable_extremes_do_not_overflow_or_change_freeze_policy() {
        let tiny = f32::from_bits(1);
        assert!(AdamWStep::new(tiny, 0.0).is_err());
        assert!(AdamWStep::new(0.0, tiny).is_err());
        assert!(AdamWConfig::new(tiny, 0.9, 1e-8).is_err());
        assert!(AdamWConfig::new(0.9, tiny, 1e-8).is_err());
        assert!(AdamWConfig::new(0.9, 0.9, tiny).is_err());
        let config =
            AdamWConfig::new(f32::from_bits(0x3f7fffff), f32::from_bits(0x3f7fffff), 1.0).unwrap();
        let step = AdamWStep::new(1.0, 0.0).unwrap();
        let first = step.candidate(config, 1.0, 1.0, 0.0, 0.0, 0).unwrap();
        assert_eq!(first.value, 0.5);
        assert!(step
            .candidate(config, 1.0, 2.0f32.powi(64), 0.0, 0.0, 0)
            .is_err());
        assert!(checked_mul(f32::MAX, 2.0).is_err());
        assert!(checked_div(f32::MAX, 0.5).is_err());
        assert!(checked_add(f32::MAX, f32::MAX).is_err());
        let oversized = f32::from_bits(0x7f000000);
        assert_eq!(checked_mul(0.0, oversized).unwrap(), 0.0);
        assert_eq!(checked_mul(0.5, oversized).unwrap(), DIVISOR_MAX);
        assert!(checked_mul(2.0, oversized).is_err());
        assert!(checked_mul(oversized, 2.0).is_err());
        assert!(checked_div(1.0, oversized).is_err());
        assert_eq!(checked_div(DIVISOR_MAX, DIVISOR_MAX).unwrap(), 1.0);
        assert!(checked_div(f32::MAX, 1.0).is_err());
        assert!(AdamWStep::new(1.0, 0.0)
            .unwrap()
            .candidate(
                AdamWConfig::new(f32::from_bits(0x3f7fffff), 0.0, 1.0).unwrap(),
                0.0,
                0.0,
                f32::MAX,
                0.0,
                u32::MAX - 1
            )
            .is_err());
    }
}
