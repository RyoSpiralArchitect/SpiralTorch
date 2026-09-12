//! Global L2 clipping of effective optimizer gradients, before learning rates.
//! The small-norm and near-one no-ops preserve ModuleTrainer's historical rule.
use thiserror::Error;

pub const NORM_FLOOR: f32 = f32::EPSILON;
pub const SCALE_EPSILON: f32 = f32::EPSILON;
pub const SCALE_CHUNK: f32 = 5.421011e-20; // 2^-64, a normal f32.
pub const MAX_SCALE_FACTORS: usize = 12;

#[derive(Debug, Error, Clone, Copy, PartialEq)]
pub enum GradientClipError {
    #[error("gradient clip max norm must be positive and finite")]
    MaxNorm,
    #[error("gradient squared norm must be finite and nonnegative")]
    NonFiniteNorm,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobalNormClip {
    max_norm: f32,
}

impl GlobalNormClip {
    pub fn new(max_norm: f32) -> Result<Self, GradientClipError> {
        if !max_norm.is_finite() || max_norm <= 0. {
            return Err(GradientClipError::MaxNorm);
        }
        Ok(Self { max_norm })
    }

    pub fn max_norm(self) -> f32 {
        self.max_norm
    }

    /// Normal mantissa and base-2 exponent, including subnormal f32 limits.
    /// Backends can transport these without flushing the configuration to zero.
    pub fn binary_parts(self) -> (f32, i32) {
        let bits = self.max_norm.to_bits();
        let exponent = (bits >> 23) as i32;
        if exponent == 0 {
            let shift = bits.leading_zeros() - 8;
            (
                f32::from_bits(((bits << shift) & 0x7fffff) | 0x3f000000),
                -125 - shift as i32,
            )
        } else {
            (
                f32::from_bits((bits & 0x7fffff) | 0x3f000000),
                exponent - 126,
            )
        }
    }

    /// Ordered normal-f32 factors avoid underflowing the *scale*, even when
    /// clipping a finite wide gradient to a very small representable value.
    pub fn factors(self, squared_norm: f64) -> Result<ClipFactors, GradientClipError> {
        if !squared_norm.is_finite() || squared_norm < 0. {
            return Err(GradientClipError::NonFiniteNorm);
        }
        let mut result = ClipFactors {
            values: [1.; MAX_SCALE_FACTORS],
            len: 0,
        };
        let norm = squared_norm.sqrt();
        if norm <= f64::from(NORM_FLOOR) || norm <= f64::from(self.max_norm) {
            return Ok(result);
        }
        let mut scale = f64::from(self.max_norm) / norm;
        if (scale as f32 - 1.).abs() <= SCALE_EPSILON {
            return Ok(result);
        }
        while scale < f64::from(SCALE_CHUNK) {
            result.values[result.len] = SCALE_CHUNK;
            result.len += 1;
            scale /= f64::from(SCALE_CHUNK);
        }
        result.values[result.len] = scale as f32;
        result.len += 1;
        Ok(result)
    }
}

#[derive(Clone, Debug)]
pub struct ClipFactors {
    values: [f32; MAX_SCALE_FACTORS],
    len: usize,
}
impl ClipFactors {
    pub fn as_slice(&self) -> &[f32] {
        &self.values[..self.len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_and_preserves_noop_boundaries() {
        for x in [0., -1., f32::INFINITY, f32::NAN] {
            assert!(GlobalNormClip::new(x).is_err());
        }
        let clip = GlobalNormClip::new(1.).unwrap();
        for sq in [0., 0.25, 1., (1. + f64::from(SCALE_EPSILON) / 2.).powi(2)] {
            assert!(clip.factors(sq).unwrap().as_slice().is_empty());
        }
        assert_eq!(clip.factors(4.).unwrap().as_slice(), &[0.5]);
        for sq in [-1., f64::NAN, f64::INFINITY] {
            assert!(clip.factors(sq).is_err());
        }
        assert!(GlobalNormClip::new(1e-30)
            .unwrap()
            .factors(f64::from(NORM_FLOOR).powi(2))
            .unwrap()
            .as_slice()
            .is_empty());
    }

    #[test]
    fn finite_wide_norms_and_underflowing_scale_remain_nonzero() {
        for limit in [1., 1e-30] {
            let factors = GlobalNormClip::new(limit)
                .unwrap()
                .factors(4. * f64::from(f32::MAX).powi(2))
                .unwrap();
            let mut value = f32::MAX;
            for &factor in factors.as_slice() {
                assert!(factor.is_normal() && factor > 0. && factor <= 1.);
                value *= factor;
            }
            assert!((f64::from(value) / f64::from(limit) - 0.5).abs() < 1e-6);
        }
        // Even the full f64 ingress range fits the bounded representation.
        for bits in [1, 0x7fffff, 0x800000, 0x3f800000, 0x7f7fffff] {
            let value = f32::from_bits(bits);
            let (mantissa, exponent) = GlobalNormClip::new(value).unwrap().binary_parts();
            assert!(mantissa.is_normal());
            assert_eq!(f64::from(mantissa) * 2f64.powi(exponent), f64::from(value));
        }
        let tiny = GlobalNormClip::new(f32::from_bits(1)).unwrap();
        assert!(tiny.factors(f64::MAX).unwrap().as_slice().len() <= MAX_SCALE_FACTORS);
    }
}
