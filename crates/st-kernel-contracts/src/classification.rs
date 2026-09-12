//! Integer-label classification contracts, independent of tensor storage/routes.
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ClassReduction {
    None = 0,
    Sum = 1,
    Mean = 2,
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum ClassificationError {
    #[error("label_smoothing must be finite and between zero and one")]
    Smoothing,
    #[error("class index target must be an integer representable as i64")]
    Label,
    #[error("cross entropy requires a nonempty final class axis")]
    Classes,
    #[error("class targets must match the sample axes, optionally ending in one")]
    Shape,
    #[error("cross entropy sample count overflows")]
    Size,
    #[error("mean cross entropy requires non-ignored labels")]
    EmptyMean,
}

pub fn decode_class_index(value: f32) -> Result<i64, ClassificationError> {
    let wide = f64::from(value);
    if !wide.is_finite()
        || wide.fract() != 0.
        || wide < i64::MIN as f64
        || wide >= -(i64::MIN as f64)
    {
        Err(ClassificationError::Label)
    } else {
        Ok(value as i64)
    }
}

/// Finite nonnegative f64 coefficient transported without flushing it to f32 zero.
/// Backends multiply the mantissa first, then apply the exponent to the product.
#[derive(Clone, Copy, Debug)]
pub struct BinaryScale {
    pub mantissa: f32,
    pub exponent: i32,
}

impl BinaryScale {
    fn new(value: f64) -> Self {
        if value == 0. {
            return Self {
                mantissa: 0.,
                exponent: 0,
            };
        }
        let (normal, shift) = if value < f64::MIN_POSITIVE {
            (value * 4_503_599_627_370_496., -52)
        } else {
            (value, 0)
        };
        let bits = normal.to_bits();
        Self {
            mantissa: f64::from_bits((bits & ((1u64 << 52) - 1)) | (1022u64 << 52)) as f32,
            exponent: ((bits >> 52) & 2047) as i32 - 1022 + shift,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CrossEntropySpec {
    reduction: ClassReduction,
    ignore_index: i64,
    smoothing: f64,
}

impl CrossEntropySpec {
    pub fn new(
        reduction: ClassReduction,
        ignore_index: i64,
        smoothing: f64,
    ) -> Result<Self, ClassificationError> {
        if !smoothing.is_finite() || !(0.0..=1.0).contains(&smoothing) {
            return Err(ClassificationError::Smoothing);
        }
        Ok(Self {
            reduction,
            ignore_index,
            smoothing,
        })
    }
    pub fn reduction(self) -> ClassReduction {
        self.reduction
    }
    pub fn smoothing(self) -> f64 {
        self.smoothing
    }
    /// None means no valid f32 label can equal this i64. Never round an ignore ID.
    pub fn ignore_transport(self) -> Option<f32> {
        let value = self.ignore_index as f32;
        (decode_class_index(value) == Ok(self.ignore_index)).then_some(value)
    }
    pub fn nll_scale(self) -> BinaryScale {
        BinaryScale::new(1. - self.smoothing)
    }
    pub fn uniform_scale(self, classes: usize) -> Result<BinaryScale, ClassificationError> {
        if classes == 0 {
            return Err(ClassificationError::Classes);
        }
        Ok(BinaryScale::new(self.smoothing / classes as f64))
    }
    /// Class-last logits; None keeps a singleton class axis in the output.
    pub fn shapes(
        self,
        logits: &[usize],
        targets: &[usize],
    ) -> Result<(usize, usize, Vec<usize>), ClassificationError> {
        let (&classes, samples) = logits.split_last().ok_or(ClassificationError::Classes)?;
        if classes == 0 {
            return Err(ClassificationError::Classes);
        }
        let mut expanded = samples.to_vec();
        expanded.push(1);
        if targets != samples && targets != expanded {
            return Err(ClassificationError::Shape);
        }
        let rows = samples
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or(ClassificationError::Size)?;
        if rows == 0 && self.reduction == ClassReduction::Mean {
            return Err(ClassificationError::EmptyMean);
        }
        Ok((
            rows,
            classes,
            if self.reduction == ClassReduction::None {
                expanded
            } else {
                vec![1, 1]
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transport_never_rounds_ignore_ids_or_saturates_i64() {
        for id in [-100, i64::MIN, 1i64 << 40] {
            assert_eq!(
                CrossEntropySpec::new(ClassReduction::Mean, id, 0.)
                    .unwrap()
                    .ignore_transport(),
                Some(id as f32)
            );
        }
        for id in [i64::MAX, 16_777_217, (1i64 << 40) + 1] {
            assert_eq!(
                CrossEntropySpec::new(ClassReduction::Mean, id, 0.)
                    .unwrap()
                    .ignore_transport(),
                None
            );
        }
        for v in [0.5, f32::NAN, f32::INFINITY, i64::MAX as f32, -f32::MAX] {
            assert!(decode_class_index(v).is_err());
        }
    }
    #[test]
    fn coefficients_keep_tiny_smoothing_and_shapes_keep_sample_axes() {
        for value in [0., 0.2, 1., 1e-40, 1e-300, f64::from_bits(1)] {
            let p = BinaryScale::new(value);
            let reconstructed = if p.exponent < -1022 {
                (f64::from(p.mantissa) * 2f64.powi(p.exponent + 52)) * 2f64.powi(-52)
            } else {
                f64::from(p.mantissa) * 2f64.powi(p.exponent)
            };
            assert!((reconstructed - value).abs() <= value * 1e-7);
        }
        let spec = CrossEntropySpec::new(ClassReduction::None, -100, 0.2).unwrap();
        for target in [&[2, 3][..], &[2, 3, 1][..]] {
            assert_eq!(
                spec.shapes(&[2, 3, 257], target).unwrap(),
                (6, 257, vec![2, 3, 1])
            );
        }
        assert!(spec.shapes(&[2, 3, 257], &[6, 1]).is_err());
        assert_eq!(spec.shapes(&[0, 3], &[0]).unwrap(), (0, 3, vec![0, 1]));
        assert!(CrossEntropySpec::new(ClassReduction::Mean, -100, 0.)
            .unwrap()
            .shapes(&[0, 3], &[0])
            .is_err());
        for s in [f64::NAN, -0.1, 1.1] {
            assert!(CrossEntropySpec::new(ClassReduction::Mean, -100, s).is_err());
        }
    }
}
