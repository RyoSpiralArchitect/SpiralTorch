//! Last-axis affine LayerNorm shape and scalar contract, independent of devices.

use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum LayerNormError {
    #[error("LayerNorm requires a nonempty last axis")]
    Axis,
    #[error("LayerNorm affine operands must have shape [cols] or [1, cols]")]
    AffineShape,
    #[error("LayerNorm epsilon must be finite and nonnegative")]
    Epsilon,
    #[error("LayerNorm affine gradient scale must be finite")]
    GradientScale,
    #[error("LayerNorm cotangent shape must equal the input shape")]
    CotangentShape,
    #[error("LayerNorm shape product overflows")]
    Overflow,
}

/// Normalize the last axis. Empty leading axes are valid and have zero rows.
#[derive(Clone, Copy, Debug)]
pub struct LayerNormShape {
    pub rows: usize,
    pub cols: usize,
}

impl LayerNormShape {
    pub fn new(shape: &[usize], affine: &[usize]) -> Result<Self, LayerNormError> {
        let cols = *shape
            .last()
            .filter(|&&n| n > 0)
            .ok_or(LayerNormError::Axis)?;
        if affine != [cols] && affine != [1, cols] {
            return Err(LayerNormError::AffineShape);
        }
        let rows = shape[..shape.len() - 1]
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .ok_or(LayerNormError::Overflow)?;
        rows.checked_mul(cols).ok_or(LayerNormError::Overflow)?;
        Ok(Self { rows, cols })
    }
}

pub fn validate_epsilon(epsilon: f32) -> Result<(), LayerNormError> {
    if !epsilon.is_finite() || epsilon < 0.0 {
        Err(LayerNormError::Epsilon)
    } else {
        Ok(())
    }
}

/// This scale is applied after the row sum to gamma/beta gradients, never dx.
pub fn validate_gradient_scale(scale: f32) -> Result<(), LayerNormError> {
    if scale.is_finite() {
        Ok(())
    } else {
        Err(LayerNormError::GradientScale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn last_axis_and_scalars_are_explicit() {
        assert_eq!(LayerNormShape::new(&[2, 3, 4], &[4]).unwrap().rows, 6);
        assert_eq!(LayerNormShape::new(&[2, 0, 4], &[1, 4]).unwrap().rows, 0);
        for shape in [vec![], vec![0], vec![3, 0]] {
            assert_eq!(
                LayerNormShape::new(&shape, &[0]).unwrap_err(),
                LayerNormError::Axis
            );
        }
        assert!(LayerNormShape::new(&[2, 3], &[3, 1]).is_err());
        assert!(LayerNormShape::new(&[usize::MAX, 2], &[2]).is_err());
        for epsilon in [0.0, f32::from_bits(1), f32::MAX] {
            assert!(validate_epsilon(epsilon).is_ok());
        }
        for epsilon in [-1.0, f32::INFINITY, f32::NAN] {
            assert!(validate_epsilon(epsilon).is_err());
        }
        assert!(validate_gradient_scale(-2.0).is_ok());
        assert!(validate_gradient_scale(f32::NAN).is_err());
    }
}
