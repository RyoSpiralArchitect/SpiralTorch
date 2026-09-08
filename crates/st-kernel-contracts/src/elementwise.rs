//! Finite f32 elementwise semantics shared by host and resident tensors.

/// Discriminants are the backend dispatch contract, not a runtime route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ElementwiseOp {
    Identity = 0,
    Add = 1,
    Multiply = 2,
    Relu = 3,
    Gelu = 4,
}

impl ElementwiseOp {
    pub fn is_binary(self) -> bool {
        matches!(self, Self::Add | Self::Multiply)
    }

    /// None means a non-finite input, intermediate or output. GELU uses the
    /// existing tanh approximation and checked, saturated dense-kernel policy.
    pub fn apply(self, a: f32, b: f32) -> Option<f32> {
        if !a.is_finite() || (self.is_binary() && !b.is_finite()) {
            return None;
        }
        let value = match self {
            Self::Identity => a,
            Self::Add => a + b,
            Self::Multiply => a * b,
            Self::Relu => a.max(0.0),
            Self::Gelu => {
                let square = a * a;
                let cubic = square * a;
                let inner = 0.7978846 * (a + 0.044715 * cubic);
                if !square.is_finite() || !cubic.is_finite() || !inner.is_finite() {
                    return None;
                }
                0.5 * a * (1.0 + inner.clamp(-10.0, 10.0).tanh())
            }
        };
        value.is_finite().then_some(value)
    }

    /// Local partials of the checked operation. The full VJP also evaluates
    /// forward intermediates, even when the incoming cotangent is zero.
    pub fn partials(self, a: f32, b: f32) -> Option<(f32, f32)> {
        if !a.is_finite() || (self.is_binary() && !b.is_finite()) {
            return None;
        }
        Some(match self {
            Self::Identity => (1., 0.),
            Self::Add => (1., 1.),
            Self::Multiply => (b, a),
            Self::Relu => (if a > 0. { 1. } else { 0. }, 0.),
            Self::Gelu => (gelu_derivative(a)?, 0.),
        })
    }
}

/// Saturated tanh-GELU derivative shared with the Tensor/Module host path.
pub fn gelu_derivative(x: f32) -> Option<f32> {
    if !x.is_finite() {
        return None;
    }
    if x.abs() >= 10. {
        return Some(if x > 0. { 1. } else { 0. });
    }
    let square = x * x;
    let inner = 0.7978846 * (x + 0.044715 * x * square);
    let t = inner.tanh();
    let derivative =
        0.5 * (1. + t) + 0.5 * x * (1. - t * t) * 0.7978846 * (1. + 3. * 0.044715 * square);
    derivative.is_finite().then_some(derivative)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_policy_includes_intermediates_and_signed_zero_copy() {
        assert_eq!(
            ElementwiseOp::Identity.apply(-0.0, 0.0).unwrap().to_bits(),
            (-0.0f32).to_bits()
        );
        assert_eq!(ElementwiseOp::Multiply.apply(f32::MAX, 2.0), None);
        assert_eq!(ElementwiseOp::Gelu.apply(-f32::MAX, 0.0), None);
        assert_eq!(ElementwiseOp::Relu.apply(f32::NEG_INFINITY, 0.0), None);
        assert_eq!(ElementwiseOp::Gelu.apply(100.0, 0.0), Some(100.0));
    }
}
