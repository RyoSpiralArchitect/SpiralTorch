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
