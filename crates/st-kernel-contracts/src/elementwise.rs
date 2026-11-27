//! Finite f32 elementwise semantics shared by host and resident tensors.

/// Scalar tanh policy for the host GELU forward and derivative.
pub const HOST_GELU_TANH: &str = if cfg!(feature = "portable-host-gelu") {
    "portable_libm_tanh_f32.v1"
} else {
    "platform_tanh_f32.v1"
};

/// Scalar tanh used inside host GELU and its input derivative. The opt-in
/// implementation is shared by native and WASM; it does not change GPU shaders.
#[inline]
pub fn gelu_tanh(inner: f32) -> f32 {
    #[cfg(feature = "portable-host-gelu")]
    {
        libm::tanhf(inner)
    }
    #[cfg(not(feature = "portable-host-gelu"))]
    {
        inner.tanh()
    }
}

/// Discriminants are the backend dispatch contract, not a runtime route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ElementwiseOp {
    Identity = 0,
    Add = 1,
    Multiply = 2,
    Relu = 3,
    Gelu = 4,
    Subtract = 5,
    Divide = 6,
}

impl ElementwiseOp {
    pub fn is_binary(self) -> bool {
        matches!(
            self,
            Self::Add | Self::Multiply | Self::Subtract | Self::Divide
        )
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
            Self::Subtract => a - b,
            Self::Divide => {
                if b == 0.0 {
                    return None;
                }
                a / b
            }
            Self::Relu => a.max(0.0),
            Self::Gelu => {
                let square = a * a;
                let cubic = square * a;
                let inner = 0.7978846 * (a + 0.044715 * cubic);
                if !square.is_finite() || !cubic.is_finite() || !inner.is_finite() {
                    return None;
                }
                0.5 * a * (1.0 + gelu_tanh(inner.clamp(-10.0, 10.0)))
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
            Self::Subtract => (1., -1.),
            Self::Divide => {
                let value = Self::Divide.apply(a, b)?;
                (Self::Divide.apply(1., b)?, Self::Divide.apply(-value, b)?)
            }
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
    let t = gelu_tanh(inner);
    let derivative =
        0.5 * (1. + t) + 0.5 * x * (1. - t * t) * 0.7978846 * (1. + 3. * 0.044715 * square);
    derivative.is_finite().then_some(derivative)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_gelu_uses_the_selected_tanh_for_values_and_partials() {
        #[cfg(feature = "portable-host-gelu")]
        let (tanh, policy) = (libm::tanhf, "portable_libm_tanh_f32.v1");
        #[cfg(not(feature = "portable-host-gelu"))]
        let (tanh, policy) = (f32::tanh, "platform_tanh_f32.v1");
        assert_eq!(HOST_GELU_TANH, policy);
        // Includes the three distinct inputs where the two fixture hosts disagreed.
        for x in [
            f32::from_bits(0xbe22c063),
            f32::from_bits(0xbe3ce4af),
            f32::from_bits(0x3df3a458),
            -2.5,
            -0.0,
            0.0,
            0.125,
            3.0,
        ] {
            let square = x * x;
            let cubic = square * x;
            let forward =
                0.5 * x * (1. + tanh((0.7978846 * (x + 0.044715 * cubic)).clamp(-10., 10.)));
            let t = tanh(0.7978846 * (x + 0.044715 * x * square));
            let derivative =
                0.5 * (1. + t) + 0.5 * x * (1. - t * t) * 0.7978846 * (1. + 3. * 0.044715 * square);
            assert_eq!(
                ElementwiseOp::Gelu.apply(x, 0.).unwrap().to_bits(),
                forward.to_bits()
            );
            assert_eq!(gelu_derivative(x).unwrap().to_bits(), derivative.to_bits());
        }
        assert_eq!(gelu_tanh(-0.).to_bits(), (-0f32).to_bits());
        assert_eq!(gelu_derivative(-10.), Some(0.));
        assert_eq!(gelu_derivative(10.), Some(1.));
        assert_eq!(gelu_derivative(f32::NAN), None);
    }

    #[test]
    fn subtract_and_divide_check_forward_and_local_partials() {
        assert_eq!(ElementwiseOp::Subtract.apply(6., 2.), Some(4.));
        assert_eq!(ElementwiseOp::Subtract.apply(f32::MAX, -f32::MAX), None);
        assert_eq!(ElementwiseOp::Divide.apply(4., 2.), Some(2.));
        for zero in [0., -0.] {
            assert_eq!(ElementwiseOp::Divide.apply(0., zero), None);
            assert_eq!(ElementwiseOp::Divide.partials(0., zero), None);
        }
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(ElementwiseOp::Subtract.apply(1., invalid), None);
            assert_eq!(ElementwiseOp::Divide.apply(invalid, 1.), None);
            assert_eq!(ElementwiseOp::Divide.apply(1., invalid), None);
        }
        assert_eq!(ElementwiseOp::Divide.apply(f32::MAX, 0.5), None);
        assert_eq!(ElementwiseOp::Subtract.partials(4., 2.), Some((1., -1.)));
        assert_eq!(ElementwiseOp::Divide.partials(4., 2.), Some((0.5, -1.)));
        // Squaring the denominator here would overflow even though both partials exist.
        let (lhs, rhs) = ElementwiseOp::Divide.partials(1e20, 1e20).unwrap();
        assert_eq!(lhs, 1e-20);
        assert_eq!(rhs, -1e-20);
        assert_eq!(
            ElementwiseOp::Divide.apply(-0., 2.).unwrap().to_bits(),
            (-0f32).to_bits()
        );
    }

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
