//! Finite Picard kernel beneath the open-topos policy and audit layer.
//! Geometry, depth/volume budgets and runtime routing remain caller-owned.

use thiserror::Error;

pub const MAX_ITERATIONS: usize = 4096;

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum ToposKernelError {
    #[error("Topos coupling must be finite and in [0, 1)")]
    Coupling,
    #[error("Topos saturation must be finite and positive")]
    Saturation,
    #[error("Topos porosity must be finite and in [0, 1]")]
    Porosity,
    #[error("Topos iterations must be in 1..=4096")]
    Iterations,
    #[error("Topos input, intermediate or gradient is non-finite")]
    NonFinite,
}

/// Validated scalar execution parameters, not an admission or geometry policy.
#[derive(Clone, Copy, Debug)]
pub struct ToposResonatorKernel {
    coupling: f32,
    saturation: f32,
    porosity: f32,
    iterations: usize,
}

impl PartialEq for ToposResonatorKernel {
    fn eq(&self, other: &Self) -> bool {
        self.coupling.to_bits() == other.coupling.to_bits()
            && self.saturation.to_bits() == other.saturation.to_bits()
            && self.porosity.to_bits() == other.porosity.to_bits()
            && self.iterations == other.iterations
    }
}

impl Eq for ToposResonatorKernel {}

impl ToposResonatorKernel {
    pub fn new(
        coupling: f32,
        saturation: f32,
        porosity: f32,
        iterations: usize,
    ) -> Result<Self, ToposKernelError> {
        if !coupling.is_finite() || !(0.0..1.0).contains(&coupling) {
            return Err(ToposKernelError::Coupling);
        }
        if !saturation.is_finite() || saturation <= 0.0 {
            return Err(ToposKernelError::Saturation);
        }
        if !porosity.is_finite() || !(0.0..=1.0).contains(&porosity) {
            return Err(ToposKernelError::Porosity);
        }
        if !(1..=MAX_ITERATIONS).contains(&iterations) {
            return Err(ToposKernelError::Iterations);
        }
        Ok(Self {
            coupling,
            saturation,
            porosity,
            iterations,
        })
    }

    pub fn coupling(self) -> f32 {
        self.coupling
    }
    pub fn saturation(self) -> f32 {
        self.saturation
    }
    pub fn porosity(self) -> f32 {
        self.porosity
    }
    pub fn iterations(self) -> usize {
        self.iterations
    }

    /// Output and exact finite-unroll drive sensitivity. No fixed-point or
    /// straight-through derivative is substituted at the saturation boundary.
    pub fn capture(self, input: f32, gate: f32) -> Result<(f32, f32), ToposKernelError> {
        checked(input)?;
        checked(gate)?;
        let drive = checked(input * gate)?;
        let mut state = 0.0;
        let mut sensitivity = 0.0;
        for _ in 0..self.iterations {
            let raw = checked(drive + self.coupling * state)?;
            let slope = porous_mix_slope(raw, self.saturation, self.porosity);
            state = checked(porous_mix(raw, self.saturation, self.porosity))?;
            sensitivity = checked(slope * (1.0 + self.coupling * sensitivity))?;
        }
        Ok((state, sensitivity))
    }

    /// Scalar contributions before any shared-gate sum. Forward validity is
    /// checked even for a zero cotangent.
    pub fn vjp(self, input: f32, gate: f32, cotangent: f32) -> Result<[f32; 2], ToposKernelError> {
        checked(cotangent)?;
        let (_, sensitivity) = self.capture(input, gate)?;
        let drive = checked(cotangent * sensitivity)?;
        Ok([checked(drive * gate)?, checked(drive * input)?])
    }
}

fn checked(value: f32) -> Result<f32, ToposKernelError> {
    value
        .is_finite()
        .then_some(value)
        .ok_or(ToposKernelError::NonFinite)
}

/// OpenCartesianTopos' existing porous rewrite, including its permissive
/// non-finite fallback. Checked execution must reject invalid operands first.
#[inline]
pub fn porous_mix(value: f32, saturation: f32, porosity: f32) -> f32 {
    if !value.is_finite() || saturation <= 0.0 {
        return 0.0;
    }
    let limit = saturation.abs();
    let magnitude = value.abs();
    if magnitude <= limit {
        return value;
    }
    if porosity <= f32::EPSILON {
        return value.signum() * limit;
    }
    let relative_limit = limit / magnitude;
    let bleed = (1.0 - relative_limit) / (1.0 + relative_limit);
    let absorb = (porosity * 0.25).min(1.0);
    let softened = limit * (1.0 - absorb * bleed.min(1.0)).max(0.0);
    value.signum() * softened
}

/// Boundary convention is slope 1 at |value| == saturation, including zero.
#[inline]
pub fn porous_mix_slope(value: f32, saturation: f32, porosity: f32) -> f32 {
    if !value.is_finite() || saturation <= 0.0 {
        return 0.0;
    }
    let limit = saturation.abs();
    let magnitude = value.abs();
    if magnitude <= limit {
        return 1.0;
    }
    if porosity <= f32::EPSILON {
        return 0.0;
    }
    let absorb = (porosity * 0.25).min(1.0);
    let ratio = f64::from(limit) / (f64::from(magnitude) + f64::from(limit));
    (-2.0 * f64::from(absorb) * ratio * ratio) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_gate_has_no_porosity_signal_but_can_learn() {
        for coupling in [0., 0.25, 0.95] {
            for iterations in [1, 4, 41] {
                let gain = (0..iterations).fold(0.0, |a, _| 1.0 + coupling * a);
                for porosity in [0., 0.3, 1.] {
                    let kernel =
                        ToposResonatorKernel::new(coupling, 1., porosity, iterations).unwrap();
                    for input in [-2., -0.25, 0., 0.25, 2.] {
                        let (output, sensitivity) = kernel.capture(input, 0.).unwrap();
                        assert_eq!(output, 0.);
                        assert_eq!(sensitivity, gain);
                        assert_eq!(kernel.vjp(input, 0., 1.).unwrap(), [0., gain * input]);
                    }
                }
            }
        }
    }

    #[test]
    fn hard_rewrite_matches_a_clipped_gain_control_away_from_kinks() {
        for coupling in [0., 0.2, 0.5, 0.95] {
            for iterations in [1, 5, 41] {
                let kernel = ToposResonatorKernel::new(coupling, 1., 0., iterations).unwrap();
                let gain = (0..iterations).fold(0.0_f64, |a, _| 1.0 + f64::from(coupling) * a);
                for input in [-3., -0.9, -0.1, 0., 0.1, 0.9, 3.] {
                    let gate = 0.8;
                    let scaled = f64::from(input) * f64::from(gate) * gain;
                    assert!((scaled.abs() - 1.).abs() > 1e-4);
                    let expected = scaled.clamp(-1., 1.) as f32;
                    let slope = if scaled.abs() < 1. { gain as f32 } else { 0. };
                    let (output, sensitivity) = kernel.capture(input, gate).unwrap();
                    assert!((output - expected).abs() <= 2e-6);
                    assert!((sensitivity - slope).abs() <= 2e-6 * (1. + slope));
                    let [dx, dg] = kernel.vjp(input, gate, 1.).unwrap();
                    assert!((dx - slope * gate).abs() <= 2e-6 * (1. + slope));
                    assert!((dg - slope * input).abs() <= 2e-6 * (1. + slope));
                }
            }
        }
    }

    #[test]
    fn porous_tail_is_distinct_from_hard_clipping() {
        let hard = ToposResonatorKernel::new(0.25, 1., 0., 4).unwrap();
        let porous = ToposResonatorKernel::new(0.25, 1., 0.3, 4).unwrap();
        for input in [-2., 2.] {
            let (hard_output, hard_slope) = hard.capture(input, 1.).unwrap();
            let (porous_output, porous_slope) = porous.capture(input, 1.).unwrap();
            assert_eq!(hard_output, input.signum());
            assert_eq!(hard_slope, 0.);
            assert!(porous_output.abs() < hard_output.abs());
            assert!(porous_slope < 0.);
            assert_eq!(hard.vjp(input, 1., 1.).unwrap(), [0., 0.]);
            let [dx, dg] = porous.vjp(input, 1., 1.).unwrap();
            assert!(dx < 0. && dg * input < 0.);
        }
    }

    #[test]
    fn parameters_and_nonfinite_intermediates_fail_closed() {
        for value in [f32::NAN, f32::INFINITY, -1., 1.] {
            assert!(ToposResonatorKernel::new(value, 1., 0., 1).is_err());
        }
        for value in [f32::NAN, f32::INFINITY, -1., 0.] {
            assert!(ToposResonatorKernel::new(0., value, 0., 1).is_err());
        }
        for value in [f32::NAN, f32::INFINITY, -1., 1.01] {
            assert!(ToposResonatorKernel::new(0., 1., value, 1).is_err());
        }
        for iterations in [0, MAX_ITERATIONS + 1, usize::MAX] {
            assert!(ToposResonatorKernel::new(0., 1., 0., iterations).is_err());
        }
        let k = ToposResonatorKernel::new(0.5, f32::MAX, 0., 2).unwrap();
        for (input, gate) in [(f32::MAX, 2.), (f32::MAX, 1.), (f32::NAN, 0.)] {
            assert_eq!(k.vjp(input, gate, 0.), Err(ToposKernelError::NonFinite));
        }
        assert!(k.vjp(1., 1., f32::MAX).is_err());
    }

    #[test]
    fn unrolled_vjp_matches_finite_difference_away_from_kinks() {
        for porosity in [0., 0.3, 1.] {
            for iterations in [1, 5, 41] {
                let k = ToposResonatorKernel::new(0.2, 1., porosity, iterations).unwrap();
                for input in [-2., -0.3, 0., 0.3, 2.] {
                    let gate = 0.9;
                    let [dx, dg] = k.vjp(input, gate, 1.).unwrap();
                    let h = 1e-3;
                    let numerical_x = (k.capture(input + h, gate).unwrap().0
                        - k.capture(input - h, gate).unwrap().0)
                        / (2. * h);
                    let numerical_g = (k.capture(input, gate + h).unwrap().0
                        - k.capture(input, gate - h).unwrap().0)
                        / (2. * h);
                    assert!((dx - numerical_x).abs() < 2e-4);
                    assert!((dg - numerical_g).abs() < 2e-4);
                }
            }
        }
        for input in [-1., 1.] {
            assert_eq!(porous_mix_slope(input, 1., 0.3), 1.);
        }
        assert_eq!(porous_mix(-0., 1., 0.3).to_bits(), (-0f32).to_bits());
        assert!(porous_mix_slope(2., 1., 0.3) < 0.);
    }
}
