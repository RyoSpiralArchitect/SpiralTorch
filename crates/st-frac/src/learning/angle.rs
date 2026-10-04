use super::{FractionalLearningError, Result};
use crate::{checked_f32, validate_slice};

/// Immutable scalar chart alpha = 1 + 2*tan(angle), restricted to alpha > 0.
///
/// Normalized K=3 history then has shape [-cos(angle), sin(angle)]. This chart
/// composes with the existing GL operator; it is not a different convolution.
/// Inputs/outputs are f32, with f64 evaluation of the map and its differential.
/// Derivatives are those of the real map, not the staircase from f32 rounding.
/// No angle wrapping, clipping, projection or optimizer policy is applied.
#[derive(Clone, Copy, Debug)]
pub struct FractionalGlAngleChart {
    angle: f32,
    alpha: f32,
    derivative: f64,
}

impl FractionalGlAngleChart {
    pub fn new(angle: f32) -> Result<Self> {
        let value = f64::from(angle);
        if !value.is_finite() || value <= (-0.5f64).atan() || value >= std::f64::consts::FRAC_PI_2 {
            return Err(FractionalLearningError::HistoryAngle);
        }
        let tangent = value.tan();
        let alpha = checked_f32("fractional angle order", 1.0 + 2.0 * tangent)?;
        if alpha <= 0.0 {
            return Err(FractionalLearningError::HistoryAngle);
        }
        let derivative = 2.0 * (1.0 + tangent * tangent);
        Ok(Self {
            angle,
            alpha,
            derivative,
        })
    }

    pub fn angle(&self) -> f32 {
        self.angle
    }

    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    pub fn alpha_derivative(&self) -> f64 {
        self.derivative
    }

    pub fn vjp(&self, alpha_upstream: f32) -> Result<f32> {
        validate_slice("fractional angle upstream", &[alpha_upstream])?;
        Ok(checked_f32(
            "fractional angle gradient",
            self.derivative * f64::from(alpha_upstream),
        )?)
    }

    pub fn jvp(&self, angle_tangent: f32) -> Result<f32> {
        validate_slice("fractional angle tangent", &[angle_tangent])?;
        Ok(checked_f32(
            "fractional angle tangent",
            self.derivative * f64::from(angle_tangent),
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning::FractionalGlKernel;

    #[test]
    fn scalar_chart_has_true_differentials() {
        for angle in [-0.4, -0.2, 0., 0.4636476, 1.2] {
            let chart = FractionalGlAngleChart::new(angle).unwrap();
            let epsilon = 0.0001;
            let plus = FractionalGlAngleChart::new(angle + epsilon)
                .unwrap()
                .alpha();
            let minus = FractionalGlAngleChart::new(angle - epsilon)
                .unwrap()
                .alpha();
            let finite_difference = (plus - minus) / (2. * epsilon);
            assert!(
                (f64::from(finite_difference) - chart.alpha_derivative()).abs()
                    < 0.002 * chart.alpha_derivative()
            );
            assert_eq!(chart.vjp(-0.3).unwrap(), chart.jvp(-0.3).unwrap());
            assert_eq!(chart.angle(), angle);
        }
        assert_eq!(FractionalGlAngleChart::new(0.).unwrap().alpha(), 1.);
        assert_eq!(FractionalGlAngleChart::new(0.4636476).unwrap().alpha(), 2.);
    }

    #[test]
    fn domain_is_not_wrapped_or_clipped() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.5, 1.6, 6.3] {
            assert!(FractionalGlAngleChart::new(value).is_err());
        }
        let lower = (-0.5f64).atan() as f32;
        assert!(FractionalGlAngleChart::new(lower).is_ok());
        assert!(FractionalGlAngleChart::new(f32::from_bits(lower.to_bits() + 1)).is_err());
        let upper = std::f32::consts::FRAC_PI_2;
        assert!(FractionalGlAngleChart::new(upper).is_err());
        assert!(FractionalGlAngleChart::new(f32::from_bits(upper.to_bits() - 1)).is_ok());
    }

    #[test]
    fn scalar_direction_guards_include_result_overflow() {
        let chart = FractionalGlAngleChart::new(0.).unwrap();
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
            assert!(chart.vjp(bad).is_err());
            assert!(chart.jvp(bad).is_err());
        }
        assert_eq!(chart.vjp(0.).unwrap(), 0.);
    }

    #[test]
    fn normalized_two_tap_matches_ordinary_shape_and_parameter_gradient() {
        let kernel = FractionalGlKernel::new(3, 0.7, 16, 48).unwrap();
        let x = [0.3, -0.7, 1.2, 0.1, -0.2, 0.8, -0.4, 0.5];
        let upstream = [0.2, -0.1, 0.3, -0.4, 0.1, 0.6, 0.2, -0.5];
        for angle in [-0.4, -0.24, 0., 0.4636476, 1.2] {
            let chart = FractionalGlAngleChart::new(angle).unwrap();
            let batch = kernel
                .forward_history_log_gain(&x, &[2, 4], 1, chart.alpha(), 0.3)
                .unwrap();
            let (sin, cos) = f64::from(angle).sin_cos();
            let gain = f64::from(batch.gain());
            let coefficients = [(-gain * cos) as f32, (gain * sin) as f32];
            let mut expected_gradient = 0.;
            for i in 0..x.len() {
                let first = if i % 4 >= 1 { f64::from(x[i - 1]) } else { 0. };
                let second = if i % 4 >= 2 { f64::from(x[i - 2]) } else { 0. };
                let expected =
                    f64::from(coefficients[0]) * first + f64::from(coefficients[1]) * second;
                assert!((f64::from(batch.output().as_slice().unwrap()[i]) - expected).abs() < 3e-7);
                expected_gradient += f64::from(upstream[i]) * gain * (sin * first + cos * second);
            }
            let actual = chart.vjp(batch.vjp_alpha(&upstream).unwrap()).unwrap();
            assert!((f64::from(actual) - expected_gradient).abs() < 3e-7);
        }
    }

    #[test]
    fn chain_rule_also_composes_with_long_nd_history() {
        let kernel = FractionalGlKernel::new(8, 1., 48, 384).unwrap();
        let x: Vec<_> = (0..48).map(|i| (i as f32 * 0.3).sin()).collect();
        let upstream: Vec<_> = x.iter().map(|v| v.cos()).collect();
        for angle in [-0.3, 0., 0.4636476] {
            let chart = FractionalGlAngleChart::new(angle).unwrap();
            let batch = kernel
                .forward_history_log_gain(&x, &[2, 8, 3], 1, chart.alpha(), 0.2)
                .unwrap();
            let derivative = chart.vjp(batch.vjp_alpha(&upstream).unwrap()).unwrap();
            let epsilon = 0.001;
            let score = |value| {
                let chart = FractionalGlAngleChart::new(value).unwrap();
                let batch = kernel
                    .forward_history_log_gain(&x, &[2, 8, 3], 1, chart.alpha(), 0.2)
                    .unwrap();
                batch
                    .output()
                    .iter()
                    .zip(&upstream)
                    .map(|(&y, &g)| f64::from(y) * f64::from(g))
                    .sum::<f64>()
            };
            let fd = (score(angle + epsilon) - score(angle - epsilon)) / f64::from(2. * epsilon);
            assert!((f64::from(derivative) - fd).abs() < 0.002);
        }
    }
}
