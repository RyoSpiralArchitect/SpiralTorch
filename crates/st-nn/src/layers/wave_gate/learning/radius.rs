use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct ProjectionRadius {
    pub radius: f64,
    pub log_radius: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radius_one_matches_legacy_and_scalar_derivative_matches_differences() {
        let kernel = WaveGateKernel::new(-0.7, 1.0, 0.2, 16).unwrap();
        let x = [0.2, -0.3, 1.5, 0.5];
        let gate = [1.4, -1.1];
        let bias = [0.15, -0.05];
        let dy = [0.35, -0.2, -0.1, 0.3];
        let legacy = kernel.forward(&x, &gate, &bias, 2, 2).unwrap();
        let unit = kernel
            .forward_with_log_radius(&x, &gate, &bias, 2, 2, 0.0)
            .unwrap();
        for (a, b) in legacy.output().data().iter().zip(unit.output().data()) {
            assert!((a - b).abs() < 1e-7);
        }
        let a = legacy.vjp(&dy).unwrap();
        let b = unit.vjp(&dy).unwrap();
        for (left, right) in [
            (&a.grad_input, &b.grad_input),
            (&a.grad_gate, &b.grad_gate),
            (&a.grad_bias, &b.grad_bias),
        ] {
            for (a, b) in left.data().iter().zip(right.data()) {
                assert!((a - b).abs() < 1e-7);
            }
        }
        assert!(legacy.vjp_with_log_radius(&dy).is_err());
        for log_radius in [-2.0, 0.0, 2.0] {
            let batch = kernel
                .forward_with_log_radius(&x, &gate, &bias, 2, 2, log_radius)
                .unwrap();
            let (_, analytic) = batch.vjp_with_log_radius(&dy).unwrap();
            let loss = |r| {
                kernel
                    .forward_with_log_radius(&x, &gate, &bias, 2, 2, r)
                    .unwrap()
                    .output()
                    .data()
                    .iter()
                    .zip(dy)
                    .map(|(&y, d)| f64::from(y) * f64::from(d))
                    .sum::<f64>()
            };
            let numeric = (loss(log_radius + 0.001) - loss(log_radius - 0.001)) / 0.002;
            assert!(
                (f64::from(analytic) - numeric).abs() < 1e-5,
                "{analytic} vs {numeric}"
            );
            assert_eq!(batch.vjp_with_log_radius(&dy).unwrap().1, analytic);
        }
    }

    #[test]
    fn radius_preserves_origin_gain_and_rejects_invalid_states() {
        let kernel = WaveGateKernel::new(-4.0, 10.0, 0.2, 4).unwrap();
        for radius in [-80.0, -2.0, 0.0, 2.0, 80.0] {
            let batch = kernel
                .forward_with_log_radius(&[0.0; 2], &[1.0; 2], &[0.0; 2], 1, 2, radius)
                .unwrap();
            let (vjp, dr) = batch.vjp_with_log_radius(&[0.4, -0.2]).unwrap();
            assert_eq!(vjp.grad_input.data(), &[0.2, -0.1]);
            assert_eq!(dr, 0.0);
            let report = batch.conditioning();
            assert_eq!(report.relative_radial_gain_mean, Some(1.0));
            assert_eq!(report.log_radius, Some(radius));
        }
        let batch = kernel
            .forward_with_log_radius(&[], &[1.0; 2], &[0.0; 2], 0, 2, 1.0)
            .unwrap();
        let (vjp, dr) = batch.vjp_with_log_radius(&[]).unwrap();
        assert_eq!(vjp.grad_gate.data(), &[0.0; 2]);
        assert_eq!(dr, 0.0);
        for radius in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -100.0, 100.0] {
            assert!(kernel
                .forward_with_log_radius(&[], &[1.0; 2], &[0.0; 2], 0, 2, radius)
                .is_err());
        }
        let batch = kernel
            .forward_with_log_radius(&[1.0; 2], &[1.0; 2], &[0.0; 2], 1, 2, 0.0)
            .unwrap();
        assert!(batch.vjp_with_log_radius(&[f32::NAN; 2]).is_err());
        assert!(batch.vjp_with_log_radius(&[1.0]).is_err());
    }

    #[test]
    fn small_radius_derivative_does_not_cancel_and_large_inputs_stay_finite() {
        let kernel = WaveGateKernel::new(-1.0, f32::MAX, 0.0, 4).unwrap();
        let batch = kernel
            .forward_with_log_radius(&[1e-8], &[1.0], &[0.0], 1, 1, 0.0)
            .unwrap();
        let (_, dr) = batch.vjp_with_log_radius(&[1.0]).unwrap();
        assert!((f64::from(dr) / (2e-24 / 3.0) - 1.0).abs() < 1e-6);
        for radius in [-80.0, 0.0, 80.0] {
            let batch = kernel
                .forward_with_log_radius(&[1e30, -1e30], &[1.0; 2], &[0.0; 2], 1, 2, radius)
                .unwrap();
            assert!(batch.output().data().iter().all(|v| v.is_finite()));
            let (vjp, dr) = batch.vjp_with_log_radius(&[1.0; 2]).unwrap();
            assert!(vjp.grad_input.data().iter().all(|v| v.is_finite()));
            assert!(dr.is_finite());
        }
    }
}

impl ProjectionRadius {
    pub fn new(log_radius: f32) -> PureResult<Self> {
        let radius = f64::from(log_radius).exp();
        if !log_radius.is_finite()
            || radius < f64::from(f32::MIN_POSITIVE)
            || radius > f64::from(f32::MAX)
        {
            return Err(TensorError::InvalidValue {
                label: "wave_gate_log_radius",
            });
        }
        Ok(Self { radius, log_radius })
    }

    fn row(&self, z: &[f32], scale: f64) -> (f64, f64, f64, f64) {
        let norm = z.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>().sqrt();
        let a = norm / (scale * self.radius);
        let t = a.tanh();
        let tangent = if a == 0.0 { 1.0 } else { t / a };
        let radial = 1.0 - t * t;
        // Avoid subtracting near-equal gains: d output / d log(R) is cubic at zero.
        let log_gain = if a < 1e-3 {
            let a2 = a * a;
            a2 * (2.0 / 3.0 + a2 * (-8.0 / 15.0 + a2 * 34.0 / 105.0))
        } else {
            tangent - radial
        };
        (norm, tangent, radial, log_gain)
    }

    pub fn forward(
        &self,
        topos: &OpenCartesianTopos,
        input: &Tensor,
        gate: &Tensor,
        bias: &Tensor,
    ) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        let scale = f64::from((-topos.curvature()).sqrt());
        let gate: Vec<_> = gate.data().iter().map(|&v| topos.saturate(v)).collect();
        let mut data = Vec::with_capacity(input.data().len());
        for row in input.data().chunks_exact(cols) {
            let z: Vec<_> = (0..cols)
                .map(|c| topos.saturate(row[c] * gate[c] + bias.data()[c]))
                .collect();
            let (_, tangent, _, _) = self.row(&z, scale);
            data.extend(z.iter().map(|&v| (f64::from(v) * tangent / scale) as f32));
        }
        let output = Tensor::from_vec(rows, cols, data)?;
        topos.guard_tensor("wave_gate_radius_output", &output)?;
        Ok(output)
    }

    pub fn vjp(
        &self,
        topos: &OpenCartesianTopos,
        input: &Tensor,
        gate: &Tensor,
        bias: &Tensor,
        upstream: &Tensor,
    ) -> PureResult<(WaveGateVjp, f32)> {
        let (rows, cols) = input.shape();
        let scale = f64::from((-topos.curvature()).sqrt());
        let gates: Vec<_> = gate
            .data()
            .iter()
            .map(|&v| topos.saturate_with_slope(v))
            .collect();
        let mut dx = Vec::with_capacity(input.data().len());
        let mut dg = vec![0.0f64; cols];
        let mut db = vec![0.0f64; cols];
        let mut dr = 0.0f64;
        for (x, dy) in input
            .data()
            .chunks_exact(cols)
            .zip(upstream.data().chunks_exact(cols))
        {
            let affine: Vec<_> = (0..cols)
                .map(|c| topos.saturate_with_slope(x[c] * gates[c].0 + bias.data()[c]))
                .collect();
            let z: Vec<_> = affine.iter().map(|&(v, _)| v).collect();
            let (norm, tangent, radial, log_gain) = self.row(&z, scale);
            let dot = if norm == 0.0 {
                0.0
            } else {
                z.iter()
                    .zip(dy)
                    .map(|(&v, &g)| f64::from(v) / norm * f64::from(g))
                    .sum()
            };
            for c in 0..cols {
                let aligned = if norm == 0.0 {
                    0.0
                } else {
                    f64::from(z[c]) / norm * dot
                };
                let dz = (tangent * (f64::from(dy[c]) - aligned) + radial * aligned) / scale;
                let da = dz * f64::from(affine[c].1);
                dx.push((da * f64::from(gates[c].0)) as f32);
                dg[c] += da * f64::from(x[c]) * f64::from(gates[c].1);
                db[c] += da;
                dr += f64::from(dy[c]) * f64::from(z[c]) * log_gain / scale;
            }
        }
        let result = WaveGateVjp {
            grad_input: Tensor::from_vec(rows, cols, dx)?,
            grad_gate: Tensor::from_vec(1, cols, dg.into_iter().map(|v| v as f32).collect())?,
            grad_bias: Tensor::from_vec(1, cols, db.into_iter().map(|v| v as f32).collect())?,
        };
        for gradient in [&result.grad_input, &result.grad_gate, &result.grad_bias] {
            topos.guard_tensor("wave_gate_radius_gradient", gradient)?;
        }
        let dr = dr as f32;
        if !dr.is_finite() {
            return Err(TensorError::InvalidValue {
                label: "wave_gate_log_radius_gradient",
            });
        }
        Ok((result, dr))
    }
}
