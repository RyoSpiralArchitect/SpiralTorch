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
        self.row_gains(norm, scale)
    }

    pub(super) fn row_gains(&self, norm: f64, scale: f64) -> (f64, f64, f64, f64) {
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
        stages: WaveStages,
    ) -> PureResult<Tensor> {
        match stages {
            WaveStages::Both => self.forward_map::<true, true>(topos, input, gate, bias),
            WaveStages::Saturation => self.forward_map::<true, false>(topos, input, gate, bias),
            WaveStages::Projection => self.forward_map::<false, true>(topos, input, gate, bias),
        }
    }

    fn forward_map<const SAT: bool, const PROJECT: bool>(
        &self,
        topos: &OpenCartesianTopos,
        input: &Tensor,
        gate: &Tensor,
        bias: &Tensor,
    ) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        let scale = f64::from((-topos.curvature()).sqrt());
        let rewrite = |v| if SAT { topos.saturate(v) } else { v };
        let gate: Vec<_> = gate.data().iter().copied().map(rewrite).collect();
        let mut data = Vec::with_capacity(input.data().len());
        let mut z = Vec::with_capacity(cols);
        for row in input.data().chunks_exact(cols) {
            z.clear();
            z.extend((0..cols).map(|c| rewrite(row[c] * gate[c] + bias.data()[c])));
            if PROJECT {
                let (_, tangent, _, _) = self.row(&z, scale);
                data.extend(z.iter().map(|&v| (f64::from(v) * tangent / scale) as f32));
            } else {
                data.extend_from_slice(&z);
            }
        }
        let output = Tensor::from_vec(rows, cols, data)?;
        topos.guard_tensor("wave_gate_radius_output", &output)?;
        Ok(output)
    }

    fn visit_affine_adjoint(
        &self,
        topos: &OpenCartesianTopos,
        tensors: [&Tensor; 4],
        need_radius: bool,
        stages: WaveStages,
        consume: impl FnMut(usize, f64, (f32, f32), f32),
    ) -> f64 {
        match stages {
            WaveStages::Both => self.visit_map::<true, true>(topos, tensors, need_radius, consume),
            WaveStages::Saturation => self.visit_map::<true, false>(topos, tensors, false, consume),
            WaveStages::Projection => self.visit_map::<false, true>(topos, tensors, false, consume),
        }
    }

    fn visit_map<const SAT: bool, const PROJECT: bool>(
        &self,
        topos: &OpenCartesianTopos,
        [input, gate, bias, upstream]: [&Tensor; 4],
        need_radius: bool,
        mut consume: impl FnMut(usize, f64, (f32, f32), f32),
    ) -> f64 {
        let (_, cols) = input.shape();
        let scale = f64::from((-topos.curvature()).sqrt());
        let rewrite = |x| {
            if SAT {
                topos.saturate_with_slope(x)
            } else {
                (x, 1.)
            }
        };
        let gates: Vec<_> = gate.data().iter().map(|&v| rewrite(v)).collect();
        let mut dr = 0.0f64;
        // Reuse row-local storage without changing reduction order or precision.
        let mut affine = Vec::with_capacity(cols);
        let mut z = Vec::with_capacity(cols);
        for (x, dy) in input
            .data()
            .chunks_exact(cols)
            .zip(upstream.data().chunks_exact(cols))
        {
            affine.clear();
            affine.extend((0..cols).map(|c| rewrite(x[c] * gates[c].0 + bias.data()[c])));
            if !PROJECT {
                for c in 0..cols {
                    consume(c, f64::from(dy[c]) * f64::from(affine[c].1), gates[c], x[c]);
                }
                continue;
            }
            z.clear();
            z.extend(affine.iter().map(|&(v, _)| v));
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
                consume(c, da, gates[c], x[c]);
                if need_radius {
                    dr += f64::from(dy[c]) * f64::from(z[c]) * log_gain / scale;
                }
            }
        }
        dr
    }

    pub(super) fn vjp_input(
        &self,
        topos: &OpenCartesianTopos,
        tensors: [&Tensor; 4],
        stages: WaveStages,
    ) -> PureResult<Tensor> {
        let (rows, cols) = tensors[0].shape();
        let mut dx = Vec::with_capacity(tensors[0].data().len());
        self.visit_affine_adjoint(topos, tensors, false, stages, |_, da, (g, _), _| {
            dx.push((da * f64::from(g)) as f32);
        });
        let input = Tensor::from_vec(rows, cols, dx)?;
        topos.guard_tensor("wave_gate_radius_gradient", &input)?;
        Ok(input)
    }

    pub fn vjp(
        &self,
        topos: &OpenCartesianTopos,
        input: &Tensor,
        gate: &Tensor,
        bias: &Tensor,
        upstream: &Tensor,
        [need_input, need_radius]: [bool; 2],
        stages: WaveStages,
    ) -> PureResult<(WaveGatePartialVjp, Option<f32>)> {
        let (rows, cols) = input.shape();
        let mut dx = need_input.then(|| Vec::with_capacity(input.data().len()));
        let mut dg = vec![0.0f64; cols];
        let mut db = vec![0.0f64; cols];
        let dr = self.visit_affine_adjoint(
            topos,
            [input, gate, bias, upstream],
            need_radius,
            stages,
            |c, da, (g, dg_draw), x| {
                if let Some(dx) = &mut dx {
                    dx.push((da * f64::from(g)) as f32);
                }
                dg[c] += da * f64::from(x) * f64::from(dg_draw);
                db[c] += da;
            },
        );
        let result = WaveGatePartialVjp {
            grad_input: dx.map(|dx| Tensor::from_vec(rows, cols, dx)).transpose()?,
            grad_gate: Tensor::from_vec(1, cols, dg.into_iter().map(|v| v as f32).collect())?,
            grad_bias: Tensor::from_vec(1, cols, db.into_iter().map(|v| v as f32).collect())?,
        };
        for gradient in result
            .grad_input
            .iter()
            .chain([&result.grad_gate, &result.grad_bias])
        {
            topos.guard_tensor("wave_gate_radius_gradient", gradient)?;
        }
        let dr = need_radius.then_some(dr as f32);
        if dr.is_some_and(|value| !value.is_finite()) {
            return Err(TensorError::InvalidValue {
                label: "wave_gate_log_radius_gradient",
            });
        }
        Ok((result, dr))
    }
}
