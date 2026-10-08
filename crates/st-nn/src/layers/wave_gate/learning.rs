use super::*;
use crate::execution::{push_backend_policy, BackendPolicy};
use st_core::backend::device_caps::DeviceCaps;

mod radius;
use radius::ProjectionRadius;

/// Immutable CPU recipe for external optimizers; no training-policy rewrites.
#[derive(Clone, Debug)]
pub struct WaveGateKernel {
    topos: OpenCartesianTopos,
}

/// Owned forward inputs and recipe. Reusable VJPs never read live parameters.
#[derive(Debug)]
pub struct WaveGateLearningBatch {
    kernel: WaveGateKernel,
    input: Tensor,
    gate: Tensor,
    bias: Tensor,
    output: Tensor,
    radius: Option<ProjectionRadius>,
}

/// Local conditioning of the captured map, not a full loss-gradient estimate.
/// Projection gains are relative to its zero-norm Jacobian (I / sqrt(-k)).
#[derive(Clone, Debug, serde::Serialize)]
pub struct WaveGateConditioning {
    pub schema: &'static str,
    pub rows: usize,
    pub features: usize,
    pub gate_outside_values: usize,
    pub affine_outside_values: usize,
    pub affine_nonfinite_values: usize,
    pub affine_abs_slope_mean: Option<f64>,
    pub dimensionless_norm_mean: Option<f64>,
    pub dimensionless_norm_max: Option<f64>,
    pub relative_radial_gain_mean: Option<f64>,
    pub relative_radial_gain_min: Option<f64>,
    pub relative_tangential_gain_mean: Option<f64>,
    pub relative_tangential_gain_min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub projection_radius: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_radius: Option<f32>,
}

impl WaveGateKernel {
    pub fn new(
        curvature: f32,
        saturation: f32,
        porosity: f32,
        max_values: usize,
    ) -> PureResult<Self> {
        Ok(Self {
            topos: OpenCartesianTopos::new(curvature, 1e-6, saturation, 64, max_values)?
                .with_porosity(porosity)?,
        })
    }

    pub fn topos(&self) -> &OpenCartesianTopos {
        &self.topos
    }

    pub fn forward(
        &self,
        input: &[f32],
        gate: &[f32],
        bias: &[f32],
        rows: usize,
        features: usize,
    ) -> PureResult<WaveGateLearningBatch> {
        self.forward_impl(input, gate, bias, rows, features, None)
    }

    /// R=exp(log_radius), with fixed origin gain I/sqrt(-curvature).
    /// The caller owns this scalar parameter; the snapshot owns its value.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_log_radius(
        &self,
        input: &[f32],
        gate: &[f32],
        bias: &[f32],
        rows: usize,
        features: usize,
        log_radius: f32,
    ) -> PureResult<WaveGateLearningBatch> {
        self.forward_impl(
            input,
            gate,
            bias,
            rows,
            features,
            Some(ProjectionRadius::new(log_radius)?),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_impl(
        &self,
        input: &[f32],
        gate: &[f32],
        bias: &[f32],
        rows: usize,
        features: usize,
        radius: Option<ProjectionRadius>,
    ) -> PureResult<WaveGateLearningBatch> {
        let count = rows.checked_mul(features);
        if features == 0 || count != Some(input.len()) {
            return Err(TensorError::InvalidDimensions {
                rows,
                cols: features,
            });
        }
        if input.len() > self.topos.max_volume() || features > self.topos.max_volume() {
            return Err(TensorError::InvalidValue {
                label: "wave_gate_value_budget",
            });
        }
        let input = Tensor::from_vec(rows, features, input.to_vec())?;
        let gate = Tensor::from_vec(1, features, gate.to_vec())?;
        let bias = Tensor::from_vec(1, features, bias.to_vec())?;
        self.topos
            .guard_tensor("wave_gate_learning_input", &input)?;
        self.topos.guard_tensor("wave_gate_learning_gate", &gate)?;
        self.topos.guard_tensor("wave_gate_learning_bias", &bias)?;
        // Client transport explicitly promises CPU execution, independent of
        // thread-local trainer policy. The guard also restores the caller's policy.
        let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
        let output = if let Some(radius) = radius {
            radius.forward(&self.topos, &input, &gate, &bias)?
        } else {
            self.layer(&gate, &bias)?.forward(&input)?
        };
        Ok(WaveGateLearningBatch {
            kernel: self.clone(),
            input,
            gate,
            bias,
            output,
            radius,
        })
    }

    fn layer(&self, gate: &Tensor, bias: &Tensor) -> PureResult<WaveGate> {
        Ok(WaveGate {
            gate: Parameter::new("learning::gate", gate.clone()),
            bias: Parameter::new("learning::bias", bias.clone()),
            topos: self.topos.clone(),
            encoder: LanguageWaveEncoder::new(self.topos.curvature(), 1.0)?,
        })
    }
}

impl WaveGateLearningBatch {
    pub fn output(&self) -> &Tensor {
        &self.output
    }

    /// Computes scalar diagnostics on demand; it never changes the saved VJP.
    pub fn conditioning(&self) -> WaveGateConditioning {
        let (rows, features) = self.input.shape();
        let topos = &self.kernel.topos;
        let scale = f64::from((-topos.curvature()).sqrt()) * self.radius.map_or(1.0, |r| r.radius);
        let gate: Vec<_> = self
            .gate
            .data()
            .iter()
            .map(|&v| topos.saturate(v))
            .collect();
        let mut affine_outside_values = 0;
        let mut affine_nonfinite_values = 0;
        let mut slope_sum = 0.0;
        let mut norm_sum = 0.0;
        let mut norm_max = 0.0f64;
        let mut radial_sum = 0.0;
        let mut radial_min = f64::INFINITY;
        let mut tangent_sum = 0.0;
        let mut tangent_min = f64::INFINITY;
        for row in self.input.data().chunks_exact(features) {
            let mut norm_sq = 0.0;
            for col in 0..features {
                let affine = row[col] * gate[col] + self.bias.data()[col];
                affine_outside_values += usize::from(affine.abs() > topos.saturation());
                affine_nonfinite_values += usize::from(!affine.is_finite());
                let (z, slope) = topos.saturate_with_slope(affine);
                slope_sum += f64::from(slope.abs());
                norm_sq += f64::from(z).powi(2);
            }
            let ratio = norm_sq.sqrt() / scale;
            let tanh = ratio.tanh();
            let radial = 1.0 - tanh * tanh;
            let tangent = if ratio == 0.0 { 1.0 } else { tanh / ratio };
            norm_sum += ratio;
            norm_max = norm_max.max(ratio);
            radial_sum += radial;
            radial_min = radial_min.min(radial);
            tangent_sum += tangent;
            tangent_min = tangent_min.min(tangent);
        }
        WaveGateConditioning {
            schema: if self.radius.is_some() {
                "spiraltorch.wave_gate_conditioning.v2"
            } else {
                "spiraltorch.wave_gate_conditioning.v1"
            },
            rows,
            features,
            gate_outside_values: self
                .gate
                .data()
                .iter()
                .filter(|v| v.abs() > topos.saturation())
                .count(),
            affine_outside_values,
            affine_nonfinite_values,
            affine_abs_slope_mean: (rows > 0).then(|| slope_sum / (rows * features) as f64),
            dimensionless_norm_mean: (rows > 0).then(|| norm_sum / rows as f64),
            dimensionless_norm_max: (rows > 0).then_some(norm_max),
            relative_radial_gain_mean: (rows > 0).then(|| radial_sum / rows as f64),
            relative_radial_gain_min: (rows > 0).then_some(radial_min),
            relative_tangential_gain_mean: (rows > 0).then(|| tangent_sum / rows as f64),
            relative_tangential_gain_min: (rows > 0).then_some(tangent_min),
            projection_radius: self.radius.map(|r| r.radius),
            log_radius: self.radius.map(|r| r.log_radius),
        }
    }

    /// Sum-reduced parameter derivatives, without mutation or optimizer policy.
    pub fn vjp(&self, upstream: &[f32]) -> PureResult<WaveGateVjp> {
        if self.radius.is_some() {
            return self.vjp_with_log_radius(upstream).map(|(vjp, _)| vjp);
        }
        let (rows, cols) = self.input.shape();
        let upstream = Tensor::from_vec(rows, cols, upstream.to_vec())?;
        let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
        self.kernel
            .layer(&self.gate, &self.bias)?
            .vjp(&self.input, &upstream)
    }

    /// Sum-reduced input/gate/bias derivatives plus the shared log-radius derivative.
    /// Rejects legacy snapshots rather than inventing a radius parameter for them.
    pub fn vjp_with_log_radius(&self, upstream: &[f32]) -> PureResult<(WaveGateVjp, f32)> {
        let radius = self.radius.ok_or(TensorError::InvalidValue {
            label: "wave_gate_snapshot_has_no_log_radius",
        })?;
        let (rows, cols) = self.input.shape();
        let upstream = Tensor::from_vec(rows, cols, upstream.to_vec())?;
        self.kernel
            .topos
            .guard_tensor("wave_gate_radius_upstream", &upstream)?;
        radius.vjp(
            &self.kernel.topos,
            &self.input,
            &self.gate,
            &self.bias,
            &upstream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conditioning_matches_radial_and_tangential_vjps_without_mutation() {
        let kernel = WaveGateKernel::new(-1.0, 10.0, 0.2, 4).unwrap();
        let batch = kernel
            .forward(&[0.3, 0.4], &[1.0; 2], &[0.0; 2], 1, 2)
            .unwrap();
        let report = batch.conditioning();
        assert_eq!(report.affine_outside_values, 0);
        assert_eq!(report.affine_abs_slope_mean, Some(1.0));
        for (seed, gain) in [
            ([0.6, 0.8], report.relative_radial_gain_mean.unwrap()),
            ([-0.8, 0.6], report.relative_tangential_gain_mean.unwrap()),
        ] {
            let actual = batch.vjp(&seed).unwrap();
            for (&v, s) in actual.grad_input.data().iter().zip(seed) {
                assert!((f64::from(v) - f64::from(s) * gain).abs() < 1e-7);
            }
        }
        let empty = kernel
            .forward(&[], &[1.0; 2], &[0.0; 2], 0, 2)
            .unwrap()
            .conditioning();
        assert!(empty.dimensionless_norm_mean.is_none());
        assert!(empty.relative_radial_gain_min.is_none());
        assert!(serde_json::to_string(&empty).is_ok());
        let zero = kernel
            .forward(&[0.0; 2], &[0.0; 2], &[0.0; 2], 1, 2)
            .unwrap()
            .conditioning();
        assert_eq!(zero.relative_radial_gain_mean, Some(1.0));
        assert_eq!(zero.relative_tangential_gain_mean, Some(1.0));
        let saturated = WaveGateKernel::new(-1.0, 1.0, 0.2, 4)
            .unwrap()
            .forward(&[4.0; 2], &[2.0; 2], &[0.0; 2], 1, 2)
            .unwrap()
            .conditioning();
        assert_eq!(saturated.gate_outside_values, 2);
        assert_eq!(saturated.affine_outside_values, 2);
        assert!(saturated.affine_abs_slope_mean.unwrap() < 1.0);
    }

    #[test]
    fn snapshot_owns_inputs_and_matches_native_parameter_sum() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<WaveGateLearningBatch>();
        let kernel = WaveGateKernel::new(-1.0, 1.0, 0.2, 16).unwrap();
        let mut input = vec![0.2, -0.3, 0.4, 0.5];
        let mut gate = vec![1.4, -1.1];
        let mut bias = vec![0.15, -0.05];
        let batch = kernel.forward(&input, &gate, &bias, 2, 2).unwrap();
        let seed = [0.35, -0.2, -0.1, 0.3];
        let expected = kernel
            .layer(
                &Tensor::from_vec(1, 2, gate.clone()).unwrap(),
                &Tensor::from_vec(1, 2, bias.clone()).unwrap(),
            )
            .unwrap()
            .vjp(
                &Tensor::from_vec(2, 2, input.clone()).unwrap(),
                &Tensor::from_vec(2, 2, seed.to_vec()).unwrap(),
            )
            .unwrap();
        input.fill(f32::NAN);
        gate.fill(0.0);
        bias.fill(f32::INFINITY);
        let actual = batch.vjp(&seed).unwrap();
        assert_eq!(actual.grad_input.data(), expected.grad_input.data());
        assert_eq!(actual.grad_gate.data(), expected.grad_gate.data());
        assert_eq!(actual.grad_bias.data(), expected.grad_bias.data());
        assert_eq!(
            batch.vjp(&seed).unwrap().grad_gate.data(),
            expected.grad_gate.data()
        );
        assert!(batch.vjp(&[f32::NAN; 4]).is_err());
        assert!(batch.vjp(&[0.0]).is_err());
    }

    #[test]
    fn snapshot_empty_shapes_budgets_and_finite_guards() {
        let kernel = WaveGateKernel::new(-1.0, 1.0, 0.2, 4).unwrap();
        let empty = kernel.forward(&[], &[1.0, 2.0], &[0.0; 2], 0, 2).unwrap();
        assert_eq!(empty.output().shape(), (0, 2));
        let gradients = empty.vjp(&[]).unwrap();
        assert_eq!(gradients.grad_gate.data(), &[0.0; 2]);
        assert_eq!(gradients.grad_bias.data(), &[0.0; 2]);
        for invalid in [
            kernel.forward(&[], &[], &[], 0, 0),
            kernel.forward(&[], &[], &[], usize::MAX, 2),
            kernel.forward(&[0.0; 6], &[1.0; 2], &[0.0; 2], 3, 2),
            kernel.forward(&[], &[0.0; 5], &[0.0; 5], 0, 5),
            kernel.forward(&[0.0; 2], &[f32::NAN; 2], &[0.0; 2], 1, 2),
            kernel.forward(&[0.0; 2], &[1.0; 2], &[f32::INFINITY; 2], 1, 2),
        ] {
            assert!(invalid.is_err());
        }
        for curvature in [0.0, 1.0, f32::NAN, f32::NEG_INFINITY] {
            assert!(WaveGateKernel::new(curvature, 1.0, 0.2, 4).is_err());
        }
    }
}
