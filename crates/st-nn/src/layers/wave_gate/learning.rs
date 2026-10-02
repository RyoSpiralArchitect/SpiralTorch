use super::*;
use crate::execution::{push_backend_policy, BackendPolicy};
use st_core::backend::device_caps::DeviceCaps;

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
        let output = self.layer(&gate, &bias)?.forward(&input)?;
        Ok(WaveGateLearningBatch {
            kernel: self.clone(),
            input,
            gate,
            bias,
            output,
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

    /// Sum-reduced parameter derivatives, without mutation or optimizer policy.
    pub fn vjp(&self, upstream: &[f32]) -> PureResult<WaveGateVjp> {
        let (rows, cols) = self.input.shape();
        let upstream = Tensor::from_vec(rows, cols, upstream.to_vec())?;
        let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
        self.kernel
            .layer(&self.gate, &self.bias)?
            .vjp(&self.input, &upstream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
