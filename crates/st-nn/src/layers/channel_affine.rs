//! Ordinary channel-wise affine control for geometry learning comparisons.
//! The existing tensor primitives own arithmetic; snapshots own the pullback.

use crate::execution::{push_backend_policy, BackendPolicy};
use st_core::backend::device_caps::DeviceCaps;
use st_tensor::{PureResult, Tensor, TensorError, TensorUtilBackend};

#[derive(Clone, Debug)]
pub struct ChannelAffineKernel {
    max_values: usize,
}

#[derive(Debug)]
pub struct ChannelAffineLearningBatch {
    input: Tensor,
    gate: Vec<f32>,
    output: Tensor,
}

#[derive(Debug)]
pub struct ChannelAffineVjp {
    pub grad_input: Tensor,
    pub grad_gate: Tensor,
    pub grad_bias: Tensor,
}

impl ChannelAffineKernel {
    pub fn new(max_values: usize) -> PureResult<Self> {
        if max_values == 0 || u32::try_from(max_values).is_err() {
            return Err(TensorError::InvalidValue {
                label: "channel_affine_value_budget",
            });
        }
        Ok(Self { max_values })
    }

    pub fn max_values(&self) -> usize {
        self.max_values
    }

    /// y[r,c] = input[r,c] * gate[c] + bias[c]. No saturation or projection.
    pub fn forward(
        &self,
        input: &[f32],
        gate: &[f32],
        bias: &[f32],
        rows: usize,
        features: usize,
    ) -> PureResult<ChannelAffineLearningBatch> {
        if features == 0 || rows.checked_mul(features) != Some(input.len()) {
            return Err(TensorError::InvalidDimensions {
                rows,
                cols: features,
            });
        }
        if input.len() > self.max_values || features > self.max_values {
            return Err(TensorError::InvalidValue {
                label: "channel_affine_value_budget",
            });
        }
        for values in [gate, bias] {
            if values.len() != features {
                return Err(TensorError::DataLength {
                    expected: features,
                    got: values.len(),
                });
            }
        }
        let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
        let input = Tensor::from_vec(rows, features, input.to_vec())?;
        let output = input.row_affine_with_backend(gate, bias, TensorUtilBackend::Cpu)?;
        Ok(ChannelAffineLearningBatch {
            input,
            gate: gate.to_vec(),
            output,
        })
    }
}

impl ChannelAffineLearningBatch {
    pub fn output(&self) -> &Tensor {
        &self.output
    }

    /// Parameter gradients are sums, not row means, using shared tensor reductions.
    pub fn vjp(&self, upstream: &[f32]) -> PureResult<ChannelAffineVjp> {
        let (rows, features) = self.input.shape();
        let expected = self.input.data().len();
        if upstream.len() != expected {
            return Err(TensorError::DataLength {
                expected,
                got: upstream.len(),
            });
        }
        let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
        let upstream = Tensor::from_vec(rows, features, upstream.to_vec())?;
        let grad_input = upstream.mul_row_with_backend(&self.gate, TensorUtilBackend::Cpu)?;
        let products = upstream.hadamard_with_backend(&self.input, TensorUtilBackend::Cpu)?;
        let grad_gate = Tensor::from_vec(
            1,
            features,
            products.try_sum_axis0_scaled_with_backend(1.0, TensorUtilBackend::Cpu)?,
        )?;
        let grad_bias = Tensor::from_vec(
            1,
            features,
            upstream.try_sum_axis0_scaled_with_backend(1.0, TensorUtilBackend::Cpu)?,
        )?;
        Ok(ChannelAffineVjp {
            grad_input,
            grad_gate,
            grad_bias,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry_artifact::GeometryAdapterArtifact;

    #[test]
    fn shared_native_python_wasm_fixture() {
        let f: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/channel_affine.fixture.json"
        ))
        .unwrap();
        let values = |name: &str| -> Vec<f32> {
            f[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        };
        let batch = ChannelAffineKernel::new(f["max_values"].as_u64().unwrap() as usize)
            .unwrap()
            .forward(
                &values("input"),
                &values("gate"),
                &values("bias"),
                f["rows"].as_u64().unwrap() as usize,
                f["features"].as_u64().unwrap() as usize,
            )
            .unwrap();
        assert_eq!(batch.output().data(), values("output"));
        let gradient = batch.vjp(&values("upstream")).unwrap();
        assert_eq!(gradient.grad_input.data(), values("grad_input"));
        assert_eq!(gradient.grad_gate.data(), values("grad_gate"));
        assert_eq!(gradient.grad_bias.data(), values("grad_bias"));
        let artifact =
            GeometryAdapterArtifact::create_json(f["artifact"].to_string().as_bytes()).unwrap();
        let loaded = GeometryAdapterArtifact::from_json(
            artifact.to_json().as_bytes(),
            artifact.artifact_id(),
            f["artifact"]["base_model_identity"].as_str().unwrap(),
        )
        .unwrap();
        assert_eq!(loaded.to_json(), artifact.to_json());
        assert_eq!(loaded.layout_id().unwrap(), artifact.layout_id().unwrap());
    }

    #[test]
    fn native_forward_pullback_and_owned_snapshot() {
        let kernel = ChannelAffineKernel::new(6).unwrap();
        let mut input = vec![1., 2., 3., 4., 5., 6.];
        let mut gate = vec![2., -1., 0.5];
        let batch = kernel.forward(&input, &gate, &[1., 2., 3.], 2, 3).unwrap();
        input.fill(99.);
        gate.fill(99.);
        assert_eq!(batch.output().data(), &[3., 0., 4.5, 9., -3., 6.]);
        let vjp = batch.vjp(&[1., 2., 3., 4., 5., 6.]).unwrap();
        assert_eq!(vjp.grad_input.data(), &[2., -2., 1.5, 8., -5., 3.]);
        assert_eq!(vjp.grad_gate.data(), &[17., 29., 45.]);
        assert_eq!(vjp.grad_bias.data(), &[5., 7., 9.]);
        assert_eq!(batch.vjp(&[1.; 6]).unwrap().grad_gate.data(), &[5., 7., 9.]);
    }

    #[test]
    fn identity_start_matches_wave_gate_first_update_derivatives() {
        let x = [1., -2., 3., 4., 0.5, -1.];
        let dy = [0.1, 0.3, -0.2, 1., 0.5, 2.];
        let affine = ChannelAffineKernel::new(6)
            .unwrap()
            .forward(&x, &[0.; 3], &[0.; 3], 2, 3)
            .unwrap();
        let wave = crate::WaveGateKernel::new(-1., 1., 0.05, 6)
            .unwrap()
            .forward(&x, &[0.; 3], &[0.; 3], 2, 3)
            .unwrap();
        assert_eq!(affine.output().data(), wave.output().data());
        let a = affine.vjp(&dy).unwrap();
        let b = wave.vjp(&dy).unwrap();
        assert_eq!(a.grad_input.data(), b.grad_input.data());
        assert_eq!(a.grad_gate.data(), b.grad_gate.data());
        assert_eq!(a.grad_bias.data(), b.grad_bias.data());
    }

    #[test]
    fn finite_budget_shape_and_empty_rows() {
        assert!(ChannelAffineKernel::new(0).is_err());
        let kernel = ChannelAffineKernel::new(2).unwrap();
        assert!(kernel.forward(&[1.; 4], &[1.; 2], &[0.; 2], 2, 2).is_err());
        assert!(kernel.forward(&[1.], &[1.], &[0.], usize::MAX, 2).is_err());
        assert!(kernel.forward(&[1.], &[], &[0.], 1, 1).is_err());
        for bad in [f32::NAN, f32::INFINITY] {
            assert!(kernel.forward(&[bad], &[1.], &[0.], 1, 1).is_err());
            assert!(kernel.forward(&[], &[bad], &[0.], 0, 1).is_err());
        }
        assert!(kernel.forward(&[f32::MAX], &[2.], &[0.], 1, 1).is_err());
        let empty = kernel.forward(&[], &[1., 2.], &[0., 0.], 0, 2).unwrap();
        assert!(empty.output().data().is_empty());
        assert_eq!(empty.vjp(&[]).unwrap().grad_bias.data(), &[0., 0.]);
        let batch = kernel.forward(&[1., 2.], &[1.], &[0.], 2, 1).unwrap();
        assert!(batch.vjp(&[1.]).is_err());
        assert!(matches!(
            batch.vjp(&[1.; 10]),
            Err(TensorError::DataLength {
                expected: 2,
                got: 10
            })
        ));
        assert!(batch.vjp(&[f32::NAN, 1.]).is_err());
        assert!(batch.vjp(&[f32::MAX; 2]).is_err());
    }
}
