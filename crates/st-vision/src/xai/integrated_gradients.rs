use st_nn::module::Module;
use st_tensor::{PureResult, Tensor, TensorError};

use super::utils::{accumulate_inplace, add, mul, scale, sub};

#[derive(Debug, Clone, Copy)]
pub struct IntegratedGradientsConfig {
    pub steps: usize,
    pub target_index: usize,
}

impl IntegratedGradientsConfig {
    pub fn new(steps: usize, target_index: usize) -> Self {
        Self {
            steps,
            target_index,
        }
    }
}

pub struct IntegratedGradients;

impl IntegratedGradients {
    pub fn attribute<M: Module>(
        model: &mut M,
        input: &Tensor,
        baseline: &Tensor,
        config: &IntegratedGradientsConfig,
    ) -> PureResult<Tensor> {
        if config.steps == 0 {
            return Err(TensorError::InvalidValue {
                label: "integrated_gradients_steps",
            });
        }
        let (rows, cols) = input.shape();
        if baseline.shape() != (rows, cols) {
            return Err(TensorError::ShapeMismatch {
                left: baseline.shape(),
                right: (rows, cols),
            });
        }
        let delta = sub(input, baseline)?;
        let mut total_grad = Tensor::zeros(rows, cols)?;
        for step in 1..=config.steps {
            let alpha = step as f32 / config.steps as f32;
            let scaled_delta = scale(&delta, alpha)?;
            let interpolated = add(baseline, &scaled_delta)?;
            let output = model.forward(&interpolated)?;
            let (out_rows, out_cols) = output.shape();
            let total = out_rows * out_cols;
            if config.target_index >= total {
                return Err(TensorError::InvalidValue {
                    label: "integrated_gradients_target",
                });
            }
            let mut grad_output = Tensor::zeros(out_rows, out_cols)?;
            grad_output.data_mut()[config.target_index] = 1.0;
            let grad_input = model.backward(&interpolated, &grad_output)?;
            accumulate_inplace(&mut total_grad, &grad_input)?;
        }
        let average_grad = scale(&total_grad, 1.0 / config.steps as f32)?;
        mul(&average_grad, &delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_tensor::Tensor;

    struct IdentityModule;

    impl Module for IdentityModule {
        fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
            Ok(input.clone())
        }

        fn backward(&mut self, _input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
            Ok(grad_output.clone())
        }

        fn visit_parameters(
            &self,
            _visitor: &mut dyn FnMut(&st_nn::module::Parameter) -> PureResult<()>,
        ) -> PureResult<()> {
            Ok(())
        }

        fn visit_parameters_mut(
            &mut self,
            _visitor: &mut dyn FnMut(&mut st_nn::module::Parameter) -> PureResult<()>,
        ) -> PureResult<()> {
            Ok(())
        }
    }

    #[test]
    fn integrated_gradients_identity_matches_difference() {
        let input = Tensor::from_vec(1, 3, vec![0.2, 0.4, 0.6]).unwrap();
        let baseline = Tensor::zeros(1, 3).unwrap();
        let mut module = IdentityModule;
        let config = IntegratedGradientsConfig::new(16, 0);
        let attribution =
            IntegratedGradients::attribute(&mut module, &input, &baseline, &config).unwrap();
        assert!((attribution.data()[0] - input.data()[0]).abs() < 1e-6);
        assert!(attribution.data()[1].abs() < 1e-6);
        assert!(attribution.data()[2].abs() < 1e-6);
    }

    #[test]
    fn integrated_gradients_rejects_zero_steps() {
        let input = Tensor::zeros(1, 1).unwrap();
        let baseline = Tensor::zeros(1, 1).unwrap();
        let mut module = IdentityModule;
        let config = IntegratedGradientsConfig::new(0, 0);
        let result = IntegratedGradients::attribute(&mut module, &input, &baseline, &config);
        assert!(result.is_err());
    }
}
