// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::Module;
use crate::{PureResult, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use st_tensor::TensorError;
use std::cell::RefCell;

/// Stochastic dropout layer that scales retained activations during training
/// and becomes a no-op when evaluation mode is enabled.
#[derive(Debug)]
pub struct Dropout {
    probability: f32,
    training: bool,
    rng: RefCell<StdRng>,
    last_mask: RefCell<Option<Tensor>>,
}

impl Dropout {
    /// Creates a new dropout layer with the provided drop probability.
    ///
    /// `probability` must lie in `[0, 1)`. During training the layer retains
    /// activations with probability `1 - probability` and rescales them by the
    /// inverse keep probability to preserve expectation.
    pub fn new(probability: f32) -> PureResult<Self> {
        Self::with_seed(probability, None)
    }

    /// Creates a new dropout layer with a deterministic seed used for unit
    /// tests and reproducible experiments.
    pub fn with_seed(probability: f32, seed: Option<u64>) -> PureResult<Self> {
        if probability < 0.0 || probability >= 1.0 {
            return Err(TensorError::InvalidValue {
                label: "dropout_probability",
            });
        }
        let rng = match seed {
            Some(value) => StdRng::seed_from_u64(value),
            None => StdRng::from_entropy(),
        };
        Ok(Self {
            probability,
            training: true,
            rng: RefCell::new(rng),
            last_mask: RefCell::new(None),
        })
    }

    /// Returns the configured drop probability.
    pub fn probability(&self) -> f32 {
        self.probability
    }

    /// Sets the layer to training (`true`) or evaluation (`false`) mode.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        if !training {
            self.last_mask.borrow_mut().take();
        }
    }

    /// Returns whether the layer is currently in training mode.
    pub fn training(&self) -> bool {
        self.training
    }

    /// Convenience helper that switches the layer into training mode and clears
    /// any cached inference mask.
    pub fn train(&mut self) {
        self.set_training(true);
    }

    /// Convenience helper that switches the layer into evaluation mode,
    /// dropping any cached stochastic mask so gradients flow transparently.
    pub fn eval(&mut self) {
        self.set_training(false);
    }

    fn keep_probability(&self) -> f32 {
        1.0 - self.probability
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let keep_probability = self.keep_probability();
        if !self.training || keep_probability == 1.0 {
            self.last_mask.borrow_mut().take();
            return Ok(input.clone());
        }

        let (rows, cols) = input.shape();
        if rows == 0 || cols == 0 {
            return Err(TensorError::EmptyInput("dropout forward input"));
        }

        let mut rng = self.rng.borrow_mut();
        let mut mask_data = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            let sample: f32 = rng.gen();
            if sample < keep_probability {
                mask_data.push(1.0 / keep_probability);
            } else {
                mask_data.push(0.0);
            }
        }
        drop(rng);

        let mask = Tensor::from_vec(rows, cols, mask_data)?;
        let output = input.hadamard(&mask)?;
        self.last_mask.borrow_mut().replace(mask);
        Ok(output)
    }

    fn backward(&mut self, _input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let keep_probability = self.keep_probability();
        if !self.training || keep_probability == 1.0 {
            return Ok(grad_output.clone());
        }

        let mask_guard = self.last_mask.borrow();
        let Some(mask) = mask_guard.as_ref() else {
            return Err(TensorError::InvalidValue {
                label: "dropout_mask_missing",
            });
        };
        grad_output.hadamard(mask)
    }

    fn visit_parameters(
        &self,
        _visitor: &mut dyn FnMut(&crate::module::Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        _visitor: &mut dyn FnMut(&mut crate::module::Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::Rng;

    #[test]
    fn dropout_forward_matches_mask() {
        let layer = Dropout::with_seed(0.25, Some(42)).unwrap();
        let input = Tensor::from_vec(1, 6, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output = layer.forward(&input).unwrap();

        let keep_probability = 1.0 - layer.probability();
        let mut rng = StdRng::seed_from_u64(42);
        let mut expected = Vec::new();
        for value in input.data() {
            let sample: f32 = rng.gen();
            if sample < keep_probability {
                expected.push(value * (1.0 / keep_probability));
            } else {
                expected.push(0.0);
            }
        }
        let expected_tensor = Tensor::from_vec(1, 6, expected).unwrap();
        assert_eq!(output, expected_tensor);
    }

    #[test]
    fn dropout_backward_uses_mask() {
        let mut layer = Dropout::with_seed(0.5, Some(7)).unwrap();
        let input = Tensor::from_vec(1, 4, vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let _ = layer.forward(&input).unwrap();
        let grad_output = Tensor::from_vec(1, 4, vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();

        let keep_probability = 1.0 - layer.probability();
        let mut rng = StdRng::seed_from_u64(7);
        let mut expected = Vec::new();
        for grad in grad_output.data() {
            let sample: f32 = rng.gen();
            if sample < keep_probability {
                expected.push(grad * (1.0 / keep_probability));
            } else {
                expected.push(0.0);
            }
        }
        let expected_tensor = Tensor::from_vec(1, 4, expected).unwrap();
        assert_eq!(grad_input, expected_tensor);
    }

    #[test]
    fn dropout_evaluation_is_identity() {
        let mut layer = Dropout::with_seed(0.4, Some(5)).unwrap();
        layer.set_training(false);
        let input = Tensor::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output, input);

        let grad_output = Tensor::from_vec(2, 2, vec![0.5, 0.25, -0.75, 1.25]).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input, grad_output);
    }
}
