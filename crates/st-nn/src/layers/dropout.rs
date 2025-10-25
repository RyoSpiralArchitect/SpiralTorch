// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::Module;
use crate::{PureResult, Tensor, TensorError};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::cell::{Cell, RefCell};

/// Bernoulli dropout layer that mirrors the behaviour of `nn.Dropout`.
pub struct Dropout {
    probability: f32,
    keep_scale: f32,
    train: Cell<bool>,
    rng: RefCell<StdRng>,
    last_mask: RefCell<Option<Tensor>>,
}

impl core::fmt::Debug for Dropout {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Dropout")
            .field("probability", &self.probability)
            .field("training", &self.train.get())
            .finish()
    }
}

impl Dropout {
    /// Builds a new dropout layer using entropy from the host.
    pub fn new(probability: f32) -> PureResult<Self> {
        Self::with_seed(probability, None)
    }

    /// Builds a new dropout layer with a deterministic RNG seed.
    pub fn with_seed(probability: f32, seed: Option<u64>) -> PureResult<Self> {
        if !(0.0..1.0).contains(&probability) {
            return Err(TensorError::InvalidValue {
                label: "dropout_probability",
            });
        }
        let keep = 1.0 - probability;
        let keep_scale = if keep == 0.0 {
            return Err(TensorError::InvalidValue {
                label: "dropout_probability",
            });
        } else {
            1.0 / keep
        };
        let rng = match seed {
            Some(value) => StdRng::seed_from_u64(value),
            None => StdRng::from_entropy(),
        };
        Ok(Self {
            probability,
            keep_scale,
            train: Cell::new(true),
            rng: RefCell::new(rng),
            last_mask: RefCell::new(None),
        })
    }

    /// Returns the probability assigned to zeroing activations.
    pub fn probability(&self) -> f32 {
        self.probability
    }

    /// Returns whether the layer currently runs in training mode.
    pub fn is_training(&self) -> bool {
        self.train.get()
    }

    /// Switches the layer into training or evaluation mode.
    pub fn set_training(&self, training: bool) {
        self.train.set(training);
        if !training {
            self.last_mask.borrow_mut().take();
        }
    }

    /// Convenience helper that enables training mode.
    pub fn train(&self) {
        self.set_training(true);
    }

    /// Convenience helper that enables evaluation mode.
    pub fn eval(&self) {
        self.set_training(false);
    }

    fn guard_input(&self, input: &Tensor) -> PureResult<()> {
        if input.len() == 0 {
            return Err(TensorError::EmptyInput("dropout_forward"));
        }
        Ok(())
    }

    fn sample_mask(&self, rows: usize, cols: usize) -> PureResult<Tensor> {
        let mut rng = self.rng.borrow_mut();
        let mut mask = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            let keep = rng.gen::<f32>() >= self.probability;
            if keep {
                mask.push(self.keep_scale);
            } else {
                mask.push(0.0);
            }
        }
        Tensor::from_vec(rows, cols, mask)
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        self.guard_input(input)?;
        if !self.is_training() || self.probability == 0.0 {
            self.last_mask.borrow_mut().take();
            return Ok(input.clone());
        }

        let (rows, cols) = input.shape();
        let mask = self.sample_mask(rows, cols)?;
        self.last_mask.borrow_mut().replace(mask.clone());
        input.hadamard(&mask)
    }

    fn backward(&mut self, _input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        if !self.is_training() || self.probability == 0.0 {
            return Ok(grad_output.clone());
        }
        let mask = {
            let borrow = self.last_mask.borrow();
            borrow.clone()
        };
        let Some(mask) = mask else {
            return Err(TensorError::InvalidValue {
                label: "dropout_mask",
            });
        };
        let grad = grad_output.hadamard(&mask)?;
        self.last_mask.borrow_mut().take();
        Ok(grad)
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
    use rand::{rngs::StdRng, Rng, SeedableRng};

    fn mask_from_seed(probability: f32, seed: u64, rows: usize, cols: usize) -> Tensor {
        let mut rng = StdRng::seed_from_u64(seed);
        let keep_scale = 1.0 / (1.0 - probability);
        let mut mask = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            let keep = rng.gen::<f32>() >= probability;
            mask.push(if keep { keep_scale } else { 0.0 });
        }
        Tensor::from_vec(rows, cols, mask).unwrap()
    }

    #[test]
    fn dropout_rejects_invalid_probability() {
        assert!(Dropout::with_seed(-0.1, Some(1)).is_err());
        assert!(Dropout::with_seed(1.0, Some(1)).is_err());
    }

    #[test]
    fn dropout_forward_and_backward_match_mask() {
        let seed = 42u64;
        let mut dropout = Dropout::with_seed(0.5, Some(seed)).unwrap();
        let input = Tensor::from_vec(2, 3, vec![0.2, -0.3, 0.5, 1.0, -1.2, 0.7]).unwrap();
        let mask = mask_from_seed(0.5, seed, 2, 3);
        let expected = input.hadamard(&mask).unwrap();
        let output = dropout.forward(&input).unwrap();
        assert_eq!(output, expected);

        let grad_out = Tensor::from_vec(2, 3, vec![0.1, 0.2, -0.5, 0.3, -0.7, 0.9]).unwrap();
        let expected_grad = grad_out.hadamard(&mask).unwrap();
        let grad_in = dropout.backward(&input, &grad_out).unwrap();
        assert_eq!(grad_in, expected_grad);
    }

    #[test]
    fn dropout_eval_mode_passthrough() {
        let mut dropout = Dropout::with_seed(0.3, Some(7)).unwrap();
        dropout.eval();
        let input = Tensor::from_vec(1, 4, vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        let output = dropout.forward(&input).unwrap();
        assert_eq!(output, input);

        let grad = Tensor::from_vec(1, 4, vec![1.0, -1.0, 0.5, -0.5]).unwrap();
        let grad_in = dropout.backward(&input, &grad).unwrap();
        assert_eq!(grad_in, grad);
    }
}
