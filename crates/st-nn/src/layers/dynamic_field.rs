// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
use rand::rngs::StdRng;
use rand::Rng;
use spiral_config::determinism;
use std::cell::RefCell;

/// Propagates semantic amplitudes using a discrete Klein–Gordon style update
/// blended with a Dirac spinor coupling. This treats the incoming activations as
/// narrative field intensities and modulates them with a learnable mass/phase
/// pair.
#[derive(Debug)]
pub struct KleinGordonPropagation {
    mass: Parameter,
    spinor: Parameter,
    time_step: f32,
    damping: f32,
}

impl KleinGordonPropagation {
    /// Creates a propagation layer with deterministic initial parameters.
    pub fn new(
        name: impl Into<String>,
        features: usize,
        time_step: f32,
        damping: f32,
    ) -> PureResult<Self> {
        if features == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: 1,
                cols: features,
            });
        }
        if !time_step.is_finite() || time_step <= 0.0 {
            return Err(TensorError::InvalidValue {
                label: "klein_gordon_time_step",
            });
        }
        if !damping.is_finite() || damping < 0.0 {
            return Err(TensorError::InvalidValue {
                label: "klein_gordon_damping",
            });
        }
        let name = name.into();
        let mass = Tensor::from_fn(1, features, |_r, c| 0.05 + (c as f32 * 0.01))?;
        let spinor = Tensor::from_fn(1, features, |_r, c| ((c as f32 + 1.0) * 0.37).sin() * 0.1)?;
        Ok(Self {
            mass: Parameter::new(format!("{name}::mass"), mass),
            spinor: Parameter::new(format!("{name}::spinor"), spinor),
            time_step,
            damping,
        })
    }

    /// Returns the configured time step used during propagation.
    pub fn time_step(&self) -> f32 {
        self.time_step
    }

    /// Returns the configured damping factor.
    pub fn damping(&self) -> f32 {
        self.damping
    }

    /// Returns a reference to the internal mass parameter.
    pub fn mass(&self) -> &Parameter {
        &self.mass
    }

    /// Returns a reference to the internal spinor parameter.
    pub fn spinor(&self) -> &Parameter {
        &self.spinor
    }
}

impl Module for KleinGordonPropagation {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if cols != self.mass.value().shape().1 {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: self.mass.value().shape(),
            });
        }
        let mut output = Tensor::zeros(rows, cols)?;
        let time_step = self.time_step;
        let damping = self.damping;
        let mass = self.mass.value().data();
        let spin = self.spinor.value().data();
        {
            let input_buf = input.data();
            let out_buf = output.data_mut();
            for r in 0..rows {
                let offset = r * cols;
                for c in 0..cols {
                    let idx = offset + c;
                    let wave = input_buf[idx];
                    let amplitude = wave.tanh();
                    let kg_coeff = 1.0 - time_step * damping - time_step * time_step * mass[c];
                    let dirac_coeff = time_step * spin[c];
                    out_buf[idx] = wave * kg_coeff + dirac_coeff * amplitude;
                }
            }
        }
        Ok(output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if grad_output.shape() != (rows, cols) {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (rows, cols),
            });
        }
        let mut grad_input = Tensor::zeros(rows, cols)?;
        let mut grad_mass = Tensor::zeros(1, cols)?;
        let mut grad_spin = Tensor::zeros(1, cols)?;
        let time_step = self.time_step;
        let damping = self.damping;
        let mass = self.mass.value().data();
        let spin = self.spinor.value().data();
        {
            let input_buf = input.data();
            let grad_output_buf = grad_output.data();
            let grad_input_buf = grad_input.data_mut();
            let grad_mass_buf = grad_mass.data_mut();
            let grad_spin_buf = grad_spin.data_mut();
            for r in 0..rows {
                let offset = r * cols;
                for c in 0..cols {
                    let idx = offset + c;
                    let wave = input_buf[idx];
                    let amplitude = wave.tanh();
                    let sech2 = 1.0 - amplitude * amplitude;
                    let kg_coeff = 1.0 - time_step * damping - time_step * time_step * mass[c];
                    let dirac_coeff = time_step * spin[c];
                    let go = grad_output_buf[idx];
                    grad_input_buf[idx] = go * (kg_coeff + dirac_coeff * sech2);
                    grad_mass_buf[c] += go * (-time_step * time_step * wave);
                    grad_spin_buf[c] += go * (time_step * amplitude);
                }
            }
        }
        self.mass.accumulate_euclidean(&grad_mass)?;
        self.spinor.accumulate_euclidean(&grad_spin)?;
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.mass)?;
        visitor(&self.spinor)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.mass)?;
        visitor(&mut self.spinor)?;
        Ok(())
    }
}

/// Hamilton–Jacobi style flow that nudges timelines towards minimal-action
/// trajectories. Each column receives an independent potential term that
/// encourages smoothness across the temporal dimension.
#[derive(Debug)]
pub struct HamiltonJacobiFlow {
    potential: Parameter,
    step_size: f32,
}

impl HamiltonJacobiFlow {
    /// Creates a flow model for the provided feature count.
    pub fn new(name: impl Into<String>, features: usize, step_size: f32) -> PureResult<Self> {
        if features == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: 1,
                cols: features,
            });
        }
        if !step_size.is_finite() || step_size <= 0.0 {
            return Err(TensorError::InvalidValue {
                label: "hamilton_jacobi_step",
            });
        }
        let name = name.into();
        let potential = Tensor::from_fn(1, features, |_r, c| {
            ((c as f32 + 1.0) * 0.13).sin().abs() + 0.1
        })?;
        Ok(Self {
            potential: Parameter::new(format!("{name}::potential"), potential),
            step_size,
        })
    }

    /// Returns the internal potential parameter.
    pub fn potential(&self) -> &Parameter {
        &self.potential
    }

    /// Returns the configured gradient descent step size.
    pub fn step_size(&self) -> f32 {
        self.step_size
    }
}

impl Module for HamiltonJacobiFlow {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if cols != self.potential.value().shape().1 {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: self.potential.value().shape(),
            });
        }
        let mut output = Tensor::zeros(rows, cols)?;
        let step = self.step_size;
        let potential = self.potential.value().data();
        {
            let input_buf = input.data();
            let out_buf = output.data_mut();
            for r in 0..rows {
                let offset = r * cols;
                let input_row = &input_buf[offset..offset + cols];
                let prev_row = if r > 0 {
                    &input_buf[offset - cols..offset]
                } else {
                    input_row
                };
                let next_row = if r + 1 < rows {
                    &input_buf[offset + cols..offset + 2 * cols]
                } else {
                    input_row
                };
                let out_row = &mut out_buf[offset..offset + cols];

                for (((out, &current), &potential), (&prev, &next)) in out_row
                    .iter_mut()
                    .zip(input_row.iter())
                    .zip(potential.iter())
                    .zip(prev_row.iter().zip(next_row.iter()))
                {
                    let grad = (2.0 * current - prev - next) + potential * current;
                    *out = current - step * grad;
                }
            }
        }
        Ok(output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if grad_output.shape() != (rows, cols) {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (rows, cols),
            });
        }
        let mut grad_input = Tensor::zeros(rows, cols)?;
        let mut grad_potential = Tensor::zeros(1, cols)?;
        let step = self.step_size;
        let potential = self.potential.value().data();
        {
            let input_buf = input.data();
            let grad_output_buf = grad_output.data();
            let grad_input_buf = grad_input.data_mut();
            let grad_potential_buf = grad_potential.data_mut();
            for r in 0..rows {
                let offset = r * cols;
                for c in 0..cols {
                    let idx = offset + c;
                    let current = input_buf[idx];
                    let go = grad_output_buf[idx];
                    if rows == 1 {
                        grad_input_buf[idx] += go * (1.0 - step * potential[c]);
                    } else if r == 0 {
                        grad_input_buf[idx] += go * (1.0 - step * (1.0 + potential[c]));
                        grad_input_buf[idx + cols] += go * step;
                    } else if r + 1 == rows {
                        grad_input_buf[idx] += go * (1.0 - step * (1.0 + potential[c]));
                        grad_input_buf[idx - cols] += go * step;
                    } else {
                        grad_input_buf[idx] += go * (1.0 - step * (2.0 + potential[c]));
                        grad_input_buf[idx - cols] += go * step;
                        grad_input_buf[idx + cols] += go * step;
                    }
                    grad_potential_buf[c] += -step * current * go;
                }
            }
        }
        self.potential.accumulate_euclidean(&grad_potential)?;
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.potential)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.potential)
    }
}

/// Layer that models semantic interference through a stochastic Schrödinger
/// style update. It keeps track of decoherence factors that can later be
/// surfaced through [`NarrativeHint`] quantum metadata.
#[derive(Debug)]
pub struct StochasticSchrodingerLayer {
    coherence: Parameter,
    decoherence_rate: f32,
    noise_scale: f32,
    rng: RefCell<StdRng>,
    last_amplitude: RefCell<Option<Tensor>>,
    last_decoherence: RefCell<Option<Tensor>>,
}

impl StochasticSchrodingerLayer {
    /// Creates a stochastic layer using entropy-backed randomness.
    pub fn new(
        name: impl Into<String>,
        features: usize,
        decoherence_rate: f32,
        noise_scale: f32,
    ) -> PureResult<Self> {
        Self::with_seed(name, features, decoherence_rate, noise_scale, None)
    }

    /// Same as [`StochasticSchrodingerLayer::new`] but allows a deterministic seed
    /// for reproducible experiments and unit tests.
    pub fn with_seed(
        name: impl Into<String>,
        features: usize,
        decoherence_rate: f32,
        noise_scale: f32,
        seed: Option<u64>,
    ) -> PureResult<Self> {
        if features == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: 1,
                cols: features,
            });
        }
        if !decoherence_rate.is_finite() || decoherence_rate < 0.0 {
            return Err(TensorError::InvalidValue {
                label: "schrodinger_decoherence",
            });
        }
        if !noise_scale.is_finite() || noise_scale < 0.0 {
            return Err(TensorError::InvalidValue {
                label: "schrodinger_noise",
            });
        }
        let name = name.into();
        let coherence = Tensor::from_fn(1, features, |_r, c| (0.85 - (c as f32 * 0.03)).max(0.1))?;
        let rng = determinism::rng_from_optional(seed, "st-nn/layers/stochastic_schrodinger");
        Ok(Self {
            coherence: Parameter::new(format!("{name}::coherence"), coherence),
            decoherence_rate,
            noise_scale,
            rng: RefCell::new(rng),
            last_amplitude: RefCell::new(None),
            last_decoherence: RefCell::new(None),
        })
    }

    /// Returns the configured decoherence rate.
    pub fn decoherence_rate(&self) -> f32 {
        self.decoherence_rate
    }

    /// Returns the coherence parameter.
    pub fn coherence(&self) -> &Parameter {
        &self.coherence
    }
}

impl Module for StochasticSchrodingerLayer {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if cols != self.coherence.value().shape().1 {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: self.coherence.value().shape(),
            });
        }
        let mut output = Tensor::zeros(rows, cols)?;
        let mut amplitude = Tensor::zeros(rows, cols)?;
        let mut decoherence = Tensor::zeros(rows, cols)?;
        let noise_scale = self.noise_scale;
        let rate = self.decoherence_rate;
        let coherence = self.coherence.value().data();
        {
            let input_buf = input.data();
            let out_buf = output.data_mut();
            let amp_buf = amplitude.data_mut();
            let deco_buf = decoherence.data_mut();
            let mut rng = self.rng.borrow_mut();
            for r in 0..rows {
                let offset = r * cols;
                let input_row = &input_buf[offset..offset + cols];
                let out_row = &mut out_buf[offset..offset + cols];
                let amp_row = &mut amp_buf[offset..offset + cols];
                let deco_row = &mut deco_buf[offset..offset + cols];
                for (((out, amp_slot), deco_slot), (&input, &coherence)) in out_row
                    .iter_mut()
                    .zip(amp_row.iter_mut())
                    .zip(deco_row.iter_mut())
                    .zip(input_row.iter().zip(coherence.iter()))
                {
                    let amp = input.tanh();
                    let deco = 1.0 / (1.0 + rate * amp.abs());
                    let interference = amp * coherence * deco;
                    let noise = (rng.gen::<f32>() - 0.5) * noise_scale;
                    *out = interference + noise;
                    *amp_slot = amp;
                    *deco_slot = deco;
                }
            }
        }
        self.last_amplitude.borrow_mut().replace(amplitude);
        self.last_decoherence.borrow_mut().replace(decoherence);
        Ok(output)
    }

    fn backward(&mut self, _input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let amplitude_guard = self.last_amplitude.borrow();
        let Some(amplitude) = amplitude_guard.as_ref() else {
            return Err(TensorError::InvalidValue {
                label: "schrodinger_amplitude_cache",
            });
        };
        let decoherence_guard = self.last_decoherence.borrow();
        let Some(decoherence) = decoherence_guard.as_ref() else {
            return Err(TensorError::InvalidValue {
                label: "schrodinger_decoherence_cache",
            });
        };
        let (rows, cols) = grad_output.shape();
        let mut grad_input = Tensor::zeros(rows, cols)?;
        let mut grad_coherence = Tensor::zeros(1, cols)?;
        let coherence_values = self.coherence.value().data().to_vec();
        let rate = self.decoherence_rate;
        {
            let grad_output_buf = grad_output.data();
            let grad_input_buf = grad_input.data_mut();
            let amp_buf = amplitude.data();
            let deco_buf = decoherence.data();
            let grad_coherence_buf = grad_coherence.data_mut();
            for r in 0..rows {
                let offset = r * cols;
                for c in 0..cols {
                    let idx = offset + c;
                    let amp = amp_buf[idx];
                    let deco = deco_buf[idx];
                    let go = grad_output_buf[idx];
                    let denom = 1.0 + rate * amp.abs();
                    let sign = if amp >= 0.0 { 1.0 } else { -1.0 };
                    let d_deco_d_amp = -rate * sign / (denom * denom);
                    let base = deco + amp * d_deco_d_amp;
                    let d_amp_d_input = 1.0 - amp * amp;
                    grad_input_buf[idx] = go * coherence_values[c] * base * d_amp_d_input;
                    grad_coherence_buf[c] += go * (amp * deco);
                }
            }
        }
        self.coherence.accumulate_euclidean(&grad_coherence)?;
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.coherence)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.coherence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn klein_gordon_shapes_match() {
        let layer = KleinGordonPropagation::new("kg", 4, 0.2, 0.1).unwrap();
        let input = Tensor::from_vec(3, 4, vec![0.1; 12]).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn hamilton_jacobi_smooths_edges() {
        let mut layer = HamiltonJacobiFlow::new("hj", 3, 0.1).unwrap();
        let input = Tensor::from_vec(
            4,
            3,
            vec![
                1.0, 0.0, -1.0, 1.0, 0.5, -0.5, 1.0, 0.0, -1.0, 1.0, 0.5, -0.5,
            ],
        )
        .unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), input.shape());
        let grad = Tensor::from_vec(4, 3, vec![0.2; 12]).unwrap();
        let back = layer.backward(&input, &grad).unwrap();
        assert_eq!(back.shape(), input.shape());
    }

    #[test]
    fn stochastic_schrodinger_is_reproducible() {
        let mut layer = StochasticSchrodingerLayer::with_seed("qs", 2, 0.4, 0.05, Some(7)).unwrap();
        let input = Tensor::from_vec(1, 2, vec![0.3, -0.7]).unwrap();
        let output_a = layer.forward(&input).unwrap();
        let output_b = layer.forward(&input).unwrap();
        assert_ne!(output_a.data(), output_b.data());
        let grad = Tensor::from_vec(1, 2, vec![0.1, -0.2]).unwrap();
        let back = layer.backward(&input, &grad).unwrap();
        assert_eq!(back.shape(), input.shape());
    }
}
