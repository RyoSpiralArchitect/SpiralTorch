use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
use st_tensor::pure::{
    topos::{OpenCartesianTopos, RewriteMonad},
    LanguageWaveEncoder,
};

/// Hyperbolic feature gate that mixes LanguageWaveEncoder spectra with module tensors.
#[derive(Debug)]
pub struct WaveGate {
    gate: Parameter,
    bias: Parameter,
    topos: OpenCartesianTopos,
    encoder: LanguageWaveEncoder,
}

impl WaveGate {
    /// Creates a wave gate with deterministic small parameters and an inferred topos.
    pub fn new(
        name: impl Into<String>,
        features: usize,
        curvature: f32,
        temperature: f32,
    ) -> PureResult<Self> {
        if features == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: 1,
                cols: features,
            });
        }
        let encoder = LanguageWaveEncoder::new(curvature, temperature)?;
        let depth = features.saturating_mul(64).max(64);
        let volume = features.saturating_mul(1024).max(1024);
        let topos = OpenCartesianTopos::new(curvature, 1e-6, 1e4, depth, volume)?;
        Self::with_topos(name, features, encoder, topos)
    }

    /// Builds a wave gate with explicit encoder/topos wiring.
    pub fn with_topos(
        name: impl Into<String>,
        features: usize,
        encoder: LanguageWaveEncoder,
        topos: OpenCartesianTopos,
    ) -> PureResult<Self> {
        if features == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: 1,
                cols: features,
            });
        }
        if (encoder.curvature() - topos.curvature()).abs() > 1e-6 {
            return Err(TensorError::CurvatureMismatch {
                expected: topos.curvature(),
                got: encoder.curvature(),
            });
        }
        let name = name.into();
        let gate = Tensor::from_fn(1, features, |_r, c| ((c as f32 + 1.0) * 0.01).sin() * 0.1)?;
        let bias = Tensor::zeros(1, features)?;
        Ok(Self {
            gate: Parameter::new(format!("{name}::gate"), gate),
            bias: Parameter::new(format!("{name}::bias"), bias),
            topos,
            encoder,
        })
    }

    /// Returns the open-cartesian guard used by the gate.
    pub fn topos(&self) -> &OpenCartesianTopos {
        &self.topos
    }

    /// Returns the internal encoder used for text-driven updates.
    pub fn encoder(&self) -> &LanguageWaveEncoder {
        &self.encoder
    }

    /// Returns an immutable reference to the gate parameter.
    pub fn gate(&self) -> &Parameter {
        &self.gate
    }

    /// Returns an immutable reference to the bias parameter.
    pub fn bias(&self) -> &Parameter {
        &self.bias
    }

    fn gate_len(&self) -> usize {
        self.gate.value().shape().1
    }
}

impl Module for WaveGate {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if cols != self.gate_len() {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: self.gate.value().shape(),
            });
        }
        self.topos.guard_tensor("wave_gate_forward_in", input)?;
        let mut gate = self.gate.value().clone();
        let monad = RewriteMonad::new(&self.topos);
        monad.rewrite_tensor("wave_gate_gate_rewrite", &mut gate)?;
        let gate_data = gate.data();
        let bias_data = self.bias.value().data();
        let input_buf = input.data();
        let mut out = Tensor::zeros(rows, cols)?;
        {
            let out_buf = out.data_mut();
            for r in 0..rows {
                let offset = r * cols;
                for c in 0..cols {
                    let value = input_buf[offset + c] * gate_data[c] + bias_data[c];
                    out_buf[offset + c] = value;
                }
            }
        }
        monad.rewrite_tensor("wave_gate_forward_out", &mut out)?;
        let projected = out.project_to_poincare(self.topos.curvature())?;
        self.topos
            .guard_tensor("wave_gate_forward_projected", &projected)?;
        Ok(projected)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if grad_output.shape() != (rows, cols) {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (rows, cols),
            });
        }
        let gate = self.gate.value().clone();
        let gate_data = gate.data();
        let mut grad_gate = Tensor::zeros(1, cols)?;
        let mut grad_bias = Tensor::zeros(1, cols)?;
        let mut grad_input = Tensor::zeros(rows, cols)?;
        {
            let grad_input_buf = grad_input.data_mut();
            let grad_gate_buf = grad_gate.data_mut();
            let grad_bias_buf = grad_bias.data_mut();
            let input_buf = input.data();
            let grad_out_buf = grad_output.data();
            for r in 0..rows {
                let offset = r * cols;
                for c in 0..cols {
                    let go = grad_out_buf[offset + c];
                    grad_input_buf[offset + c] = go * gate_data[c];
                    grad_gate_buf[c] += go * input_buf[offset + c];
                    grad_bias_buf[c] += go;
                }
            }
        }
        let monad = RewriteMonad::new(&self.topos);
        monad.rewrite_tensor("wave_gate_grad_gate", &mut grad_gate)?;
        monad.rewrite_tensor("wave_gate_grad_bias", &mut grad_bias)?;
        self.gate.accumulate_euclidean(&grad_gate)?;
        self.bias.accumulate_euclidean(&grad_bias)?;
        self.topos
            .guard_tensor("wave_gate_backward_out", &grad_input)?;
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.gate)?;
        visitor(&self.bias)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.gate)?;
        visitor(&mut self.bias)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wave_gate_forwards_and_backwards() {
        let mut gate = WaveGate::new("wg", 4, -1.0, 0.5).unwrap();
        let input =
            Tensor::from_vec(2, 4, vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]).unwrap();
        let output = gate.forward(&input).unwrap();
        assert_eq!(output.shape(), input.shape());
        let grad_out =
            Tensor::from_vec(2, 4, vec![0.2, -0.1, 0.05, -0.3, 0.4, -0.2, 0.1, -0.05]).unwrap();
        let grad_in = gate.backward(&input, &grad_out).unwrap();
        assert_eq!(grad_in.shape(), input.shape());
        let before = gate.gate().value().clone();
        gate.apply_step(0.01).unwrap();
        let after = gate.gate().value();
        assert_ne!(before, *after);
    }

    #[test]
    fn wave_gate_attaches_hypergrad() {
        let mut gate = WaveGate::new("wg", 8, -0.75, 0.9).unwrap();
        let topos = gate.topos().clone();
        gate.attach_hypergrad_with_topos(-0.75, 0.04, topos)
            .unwrap();
        let encoder = gate.encoder().clone();
        gate
            .visit_parameters_mut(&mut |param| param.absorb_text(&encoder, "wave"))
            .unwrap();
        gate.apply_step(0.01).unwrap();
        assert!(gate.gate().value().squared_l2_norm() > 0.0);
    }
}
