use crate::schedule::GradientBands;
use st_core::backend::device_caps::DeviceCaps;
use st_tensor::pure::{
    topos::OpenCartesianTopos, AmegaHypergrad, ComplexTensor, LanguageWaveEncoder, PureResult,
    Tensor, TensorError,
};

/// Trainable parameter that can either rely on the hypergrad tape or fall back
/// to standard Euclidean accumulation.
#[derive(Debug)]
pub struct Parameter {
    name: String,
    value: Tensor,
    gradient: Option<Tensor>,
    hypergrad: Option<AmegaHypergrad>,
}

impl Parameter {
    /// Creates a new parameter with the provided tensor value.
    pub fn new(name: impl Into<String>, value: Tensor) -> Self {
        Self {
            name: name.into(),
            value,
            gradient: None,
            hypergrad: None,
        }
    }

    /// Returns the identifier assigned to the parameter.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Overrides the parameter name.
    pub fn rename(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    /// Provides an immutable view into the underlying tensor value.
    pub fn value(&self) -> &Tensor {
        &self.value
    }

    /// Provides a mutable view into the underlying tensor value.
    pub fn value_mut(&mut self) -> &mut Tensor {
        &mut self.value
    }

    /// Returns the currently cached Euclidean gradient when no hypergrad tape is active.
    pub fn gradient(&self) -> Option<&Tensor> {
        self.gradient.as_ref()
    }

    /// Attaches a hypergrad tape to the parameter.
    pub fn attach_hypergrad(&mut self, curvature: f32, learning_rate: f32) -> PureResult<()> {
        let (rows, cols) = self.value.shape();
        let tape = AmegaHypergrad::new(curvature, learning_rate, rows, cols)?;
        self.hypergrad = Some(tape);
        self.gradient = None;
        Ok(())
    }

    /// Attaches a hypergrad tape using a caller-supplied topos.
    pub fn attach_hypergrad_with_topos(
        &mut self,
        curvature: f32,
        learning_rate: f32,
        topos: OpenCartesianTopos,
    ) -> PureResult<()> {
        let (rows, cols) = self.value.shape();
        let tape = AmegaHypergrad::with_topos(curvature, learning_rate, rows, cols, topos)?;
        self.hypergrad = Some(tape);
        self.gradient = None;
        Ok(())
    }

    /// Provides direct access to the hypergrad tape when attached.
    pub fn hypergrad(&self) -> Option<&AmegaHypergrad> {
        self.hypergrad.as_ref()
    }

    /// Provides mutable access to the hypergrad tape when attached.
    pub fn hypergrad_mut(&mut self) -> Option<&mut AmegaHypergrad> {
        self.hypergrad.as_mut()
    }

    fn assert_shape(&self, tensor: &Tensor) -> PureResult<()> {
        if self.value.shape() != tensor.shape() {
            return Err(TensorError::ShapeMismatch {
                left: self.value.shape(),
                right: tensor.shape(),
            });
        }
        Ok(())
    }

    /// Accumulates a Euclidean gradient update. When a hypergrad tape is
    /// attached the value is streamed through the tape, otherwise a local
    /// gradient buffer is maintained.
    pub fn accumulate_euclidean(&mut self, update: &Tensor) -> PureResult<()> {
        self.assert_shape(update)?;
        if let Some(tape) = self.hypergrad.as_mut() {
            tape.accumulate_wave(update)?;
        } else {
            match self.gradient.as_mut() {
                Some(existing) => existing.add_scaled(update, 1.0)?,
                None => {
                    self.gradient = Some(update.clone());
                }
            }
        }
        Ok(())
    }

    /// Streams a complex wave through the attached hypergrad tape or caches an
    /// Euclidean equivalent when no tape is present.
    pub fn accumulate_complex_wave(&mut self, wave: &ComplexTensor) -> PureResult<()> {
        let tensor = wave.to_tensor()?;
        self.assert_shape(&tensor)?;
        if let Some(tape) = self.hypergrad.as_mut() {
            tape.accumulate_complex_wave(wave)
        } else {
            match self.gradient.as_mut() {
                Some(existing) => existing.add_scaled(&tensor, 1.0)?,
                None => {
                    self.gradient = Some(tensor);
                }
            }
            Ok(())
        }
    }

    /// Absorbs free-form text directly into the parameter's accumulator by
    /// delegating to the hypergrad tape or caching the encoded tensor.
    pub fn absorb_text(&mut self, encoder: &LanguageWaveEncoder, text: &str) -> PureResult<()> {
        let tensor = encoder.encode_z_space(text)?;
        self.assert_shape(&tensor)?;
        if let Some(tape) = self.hypergrad.as_mut() {
            tape.absorb_text(encoder, text)
        } else {
            match self.gradient.as_mut() {
                Some(existing) => existing.add_scaled(&tensor, 1.0)?,
                None => {
                    self.gradient = Some(tensor);
                }
            }
            Ok(())
        }
    }

    /// Clears the cached gradient or resets the hypergrad tape accumulator.
    pub fn zero_gradient(&mut self) {
        if let Some(tape) = self.hypergrad.as_mut() {
            tape.reset();
        }
        if let Some(grad) = self.gradient.as_mut() {
            for value in grad.data_mut() {
                *value = 0.0;
            }
        }
    }

    /// Applies the accumulated update either via the hypergrad tape or by using
    /// the supplied fallback learning rate.
    pub fn apply_step(&mut self, fallback_lr: f32) -> PureResult<()> {
        if let Some(tape) = self.hypergrad.as_mut() {
            tape.apply(&mut self.value)?;
        } else if let Some(grad) = self.gradient.as_mut() {
            self.value.add_scaled(grad, -fallback_lr)?;
            for value in grad.data_mut() {
                *value = 0.0;
            }
        }
        Ok(())
    }
}

/// High-level module trait inspired by PyTorch's `nn.Module` but expressed in
/// pure Rust so it can be used from WebGPU, HIP, or CPU flows alike.
pub trait Module {
    /// Runs a forward pass.
    fn forward(&self, input: &Tensor) -> PureResult<Tensor>;

    /// Propagates a gradient backwards. Implementations should populate the
    /// relevant parameter accumulators before returning the gradient with
    /// respect to `input`.
    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor>;

    /// Visits immutable parameters.
    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()>;

    /// Visits mutable parameters.
    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()>;

    /// Propagates an Above/Here/Beneath gradient schedule through the module.
    fn backward_bands(&mut self, input: &Tensor, bands: &GradientBands) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        let mut total = Tensor::zeros(rows, cols)?;
        for grad in bands.iter() {
            if grad.squared_l2_norm() == 0.0 {
                continue;
            }
            let contribution = self.backward(input, grad)?;
            total.add_scaled(&contribution, 1.0)?;
        }
        Ok(total)
    }

    /// Attaches a hypergrad tape to every parameter.
    fn attach_hypergrad(&mut self, curvature: f32, learning_rate: f32) -> PureResult<()> {
        self.visit_parameters_mut(&mut |param| param.attach_hypergrad(curvature, learning_rate))
    }

    /// Attaches a hypergrad tape using a shared topos.
    fn attach_hypergrad_with_topos(
        &mut self,
        curvature: f32,
        learning_rate: f32,
        topos: OpenCartesianTopos,
    ) -> PureResult<()> {
        self.visit_parameters_mut(&mut |param| {
            param.attach_hypergrad_with_topos(curvature, learning_rate, topos.clone())
        })
    }

    /// Applies every parameter update.
    fn apply_step(&mut self, fallback_lr: f32) -> PureResult<()> {
        self.visit_parameters_mut(&mut |param| param.apply_step(fallback_lr))
    }

    /// Clears accumulators across every parameter.
    fn zero_accumulators(&mut self) -> PureResult<()> {
        self.visit_parameters_mut(&mut |param| {
            param.zero_gradient();
            Ok(())
        })
    }

    /// Allows modules to describe the device they expect to run on. The default
    /// implementation simply returns `None` which indicates the module is agnostic.
    fn preferred_device(&self) -> Option<DeviceCaps> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_absorbs_waves_without_hypergrad() {
        let encoder = LanguageWaveEncoder::new(-1.1, 0.6).unwrap();
        let wave = encoder.encode_wave("flux").unwrap();
        let tensor = wave.to_tensor().unwrap();
        let mut param = Parameter::new("gate", Tensor::zeros(1, tensor.shape().1).unwrap());
        param.accumulate_complex_wave(&wave).unwrap();
        assert!(param.gradient().is_some());
        param.absorb_text(&encoder, "flux").unwrap();
        assert!(param.gradient().unwrap().squared_l2_norm() > 0.0);
    }

    #[test]
    fn parameter_streams_wave_through_hypergrad() {
        let encoder = LanguageWaveEncoder::new(-0.95, 0.8).unwrap();
        let wave = encoder.encode_wave("spiral").unwrap();
        let tensor = wave.to_tensor().unwrap();
        let mut param = Parameter::new("gate", Tensor::zeros(1, tensor.shape().1).unwrap());
        param.attach_hypergrad(encoder.curvature(), 0.05).unwrap();
        param.accumulate_complex_wave(&wave).unwrap();
        param.absorb_text(&encoder, "spiral").unwrap();
        assert!(param.hypergrad().is_some());
        param.apply_step(0.01).unwrap();
    }
}
