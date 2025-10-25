// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// ============================================================================
//  SpiralReality Proprietary
//  Copyright (c) 2025 SpiralReality. All Rights Reserved.
//
//  NOTICE: This file contains confidential and proprietary information of
//  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
//  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
//  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
//
//  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
//  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
//  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
//  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
//  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
//  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
//  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ============================================================================

use crate::gnn::RoundtableBandSignal;
use crate::schedule::GradientBands;
use st_core::backend::device_caps::DeviceCaps;
#[cfg(feature = "psychoid")]
use st_core::telemetry::psychoid::PsychoidSample;
use st_tensor::{
    topos::OpenCartesianTopos, AmegaHypergrad, AmegaRealgrad, ComplexTensor, LanguageWaveEncoder,
    PackedB, PureResult, Tensor, TensorError, Tile,
};
use std::cell::RefCell;
use std::collections::HashMap;

/// Trainable parameter that can either rely on the hypergrad tape or fall back
/// to standard Euclidean accumulation.
pub struct Parameter {
    name: String,
    value: Tensor,
    gradient: Option<Tensor>,
    hypergrad: Option<AmegaHypergrad>,
    realgrad: Option<AmegaRealgrad>,
    packed_matmul: RefCell<Option<PackedB>>,
    packed_matmul_transpose: RefCell<Option<PackedB>>,
}

impl core::fmt::Debug for Parameter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (rows, cols) = self.value.shape();
        write!(
            f,
            "Parameter(name={},shape=({},{}),has_grad={},has_hypergrad={},has_realgrad={})",
            self.name,
            rows,
            cols,
            self.gradient.is_some(),
            self.hypergrad.is_some(),
            self.realgrad.is_some()
        )
    }
}

impl Parameter {
    /// Creates a new parameter with the provided tensor value.
    pub fn new(name: impl Into<String>, value: Tensor) -> Self {
        Self {
            name: name.into(),
            value,
            gradient: None,
            hypergrad: None,
            realgrad: None,
            packed_matmul: RefCell::new(None),
            packed_matmul_transpose: RefCell::new(None),
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
        self.invalidate_matmul_pack();
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

    /// Attaches an Euclidean realgrad tape to the parameter.
    pub fn attach_realgrad(&mut self, learning_rate: f32) -> PureResult<()> {
        let (rows, cols) = self.value.shape();
        let tape = AmegaRealgrad::new(learning_rate, rows, cols)?;
        self.realgrad = Some(tape);
        if self.hypergrad.is_none() {
            self.gradient = None;
        }
        Ok(())
    }

    /// Provides direct access to the realgrad tape when attached.
    pub fn realgrad(&self) -> Option<&AmegaRealgrad> {
        self.realgrad.as_ref()
    }

    /// Provides mutable access to the realgrad tape when attached.
    pub fn realgrad_mut(&mut self) -> Option<&mut AmegaRealgrad> {
        self.realgrad.as_mut()
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
        }
        if let Some(tape) = self.realgrad.as_mut() {
            tape.accumulate_wave(update)?;
        }
        if self.hypergrad.is_none() && self.realgrad.is_none() {
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
            Ok(())
        }
        .and_then(|_| {
            if let Some(tape) = self.realgrad.as_mut() {
                tape.accumulate_complex_wave(wave)?;
            }
            if self.hypergrad.is_none() && self.realgrad.is_none() {
                match self.gradient.as_mut() {
                    Some(existing) => existing.add_scaled(&tensor, 1.0)?,
                    None => {
                        self.gradient = Some(tensor);
                    }
                }
            }
            Ok(())
        })
    }

    /// Absorbs free-form text directly into the parameter's accumulator by
    /// delegating to the hypergrad tape or caching the encoded tensor.
    pub fn absorb_text(&mut self, encoder: &LanguageWaveEncoder, text: &str) -> PureResult<()> {
        let tensor = encoder.encode_z_space(text)?;
        self.assert_shape(&tensor)?;
        if let Some(tape) = self.hypergrad.as_mut() {
            tape.absorb_text(encoder, text)
        } else {
            Ok(())
        }
        .and_then(|_| {
            if let Some(tape) = self.realgrad.as_mut() {
                tape.absorb_text(encoder, text)?;
            }
            if self.hypergrad.is_none() && self.realgrad.is_none() {
                match self.gradient.as_mut() {
                    Some(existing) => existing.add_scaled(&tensor, 1.0)?,
                    None => {
                        self.gradient = Some(tensor);
                    }
                }
            }
            Ok(())
        })
    }

    /// Clears the cached gradient or resets the hypergrad tape accumulator.
    pub fn zero_gradient(&mut self) {
        if let Some(tape) = self.hypergrad.as_mut() {
            tape.reset();
        }
        if let Some(tape) = self.realgrad.as_mut() {
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
        let mut applied = false;
        if let Some(tape) = self.hypergrad.as_mut() {
            tape.apply(&mut self.value)?;
            applied = true;
        }
        if let Some(tape) = self.realgrad.as_mut() {
            tape.apply(&mut self.value)?;
            applied = true;
        }
        if !applied {
            if let Some(grad) = self.gradient.as_mut() {
                self.value.add_scaled(grad, -fallback_lr)?;
                for value in grad.data_mut() {
                    *value = 0.0;
                }
            }
        }
        self.invalidate_matmul_pack();
        Ok(())
    }

    /// Scales any accumulated gradient or hypergradient buffers by the provided factor.
    pub fn scale_accumulators(&mut self, factor: f32) {
        if !factor.is_finite() {
            return;
        }
        if let Some(tape) = self.hypergrad.as_mut() {
            for grad in tape.gradient_mut() {
                *grad *= factor;
            }
        }
        if let Some(tape) = self.realgrad.as_mut() {
            for grad in tape.gradient_mut() {
                *grad *= factor;
            }
        }
        if let Some(grad) = self.gradient.as_mut() {
            for value in grad.data_mut() {
                *value *= factor;
            }
        }
    }

    /// Returns the squared L2 norm of any accumulated gradients.
    pub fn accumulators_norm_sq(&self) -> f64 {
        let mut total = 0.0;
        if let Some(tape) = self.hypergrad.as_ref() {
            total += tape
                .gradient()
                .iter()
                .map(|&value| {
                    let v = value as f64;
                    v * v
                })
                .sum::<f64>();
        }
        if let Some(tape) = self.realgrad.as_ref() {
            total += tape
                .gradient()
                .iter()
                .map(|&value| {
                    let v = value as f64;
                    v * v
                })
                .sum::<f64>();
        }
        if self.hypergrad.is_none() && self.realgrad.is_none() {
            if let Some(grad) = self.gradient.as_ref() {
                total += grad
                    .data()
                    .iter()
                    .map(|&value| {
                        let v = value as f64;
                        v * v
                    })
                    .sum::<f64>();
            }
        }
        total
    }

    /// Scales the learning rate inside the attached hypergrad tape, if present.
    pub fn scale_learning_rate(&mut self, factor: f32) {
        if let Some(tape) = self.hypergrad.as_mut() {
            tape.scale_learning_rate(factor);
        }
        if let Some(tape) = self.realgrad.as_mut() {
            tape.scale_learning_rate(factor);
        }
    }

    /// Ensures a prepacked representation of the parameter is available for matmul.
    pub fn ensure_matmul_pack(&self) -> PureResult<PackedB> {
        if let Some(existing) = self.packed_matmul.borrow().clone() {
            return Ok(existing);
        }
        let pack = PackedB::from_tensor(self.value(), Tile::col_major())?;
        *self.packed_matmul.borrow_mut() = Some(pack.clone());
        Ok(pack)
    }

    /// Ensures a prepacked representation of the parameter transpose is available for matmul.
    pub fn ensure_matmul_transpose_pack(&self) -> PureResult<PackedB> {
        if let Some(existing) = self.packed_matmul_transpose.borrow().clone() {
            return Ok(existing);
        }
        let pack = PackedB::from_tensor_transpose(self.value(), Tile::col_major())?;
        *self.packed_matmul_transpose.borrow_mut() = Some(pack.clone());
        Ok(pack)
    }

    fn invalidate_matmul_pack(&self) {
        self.packed_matmul.borrow_mut().take();
        self.packed_matmul_transpose.borrow_mut().take();
    }

    /// Replaces the parameter value with the provided tensor.
    pub fn load_value(&mut self, value: &Tensor) -> PureResult<()> {
        self.assert_shape(value)?;
        *self.value_mut() = value.clone();
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

    /// Applies the latest roundtable band signal before a backward pass.
    fn apply_roundtable_band(&mut self, _signal: &RoundtableBandSignal) -> PureResult<()> {
        Ok(())
    }

    /// Clears any previously applied roundtable directives.
    fn clear_roundtable_band(&mut self) -> PureResult<()> {
        Ok(())
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

    /// Attaches a realgrad tape to every parameter.
    fn attach_realgrad(&mut self, learning_rate: f32) -> PureResult<()> {
        self.visit_parameters_mut(&mut |param| param.attach_realgrad(learning_rate))
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

    /// Optional hook that surfaces activation drift telemetry for ψ metering.
    ///
    /// Implementations may override this to provide a smoothed scalar that
    /// captures how far their activations drifted during the most recent step.
    /// Returning `None` indicates that the module does not contribute drift
    /// telemetry, allowing the ψ meter to fall back to zero for that component.
    fn psi_probe(&self) -> Option<f32> {
        None
    }

    #[cfg(feature = "psychoid")]
    fn psychoid_sample(&self, _input: &Tensor, _output: &Tensor) -> Option<PsychoidSample> {
        None
    }

    /// Allows modules to describe the device they expect to run on. The default
    /// implementation simply returns `None` which indicates the module is agnostic.
    fn preferred_device(&self) -> Option<DeviceCaps> {
        None
    }

    /// Captures a copy of every parameter tensor keyed by its canonical name.
    fn state_dict(&self) -> PureResult<HashMap<String, Tensor>> {
        let mut state = HashMap::new();
        self.visit_parameters(&mut |param| {
            state.insert(param.name().to_string(), param.value().clone());
            Ok(())
        })?;
        Ok(state)
    }

    /// Restores parameters from a state dictionary produced by [`Module::state_dict`].
    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>) -> PureResult<()> {
        self.visit_parameters_mut(&mut |param| {
            let Some(value) = state.get(param.name()) else {
                return Err(TensorError::MissingParameter {
                    name: param.name().to_string(),
                });
            };
            param.load_value(value)
        })
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

    #[test]
    fn parameter_streams_wave_through_realgrad() {
        let encoder = LanguageWaveEncoder::new(-0.9, 0.7).unwrap();
        let wave = encoder.encode_wave("realgrad").unwrap();
        let tensor = wave.to_tensor().unwrap();
        let mut param = Parameter::new("gate", Tensor::zeros(1, tensor.shape().1).unwrap());
        param.attach_realgrad(0.02).unwrap();
        param.accumulate_complex_wave(&wave).unwrap();
        param.absorb_text(&encoder, "realgrad").unwrap();
        assert!(param.realgrad().is_some());
        param.apply_step(0.05).unwrap();
    }
}
