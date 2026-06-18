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

use crate::execution::current_tensor_util_backend_for_values;
use crate::gnn::RoundtableBandSignal;
use crate::optim::LocalLearningRateAdapter;
use crate::schedule::GradientBands;
use st_core::backend::device_caps::DeviceCaps;
use st_core::ops::zspace_round::RoundtableBand;
use st_core::ops::zspace_round::SpectralFeatureSample;
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

    fn validate_fallback_lr(fallback_lr: f32) -> PureResult<()> {
        if fallback_lr <= 0.0 || !fallback_lr.is_finite() {
            return Err(TensorError::NonPositiveLearningRate { rate: fallback_lr });
        }
        Ok(())
    }

    fn validate_synchronized_accumulator(values: &[f32]) -> PureResult<()> {
        for value in values.iter().copied() {
            if !value.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "synchronized_accumulator",
                    value,
                });
            }
        }
        Ok(())
    }

    fn validate_scaled_accumulator(values: &[f32], factor: f32) -> PureResult<()> {
        for value in values.iter().copied() {
            let scaled = value * factor;
            if !scaled.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "scaled_accumulator",
                    value: scaled,
                });
            }
        }
        Ok(())
    }

    fn validate_fallback_update(
        value: &Tensor,
        gradient: &Tensor,
        fallback_lr: f32,
    ) -> PureResult<()> {
        if value.shape() != gradient.shape() {
            return Err(TensorError::ShapeMismatch {
                left: value.shape(),
                right: gradient.shape(),
            });
        }
        for (&weight, &grad) in value.data().iter().zip(gradient.data().iter()) {
            let delta = fallback_lr * grad;
            if !delta.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "parameter_delta",
                    value: delta,
                });
            }
            let next = weight - delta;
            if !next.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "parameter_update",
                    value: next,
                });
            }
        }
        Ok(())
    }

    fn synchronize_accumulator_buffer<F>(
        gradient: &mut [f32],
        synchronize: &mut F,
    ) -> PureResult<()>
    where
        F: FnMut(&mut [f32]) -> PureResult<()>,
    {
        let mut synchronized = gradient.to_vec();
        synchronize(&mut synchronized)?;
        Self::validate_synchronized_accumulator(&synchronized)?;
        gradient.copy_from_slice(&synchronized);
        Ok(())
    }

    /// Accumulates a Euclidean gradient update. When a hypergrad tape is
    /// attached the value is streamed through the tape, otherwise a local
    /// gradient buffer is maintained.
    pub fn accumulate_euclidean(&mut self, update: &Tensor) -> PureResult<()> {
        self.assert_shape(update)?;
        if let Some(tape) = self.hypergrad.as_mut() {
            let backend = current_tensor_util_backend_for_values(update.data().len());
            tape.accumulate_wave_with_backend(update, backend)?;
        }
        if let Some(tape) = self.realgrad.as_mut() {
            tape.accumulate_wave(update)?;
        }
        if self.hypergrad.is_none() && self.realgrad.is_none() {
            match self.gradient.as_mut() {
                Some(existing) => {
                    let backend = current_tensor_util_backend_for_values(existing.data().len());
                    existing.add_scaled_with_backend(update, 1.0, backend)?;
                }
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
                    Some(existing) => {
                        let backend = current_tensor_util_backend_for_values(existing.data().len());
                        existing.add_scaled_with_backend(&tensor, 1.0, backend)?;
                    }
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
                    Some(existing) => {
                        let backend = current_tensor_util_backend_for_values(existing.data().len());
                        existing.add_scaled_with_backend(&tensor, 1.0, backend)?;
                    }
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
        Self::validate_fallback_lr(fallback_lr)?;
        let mut applied = false;
        if let Some(tape) = self.hypergrad.as_mut() {
            let backend = current_tensor_util_backend_for_values(self.value.data().len());
            tape.apply_with_backend(&mut self.value, backend)?;
            applied = true;
        }
        if let Some(tape) = self.realgrad.as_mut() {
            let backend = current_tensor_util_backend_for_values(self.value.data().len());
            tape.apply_with_backend(&mut self.value, backend)?;
            applied = true;
        }
        if !applied {
            if let Some(grad) = self.gradient.as_mut() {
                Self::validate_fallback_update(&self.value, grad, fallback_lr)?;
                let backend = current_tensor_util_backend_for_values(self.value.data().len());
                self.value
                    .add_scaled_with_backend(grad, -fallback_lr, backend)?;
                for value in grad.data_mut() {
                    *value = 0.0;
                }
            }
        }
        self.invalidate_matmul_pack();
        Ok(())
    }

    /// Applies the accumulated update using the provided adapter to modulate the
    /// effective learning rate.
    pub fn apply_step_with_adapter(
        &mut self,
        fallback_lr: f32,
        adapter: Option<&mut dyn LocalLearningRateAdapter>,
    ) -> PureResult<()> {
        Self::validate_fallback_lr(fallback_lr)?;
        if let Some(adapter) = adapter {
            if let Some(view) = self.primary_gradient_view() {
                let hint = adapter.sheet_hint().max(1);
                if let Some(features) = SpectralFeatureSample::from_slice(view, hint) {
                    let raw = adapter.scale_factor(self.name(), &features);
                    if raw.is_finite() && raw > 0.0 && (raw - 1.0).abs() > f32::EPSILON {
                        self.scale_accumulators_with_backend_policy(raw)?;
                    }
                }
            }
        }
        self.apply_step(fallback_lr)
    }

    fn primary_gradient_view(&self) -> Option<&[f32]> {
        if let Some(tape) = self.hypergrad.as_ref() {
            let gradient = tape.gradient();
            if !gradient.is_empty() {
                return Some(gradient);
            }
        }
        if let Some(tape) = self.realgrad.as_ref() {
            let gradient = tape.gradient();
            if !gradient.is_empty() {
                return Some(gradient);
            }
        }
        self.gradient.as_ref().map(|tensor| tensor.data())
    }

    /// Scales any accumulated gradient or hypergradient buffers by the provided factor.
    pub fn scale_accumulators(&mut self, factor: f32) {
        if !factor.is_finite() {
            return;
        }
        if let Some(tape) = self.hypergrad.as_ref() {
            if Self::validate_scaled_accumulator(tape.gradient(), factor).is_err() {
                return;
            }
        }
        if let Some(tape) = self.realgrad.as_ref() {
            if Self::validate_scaled_accumulator(tape.gradient(), factor).is_err() {
                return;
            }
        }
        if let Some(grad) = self.gradient.as_ref() {
            if Self::validate_scaled_accumulator(grad.data(), factor).is_err() {
                return;
            }
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

    /// Scales accumulator buffers while routing tensor-backed gradients through backend policy.
    pub fn scale_accumulators_with_backend_policy(&mut self, factor: f32) -> PureResult<()> {
        if !factor.is_finite() {
            return Ok(());
        }
        if let Some(tape) = self.hypergrad.as_ref() {
            Self::validate_scaled_accumulator(tape.gradient(), factor)?;
        }
        if let Some(tape) = self.realgrad.as_ref() {
            Self::validate_scaled_accumulator(tape.gradient(), factor)?;
        }
        if let Some(grad) = self.gradient.as_ref() {
            Self::validate_scaled_accumulator(grad.data(), factor)?;
        }
        if let Some(tape) = self.hypergrad.as_mut() {
            let backend = current_tensor_util_backend_for_values(tape.gradient().len());
            tape.scale_gradient_with_backend(factor, backend)?;
        }
        if let Some(tape) = self.realgrad.as_mut() {
            let backend = current_tensor_util_backend_for_values(tape.gradient().len());
            tape.scale_gradient_with_backend(factor, backend)?;
        }
        if let Some(grad) = self.gradient.as_mut() {
            let backend = current_tensor_util_backend_for_values(grad.data().len());
            *grad = grad.scale_with_backend(factor, backend)?;
        }
        Ok(())
    }

    /// Exposes accumulated gradient buffers to a synchronization callback.
    ///
    /// Hypergrad and realgrad tapes are synchronized independently when present.
    /// The Euclidean fallback buffer is synchronized only when no tape is active,
    /// matching the update path used by [`apply_step`].
    pub fn synchronize_accumulators_with<F>(&mut self, mut synchronize: F) -> PureResult<usize>
    where
        F: FnMut(&mut [f32]) -> PureResult<()>,
    {
        let mut synchronized = 0usize;
        if let Some(tape) = self.hypergrad.as_mut() {
            let gradient = tape.gradient_mut();
            if !gradient.is_empty() {
                Self::synchronize_accumulator_buffer(gradient, &mut synchronize)?;
                synchronized += 1;
            }
        }
        if let Some(tape) = self.realgrad.as_mut() {
            let gradient = tape.gradient_mut();
            if !gradient.is_empty() {
                Self::synchronize_accumulator_buffer(gradient, &mut synchronize)?;
                synchronized += 1;
            }
        }
        if self.hypergrad.is_none() && self.realgrad.is_none() {
            if let Some(grad) = self.gradient.as_mut() {
                let data = grad.data_mut();
                if !data.is_empty() {
                    Self::synchronize_accumulator_buffer(data, &mut synchronize)?;
                    synchronized += 1;
                }
            }
        }
        Ok(synchronized)
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
        for (band, grad) in bands.iter_labeled() {
            let backend = current_tensor_util_backend_for_values(grad.data().len());
            if grad.squared_l2_norm_with_backend(backend)? == 0.0 {
                continue;
            }
            self.begin_backward_band_pass(band, grad)?;
            let result = self.backward(input, grad);
            self.end_backward_band_pass(band)?;
            let contribution = result?;
            let backend = current_tensor_util_backend_for_values(total.data().len());
            total.add_scaled_with_backend(&contribution, 1.0, backend)?;
        }
        Ok(total)
    }

    /// Announces the start of a band-specific backward replay.
    fn begin_backward_band_pass(
        &mut self,
        _band: RoundtableBand,
        _gradient: &Tensor,
    ) -> PureResult<()> {
        Ok(())
    }

    /// Announces the end of a band-specific backward replay.
    fn end_backward_band_pass(&mut self, _band: RoundtableBand) -> PureResult<()> {
        Ok(())
    }

    /// Applies the latest roundtable band signal before a backward pass.
    fn apply_roundtable_band(&mut self, _signal: &RoundtableBandSignal) -> PureResult<()> {
        Ok(())
    }

    /// Clears any previously applied roundtable directives.
    fn clear_roundtable_band(&mut self) -> PureResult<()> {
        Ok(())
    }

    /// Optional hook that allows modules to infuse raw text into their state.
    ///
    /// Layers that support Z-space resonance (e.g. wave gates) can override this
    /// and translate the provided string into parameter updates via
    /// [`Parameter::absorb_text`].
    ///
    /// The default implementation is a no-op so callers can broadcast a single
    /// infusion signal through composite modules (such as [`Sequential`])
    /// without needing per-layer feature checks.
    fn infuse_text(&mut self, _text: &str) -> PureResult<()> {
        Ok(())
    }

    /// Enables or disables training mode.
    ///
    /// The default implementation is a no-op so stateless layers (or layers
    /// without distinct training/evaluation behaviour) don't need to implement
    /// it. Stateful layers such as dropout or batch normalisation should
    /// override it to toggle their internal mode.
    fn set_training(&mut self, _training: bool) -> PureResult<()> {
        Ok(())
    }

    /// Convenience helper that switches the module into training mode.
    fn train(&mut self) -> PureResult<()> {
        self.set_training(true)
    }

    /// Convenience helper that switches the module into evaluation mode.
    fn eval(&mut self) -> PureResult<()> {
        self.set_training(false)
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

    /// Applies every parameter update while routing the gradients through the provided adapter.
    fn apply_step_with_adapter(
        &mut self,
        fallback_lr: f32,
        adapter: Option<&mut dyn LocalLearningRateAdapter>,
    ) -> PureResult<()> {
        if let Some(adapter) = adapter {
            let adapter_ref = adapter;
            self.visit_parameters_mut(&mut |param| {
                param.apply_step_with_adapter(fallback_lr, Some(&mut *adapter_ref))
            })
        } else {
            self.apply_step(fallback_lr)
        }
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

    struct FixedAdapter {
        factor: f32,
    }

    impl LocalLearningRateAdapter for FixedAdapter {
        fn scale_factor(&mut self, _: &str, _: &SpectralFeatureSample) -> f32 {
            self.factor
        }
    }

    #[test]
    fn parameter_respects_local_lr_adapter() {
        let mut param = Parameter::new("weight", Tensor::zeros(1, 2).unwrap());
        let update = Tensor::from_vec(1, 2, vec![0.5, -0.25]).unwrap();
        param.accumulate_euclidean(&update).unwrap();
        let mut adapter = FixedAdapter { factor: 2.0 };
        param
            .apply_step_with_adapter(0.1, Some(&mut adapter))
            .unwrap();
        let values = param.value().data();
        assert!((values[0] + 0.1 * 0.5 * 2.0).abs() < 1e-6);
        assert!((values[1] - 0.1 * 0.25 * 2.0).abs() < 1e-6);
    }

    #[test]
    fn parameter_rejects_invalid_fallback_lr_without_mutating_accumulators() {
        let mut param = Parameter::new("weight", Tensor::zeros(1, 2).unwrap());
        let update = Tensor::from_vec(1, 2, vec![0.5, -0.25]).unwrap();
        param.accumulate_euclidean(&update).unwrap();
        let value_before = param.value().clone();
        let grad_before = param.gradient().unwrap().clone();

        let err = param.apply_step(f32::NAN).unwrap_err();
        assert!(matches!(
            err,
            TensorError::NonPositiveLearningRate { rate } if rate.is_nan()
        ));
        assert_eq!(*param.value(), value_before);
        assert_eq!(*param.gradient().unwrap(), grad_before);

        let mut adapter = FixedAdapter { factor: 2.0 };
        let err = param
            .apply_step_with_adapter(-0.1, Some(&mut adapter))
            .unwrap_err();
        assert!(matches!(
            err,
            TensorError::NonPositiveLearningRate { rate } if (rate + 0.1).abs() < f32::EPSILON
        ));
        assert_eq!(*param.value(), value_before);
        assert_eq!(*param.gradient().unwrap(), grad_before);
    }

    #[test]
    fn parameter_rejects_overflowing_fallback_delta_without_mutating_accumulators() {
        let mut param = Parameter::new("weight", Tensor::zeros(1, 1).unwrap());
        let update = Tensor::from_vec(1, 1, vec![2.0]).unwrap();
        param.accumulate_euclidean(&update).unwrap();
        let value_before = param.value().clone();
        let grad_before = param.gradient().unwrap().clone();

        let err = param.apply_step(f32::MAX).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "parameter_delta",
                value,
            } if value.is_infinite()
        ));
        assert_eq!(*param.value(), value_before);
        assert_eq!(*param.gradient().unwrap(), grad_before);
    }

    #[test]
    fn parameter_rejects_overflowing_fallback_update_without_mutating_accumulators() {
        let mut param = Parameter::new("weight", Tensor::from_vec(1, 1, vec![f32::MAX]).unwrap());
        let update = Tensor::from_vec(1, 1, vec![-0.5]).unwrap();
        param.accumulate_euclidean(&update).unwrap();
        let value_before = param.value().clone();
        let grad_before = param.gradient().unwrap().clone();

        let err = param.apply_step(f32::MAX).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "parameter_update",
                value,
            } if value.is_infinite()
        ));
        assert_eq!(*param.value(), value_before);
        assert_eq!(*param.gradient().unwrap(), grad_before);
    }

    #[test]
    fn parameter_rejects_overflowing_adapter_scale_without_mutating_accumulators() {
        let mut param = Parameter::new("weight", Tensor::zeros(1, 1).unwrap());
        let update = Tensor::from_vec(1, 1, vec![2.0]).unwrap();
        param.accumulate_euclidean(&update).unwrap();
        let value_before = param.value().clone();
        let grad_before = param.gradient().unwrap().clone();
        let mut adapter = FixedAdapter { factor: f32::MAX };

        let err = param
            .apply_step_with_adapter(0.1, Some(&mut adapter))
            .unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "scaled_accumulator",
                value,
            } if value.is_infinite()
        ));
        assert_eq!(*param.value(), value_before);
        assert_eq!(*param.gradient().unwrap(), grad_before);
    }

    #[test]
    fn parameter_legacy_scale_accumulators_skips_overflow_without_mutating_buffer() {
        let mut param = Parameter::new("weight", Tensor::zeros(1, 1).unwrap());
        let update = Tensor::from_vec(1, 1, vec![2.0]).unwrap();
        param.accumulate_euclidean(&update).unwrap();
        let grad_before = param.gradient().unwrap().clone();

        param.scale_accumulators(f32::MAX);

        assert_eq!(*param.gradient().unwrap(), grad_before);
    }

    #[test]
    fn parameter_synchronizes_euclidean_accumulator_buffer() {
        let mut param = Parameter::new("weight", Tensor::zeros(1, 2).unwrap());
        let update = Tensor::from_vec(1, 2, vec![0.5, -0.25]).unwrap();
        param.accumulate_euclidean(&update).unwrap();

        let buffers = param
            .synchronize_accumulators_with(|gradient| {
                for value in gradient {
                    *value *= 2.0;
                }
                Ok(())
            })
            .unwrap();

        assert_eq!(buffers, 1);
        assert_eq!(param.gradient().unwrap().data(), &[1.0, -0.5]);
    }

    #[test]
    fn parameter_rejects_non_finite_synchronized_accumulator_without_mutating_buffer() {
        let mut param = Parameter::new("weight", Tensor::zeros(1, 2).unwrap());
        let update = Tensor::from_vec(1, 2, vec![0.5, -0.25]).unwrap();
        param.accumulate_euclidean(&update).unwrap();
        let grad_before = param.gradient().unwrap().clone();

        let err = param
            .synchronize_accumulators_with(|gradient| {
                gradient[0] = f32::INFINITY;
                Ok(())
            })
            .unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "synchronized_accumulator",
                value,
            } if value.is_infinite()
        ));
        assert_eq!(*param.gradient().unwrap(), grad_before);
    }

    #[test]
    fn parameter_synchronizes_hypergrad_and_realgrad_buffers() {
        let mut param = Parameter::new("weight", Tensor::zeros(1, 2).unwrap());
        param.attach_hypergrad(-1.0, 0.1).unwrap();
        param.attach_realgrad(0.1).unwrap();

        let mut calls = 0usize;
        let buffers = param
            .synchronize_accumulators_with(|gradient| {
                calls += 1;
                for value in gradient {
                    *value = calls as f32;
                }
                Ok(())
            })
            .unwrap();

        assert_eq!(buffers, 2);
        assert_eq!(calls, 2);
        assert_eq!(param.hypergrad().unwrap().gradient(), &[1.0, 1.0]);
        assert_eq!(param.realgrad().unwrap().gradient(), &[2.0, 2.0]);
        assert!(param.gradient().is_none());
    }
}
