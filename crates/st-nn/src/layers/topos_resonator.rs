// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::execution::current_tensor_util_backend_for_values;
use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor};
use st_tensor::topos::OpenCartesianTopos;
use st_tensor::{
    emit_tensor_op, emit_tensor_op_meta, LanguageWaveEncoder, TensorError, TensorUtilBackend,
};

/// Element-wise gate that keeps a persistent Z-space resonance attached to the
/// hypergrad tape. The resonator can ingest text or complex waves and amplify
/// the incoming signal before passing it downstream.
#[derive(Debug)]
pub struct ToposResonator {
    gate: Parameter,
    encoder: Option<LanguageWaveEncoder>,
}

fn tensor_util_backend_label(backend: TensorUtilBackend) -> &'static str {
    match backend {
        TensorUtilBackend::Auto => "auto",
        TensorUtilBackend::Cpu => "cpu",
        TensorUtilBackend::GpuWgpu => "wgpu",
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_topos_resonator_meta(
    op_name: &'static str,
    kind: &'static str,
    rows: usize,
    cols: usize,
    gate_shape: (usize, usize),
    requested_backend: TensorUtilBackend,
    encoder_attached: bool,
    backward: bool,
) {
    let values = rows.saturating_mul(cols);
    let gate_values = gate_shape.0.saturating_mul(gate_shape.1);
    emit_tensor_op(
        op_name,
        &[rows, cols, gate_shape.0, gate_shape.1],
        &[rows, cols],
    );
    emit_tensor_op_meta(op_name, || {
        serde_json::json!({
            "backend": "composite",
            "requested_backend": tensor_util_backend_label(requested_backend),
            "delegate_backend": tensor_util_backend_label(requested_backend),
            "kernel": "topos_resonator.hadamard_gate",
            "kind": kind,
            "rows": rows,
            "cols": cols,
            "values": values,
            "output_rows": rows,
            "output_cols": cols,
            "output_values": values,
            "gate_rows": gate_shape.0,
            "gate_cols": gate_shape.1,
            "gate_values": gate_values,
            "trainable_parameters": gate_values,
            "encoder_attached": encoder_attached,
            "estimated_gate_multiply_ops": values,
            "estimated_gradient_multiply_ops": if backward { values.saturating_mul(2) } else { 0 },
            "accumulates_gate_gradient": backward,
            "backward": backward,
            "empty": values == 0,
        })
    });
}

impl ToposResonator {
    /// Creates a new resonator with an identity gate.
    pub fn new(name: impl Into<String>, rows: usize, cols: usize) -> PureResult<Self> {
        let weights = Tensor::from_vec(rows, cols, vec![1.0; rows * cols])?;
        Ok(Self {
            gate: Parameter::new(name, weights),
            encoder: None,
        })
    }

    /// Provides immutable access to the parameter for external inspection.
    pub fn parameter(&self) -> &Parameter {
        &self.gate
    }

    /// Provides mutable access to the parameter so callers can attach
    /// hypergrad tapes or rename the gate.
    pub fn parameter_mut(&mut self) -> &mut Parameter {
        &mut self.gate
    }

    /// Attaches a dedicated text encoder that can stream descriptions directly
    /// into the hypergrad tape.
    pub fn with_encoder(mut self, encoder: LanguageWaveEncoder) -> Self {
        self.encoder = Some(encoder);
        self
    }

    /// Streams raw text into the resonator when an encoder has been attached.
    pub fn infuse_text(&mut self, text: &str) -> PureResult<()> {
        let encoder = self
            .encoder
            .as_ref()
            .ok_or(TensorError::EmptyInput("topos resonator encoder"))?;
        self.gate.absorb_text(encoder, text)
    }

    /// Connects the gate to a caller supplied open topos.
    pub fn attach_open_topos(
        &mut self,
        curvature: f32,
        learning_rate: f32,
        topos: OpenCartesianTopos,
    ) -> PureResult<()> {
        self.gate
            .attach_hypergrad_with_topos(curvature, learning_rate, topos)
    }
}

impl Module for ToposResonator {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let backend = current_tensor_util_backend_for_values(input.data().len());
        let output = input.hadamard_with_backend(self.gate.value(), backend)?;
        let (rows, cols) = input.shape();
        emit_topos_resonator_meta(
            "topos_resonator_forward",
            "topos_resonator_forward_gate",
            rows,
            cols,
            self.gate.value().shape(),
            backend,
            self.encoder.is_some(),
            false,
        );
        Ok(output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let backend = current_tensor_util_backend_for_values(grad_output.data().len());
        let grad_gate = grad_output.hadamard_with_backend(input, backend)?;
        self.gate.accumulate_euclidean(&grad_gate)?;
        let grad_input = grad_output.hadamard_with_backend(self.gate.value(), backend)?;
        let (rows, cols) = grad_output.shape();
        emit_topos_resonator_meta(
            "topos_resonator_backward",
            "topos_resonator_backward_gate",
            rows,
            cols,
            self.gate.value().shape(),
            backend,
            self.encoder.is_some(),
            true,
        );
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.gate)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.gate)
    }

    fn infuse_text(&mut self, text: &str) -> PureResult<()> {
        let Some(encoder) = self.encoder.as_ref() else {
            return Ok(());
        };
        self.gate.absorb_text(encoder, text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

    fn observer_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    #[test]
    fn resonator_behaves_like_identity_by_default() {
        let resonator = ToposResonator::new("gate", 2, 2).unwrap();
        let input = Tensor::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = resonator.forward(&input).unwrap();
        assert_eq!(out.data(), input.data());
    }

    #[test]
    fn backward_accumulates_gate_gradient() {
        let mut resonator = ToposResonator::new("gate", 2, 2).unwrap();
        let input = Tensor::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grad_output = Tensor::from_vec(2, 2, vec![0.5, 0.25, 0.1, 0.0]).unwrap();
        let grad_input = resonator.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input.data(), grad_output.data());
        let mut captured: Option<Tensor> = None;
        resonator
            .visit_parameters(&mut |param: &Parameter| {
                captured = param.gradient().cloned();
                Ok(())
            })
            .unwrap();
        let gradient = captured.expect("gradient present");
        assert_eq!(
            gradient.data(),
            grad_output.hadamard(&input).unwrap().data()
        );
    }

    #[test]
    fn resonator_forward_backward_emit_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut resonator = ToposResonator::new("gate", 2, 2).unwrap();
        let input = Tensor::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = resonator.forward(&input).unwrap();
        let grad_output = Tensor::from_vec(2, 2, vec![0.5, 0.25, 0.1, 0.0]).unwrap();
        let grad_input = resonator.backward(&input, &grad_output).unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        assert_eq!(out.shape(), input.shape());
        assert_eq!(grad_input.shape(), input.shape());
        let events = events.lock().unwrap();
        for (op_name, kind, backward) in [
            (
                "topos_resonator_forward",
                "topos_resonator_forward_gate",
                false,
            ),
            (
                "topos_resonator_backward",
                "topos_resonator_backward_gate",
                true,
            ),
        ] {
            let event = events
                .iter()
                .find(|(name, data)| {
                    *name == op_name
                        && data["backend"] == "composite"
                        && data["kind"] == kind
                        && data["rows"] == 2
                        && data["cols"] == 2
                        && data["gate_values"] == 4
                })
                .unwrap_or_else(|| panic!("{op_name} metadata event"));
            assert_eq!(event.1["requested_backend"], "auto");
            assert_eq!(event.1["delegate_backend"], "auto");
            assert_eq!(event.1["values"], 4);
            assert_eq!(event.1["trainable_parameters"], 4);
            assert_eq!(event.1["encoder_attached"], false);
            assert_eq!(event.1["backward"], backward);
        }
        assert!(
            events
                .iter()
                .any(|(name, data)| *name == "hadamard" && data["backend"] == "cpu"),
            "resonator should still expose delegated hadamard backend"
        );
    }
}
