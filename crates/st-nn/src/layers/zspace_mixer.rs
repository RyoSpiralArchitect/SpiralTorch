// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::execution::current_tensor_util_backend_for_values;
use crate::module::{Module, Parameter};
use num_complex::Complex32;
use st_tensor::{emit_tensor_op, emit_tensor_op_meta, PureResult, Tensor, TensorError};
use thiserror::Error;

fn emit_zspace_mixer_meta(
    op_name: &'static str,
    rows: usize,
    cols: usize,
    backward: bool,
    broadcast_backend: Option<String>,
    gradient_reduction_backend: Option<String>,
    gradient_scale: Option<f32>,
) {
    let input_shape = if backward {
        vec![rows, cols, rows, cols, 1, cols]
    } else {
        vec![rows, cols, 1, cols]
    };
    emit_tensor_op(op_name, &input_shape, &[rows, cols]);
    emit_tensor_op_meta(op_name, || {
        let values = rows.saturating_mul(cols);
        serde_json::json!({
            "backend": "composite",
            "requested_backend": "auto",
            "kernel": "zspace_mixer.scalar",
            "kind": if backward { "broadcast_gate_backward" } else { "broadcast_gate_forward" },
            "rows": rows,
            "cols": cols,
            "values": values,
            "output_rows": rows,
            "output_cols": cols,
            "output_values": values,
            "gate_cols": cols,
            "trainable_parameters": cols,
            "broadcast_backend": broadcast_backend,
            "gradient_reduction_backend": gradient_reduction_backend,
            "gradient_scale": gradient_scale,
            "parameter_gradient_scale": gradient_scale,
            "input_gradient_scale": if backward && gradient_scale.is_some() {
                Some(1.0f32)
            } else {
                None
            },
            "estimated_broadcast_ops": values,
            "estimated_gate_gradient_ops": if backward { values.saturating_mul(2) } else { 0 },
            "estimated_total_ops": if backward {
                values.saturating_mul(3)
            } else {
                values
            },
            "empty": rows == 0 || cols == 0,
        })
    });
}

/// Lightweight gating module that modulates Z-space activations column-wise.
///
/// The mixer keeps a single row of parameters and broadcasts it across the
/// incoming batch, performing an element-wise product. This keeps the module
/// compatible with the hypergrad tape while remaining fully deterministic in
/// CPU-only environments.
pub struct ZSpaceMixer {
    gate: Parameter,
}

impl ZSpaceMixer {
    /// Builds a mixer with the provided number of features. Parameters start at
    /// `1.0` so the module initially acts as a pass-through.
    pub fn new(name: impl Into<String>, features: usize) -> PureResult<Self> {
        if features == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: 1,
                cols: features,
            });
        }
        let weights = Tensor::from_fn(1, features, |_, _| 1.0)?;
        Ok(Self {
            gate: Parameter::new(name, weights),
        })
    }

    /// Returns a view into the underlying parameter.
    pub fn gate(&self) -> &Parameter {
        &self.gate
    }

    /// Returns a mutable view into the parameter.
    pub fn gate_mut(&mut self) -> &mut Parameter {
        &mut self.gate
    }

    fn assert_input(&self, input: &Tensor) -> PureResult<()> {
        let (_, cols) = input.shape();
        let gate_shape = self.gate.value().shape();
        if gate_shape.1 != cols {
            return Err(TensorError::ShapeMismatch {
                left: gate_shape,
                right: (1, cols),
            });
        }
        Ok(())
    }

    fn gate_row(&self) -> &[f32] {
        self.gate.value().data()
    }
}

impl Module for ZSpaceMixer {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        self.assert_input(input)?;
        let (rows, cols) = input.shape();
        let gate: Vec<f32> = self.gate_row().to_vec();
        let util_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        let output = input.mul_row_with_backend(&gate, util_backend)?;
        emit_zspace_mixer_meta(
            "zspace_mixer_forward",
            rows,
            cols,
            false,
            Some(util_backend.to_string()),
            None,
            None,
        );
        Ok(output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        self.assert_input(input)?;
        if input.shape() != grad_output.shape() {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: grad_output.shape(),
            });
        }

        let (rows, cols) = input.shape();
        let gate: Vec<f32> = self.gate_row().to_vec();

        if rows == 0 {
            let output = Tensor::zeros(rows, cols)?;
            emit_zspace_mixer_meta("zspace_mixer_backward", rows, cols, true, None, None, None);
            return Ok(output);
        }

        let util_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        let grad_gate_product = grad_output.hadamard_with_backend(input, util_backend)?;
        let gradient_scale = 1.0 / rows as f32;
        let grad_gate =
            grad_gate_product.try_sum_axis0_scaled_with_backend(gradient_scale, util_backend)?;
        let grad_tensor = Tensor::from_vec(1, cols, grad_gate)?;
        self.gate.accumulate_euclidean(&grad_tensor)?;

        let output = grad_output.mul_row_with_backend(&gate, util_backend)?;
        emit_zspace_mixer_meta(
            "zspace_mixer_backward",
            rows,
            cols,
            true,
            Some(util_backend.to_string()),
            Some(util_backend.to_string()),
            Some(gradient_scale),
        );
        Ok(output)
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
}

// ===== SpiralTorch: zspace_mixer hardening additions (non-breaking) =====
#[derive(Debug, Error)]
pub enum MixerError {
    #[error("no channels")]
    EmptyChannels,
    #[error("channel length mismatch at chan {chan}: got {got}, expected {expected}")]
    LengthMismatch {
        chan: usize,
        got: usize,
        expected: usize,
    },
    #[error("non-finite weight at index {index}: {value}")]
    NonFiniteWeight { index: usize, value: f32 },
    #[error("non-finite sample at chan {chan} index {index}")]
    NonFiniteSample { chan: usize, index: usize },
    #[error("degenerate normalization (zero norm)")]
    DegenerateNorm,
}
pub type MixerResult<T> = Result<T, MixerError>;

#[inline]
fn st_is_finite_f32(x: f32) -> bool {
    x.is_finite()
}
#[inline]
fn st_is_finite_c32(z: Complex32) -> bool {
    z.re.is_finite() && z.im.is_finite()
}

/// L1 正規化（重み総和=1）。総和が小さすぎる場合はエラー。
pub fn st_norm_l1(weights: &[f32]) -> MixerResult<Vec<f32>> {
    if weights.is_empty() {
        return Err(MixerError::EmptyChannels);
    }
    let mut sum = 0f32;
    for (i, &w) in weights.iter().enumerate() {
        if !st_is_finite_f32(w) {
            return Err(MixerError::NonFiniteWeight { index: i, value: w });
        }
        sum += w;
    }
    if sum.abs() < 1e-30 {
        return Err(MixerError::DegenerateNorm);
    }
    Ok(weights.iter().map(|&w| w / sum).collect())
}

/// L2 正規化（||w||2 = 1）。ノルムが小さすぎる場合はエラー。
pub fn st_norm_l2(weights: &[f32]) -> MixerResult<Vec<f32>> {
    if weights.is_empty() {
        return Err(MixerError::EmptyChannels);
    }
    let mut s2 = 0f32;
    for (i, &w) in weights.iter().enumerate() {
        if !st_is_finite_f32(w) {
            return Err(MixerError::NonFiniteWeight { index: i, value: w });
        }
        s2 += w * w;
    }
    if s2 <= 0.0 {
        return Err(MixerError::DegenerateNorm);
    }
    let inv = s2.sqrt().recip();
    Ok(weights.iter().map(|&w| w * inv).collect())
}

/// 複数の複素系列を**安定に重み付き合成**する（Kahan/Neumaierでサンプル毎に補正）。
/// - `channels`: 各チャネルは同一長の &[Complex32]
/// - `weights`: channels.len() と同じ長さ
/// - `use_l2`: trueでL2正規化、falseでL1正規化
pub fn st_mix_series(
    channels: &[&[Complex32]],
    weights: &[f32],
    use_l2: bool,
) -> MixerResult<Vec<Complex32>> {
    if channels.is_empty() {
        return Err(MixerError::EmptyChannels);
    }
    if channels.len() != weights.len() {
        return Err(MixerError::LengthMismatch {
            chan: usize::MAX,
            got: weights.len(),
            expected: channels.len(),
        });
    }

    // 長さ検証
    let n = channels[0].len();
    for (cidx, ch) in channels.iter().enumerate() {
        if ch.len() != n {
            return Err(MixerError::LengthMismatch {
                chan: cidx,
                got: ch.len(),
                expected: n,
            });
        }
    }
    // 正規化
    let w = if use_l2 {
        st_norm_l2(weights)?
    } else {
        st_norm_l1(weights)?
    };

    // 出力と補正項
    let mut out = vec![Complex32::new(0.0, 0.0); n];
    let mut c_re = vec![0f32; n];
    let mut c_im = vec![0f32; n];

    for (cidx, (ch, &w_i)) in channels.iter().zip(w.iter()).enumerate() {
        for i in 0..n {
            let x = ch[i];
            if !st_is_finite_c32(x) {
                return Err(MixerError::NonFiniteSample {
                    chan: cidx,
                    index: i,
                });
            }
            // 加算する量
            let add_re = x.re * w_i;
            let add_im = x.im * w_i;
            // Kahan/Neumaier 的補正（実部）
            let t_re = add_re - c_re[i];
            let u_re = out[i].re + t_re;
            c_re[i] = (u_re - out[i].re) - t_re;
            out[i].re = u_re;
            // 虚部
            let t_im = add_im - c_im[i];
            let u_im = out[i].im + t_im;
            c_im[i] = (u_im - out[i].im) - t_im;
            out[i].im = u_im;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod st_mixer_tests {
    use super::*;

    #[test]
    fn l1_norm_basic() {
        let w = st_norm_l1(&[1.0, 1.0, 2.0]).unwrap();
        let s: f32 = w.iter().sum();
        assert!((s - 1.0).abs() < 1e-7, "sum={}", s);
    }

    #[test]
    fn l2_norm_basic() {
        let w = st_norm_l2(&[3.0, 4.0]).unwrap();
        let s2: f32 = w.iter().map(|x| x * x).sum();
        assert!((s2 - 1.0).abs() < 1e-6, "||w||2^2={}", s2);
    }

    #[test]
    fn mix_series_matches_naive_small() {
        let a = vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, 1.0)];
        let b = vec![Complex32::new(0.5, 0.5), Complex32::new(-0.5, 0.0)];
        let out = st_mix_series(&[&a, &b], &[2.0, 1.0], false).unwrap(); // L1 -> [2/3, 1/3]
        let naive: Vec<Complex32> = (0..a.len())
            .map(|i| a[i] * (2.0 / 3.0) + b[i] * (1.0 / 3.0))
            .collect();
        let err: f32 = (0..a.len()).map(|i| (out[i] - naive[i]).norm()).sum();
        assert!(err < 1e-5, "err={}", err);
    }
}
// ===== end additions =====

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "wgpu")]
    use crate::execution::{push_backend_policy, BackendPolicy};
    #[cfg(feature = "wgpu")]
    use st_core::backend::device_caps::DeviceCaps;
    #[cfg(feature = "wgpu")]
    use st_tensor::backend::wgpu_dense;
    use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

    fn observer_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    #[test]
    fn mixer_scales_and_accumulates_gradients() {
        let mut mixer = ZSpaceMixer::new("mixer", 3).unwrap();
        let input = Tensor::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output = mixer.forward(&input).unwrap();
        assert_eq!(output.data(), input.data());

        let grad_output = Tensor::from_vec(2, 3, vec![0.5, 1.0, -1.0, 0.25, 0.5, -0.5]).unwrap();
        let grad_input = mixer.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input.data(), grad_output.data());

        let gate = mixer.gate().value();
        let expected_grad = [
            (1.0 * 0.5 + 4.0 * 0.25) * 0.5,
            (2.0 * 1.0 + 5.0 * 0.5) * 0.5,
            (-3.0 + 6.0 * -0.5) * 0.5,
        ];
        let grads = mixer.gate().gradient().unwrap();
        for (expected, actual) in expected_grad.iter().zip(grads.data()) {
            assert!((expected - actual).abs() < 1e-6);
        }
        assert_eq!(gate.shape(), (1, 3));
    }

    #[test]
    fn mixer_forward_backward_emit_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut mixer = ZSpaceMixer::new("mixer", 3).unwrap();
        let input = Tensor::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let grad_output = Tensor::from_vec(2, 3, vec![0.5, 1.0, -1.0, 0.25, 0.5, -0.5]).unwrap();
        let _ = mixer.forward(&input).unwrap();
        let _ = mixer.backward(&input, &grad_output).unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        let forward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "zspace_mixer_forward" && data["rows"] == 2 && data["cols"] == 3
            })
            .expect("zspace mixer forward metadata event");
        assert_eq!(forward.1["backend"], "composite");
        assert_eq!(forward.1["kind"], "broadcast_gate_forward");
        assert_eq!(forward.1["gate_cols"], 3);
        assert_eq!(forward.1["broadcast_backend"], "auto");

        let backward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "zspace_mixer_backward" && data["rows"] == 2 && data["cols"] == 3
            })
            .expect("zspace mixer backward metadata event");
        assert_eq!(backward.1["backend"], "composite");
        assert_eq!(backward.1["kind"], "broadcast_gate_backward");
        assert_eq!(backward.1["broadcast_backend"], "auto");
        assert_eq!(backward.1["gradient_reduction_backend"], "auto");
        assert_eq!(backward.1["gradient_scale"], 0.5);
        assert_eq!(backward.1["parameter_gradient_scale"], 0.5);
        assert_eq!(backward.1["input_gradient_scale"], 1.0);
        assert!(
            backward.1["estimated_gate_gradient_ops"]
                .as_u64()
                .unwrap_or(0)
                > 0
        );

        let hadamard = events
            .iter()
            .any(|(op_name, data)| *op_name == "hadamard" && data["rows"] == 2);
        let mul_row = events
            .iter()
            .filter(|(op_name, data)| *op_name == "mul_row" && data["rows"] == 2)
            .count();
        let reduction = events
            .iter()
            .any(|(op_name, data)| *op_name == "sum_axis0_scaled" && data["cols"] == 3);
        assert!(hadamard);
        assert_eq!(mul_row, 2);
        assert!(reduction);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn mixer_forced_wgpu_uses_mul_row_and_matches_cpu_reference() {
        if !wgpu_dense::is_available() {
            return;
        }
        let _lock = observer_lock();
        let previous_threshold = std::env::var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES").ok();
        std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", "1");

        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let input = Tensor::from_fn(5, 4, |row, col| {
            ((row * 19 + col * 7) % 17) as f32 * 0.041 - 0.3
        })
        .unwrap();
        let grad_output = Tensor::from_fn(5, 4, |row, col| {
            ((row * 23 + col * 11) % 13) as f32 * 0.027 - 0.16
        })
        .unwrap();
        let cpu_policy = BackendPolicy::from_device_caps(DeviceCaps::cpu());
        let mut cpu_mixer = ZSpaceMixer::new("mixer_cpu", 4).unwrap();
        for (idx, value) in cpu_mixer
            .gate_mut()
            .value_mut()
            .data_mut()
            .iter_mut()
            .enumerate()
        {
            *value = 0.7 + idx as f32 * 0.13;
        }
        let cpu_forward = {
            let _guard = push_backend_policy(cpu_policy);
            cpu_mixer.forward(&input).unwrap()
        };
        let cpu_grad_input = {
            let _guard = push_backend_policy(cpu_policy);
            cpu_mixer.backward(&input, &grad_output).unwrap()
        };

        let wgpu_policy = BackendPolicy::from_device_caps(DeviceCaps::wgpu(32, true, 256));
        let mut wgpu_mixer = ZSpaceMixer::new("mixer_wgpu", 4).unwrap();
        for (idx, value) in wgpu_mixer
            .gate_mut()
            .value_mut()
            .data_mut()
            .iter_mut()
            .enumerate()
        {
            *value = 0.7 + idx as f32 * 0.13;
        }
        let (wgpu_forward, wgpu_grad_input) = {
            let _guard = push_backend_policy(wgpu_policy);
            (
                wgpu_mixer.forward(&input).unwrap(),
                wgpu_mixer.backward(&input, &grad_output).unwrap(),
            )
        };

        st_tensor::set_tensor_op_meta_observer(previous);
        match previous_threshold {
            Some(value) => std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", value),
            None => std::env::remove_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES"),
        }

        assert_eq!(cpu_forward.shape(), wgpu_forward.shape());
        for (idx, (&cpu, &wgpu)) in cpu_forward
            .data()
            .iter()
            .zip(wgpu_forward.data().iter())
            .enumerate()
        {
            let delta = (cpu - wgpu).abs();
            assert!(
                delta <= 1e-6,
                "mixer forward mismatch at {idx}: cpu={cpu} wgpu={wgpu} delta={delta}"
            );
        }
        assert_eq!(cpu_grad_input.shape(), wgpu_grad_input.shape());
        for (idx, (&cpu, &wgpu)) in cpu_grad_input
            .data()
            .iter()
            .zip(wgpu_grad_input.data().iter())
            .enumerate()
        {
            let delta = (cpu - wgpu).abs();
            assert!(
                delta <= 1e-6,
                "mixer backward input mismatch at {idx}: cpu={cpu} wgpu={wgpu} delta={delta}"
            );
        }

        let cpu_gate_grad = cpu_mixer.gate().gradient().unwrap();
        let wgpu_gate_grad = wgpu_mixer.gate().gradient().unwrap();
        for (idx, (&cpu, &wgpu)) in cpu_gate_grad
            .data()
            .iter()
            .zip(wgpu_gate_grad.data().iter())
            .enumerate()
        {
            let delta = (cpu - wgpu).abs();
            assert!(
                delta <= 1e-6,
                "mixer gate gradient mismatch at {idx}: cpu={cpu} wgpu={wgpu} delta={delta}"
            );
        }

        let events = events.lock().unwrap();
        assert!(events
            .iter()
            .any(|(op_name, data)| *op_name == "mul_row" && data["backend"] == "wgpu_dense"));
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "zspace_mixer_forward"
                && data["backend"] == "composite"
                && data["broadcast_backend"] == "wgpu"
        }));
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "zspace_mixer_backward"
                && data["backend"] == "composite"
                && data["broadcast_backend"] == "wgpu"
        }));
    }

    #[test]
    fn mixer_input_gradient_matches_numeric_gradients_without_batch_scaling() {
        let mut mixer = ZSpaceMixer::new("mixer", 2).unwrap();
        mixer.gate_mut().value_mut().data_mut()[0] = 1.25;
        mixer.gate_mut().value_mut().data_mut()[1] = -0.75;

        let input = Tensor::from_vec(2, 2, vec![0.2, -0.4, 0.6, 0.8]).unwrap();
        let grad_output = Tensor::from_vec(2, 2, vec![0.3, -0.2, 0.5, -0.7]).unwrap();
        let grad_input = mixer.backward(&input, &grad_output).unwrap();
        let grad_output_vec = grad_output.data().to_vec();
        let epsilon = 1e-3;

        for idx in 0..input.data().len() {
            let mut plus = input.clone();
            plus.data_mut()[idx] += epsilon;
            let out_plus = mixer.forward(&plus).unwrap();
            let loss_plus = out_plus
                .data()
                .iter()
                .zip(grad_output_vec.iter())
                .map(|(out, grad)| out * grad)
                .sum::<f32>();

            let mut minus = input.clone();
            minus.data_mut()[idx] -= epsilon;
            let out_minus = mixer.forward(&minus).unwrap();
            let loss_minus = out_minus
                .data()
                .iter()
                .zip(grad_output_vec.iter())
                .map(|(out, grad)| out * grad)
                .sum::<f32>();

            let numeric = (loss_plus - loss_minus) / (2.0 * epsilon);
            assert!((numeric - grad_input.data()[idx]).abs() < 1e-4);
        }
    }

    #[test]
    fn mixer_empty_batch_returns_empty_grad_without_gate_update() {
        let mut mixer = ZSpaceMixer::new("mixer", 3).unwrap();
        let input = Tensor::from_vec(0, 3, Vec::new()).unwrap();
        let grad_output = Tensor::from_vec(0, 3, Vec::new()).unwrap();

        let output = mixer.forward(&input).unwrap();
        let grad_input = mixer.backward(&input, &grad_output).unwrap();

        assert_eq!(output.shape(), input.shape());
        assert!(output.data().is_empty());
        assert_eq!(grad_input.shape(), input.shape());
        assert!(grad_input.data().is_empty());
        assert!(mixer.gate().gradient().is_none());
    }
}
