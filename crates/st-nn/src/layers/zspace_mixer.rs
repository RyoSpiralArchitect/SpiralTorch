// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use num_complex::Complex32;
use st_tensor::{PureResult, Tensor, TensorError};
use thiserror::Error;

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
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            let offset = r * cols;
            for c in 0..cols {
                data.push(input.data()[offset + c] * gate[c]);
            }
        }
        Tensor::from_vec(rows, cols, data)
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

        let mut grad_gate = vec![0.0f32; cols];
        for r in 0..rows {
            let offset = r * cols;
            for c in 0..cols {
                let idx = offset + c;
                grad_gate[c] += grad_output.data()[idx] * input.data()[idx];
            }
        }
        let grad_tensor = Tensor::from_vec(1, cols, grad_gate)?;
        self.gate.accumulate_euclidean(&grad_tensor)?;

        let mut grad_input = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            let offset = r * cols;
            for c in 0..cols {
                let idx = offset + c;
                grad_input.push(grad_output.data()[idx] * gate[c]);
            }
        }
        Tensor::from_vec(rows, cols, grad_input)
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
        let expected_grad = vec![
            1.0 * 0.5 + 4.0 * 0.25,
            2.0 * 1.0 + 5.0 * 0.5,
            3.0 * -1.0 + 6.0 * -0.5,
        ];
        let grads = mixer.gate().gradient().unwrap();
        for (expected, actual) in expected_grad.iter().zip(grads.data()) {
            assert!((expected - actual).abs() < 1e-6);
        }
        assert_eq!(gate.shape(), (1, 3));
    }
}
