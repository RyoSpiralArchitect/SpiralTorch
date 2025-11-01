// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Hardmax fusion utilities bridging abstract einsum-style planning with
//! backend-specific fused kernels. The goal is to keep the higher level tensor
//! API agnostic of where the computation runs while still surfacing metadata
//! describing the dynamic-programming reductions that happen along the way.

#[cfg(feature = "wgpu")]
use crate::backend::wgpu_dense;
#[cfg(not(feature = "wgpu"))]
use crate::pure::{Layout, PureResult};
#[cfg(feature = "wgpu")]
use crate::pure::{Layout, PureResult, TensorError};
use core::fmt;
use rayon::{current_num_threads, prelude::*};
use std::sync::atomic::{AtomicUsize, Ordering};

const DP_CPU_THRESHOLD: usize = 4_096;
const PARALLEL_ROW_THRESHOLD: usize = 32;
const PARALLEL_MIN_VOLUME: usize = 16_384;

/// Explicit backend selection for row-wise hardmax style reductions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HardmaxBackend {
    /// Allow SpiralTorch to pick the most appropriate backend.
    Auto,
    /// Force the pure Rust implementation.
    Cpu,
    /// Execute on the WGPU accelerator backend when available.
    #[cfg(feature = "wgpu")]
    GpuWgpu,
}

impl HardmaxBackend {
    pub(crate) fn label(self) -> &'static str {
        match self {
            HardmaxBackend::Auto => "auto",
            HardmaxBackend::Cpu => "cpu",
            #[cfg(feature = "wgpu")]
            HardmaxBackend::GpuWgpu => "wgpu",
        }
    }
}

impl fmt::Display for HardmaxBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Logical output requested from the fusion plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HardmaxMode {
    /// Produce both the softmax probabilities and hardmax mask.
    SoftmaxAndMask,
    /// Only compute the argmax mask.
    MaskOnly,
}

/// Result of executing a hardmax fusion plan.
#[derive(Clone, Debug)]
pub(crate) struct HardmaxFusionResult {
    pub(crate) softmax: Option<Vec<f32>>,
    pub(crate) hardmax: Vec<f32>,
    pub(crate) dp_reductions: usize,
    pub(crate) einsum: &'static str,
    pub(crate) fused_ops: &'static str,
}

/// Builder linking an abstract einsum description with a backend execution.
#[derive(Clone, Debug)]
pub(crate) struct HardmaxFusionPlan<'a> {
    input: &'a [f32],
    rows: usize,
    cols: usize,
    layout: Layout,
    backend: HardmaxBackend,
    mode: HardmaxMode,
}

impl<'a> HardmaxFusionPlan<'a> {
    pub(crate) fn new(input: &'a [f32], rows: usize, cols: usize) -> Self {
        Self {
            input,
            rows,
            cols,
            layout: Layout::RowMajor,
            backend: HardmaxBackend::Auto,
            mode: HardmaxMode::SoftmaxAndMask,
        }
    }

    pub(crate) fn layout(mut self, layout: Layout) -> Self {
        self.layout = layout;
        self
    }

    pub(crate) fn backend(mut self, backend: HardmaxBackend) -> Self {
        self.backend = backend;
        self
    }

    pub(crate) fn mode(mut self, mode: HardmaxMode) -> Self {
        self.mode = mode;
        self
    }

    pub(crate) fn execute(&self) -> PureResult<HardmaxFusionResult> {
        match self.backend {
            HardmaxBackend::Auto => self.execute_auto(),
            HardmaxBackend::Cpu => self.execute_cpu(),
            #[cfg(feature = "wgpu")]
            HardmaxBackend::GpuWgpu => self.execute_gpu(),
        }
    }

    fn execute_auto(&self) -> PureResult<HardmaxFusionResult> {
        let volume = self.rows.saturating_mul(self.cols);
        if volume <= DP_CPU_THRESHOLD {
            return self.execute_cpu();
        }

        #[cfg(feature = "wgpu")]
        {
            if self.gpu_supported() {
                return self.execute_gpu();
            }
        }

        self.execute_cpu()
    }

    fn execute_cpu(&self) -> PureResult<HardmaxFusionResult> {
        let einsum_signature = match self.mode {
            HardmaxMode::SoftmaxAndMask => "bi->bi",
            HardmaxMode::MaskOnly => "bi->argmax",
        };
        let fused_ops = match self.mode {
            HardmaxMode::SoftmaxAndMask => "softmax|hardmax",
            HardmaxMode::MaskOnly => "hardmax",
        };

        let executor = HardmaxEinsum::new(self.input, self.rows, self.cols);
        let result = executor.evaluate(self.mode);
        let HardmaxEinsumResult {
            softmax,
            hardmax,
            dp_reductions,
            parallelized,
        } = result;
        Ok(HardmaxFusionResult {
            softmax,
            hardmax,
            dp_reductions,
            einsum: einsum_signature,
            fused_ops: match (self.mode, parallelized) {
                (HardmaxMode::SoftmaxAndMask, true) => "softmax|hardmax|par",
                (HardmaxMode::SoftmaxAndMask, false) => fused_ops,
                (HardmaxMode::MaskOnly, true) => "hardmax|par",
                (HardmaxMode::MaskOnly, false) => fused_ops,
            },
        })
    }

    #[cfg(feature = "wgpu")]
    fn execute_gpu(&self) -> PureResult<HardmaxFusionResult> {
        if !self.gpu_supported() {
            return Err(TensorError::BackendFailure {
                backend: "wgpu",
                message: format!(
                    "hardmax fusion is not supported for shape {}x{} on the active device",
                    self.rows, self.cols
                ),
            });
        }

        match self.mode {
            HardmaxMode::SoftmaxAndMask => {
                let (softmax, hardmax) =
                    wgpu_dense::row_softmax_hardmax(self.input, self.rows, self.cols, self.layout)
                        .map_err(|message| TensorError::BackendFailure {
                            backend: "wgpu",
                            message,
                        })?;

                Ok(HardmaxFusionResult {
                    softmax: Some(softmax),
                    hardmax,
                    dp_reductions: self.rows.saturating_mul(self.cols),
                    einsum: "bi->bi",
                    fused_ops: "wgpu::softmax_hardmax",
                })
            }
            HardmaxMode::MaskOnly => {
                let hardmax =
                    wgpu_dense::row_hardmax(self.input, self.rows, self.cols, self.layout)
                        .map_err(|message| TensorError::BackendFailure {
                            backend: "wgpu",
                            message,
                        })?;

                Ok(HardmaxFusionResult {
                    softmax: None,
                    hardmax,
                    dp_reductions: self.rows.saturating_mul(self.cols),
                    einsum: "bi->argmax",
                    fused_ops: "wgpu::hardmax",
                })
            }
        }
    }

    #[cfg(feature = "wgpu")]
    fn gpu_supported(&self) -> bool {
        if !wgpu_dense::is_available() {
            return false;
        }

        match self.mode {
            HardmaxMode::SoftmaxAndMask => {
                wgpu_dense::supports_row_softmax_hardmax(self.rows, self.cols)
            }
            HardmaxMode::MaskOnly => wgpu_dense::supports_row_hardmax(self.rows, self.cols),
        }
    }
}

#[derive(Clone, Debug)]
struct HardmaxEinsum<'a> {
    input: &'a [f32],
    rows: usize,
    cols: usize,
}

impl<'a> HardmaxEinsum<'a> {
    fn new(input: &'a [f32], rows: usize, cols: usize) -> Self {
        Self { input, rows, cols }
    }

    fn evaluate(&self, mode: HardmaxMode) -> HardmaxEinsumResult {
        let expected = self.rows.saturating_mul(self.cols);
        let mut hardmax = vec![0.0; expected];
        if expected == 0 || self.input.len() != expected {
            let softmax = match mode {
                HardmaxMode::SoftmaxAndMask => Some(vec![0.0; expected]),
                HardmaxMode::MaskOnly => None,
            };
            return HardmaxEinsumResult {
                softmax,
                hardmax,
                dp_reductions: 0,
                parallelized: false,
            };
        }

        let should_parallelize = || {
            let volume = expected;
            volume >= PARALLEL_MIN_VOLUME
                && self.rows >= PARALLEL_ROW_THRESHOLD
                && current_num_threads() > 1
        };

        match mode {
            HardmaxMode::SoftmaxAndMask => {
                let mut softmax = vec![0.0; expected];
                let mut dp_reductions = 0usize;
                let mut parallelized = false;

                if should_parallelize() {
                    parallelized = true;
                    let reductions = AtomicUsize::new(0);
                    self.input
                        .par_chunks(self.cols)
                        .zip(softmax.par_chunks_mut(self.cols))
                        .zip(hardmax.par_chunks_mut(self.cols))
                        .for_each(|((input_row, soft_row), hard_row)| {
                            let dp = compute_softmax_row(input_row, soft_row, hard_row);
                            reductions.fetch_add(dp, Ordering::Relaxed);
                        });
                    dp_reductions = reductions.load(Ordering::Relaxed);
                } else {
                    for row in 0..self.rows {
                        let offset = row * self.cols;
                        let input_row = &self.input[offset..offset + self.cols];
                        let soft_row = &mut softmax[offset..offset + self.cols];
                        let hard_row = &mut hardmax[offset..offset + self.cols];
                        dp_reductions += compute_softmax_row(input_row, soft_row, hard_row);
                    }
                }

                HardmaxEinsumResult {
                    softmax: Some(softmax),
                    hardmax,
                    dp_reductions,
                    parallelized,
                }
            }
            HardmaxMode::MaskOnly => {
                let mut dp_reductions = 0usize;
                let mut parallelized = false;

                if should_parallelize() {
                    parallelized = true;
                    let reductions = AtomicUsize::new(0);
                    self.input
                        .par_chunks(self.cols)
                        .zip(hardmax.par_chunks_mut(self.cols))
                        .for_each(|(input_row, hard_row)| {
                            let dp = compute_mask_row(input_row, hard_row);
                            reductions.fetch_add(dp, Ordering::Relaxed);
                        });
                    dp_reductions = reductions.load(Ordering::Relaxed);
                } else {
                    for row in 0..self.rows {
                        let offset = row * self.cols;
                        let input_row = &self.input[offset..offset + self.cols];
                        let hard_row = &mut hardmax[offset..offset + self.cols];
                        dp_reductions =
                            dp_reductions.saturating_add(compute_mask_row(input_row, hard_row));
                    }
                }

                HardmaxEinsumResult {
                    softmax: None,
                    hardmax,
                    dp_reductions,
                    parallelized,
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
struct HardmaxEinsumResult {
    softmax: Option<Vec<f32>>,
    hardmax: Vec<f32>,
    dp_reductions: usize,
    parallelized: bool,
}

fn compute_softmax_row(input_row: &[f32], soft_row: &mut [f32], hard_row: &mut [f32]) -> usize {
    debug_assert_eq!(input_row.len(), soft_row.len());
    debug_assert_eq!(input_row.len(), hard_row.len());

    soft_row.fill(0.0);
    hard_row.fill(0.0);

    let mut row_max = f32::NEG_INFINITY;
    let mut argmax_index: Option<usize> = None;
    let mut finite_values = 0usize;

    for (index, &value) in input_row.iter().enumerate() {
        if value.is_nan() {
            continue;
        }

        finite_values += 1;
        if value > row_max || argmax_index.is_none() {
            row_max = value;
            argmax_index = Some(index);
        }
    }

    if finite_values == 0 {
        return 0;
    }

    let mut sum = 0.0f32;
    for (prob_slot, &value) in soft_row.iter_mut().zip(input_row.iter()) {
        if value.is_nan() {
            *prob_slot = 0.0;
            continue;
        }

        let shifted = if row_max.is_infinite() {
            if value == row_max {
                1.0
            } else {
                0.0
            }
        } else {
            (value - row_max).exp()
        };

        sum += shifted;
        *prob_slot = shifted;
    }

    let inv_sum = if sum.is_finite() && sum > 0.0 {
        sum.recip()
    } else {
        0.0
    };

    if inv_sum > 0.0 {
        for prob in soft_row.iter_mut() {
            *prob *= inv_sum;
        }
    } else {
        soft_row.fill(0.0);
    }

    if let Some(idx) = argmax_index {
        if let Some(slot) = hard_row.get_mut(idx) {
            *slot = 1.0;
        }
    }

    finite_values.saturating_sub(1)
}

fn compute_mask_row(input_row: &[f32], hard_row: &mut [f32]) -> usize {
    debug_assert_eq!(input_row.len(), hard_row.len());

    hard_row.fill(0.0);

    if input_row.is_empty() {
        return 0;
    }

    let mut row_max = f32::NEG_INFINITY;
    let mut argmax_index: Option<usize> = None;
    let mut finite_values = 0usize;
    for (index, &value) in input_row.iter().enumerate() {
        if value.is_nan() {
            continue;
        }

        finite_values += 1;
        if value > row_max || argmax_index.is_none() {
            row_max = value;
            argmax_index = Some(index);
        }
    }

    if let Some(idx) = argmax_index {
        if let Some(slot) = hard_row.get_mut(idx) {
            *slot = 1.0;
        }
    }

    finite_values.saturating_sub(1)
}
