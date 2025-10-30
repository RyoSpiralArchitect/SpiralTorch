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

const DP_CPU_THRESHOLD: usize = 4_096;

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
        Ok(HardmaxFusionResult {
            softmax: result.softmax,
            hardmax: result.hardmax,
            dp_reductions: result.dp_reductions,
            einsum: einsum_signature,
            fused_ops,
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
            };
        }

        match mode {
            HardmaxMode::SoftmaxAndMask => {
                let mut softmax = vec![0.0; expected];
                let mut dp_reductions = 0usize;

                for row in 0..self.rows {
                    let offset = row * self.cols;
                    let mut row_max = f32::NEG_INFINITY;
                    let mut sum = 0.0f32;

                    for c in 0..self.cols {
                        let value = self.input[offset + c];
                        if value > row_max {
                            if row_max.is_finite() {
                                let scale = (row_max - value).exp();
                                sum = sum * scale + 1.0;
                                dp_reductions += 1;
                            } else {
                                sum = 1.0;
                            }
                            row_max = value;
                        } else {
                            sum += (value - row_max).exp();
                        }
                    }

                    let inv_sum = if sum.is_finite() && sum > f32::EPSILON {
                        sum.recip()
                    } else {
                        0.0
                    };

                    for c in 0..self.cols {
                        let value = self.input[offset + c];
                        let prob = ((value - row_max).exp()) * inv_sum;
                        softmax[offset + c] = prob;
                        hardmax[offset + c] = if value == row_max { 1.0 } else { 0.0 };
                    }
                }

                HardmaxEinsumResult {
                    softmax: Some(softmax),
                    hardmax,
                    dp_reductions,
                }
            }
            HardmaxMode::MaskOnly => {
                let mut dp_reductions = 0usize;

                for row in 0..self.rows {
                    let offset = row * self.cols;
                    let mut row_max = f32::NEG_INFINITY;
                    for c in 0..self.cols {
                        row_max = row_max.max(self.input[offset + c]);
                    }

                    for c in 0..self.cols {
                        let value = self.input[offset + c];
                        hardmax[offset + c] = if value == row_max { 1.0 } else { 0.0 };
                    }

                    dp_reductions = dp_reductions.saturating_add(self.cols.saturating_sub(1));
                }

                HardmaxEinsumResult {
                    softmax: None,
                    hardmax,
                    dp_reductions,
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
}
