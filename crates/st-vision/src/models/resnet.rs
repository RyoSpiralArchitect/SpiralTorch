// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use st_nn::io;
use st_nn::layers::activation::Relu;
use st_nn::layers::conv::{AvgPool2d, Conv2d, MaxPool2d};
use st_nn::layers::normalization::LayerNorm;
use st_nn::module::{Module, Parameter};
use st_nn::PureResult;
use st_tensor::{Tensor, TensorError};

pub fn conv_output_hw(
    input_hw: (usize, usize),
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> PureResult<(usize, usize)> {
    let (h, w) = input_hw;
    if h == 0 || w == 0 {
        return Err(TensorError::InvalidDimensions { rows: h, cols: w });
    }
    let effective_kh = (kernel.0 - 1) * dilation.0 + 1;
    let effective_kw = (kernel.1 - 1) * dilation.1 + 1;
    if h + 2 * padding.0 < effective_kh || w + 2 * padding.1 < effective_kw {
        return Err(TensorError::InvalidDimensions {
            rows: h + 2 * padding.0,
            cols: effective_kh.max(effective_kw),
        });
    }
    let oh = (h + 2 * padding.0 - effective_kh) / stride.0 + 1;
    let ow = (w + 2 * padding.1 - effective_kw) / stride.1 + 1;
    Ok((oh, ow))
}

pub fn pool_output_hw(
    input_hw: (usize, usize),
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> PureResult<(usize, usize)> {
    let (h, w) = input_hw;
    if h == 0 || w == 0 {
        return Err(TensorError::InvalidDimensions { rows: h, cols: w });
    }
    if h + 2 * padding.0 < kernel.0 || w + 2 * padding.1 < kernel.1 {
        return Err(TensorError::InvalidDimensions {
            rows: h + 2 * padding.0,
            cols: kernel.0.max(kernel.1),
        });
    }
    let oh = (h + 2 * padding.0 - kernel.0) / stride.0 + 1;
    let ow = (w + 2 * padding.1 - kernel.1) / stride.1 + 1;
    Ok((oh, ow))
}

#[derive(Clone, Debug)]
pub struct ResNetConfig {
    pub input_channels: usize,
    pub input_hw: (usize, usize),
    pub stage_channels: Vec<usize>,
    pub block_depths: Vec<usize>,
    pub stem_kernel: (usize, usize),
    pub stem_stride: (usize, usize),
    pub stem_padding: (usize, usize),
    pub curvature: f32,
    pub epsilon: f32,
    pub use_max_pool: bool,
    pub global_pool: bool,
    pub skip_init: f32,
    pub skip_learnable: bool,
    pub skip_slip: Option<SkipSlipSchedule>,
}

impl Default for ResNetConfig {
    fn default() -> Self {
        Self {
            input_channels: 3,
            input_hw: (224, 224),
            stage_channels: vec![64, 128, 256, 512],
            block_depths: vec![2, 2, 2, 2],
            stem_kernel: (7, 7),
            stem_stride: (2, 2),
            stem_padding: (3, 3),
            curvature: -1.0,
            epsilon: 1.0e-5,
            use_max_pool: true,
            global_pool: true,
            skip_init: 1.0,
            skip_learnable: false,
            skip_slip: None,
        }
    }
}

impl ResNetConfig {
    /// Returns a CIFAR-oriented ResNet-56 configuration with optional learnable skip scaling.
    pub fn resnet56_cifar(skip_learnable: bool) -> Self {
        let skip_slip = if skip_learnable {
            Some(
                SkipSlipSchedule::linear(0.35, 1.0)
                    .with_power(1.5)
                    .per_stage(),
            )
        } else {
            None
        };
        Self {
            input_channels: 3,
            input_hw: (32, 32),
            stage_channels: vec![16, 32, 64],
            block_depths: vec![9, 9, 9],
            stem_kernel: (3, 3),
            stem_stride: (1, 1),
            stem_padding: (1, 1),
            curvature: -1.0,
            epsilon: 1.0e-5,
            use_max_pool: false,
            global_pool: true,
            skip_init: 1.0,
            skip_learnable,
            skip_slip,
        }
    }

    /// Applies a skip slip schedule to the configuration.
    pub fn with_skip_slip(mut self, schedule: SkipSlipSchedule) -> Self {
        self.skip_slip = Some(schedule);
        self
    }
}

#[derive(Clone, Debug)]
pub struct SkipSlipSchedule {
    start: f32,
    end: f32,
    power: f32,
    per_stage: bool,
}

#[derive(Clone, Copy)]
enum IdentityBlend<'a> {
    Global(f32),
    PerStage(&'a [f32]),
}

impl SkipSlipSchedule {
    pub fn linear(start: f32, end: f32) -> Self {
        Self {
            start,
            end,
            power: 1.0,
            per_stage: false,
        }
    }

    pub fn constant(value: f32) -> Self {
        Self::linear(value, value)
    }

    pub fn with_power(mut self, power: f32) -> Self {
        self.power = power;
        self
    }

    pub fn per_stage(mut self) -> Self {
        self.per_stage = true;
        self
    }

    /// Returns the slip factors that would be applied to a sequence of stages.
    ///
    /// Each element in the returned vector corresponds to a stage and contains
    /// one slip factor per residual block in that stage. The schedule is
    /// validated before previewing.
    pub fn preview(&self, block_depths: &[usize]) -> PureResult<Vec<Vec<f32>>> {
        self.preview_inner(block_depths, None)
    }

    /// Returns the slip factors after blending the schedule with the identity bridge.
    ///
    /// A blend progress of `0.0` keeps all residual paths at identity strength,
    /// while `1.0` applies the full schedule. Values outside the unit interval are
    /// clamped for convenience.
    pub fn preview_with_identity_blend(
        &self,
        block_depths: &[usize],
        identity_progress: f32,
    ) -> PureResult<Vec<Vec<f32>>> {
        let progress = Self::normalise_identity_progress(identity_progress)?;
        self.preview_inner(block_depths, Some(IdentityBlend::Global(progress)))
    }

    /// Returns slip factors blended with per-stage identity progress markers.
    ///
    /// Each entry in `stage_progress` corresponds to one stage in `block_depths` and will
    /// be clamped to the unit interval. Missing trailing values default to the last
    /// provided stage when applying the blend.
    pub fn preview_with_stage_identity_blend(
        &self,
        block_depths: &[usize],
        stage_progress: &[f32],
    ) -> PureResult<Vec<Vec<f32>>> {
        let progress = Self::normalise_stage_identity_progress(stage_progress)?;
        self.preview_inner(block_depths, Some(IdentityBlend::PerStage(&progress)))
    }

    fn preview_inner(
        &self,
        block_depths: &[usize],
        identity_progress: Option<IdentityBlend<'_>>,
    ) -> PureResult<Vec<Vec<f32>>> {
        self.validate()?;
        let total_blocks: usize = block_depths.iter().sum();
        let mut preview = Vec::with_capacity(block_depths.len());
        let mut global_block_idx = 0usize;
        for (stage_idx, &depth) in block_depths.iter().enumerate() {
            let mut stage_factors = Vec::with_capacity(depth);
            for block_idx in 0..depth {
                let slip = self.factor_internal(
                    stage_idx,
                    block_idx,
                    depth,
                    global_block_idx,
                    total_blocks,
                );
                let factor = match identity_progress {
                    Some(IdentityBlend::Global(progress)) => {
                        Self::blend_with_identity(slip, progress)
                    }
                    Some(IdentityBlend::PerStage(per_stage)) => {
                        let progress = Self::stage_identity_progress(per_stage, stage_idx);
                        Self::blend_with_identity(slip, progress)
                    }
                    None => slip,
                };
                stage_factors.push(factor);
                global_block_idx += 1;
            }
            preview.push(stage_factors);
        }
        Ok(preview)
    }

    fn validate(&self) -> PureResult<()> {
        if !self.start.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "resnet_skip_slip_start",
                value: self.start,
            });
        }
        if !self.end.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "resnet_skip_slip_end",
                value: self.end,
            });
        }
        if !self.power.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "resnet_skip_slip_power",
                value: self.power,
            });
        }
        if self.start < 0.0 || self.end < 0.0 {
            return Err(TensorError::InvalidValue {
                label: "resnet_skip_slip_range",
            });
        }
        if self.power <= 0.0 {
            return Err(TensorError::InvalidValue {
                label: "resnet_skip_slip_power",
            });
        }
        Ok(())
    }

    fn factor_internal(
        &self,
        _stage_idx: usize,
        block_idx: usize,
        stage_depth: usize,
        global_block_idx: usize,
        total_blocks: usize,
    ) -> f32 {
        let numerator = if self.per_stage {
            block_idx as f32
        } else {
            global_block_idx as f32
        };
        let denominator = if self.per_stage {
            if stage_depth > 1 {
                (stage_depth - 1) as f32
            } else {
                1.0
            }
        } else if total_blocks > 1 {
            (total_blocks - 1) as f32
        } else {
            1.0
        };
        let progress = if denominator <= f32::EPSILON {
            0.0
        } else {
            (numerator / denominator).clamp(0.0, 1.0)
        };
        let powered = progress.powf(self.power);
        let slip = self.start + (self.end - self.start) * powered;
        if slip <= 0.0 {
            0.0
        } else {
            slip
        }
    }

    fn blend_with_identity(slip: f32, progress: f32) -> f32 {
        if !progress.is_finite() {
            return slip;
        }
        let progress = progress.clamp(0.0, 1.0);
        1.0 + (slip - 1.0) * progress
    }

    fn normalise_identity_progress(progress: f32) -> PureResult<f32> {
        if !progress.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "resnet_skip_slip_identity_progress",
                value: progress,
            });
        }
        Ok(progress.clamp(0.0, 1.0))
    }

    fn normalise_stage_identity_progress(progress: &[f32]) -> PureResult<Vec<f32>> {
        if progress.is_empty() {
            return Err(TensorError::InvalidValue {
                label: "resnet_skip_slip_identity_stage_progress",
            });
        }
        let mut normalised = Vec::with_capacity(progress.len());
        for &value in progress {
            if !value.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "resnet_skip_slip_identity_stage_progress",
                    value,
                });
            }
            normalised.push(value.clamp(0.0, 1.0));
        }
        Ok(normalised)
    }

    fn stage_identity_progress(progress: &[f32], stage_idx: usize) -> f32 {
        match progress.get(stage_idx).copied() {
            Some(value) => value,
            None => *progress
                .last()
                .expect("stage identity progress normalised to non-empty"),
        }
    }

    fn factor(
        &self,
        stage_idx: usize,
        block_idx: usize,
        stage_depth: usize,
        global_block_idx: usize,
        total_blocks: usize,
    ) -> f32 {
        self.factor_internal(
            stage_idx,
            block_idx,
            stage_depth,
            global_block_idx,
            total_blocks,
        )
    }

    fn factor_with_identity_blend(
        &self,
        stage_idx: usize,
        block_idx: usize,
        stage_depth: usize,
        global_block_idx: usize,
        total_blocks: usize,
        identity_progress: f32,
    ) -> f32 {
        let slip = self.factor_internal(
            stage_idx,
            block_idx,
            stage_depth,
            global_block_idx,
            total_blocks,
        );
        Self::blend_with_identity(slip, identity_progress)
    }

    fn factor_with_stage_identity_blend(
        &self,
        stage_idx: usize,
        block_idx: usize,
        stage_depth: usize,
        global_block_idx: usize,
        total_blocks: usize,
        stage_progress: &[f32],
    ) -> f32 {
        let slip = self.factor_internal(
            stage_idx,
            block_idx,
            stage_depth,
            global_block_idx,
            total_blocks,
        );
        let progress = Self::stage_identity_progress(stage_progress, stage_idx);
        Self::blend_with_identity(slip, progress)
    }

    #[allow(dead_code)]
    fn description(&self) -> String {
        let scope = if self.per_stage {
            "per-stage"
        } else {
            "global"
        };
        format!(
            "{scope} slip: start={:.3}, end={:.3}, power={:.3}",
            self.start, self.end, self.power
        )
    }
}

#[derive(Debug)]
enum SkipStyle {
    Identity,
    Fixed(f32),
    Learnable(Parameter),
}

impl SkipStyle {
    fn from_config(name: &str, skip_init: f32, skip_learnable: bool) -> PureResult<Self> {
        if skip_learnable {
            let tensor = Tensor::from_vec(1, 1, vec![skip_init])?;
            Ok(Self::Learnable(Parameter::new(
                format!("{name}.skip_gate"),
                tensor,
            )))
        } else if (skip_init - 1.0).abs() > f32::EPSILON {
            Ok(Self::Fixed(skip_init))
        } else {
            Ok(Self::Identity)
        }
    }

    fn apply_forward(&self, residual: &Tensor) -> PureResult<Tensor> {
        match self {
            SkipStyle::Identity => Ok(residual.clone()),
            SkipStyle::Fixed(scale) => residual.scale(*scale),
            SkipStyle::Learnable(param) => {
                let scale = param.value().data()[0];
                residual.scale(scale)
            }
        }
    }

    fn propagate_backward(
        &mut self,
        residual_pre: &Tensor,
        grad_scaled: &Tensor,
    ) -> PureResult<Tensor> {
        match self {
            SkipStyle::Identity => Ok(grad_scaled.clone()),
            SkipStyle::Fixed(scale) => grad_scaled.scale(*scale),
            SkipStyle::Learnable(param) => {
                let scale = param.value().data()[0];
                let grad_residual = grad_scaled.scale(scale)?;
                let grad_value = grad_scaled
                    .data()
                    .iter()
                    .zip(residual_pre.data().iter())
                    .map(|(g, r)| g * r)
                    .sum::<f32>();
                let grad_tensor = Tensor::from_vec(1, 1, vec![grad_value])?;
                param.accumulate_euclidean(&grad_tensor)?;
                Ok(grad_residual)
            }
        }
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        if let SkipStyle::Learnable(param) = self {
            visitor(param)?;
        }
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        if let SkipStyle::Learnable(param) = self {
            visitor(param)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct ResNetBlock {
    conv1: Conv2d,
    norm1: LayerNorm,
    conv2: Conv2d,
    norm2: LayerNorm,
    activation1: Relu,
    activation2: Relu,
    downsample: Option<(Conv2d, LayerNorm)>,
    output_hw: (usize, usize),
    skip_style: SkipStyle,
    slip_factor: f32,
}

impl ResNetBlock {
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: &str,
        in_channels: usize,
        out_channels: usize,
        stride: (usize, usize),
        input_hw: (usize, usize),
        curvature: f32,
        epsilon: f32,
        skip_init: f32,
        skip_learnable: bool,
        slip_factor: f32,
    ) -> PureResult<Self> {
        let conv1 = Conv2d::new(
            format!("{name}.conv1"),
            in_channels,
            out_channels,
            (3, 3),
            stride,
            (1, 1),
            (1, 1),
            input_hw,
        )?;
        let conv1_hw = conv_output_hw(input_hw, (3, 3), stride, (1, 1), (1, 1))?;
        let norm1 = LayerNorm::new(
            format!("{name}.ln1"),
            out_channels * conv1_hw.0 * conv1_hw.1,
            curvature,
            epsilon,
        )?;
        let activation1 = Relu::new();
        let conv2 = Conv2d::new(
            format!("{name}.conv2"),
            out_channels,
            out_channels,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            conv1_hw,
        )?;
        let conv2_hw = conv_output_hw(conv1_hw, (3, 3), (1, 1), (1, 1), (1, 1))?;
        let norm2 = LayerNorm::new(
            format!("{name}.ln2"),
            out_channels * conv2_hw.0 * conv2_hw.1,
            curvature,
            epsilon,
        )?;
        let activation2 = Relu::new();
        let downsample = if stride != (1, 1) || in_channels != out_channels {
            let down_conv = Conv2d::new(
                format!("{name}.downsample"),
                in_channels,
                out_channels,
                (1, 1),
                stride,
                (0, 0),
                (1, 1),
                input_hw,
            )?;
            let down_hw = conv_output_hw(input_hw, (1, 1), stride, (0, 0), (1, 1))?;
            let down_norm = LayerNorm::new(
                format!("{name}.down_ln"),
                out_channels * down_hw.0 * down_hw.1,
                curvature,
                epsilon,
            )?;
            Some((down_conv, down_norm))
        } else {
            None
        };
        if !slip_factor.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "resnet_skip_slip_factor",
                value: slip_factor,
            });
        }
        if slip_factor < 0.0 {
            return Err(TensorError::InvalidValue {
                label: "resnet_skip_slip_factor",
            });
        }
        let skip_style = SkipStyle::from_config(name, skip_init, skip_learnable)?;
        Ok(Self {
            conv1,
            norm1,
            conv2,
            norm2,
            activation1,
            activation2,
            downsample,
            output_hw: conv2_hw,
            skip_style,
            slip_factor,
        })
    }

    fn output_hw(&self) -> (usize, usize) {
        self.output_hw
    }

    fn forward_impl(&self, input: &Tensor) -> PureResult<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let residual = if let Some((down_conv, down_norm)) = &self.downsample {
            let down = down_conv.forward(input)?;
            let down = down_norm.forward(&down)?;
            down
        } else {
            input.clone()
        };
        let residual = self.skip_style.apply_forward(&residual)?;
        let residual = if (self.slip_factor - 1.0).abs() > f32::EPSILON {
            residual.scale(self.slip_factor)?
        } else {
            residual
        };
        let conv1_out = self.conv1.forward(input)?;
        let norm1_out = self.norm1.forward(&conv1_out)?;
        let act1_out = self.activation1.forward(&norm1_out)?;
        let conv2_out = self.conv2.forward(&act1_out)?;
        let norm2_out = self.norm2.forward(&conv2_out)?;
        let summed = norm2_out.add(&residual)?;
        let out = self.activation2.forward(&summed)?;
        Ok((out, summed, conv2_out, act1_out, conv1_out))
    }

    fn slip_factor(&self) -> f32 {
        self.slip_factor
    }

    fn set_slip_factor(&mut self, slip: f32) -> PureResult<()> {
        if !slip.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "resnet_skip_slip_factor",
                value: slip,
            });
        }
        if slip < 0.0 {
            return Err(TensorError::InvalidValue {
                label: "resnet_skip_slip_factor",
            });
        }
        self.slip_factor = slip;
        Ok(())
    }
}

impl Module for ResNetBlock {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (out, _, _, _, _) = self.forward_impl(input)?;
        Ok(out)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let residual_pre = if let Some((down_conv, down_norm)) = &self.downsample {
            let down = down_conv.forward(input)?;
            down_norm.forward(&down)?
        } else {
            input.clone()
        };
        let conv1_out = self.conv1.forward(input)?;
        let norm1_out = self.norm1.forward(&conv1_out)?;
        let act1_out = self.activation1.forward(&norm1_out)?;
        let conv2_out = self.conv2.forward(&act1_out)?;
        let norm2_out = self.norm2.forward(&conv2_out)?;
        let residual_scaled = self.skip_style.apply_forward(&residual_pre)?;
        let residual_slipped = if (self.slip_factor - 1.0).abs() > f32::EPSILON {
            residual_scaled.scale(self.slip_factor)?
        } else {
            residual_scaled.clone()
        };
        let summed = norm2_out.add(&residual_slipped)?;
        let grad = self.activation2.backward(&summed, grad_output)?;
        let grad_residual_slipped = grad.clone();
        let mut grad_main = grad;
        grad_main = self.norm2.backward(&conv2_out, &grad_main)?;
        grad_main = self.conv2.backward(&act1_out, &grad_main)?;
        grad_main = self.activation1.backward(&norm1_out, &grad_main)?;
        grad_main = self.norm1.backward(&conv1_out, &grad_main)?;
        let mut grad_input = self.conv1.backward(input, &grad_main)?;
        let grad_residual_scaled = if (self.slip_factor - 1.0).abs() > f32::EPSILON {
            grad_residual_slipped.scale(self.slip_factor)?
        } else {
            grad_residual_slipped.clone()
        };
        let grad_residual_pre = self
            .skip_style
            .propagate_backward(&residual_pre, &grad_residual_scaled)?;
        if let Some((down_conv, down_norm)) = &mut self.downsample {
            let down_out = down_conv.forward(input)?;
            let grad_down = down_norm.backward(&down_out, &grad_residual_pre)?;
            let grad_skip = down_conv.backward(input, &grad_down)?;
            grad_input = grad_input.add(&grad_skip)?;
        } else {
            grad_input = grad_input.add(&grad_residual_pre)?;
        }
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.conv1.visit_parameters(visitor)?;
        self.norm1.visit_parameters(visitor)?;
        self.conv2.visit_parameters(visitor)?;
        self.norm2.visit_parameters(visitor)?;
        self.skip_style.visit_parameters(visitor)?;
        if let Some((down_conv, down_norm)) = &self.downsample {
            down_conv.visit_parameters(visitor)?;
            down_norm.visit_parameters(visitor)?;
        }
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.conv1.visit_parameters_mut(visitor)?;
        self.norm1.visit_parameters_mut(visitor)?;
        self.conv2.visit_parameters_mut(visitor)?;
        self.norm2.visit_parameters_mut(visitor)?;
        self.skip_style.visit_parameters_mut(visitor)?;
        if let Some((down_conv, down_norm)) = &mut self.downsample {
            down_conv.visit_parameters_mut(visitor)?;
            down_norm.visit_parameters_mut(visitor)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct ResNetBackbone {
    stem_conv: Conv2d,
    stem_norm: LayerNorm,
    stem_activation: Relu,
    stem_pool: Option<MaxPool2d>,
    blocks: Vec<ResNetBlock>,
    global_pool: Option<AvgPool2d>,
    output_channels: usize,
    output_hw: (usize, usize),
    tokens_per_stage: Vec<(usize, (usize, usize))>,
    block_depths: Vec<usize>,
}

impl ResNetBackbone {
    pub fn new(config: ResNetConfig) -> PureResult<Self> {
        if config.stage_channels.is_empty() {
            return Err(TensorError::EmptyInput("resnet_stage_channels"));
        }
        if config.stage_channels.len() != config.block_depths.len() {
            return Err(TensorError::InvalidDimensions {
                rows: config.stage_channels.len(),
                cols: config.block_depths.len(),
            });
        }
        if config.curvature >= 0.0 || !config.curvature.is_finite() {
            return Err(TensorError::NonHyperbolicCurvature {
                curvature: config.curvature,
            });
        }
        if config.epsilon <= 0.0 || !config.epsilon.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "resnet_layernorm_epsilon",
                value: config.epsilon,
            });
        }
        if !config.skip_init.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "resnet_skip_init",
                value: config.skip_init,
            });
        }
        if let Some(schedule) = &config.skip_slip {
            schedule.validate()?;
        }
        let stem_out_channels = config.stage_channels[0];
        let stem_conv = Conv2d::new(
            "resnet.stem",
            config.input_channels,
            stem_out_channels,
            config.stem_kernel,
            config.stem_stride,
            config.stem_padding,
            (1, 1),
            config.input_hw,
        )?;
        let mut tokens_per_stage = Vec::new();
        let mut current_hw = conv_output_hw(
            config.input_hw,
            config.stem_kernel,
            config.stem_stride,
            config.stem_padding,
            (1, 1),
        )?;
        let stem_norm = LayerNorm::new(
            "resnet.stem_ln",
            stem_out_channels * current_hw.0 * current_hw.1,
            config.curvature,
            config.epsilon,
        )?;
        let stem_activation = Relu::new();
        let stem_pool = if config.use_max_pool {
            let pool = MaxPool2d::new(stem_out_channels, (3, 3), (2, 2), (1, 1), current_hw)?;
            current_hw = pool_output_hw(current_hw, (3, 3), (2, 2), (1, 1))?;
            Some(pool)
        } else {
            None
        };
        let mut blocks = Vec::new();
        let mut current_channels = stem_out_channels;
        let total_blocks: usize = config.block_depths.iter().sum();
        let mut global_block_idx = 0usize;
        for (stage_idx, (&channels, &depth)) in config
            .stage_channels
            .iter()
            .zip(config.block_depths.iter())
            .enumerate()
        {
            for block_idx in 0..depth {
                let stride = if stage_idx > 0 && block_idx == 0 {
                    (2, 2)
                } else {
                    (1, 1)
                };
                let slip_factor = config
                    .skip_slip
                    .as_ref()
                    .map(|schedule| {
                        schedule.factor(stage_idx, block_idx, depth, global_block_idx, total_blocks)
                    })
                    .unwrap_or(1.0);
                let block = ResNetBlock::new(
                    &format!("resnet.stage{stage_idx}.block{block_idx}"),
                    current_channels,
                    channels,
                    stride,
                    current_hw,
                    config.curvature,
                    config.epsilon,
                    config.skip_init,
                    config.skip_learnable,
                    slip_factor,
                )?;
                tokens_per_stage.push((channels, block.output_hw()));
                current_hw = block.output_hw();
                current_channels = channels;
                blocks.push(block);
                global_block_idx += 1;
            }
        }
        let global_pool = if config.global_pool {
            let pool = AvgPool2d::new(current_channels, current_hw, (1, 1), (0, 0), current_hw)?;
            current_hw = (1, 1);
            Some(pool)
        } else {
            None
        };
        Ok(Self {
            stem_conv,
            stem_norm,
            stem_activation,
            stem_pool,
            blocks,
            global_pool,
            output_channels: current_channels,
            output_hw: current_hw,
            tokens_per_stage,
            block_depths: config.block_depths.clone(),
        })
    }

    pub fn output_channels(&self) -> usize {
        self.output_channels
    }

    pub fn output_hw(&self) -> (usize, usize) {
        self.output_hw
    }

    pub fn output_features(&self) -> usize {
        self.output_channels * self.output_hw.0 * self.output_hw.1
    }

    pub fn stage_shapes(&self) -> &Vec<(usize, (usize, usize))> {
        &self.tokens_per_stage
    }

    pub fn block_depths(&self) -> &[usize] {
        &self.block_depths
    }

    pub fn skip_slip_factors(&self) -> Vec<Vec<f32>> {
        let mut factors = Vec::with_capacity(self.block_depths.len());
        let mut global_idx = 0usize;
        for &depth in &self.block_depths {
            let mut stage = Vec::with_capacity(depth);
            for _ in 0..depth {
                let block = self
                    .blocks
                    .get(global_idx)
                    .expect("skip slip factors align with block layout");
                stage.push(block.slip_factor());
                global_idx += 1;
            }
            factors.push(stage);
        }
        factors
    }

    pub fn set_skip_slip(&mut self, schedule: Option<SkipSlipSchedule>) -> PureResult<()> {
        self.apply_skip_slip(schedule, None, None)
    }

    /// Applies a skip slip schedule with a blend factor against the identity bridge.
    ///
    /// A progress value of `0.0` leaves all residual paths at identity strength,
    /// while `1.0` applies the schedule exactly. Intermediate values smoothly blend
    /// between the two, enabling annealing over the course of training.
    pub fn set_skip_slip_progress(
        &mut self,
        schedule: Option<SkipSlipSchedule>,
        identity_progress: f32,
    ) -> PureResult<()> {
        self.apply_skip_slip(schedule, Some(identity_progress), None)
    }

    /// Applies a skip slip schedule with per-stage identity blend progress markers.
    ///
    /// The provided slice may be shorter than the number of stages; the final value will
    /// be re-used for any additional stages. Entries are clamped to the unit interval.
    pub fn set_skip_slip_stage_progress(
        &mut self,
        schedule: Option<SkipSlipSchedule>,
        stage_progress: &[f32],
    ) -> PureResult<()> {
        self.apply_skip_slip(schedule, None, Some(stage_progress))
    }

    fn apply_skip_slip(
        &mut self,
        schedule: Option<SkipSlipSchedule>,
        identity_progress: Option<f32>,
        stage_progress: Option<&[f32]>,
    ) -> PureResult<()> {
        if identity_progress.is_some() && stage_progress.is_some() {
            return Err(TensorError::InvalidValue {
                label: "resnet_skip_slip_identity_blend_conflict",
            });
        }
        match schedule {
            Some(schedule) => {
                schedule.validate()?;
                let blend = match identity_progress {
                    Some(value) => Some(SkipSlipSchedule::normalise_identity_progress(value)?),
                    None => None,
                };
                let stage_blend = match stage_progress {
                    Some(values) => {
                        Some(SkipSlipSchedule::normalise_stage_identity_progress(values)?)
                    }
                    None => None,
                };
                let stage_blend_ref = stage_blend.as_deref();
                let total_blocks: usize = self.block_depths.iter().sum();
                debug_assert_eq!(self.blocks.len(), total_blocks);
                let mut global_block_idx = 0usize;
                let mut block_iter = self.blocks.iter_mut();
                for (stage_idx, &depth) in self.block_depths.iter().enumerate() {
                    for block_idx in 0..depth {
                        let slip = if let Some(per_stage) = stage_blend_ref {
                            schedule.factor_with_stage_identity_blend(
                                stage_idx,
                                block_idx,
                                depth,
                                global_block_idx,
                                total_blocks,
                                per_stage,
                            )
                        } else if let Some(progress) = blend {
                            schedule.factor_with_identity_blend(
                                stage_idx,
                                block_idx,
                                depth,
                                global_block_idx,
                                total_blocks,
                                progress,
                            )
                        } else {
                            schedule.factor(
                                stage_idx,
                                block_idx,
                                depth,
                                global_block_idx,
                                total_blocks,
                            )
                        };
                        let block = block_iter
                            .next()
                            .expect("skip slip schedule length matches blocks");
                        block.set_slip_factor(slip)?;
                        global_block_idx += 1;
                    }
                }
                debug_assert!(block_iter.next().is_none());
            }
            None => {
                for block in &mut self.blocks {
                    block.set_slip_factor(1.0)?;
                }
            }
        }
        Ok(())
    }

    pub fn load_weights_json<P: AsRef<std::path::Path>>(&mut self, path: P) -> PureResult<()> {
        io::load_json(self, path)
    }

    pub fn load_weights_bincode<P: AsRef<std::path::Path>>(&mut self, path: P) -> PureResult<()> {
        io::load_bincode(self, path)
    }
}

impl Module for ResNetBackbone {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let stem_out = self.stem_conv.forward(input)?;
        let stem_norm = self.stem_norm.forward(&stem_out)?;
        let mut activ = self.stem_activation.forward(&stem_norm)?;
        if let Some(pool) = &self.stem_pool {
            activ = pool.forward(&activ)?;
        }
        for block in &self.blocks {
            activ = block.forward(&activ)?;
        }
        if let Some(pool) = &self.global_pool {
            activ = pool.forward(&activ)?;
        }
        Ok(activ)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let stem_out = self.stem_conv.forward(input)?;
        let stem_norm = self.stem_norm.forward(&stem_out)?;
        let stem_act = self.stem_activation.forward(&stem_norm)?;
        let stem_pool_out = if let Some(pool) = &self.stem_pool {
            pool.forward(&stem_act)?
        } else {
            stem_act.clone()
        };
        let mut block_inputs = Vec::with_capacity(self.blocks.len());
        let mut activ = stem_pool_out.clone();
        for block in &self.blocks {
            block_inputs.push(activ.clone());
            activ = block.forward(&activ)?;
        }
        let mut grad = if let Some(pool) = &mut self.global_pool {
            pool.backward(&activ, grad_output)?
        } else {
            grad_output.clone()
        };
        for (block, block_input) in self
            .blocks
            .iter_mut()
            .rev()
            .zip(block_inputs.into_iter().rev())
        {
            grad = block.backward(&block_input, &grad)?;
        }
        if let Some(pool) = &mut self.stem_pool {
            grad = pool.backward(&stem_act, &grad)?;
        }
        grad = self.stem_activation.backward(&stem_norm, &grad)?;
        grad = self.stem_norm.backward(&stem_out, &grad)?;
        grad = self.stem_conv.backward(input, &grad)?;
        Ok(grad)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.stem_conv.visit_parameters(visitor)?;
        self.stem_norm.visit_parameters(visitor)?;
        for block in &self.blocks {
            block.visit_parameters(visitor)?;
        }
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.stem_conv.visit_parameters_mut(visitor)?;
        self.stem_norm.visit_parameters_mut(visitor)?;
        for block in &mut self.blocks {
            block.visit_parameters_mut(visitor)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct SkipBridge {
    projection: Conv2d,
    norm: LayerNorm,
    activation: Relu,
}

impl SkipBridge {
    fn new(
        name: &str,
        in_channels: usize,
        in_hw: (usize, usize),
        out_channels: usize,
        out_hw: (usize, usize),
        curvature: f32,
        epsilon: f32,
    ) -> PureResult<Self> {
        if out_hw.0 == 0 || out_hw.1 == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: out_hw.0,
                cols: out_hw.1,
            });
        }
        if in_hw.0 % out_hw.0 != 0 || in_hw.1 % out_hw.1 != 0 {
            return Err(TensorError::InvalidDimensions {
                rows: in_hw.0,
                cols: in_hw.1,
            });
        }
        let stride = (in_hw.0 / out_hw.0, in_hw.1 / out_hw.1);
        let projection = Conv2d::new(
            format!("{name}.proj"),
            in_channels,
            out_channels,
            (1, 1),
            stride,
            (0, 0),
            (1, 1),
            in_hw,
        )?;
        let projected_hw = conv_output_hw(in_hw, (1, 1), stride, (0, 0), (1, 1))?;
        if projected_hw != out_hw {
            return Err(TensorError::InvalidDimensions {
                rows: projected_hw.0,
                cols: projected_hw.1,
            });
        }
        let norm = LayerNorm::new(
            format!("{name}.ln"),
            out_channels * out_hw.0 * out_hw.1,
            curvature,
            epsilon,
        )?;
        let activation = Relu::new();
        Ok(Self {
            projection,
            norm,
            activation,
        })
    }

    fn forward_impl(&self, input: &Tensor) -> PureResult<(Tensor, Tensor, Tensor)> {
        let conv = self.projection.forward(input)?;
        let norm = self.norm.forward(&conv)?;
        let act = self.activation.forward(&norm)?;
        Ok((act, norm, conv))
    }
}

impl Module for SkipBridge {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (out, _, _) = self.forward_impl(input)?;
        Ok(out)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let conv = self.projection.forward(input)?;
        let norm = self.norm.forward(&conv)?;
        let mut grad = self.activation.backward(&norm, grad_output)?;
        grad = self.norm.backward(&conv, &grad)?;
        let grad_input = self.projection.backward(input, &grad)?;
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.projection.visit_parameters(visitor)?;
        self.norm.visit_parameters(visitor)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.projection.visit_parameters_mut(visitor)?;
        self.norm.visit_parameters_mut(visitor)?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ResNet56WithSkipConfig {
    pub base: ResNetConfig,
    pub skip_scale: f32,
}

impl Default for ResNet56WithSkipConfig {
    fn default() -> Self {
        let mut base = ResNetConfig::default();
        base.input_hw = (32, 32);
        base.stage_channels = vec![16, 32, 64];
        base.block_depths = vec![9, 9, 9];
        base.stem_kernel = (3, 3);
        base.stem_stride = (1, 1);
        base.stem_padding = (1, 1);
        base.use_max_pool = false;
        base.global_pool = true;
        Self {
            base,
            skip_scale: 0.5,
        }
    }
}

#[derive(Debug)]
pub struct ResNet56WithSkip {
    stem_conv: Conv2d,
    stem_norm: LayerNorm,
    stem_activation: Relu,
    stem_pool: Option<MaxPool2d>,
    stages: Vec<Vec<ResNetBlock>>,
    stage_shapes: Vec<(usize, (usize, usize))>,
    skip_bridges: Vec<SkipBridge>,
    fusion_norm: LayerNorm,
    fusion_activation: Relu,
    global_pool: Option<AvgPool2d>,
    output_channels: usize,
    output_hw: (usize, usize),
    skip_scale: f32,
}

impl ResNet56WithSkip {
    pub fn new(config: ResNet56WithSkipConfig) -> PureResult<Self> {
        if !config.skip_scale.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "resnet56_skip_scale",
                value: config.skip_scale,
            });
        }
        let base = config.base;
        if base.stage_channels.is_empty() {
            return Err(TensorError::EmptyInput("resnet56_stage_channels"));
        }
        if base.stage_channels.len() != base.block_depths.len() {
            return Err(TensorError::InvalidDimensions {
                rows: base.stage_channels.len(),
                cols: base.block_depths.len(),
            });
        }
        if let Some(schedule) = &base.skip_slip {
            schedule.validate()?;
        }
        let stem_out_channels = base.stage_channels[0];
        let stem_conv = Conv2d::new(
            "resnet56.stem",
            base.input_channels,
            stem_out_channels,
            base.stem_kernel,
            base.stem_stride,
            base.stem_padding,
            (1, 1),
            base.input_hw,
        )?;
        let mut current_hw = conv_output_hw(
            base.input_hw,
            base.stem_kernel,
            base.stem_stride,
            base.stem_padding,
            (1, 1),
        )?;
        let stem_norm = LayerNorm::new(
            "resnet56.stem_ln",
            stem_out_channels * current_hw.0 * current_hw.1,
            base.curvature,
            base.epsilon,
        )?;
        let stem_activation = Relu::new();
        let stem_pool = if base.use_max_pool {
            let pool = MaxPool2d::new(stem_out_channels, (3, 3), (2, 2), (1, 1), current_hw)?;
            current_hw = pool_output_hw(current_hw, (3, 3), (2, 2), (1, 1))?;
            Some(pool)
        } else {
            None
        };
        let mut stages = Vec::new();
        let mut stage_shapes = Vec::new();
        let mut current_channels = stem_out_channels;
        let total_blocks: usize = base.block_depths.iter().sum();
        let mut global_block_idx = 0usize;
        for (stage_idx, (&channels, &depth)) in base
            .stage_channels
            .iter()
            .zip(base.block_depths.iter())
            .enumerate()
        {
            let mut blocks = Vec::new();
            for block_idx in 0..depth {
                let stride = if stage_idx > 0 && block_idx == 0 {
                    (2, 2)
                } else {
                    (1, 1)
                };
                let slip_factor = base
                    .skip_slip
                    .as_ref()
                    .map(|schedule| {
                        schedule.factor(stage_idx, block_idx, depth, global_block_idx, total_blocks)
                    })
                    .unwrap_or(1.0);
                let block = ResNetBlock::new(
                    &format!("resnet56.stage{stage_idx}.block{block_idx}"),
                    current_channels,
                    channels,
                    stride,
                    current_hw,
                    base.curvature,
                    base.epsilon,
                    base.skip_init,
                    base.skip_learnable,
                    slip_factor,
                )?;
                current_hw = block.output_hw();
                current_channels = channels;
                blocks.push(block);
                global_block_idx += 1;
            }
            stage_shapes.push((current_channels, current_hw));
            stages.push(blocks);
        }
        let Some(&(final_channels, final_hw)) = stage_shapes.last() else {
            return Err(TensorError::EmptyInput("resnet56_stages"));
        };
        let fusion_norm = LayerNorm::new(
            "resnet56.fusion_ln",
            final_channels * final_hw.0 * final_hw.1,
            base.curvature,
            base.epsilon,
        )?;
        let fusion_activation = Relu::new();
        let mut skip_bridges = Vec::new();
        if stage_shapes.len() > 1 {
            for (idx, &(stage_channels, stage_hw)) in
                stage_shapes.iter().enumerate().take(stage_shapes.len() - 1)
            {
                let bridge = SkipBridge::new(
                    &format!("resnet56.skip.stage{idx}"),
                    stage_channels,
                    stage_hw,
                    final_channels,
                    final_hw,
                    base.curvature,
                    base.epsilon,
                )?;
                skip_bridges.push(bridge);
            }
        }
        let global_pool = if base.global_pool {
            let pool = AvgPool2d::new(final_channels, final_hw, (1, 1), (0, 0), final_hw)?;
            Some(pool)
        } else {
            None
        };
        let output_hw = if base.global_pool { (1, 1) } else { final_hw };
        Ok(Self {
            stem_conv,
            stem_norm,
            stem_activation,
            stem_pool,
            stages,
            stage_shapes,
            skip_bridges,
            fusion_norm,
            fusion_activation,
            global_pool,
            output_channels: final_channels,
            output_hw,
            skip_scale: config.skip_scale,
        })
    }

    pub fn output_channels(&self) -> usize {
        self.output_channels
    }

    pub fn output_hw(&self) -> (usize, usize) {
        self.output_hw
    }

    pub fn output_features(&self) -> usize {
        self.output_channels * self.output_hw.0 * self.output_hw.1
    }

    pub fn stage_shapes(&self) -> &Vec<(usize, (usize, usize))> {
        &self.stage_shapes
    }

    pub fn load_weights_json<P: AsRef<std::path::Path>>(&mut self, path: P) -> PureResult<()> {
        io::load_json(self, path)
    }

    pub fn load_weights_bincode<P: AsRef<std::path::Path>>(&mut self, path: P) -> PureResult<()> {
        io::load_bincode(self, path)
    }
}

impl Module for ResNet56WithSkip {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let stem_out = self.stem_conv.forward(input)?;
        let stem_norm = self.stem_norm.forward(&stem_out)?;
        let mut activ = self.stem_activation.forward(&stem_norm)?;
        if let Some(pool) = &self.stem_pool {
            activ = pool.forward(&activ)?;
        }
        let mut stage_outputs = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            for block in stage {
                activ = block.forward(&activ)?;
            }
            stage_outputs.push(activ.clone());
        }
        let mut fused = stage_outputs
            .last()
            .cloned()
            .ok_or(TensorError::EmptyInput("resnet56_forward_stage"))?;
        for (idx, bridge) in self.skip_bridges.iter().enumerate() {
            let skip = bridge.forward(&stage_outputs[idx])?;
            let scaled = skip.scale(self.skip_scale)?;
            fused = fused.add(&scaled)?;
        }
        let fusion_norm = self.fusion_norm.forward(&fused)?;
        let mut output = self.fusion_activation.forward(&fusion_norm)?;
        if let Some(pool) = &self.global_pool {
            output = pool.forward(&output)?;
        }
        Ok(output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let stem_out = self.stem_conv.forward(input)?;
        let stem_norm = self.stem_norm.forward(&stem_out)?;
        let stem_act = self.stem_activation.forward(&stem_norm)?;
        let stem_pool_out = if let Some(pool) = &self.stem_pool {
            pool.forward(&stem_act)?
        } else {
            stem_act.clone()
        };
        let mut stage_inputs: Vec<Vec<Tensor>> = Vec::with_capacity(self.stages.len());
        let mut stage_outputs: Vec<Tensor> = Vec::with_capacity(self.stages.len());
        let mut activ = stem_pool_out.clone();
        for stage in &self.stages {
            let mut block_inputs = Vec::with_capacity(stage.len());
            for block in stage {
                block_inputs.push(activ.clone());
                activ = block.forward(&activ)?;
            }
            stage_inputs.push(block_inputs);
            stage_outputs.push(activ.clone());
        }
        let mut fused = stage_outputs
            .last()
            .cloned()
            .ok_or(TensorError::EmptyInput("resnet56_backward_stage"))?;
        for (idx, bridge) in self.skip_bridges.iter().enumerate() {
            let skip = bridge.forward(&stage_outputs[idx])?;
            let scaled = skip.scale(self.skip_scale)?;
            fused = fused.add(&scaled)?;
        }
        let fusion_norm = self.fusion_norm.forward(&fused)?;
        let fusion_act = self.fusion_activation.forward(&fusion_norm)?;
        let mut grad = if let Some(pool) = &mut self.global_pool {
            pool.backward(&fusion_act, grad_output)?
        } else {
            grad_output.clone()
        };
        grad = self.fusion_activation.backward(&fusion_norm, &grad)?;
        grad = self.fusion_norm.backward(&fused, &grad)?;
        let stage_count = self.stages.len();
        let mut stage_output_grads: Vec<Option<Tensor>> = vec![None; stage_count];
        let last_idx = stage_count - 1;
        stage_output_grads[last_idx] = Some(grad.clone());
        for (idx, bridge) in self.skip_bridges.iter_mut().enumerate() {
            let grad_branch = grad.scale(self.skip_scale)?;
            let grad_in = bridge.backward(&stage_outputs[idx], &grad_branch)?;
            match &mut stage_output_grads[idx] {
                Some(existing) => {
                    *existing = existing.add(&grad_in)?;
                }
                None => {
                    stage_output_grads[idx] = Some(grad_in);
                }
            }
        }
        let mut next_grad = stage_output_grads[last_idx]
            .take()
            .ok_or(TensorError::EmptyInput("resnet56_final_grad"))?;
        for stage_idx in (0..stage_count).rev() {
            let blocks = &mut self.stages[stage_idx];
            let block_inputs = &stage_inputs[stage_idx];
            for (block, block_input) in blocks.iter_mut().rev().zip(block_inputs.iter().rev()) {
                next_grad = block.backward(block_input, &next_grad)?;
            }
            if stage_idx > 0 {
                let entry = &mut stage_output_grads[stage_idx - 1];
                match entry.take() {
                    Some(existing) => {
                        let combined = existing.add(&next_grad)?;
                        *entry = Some(combined.clone());
                        next_grad = combined;
                    }
                    None => {
                        *entry = Some(next_grad.clone());
                    }
                }
            }
        }
        let mut grad_input = next_grad;
        if let Some(pool) = &mut self.stem_pool {
            grad_input = pool.backward(&stem_act, &grad_input)?;
        }
        grad_input = self.stem_activation.backward(&stem_norm, &grad_input)?;
        grad_input = self.stem_norm.backward(&stem_out, &grad_input)?;
        grad_input = self.stem_conv.backward(input, &grad_input)?;
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.stem_conv.visit_parameters(visitor)?;
        self.stem_norm.visit_parameters(visitor)?;
        for stage in &self.stages {
            for block in stage {
                block.visit_parameters(visitor)?;
            }
        }
        self.fusion_norm.visit_parameters(visitor)?;
        for bridge in &self.skip_bridges {
            bridge.visit_parameters(visitor)?;
        }
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.stem_conv.visit_parameters_mut(visitor)?;
        self.stem_norm.visit_parameters_mut(visitor)?;
        for stage in &mut self.stages {
            for block in stage {
                block.visit_parameters_mut(visitor)?;
            }
        }
        self.fusion_norm.visit_parameters_mut(visitor)?;
        for bridge in &mut self.skip_bridges {
            bridge.visit_parameters_mut(visitor)?;
        }
        Ok(())
    }
}
