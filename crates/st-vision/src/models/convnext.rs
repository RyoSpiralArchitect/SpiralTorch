// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::path::Path;

use st_nn::io;
use st_nn::layers::conv::Conv2d;
use st_nn::layers::gelu::Gelu;
use st_nn::layers::linear::Linear;
use st_nn::layers::normalization::LayerNorm;
use st_nn::module::{Module, Parameter};
use st_nn::PureResult;
use st_tensor::{Tensor, TensorError};

use crate::models::resnet::conv_output_hw;

fn conv_to_tokens(input: &Tensor, channels: usize, hw: (usize, usize)) -> PureResult<Tensor> {
    let (batch, cols) = input.shape();
    let expected = channels * hw.0 * hw.1;
    if cols != expected {
        return Err(TensorError::ShapeMismatch {
            left: (batch, cols),
            right: (batch, expected),
        });
    }
    let tokens_per_batch = hw.0 * hw.1;
    let mut data = Vec::with_capacity(batch * tokens_per_batch * channels);
    for b in 0..batch {
        let row = &input.data()[b * cols..(b + 1) * cols];
        for token in 0..tokens_per_batch {
            for c in 0..channels {
                let offset = c * tokens_per_batch + token;
                data.push(row[offset]);
            }
        }
    }
    Tensor::from_vec(batch * tokens_per_batch, channels, data)
}

fn tokens_to_conv(
    tokens: &Tensor,
    batch: usize,
    channels: usize,
    hw: (usize, usize),
) -> PureResult<Tensor> {
    let tokens_per_batch = hw.0 * hw.1;
    if tokens.shape().0 != batch * tokens_per_batch || tokens.shape().1 != channels {
        return Err(TensorError::ShapeMismatch {
            left: tokens.shape(),
            right: (batch * tokens_per_batch, channels),
        });
    }
    let mut data = vec![0.0f32; batch * channels * tokens_per_batch];
    for b in 0..batch {
        for token in 0..tokens_per_batch {
            for c in 0..channels {
                let src = (b * tokens_per_batch + token) * channels + c;
                let dst = b * channels * tokens_per_batch + c * tokens_per_batch + token;
                data[dst] = tokens.data()[src];
            }
        }
    }
    Tensor::from_vec(batch, channels * tokens_per_batch, data)
}

#[derive(Clone, Debug)]
pub struct ConvNeXtConfig {
    pub input_channels: usize,
    pub input_hw: (usize, usize),
    pub stage_dims: Vec<usize>,
    pub stage_depths: Vec<usize>,
    pub patch_size: (usize, usize),
    pub curvature: f32,
    pub epsilon: f32,
}

impl Default for ConvNeXtConfig {
    fn default() -> Self {
        Self {
            input_channels: 3,
            input_hw: (224, 224),
            stage_dims: vec![96, 192, 384, 768],
            stage_depths: vec![3, 3, 9, 3],
            patch_size: (4, 4),
            curvature: -1.0,
            epsilon: 1e-6,
        }
    }
}

#[derive(Debug)]
struct ConvNeXtBlock {
    depthwise: Conv2d,
    norm: LayerNorm,
    mlp1: Linear,
    activation: Gelu,
    mlp2: Linear,
    channels: usize,
    hw: (usize, usize),
}

impl ConvNeXtBlock {
    fn new(
        name: &str,
        channels: usize,
        input_hw: (usize, usize),
        curvature: f32,
        epsilon: f32,
    ) -> PureResult<Self> {
        let depthwise = Conv2d::new(
            format!("{name}.dw"),
            channels,
            channels,
            (7, 7),
            (1, 1),
            (3, 3),
            (1, 1),
            input_hw,
        )?;
        let norm = LayerNorm::new(format!("{name}.ln"), channels, curvature, epsilon)?;
        let mlp1 = Linear::new(format!("{name}.fc1"), channels, channels * 4)?;
        let activation = Gelu::new();
        let mlp2 = Linear::new(format!("{name}.fc2"), channels * 4, channels)?;
        Ok(Self {
            depthwise,
            norm,
            mlp1,
            activation,
            mlp2,
            channels,
            hw: input_hw,
        })
    }
}

impl Module for ConvNeXtBlock {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let dw = self.depthwise.forward(input)?;
        let tokens = conv_to_tokens(&dw, self.channels, self.hw)?;
        let normed = self.norm.forward(&tokens)?;
        let hidden = self.mlp1.forward(&normed)?;
        let activated = self.activation.forward(&hidden)?;
        let projected = self.mlp2.forward(&activated)?;
        let conv_layout = tokens_to_conv(&projected, input.shape().0, self.channels, self.hw)?;
        conv_layout.add(input)
    }

    fn backward(&mut self, _input: &Tensor, _grad_output: &Tensor) -> PureResult<Tensor> {
        Err(TensorError::InvalidValue {
            label: "convnext_block_backward",
        })
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.depthwise.visit_parameters(visitor)?;
        self.norm.visit_parameters(visitor)?;
        self.mlp1.visit_parameters(visitor)?;
        self.mlp2.visit_parameters(visitor)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.depthwise.visit_parameters_mut(visitor)?;
        self.norm.visit_parameters_mut(visitor)?;
        self.mlp1.visit_parameters_mut(visitor)?;
        self.mlp2.visit_parameters_mut(visitor)
    }
}

#[derive(Debug)]
struct ConvNeXtStage {
    blocks: Vec<ConvNeXtBlock>,
    downsample: Option<Conv2d>,
}

impl ConvNeXtStage {
    fn new(
        name: &str,
        channels: usize,
        depth: usize,
        input_hw: (usize, usize),
        next_channels: Option<usize>,
        curvature: f32,
        epsilon: f32,
    ) -> PureResult<(Self, (usize, usize), usize)> {
        let mut blocks = Vec::with_capacity(depth);
        for idx in 0..depth {
            blocks.push(ConvNeXtBlock::new(
                &format!("{name}.block{idx}"),
                channels,
                input_hw,
                curvature,
                epsilon,
            )?);
        }
        let (downsample, next_hw, next_channels) = if let Some(next) = next_channels {
            let conv = Conv2d::new(
                format!("{name}.downsample"),
                channels,
                next,
                (2, 2),
                (2, 2),
                (0, 0),
                (1, 1),
                input_hw,
            )?;
            let hw = conv_output_hw(input_hw, (2, 2), (2, 2), (0, 0), (1, 1))?;
            (Some(conv), hw, next)
        } else {
            (None, input_hw, channels)
        };
        Ok((Self { blocks, downsample }, next_hw, next_channels))
    }
}

impl Module for ConvNeXtStage {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let mut activ = input.clone();
        for block in &self.blocks {
            activ = block.forward(&activ)?;
        }
        if let Some(down) = &self.downsample {
            activ = down.forward(&activ)?;
        }
        Ok(activ)
    }

    fn backward(&mut self, _input: &Tensor, _grad_output: &Tensor) -> PureResult<Tensor> {
        Err(TensorError::InvalidValue {
            label: "convnext_stage_backward",
        })
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        for block in &self.blocks {
            block.visit_parameters(visitor)?;
        }
        if let Some(down) = &self.downsample {
            down.visit_parameters(visitor)?;
        }
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        for block in &mut self.blocks {
            block.visit_parameters_mut(visitor)?;
        }
        if let Some(down) = &mut self.downsample {
            down.visit_parameters_mut(visitor)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct ConvNeXtBackbone {
    stem: Conv2d,
    stages: Vec<ConvNeXtStage>,
    final_norm: LayerNorm,
    output_channels: usize,
    output_hw: (usize, usize),
}

impl ConvNeXtBackbone {
    pub fn new(config: ConvNeXtConfig) -> PureResult<Self> {
        if config.stage_dims.len() != config.stage_depths.len() {
            return Err(TensorError::InvalidDimensions {
                rows: config.stage_dims.len(),
                cols: config.stage_depths.len(),
            });
        }
        if config.curvature >= 0.0 || !config.curvature.is_finite() {
            return Err(TensorError::NonHyperbolicCurvature {
                curvature: config.curvature,
            });
        }
        if config.epsilon <= 0.0 || !config.epsilon.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "convnext_layernorm_epsilon",
                value: config.epsilon,
            });
        }
        let stem = Conv2d::new(
            "convnext.stem",
            config.input_channels,
            config.stage_dims[0],
            config.patch_size,
            config.patch_size,
            (0, 0),
            (1, 1),
            config.input_hw,
        )?;
        let mut current_hw = conv_output_hw(
            config.input_hw,
            config.patch_size,
            config.patch_size,
            (0, 0),
            (1, 1),
        )?;
        let mut stages = Vec::with_capacity(config.stage_dims.len());
        let mut current_channels = config.stage_dims[0];
        for (idx, (&channels, &depth)) in config
            .stage_dims
            .iter()
            .zip(config.stage_depths.iter())
            .enumerate()
        {
            let next_channels = config.stage_dims.get(idx + 1).copied();
            let (stage, next_hw, next_ch) = ConvNeXtStage::new(
                &format!("convnext.stage{idx}"),
                channels,
                depth,
                current_hw,
                next_channels,
                config.curvature,
                config.epsilon,
            )?;
            current_hw = next_hw;
            current_channels = next_ch;
            stages.push(stage);
        }
        let final_norm = LayerNorm::new(
            "convnext.final_norm",
            current_channels * current_hw.0 * current_hw.1,
            config.curvature,
            config.epsilon,
        )?;
        Ok(Self {
            stem,
            stages,
            final_norm,
            output_channels: current_channels,
            output_hw: current_hw,
        })
    }

    pub fn load_weights_json<P: AsRef<Path>>(&mut self, path: P) -> PureResult<()> {
        io::load_json(self, path)
    }

    pub fn load_weights_bincode<P: AsRef<Path>>(&mut self, path: P) -> PureResult<()> {
        io::load_bincode(self, path)
    }

    pub fn output_shape(&self) -> (usize, (usize, usize)) {
        (self.output_channels, self.output_hw)
    }
}

impl Module for ConvNeXtBackbone {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let mut activ = self.stem.forward(input)?;
        for stage in &self.stages {
            activ = stage.forward(&activ)?;
        }
        self.final_norm.forward(&activ)
    }

    fn backward(&mut self, _input: &Tensor, _grad_output: &Tensor) -> PureResult<Tensor> {
        Err(TensorError::InvalidValue {
            label: "convnext_backbone_backward",
        })
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.stem.visit_parameters(visitor)?;
        for stage in &self.stages {
            stage.visit_parameters(visitor)?;
        }
        self.final_norm.visit_parameters(visitor)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.stem.visit_parameters_mut(visitor)?;
        for stage in &mut self.stages {
            stage.visit_parameters_mut(visitor)?;
        }
        self.final_norm.visit_parameters_mut(visitor)
    }
}
