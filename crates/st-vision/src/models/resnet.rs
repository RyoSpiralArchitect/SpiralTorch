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

pub(crate) fn conv_output_hw(
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

pub(crate) fn pool_output_hw(
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
        }
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
        Ok(Self {
            conv1,
            norm1,
            conv2,
            norm2,
            activation1,
            activation2,
            downsample,
            output_hw: conv2_hw,
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
        let conv1_out = self.conv1.forward(input)?;
        let norm1_out = self.norm1.forward(&conv1_out)?;
        let act1_out = self.activation1.forward(&norm1_out)?;
        let conv2_out = self.conv2.forward(&act1_out)?;
        let norm2_out = self.norm2.forward(&conv2_out)?;
        let summed = norm2_out.add(&residual)?;
        let out = self.activation2.forward(&summed)?;
        Ok((out, summed, conv2_out, act1_out, conv1_out))
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
        let summed = norm2_out.add(&residual_pre)?;
        let grad = self.activation2.backward(&summed, grad_output)?;
        let grad_residual = grad.clone();
        let mut grad_main = grad;
        grad_main = self.norm2.backward(&conv2_out, &grad_main)?;
        grad_main = self.conv2.backward(&act1_out, &grad_main)?;
        grad_main = self.activation1.backward(&norm1_out, &grad_main)?;
        grad_main = self.norm1.backward(&conv1_out, &grad_main)?;
        let mut grad_input = self.conv1.backward(input, &grad_main)?;
        if let Some((down_conv, down_norm)) = &mut self.downsample {
            let down_out = down_conv.forward(input)?;
            let grad_down = down_norm.backward(&down_out, &grad_residual)?;
            let grad_skip = down_conv.backward(input, &grad_down)?;
            grad_input = grad_input.add(&grad_skip)?;
        } else {
            grad_input = grad_input.add(&grad_residual)?;
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
                let block = ResNetBlock::new(
                    &format!("resnet.stage{stage_idx}.block{block_idx}"),
                    current_channels,
                    channels,
                    stride,
                    current_hw,
                    config.curvature,
                    config.epsilon,
                )?;
                tokens_per_stage.push((channels, block.output_hw()));
                current_hw = block.output_hw();
                current_channels = channels;
                blocks.push(block);
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
