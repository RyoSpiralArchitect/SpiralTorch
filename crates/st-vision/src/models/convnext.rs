// SPDX-License-Identifier: AGPL-3.0-or-later

use std::path::Path;

use crate::models::utils::{conv_output_hw, conv_to_tokens, tokens_to_conv};
use st_nn::io;
use st_nn::layers::gelu::Gelu;
use st_nn::layers::linear::Linear;
use st_nn::layers::normalization::LayerNorm;
use st_nn::module::{Module, Parameter};
use st_nn::PureResult;
use st_tensor::{Tensor, TensorError};

use st_nn::layers::conv::Conv2d;

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
            epsilon: 1.0e-5,
        }
    }
}

#[derive(Debug)]
struct ConvNeXtBlock {
    norm: LayerNorm,
    mlp1: Linear,
    activation: Gelu,
    mlp2: Linear,
}

impl ConvNeXtBlock {
    fn new(name: &str, channels: usize, curvature: f32, epsilon: f32) -> PureResult<Self> {
        if channels == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: 1,
                cols: channels,
            });
        }
        if curvature >= 0.0 || !curvature.is_finite() {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        if epsilon <= 0.0 || !epsilon.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "convnext_layernorm_epsilon",
                value: epsilon,
            });
        }
        Ok(Self {
            norm: LayerNorm::new(format!("{name}.ln"), channels, curvature, epsilon)?,
            mlp1: Linear::new(format!("{name}.mlp1"), channels, channels * 4)?,
            activation: Gelu::new(),
            mlp2: Linear::new(format!("{name}.mlp2"), channels * 4, channels)?,
        })
    }
}

impl Module for ConvNeXtBlock {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let norm = self.norm.forward(input)?;
        let mlp = self.mlp1.forward(&norm)?;
        let act = self.activation.forward(&mlp)?;
        let proj = self.mlp2.forward(&act)?;
        input.add(&proj)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let norm = self.norm.forward(input)?;
        let mlp = self.mlp1.forward(&norm)?;
        let act = self.activation.forward(&mlp)?;

        let mut grad = self.mlp2.backward(&act, grad_output)?;
        grad = self.activation.backward(&mlp, &grad)?;
        grad = self.mlp1.backward(&norm, &grad)?;
        let grad_norm = self.norm.backward(input, &grad)?;
        grad_norm.add(grad_output)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.norm.visit_parameters(visitor)?;
        self.mlp1.visit_parameters(visitor)?;
        self.mlp2.visit_parameters(visitor)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.norm.visit_parameters_mut(visitor)?;
        self.mlp1.visit_parameters_mut(visitor)?;
        self.mlp2.visit_parameters_mut(visitor)?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct ConvNeXtBackbone {
    patch_embed: Conv2d,
    stages: Vec<Vec<ConvNeXtBlock>>,
    merges: Vec<Conv2d>,
    stage_dims: Vec<usize>,
    stage_hw: Vec<(usize, usize)>,
    stage_shapes: Vec<(usize, (usize, usize))>,
}

impl ConvNeXtBackbone {
    pub fn new(config: ConvNeXtConfig) -> PureResult<Self> {
        if config.stage_dims.is_empty() {
            return Err(TensorError::EmptyInput("convnext_stage_dims"));
        }
        if config.stage_dims.len() != config.stage_depths.len() {
            return Err(TensorError::InvalidDimensions {
                rows: config.stage_dims.len(),
                cols: config.stage_depths.len(),
            });
        }
        if config.input_hw.0 == 0 || config.input_hw.1 == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: config.input_hw.0,
                cols: config.input_hw.1,
            });
        }
        let patch_hw = conv_output_hw(
            config.input_hw,
            config.patch_size,
            config.patch_size,
            (0, 0),
            (1, 1),
        )?;
        let patch_embed = Conv2d::new(
            "convnext.patch",
            config.input_channels,
            config.stage_dims[0],
            config.patch_size,
            config.patch_size,
            (0, 0),
            (1, 1),
            config.input_hw,
        )?;

        let mut stages = Vec::with_capacity(config.stage_dims.len());
        let mut merges = Vec::with_capacity(config.stage_dims.len().saturating_sub(1));
        let mut stage_hw = Vec::with_capacity(config.stage_dims.len());
        let mut stage_shapes = Vec::with_capacity(config.stage_dims.len());
        let mut current_hw = patch_hw;
        for (idx, (&dim, &depth)) in config
            .stage_dims
            .iter()
            .zip(config.stage_depths.iter())
            .enumerate()
        {
            let mut blocks = Vec::with_capacity(depth);
            for block_idx in 0..depth {
                blocks.push(ConvNeXtBlock::new(
                    &format!("convnext.stage{idx}.block{block_idx}"),
                    dim,
                    config.curvature,
                    config.epsilon,
                )?);
            }
            stage_hw.push(current_hw);
            stage_shapes.push((dim, current_hw));
            stages.push(blocks);
            if idx + 1 < config.stage_dims.len() {
                let next_dim = config.stage_dims[idx + 1];
                let down = Conv2d::new(
                    format!("convnext.down{idx}"),
                    dim,
                    next_dim,
                    (2, 2),
                    (2, 2),
                    (0, 0),
                    (1, 1),
                    current_hw,
                )?;
                current_hw = conv_output_hw(current_hw, (2, 2), (2, 2), (0, 0), (1, 1))?;
                merges.push(down);
            }
        }

        Ok(Self {
            patch_embed,
            stages,
            merges,
            stage_dims: config.stage_dims,
            stage_hw,
            stage_shapes,
        })
    }

    pub fn output_shape(&self) -> (usize, (usize, usize)) {
        let last = self.stage_dims.len() - 1;
        (self.stage_dims[last], self.stage_hw[last])
    }

    pub fn stage_shapes(&self) -> &[(usize, (usize, usize))] {
        &self.stage_shapes
    }

    pub fn load_weights_json<P: AsRef<Path>>(&mut self, path: P) -> PureResult<()> {
        io::load_json(self, path)
    }

    pub fn load_weights_bincode<P: AsRef<Path>>(&mut self, path: P) -> PureResult<()> {
        io::load_bincode(self, path)
    }
}

impl Module for ConvNeXtBackbone {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let batch = input.shape().0;
        let mut activ = self.patch_embed.forward(input)?;
        let mut tokens = conv_to_tokens(&activ, self.stage_dims[0], self.stage_hw[0])?;
        for (idx, blocks) in self.stages.iter().enumerate() {
            for block in blocks {
                tokens = block.forward(&tokens)?;
            }
            if let Some(merge) = self.merges.get(idx) {
                let conv_in =
                    tokens_to_conv(&tokens, batch, self.stage_dims[idx], self.stage_hw[idx])?;
                activ = merge.forward(&conv_in)?;
                tokens = conv_to_tokens(&activ, self.stage_dims[idx + 1], self.stage_hw[idx + 1])?;
            }
        }
        let last_idx = self.stage_dims.len() - 1;
        tokens_to_conv(
            &tokens,
            batch,
            self.stage_dims[last_idx],
            self.stage_hw[last_idx],
        )
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let batch = input.shape().0;
        let mut activ = self.patch_embed.forward(input)?;
        let mut tokens = conv_to_tokens(&activ, self.stage_dims[0], self.stage_hw[0])?;
        let mut block_inputs: Vec<Vec<Tensor>> = Vec::with_capacity(self.stages.len());
        let mut merge_inputs = Vec::with_capacity(self.merges.len());
        for (idx, blocks) in self.stages.iter().enumerate() {
            let mut inputs = Vec::with_capacity(blocks.len());
            for block in blocks {
                inputs.push(tokens.clone());
                tokens = block.forward(&tokens)?;
            }
            block_inputs.push(inputs);
            if let Some(merge) = self.merges.get(idx) {
                let conv_in =
                    tokens_to_conv(&tokens, batch, self.stage_dims[idx], self.stage_hw[idx])?;
                merge_inputs.push(conv_in.clone());
                activ = merge.forward(&conv_in)?;
                tokens = conv_to_tokens(&activ, self.stage_dims[idx + 1], self.stage_hw[idx + 1])?;
            }
        }

        let last_idx = self.stage_dims.len() - 1;
        let mut grad_tokens = conv_to_tokens(
            grad_output,
            self.stage_dims[last_idx],
            self.stage_hw[last_idx],
        )?;
        for stage_idx in (0..self.stages.len()).rev() {
            let blocks = &mut self.stages[stage_idx];
            let inputs = &block_inputs[stage_idx];
            for (block, input_tokens) in blocks.iter_mut().zip(inputs.iter()).rev() {
                grad_tokens = block.backward(input_tokens, &grad_tokens)?;
            }
            if stage_idx > 0 {
                let merge_idx = stage_idx - 1;
                let merge = &mut self.merges[merge_idx];
                let grad_conv = tokens_to_conv(
                    &grad_tokens,
                    batch,
                    self.stage_dims[stage_idx],
                    self.stage_hw[stage_idx],
                )?;
                let grad_prev = merge.backward(&merge_inputs[merge_idx], &grad_conv)?;
                grad_tokens = conv_to_tokens(
                    &grad_prev,
                    self.stage_dims[merge_idx],
                    self.stage_hw[merge_idx],
                )?;
            }
        }
        let grad_patches =
            tokens_to_conv(&grad_tokens, batch, self.stage_dims[0], self.stage_hw[0])?;
        self.patch_embed.backward(input, &grad_patches)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.patch_embed.visit_parameters(visitor)?;
        for stage in &self.stages {
            for block in stage {
                block.visit_parameters(visitor)?;
            }
        }
        for merge in &self.merges {
            merge.visit_parameters(visitor)?;
        }
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.patch_embed.visit_parameters_mut(visitor)?;
        for stage in &mut self.stages {
            for block in stage {
                block.visit_parameters_mut(visitor)?;
            }
        }
        for merge in &mut self.merges {
            merge.visit_parameters_mut(visitor)?;
        }
        Ok(())
    }
}
