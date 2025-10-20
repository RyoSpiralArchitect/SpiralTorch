// SPDX-License-Identifier: AGPL-3.0-or-later

use std::path::Path;

use crate::models::utils::{
    conv_to_tokens, mean_pool_tokens, mean_pool_tokens_backward, tokens_to_conv,
};
use st_nn::io;
use st_nn::layers::gelu::Gelu;
use st_nn::layers::linear::Linear;
use st_nn::layers::normalization::LayerNorm;
use st_nn::module::{Module, Parameter};
use st_nn::PureResult;
use st_tensor::{Tensor, TensorError};

use st_nn::layers::conv::Conv2d;

#[derive(Clone, Debug)]
pub struct ViTConfig {
    pub image_hw: (usize, usize),
    pub patch_size: (usize, usize),
    pub in_channels: usize,
    pub embed_dim: usize,
    pub depth: usize,
    pub mlp_dim: usize,
    pub curvature: f32,
    pub epsilon: f32,
}

impl Default for ViTConfig {
    fn default() -> Self {
        Self {
            image_hw: (224, 224),
            patch_size: (16, 16),
            in_channels: 3,
            embed_dim: 768,
            depth: 12,
            mlp_dim: 3072,
            curvature: -1.0,
            epsilon: 1.0e-5,
        }
    }
}

#[derive(Debug)]
struct TransformerBlock {
    norm1: LayerNorm,
    proj: Linear,
    norm2: LayerNorm,
    mlp1: Linear,
    activation: Gelu,
    mlp2: Linear,
}

impl TransformerBlock {
    fn new(
        name: &str,
        embed_dim: usize,
        mlp_dim: usize,
        curvature: f32,
        epsilon: f32,
    ) -> PureResult<Self> {
        if embed_dim == 0 || mlp_dim == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: embed_dim,
                cols: mlp_dim,
            });
        }
        if curvature >= 0.0 || !curvature.is_finite() {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        if epsilon <= 0.0 || !epsilon.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "vit_layernorm_epsilon",
                value: epsilon,
            });
        }
        Ok(Self {
            norm1: LayerNorm::new(format!("{name}.ln1"), embed_dim, curvature, epsilon)?,
            proj: Linear::new(format!("{name}.proj"), embed_dim, embed_dim)?,
            norm2: LayerNorm::new(format!("{name}.ln2"), embed_dim, curvature, epsilon)?,
            mlp1: Linear::new(format!("{name}.mlp1"), embed_dim, mlp_dim)?,
            activation: Gelu::new(),
            mlp2: Linear::new(format!("{name}.mlp2"), mlp_dim, embed_dim)?,
        })
    }
}

impl Module for TransformerBlock {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let norm1 = self.norm1.forward(input)?;
        let attn = self.proj.forward(&norm1)?;
        let resid = input.add(&attn)?;
        let norm2 = self.norm2.forward(&resid)?;
        let mlp_in = self.mlp1.forward(&norm2)?;
        let mlp_act = self.activation.forward(&mlp_in)?;
        resid.add(&self.mlp2.forward(&mlp_act)?)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let norm1 = self.norm1.forward(input)?;
        let attn = self.proj.forward(&norm1)?;
        let resid = input.add(&attn)?;
        let norm2 = self.norm2.forward(&resid)?;
        let mlp_in = self.mlp1.forward(&norm2)?;
        let mlp_act = self.activation.forward(&mlp_in)?;
        let grad_resid_from_output = grad_output.clone();
        let mut grad_mlp = self.mlp2.backward(&mlp_act, grad_output)?;
        grad_mlp = self.activation.backward(&mlp_in, &grad_mlp)?;
        grad_mlp = self.mlp1.backward(&norm2, &grad_mlp)?;
        let grad_resid_from_mlp = self.norm2.backward(&resid, &grad_mlp)?;
        let grad_resid = grad_resid_from_output.add(&grad_resid_from_mlp)?;

        let grad_attn = self.proj.backward(&norm1, &grad_resid)?;
        let grad_input_from_attn = self.norm1.backward(input, &grad_attn)?;
        grad_input_from_attn.add(&grad_resid)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.norm1.visit_parameters(visitor)?;
        self.proj.visit_parameters(visitor)?;
        self.norm2.visit_parameters(visitor)?;
        self.mlp1.visit_parameters(visitor)?;
        self.mlp2.visit_parameters(visitor)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.norm1.visit_parameters_mut(visitor)?;
        self.proj.visit_parameters_mut(visitor)?;
        self.norm2.visit_parameters_mut(visitor)?;
        self.mlp1.visit_parameters_mut(visitor)?;
        self.mlp2.visit_parameters_mut(visitor)?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct ViTBackbone {
    patch_embed: Conv2d,
    blocks: Vec<TransformerBlock>,
    embed_dim: usize,
    patch_grid: (usize, usize),
    num_patches: usize,
}

impl ViTBackbone {
    pub fn new(config: ViTConfig) -> PureResult<Self> {
        if config.image_hw.0 == 0 || config.image_hw.1 == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: config.image_hw.0,
                cols: config.image_hw.1,
            });
        }
        if config.patch_size.0 == 0
            || config.patch_size.1 == 0
            || config.image_hw.0 % config.patch_size.0 != 0
            || config.image_hw.1 % config.patch_size.1 != 0
        {
            return Err(TensorError::InvalidDimensions {
                rows: config.patch_size.0,
                cols: config.patch_size.1,
            });
        }
        if config.embed_dim == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: 1,
                cols: config.embed_dim,
            });
        }
        let grid = (
            config.image_hw.0 / config.patch_size.0,
            config.image_hw.1 / config.patch_size.1,
        );
        let num_patches = grid.0 * grid.1;
        let patch_embed = Conv2d::new(
            "vit.patch",
            config.in_channels,
            config.embed_dim,
            config.patch_size,
            config.patch_size,
            (0, 0),
            (1, 1),
            config.image_hw,
        )?;
        let mut blocks = Vec::with_capacity(config.depth);
        for idx in 0..config.depth {
            blocks.push(TransformerBlock::new(
                &format!("vit.block{idx}"),
                config.embed_dim,
                config.mlp_dim,
                config.curvature,
                config.epsilon,
            )?);
        }
        Ok(Self {
            patch_embed,
            blocks,
            embed_dim: config.embed_dim,
            patch_grid: grid,
            num_patches,
        })
    }

    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    pub fn patches(&self) -> usize {
        self.num_patches
    }

    pub fn load_weights_json<P: AsRef<Path>>(&mut self, path: P) -> PureResult<()> {
        io::load_json(self, path)
    }

    pub fn load_weights_bincode<P: AsRef<Path>>(&mut self, path: P) -> PureResult<()> {
        io::load_bincode(self, path)
    }
}

impl Module for ViTBackbone {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let batch = input.shape().0;
        let patches = self.patch_embed.forward(input)?;
        let mut tokens = conv_to_tokens(&patches, self.embed_dim, self.patch_grid)?;
        for block in &self.blocks {
            tokens = block.forward(&tokens)?;
        }
        mean_pool_tokens(&tokens, batch, self.num_patches)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let batch = input.shape().0;
        let patches = self.patch_embed.forward(input)?;
        let mut tokens = conv_to_tokens(&patches, self.embed_dim, self.patch_grid)?;
        let mut block_inputs = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            block_inputs.push(tokens.clone());
            tokens = block.forward(&tokens)?;
        }
        let mut grad = mean_pool_tokens_backward(grad_output, batch, self.num_patches)?;
        for (block, activ) in self.blocks.iter_mut().zip(block_inputs.into_iter()).rev() {
            grad = block.backward(&activ, &grad)?;
        }
        let grad_patches = tokens_to_conv(&grad, batch, self.embed_dim, self.patch_grid)?;
        self.patch_embed.backward(input, &grad_patches)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.patch_embed.visit_parameters(visitor)?;
        for block in &self.blocks {
            block.visit_parameters(visitor)?;
        }
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.patch_embed.visit_parameters_mut(visitor)?;
        for block in &mut self.blocks {
            block.visit_parameters_mut(visitor)?;
        }
        Ok(())
    }
}
