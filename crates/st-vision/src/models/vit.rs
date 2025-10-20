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

fn flatten_patches_to_tokens(
    input: &Tensor,
    channels: usize,
    hw: (usize, usize),
) -> PureResult<Tensor> {
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
                let channel_offset = c * tokens_per_batch;
                data.push(row[channel_offset + token]);
            }
        }
    }
    Tensor::from_vec(batch * tokens_per_batch, channels, data)
}

fn repeat_rows(tensor: &Tensor, repeats: usize) -> PureResult<Tensor> {
    let (rows, cols) = tensor.shape();
    let mut data = Vec::with_capacity(rows * repeats * cols);
    for _ in 0..repeats {
        data.extend_from_slice(tensor.data());
    }
    Tensor::from_vec(rows * repeats, cols, data)
}

#[derive(Clone, Debug)]
pub struct ViTConfig {
    pub image_hw: (usize, usize),
    pub patch_size: (usize, usize),
    pub in_channels: usize,
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub mlp_dim: usize,
    pub include_cls_token: bool,
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
            num_heads: 12,
            mlp_dim: 3072,
            include_cls_token: true,
            curvature: -1.0,
            epsilon: 1e-6,
        }
    }
}

#[derive(Debug)]
struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    proj: Linear,
    num_heads: usize,
    tokens_per_batch: usize,
}

impl MultiHeadAttention {
    fn new(
        name: &str,
        embed_dim: usize,
        num_heads: usize,
        tokens_per_batch: usize,
    ) -> PureResult<Self> {
        if embed_dim % num_heads != 0 {
            return Err(TensorError::InvalidDimensions {
                rows: embed_dim,
                cols: num_heads,
            });
        }
        Ok(Self {
            query: Linear::new(format!("{name}.q"), embed_dim, embed_dim)?,
            key: Linear::new(format!("{name}.k"), embed_dim, embed_dim)?,
            value: Linear::new(format!("{name}.v"), embed_dim, embed_dim)?,
            proj: Linear::new(format!("{name}.proj"), embed_dim, embed_dim)?,
            num_heads,
            tokens_per_batch,
        })
    }

    fn softmax(vector: &mut [f32]) {
        let max = vector.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for value in vector.iter_mut() {
            *value = (*value - max).exp();
            sum += *value;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 }; // guard degeneracy
        for value in vector.iter_mut() {
            *value *= inv;
        }
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let q = self.query.forward(input)?;
        let k = self.key.forward(input)?;
        let v = self.value.forward(input)?;
        let head_dim = q.shape().1 / self.num_heads;
        let total_tokens = q.shape().0;
        if total_tokens % self.tokens_per_batch != 0 {
            return Err(TensorError::InvalidDimensions {
                rows: total_tokens,
                cols: self.tokens_per_batch,
            });
        }
        let batch = total_tokens / self.tokens_per_batch;
        let mut context = vec![0.0f32; total_tokens * q.shape().1];
        let q_data = q.data();
        let k_data = k.data();
        let v_data = v.data();
        for b in 0..batch {
            for head in 0..self.num_heads {
                let head_offset = head * head_dim;
                for token in 0..self.tokens_per_batch {
                    let row_idx = b * self.tokens_per_batch + token;
                    let q_offset = row_idx * q.shape().1 + head_offset;
                    let mut logits = vec![0.0f32; self.tokens_per_batch];
                    for other in 0..self.tokens_per_batch {
                        let k_row = b * self.tokens_per_batch + other;
                        let k_offset = k_row * k.shape().1 + head_offset;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q_data[q_offset + d] * k_data[k_offset + d];
                        }
                        logits[other] = dot / (head_dim as f32).sqrt();
                    }
                    Self::softmax(&mut logits);
                    let mut out = vec![0.0f32; head_dim];
                    for other in 0..self.tokens_per_batch {
                        let weight = logits[other];
                        let v_row = b * self.tokens_per_batch + other;
                        let v_offset = v_row * v.shape().1 + head_offset;
                        for d in 0..head_dim {
                            out[d] += weight * v_data[v_offset + d];
                        }
                    }
                    let out_offset = row_idx * q.shape().1 + head_offset;
                    context[out_offset..out_offset + head_dim].copy_from_slice(&out);
                }
            }
        }
        let context = Tensor::from_vec(total_tokens, q.shape().1, context)?;
        self.proj.forward(&context)
    }

    fn backward(&mut self, _input: &Tensor, _grad_output: &Tensor) -> PureResult<Tensor> {
        Err(TensorError::InvalidValue {
            label: "vit_attention_backward",
        })
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.query.visit_parameters(visitor)?;
        self.key.visit_parameters(visitor)?;
        self.value.visit_parameters(visitor)?;
        self.proj.visit_parameters(visitor)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.query.visit_parameters_mut(visitor)?;
        self.key.visit_parameters_mut(visitor)?;
        self.value.visit_parameters_mut(visitor)?;
        self.proj.visit_parameters_mut(visitor)
    }
}

#[derive(Debug)]
struct TransformerBlock {
    norm1: LayerNorm,
    attn: MultiHeadAttention,
    norm2: LayerNorm,
    mlp_fc1: Linear,
    mlp_act: Gelu,
    mlp_fc2: Linear,
}

impl TransformerBlock {
    fn new(
        name: &str,
        embed_dim: usize,
        mlp_dim: usize,
        num_heads: usize,
        tokens_per_batch: usize,
        curvature: f32,
        epsilon: f32,
    ) -> PureResult<Self> {
        let norm1 = LayerNorm::new(format!("{name}.ln1"), embed_dim, curvature, epsilon)?;
        let attn = MultiHeadAttention::new(
            &format!("{name}.attn"),
            embed_dim,
            num_heads,
            tokens_per_batch,
        )?;
        let norm2 = LayerNorm::new(format!("{name}.ln2"), embed_dim, curvature, epsilon)?;
        let mlp_fc1 = Linear::new(format!("{name}.fc1"), embed_dim, mlp_dim)?;
        let mlp_act = Gelu::new();
        let mlp_fc2 = Linear::new(format!("{name}.fc2"), mlp_dim, embed_dim)?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp_fc1,
            mlp_act,
            mlp_fc2,
        })
    }
}

impl Module for TransformerBlock {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let normed = self.norm1.forward(input)?;
        let attn_out = self.attn.forward(&normed)?;
        let resid1 = attn_out.add(input)?;
        let normed2 = self.norm2.forward(&resid1)?;
        let hidden = self.mlp_fc1.forward(&normed2)?;
        let activated = self.mlp_act.forward(&hidden)?;
        let mlp_out = self.mlp_fc2.forward(&activated)?;
        mlp_out.add(&resid1)
    }

    fn backward(&mut self, _input: &Tensor, _grad_output: &Tensor) -> PureResult<Tensor> {
        Err(TensorError::InvalidValue {
            label: "vit_block_backward",
        })
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.norm1.visit_parameters(visitor)?;
        self.attn.visit_parameters(visitor)?;
        self.norm2.visit_parameters(visitor)?;
        self.mlp_fc1.visit_parameters(visitor)?;
        self.mlp_fc2.visit_parameters(visitor)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.norm1.visit_parameters_mut(visitor)?;
        self.attn.visit_parameters_mut(visitor)?;
        self.norm2.visit_parameters_mut(visitor)?;
        self.mlp_fc1.visit_parameters_mut(visitor)?;
        self.mlp_fc2.visit_parameters_mut(visitor)
    }
}

#[derive(Debug)]
pub struct ViTBackbone {
    patch_embed: Conv2d,
    cls_token: Option<Parameter>,
    pos_embed: Parameter,
    blocks: Vec<TransformerBlock>,
    final_norm: LayerNorm,
    tokens_per_batch: usize,
    grid_hw: (usize, usize),
    embed_dim: usize,
}

impl ViTBackbone {
    pub fn new(config: ViTConfig) -> PureResult<Self> {
        if config.curvature >= 0.0 || !config.curvature.is_finite() {
            return Err(TensorError::NonHyperbolicCurvature {
                curvature: config.curvature,
            });
        }
        if config.epsilon <= 0.0 || !config.epsilon.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "vit_layernorm_epsilon",
                value: config.epsilon,
            });
        }
        if config.image_hw.0 % config.patch_size.0 != 0
            || config.image_hw.1 % config.patch_size.1 != 0
        {
            return Err(TensorError::InvalidDimensions {
                rows: config.image_hw.0,
                cols: config.patch_size.0,
            });
        }
        let patch_embed = Conv2d::new(
            "vit.patch_embed",
            config.in_channels,
            config.embed_dim,
            config.patch_size,
            config.patch_size,
            (0, 0),
            (1, 1),
            config.image_hw,
        )?;
        let grid_hw = conv_output_hw(
            config.image_hw,
            config.patch_size,
            config.patch_size,
            (0, 0),
            (1, 1),
        )?;
        let mut tokens_per_batch = grid_hw.0 * grid_hw.1;
        let cls_token = if config.include_cls_token {
            tokens_per_batch += 1;
            Some(Parameter::new(
                "vit.cls_token",
                Tensor::zeros(1, config.embed_dim)?,
            ))
        } else {
            None
        };
        let pos_embed = Parameter::new(
            "vit.pos_embed",
            Tensor::random_normal(tokens_per_batch, config.embed_dim, 0.0, 0.02, Some(7))?,
        );
        let mut blocks = Vec::with_capacity(config.depth);
        for idx in 0..config.depth {
            blocks.push(TransformerBlock::new(
                &format!("vit.encoder{idx}"),
                config.embed_dim,
                config.mlp_dim,
                config.num_heads,
                tokens_per_batch,
                config.curvature,
                config.epsilon,
            )?);
        }
        let final_norm = LayerNorm::new(
            "vit.final_norm",
            config.embed_dim,
            config.curvature,
            config.epsilon,
        )?;
        Ok(Self {
            patch_embed,
            cls_token,
            pos_embed,
            blocks,
            final_norm,
            tokens_per_batch,
            grid_hw,
            embed_dim: config.embed_dim,
        })
    }

    pub fn load_weights_json<P: AsRef<Path>>(&mut self, path: P) -> PureResult<()> {
        io::load_json(self, path)
    }

    pub fn load_weights_bincode<P: AsRef<Path>>(&mut self, path: P) -> PureResult<()> {
        io::load_bincode(self, path)
    }

    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }
}

impl Module for ViTBackbone {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let patches = self.patch_embed.forward(input)?;
        let mut tokens = flatten_patches_to_tokens(&patches, self.embed_dim, self.grid_hw)?;
        let batch = input.shape().0;
        if let Some(token) = &self.cls_token {
            let per_batch = tokens.shape().0 / batch;
            let mut data = Vec::with_capacity(batch * self.tokens_per_batch * self.embed_dim);
            for b in 0..batch {
                data.extend_from_slice(token.value().data());
                let start = b * per_batch * self.embed_dim;
                let end = start + per_batch * self.embed_dim;
                data.extend_from_slice(&tokens.data()[start..end]);
            }
            tokens = Tensor::from_vec(batch * self.tokens_per_batch, self.embed_dim, data)?;
        } else if tokens.shape().0 != batch * self.tokens_per_batch {
            let repeats = self.tokens_per_batch / (self.grid_hw.0 * self.grid_hw.1);
            tokens = repeat_rows(&tokens, repeats)?;
        }
        let pos = repeat_rows(self.pos_embed.value(), batch)?;
        tokens = tokens.add(&pos)?;
        let mut activ = tokens;
        for block in &self.blocks {
            activ = block.forward(&activ)?;
        }
        activ = self.final_norm.forward(&activ)?;
        if self.cls_token.is_some() {
            let mut data = Vec::with_capacity(batch * self.embed_dim);
            for b in 0..batch {
                let row = &activ.data()[b * self.tokens_per_batch * self.embed_dim
                    ..(b * self.tokens_per_batch + 1) * self.embed_dim];
                data.extend_from_slice(row);
            }
            Tensor::from_vec(batch, self.embed_dim, data)
        } else {
            let mut accum = vec![0.0f32; batch * self.embed_dim];
            for b in 0..batch {
                for token in 0..self.tokens_per_batch {
                    let row_idx = b * self.tokens_per_batch + token;
                    let row =
                        &activ.data()[row_idx * self.embed_dim..(row_idx + 1) * self.embed_dim];
                    for d in 0..self.embed_dim {
                        accum[b * self.embed_dim + d] += row[d];
                    }
                }
                let inv = 1.0 / self.tokens_per_batch as f32;
                for d in 0..self.embed_dim {
                    accum[b * self.embed_dim + d] *= inv;
                }
            }
            Tensor::from_vec(batch, self.embed_dim, accum)
        }
    }

    fn backward(&mut self, _input: &Tensor, _grad_output: &Tensor) -> PureResult<Tensor> {
        Err(TensorError::InvalidValue {
            label: "vit_backbone_backward",
        })
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.patch_embed.visit_parameters(visitor)?;
        if let Some(token) = &self.cls_token {
            visitor(token)?;
        }
        visitor(&self.pos_embed)?;
        for block in &self.blocks {
            block.visit_parameters(visitor)?;
        }
        self.final_norm.visit_parameters(visitor)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.patch_embed.visit_parameters_mut(visitor)?;
        if let Some(token) = &mut self.cls_token {
            visitor(token)?;
        }
        visitor(&mut self.pos_embed)?;
        for block in &mut self.blocks {
            block.visit_parameters_mut(visitor)?;
        }
        self.final_norm.visit_parameters_mut(visitor)
    }
}
