// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

use serde::Serialize;
use st_nn::{Embedding, Linear, Module, Parameter, PureResult, Sequential, Tensor, TensorError};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum CharLmInputMode {
    TokenIndices,
    OneHot,
}

#[derive(Debug, Clone, Serialize)]
pub struct LanguageEvalMetric {
    pub tokens: usize,
    pub windows: usize,
    pub mean_nll: f32,
    pub perplexity: Option<f32>,
    pub accuracy: f32,
    pub mean_target_probability: f32,
    pub mean_top_probability: f32,
    pub mean_entropy: f32,
    pub mean_normalized_entropy: f32,
    pub mean_target_rank: f32,
    pub mean_target_percentile: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_target_logprob_lift: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_target_probability_lift: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_target_rank_lift: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_kl_to_unigram: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_top5_overlap_with_unigram: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainingSummary {
    pub initial_validation: Option<LanguageEvalMetric>,
    pub final_validation: Option<LanguageEvalMetric>,
    pub unigram_validation: Option<LanguageEvalMetric>,
    pub best_validation: Option<LanguageEvalMetric>,
    pub best_validation_epoch: Option<usize>,
    pub best_validation_mean_nll: Option<f32>,
    pub validation_nll_delta: Option<f32>,
    pub validation_accuracy_delta: Option<f32>,
    pub final_vs_unigram_nll_delta: Option<f32>,
    pub best_validation_nll_delta: Option<f32>,
    pub best_vs_unigram_nll_delta: Option<f32>,
    pub final_minus_best_validation_nll: Option<f32>,
    pub best_checkpoint_path: Option<String>,
    pub best_sample_path: Option<String>,
    pub epochs_completed: usize,
    pub early_stopped_epoch: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct TokenSplit {
    pub train: Vec<usize>,
    pub validation: Vec<usize>,
    pub actual_validation_fraction: f32,
}

#[derive(Debug, Clone)]
pub struct ParameterSnapshot {
    values: BTreeMap<String, Tensor>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ParameterLearnabilityMetric {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub elements: usize,
    pub value_l2: f32,
    pub value_rms: f32,
    pub value_mean_abs: f32,
    pub value_max_abs: f32,
    pub update_l2: Option<f32>,
    pub update_rms: Option<f32>,
    pub update_to_value_l2: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LearnabilityMetric {
    pub parameters: Vec<ParameterLearnabilityMetric>,
    pub parameter_count: usize,
    pub total_value_l2: f32,
    pub total_update_l2: Option<f32>,
    pub max_update_to_value_l2: Option<f32>,
    pub mean_update_to_value_l2: Option<f32>,
    pub train_loss: Option<f32>,
    pub validation_nll: Option<f32>,
    pub validation_entropy: Option<f32>,
    pub validation_top_probability: Option<f32>,
    pub validation_target_probability: Option<f32>,
    pub validation_target_logprob_lift: Option<f32>,
    pub validation_target_rank_lift: Option<f32>,
    pub validation_kl_to_unigram: Option<f32>,
}

pub const HEAD_PRIOR_NONE: &str = "none";
pub const HEAD_PRIOR_UNIGRAM: &str = "unigram";
pub const HEAD_PRIOR_LEARNED_UNIGRAM: &str = "learned-unigram";
pub const CHAR_FEATURE_TOKEN: &str = "token";
pub const CHAR_FEATURE_TOKEN_BIGRAM: &str = "token-bigram";
pub const DEFAULT_CHAR_FEATURE: &str = CHAR_FEATURE_TOKEN_BIGRAM;
pub const DEFAULT_HEAD_RESIDUAL_SCALE: f32 = 1.0;

pub fn validate_char_feature(value: &str) -> PureResult<()> {
    match value {
        CHAR_FEATURE_TOKEN | CHAR_FEATURE_TOKEN_BIGRAM => Ok(()),
        _ => Err(TensorError::Generic(format!(
            "invalid --char-feature {value}; expected {CHAR_FEATURE_TOKEN} or {CHAR_FEATURE_TOKEN_BIGRAM}"
        ))),
    }
}

fn token_to_index(value: f32, vocab_size: usize) -> usize {
    if vocab_size == 0 || !value.is_finite() {
        return 0;
    }
    let rounded = value.round();
    if !rounded.is_finite() {
        return 0;
    }
    let idx = rounded as isize;
    if idx <= 0 {
        return 0;
    }
    let max = vocab_size.saturating_sub(1) as isize;
    idx.min(max) as usize
}

pub fn validate_head_prior(value: &str) -> PureResult<()> {
    match value {
        HEAD_PRIOR_NONE | HEAD_PRIOR_UNIGRAM | HEAD_PRIOR_LEARNED_UNIGRAM => Ok(()),
        _ => Err(TensorError::Generic(format!(
            "invalid --head-prior {value}; expected {HEAD_PRIOR_NONE}, {HEAD_PRIOR_UNIGRAM}, or {HEAD_PRIOR_LEARNED_UNIGRAM}"
        ))),
    }
}

pub fn head_prior_is_enabled(value: &str) -> bool {
    matches!(value, HEAD_PRIOR_UNIGRAM | HEAD_PRIOR_LEARNED_UNIGRAM)
}

#[derive(Debug)]
pub struct CharFeatureEmbedding {
    token: Embedding,
    bigram: Parameter,
    vocab_size: usize,
    embed_dim: usize,
}

impl CharFeatureEmbedding {
    pub fn new(name: impl Into<String>, vocab_size: usize, embed_dim: usize) -> PureResult<Self> {
        if vocab_size == 0 || embed_dim == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: vocab_size.max(1),
                cols: embed_dim.max(1),
            });
        }
        let name = name.into();
        let bigram_rows =
            vocab_size
                .checked_mul(vocab_size)
                .ok_or(TensorError::InvalidDimensions {
                    rows: vocab_size,
                    cols: vocab_size,
                })?;
        Ok(Self {
            token: Embedding::new(name.clone(), vocab_size, embed_dim)?,
            bigram: Parameter::new(
                format!("{name}::bigram_weight"),
                Tensor::zeros(bigram_rows, embed_dim)?,
            ),
            vocab_size,
            embed_dim,
        })
    }

    fn bigram_index(&self, previous: usize, current: usize) -> usize {
        previous * self.vocab_size + current
    }
}

impl Module for CharFeatureEmbedding {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (batch, steps) = input.shape();
        let output_cols = steps * self.embed_dim;
        let mut output = self.token.forward(input)?;
        if output.shape() != (batch, output_cols) {
            return Err(TensorError::ShapeMismatch {
                left: output.shape(),
                right: (batch, output_cols),
            });
        }
        if batch == 0 || steps == 0 {
            return Ok(output);
        }

        let input_data = input.data();
        let bigram_data = self.bigram.value().data();
        let output_data = output.data_mut();
        for b in 0..batch {
            let input_row = b * steps;
            let output_row = b * output_cols;
            let mut previous = 0usize;
            for t in 0..steps {
                let current = token_to_index(input_data[input_row + t], self.vocab_size);
                let bigram_base = self.bigram_index(previous, current) * self.embed_dim;
                let output_base = output_row + t * self.embed_dim;
                for c in 0..self.embed_dim {
                    output_data[output_base + c] += bigram_data[bigram_base + c];
                }
                previous = current;
            }
        }
        Ok(output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let (batch, steps) = input.shape();
        let output_cols = steps * self.embed_dim;
        if grad_output.shape() != (batch, output_cols) {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (batch, output_cols),
            });
        }
        let grad_input = self.token.backward(input, grad_output)?;
        if batch == 0 || steps == 0 {
            return Ok(grad_input);
        }

        let input_data = input.data();
        let grad_data = grad_output.data();
        let mut grad_bigram = vec![0.0f32; self.vocab_size * self.vocab_size * self.embed_dim];
        for b in 0..batch {
            let input_row = b * steps;
            let grad_row = b * output_cols;
            let mut previous = 0usize;
            for t in 0..steps {
                let current = token_to_index(input_data[input_row + t], self.vocab_size);
                let gb_base = self.bigram_index(previous, current) * self.embed_dim;
                let go_base = grad_row + t * self.embed_dim;
                for c in 0..self.embed_dim {
                    grad_bigram[gb_base + c] += grad_data[go_base + c];
                }
                previous = current;
            }
        }
        let grad_w = Tensor::from_vec(
            self.vocab_size * self.vocab_size,
            self.embed_dim,
            grad_bigram,
        )?
        .scale(1.0 / batch as f32)?;
        self.bigram.accumulate_euclidean(&grad_w)?;
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.token.visit_parameters(visitor)?;
        visitor(&self.bigram)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.token.visit_parameters_mut(visitor)?;
        visitor(&mut self.bigram)?;
        Ok(())
    }
}

pub fn push_char_embedding(
    model: &mut Sequential,
    name: &str,
    vocab_size: usize,
    embed_dim: usize,
    char_feature: &str,
) -> PureResult<()> {
    match char_feature {
        CHAR_FEATURE_TOKEN => model.push(Embedding::new(name, vocab_size, embed_dim)?),
        CHAR_FEATURE_TOKEN_BIGRAM => {
            model.push(CharFeatureEmbedding::new(name, vocab_size, embed_dim)?)
        }
        _ => validate_char_feature(char_feature)?,
    }
    Ok(())
}

pub fn linear_with_weight_rms(
    name: &str,
    input_dim: usize,
    output_dim: usize,
    target_rms: f32,
) -> PureResult<Linear> {
    if target_rms <= 0.0 || !target_rms.is_finite() {
        return Err(TensorError::NonFiniteValue {
            label: "char_lm_linear_weight_rms",
            value: target_rms,
        });
    }
    let mut layer = Linear::new(name, input_dim, output_dim)?;
    let weight_name = format!("{name}::weight");
    layer.visit_parameters_mut(&mut |param| {
        if param.name() != weight_name {
            return Ok(());
        }
        let current = param.value().squared_l2_norm().sqrt()
            / (param.value().data().len() as f32).sqrt().max(1.0);
        if current <= f32::EPSILON || !current.is_finite() {
            return Ok(());
        }
        let scale = target_rms / current;
        for value in param.value_mut().data_mut() {
            *value *= scale;
        }
        Ok(())
    })?;
    Ok(layer)
}

pub fn validate_head_residual_scale(value: f32) -> PureResult<()> {
    if value <= 0.0 || !value.is_finite() {
        return Err(TensorError::NonFiniteValue {
            label: "char_lm_head_residual_scale",
            value,
        });
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct FixedLogitScale {
    _name: String,
    scale: f32,
}

impl FixedLogitScale {
    pub fn new(name: impl Into<String>, scale: f32) -> PureResult<Self> {
        validate_head_residual_scale(scale)?;
        Ok(Self {
            _name: name.into(),
            scale,
        })
    }
}

impl Module for FixedLogitScale {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        input.scale(self.scale)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        if input.shape() != grad_output.shape() {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: grad_output.shape(),
            });
        }
        grad_output.scale(self.scale)
    }

    fn visit_parameters(
        &self,
        _visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        _visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }
}

pub fn residual_logit_scaler(
    name: impl Into<String>,
    _vocab_size: usize,
    scale: f32,
) -> PureResult<FixedLogitScale> {
    FixedLogitScale::new(name, scale)
}

fn smoothed_unigram_probabilities(vocab_size: usize, train_tokens: &[usize]) -> Vec<f32> {
    if vocab_size == 0 {
        return Vec::new();
    }
    let mut counts = vec![1.0f32; vocab_size];
    for &token in train_tokens {
        if token < vocab_size {
            counts[token] += 1.0;
        }
    }
    let total = counts.iter().sum::<f32>().max(f32::EPSILON);
    counts.into_iter().map(|count| count / total).collect()
}

#[derive(Debug, Clone)]
pub struct FixedLogitPrior {
    name: String,
    prior: Tensor,
}

impl FixedLogitPrior {
    pub fn from_unigram(
        name: impl Into<String>,
        vocab_size: usize,
        train_tokens: &[usize],
    ) -> PureResult<Self> {
        let probabilities = smoothed_unigram_probabilities(vocab_size, train_tokens);
        let values = probabilities
            .into_iter()
            .map(|probability| probability.clamp(1.0e-9, 1.0).ln())
            .collect::<Vec<_>>();
        Self::from_values(name, Tensor::from_vec(1, vocab_size, values)?)
    }

    pub fn zeros(name: impl Into<String>, vocab_size: usize) -> PureResult<Self> {
        Self::from_values(name, Tensor::zeros(1, vocab_size)?)
    }

    fn from_values(name: impl Into<String>, prior: Tensor) -> PureResult<Self> {
        let (rows, cols) = prior.shape();
        if rows != 1 || cols == 0 {
            return Err(TensorError::ShapeMismatch {
                left: (rows, cols),
                right: (1, cols),
            });
        }
        Ok(Self {
            name: name.into(),
            prior,
        })
    }

    fn key(&self) -> String {
        format!("{}::prior", self.name)
    }
}

impl Module for FixedLogitPrior {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if self.prior.shape() != (1, cols) {
            return Err(TensorError::ShapeMismatch {
                left: self.prior.shape(),
                right: (1, cols),
            });
        }
        let mut output = input.clone();
        if rows > 0 {
            output.add_row_inplace(self.prior.data())?;
        }
        Ok(output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        if input.shape() != grad_output.shape() {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: grad_output.shape(),
            });
        }
        Ok(grad_output.clone())
    }

    fn visit_parameters(
        &self,
        _visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        _visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }

    fn state_dict(&self) -> PureResult<HashMap<String, Tensor>> {
        Ok(HashMap::from([(self.key(), self.prior.clone())]))
    }

    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>) -> PureResult<()> {
        let key = self.key();
        let Some(value) = state.get(&key) else {
            return Err(TensorError::MissingParameter { name: key });
        };
        if value.shape() != self.prior.shape() {
            return Err(TensorError::ShapeMismatch {
                left: value.shape(),
                right: self.prior.shape(),
            });
        }
        self.prior = value.clone();
        Ok(())
    }
}

#[derive(Debug)]
pub struct LearnedLogitPrior {
    name: String,
    base: Tensor,
    delta: Parameter,
}

impl LearnedLogitPrior {
    pub fn from_unigram(
        name: impl Into<String>,
        vocab_size: usize,
        train_tokens: &[usize],
    ) -> PureResult<Self> {
        let probabilities = smoothed_unigram_probabilities(vocab_size, train_tokens);
        let values = probabilities
            .into_iter()
            .map(|probability| probability.clamp(1.0e-9, 1.0).ln())
            .collect::<Vec<_>>();
        Self::from_values(name, Tensor::from_vec(1, vocab_size, values)?)
    }

    pub fn zeros(name: impl Into<String>, vocab_size: usize) -> PureResult<Self> {
        Self::from_values(name, Tensor::zeros(1, vocab_size)?)
    }

    fn from_values(name: impl Into<String>, prior: Tensor) -> PureResult<Self> {
        let (rows, cols) = prior.shape();
        if rows != 1 || cols == 0 {
            return Err(TensorError::ShapeMismatch {
                left: (rows, cols),
                right: (1, cols),
            });
        }
        let name = name.into();
        let delta = Tensor::zeros(rows, cols)?;
        Ok(Self {
            delta: Parameter::new(format!("{name}::delta"), delta),
            name,
            base: prior,
        })
    }

    fn key(&self) -> String {
        format!("{}::prior", self.name)
    }

    fn combined_prior(&self) -> PureResult<Tensor> {
        self.base.add(self.delta.value())
    }
}

impl Module for LearnedLogitPrior {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if self.base.shape() != (1, cols) || self.delta.value().shape() != (1, cols) {
            return Err(TensorError::ShapeMismatch {
                left: self.base.shape(),
                right: (1, cols),
            });
        }
        let mut output = input.clone();
        if rows > 0 {
            let prior = self.combined_prior()?;
            output.add_row_inplace(prior.data())?;
        }
        Ok(output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        if input.shape() != grad_output.shape() {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: grad_output.shape(),
            });
        }
        let (rows, cols) = grad_output.shape();
        let summed = grad_output.sum_axis0();
        let grad_prior = Tensor::from_vec(1, cols, summed)?.scale(1.0 / rows.max(1) as f32)?;
        self.delta.accumulate_euclidean(&grad_prior)?;
        Ok(grad_output.clone())
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.delta)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.delta)?;
        Ok(())
    }

    fn state_dict(&self) -> PureResult<HashMap<String, Tensor>> {
        Ok(HashMap::from([(self.key(), self.combined_prior()?)]))
    }

    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>) -> PureResult<()> {
        let key = self.key();
        let Some(value) = state.get(&key) else {
            return Err(TensorError::MissingParameter { name: key });
        };
        if value.shape() != self.base.shape() {
            return Err(TensorError::ShapeMismatch {
                left: value.shape(),
                right: self.base.shape(),
            });
        }
        self.base = value.clone();
        self.delta
            .load_value(&Tensor::zeros(value.shape().0, value.shape().1)?)?;
        Ok(())
    }
}

pub fn insert_head_prior(
    model: &mut Sequential,
    head_prior: &str,
    vocab_size: usize,
    train_tokens: Option<&[usize]>,
) -> PureResult<()> {
    let index = model.len().saturating_sub(1);
    match head_prior {
        HEAD_PRIOR_NONE => Ok(()),
        HEAD_PRIOR_UNIGRAM => {
            if let Some(tokens) = train_tokens {
                model.insert(
                    index,
                    FixedLogitPrior::from_unigram("head_prior", vocab_size, tokens)?,
                )
            } else {
                model.insert(index, FixedLogitPrior::zeros("head_prior", vocab_size)?)
            }
        }
        HEAD_PRIOR_LEARNED_UNIGRAM => {
            if let Some(tokens) = train_tokens {
                model.insert(
                    index,
                    LearnedLogitPrior::from_unigram("head_prior", vocab_size, tokens)?,
                )
            } else {
                model.insert(index, LearnedLogitPrior::zeros("head_prior", vocab_size)?)
            }
        }
        _ => validate_head_prior(head_prior),
    }
}

pub fn split_train_validation_tokens(
    tokens: &[usize],
    steps: usize,
    validation_fraction: f32,
) -> TokenSplit {
    let min_window_tokens = steps + 1;
    if validation_fraction <= 0.0 || tokens.len() < min_window_tokens * 2 {
        return TokenSplit {
            train: tokens.to_vec(),
            validation: Vec::new(),
            actual_validation_fraction: 0.0,
        };
    }

    let requested = (tokens.len() as f32 * validation_fraction).round() as usize;
    let max_validation = tokens.len() - min_window_tokens;
    let validation_len = requested.clamp(min_window_tokens, max_validation);
    let split_at = tokens.len() - validation_len;
    let validation = tokens[split_at..].to_vec();
    let actual_validation_fraction = validation.len() as f32 / tokens.len() as f32;

    TokenSplit {
        train: tokens[..split_at].to_vec(),
        validation,
        actual_validation_fraction,
    }
}

pub fn encode_context_one_hot(context: &[usize], vocab_size: usize) -> PureResult<Tensor> {
    let steps = context.len();
    let cols = vocab_size * steps;
    let mut x = Tensor::zeros(1, cols)?;
    let data = x.data_mut();
    for (t, &idx) in context.iter().enumerate() {
        if idx < vocab_size {
            data[t * vocab_size + idx] = 1.0;
        }
    }
    Ok(x)
}

pub fn encode_context_indices(context: &[usize]) -> PureResult<Tensor> {
    let mut data = Vec::with_capacity(context.len());
    for &idx in context {
        data.push(idx as f32);
    }
    Tensor::from_vec(1, context.len(), data)
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &value) in values.iter().enumerate() {
        if value.is_finite() && value > best_val {
            best = idx;
            best_val = value;
        }
    }
    best
}

fn target_rank(values: &[f32], target: usize) -> usize {
    let Some(&target_value) = values.get(target) else {
        return values.len().max(1);
    };
    1 + values
        .iter()
        .enumerate()
        .filter(|(idx, value)| *idx != target && value.is_finite() && **value > target_value)
        .count()
}

fn top_k_indices(values: &[f32], k: usize) -> Vec<usize> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_by(|&left, &right| {
        values[right]
            .partial_cmp(&values[left])
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(&right))
    });
    indices.truncate(k.min(indices.len()));
    indices
}

fn top_k_overlap_ratio(left: &[f32], right: &[f32], k: usize) -> f32 {
    let k = k.min(left.len()).min(right.len());
    if k == 0 {
        return 0.0;
    }
    let left_top = top_k_indices(left, k);
    let right_top = top_k_indices(right, k);
    let overlap = left_top
        .iter()
        .filter(|index| right_top.contains(index))
        .count();
    overlap as f32 / k as f32
}

fn kl_to_unigram(probabilities: &[f32], unigram_probabilities: &[f32]) -> f32 {
    probabilities
        .iter()
        .zip(unigram_probabilities.iter())
        .filter_map(|(&probability, &unigram_probability)| {
            let probability = probability.clamp(0.0, 1.0);
            if probability <= 0.0 {
                return None;
            }
            let unigram_probability = unigram_probability.clamp(1.0e-9, 1.0);
            Some(probability * (probability.clamp(1.0e-9, 1.0).ln() - unigram_probability.ln()))
        })
        .sum()
}

fn l2(values: &[f32]) -> f32 {
    values
        .iter()
        .map(|&value| {
            let v = value as f64;
            v * v
        })
        .sum::<f64>()
        .sqrt() as f32
}

fn mean_abs(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().map(|value| value.abs()).sum::<f32>() / values.len() as f32
}

fn max_abs(values: &[f32]) -> f32 {
    values
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max)
}

fn entropy(values: &[f32]) -> f32 {
    let mut out = 0.0f32;
    for &value in values {
        let probability = value.clamp(0.0, 1.0);
        if probability > 0.0 {
            out -= probability * probability.ln();
        }
    }
    out
}

#[derive(Debug, Clone, Copy, Default)]
struct ContextLiftSums {
    target_logprob: f32,
    target_probability: f32,
    target_rank: f32,
    kl_to_unigram: f32,
    top5_overlap_with_unigram: f32,
}

fn metric_from_sums(
    tokens: usize,
    windows: usize,
    vocab_size: usize,
    nll_sum: f32,
    correct: usize,
    target_probability_sum: f32,
    top_probability_sum: f32,
    entropy_sum: f32,
    target_rank_sum: f32,
    lift_sums: Option<ContextLiftSums>,
) -> LanguageEvalMetric {
    let mean_nll = nll_sum / windows as f32;
    let perplexity = if mean_nll.is_finite() && mean_nll < 80.0 {
        Some(mean_nll.exp())
    } else {
        None
    };
    let mean_target_rank = target_rank_sum / windows as f32;
    LanguageEvalMetric {
        tokens,
        windows,
        mean_nll,
        perplexity,
        accuracy: correct as f32 / windows as f32,
        mean_target_probability: target_probability_sum / windows as f32,
        mean_top_probability: top_probability_sum / windows as f32,
        mean_entropy: entropy_sum / windows as f32,
        mean_normalized_entropy: if vocab_size <= 1 {
            0.0
        } else {
            (entropy_sum / windows as f32) / (vocab_size as f32).ln()
        },
        mean_target_rank,
        mean_target_percentile: if vocab_size <= 1 {
            1.0
        } else {
            1.0 - ((mean_target_rank - 1.0) / (vocab_size - 1) as f32)
        },
        mean_target_logprob_lift: lift_sums.map(|sums| sums.target_logprob / windows as f32),
        mean_target_probability_lift: lift_sums
            .map(|sums| sums.target_probability / windows as f32),
        mean_target_rank_lift: lift_sums.map(|sums| sums.target_rank / windows as f32),
        mean_kl_to_unigram: lift_sums.map(|sums| sums.kl_to_unigram / windows as f32),
        mean_top5_overlap_with_unigram: lift_sums
            .map(|sums| sums.top5_overlap_with_unigram / windows as f32),
    }
}

pub fn evaluate_unigram_next_token(
    vocab_size: usize,
    steps: usize,
    train_tokens: &[usize],
    validation_tokens: &[usize],
    max_samples: usize,
) -> PureResult<Option<LanguageEvalMetric>> {
    if validation_tokens.len() <= steps || vocab_size == 0 {
        return Ok(None);
    }

    let probabilities = smoothed_unigram_probabilities(vocab_size, train_tokens);

    let available = validation_tokens.len() - steps;
    let windows = if max_samples == 0 {
        available
    } else {
        available.min(max_samples)
    };
    if windows == 0 {
        return Ok(None);
    }

    let top = argmax(&probabilities);
    let top_probability = probabilities[top].clamp(0.0, 1.0);
    let distribution_entropy = entropy(&probabilities);
    let mut nll_sum = 0.0f32;
    let mut correct = 0usize;
    let mut target_probability_sum = 0.0f32;
    let mut target_rank_sum = 0.0f32;

    for sample_idx in 0..windows {
        let start = if windows <= 1 || available <= 1 {
            0
        } else {
            sample_idx * (available - 1) / (windows - 1)
        };
        let target = validation_tokens[start + steps];
        let target_probability = probabilities
            .get(target)
            .copied()
            .unwrap_or(0.0)
            .clamp(1.0e-9, 1.0);
        if top == target {
            correct += 1;
        }
        nll_sum += -target_probability.ln();
        target_probability_sum += target_probability;
        target_rank_sum += target_rank(&probabilities, target) as f32;
    }

    Ok(Some(metric_from_sums(
        validation_tokens.len(),
        windows,
        vocab_size,
        nll_sum,
        correct,
        target_probability_sum,
        top_probability * windows as f32,
        distribution_entropy * windows as f32,
        target_rank_sum,
        None,
    )))
}

pub fn capture_parameter_snapshot<M>(model: &M) -> PureResult<ParameterSnapshot>
where
    M: Module + ?Sized,
{
    let mut values = BTreeMap::new();
    model.visit_parameters(&mut |param| {
        values.insert(param.name().to_string(), param.value().clone());
        Ok(())
    })?;
    Ok(ParameterSnapshot { values })
}

pub fn summarize_learnability<M>(
    model: &M,
    before: Option<&ParameterSnapshot>,
    train_loss: Option<f32>,
    validation: Option<&LanguageEvalMetric>,
) -> PureResult<LearnabilityMetric>
where
    M: Module + ?Sized,
{
    let mut parameters = Vec::new();
    let mut total_value_sq = 0.0f64;
    let mut total_update_sq = 0.0f64;
    let mut update_ratios = Vec::new();
    let mut saw_update = false;

    model.visit_parameters(&mut |param| {
        let value = param.value();
        let (rows, cols) = value.shape();
        let values = value.data();
        let elements = values.len();
        let value_l2 = l2(values);
        total_value_sq += (value_l2 as f64) * (value_l2 as f64);

        let mut update_l2 = None;
        let mut update_rms = None;
        let mut update_to_value_l2 = None;
        if let Some(previous) = before.and_then(|snapshot| snapshot.values.get(param.name())) {
            if previous.shape() == value.shape() {
                let mut update_sq = 0.0f64;
                for (&after, &before) in values.iter().zip(previous.data().iter()) {
                    let delta = (after - before) as f64;
                    update_sq += delta * delta;
                }
                let update = update_sq.sqrt() as f32;
                let ratio = update / value_l2.max(1.0e-12);
                update_l2 = Some(update);
                update_rms = Some(if elements == 0 {
                    0.0
                } else {
                    update / (elements as f32).sqrt()
                });
                update_to_value_l2 = Some(ratio);
                total_update_sq += update_sq;
                update_ratios.push(ratio);
                saw_update = true;
            }
        }

        parameters.push(ParameterLearnabilityMetric {
            name: param.name().to_string(),
            rows,
            cols,
            elements,
            value_l2,
            value_rms: if elements == 0 {
                0.0
            } else {
                value_l2 / (elements as f32).sqrt()
            },
            value_mean_abs: mean_abs(values),
            value_max_abs: max_abs(values),
            update_l2,
            update_rms,
            update_to_value_l2,
        });
        Ok(())
    })?;

    let total_update_l2 = saw_update.then(|| total_update_sq.sqrt() as f32);
    let max_update_to_value_l2 = update_ratios.iter().copied().reduce(f32::max);
    let mean_update_to_value_l2 = if update_ratios.is_empty() {
        None
    } else {
        Some(update_ratios.iter().sum::<f32>() / update_ratios.len() as f32)
    };

    Ok(LearnabilityMetric {
        parameter_count: parameters.len(),
        parameters,
        total_value_l2: total_value_sq.sqrt() as f32,
        total_update_l2,
        max_update_to_value_l2,
        mean_update_to_value_l2,
        train_loss,
        validation_nll: validation.map(|metric| metric.mean_nll),
        validation_entropy: validation.map(|metric| metric.mean_entropy),
        validation_top_probability: validation.map(|metric| metric.mean_top_probability),
        validation_target_probability: validation.map(|metric| metric.mean_target_probability),
        validation_target_logprob_lift: validation
            .and_then(|metric| metric.mean_target_logprob_lift),
        validation_target_rank_lift: validation.and_then(|metric| metric.mean_target_rank_lift),
        validation_kl_to_unigram: validation.and_then(|metric| metric.mean_kl_to_unigram),
    })
}

#[allow(dead_code)]
pub fn evaluate_next_token<M>(
    model: &mut M,
    vocab_size: usize,
    input_mode: CharLmInputMode,
    steps: usize,
    tokens: &[usize],
    max_samples: usize,
) -> PureResult<Option<LanguageEvalMetric>>
where
    M: Module + ?Sized,
{
    evaluate_next_token_inner(
        model,
        vocab_size,
        input_mode,
        steps,
        None,
        tokens,
        max_samples,
    )
}

pub fn evaluate_next_token_with_unigram_lift<M>(
    model: &mut M,
    vocab_size: usize,
    input_mode: CharLmInputMode,
    steps: usize,
    train_tokens: &[usize],
    tokens: &[usize],
    max_samples: usize,
) -> PureResult<Option<LanguageEvalMetric>>
where
    M: Module + ?Sized,
{
    evaluate_next_token_inner(
        model,
        vocab_size,
        input_mode,
        steps,
        Some(train_tokens),
        tokens,
        max_samples,
    )
}

fn evaluate_next_token_inner<M>(
    model: &mut M,
    vocab_size: usize,
    input_mode: CharLmInputMode,
    steps: usize,
    train_tokens: Option<&[usize]>,
    tokens: &[usize],
    max_samples: usize,
) -> PureResult<Option<LanguageEvalMetric>>
where
    M: Module + ?Sized,
{
    if tokens.len() <= steps {
        return Ok(None);
    }
    let available = tokens.len() - steps;
    let windows = if max_samples == 0 {
        available
    } else {
        available.min(max_samples)
    };
    if windows == 0 {
        return Ok(None);
    }
    let unigram_probabilities =
        train_tokens.map(|train_tokens| smoothed_unigram_probabilities(vocab_size, train_tokens));

    model.eval()?;
    let result = (|| -> PureResult<LanguageEvalMetric> {
        let mut nll_sum = 0.0f32;
        let mut correct = 0usize;
        let mut target_probability_sum = 0.0f32;
        let mut top_probability_sum = 0.0f32;
        let mut entropy_sum = 0.0f32;
        let mut target_rank_sum = 0.0f32;
        let mut lift_sums = unigram_probabilities
            .as_ref()
            .map(|_| ContextLiftSums::default());

        for sample_idx in 0..windows {
            let start = if windows <= 1 || available <= 1 {
                0
            } else {
                sample_idx * (available - 1) / (windows - 1)
            };
            let context = &tokens[start..start + steps];
            let target = tokens[start + steps];
            let x = match input_mode {
                CharLmInputMode::TokenIndices => encode_context_indices(context)?,
                CharLmInputMode::OneHot => encode_context_one_hot(context, vocab_size)?,
            };
            let prediction = model.forward(&x)?;
            if prediction.shape() != (1, vocab_size) {
                return Err(TensorError::ShapeMismatch {
                    left: prediction.shape(),
                    right: (1, vocab_size),
                });
            }
            let row = &prediction.data()[..vocab_size];
            let top = argmax(row);
            let top_probability = row[top].clamp(0.0, 1.0);
            let target_probability = row.get(target).copied().unwrap_or(0.0).clamp(1.0e-9, 1.0);
            let rank = target_rank(row, target);

            if top == target {
                correct += 1;
            }
            nll_sum += -target_probability.ln();
            target_probability_sum += target_probability;
            top_probability_sum += top_probability;
            entropy_sum += entropy(row);
            target_rank_sum += rank as f32;
            if let (Some(unigram_probabilities), Some(lift_sums)) =
                (unigram_probabilities.as_ref(), lift_sums.as_mut())
            {
                let unigram_target_probability = unigram_probabilities
                    .get(target)
                    .copied()
                    .unwrap_or(0.0)
                    .clamp(1.0e-9, 1.0);
                lift_sums.target_logprob +=
                    target_probability.ln() - unigram_target_probability.ln();
                lift_sums.target_probability += target_probability - unigram_target_probability;
                lift_sums.target_rank +=
                    target_rank(unigram_probabilities, target) as f32 - rank as f32;
                lift_sums.kl_to_unigram += kl_to_unigram(row, unigram_probabilities);
                lift_sums.top5_overlap_with_unigram +=
                    top_k_overlap_ratio(row, unigram_probabilities, 5);
            }
        }

        Ok(metric_from_sums(
            tokens.len(),
            windows,
            vocab_size,
            nll_sum,
            correct,
            target_probability_sum,
            top_probability_sum,
            entropy_sum,
            target_rank_sum,
            lift_sums,
        ))
    })();
    let restore = model.train();
    match (result, restore) {
        (Ok(metric), Ok(())) => Ok(Some(metric)),
        (Err(err), _) => Err(err),
        (Ok(_), Err(err)) => Err(err),
    }
}

pub fn write_summary(path: &Path, summary: &TrainingSummary) -> PureResult<()> {
    let writer = File::create(path).map_err(|err| TensorError::IoError {
        message: err.to_string(),
    })?;
    serde_json::to_writer_pretty(writer, summary).map_err(|err| {
        TensorError::SerializationError {
            message: err.to_string(),
        }
    })?;
    Ok(())
}
