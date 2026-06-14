// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

use serde::Serialize;
use st_nn::{Linear, Module, PureResult, Tensor, TensorError};
use std::collections::BTreeMap;
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
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainingSummary {
    pub initial_validation: Option<LanguageEvalMetric>,
    pub final_validation: Option<LanguageEvalMetric>,
    pub best_validation_epoch: Option<usize>,
    pub best_validation_mean_nll: Option<f32>,
    pub validation_nll_delta: Option<f32>,
    pub validation_accuracy_delta: Option<f32>,
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
    })
}

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

    model.eval()?;
    let result = (|| -> PureResult<LanguageEvalMetric> {
        let mut nll_sum = 0.0f32;
        let mut correct = 0usize;
        let mut target_probability_sum = 0.0f32;
        let mut top_probability_sum = 0.0f32;
        let mut entropy_sum = 0.0f32;

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

            if top == target {
                correct += 1;
            }
            nll_sum += -target_probability.ln();
            target_probability_sum += target_probability;
            top_probability_sum += top_probability;
            entropy_sum += entropy(row);
        }

        let mean_nll = nll_sum / windows as f32;
        let perplexity = if mean_nll.is_finite() && mean_nll < 80.0 {
            Some(mean_nll.exp())
        } else {
            None
        };
        Ok(LanguageEvalMetric {
            tokens: tokens.len(),
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
        })
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
