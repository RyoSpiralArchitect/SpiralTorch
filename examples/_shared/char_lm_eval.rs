// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

use serde::Serialize;
use st_nn::{Module, PureResult, Tensor, TensorError};
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
