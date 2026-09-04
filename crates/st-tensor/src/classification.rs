// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Ryo SpiralArchitect

//! Stable classification kernels shared by native training, autograd and bindings.
//! Inputs and results are f32; CPU row normalization and reductions use f64.

use crate::{emit_tensor_op, emit_tensor_op_meta, Layout, PureResult, Tensor, TensorError};
use std::str::FromStr;

/// Reduction over non-ignored samples, not over classes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LossReduction {
    None,
    Sum,
    #[default]
    Mean,
}

impl LossReduction {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Sum => "sum",
            Self::Mean => "mean",
        }
    }
}

impl FromStr for LossReduction {
    type Err = TensorError;

    fn from_str(value: &str) -> PureResult<Self> {
        match value {
            "none" => Ok(Self::None),
            "sum" => Ok(Self::Sum),
            "mean" => Ok(Self::Mean),
            _ => Err(TensorError::InvalidValue {
                label: "loss reduction must be none, sum or mean",
            }),
        }
    }
}

/// Integer-label cross entropy. Smoothing mixes the target with a uniform distribution.
/// Mean reduction rejects a batch with no non-ignored labels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrossEntropyConfig {
    pub reduction: LossReduction,
    pub ignore_index: i64,
    pub label_smoothing: f64,
}

impl Default for CrossEntropyConfig {
    fn default() -> Self {
        Self {
            reduction: LossReduction::Mean,
            ignore_index: -100,
            label_smoothing: 0.0,
        }
    }
}

impl CrossEntropyConfig {
    pub fn validate(self) -> PureResult<()> {
        if !self.label_smoothing.is_finite() || !(0.0..=1.0).contains(&self.label_smoothing) {
            return Err(TensorError::InvalidValue {
                label: "label_smoothing must be finite and between zero and one",
            });
        }
        Ok(())
    }

    pub fn output_shape(self, rows: usize) -> (usize, usize) {
        match self.reduction {
            LossReduction::None => (rows, 1),
            LossReduction::Sum | LossReduction::Mean => (1, 1),
        }
    }
}

/// Decodes the `(samples, 1)` integral target transport used by `st-nn::Loss`.
/// Prefer integer slices directly when labels originate outside a tensor loader.
pub fn class_indices_from_tensor(target: &Tensor) -> PureResult<Vec<i64>> {
    if target.shape().1 != 1 {
        return Err(TensorError::ShapeMismatch {
            left: target.shape(),
            right: (target.shape().0, 1),
        });
    }
    let target = finite_row_major(target, "class_index_target")?;
    target
        .data()
        .iter()
        .map(|&value| {
            let wide = f64::from(value);
            // i64::MAX rounds up in f64, so the upper bound is exclusive.
            if wide.fract() != 0.0 || wide < i64::MIN as f64 || wide >= -(i64::MIN as f64) {
                Err(TensorError::InvalidValue {
                    label: "class index target must be an integer representable as i64",
                })
            } else {
                Ok(value as i64)
            }
        })
        .collect()
}

struct RowPartition {
    maximum: f64,
    maximum_index: usize,
    tail: f64,
}

impl RowPartition {
    fn new(row: &[f32]) -> Self {
        let mut maximum_index = 0;
        for index in 1..row.len() {
            if row[index] > row[maximum_index] {
                maximum_index = index;
            }
        }
        let maximum = f64::from(row[maximum_index]);
        // Excluding one maximum preserves tiny tails with ln_1p instead of ln(1 + tail).
        let tail = row
            .iter()
            .enumerate()
            .filter(|&(index, _)| index != maximum_index)
            .map(|(_, &value)| (f64::from(value) - maximum).exp())
            .sum();
        Self {
            maximum,
            maximum_index,
            tail,
        }
    }

    fn log_partition(&self) -> f64 {
        self.tail.ln_1p()
    }

    fn tail_probability(&self) -> f64 {
        self.tail / (1.0 + self.tail)
    }

    fn probability(&self, value: f32) -> f64 {
        (f64::from(value) - self.maximum).exp() / (1.0 + self.tail)
    }
}

fn finite_row_major(input: &Tensor, label: &'static str) -> PureResult<Tensor> {
    for &value in input.data() {
        if !value.is_finite() {
            return Err(TensorError::NonFiniteValue { label, value });
        }
    }
    if input.layout() == Layout::RowMajor {
        Ok(input.clone())
    } else {
        input.to_layout(Layout::RowMajor)
    }
}

fn checked_f32(value: f64, label: &'static str) -> PureResult<f32> {
    let value = value as f32;
    if !value.is_finite() {
        return Err(TensorError::NonFiniteValue { label, value });
    }
    Ok(value)
}

fn sum_cotangents(values: &[f32]) -> (f64, f64) {
    let mut sum: f64 = 0.0;
    let mut correction = 0.0;
    for &value in values {
        let value = f64::from(value);
        let next = sum + value;
        // Neumaier compensation retains a small seed between large cancelling terms.
        correction += if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
    }
    // Keep the low part separate until the VJP subtracts the dominant contribution.
    (sum, correction)
}

fn validate_labels(
    logits: &Tensor,
    labels: &[i64],
    config: CrossEntropyConfig,
) -> PureResult<usize> {
    config.validate()?;
    let (rows, classes) = logits.shape();
    if classes == 0 {
        return Err(TensorError::EmptyInput("cross entropy classes"));
    }
    if labels.len() != rows {
        return Err(TensorError::DataLength {
            expected: rows,
            got: labels.len(),
        });
    }
    let mut active = 0;
    for &label in labels {
        if label == config.ignore_index {
            continue;
        }
        if usize::try_from(label).map_or(true, |index| index >= classes) {
            return Err(TensorError::InvalidValue {
                label: "cross entropy class index out of range",
            });
        }
        active += 1;
    }
    if active == 0 && config.reduction == LossReduction::Mean {
        return Err(TensorError::EmptyInput("cross entropy non-ignored labels"));
    }
    Ok(active)
}

fn emit_classification_op(
    name: &'static str,
    input: &Tensor,
    output: &Tensor,
    loss: Option<(CrossEntropyConfig, usize)>,
) {
    let (rows, cols) = input.shape();
    emit_tensor_op(name, &[rows, cols], &[output.shape().0, output.shape().1]);
    emit_tensor_op_meta(name, || {
        let mut meta = serde_json::json!({
            "backend": "cpu", "requested_backend": "cpu",
            "kernel": "stable_f64", "accumulator_dtype": "f64",
            "semantic_owner": "st-tensor", "rows": rows, "cols": cols,
            "output_rows": output.shape().0, "output_cols": output.shape().1,
        });
        if let Some((config, active)) = loss {
            meta["reduction"] = config.reduction.as_str().into();
            meta["ignore_index"] = config.ignore_index.into();
            meta["label_smoothing"] = config.label_smoothing.into();
            meta["active_rows"] = active.into();
        }
        meta
    });
}

impl Tensor {
    /// Stable CPU log-softmax over each row. Empty tensors retain their shape.
    pub fn row_log_softmax(&self) -> PureResult<Tensor> {
        let input = finite_row_major(self, "log_softmax_logits")?;
        let (rows, cols) = input.shape();
        let mut values = Vec::with_capacity(input.len());
        if cols > 0 {
            for row in input.data().chunks_exact(cols) {
                let partition = RowPartition::new(row);
                let log_partition = partition.log_partition();
                for &value in row {
                    values.push(checked_f32(
                        (f64::from(value) - partition.maximum) - log_partition,
                        "log_softmax_output",
                    )?);
                }
            }
        }
        let output = Tensor::from_vec(rows, cols, values)?;
        emit_classification_op("row_log_softmax", self, &output, None);
        Ok(output)
    }

    /// VJP with respect to these logits, not to an already normalized output.
    pub fn row_log_softmax_backward(&self, seed: &Tensor) -> PureResult<Tensor> {
        if seed.shape() != self.shape() {
            return Err(TensorError::ShapeMismatch {
                left: seed.shape(),
                right: self.shape(),
            });
        }
        let input = finite_row_major(self, "log_softmax_logits")?;
        let seed = finite_row_major(seed, "log_softmax_seed")?;
        let (rows, cols) = input.shape();
        let mut values = Vec::with_capacity(input.len());
        if cols > 0 {
            for (row, upstream) in input
                .data()
                .chunks_exact(cols)
                .zip(seed.data().chunks_exact(cols))
            {
                let partition = RowPartition::new(row);
                let (sum, correction) = sum_cotangents(upstream);
                for (&value, &gradient) in row.iter().zip(upstream) {
                    let weight = (f64::from(value) - partition.maximum).exp();
                    let gradient = f64::from(gradient);
                    // Cancel before dividing by the partition, so wide uniform rows
                    // never subtract independently rounded normalized probabilities.
                    let high = (gradient - sum * weight) + gradient * partition.tail;
                    let gradient = (high - correction * weight) / (1.0 + partition.tail);
                    values.push(checked_f32(gradient, "log_softmax_gradient")?);
                }
            }
        }
        let output = Tensor::from_vec(rows, cols, values)?;
        emit_classification_op("row_log_softmax_backward", self, &output, None);
        Ok(output)
    }

    /// Stable CPU multiclass cross entropy from unnormalized logits and integer labels.
    /// `None` returns `(rows, 1)`; `Sum` and `Mean` return `(1, 1)`.
    pub fn cross_entropy_with_logits(
        &self,
        labels: &[i64],
        config: CrossEntropyConfig,
    ) -> PureResult<Tensor> {
        let active = validate_labels(self, labels, config)?;
        let input = finite_row_major(self, "cross_entropy_logits")?;
        let (_, classes) = input.shape();
        let mut values = Vec::with_capacity(if config.reduction == LossReduction::None {
            labels.len()
        } else {
            1
        });
        let mut total = 0.0;
        for (row, &label) in input.data().chunks_exact(classes).zip(labels) {
            let loss = if label == config.ignore_index {
                0.0
            } else {
                let partition = RowPartition::new(row);
                let nll = partition.maximum - f64::from(row[label as usize]);
                let uniform = if config.label_smoothing == 0.0 {
                    0.0
                } else {
                    row.iter()
                        .map(|&value| partition.maximum - f64::from(value))
                        .sum::<f64>()
                        / classes as f64
                };
                (1.0 - config.label_smoothing) * nll
                    + config.label_smoothing * uniform
                    + partition.log_partition()
            };
            if config.reduction == LossReduction::None {
                values.push(checked_f32(loss, "cross_entropy_output")?);
            } else {
                total += loss;
            }
        }
        if config.reduction != LossReduction::None {
            if config.reduction == LossReduction::Mean {
                total /= active as f64;
            }
            values.push(checked_f32(total, "cross_entropy_output")?);
        }
        let shape = config.output_shape(labels.len());
        let output = Tensor::from_vec(shape.0, shape.1, values)?;
        emit_classification_op(
            "cross_entropy_with_logits",
            self,
            &output,
            Some((config, active)),
        );
        Ok(output)
    }

    /// VJP of cross entropy with a seed matching the selected reduction's output shape.
    pub fn cross_entropy_with_logits_backward(
        &self,
        labels: &[i64],
        config: CrossEntropyConfig,
        seed: &Tensor,
    ) -> PureResult<Tensor> {
        let active = validate_labels(self, labels, config)?;
        let expected = config.output_shape(labels.len());
        if seed.shape() != expected {
            return Err(TensorError::ShapeMismatch {
                left: seed.shape(),
                right: expected,
            });
        }
        let input = finite_row_major(self, "cross_entropy_logits")?;
        let seed = finite_row_major(seed, "cross_entropy_seed")?;
        let (rows, classes) = input.shape();
        let mut values = Vec::with_capacity(input.len());
        let smoothing = config.label_smoothing;
        let uniform_mass = smoothing / classes as f64;
        let target_adjustment = smoothing * (1.0 - 1.0 / classes as f64);
        let normalizer = if config.reduction == LossReduction::Mean {
            active as f64
        } else {
            1.0
        };
        for (row_index, (row, &label)) in input.data().chunks_exact(classes).zip(labels).enumerate()
        {
            if label == config.ignore_index {
                values.resize(values.len() + classes, 0.0);
                continue;
            }
            let partition = RowPartition::new(row);
            let upstream = f64::from(
                seed.data()[if config.reduction == LossReduction::None {
                    row_index
                } else {
                    0
                }],
            );
            for (index, &value) in row.iter().enumerate() {
                let target = index == label as usize;
                let gradient = if index == partition.maximum_index {
                    let residual = if target {
                        target_adjustment
                    } else {
                        1.0 - uniform_mass
                    };
                    residual - partition.tail_probability()
                } else if target {
                    (partition.probability(value) - 1.0) + target_adjustment
                } else {
                    partition.probability(value) - uniform_mass
                };
                values.push(checked_f32(
                    gradient * upstream / normalizer,
                    "cross_entropy_gradient",
                )?);
            }
        }
        let output = Tensor::from_vec(rows, classes, values)?;
        emit_classification_op(
            "cross_entropy_with_logits_backward",
            self,
            &output,
            Some((config, active)),
        );
        Ok(output)
    }
}
