use self::contrastive::InfoNCEResult;
use crate::execution::{
    current_matmul_backend, current_softmax_backend, current_tensor_util_backend_for_values,
};
use crate::lightning::LightningConfig;
use crate::loss::Loss;
use crate::trainer::{EpochStats, IntoBatch};
use crate::{PureResult, Tensor};
use st_core::telemetry::atlas::AtlasFragment;
use st_core::telemetry::hub;
use std::fmt;
use std::mem;

mod contrastive {
    #[derive(Debug, Clone)]
    pub struct InfoNCEResult {
        pub loss: f32,
        pub logits: Vec<f32>,
        pub labels: Vec<usize>,
        pub batch: usize,
    }

    #[derive(Debug)]
    pub(super) enum InfoNCEError {
        Shape,
        InvalidArgument,
    }

    #[cfg(test)]
    pub fn info_nce_loss(
        anchors: &[Vec<f32>],
        positives: &[Vec<f32>],
        temperature: f32,
        normalize: bool,
    ) -> Result<InfoNCEResult, InfoNCEError> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(InfoNCEError::InvalidArgument);
        }

        let (batch, feature_dim) = validate_batches(anchors, positives)?;
        let mut anchor_norms = vec![1.0f32; batch];
        let mut positive_norms = vec![1.0f32; batch];

        let mut anchors_flat = flatten_row_major(anchors, feature_dim);
        let mut positives_t = transpose_to_row_major(positives, batch, feature_dim);

        if normalize {
            normalise_rows(&mut anchors_flat, feature_dim, &mut anchor_norms);
            normalise_cols(&mut positives_t, batch, feature_dim, &mut positive_norms);
        }

        let mut logits = compute_logits(&anchors_flat, &positives_t, batch, feature_dim);

        if normalize {
            apply_normalization(&mut logits, &anchor_norms, &positive_norms, batch);
        }

        finish_info_nce_loss(logits, batch, temperature)
    }

    pub(super) fn finish_info_nce_loss(
        mut logits: Vec<f32>,
        batch: usize,
        temperature: f32,
    ) -> Result<InfoNCEResult, InfoNCEError> {
        if !temperature.is_finite() || temperature <= 0.0 || batch == 0 {
            return Err(InfoNCEError::InvalidArgument);
        }
        if logits.len() != batch * batch {
            return Err(InfoNCEError::Shape);
        }

        for value in &mut logits {
            if !value.is_finite() {
                return Err(InfoNCEError::InvalidArgument);
            }
            *value /= temperature;
            if !value.is_finite() {
                return Err(InfoNCEError::InvalidArgument);
            }
        }

        let mut loss = 0.0f64;
        for i in 0..batch {
            let row = &logits[i * batch..(i + 1) * batch];
            let max_logit = row.iter().fold(f32::NEG_INFINITY, |acc, &v| acc.max(v));
            let exp_sum: f32 = row
                .iter()
                .map(|&v| ((v - max_logit) as f64).exp() as f32)
                .sum();
            if exp_sum <= 0.0 || !exp_sum.is_finite() {
                return Err(InfoNCEError::InvalidArgument);
            }
            let positive_logit = row[i];
            let term = -(positive_logit - max_logit - exp_sum.ln());
            if !term.is_finite() {
                return Err(InfoNCEError::InvalidArgument);
            }
            loss += f64::from(term);
        }
        loss /= batch as f64;
        if !loss.is_finite() || loss.abs() > f64::from(f32::MAX) {
            return Err(InfoNCEError::InvalidArgument);
        }
        let loss = loss as f32;
        Ok(InfoNCEResult {
            loss,
            logits,
            labels: (0..batch).collect(),
            batch,
        })
    }

    #[cfg(test)]
    fn validate_batches(
        anchors: &[Vec<f32>],
        positives: &[Vec<f32>],
    ) -> Result<(usize, usize), InfoNCEError> {
        if anchors.is_empty() || positives.is_empty() {
            return Err(InfoNCEError::InvalidArgument);
        }
        if anchors.len() != positives.len() {
            return Err(InfoNCEError::Shape);
        }
        let feature_dim = anchors[0].len();
        if feature_dim == 0 {
            return Err(InfoNCEError::InvalidArgument);
        }
        for row in anchors.iter() {
            if row.len() != feature_dim {
                return Err(InfoNCEError::Shape);
            }
        }
        for row in positives.iter() {
            if row.len() != feature_dim {
                return Err(InfoNCEError::Shape);
            }
        }
        Ok((anchors.len(), feature_dim))
    }

    #[cfg(test)]
    fn flatten_row_major(rows: &[Vec<f32>], feature_dim: usize) -> Vec<f32> {
        let mut flat = Vec::with_capacity(rows.len() * feature_dim);
        for row in rows {
            flat.extend_from_slice(row);
        }
        flat
    }

    #[cfg(test)]
    fn transpose_to_row_major(rows: &[Vec<f32>], batch: usize, feature_dim: usize) -> Vec<f32> {
        let mut transposed = vec![0.0f32; batch * feature_dim];
        for (row_idx, row) in rows.iter().enumerate() {
            for (col_idx, &value) in row.iter().enumerate() {
                transposed[col_idx * batch + row_idx] = value;
            }
        }
        transposed
    }

    #[cfg(test)]
    fn normalise_rows(data: &mut [f32], feature_dim: usize, norms: &mut [f32]) {
        for (row_idx, chunk) in data.chunks_exact_mut(feature_dim).enumerate() {
            let norm = chunk
                .iter()
                .map(|&v| (v as f64).powi(2))
                .sum::<f64>()
                .sqrt() as f32;
            let norm = norm.max(f32::EPSILON);
            norms[row_idx] = norm;
            for value in chunk.iter_mut() {
                *value /= norm;
            }
        }
    }

    #[cfg(test)]
    fn normalise_cols(data: &mut [f32], batch: usize, feature_dim: usize, norms: &mut [f32]) {
        for col_idx in 0..batch {
            let mut norm_sq = 0.0f64;
            for row_idx in 0..feature_dim {
                let value = data[row_idx * batch + col_idx];
                norm_sq += (value as f64).powi(2);
            }
            let norm = norm_sq.sqrt() as f32;
            let norm = norm.max(f32::EPSILON);
            norms[col_idx] = norm;
            for row_idx in 0..feature_dim {
                let idx = row_idx * batch + col_idx;
                data[idx] /= norm;
            }
        }
    }

    #[cfg(test)]
    fn compute_logits(
        anchors_flat: &[f32],
        positives_t: &[f32],
        batch: usize,
        feature_dim: usize,
    ) -> Vec<f32> {
        let mut logits = vec![0.0f32; batch * batch];
        for i in 0..batch {
            let anchor = &anchors_flat[i * feature_dim..(i + 1) * feature_dim];
            for j in 0..batch {
                let mut dot = 0.0f32;
                for k in 0..feature_dim {
                    dot += anchor[k] * positives_t[k * batch + j];
                }
                logits[i * batch + j] = dot;
            }
        }
        logits
    }

    #[cfg(test)]
    fn apply_normalization(
        logits: &mut [f32],
        anchor_norms: &[f32],
        positive_norms: &[f32],
        batch: usize,
    ) {
        for (row, &anchor_norm) in logits
            .chunks_exact_mut(batch)
            .zip(anchor_norms.iter().take(batch))
        {
            let a = anchor_norm.max(f32::EPSILON);
            for (logit, &positive_norm) in row.iter_mut().zip(positive_norms.iter().take(batch)) {
                *logit /= a * positive_norm.max(f32::EPSILON);
            }
        }
    }
}

type SplitPredictionBatch = (Vec<Vec<f32>>, Vec<Vec<f32>>);

fn validate_selfsup_scalar(label: &'static str, value: f32) -> PureResult<f32> {
    if !value.is_finite() {
        return Err(crate::TensorError::NonFiniteValue { label, value });
    }
    Ok(value)
}

fn checked_selfsup_sum(
    label: &'static str,
    values: impl IntoIterator<Item = f32>,
) -> PureResult<f32> {
    let mut sum = 0.0f64;
    for value in values {
        validate_selfsup_scalar(label, value)?;
        sum += f64::from(value);
    }
    if !sum.is_finite() || sum.abs() > f64::from(f32::MAX) {
        let value = if sum.is_sign_negative() {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
        return Err(crate::TensorError::NonFiniteValue { label, value });
    }
    Ok(sum as f32)
}

fn project_normalized_gradient_row(
    grad_row: &mut [f32],
    hat_row: &[f32],
    norm: f32,
    label: &'static str,
) -> PureResult<()> {
    if grad_row.len() != hat_row.len() {
        return Err(crate::TensorError::ShapeMismatch {
            left: (1, grad_row.len()),
            right: (1, hat_row.len()),
        });
    }
    let norm = validate_selfsup_scalar(label, norm)?.max(f32::EPSILON);
    let dot = checked_selfsup_sum(
        label,
        grad_row
            .iter()
            .zip(hat_row.iter())
            .map(|(grad, hat)| *grad * *hat),
    )?;
    let mut projected = Vec::with_capacity(grad_row.len());
    for (&grad, &hat) in grad_row.iter().zip(hat_row.iter()) {
        validate_selfsup_scalar(label, grad)?;
        validate_selfsup_scalar(label, hat)?;
        let correction = hat * dot;
        validate_selfsup_scalar(label, correction)?;
        let value = (grad - correction) / norm;
        projected.push(validate_selfsup_scalar(label, value)?);
    }
    grad_row.copy_from_slice(&projected);
    Ok(())
}

#[derive(Debug, Clone)]
pub struct SelfSupBatch {
    combined: Tensor,
}

impl SelfSupBatch {
    pub fn from_pairs(anchors: Tensor, positives: Tensor) -> PureResult<Self> {
        if anchors.shape() != positives.shape() {
            return Err(anchors.shape_mismatch_error(positives.shape()));
        }
        let (rows, cols) = anchors.shape();
        if rows == 0 {
            return Err(anchors.empty_tensor_error("SelfSupBatch::from_pairs"));
        }
        if cols == 0 {
            return Err(crate::TensorError::InvalidDimensions { rows, cols });
        }
        let mut data = Vec::with_capacity(rows * cols * 2);
        data.extend_from_slice(anchors.data());
        data.extend_from_slice(positives.data());
        let combined = Tensor::from_vec(rows * 2, cols, data)?;
        Ok(Self { combined })
    }

    pub fn from_combined(combined: Tensor) -> PureResult<Self> {
        let (rows, cols) = combined.shape();
        if rows % 2 != 0 {
            return Err(combined.shape_mismatch_error((rows + 1, 0)));
        }
        if rows == 0 {
            return Err(combined.empty_tensor_error("SelfSupBatch::from_combined"));
        }
        if cols == 0 {
            return Err(crate::TensorError::InvalidDimensions { rows, cols });
        }
        Ok(Self { combined })
    }

    pub fn combined(&self) -> &Tensor {
        &self.combined
    }

    pub(crate) fn into_supervised(self) -> PureResult<(Tensor, Tensor)> {
        let target = Tensor::zeros(1, 1)?;
        Ok((self.combined, target))
    }
}

impl IntoBatch for SelfSupBatch {
    fn into_batch(self) -> PureResult<(Tensor, Tensor)> {
        self.into_supervised()
    }
}

#[derive(Debug, Clone)]
pub struct SelfSupEpoch {
    batches: Vec<SelfSupBatch>,
}

impl SelfSupEpoch {
    pub fn new(batches: Vec<SelfSupBatch>) -> Self {
        Self { batches }
    }

    pub fn batches(&self) -> &[SelfSupBatch] {
        &self.batches
    }

    pub(crate) fn into_supervised(self) -> PureResult<Vec<(Tensor, Tensor)>> {
        self.batches
            .into_iter()
            .map(SelfSupBatch::into_supervised)
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InfoNCEConfig {
    pub temperature: f32,
    pub normalize: bool,
}

impl InfoNCEConfig {
    pub fn new(temperature: f32, normalize: bool) -> Self {
        Self {
            temperature,
            normalize,
        }
    }
}

#[derive(Debug, Clone)]
pub enum SelfSupObjective {
    InfoNCE(InfoNCEConfig),
}

impl SelfSupObjective {}

#[derive(Debug, Clone)]
pub struct SelfSupStage {
    config: LightningConfig,
    label: Option<String>,
    objective: SelfSupObjective,
    epochs: Vec<SelfSupEpoch>,
}

impl SelfSupStage {
    pub fn new(config: LightningConfig, objective: SelfSupObjective) -> Self {
        Self {
            config,
            label: None,
            objective,
            epochs: Vec::new(),
        }
    }

    pub fn with_epochs(
        config: LightningConfig,
        objective: SelfSupObjective,
        epochs: Vec<SelfSupEpoch>,
    ) -> Self {
        Self {
            config,
            label: None,
            objective,
            epochs,
        }
    }

    pub fn add_epoch(mut self, epoch: SelfSupEpoch) -> Self {
        self.epochs.push(epoch);
        self
    }

    pub fn push_epoch(&mut self, epoch: SelfSupEpoch) {
        self.epochs.push(epoch);
    }

    pub fn set_label(&mut self, label: impl Into<String>) {
        self.label = Some(label.into());
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.set_label(label);
        self
    }

    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    pub fn config(&self) -> &LightningConfig {
        &self.config
    }

    pub fn objective(&self) -> &SelfSupObjective {
        &self.objective
    }

    pub fn epochs(&self) -> &[SelfSupEpoch] {
        &self.epochs
    }

    pub fn into_parts(
        self,
    ) -> (
        LightningConfig,
        SelfSupObjective,
        Vec<SelfSupEpoch>,
        Option<String>,
    ) {
        (self.config, self.objective, self.epochs, self.label)
    }
}

#[derive(Debug, Clone)]
pub struct InfoNCEEpochMetrics {
    pub mean_loss: f32,
    pub batches: usize,
}

impl InfoNCEEpochMetrics {
    pub(crate) fn from_batches(results: &[InfoNCEResult]) -> PureResult<Option<Self>> {
        if results.is_empty() {
            return Ok(None);
        }
        let sum = checked_selfsup_sum(
            "selfsup.info_nce.epoch_loss",
            results.iter().map(|res| res.loss),
        )?;
        let mean_loss = validate_selfsup_scalar(
            "selfsup.info_nce.epoch_mean_loss",
            sum / results.len() as f32,
        )?;
        Ok(Some(Self {
            mean_loss,
            batches: results.len(),
        }))
    }
}

#[derive(Debug, Clone)]
pub enum SelfSupEpochTelemetry {
    InfoNCE(InfoNCEEpochMetrics),
}

#[derive(Debug, Clone)]
pub struct SelfSupEpochReport {
    stats: EpochStats,
    telemetry: Option<SelfSupEpochTelemetry>,
}

impl SelfSupEpochReport {
    pub fn stats(&self) -> &EpochStats {
        &self.stats
    }

    pub fn telemetry(&self) -> Option<&SelfSupEpochTelemetry> {
        self.telemetry.as_ref()
    }

    pub(crate) fn new(stats: EpochStats, telemetry: Option<SelfSupEpochTelemetry>) -> Self {
        Self { stats, telemetry }
    }
}

#[derive(Debug, Clone)]
pub struct SelfSupStageReport {
    config: LightningConfig,
    label: Option<String>,
    epochs: Vec<SelfSupEpochReport>,
}

impl SelfSupStageReport {
    pub fn new(
        config: LightningConfig,
        label: Option<String>,
        epochs: Vec<SelfSupEpochReport>,
    ) -> Self {
        Self {
            config,
            label,
            epochs,
        }
    }

    pub fn config(&self) -> &LightningConfig {
        &self.config
    }

    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    pub fn epochs(&self) -> &[SelfSupEpochReport] {
        &self.epochs
    }
}

#[derive(Debug, Clone)]
pub struct SelfSupPlanReport {
    stages: Vec<SelfSupStageReport>,
}

impl SelfSupPlanReport {
    pub fn new(stages: Vec<SelfSupStageReport>) -> Self {
        Self { stages }
    }

    pub fn stages(&self) -> &[SelfSupStageReport] {
        &self.stages
    }
}

pub(crate) fn publish_selfsup_metrics(stage: Option<&str>, metrics: &InfoNCEEpochMetrics) {
    let mut fragment = AtlasFragment::new();
    if let Some(label) = stage {
        fragment.push_note(format!("selfsup.stage:{label}"));
    }
    fragment.push_metric("selfsup.info_nce.loss", metrics.mean_loss);
    fragment.push_metric("selfsup.info_nce.batches", metrics.batches as f32);
    hub::merge_atlas_fragment(fragment);
}

struct InfoNCECache {
    anchor_hat: Vec<Vec<f32>>,
    positive_hat: Vec<Vec<f32>>,
    anchor_norms: Vec<f32>,
    positive_norms: Vec<f32>,
    logits: Vec<f32>,
    prediction_shape: (usize, usize),
}

pub struct InfoNCELoss {
    temperature: f32,
    normalize: bool,
    cache: Option<InfoNCECache>,
    epoch_metrics: Vec<InfoNCEResult>,
}

impl fmt::Debug for InfoNCELoss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InfoNCELoss")
            .field("temperature", &self.temperature)
            .field("normalize", &self.normalize)
            .finish()
    }
}

impl InfoNCELoss {
    pub fn new(temperature: f32, normalize: bool) -> Self {
        let temperature = if temperature.is_finite() && temperature > 0.0 {
            temperature
        } else {
            0.1
        };
        Self {
            temperature,
            normalize,
            cache: None,
            epoch_metrics: Vec::new(),
        }
    }

    pub fn take_epoch_metrics(&mut self) -> Vec<InfoNCEResult> {
        mem::take(&mut self.epoch_metrics)
    }

    fn split_predictions(&self, prediction: &Tensor) -> PureResult<SplitPredictionBatch> {
        let (rows, cols) = prediction.shape();
        if rows % 2 != 0 {
            return Err(prediction.shape_mismatch_error((rows + 1, cols)));
        }
        let batch = rows / 2;
        let mut anchors = Vec::with_capacity(batch);
        let mut positives = Vec::with_capacity(batch);
        for i in 0..batch {
            let start = i * cols;
            let end = start + cols;
            anchors.push(prediction.data()[start..end].to_vec());
        }
        for i in 0..batch {
            let start = (batch + i) * cols;
            let end = start + cols;
            positives.push(prediction.data()[start..end].to_vec());
        }
        Ok((anchors, positives))
    }

    fn normalise_batch(&self, batch: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let mut normalised = Vec::with_capacity(batch.len());
        let mut norms = Vec::with_capacity(batch.len());
        for row in batch {
            let norm = if self.normalize {
                let squared = row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() as f32;
                squared.sqrt().max(f32::EPSILON)
            } else {
                1.0
            };
            if self.normalize {
                let scaled = row.iter().map(|v| *v / norm).collect::<Vec<f32>>();
                normalised.push(scaled);
            } else {
                normalised.push(row.clone());
            }
            norms.push(norm);
        }
        (normalised, norms)
    }

    fn batch_to_tensor(&self, batch: &[Vec<f32>], label: &'static str) -> PureResult<Tensor> {
        if batch.is_empty() {
            return Err(crate::TensorError::EmptyInput(label));
        }
        let cols = batch[0].len();
        if cols == 0 {
            return Err(crate::TensorError::InvalidDimensions {
                rows: batch.len(),
                cols,
            });
        }
        let mut data = Vec::with_capacity(batch.len() * cols);
        for row in batch {
            if row.len() != cols {
                return Err(crate::TensorError::ShapeMismatch {
                    left: (1, cols),
                    right: (1, row.len()),
                });
            }
            data.extend_from_slice(row);
        }
        Tensor::from_vec(batch.len(), cols, data)
    }

    fn logits_with_backend(
        &self,
        anchors: &[Vec<f32>],
        positives: &[Vec<f32>],
    ) -> PureResult<Vec<f32>> {
        let anchor_tensor = self.batch_to_tensor(anchors, "selfsup.info_nce.anchors")?;
        let positive_tensor = self.batch_to_tensor(positives, "selfsup.info_nce.positives")?;
        let util_backend = current_tensor_util_backend_for_values(positive_tensor.data().len());
        let positive_t = positive_tensor.transpose_with_backend(util_backend)?;
        let logits = anchor_tensor.matmul_with_backend(&positive_t, current_matmul_backend())?;
        Ok(logits.data().to_vec())
    }
}

impl Loss for InfoNCELoss {
    fn forward(&mut self, prediction: &Tensor, _target: &Tensor) -> PureResult<Tensor> {
        let (anchors, positives) = self.split_predictions(prediction)?;
        let (anchor_hat, anchor_norms) = self.normalise_batch(&anchors);
        let (positive_hat, positive_norms) = self.normalise_batch(&positives);
        let logits = self.logits_with_backend(&anchor_hat, &positive_hat)?;
        let result = contrastive::finish_info_nce_loss(logits, anchor_hat.len(), self.temperature)
            .map_err(|_| crate::TensorError::InvalidValue {
                label: "selfsup.info_nce",
            })?;
        self.cache = Some(InfoNCECache {
            anchor_hat,
            positive_hat,
            anchor_norms,
            positive_norms,
            logits: result.logits.clone(),
            prediction_shape: prediction.shape(),
        });
        self.epoch_metrics.push(result.clone());
        Tensor::from_vec(1, 1, vec![result.loss])
    }

    fn backward(&mut self, prediction: &Tensor, _target: &Tensor) -> PureResult<Tensor> {
        let cache = self
            .cache
            .take()
            .ok_or_else(|| prediction.shape_mismatch_error(prediction.shape()))?;
        let (rows, cols) = prediction.shape();
        if prediction.shape() != cache.prediction_shape {
            return Err(prediction.shape_mismatch_error(cache.prediction_shape));
        }
        let batch = rows / 2;
        let scale = 1.0 / (self.temperature * batch as f32);
        let probabilities = Tensor::from_vec(batch, batch, cache.logits.clone())?
            .row_softmax_with_backend(current_softmax_backend())?;
        let anchor_hat_tensor =
            self.batch_to_tensor(&cache.anchor_hat, "selfsup.info_nce.anchor_hat")?;
        let positive_hat_tensor =
            self.batch_to_tensor(&cache.positive_hat, "selfsup.info_nce.positive_hat")?;
        let matmul_backend = current_matmul_backend();
        let util_backend = current_tensor_util_backend_for_values(batch.saturating_mul(cols));

        let mut anchor_grad = probabilities.matmul_scaled_with_backend(
            &positive_hat_tensor,
            scale,
            matmul_backend,
        )?;
        anchor_grad.add_scaled_with_backend(&positive_hat_tensor, -scale, util_backend)?;

        let transpose_backend = current_tensor_util_backend_for_values(probabilities.data().len());
        let probability_t = probabilities.transpose_with_backend(transpose_backend)?;
        let mut positive_grad =
            probability_t.matmul_scaled_with_backend(&anchor_hat_tensor, scale, matmul_backend)?;
        positive_grad.add_scaled_with_backend(&anchor_hat_tensor, -scale, util_backend)?;

        if self.normalize {
            for (row_idx, (hat_row, &norm)) in cache
                .anchor_hat
                .iter()
                .zip(cache.anchor_norms.iter())
                .enumerate()
            {
                let start = row_idx * cols;
                let grad_row = &mut anchor_grad.data_mut()[start..start + cols];
                project_normalized_gradient_row(
                    grad_row,
                    hat_row,
                    norm,
                    "selfsup.info_nce.anchor_normalization_grad",
                )?;
            }
            for (row_idx, (hat_row, &norm)) in cache
                .positive_hat
                .iter()
                .zip(cache.positive_norms.iter())
                .enumerate()
            {
                let start = row_idx * cols;
                let grad_row = &mut positive_grad.data_mut()[start..start + cols];
                project_normalized_gradient_row(
                    grad_row,
                    hat_row,
                    norm,
                    "selfsup.info_nce.positive_normalization_grad",
                )?;
            }
        }

        let mut data = Vec::with_capacity(rows * cols);
        data.extend_from_slice(anchor_grad.data());
        data.extend_from_slice(positive_grad.data());
        Tensor::from_vec(rows, cols, data)
    }
}

trait TensorShapeExt {
    fn shape_mismatch_error(&self, other: (usize, usize)) -> crate::TensorError;
    fn empty_tensor_error(&self, context: &str) -> crate::TensorError;
}

impl TensorShapeExt for Tensor {
    fn shape_mismatch_error(&self, other: (usize, usize)) -> crate::TensorError {
        crate::TensorError::ShapeMismatch {
            left: self.shape(),
            right: other,
        }
    }

    fn empty_tensor_error(&self, _context: &str) -> crate::TensorError {
        crate::TensorError::EmptyInput("selfsup.batch")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "wgpu")]
    use crate::execution::{push_backend_policy, BackendPolicy};
    use crate::trainer::tensor_meta_observer_test_lock;
    #[cfg(feature = "wgpu")]
    use st_core::backend::device_caps::DeviceCaps;
    #[cfg(feature = "wgpu")]
    use st_tensor::wgpu_dense;
    use std::sync::{Arc, Mutex};

    fn sample_pairs() -> (Tensor, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let anchors = vec![vec![0.2, -0.4, 0.6], vec![0.1, 0.5, -0.3]];
        let positives = vec![vec![0.25, -0.35, 0.55], vec![-0.2, 0.45, 0.15]];
        let mut data = Vec::new();
        for row in anchors.iter().chain(positives.iter()) {
            data.extend_from_slice(row);
        }
        (Tensor::from_vec(4, 3, data).unwrap(), anchors, positives)
    }

    #[test]
    fn selfsup_batch_rejects_empty_or_zero_feature_batches() {
        let empty_anchors = Tensor::from_vec(0, 3, Vec::new()).unwrap();
        let empty_positives = Tensor::from_vec(0, 3, Vec::new()).unwrap();
        assert!(matches!(
            SelfSupBatch::from_pairs(empty_anchors, empty_positives),
            Err(crate::TensorError::EmptyInput(_))
        ));

        let zero_feature_anchors = Tensor::from_vec(2, 0, Vec::new()).unwrap();
        let zero_feature_positives = Tensor::from_vec(2, 0, Vec::new()).unwrap();
        assert!(matches!(
            SelfSupBatch::from_pairs(zero_feature_anchors, zero_feature_positives),
            Err(crate::TensorError::InvalidDimensions { rows: 2, cols: 0 })
        ));

        let empty_combined = Tensor::from_vec(0, 3, Vec::new()).unwrap();
        assert!(matches!(
            SelfSupBatch::from_combined(empty_combined),
            Err(crate::TensorError::EmptyInput(_))
        ));

        let zero_feature_combined = Tensor::from_vec(4, 0, Vec::new()).unwrap();
        assert!(matches!(
            SelfSupBatch::from_combined(zero_feature_combined),
            Err(crate::TensorError::InvalidDimensions { rows: 4, cols: 0 })
        ));
    }

    #[test]
    fn info_nce_forward_matches_reference_after_backend_matmul() {
        let (prediction, anchors, positives) = sample_pairs();
        let target = Tensor::zeros(1, 1).unwrap();
        let reference_loss = InfoNCELoss::new(0.7, true);
        let (anchor_hat, _) = reference_loss.normalise_batch(&anchors);
        let (positive_hat, _) = reference_loss.normalise_batch(&positives);
        let expected = contrastive::info_nce_loss(&anchor_hat, &positive_hat, 0.7, false).unwrap();

        let mut loss = InfoNCELoss::new(0.7, true);
        let output = loss.forward(&prediction, &target).unwrap();
        let observed = loss.take_epoch_metrics();

        assert!((output.data()[0] - expected.loss).abs() < 1e-6);
        assert_eq!(observed.len(), 1);
        assert_eq!(observed[0].batch, expected.batch);
        assert_eq!(observed[0].labels, expected.labels);
        for (actual, reference) in observed[0].logits.iter().zip(expected.logits.iter()) {
            assert!((actual - reference).abs() < 1e-6);
        }
    }

    fn info_nce_loss_value_at(
        prediction: &Tensor,
        target: &Tensor,
        temperature: f32,
        normalize: bool,
        index: usize,
        delta: f32,
    ) -> f32 {
        let mut perturbed = prediction.clone();
        perturbed.data_mut()[index] += delta;
        InfoNCELoss::new(temperature, normalize)
            .forward(&perturbed, target)
            .unwrap()
            .data()[0]
    }

    fn assert_info_nce_finite_difference(
        prediction: &Tensor,
        target: &Tensor,
        temperature: f32,
        normalize: bool,
        index: usize,
    ) {
        let mut loss = InfoNCELoss::new(temperature, normalize);
        let _ = loss.forward(prediction, target).unwrap();
        let analytic = loss.backward(prediction, target).unwrap().data()[index];
        let epsilon = 1.0e-3;
        let plus =
            info_nce_loss_value_at(prediction, target, temperature, normalize, index, epsilon);
        let minus =
            info_nce_loss_value_at(prediction, target, temperature, normalize, index, -epsilon);
        let numerical = (plus - minus) / (2.0 * epsilon);
        let tolerance = 1.0e-3_f32.max(numerical.abs() * 0.08);
        assert!(
            (analytic - numerical).abs() <= tolerance,
            "InfoNCE finite difference mismatch at {index} normalize={normalize}: analytic={analytic} numerical={numerical} tolerance={tolerance}"
        );
    }

    #[test]
    fn info_nce_backward_matches_forward_finite_difference() {
        let (prediction, _, _) = sample_pairs();
        let target = Tensor::zeros(1, 1).unwrap();
        for normalize in [false, true] {
            assert_info_nce_finite_difference(&prediction, &target, 0.7, normalize, 0);
            assert_info_nce_finite_difference(&prediction, &target, 0.7, normalize, 5);
            assert_info_nce_finite_difference(&prediction, &target, 0.7, normalize, 8);
        }
    }

    #[test]
    fn info_nce_loss_rejects_temperature_scaled_non_finite_logits() {
        let result = contrastive::finish_info_nce_loss(vec![f32::MAX], 1, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn info_nce_epoch_metrics_reject_non_finite_or_overflowed_loss() {
        let results = vec![
            InfoNCEResult {
                loss: f32::MAX,
                logits: Vec::new(),
                labels: Vec::new(),
                batch: 1,
            },
            InfoNCEResult {
                loss: f32::MAX,
                logits: Vec::new(),
                labels: Vec::new(),
                batch: 1,
            },
        ];
        let err = InfoNCEEpochMetrics::from_batches(&results)
            .expect_err("epoch metrics should reject overflowed loss sums");
        match err {
            crate::TensorError::NonFiniteValue { label, value } => {
                assert_eq!(label, "selfsup.info_nce.epoch_loss");
                assert!(value.is_infinite());
            }
            other => panic!("unexpected error: {other:?}"),
        }

        let results = vec![InfoNCEResult {
            loss: f32::NAN,
            logits: Vec::new(),
            labels: Vec::new(),
            batch: 1,
        }];
        let err = InfoNCEEpochMetrics::from_batches(&results)
            .expect_err("epoch metrics should reject non-finite losses");
        match err {
            crate::TensorError::NonFiniteValue { label, value } => {
                assert_eq!(label, "selfsup.info_nce.epoch_loss");
                assert!(value.is_nan());
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn info_nce_normalization_projection_is_commit_safe() {
        let mut grad = vec![1.0, f32::MAX];
        let original = grad.clone();
        let err = project_normalized_gradient_row(
            &mut grad,
            &[f32::MAX, 1.0],
            1.0,
            "selfsup.info_nce.anchor_normalization_grad",
        )
        .expect_err("projection should reject overflowed dot products");
        match err {
            crate::TensorError::NonFiniteValue { label, value } => {
                assert_eq!(label, "selfsup.info_nce.anchor_normalization_grad");
                assert!(value.is_infinite());
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert_eq!(grad, original);
    }

    #[test]
    fn info_nce_forward_backward_emit_tensor_backend_meta() {
        let (prediction, _, _) = sample_pairs();
        let target = Tensor::zeros(1, 1).unwrap();
        let mut last_events = Vec::new();
        for _ in 0..3 {
            let _lock = tensor_meta_observer_test_lock();
            let events = Arc::new(Mutex::new(Vec::new()));
            let captured = events.clone();
            let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
                captured
                    .lock()
                    .unwrap()
                    .push((event.op_name, event.data.clone()));
            })));

            let mut loss = InfoNCELoss::new(0.5, true);
            let _ = loss.forward(&prediction, &target).unwrap();
            let grad = loss.backward(&prediction, &target).unwrap();
            st_tensor::set_tensor_op_meta_observer(previous);

            assert_eq!(grad.shape(), prediction.shape());
            let events = events.lock().unwrap().clone();
            let matmul = events.iter().find(|(op_name, data)| {
                *op_name == "matmul" && data["rows"] == 2 && data["inner"] == 3 && data["cols"] == 2
            });
            let softmax = events.iter().find(|(op_name, data)| {
                *op_name == "row_softmax" && data["rows"] == 2 && data["cols"] == 2
            });
            if let (Some(matmul), Some(softmax)) = (matmul, softmax) {
                assert!(matmul.1["backend"].as_str().is_some());
                assert!(softmax.1["backend"].as_str().is_some());
                return;
            }
            last_events = events;
            std::thread::yield_now();
        }
        let observed = last_events
            .iter()
            .map(|(op_name, data)| {
                format!(
                    "{}:{}x{}",
                    op_name,
                    data.get("rows")
                        .and_then(|value| value.as_u64())
                        .unwrap_or(0),
                    data.get("cols")
                        .and_then(|value| value.as_u64())
                        .unwrap_or(0)
                )
            })
            .collect::<Vec<_>>();
        panic!("InfoNCE matmul/softmax metadata events missing; observed {observed:?}");
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn info_nce_forced_wgpu_routes_transpose_and_matches_cpu_reference() {
        if !wgpu_dense::is_available() {
            return;
        }
        let (prediction, _, _) = sample_pairs();
        let target = Tensor::zeros(1, 1).unwrap();
        let previous_threshold = std::env::var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES").ok();
        std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", "1");

        let cpu_policy = BackendPolicy::from_device_caps(DeviceCaps::cpu());
        let (cpu_output, cpu_grad) = {
            let _guard = push_backend_policy(cpu_policy);
            let mut loss = InfoNCELoss::new(0.5, true);
            let output = loss.forward(&prediction, &target).unwrap();
            let grad = loss.backward(&prediction, &target).unwrap();
            (output, grad)
        };

        let _lock = tensor_meta_observer_test_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let wgpu_policy = BackendPolicy::from_device_caps(DeviceCaps::wgpu(32, true, 256));
        let (wgpu_output, wgpu_grad) = {
            let _guard = push_backend_policy(wgpu_policy);
            let mut loss = InfoNCELoss::new(0.5, true);
            let output = loss.forward(&prediction, &target).unwrap();
            let grad = loss.backward(&prediction, &target).unwrap();
            (output, grad)
        };
        st_tensor::set_tensor_op_meta_observer(previous);
        match previous_threshold {
            Some(value) => std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", value),
            None => std::env::remove_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES"),
        }

        for (idx, (expected, actual)) in cpu_output
            .data()
            .iter()
            .zip(wgpu_output.data().iter())
            .enumerate()
        {
            assert!(
                (expected - actual).abs() <= 1.0e-5,
                "output mismatch at {idx}: expected={expected}, actual={actual}"
            );
        }
        for (idx, (expected, actual)) in cpu_grad
            .data()
            .iter()
            .zip(wgpu_grad.data().iter())
            .enumerate()
        {
            assert!(
                (expected - actual).abs() <= 1.0e-5,
                "gradient mismatch at {idx}: expected={expected}, actual={actual}"
            );
        }

        let events = events.lock().unwrap();
        let wgpu_transposes = events
            .iter()
            .filter(|(op_name, data)| {
                *op_name == "transpose"
                    && data["backend"] == "wgpu_dense"
                    && data["requested_backend"] == "wgpu"
                    && data["kernel"] == "tensor_util.transpose"
            })
            .count();
        assert!(
            wgpu_transposes >= 2,
            "expected InfoNCE forward/backward transposes on WGPU, observed {wgpu_transposes}"
        );
    }

    #[test]
    fn info_nce_backward_rejects_prediction_shape_mismatch() {
        let (prediction, _, _) = sample_pairs();
        let target = Tensor::zeros(1, 1).unwrap();
        let mut loss = InfoNCELoss::new(0.5, true);
        let _ = loss.forward(&prediction, &target).unwrap();

        let wrong_prediction = Tensor::from_vec(4, 4, vec![0.0; 16]).unwrap();
        let err = loss.backward(&wrong_prediction, &target).unwrap_err();

        assert!(matches!(
            err,
            crate::TensorError::ShapeMismatch {
                left: (4, 4),
                right: (4, 3)
            }
        ));
    }
}
