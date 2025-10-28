use crate::lightning::LightningConfig;
use crate::loss::Loss;
use crate::trainer::{EpochStats, IntoBatch};
use crate::{PureResult, Tensor};
use self::contrastive::InfoNCEResult;
use st_core::telemetry::atlas::AtlasFragment;
use st_core::telemetry::hub;
use std::fmt;
use std::mem;

mod contrastive {
    use std::f32::EPSILON;

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

        for value in &mut logits {
            *value /= temperature;
        }

        let mut loss = 0.0f32;
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
            loss += -(positive_logit - max_logit - exp_sum.ln());
        }
        loss /= batch as f32;

        Ok(InfoNCEResult {
            loss,
            logits,
            labels: (0..batch).collect(),
            batch,
        })
    }

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

    fn flatten_row_major(rows: &[Vec<f32>], feature_dim: usize) -> Vec<f32> {
        let mut flat = Vec::with_capacity(rows.len() * feature_dim);
        for row in rows {
            flat.extend_from_slice(row);
        }
        flat
    }

    fn transpose_to_row_major(rows: &[Vec<f32>], batch: usize, feature_dim: usize) -> Vec<f32> {
        let mut transposed = vec![0.0f32; batch * feature_dim];
        for (row_idx, row) in rows.iter().enumerate() {
            for (col_idx, &value) in row.iter().enumerate() {
                transposed[col_idx * batch + row_idx] = value;
            }
        }
        transposed
    }

    fn normalise_rows(data: &mut [f32], feature_dim: usize, norms: &mut [f32]) {
        for (row_idx, chunk) in data.chunks_exact_mut(feature_dim).enumerate() {
            let norm = chunk
                .iter()
                .map(|&v| (v as f64).powi(2))
                .sum::<f64>()
                .sqrt() as f32;
            let norm = norm.max(EPSILON);
            norms[row_idx] = norm;
            for value in chunk.iter_mut() {
                *value /= norm;
            }
        }
    }

    fn normalise_cols(data: &mut [f32], batch: usize, feature_dim: usize, norms: &mut [f32]) {
        for col_idx in 0..batch {
            let mut norm_sq = 0.0f64;
            for row_idx in 0..feature_dim {
                let value = data[row_idx * batch + col_idx];
                norm_sq += (value as f64).powi(2);
            }
            let norm = norm_sq.sqrt() as f32;
            let norm = norm.max(EPSILON);
            norms[col_idx] = norm;
            for row_idx in 0..feature_dim {
                let idx = row_idx * batch + col_idx;
                data[idx] /= norm;
            }
        }
    }

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

    fn apply_normalization(
        logits: &mut [f32],
        anchor_norms: &[f32],
        positive_norms: &[f32],
        batch: usize,
    ) {
        for i in 0..batch {
            let a = anchor_norms[i].max(EPSILON);
            for j in 0..batch {
                let idx = i * batch + j;
                logits[idx] /= a * positive_norms[j].max(EPSILON);
            }
        }
    }
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
        let mut data = Vec::with_capacity(rows * cols * 2);
        data.extend_from_slice(anchors.data());
        data.extend_from_slice(positives.data());
        let combined = Tensor::from_vec(rows * 2, cols, data)?;
        Ok(Self { combined })
    }

    pub fn from_combined(combined: Tensor) -> PureResult<Self> {
        let (rows, _) = combined.shape();
        if rows % 2 != 0 {
            return Err(combined.shape_mismatch_error((rows + 1, 0)));
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

impl SelfSupObjective {
}

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
    pub(crate) fn from_batches(results: &[InfoNCEResult]) -> Option<Self> {
        if results.is_empty() {
            return None;
        }
        let sum = results.iter().map(|res| res.loss).sum::<f32>();
        let mean_loss = sum / results.len() as f32;
        Some(Self {
            mean_loss,
            batches: results.len(),
        })
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

    fn split_predictions(&self, prediction: &Tensor) -> PureResult<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
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
                squared.sqrt().max(std::f32::EPSILON)
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
}

impl Loss for InfoNCELoss {
    fn forward(&mut self, prediction: &Tensor, _target: &Tensor) -> PureResult<Tensor> {
        let (anchors, positives) = self.split_predictions(prediction)?;
        let (anchor_hat, anchor_norms) = self.normalise_batch(&anchors);
        let (positive_hat, positive_norms) = self.normalise_batch(&positives);
        let result = contrastive::info_nce_loss(&anchor_hat, &positive_hat, self.temperature, false)
            .map_err(|_| crate::TensorError::InvalidValue { label: "selfsup.info_nce" })?;
        self.cache = Some(InfoNCECache {
            anchor_hat,
            positive_hat,
            anchor_norms,
            positive_norms,
            logits: result.logits.clone(),
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
        let batch = rows / 2;
        let mut anchor_grads = vec![vec![0.0f32; cols]; batch];
        let mut positive_grads = vec![vec![0.0f32; cols]; batch];
        let scale = 1.0 / (self.temperature * batch as f32);
        for i in 0..batch {
            let row = &cache.logits[i * batch..(i + 1) * batch];
            let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut exp = vec![0.0f32; batch];
            let mut sum = 0.0f32;
            for (j, value) in row.iter().enumerate() {
                let val = ((value - max_logit) as f64).exp() as f32;
                exp[j] = val;
                sum += val;
            }
            for j in 0..batch {
                let weight = if sum > 0.0 { exp[j] / sum } else { 0.0 };
                for k in 0..cols {
                    anchor_grads[i][k] += weight * cache.positive_hat[j][k] * scale;
                    positive_grads[j][k] += weight * cache.anchor_hat[i][k] * scale;
                }
            }
            for k in 0..cols {
                anchor_grads[i][k] -= cache.positive_hat[i][k] * scale;
                positive_grads[i][k] -= cache.anchor_hat[i][k] * scale;
            }
        }

        if self.normalize {
            for i in 0..batch {
                let norm = cache.anchor_norms[i].max(std::f32::EPSILON);
                let hat = &cache.anchor_hat[i];
                let dot = anchor_grads[i]
                    .iter()
                    .zip(hat.iter())
                    .map(|(g, h)| g * h)
                    .sum::<f32>();
                for k in 0..cols {
                    anchor_grads[i][k] = (anchor_grads[i][k] - hat[k] * dot) / norm;
                }
            }
            for i in 0..batch {
                let norm = cache.positive_norms[i].max(std::f32::EPSILON);
                let hat = &cache.positive_hat[i];
                let dot = positive_grads[i]
                    .iter()
                    .zip(hat.iter())
                    .map(|(g, h)| g * h)
                    .sum::<f32>();
                for k in 0..cols {
                    positive_grads[i][k] = (positive_grads[i][k] - hat[k] * dot) / norm;
                }
            }
        }

        let mut data = Vec::with_capacity(rows * cols);
        for grad in anchor_grads.iter().chain(positive_grads.iter()) {
            data.extend_from_slice(grad);
        }
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
