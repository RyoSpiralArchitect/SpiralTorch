use crate::{ObjectiveError, Result};
use st_tensor::{backend::cpu_dense, Layout, Tensor};

#[cfg(feature = "wgpu")]
use st_tensor::backend::wgpu_dense;

/// Result container for the InfoNCE contrastive objective.
#[derive(Debug, Clone, PartialEq)]
pub struct InfoNCEResult {
    pub loss: f32,
    pub logits: Vec<f32>,
    pub labels: Vec<usize>,
    pub batch: usize,
}

/// Result container for the InfoNCE objective operating on [`Tensor`] inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfoNCEResult {
    pub loss: f32,
    pub logits: Tensor,
    pub labels: Tensor,
    pub batch: usize,
}

struct PreparedInfoNCE {
    loss: f32,
    logits: Vec<f32>,
    batch: usize,
}

impl PreparedInfoNCE {
    fn into_result(self) -> InfoNCEResult {
        InfoNCEResult {
            loss: self.loss,
            logits: self.logits,
            labels: (0..self.batch).collect(),
            batch: self.batch,
        }
    }
}

fn validate_batches(a: &[Vec<f32>], b: &[Vec<f32>]) -> Result<(usize, usize)> {
    if a.is_empty() || b.is_empty() {
        return Err(ObjectiveError::InvalidArgument(
            "contrastive batches must be non-empty".to_string(),
        ));
    }
    if a.len() != b.len() {
        return Err(ObjectiveError::Shape(format!(
            "batch mismatch (anchors={}, positives={})",
            a.len(),
            b.len()
        )));
    }
    let feature_dim = a[0].len();
    if feature_dim == 0 {
        return Err(ObjectiveError::InvalidArgument(
            "feature dimension must be > 0".to_string(),
        ));
    }
    for (idx, row) in a.iter().enumerate() {
        if row.len() != feature_dim {
            return Err(ObjectiveError::Shape(format!(
                "anchor row {idx} has dim {} (expected {feature_dim})",
                row.len()
            )));
        }
    }
    for (idx, row) in b.iter().enumerate() {
        if row.len() != feature_dim {
            return Err(ObjectiveError::Shape(format!(
                "positive row {idx} has dim {} (expected {feature_dim})",
                row.len()
            )));
        }
    }
    Ok((a.len(), feature_dim))
}

fn validate_tensor_batches(anchors: &Tensor, positives: &Tensor) -> Result<(usize, usize)> {
    let (anchor_batch, feature_dim) = anchors.shape();
    let (positive_batch, positive_dim) = positives.shape();
    if anchor_batch == 0 || feature_dim == 0 {
        return Err(ObjectiveError::InvalidArgument(
            "contrastive batches must be non-empty".to_string(),
        ));
    }
    if positive_batch == 0 || positive_dim == 0 {
        return Err(ObjectiveError::InvalidArgument(
            "contrastive batches must be non-empty".to_string(),
        ));
    }
    if anchor_batch != positive_batch {
        return Err(ObjectiveError::Shape(format!(
            "batch mismatch (anchors={}, positives={})",
            anchor_batch, positive_batch
        )));
    }
    if feature_dim != positive_dim {
        return Err(ObjectiveError::Shape(format!(
            "feature mismatch (anchors={}, positives={})",
            feature_dim, positive_dim
        )));
    }
    Ok((anchor_batch, feature_dim))
}

fn l2_norm(values: impl Iterator<Item = f32>) -> f32 {
    values.map(|v| f64::from(v).powi(2)).sum::<f64>().sqrt() as f32
}

/// Compute the InfoNCE loss across a batch of anchor and positive representations.
pub fn info_nce_loss(
    anchors: &[Vec<f32>],
    positives: &[Vec<f32>],
    temperature: f32,
    normalize: bool,
) -> Result<InfoNCEResult> {
    validate_temperature(temperature)?;
    let (batch, feature_dim) = validate_batches(anchors, positives)?;
    let anchors_flat = flatten_row_major(anchors, feature_dim);
    let positives_t = transpose_to_row_major(positives, batch, feature_dim);
    info_nce_prepared(
        &anchors_flat,
        &positives_t,
        batch,
        feature_dim,
        temperature,
        normalize,
    )
    .map(PreparedInfoNCE::into_result)
}

/// Compute the InfoNCE loss for batches expressed as [`Tensor`]s.
pub fn info_nce_loss_tensor(
    anchors: &Tensor,
    positives: &Tensor,
    temperature: f32,
    normalize: bool,
) -> Result<TensorInfoNCEResult> {
    let result = prepare_tensor_inputs(anchors, positives, temperature, normalize)?;
    let batch = result.batch;
    let logits = Tensor::from_vec(batch, batch, result.logits)?;
    let labels = (0..batch).map(|value| value as f32).collect();
    let labels = Tensor::from_vec(batch, 1, labels)?;
    Ok(TensorInfoNCEResult {
        loss: result.loss,
        logits,
        labels,
        batch,
    })
}

fn validate_temperature(temperature: f32) -> Result<()> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(ObjectiveError::InvalidArgument(format!(
            "temperature must be > 0, got {temperature}"
        )));
    }
    Ok(())
}

fn info_nce_prepared(
    anchors_flat: &[f32],
    positives_t: &[f32],
    batch: usize,
    feature_dim: usize,
    temperature: f32,
    normalize: bool,
) -> Result<PreparedInfoNCE> {
    let mut logits = compute_logits(anchors_flat, positives_t, batch, feature_dim)?;
    if normalize {
        let anchor_norms: Vec<_> = anchors_flat
            .chunks_exact(feature_dim)
            .map(|row| l2_norm(row.iter().copied()).max(f32::EPSILON))
            .collect();
        let positive_norms: Vec<_> = (0..batch)
            .map(|column| {
                l2_norm((0..feature_dim).map(|row| positives_t[row * batch + column]))
                    .max(f32::EPSILON)
            })
            .collect();
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
        let positive_logit = row[i];
        let log_prob = positive_logit - max_logit - exp_sum.ln();
        loss += -log_prob;
    }
    loss /= batch as f32;

    Ok(PreparedInfoNCE {
        loss,
        logits,
        batch,
    })
}

fn flatten_row_major(rows: &[Vec<f32>], cols: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(rows.len() * cols);
    for row in rows {
        debug_assert_eq!(row.len(), cols);
        data.extend_from_slice(row);
    }
    data
}

fn transpose_to_row_major(rows: &[Vec<f32>], batch: usize, feature_dim: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; feature_dim * batch];
    for (row_idx, row) in rows.iter().enumerate() {
        for (col_idx, &value) in row.iter().enumerate() {
            transposed[col_idx * batch + row_idx] = value;
        }
    }
    transposed
}

fn apply_normalization(
    logits: &mut [f32],
    anchor_norms: &[f32],
    positive_norms: &[f32],
    batch: usize,
) {
    for i in 0..batch {
        for j in 0..batch {
            let denom = anchor_norms[i] * positive_norms[j];
            if denom > 0.0 {
                logits[i * batch + j] /= denom;
            }
        }
    }
}

fn compute_logits(
    anchors_flat: &[f32],
    positives_t: &[f32],
    batch: usize,
    feature_dim: usize,
) -> Result<Vec<f32>> {
    #[cfg(feature = "wgpu")]
    {
        if let Ok(buffer) = wgpu_dense::matmul(anchors_flat, positives_t, batch, feature_dim, batch)
        {
            return Ok(buffer);
        }
    }

    let mut logits = vec![0.0f32; batch * batch];
    cpu_dense::matmul_into(
        &mut logits,
        anchors_flat,
        positives_t,
        batch,
        feature_dim,
        batch,
    )
    .map_err(ObjectiveError::Shape)?;
    Ok(logits)
}

/// Compute the InfoNCE loss using [`Tensor`] operands and return the vector-form result.
pub fn info_nce_loss_tensor_as_result(
    anchors: &Tensor,
    positives: &Tensor,
    temperature: f32,
    normalize: bool,
) -> Result<InfoNCEResult> {
    prepare_tensor_inputs(anchors, positives, temperature, normalize)
        .map(PreparedInfoNCE::into_result)
}

fn prepare_tensor_inputs(
    anchors: &Tensor,
    positives: &Tensor,
    temperature: f32,
    normalize: bool,
) -> Result<PreparedInfoNCE> {
    validate_temperature(temperature)?;
    let (batch, feature_dim) = validate_tensor_batches(anchors, positives)?;
    let anchors_rm = anchors.to_layout(Layout::RowMajor)?;
    // Column-major positive storage is the row-major RHS transpose. Already
    // compatible Tensor buffers are shared, without allocating per-example Vecs.
    let positives_t = positives.to_layout(Layout::ColMajor)?;
    info_nce_prepared(
        anchors_rm.data(),
        positives_t.data(),
        batch,
        feature_dim,
        temperature,
        normalize,
    )
}
