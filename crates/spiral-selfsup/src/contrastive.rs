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

fn l2_norm(vec: &[f32]) -> f32 {
    vec.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt() as f32
}

/// Compute the InfoNCE loss across a batch of anchor and positive representations.
pub fn info_nce_loss(
    anchors: &[Vec<f32>],
    positives: &[Vec<f32>],
    temperature: f32,
    normalize: bool,
) -> Result<InfoNCEResult> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(ObjectiveError::InvalidArgument(format!(
            "temperature must be > 0, got {temperature}"
        )));
    }

    let (batch, feature_dim) = validate_batches(anchors, positives)?;
    let mut anchor_norms = vec![1.0f32; batch];
    let mut positive_norms = vec![1.0f32; batch];

    if normalize {
        for (idx, vec) in anchors.iter().enumerate() {
            let norm = l2_norm(vec).max(std::f32::EPSILON);
            anchor_norms[idx] = norm;
        }
        for (idx, vec) in positives.iter().enumerate() {
            let norm = l2_norm(vec).max(std::f32::EPSILON);
            positive_norms[idx] = norm;
        }
    }

    let anchors_flat = flatten_row_major(anchors, feature_dim);
    let positives_t = transpose_to_row_major(positives, batch, feature_dim);

    let mut logits = compute_logits(&anchors_flat, &positives_t, batch, feature_dim)?;

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
        let positive_logit = row[i];
        let log_prob = positive_logit - max_logit - exp_sum.ln();
        loss += -log_prob;
    }
    loss /= batch as f32;

    Ok(InfoNCEResult {
        loss,
        logits,
        labels: (0..batch).collect(),
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
    .map_err(|message| ObjectiveError::Shape(message))?;
    Ok(logits)
}

/// Compute the InfoNCE loss using [`Tensor`] operands.
pub fn info_nce_loss_tensor(
    anchors: &Tensor,
    positives: &Tensor,
    temperature: f32,
    normalize: bool,
) -> Result<InfoNCEResult> {
    let (anchor_rows, anchor_cols) = anchors.shape();
    let (positive_rows, positive_cols) = positives.shape();
    if anchor_rows != positive_rows || anchor_cols != positive_cols {
        return Err(ObjectiveError::Shape(format!(
            "tensor batch mismatch (anchors={}x{}, positives={}x{})",
            anchor_rows, anchor_cols, positive_rows, positive_cols
        )));
    }

    let anchors_rm = anchors
        .to_layout(Layout::RowMajor)
        .map_err(|err| ObjectiveError::InvalidArgument(err.to_string()))?;
    let positives_rm = positives
        .to_layout(Layout::RowMajor)
        .map_err(|err| ObjectiveError::InvalidArgument(err.to_string()))?;

    let anchors_vec = anchors_rm
        .data()
        .chunks(anchor_cols)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();
    let positives_vec = positives_rm
        .data()
        .chunks(positive_cols)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();

    info_nce_loss(&anchors_vec, &positives_vec, temperature, normalize)
}
