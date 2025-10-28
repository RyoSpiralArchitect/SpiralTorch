use crate::{ObjectiveError, Result};
use st_tensor::Tensor;

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

    let mut logits = vec![0.0f32; batch * batch];
    for i in 0..batch {
        for j in 0..batch {
            let mut dot = 0.0f32;
            for k in 0..feature_dim {
                dot += anchors[i][k] * positives[j][k];
            }
            if normalize {
                dot /= anchor_norms[i] * positive_norms[j];
            }
            logits[i * batch + j] = dot / temperature;
        }
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

/// Compute the InfoNCE loss for batches expressed as [`Tensor`]s.
pub fn info_nce_loss_tensor(
    anchors: &Tensor,
    positives: &Tensor,
    temperature: f32,
    normalize: bool,
) -> Result<TensorInfoNCEResult> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(ObjectiveError::InvalidArgument(format!(
            "temperature must be > 0, got {temperature}"
        )));
    }

    let (batch, feature_dim) = validate_tensor_batches(anchors, positives)?;
    let anchor_data = anchors.data();
    let positive_data = positives.data();

    let mut anchor_norms = vec![1.0f32; batch];
    let mut positive_norms = vec![1.0f32; batch];

    if normalize {
        for (idx, chunk) in anchor_data.chunks_exact(feature_dim).enumerate() {
            let norm = l2_norm(chunk).max(std::f32::EPSILON);
            anchor_norms[idx] = norm;
        }
        for (idx, chunk) in positive_data.chunks_exact(feature_dim).enumerate() {
            let norm = l2_norm(chunk).max(std::f32::EPSILON);
            positive_norms[idx] = norm;
        }
    }

    let mut logits = vec![0.0f32; batch * batch];
    for i in 0..batch {
        let anchor_row = &anchor_data[i * feature_dim..(i + 1) * feature_dim];
        for j in 0..batch {
            let positive_row = &positive_data[j * feature_dim..(j + 1) * feature_dim];
            let mut dot = 0.0f32;
            for k in 0..feature_dim {
                dot += anchor_row[k] * positive_row[k];
            }
            if normalize {
                dot /= anchor_norms[i] * positive_norms[j];
            }
            logits[i * batch + j] = dot / temperature;
        }
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

    let logits_tensor = Tensor::from_vec(batch, batch, logits).map_err(ObjectiveError::from)?;
    let labels = (0..batch).map(|value| value as f32).collect();
    let labels_tensor = Tensor::from_vec(batch, 1, labels).map_err(ObjectiveError::from)?;

    Ok(TensorInfoNCEResult {
        loss,
        logits: logits_tensor,
        labels: labels_tensor,
        batch,
    })
}
