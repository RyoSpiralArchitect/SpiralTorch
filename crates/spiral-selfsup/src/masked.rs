use crate::{ObjectiveError, Result};

/// Result of the masked modeling loss computation.
#[derive(Debug, Clone, PartialEq)]
pub struct MaskedResult {
    pub loss: f32,
    pub total_masked: usize,
    pub per_example: Vec<f32>,
}

fn validate_inputs(
    predictions: &[Vec<f32>],
    targets: &[Vec<f32>],
    mask_indices: &[Vec<usize>],
) -> Result<(usize, usize)> {
    if predictions.is_empty() {
        return Err(ObjectiveError::InvalidArgument(
            "predictions must be non-empty".to_string(),
        ));
    }
    if predictions.len() != targets.len() || predictions.len() != mask_indices.len() {
        return Err(ObjectiveError::Shape(format!(
            "batch mismatch (pred={}, target={}, mask={})",
            predictions.len(),
            targets.len(),
            mask_indices.len()
        )));
    }
    let feature_dim = predictions[0].len();
    for (idx, (pred_row, tgt_row)) in predictions.iter().zip(targets.iter()).enumerate() {
        if pred_row.len() != feature_dim {
            return Err(ObjectiveError::Shape(format!(
                "prediction row {idx} has dim {} (expected {feature_dim})",
                pred_row.len()
            )));
        }
        if tgt_row.len() != feature_dim {
            return Err(ObjectiveError::Shape(format!(
                "target row {idx} has dim {} (expected {feature_dim})",
                tgt_row.len()
            )));
        }
    }
    Ok((predictions.len(), feature_dim))
}

/// Compute a masked mean-squared-error objective.
pub fn masked_mse_loss(
    predictions: &[Vec<f32>],
    targets: &[Vec<f32>],
    mask_indices: &[Vec<usize>],
) -> Result<MaskedResult> {
    let (batch, _feature_dim) = validate_inputs(predictions, targets, mask_indices)?;
    let mut total_loss = 0.0f32;
    let mut total_count = 0usize;
    let mut per_example = vec![0.0f32; batch];

    for i in 0..batch {
        let mut sample_loss = 0.0f32;
        let mut sample_count = 0usize;
        for &idx in &mask_indices[i] {
            if let (Some(&pred), Some(&tgt)) = (predictions[i].get(idx), targets[i].get(idx)) {
                let diff = pred - tgt;
                sample_loss += diff * diff;
                sample_count += 1;
            } else {
                return Err(ObjectiveError::Shape(format!(
                    "mask index {idx} out of bounds for sample {i}"
                )));
            }
        }
        if sample_count > 0 {
            sample_loss /= sample_count as f32;
        }
        per_example[i] = sample_loss;
        total_loss += sample_loss;
        total_count += sample_count;
    }

    if total_count == 0 {
        return Err(ObjectiveError::InvalidArgument(
            "mask selects zero elements".to_string(),
        ));
    }

    Ok(MaskedResult {
        loss: total_loss / batch as f32,
        total_masked: total_count,
        per_example,
    })
}
