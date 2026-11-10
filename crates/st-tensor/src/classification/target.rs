// SPDX-License-Identifier: AGPL-3.0-or-later
use super::{PureResult, RowPartition, TensorError};

/// One target against every other class in an unmodified f32 logit row.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct TargetClassScore {
    pub vocabulary: usize,
    pub target_id: usize,
    /// One-based; among equal logits, smaller IDs come first.
    pub target_rank: usize,
    pub greedy_id: usize,
    pub strongest_rival_id: usize,
    pub target_logit: f64,
    pub strongest_rival_logit: f64,
    pub target_minus_best_rival: f64,
    /// Unsmoothed, full-vocabulary negative log likelihood, not restricted to two classes.
    pub nll: f64,
}

pub(super) fn rank_visiting_rivals(
    row: &[f32],
    target: usize,
    mut visit: impl FnMut(usize, f32),
) -> usize {
    let gold = row[target];
    let mut rank = 1;
    for (i, &v) in row.iter().enumerate() {
        if i != target {
            rank += usize::from(v > gold || (v == gold && i < target));
            visit(i, v);
        }
    }
    rank
}

impl TargetClassScore {
    /// Allows clients to reject a bad target/shape before copying or computing logits.
    pub fn validate_domain(vocabulary: usize, target_id: usize) -> PureResult<()> {
        if vocabulary < 2 || target_id >= vocabulary {
            return Err(TensorError::InvalidValue {
                label: "target scoring needs at least two classes and an in-vocabulary target",
            });
        }
        Ok(())
    }

    pub fn from_logits(row: &[f32], target_id: usize) -> PureResult<Self> {
        Self::validate_domain(row.len(), target_id)?;
        if let Some(&value) = row.iter().find(|v| !v.is_finite()) {
            return Err(TensorError::NonFiniteValue {
                label: "target score logits",
                value,
            });
        }
        let mut rival = usize::from(target_id == 0);
        let target_rank = rank_visiting_rivals(row, target_id, |i, v| {
            if v > row[rival] {
                rival = i;
            }
        });
        let greedy_id = if target_rank == 1 { target_id } else { rival };
        let partition = RowPartition::with_maximum(row, greedy_id);
        let target_logit = f64::from(row[target_id]);
        let strongest_rival_logit = f64::from(row[rival]);
        Ok(Self {
            vocabulary: row.len(),
            target_id,
            target_rank,
            greedy_id,
            strongest_rival_id: rival,
            target_logit,
            strongest_rival_logit,
            target_minus_best_rival: target_logit - strongest_rival_logit,
            nll: (partition.maximum - target_logit) + partition.log_partition(),
        })
    }
}
