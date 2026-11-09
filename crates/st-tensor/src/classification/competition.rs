// SPDX-License-Identifier: AGPL-3.0-or-later
//! A convex blend of full-vocabulary CE and CE on the target plus its strongest incorrect classes.
use super::*;

pub const MAX_RIVAL_CLASSES: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RivalCrossEntropy {
    pub rivals: usize,
    pub mix: f64,
}

impl RivalCrossEntropy {
    pub fn validate(self, classes: usize) -> PureResult<()> {
        if self.rivals == 0 || self.rivals > MAX_RIVAL_CLASSES || self.rivals >= classes {
            return Err(TensorError::InvalidValue {
                label: "rivals must be in 1..=32 and smaller than vocabulary",
            });
        }
        if !(0.0..=1.0).contains(&self.mix) {
            return Err(TensorError::InvalidValue {
                label: "competition mix must be in [0,1]",
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct RivalCrossEntropyReport {
    pub rivals: usize,
    pub mix: f64,
    pub target_tokens: usize,
    pub full_ce: f32,
    pub rival_ce: f64,
    pub loss: f32,
    pub top1_correct: usize,
    pub mean_target_rank: f64,
    pub mean_target_minus_best_rival: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coverage_guard: Option<RivalCoverageGuardReport>,
}

pub(super) fn strongest_rivals(
    row: &[f32],
    gold: usize,
    count: usize,
    rivals: &mut Vec<usize>,
) -> usize {
    rivals.clear();
    super::target::rank_visiting_rivals(row, gold, |i, z| {
        if rivals.len() == count {
            if z <= row[*rivals.last().unwrap()] {
                return;
            }
            rivals.pop();
        }
        let pos = rivals.partition_point(|&j| row[j] >= z);
        rivals.insert(pos, i);
    })
}

pub(super) struct RestrictedRow {
    pub loss: f64,
    pub margin: f64,
    peak: f64,
    wrong_mass: f64,
    partition: f64,
}

impl RestrictedRow {
    pub fn new(row: &[f32], gold: usize, rivals: &[usize], best: f64) -> Self {
        let target = f64::from(row[gold]);
        let peak = target.max(best);
        let wrong_mass: f64 = rivals
            .iter()
            .map(|&j| (f64::from(row[j]) - peak).exp())
            .sum();
        let partition = (target - peak).exp() + wrong_mass;
        let loss = if target >= best {
            wrong_mass.ln_1p()
        } else {
            (peak - target) + partition.ln()
        };
        Self {
            loss,
            margin: target - best,
            peak,
            wrong_mass,
            partition,
        }
    }

    pub fn blend(
        &self,
        row: &[f32],
        gold: usize,
        sorted: &[usize],
        grad: &mut [f32],
        n: usize,
        mix: f64,
    ) -> PureResult<()> {
        let mut cursor = 0;
        for (i, v) in grad.iter_mut().enumerate() {
            let restricted = if i == gold {
                -self.wrong_mass / self.partition
            } else if sorted.get(cursor) == Some(&i) {
                cursor += 1;
                (f64::from(row[i]) - self.peak).exp() / self.partition
            } else {
                0.0
            };
            *v = checked_f32(
                (1.0 - mix) * f64::from(*v) + mix * restricted / n as f64,
                "rival_ce_gradient",
            )?;
        }
        Ok(())
    }
}

impl Tensor {
    /// Unsmoothed mean CE and its unit-seed logit VJP; -100 labels are ignored.
    /// The rival set is selected from the unfettered row, excluding the target.
    /// At ties the lowest class ID wins; the VJP follows that selected branch.
    pub fn rival_cross_entropy_mean_gradient(
        &self,
        labels: &[i64],
        recipe: RivalCrossEntropy,
    ) -> PureResult<(RivalCrossEntropyReport, Tensor)> {
        let (_, classes) = self.shape();
        recipe.validate(classes)?;
        let ce = CrossEntropyConfig::default();
        let n = validate_labels(self, labels, ce)?;
        let input = finite_row_major(self, "rival_ce_logits")?;
        let full_ce = input.cross_entropy_with_logits(labels, ce)?.data()[0];
        let seed = Tensor::from_vec(1, 1, vec![1.0])?;
        let mut grad = input.cross_entropy_with_logits_backward(labels, ce, &seed)?;
        let mut rivals = Vec::<usize>::with_capacity(recipe.rivals);
        let (mut sum_loss, mut sum_rank, mut sum_margin, mut correct) = (0.0, 0.0, 0.0, 0);
        let change = recipe.mix > 0.0 && recipe.rivals != classes - 1;
        for ((row, grad), &label) in input
            .data()
            .chunks_exact(classes)
            .zip(grad.data_mut().chunks_exact_mut(classes))
            .zip(labels)
        {
            if label == ce.ignore_index {
                continue;
            }
            let gold = label as usize;
            let rank = strongest_rivals(row, gold, recipe.rivals, &mut rivals);
            let restricted = RestrictedRow::new(row, gold, &rivals, f64::from(row[rivals[0]]));
            sum_loss += restricted.loss;
            sum_rank += rank as f64;
            sum_margin += restricted.margin;
            correct += usize::from(rank == 1);
            if change {
                // Walk class IDs once; the rival scratch is bounded independently of vocabulary size.
                rivals.sort_unstable();
                restricted.blend(row, gold, &rivals, grad, n, recipe.mix)?;
            }
        }
        let rival_ce = sum_loss / n as f64;
        let loss = if change {
            checked_f32(
                (1.0 - recipe.mix) * f64::from(full_ce) + recipe.mix * rival_ce,
                "rival_ce_loss",
            )?
        } else {
            full_ce
        };
        let report = RivalCrossEntropyReport {
            rivals: recipe.rivals,
            mix: recipe.mix,
            target_tokens: n,
            full_ce,
            rival_ce,
            loss,
            top1_correct: correct,
            mean_target_rank: sum_rank / n as f64,
            mean_target_minus_best_rival: sum_margin / n as f64,
            coverage_guard: None,
        };
        Ok((report, grad))
    }
}

#[cfg(test)]
mod tests;
