//! Supervised class contrasts. Membership is explicit, not selected by the current logits.
use super::competition::RestrictedRow;
use super::*;

pub const MAX_DECLARED_RIVAL_IDS: usize = 65_536;

#[derive(Clone, Debug)]
pub struct DeclaredRivalRow {
    pub position: usize,
    pub target: usize,
    pub rivals: Vec<usize>,
}

impl DeclaredRivalRow {
    pub fn validate_plan(plan: &[Self], rows: usize, classes: usize, mix: f64) -> PureResult<()> {
        if !(0.0..=1.0).contains(&mix) || plan.len() > rows.min(MAX_DECLARED_RIVAL_IDS) {
            return Err(TensorError::InvalidValue {
                label: "invalid declared rival mix or plan size",
            });
        }
        let mut previous = None;
        let mut total = 0;
        for r in plan {
            if r.position >= rows
                || previous.is_some_and(|i| i >= r.position)
                || r.target >= classes
                || r.rivals.is_empty()
                || r.rivals.len() > MAX_RIVAL_CLASSES
                || r.rivals.windows(2).any(|p| p[0] >= p[1])
                || r.rivals.iter().any(|&j| j >= classes || j == r.target)
            {
                return Err(TensorError::InvalidValue {
                    label: "invalid declared rival position, target or sorted rivals",
                });
            }
            previous = Some(r.position);
            total += r.rivals.len();
            if total > MAX_DECLARED_RIVAL_IDS {
                return Err(TensorError::InvalidValue {
                    label: "declared rival ID budget exceeded",
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct DeclaredRivalReport {
    pub mix: f64,
    pub target_tokens: usize,
    pub contrasted_tokens: usize,
    pub rival_ids: usize,
    pub full_ce: f32,
    /// Restricted mean over declared targets only; None if the plan is empty.
    pub declared_ce: Option<f64>,
    pub mean_target_minus_declared_rival: Option<f64>,
    /// Mean over all non-ignored targets, retaining ordinary CE on unpaired rows.
    pub loss: f32,
}

impl Tensor {
    pub fn declared_rival_cross_entropy_mean_gradient(
        &self,
        labels: &[i64],
        plan: &[DeclaredRivalRow],
        mix: f64,
    ) -> PureResult<(DeclaredRivalReport, Tensor)> {
        let (rows, classes) = self.shape();
        DeclaredRivalRow::validate_plan(plan, rows, classes, mix)?;
        let ce = CrossEntropyConfig::default();
        let n = validate_labels(self, labels, ce)?;
        if plan
            .iter()
            .any(|r| usize::try_from(labels[r.position]) != Ok(r.target))
        {
            return Err(TensorError::InvalidValue {
                label: "declared rival row does not match its active label",
            });
        }
        let input = finite_row_major(self, "declared_rival_logits")?;
        let seed = Tensor::from_vec(1, 1, vec![1.])?;
        let mut grad = input.cross_entropy_with_logits_backward(labels, ce, &seed)?;
        let mut total = 0.;
        let mut restricted_sum = 0.;
        let mut margin_sum = 0.;
        // Keep the full row losses in f64 until reduced. Subtracting a rounded full
        // mean from itself at mix=1 can erase a very small restricted loss.
        let mut restricted = Vec::with_capacity(plan.len());
        for r in plan {
            let start = r.position * classes;
            let row = &input.data()[start..start + classes];
            let best = r
                .rivals
                .iter()
                .map(|&j| f64::from(row[j]))
                .fold(f64::NEG_INFINITY, f64::max);
            let part = RestrictedRow::new(row, r.target, &r.rivals, best);
            restricted_sum += part.loss;
            margin_sum += part.margin;
            if mix > 0. && r.rivals.len() != classes - 1 {
                part.blend(
                    row,
                    r.target,
                    &r.rivals,
                    &mut grad.data_mut()[start..start + classes],
                    n,
                    mix,
                )?;
            }
            restricted.push(part.loss);
        }
        let mut costs = plan.iter().zip(restricted).peekable();
        let full_ce = input
            .cross_entropy_visit(labels, ce, |i, full, _| {
                let row_loss = if let Some((r, part)) = costs.next_if(|(r, _)| r.position == i) {
                    if mix == 0. || r.rivals.len() == classes - 1 {
                        full
                    } else {
                        (1. - mix) * full + mix * part
                    }
                } else {
                    full
                };
                total += row_loss;
            })?
            .data()[0];
        let count = plan.len();
        Ok((
            DeclaredRivalReport {
                mix,
                target_tokens: n,
                contrasted_tokens: count,
                rival_ids: plan.iter().map(|r| r.rivals.len()).sum(),
                full_ce,
                declared_ce: (count != 0).then(|| restricted_sum / count as f64),
                mean_target_minus_declared_rival: (count != 0).then(|| margin_sum / count as f64),
                loss: checked_f32(total / n as f64, "declared_rival_loss")?,
            },
            grad,
        ))
    }
}

#[cfg(test)]
mod tests;
