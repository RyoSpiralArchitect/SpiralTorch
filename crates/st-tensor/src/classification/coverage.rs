// SPDX-License-Identifier: AGPL-3.0-or-later
//! One-sided coverage loss on the reference-selected target plus strongest incorrect classes.
use super::*;

#[derive(Clone, Copy, Debug)]
pub struct RivalCoverageGuard(f64);

impl RivalCoverageGuard {
    pub fn new(strength: f64) -> PureResult<Self> {
        if !(0.0..=1.0).contains(&strength) {
            return Err(TensorError::InvalidValue {
                label: "coverage guard strength must be in [0,1]",
            });
        }
        Ok(Self(strength))
    }

    pub fn enabled(self) -> bool {
        self.0 > 0.
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct RivalCoverageGuardReport {
    pub strength: f64,
    pub active_targets: usize,
    pub unguarded_loss: f32,
    pub mean_reference_coverage_nll: f64,
    pub mean_candidate_coverage_nll: f64,
    /// Mean positive excess over all supervised positions, before multiplying by strength.
    pub mean_violation: f64,
}

/// Detached, batch-local reference. Retains k+1 class IDs and one log mass per active row,
/// never the full reference logits. Callers must align examples, not just shapes/labels.
#[derive(Debug)]
pub struct RivalCoverageReference {
    shape: (usize, usize),
    rivals: usize,
    rows: Vec<(usize, usize, f64)>,
    ids: Vec<usize>,
}

fn membership(count: usize, sorted: &[usize]) -> impl Iterator<Item = bool> + '_ {
    let mut j = 0;
    (0..count).map(move |i| {
        let inside = sorted.get(j) == Some(&i);
        j += usize::from(inside);
        inside
    })
}

struct Split {
    in_peak: f64,
    in_sum: f64,
    out_peak: f64,
    out_sum: f64,
    out_probability: f64,
    cost: f64,
}

impl Split {
    fn of(row: &[f32], inside: &[usize]) -> Option<Self> {
        if inside.len() == row.len() {
            return None;
        }
        let in_peak = inside
            .iter()
            .map(|&j| f64::from(row[j]))
            .fold(f64::NEG_INFINITY, f64::max);
        let out_peak = row
            .iter()
            .zip(membership(row.len(), inside))
            .filter(|(_, s)| !s)
            .map(|(&v, _)| f64::from(v))
            .fold(f64::NEG_INFINITY, f64::max);
        let in_sum = inside
            .iter()
            .map(|&j| (f64::from(row[j]) - in_peak).exp())
            .sum::<f64>();
        let out_sum = row
            .iter()
            .zip(membership(row.len(), inside))
            .filter(|(_, s)| !s)
            .map(|(&v, _)| (f64::from(v) - out_peak).exp())
            .sum::<f64>();
        let odds = (out_peak - in_peak) + (out_sum.ln() - in_sum.ln());
        let e = (-odds.abs()).exp();
        let cost = odds.max(0.) + e.ln_1p();
        let out_probability = if odds >= 0. {
            1. / (1. + e)
        } else {
            e / (1. + e)
        };
        Some(Self {
            in_peak,
            in_sum,
            out_peak,
            out_sum,
            out_probability,
            cost,
        })
    }

    fn add_gradient(
        &self,
        row: &[f32],
        inside: &[usize],
        grad: &mut [f32],
        weight: f64,
    ) -> PureResult<()> {
        // d[-log P(S)]: -P(out)*P(i|S) inside, +P(out)*P(i|out) outside.
        // Avoid subtracting near-equal probabilities when the target set is already very likely.
        for ((v, &z), member) in grad.iter_mut().zip(row).zip(membership(row.len(), inside)) {
            let conditional = if member {
                -(f64::from(z) - self.in_peak).exp() / self.in_sum
            } else {
                (f64::from(z) - self.out_peak).exp() / self.out_sum
            };
            *v = checked_f32(
                f64::from(*v) + weight * self.out_probability * conditional,
                "rival_coverage_gradient",
            )?;
        }
        Ok(())
    }
}

impl Tensor {
    pub fn rival_coverage_reference(
        &self,
        labels: &[i64],
        rivals: usize,
    ) -> PureResult<RivalCoverageReference> {
        let shape = self.shape();
        RivalCrossEntropy { rivals, mix: 0. }.validate(shape.1)?;
        let n = validate_labels(self, labels, CrossEntropyConfig::default())?;
        let input = finite_row_major(self, "coverage_reference_logits")?;
        let mut rows = Vec::with_capacity(n);
        let mut ids = Vec::with_capacity(n * (rivals + 1)); // bounded by the already admitted logit count
        let mut members = Vec::with_capacity(rivals + 1);
        for (i, (row, &label)) in input.data().chunks_exact(shape.1).zip(labels).enumerate() {
            if label == -100 {
                continue;
            }
            let gold = label as usize;
            competition::strongest_rivals(row, gold, rivals, &mut members);
            members.push(gold);
            members.sort_unstable();
            rows.push((i, gold, Split::of(row, &members).map_or(0., |s| s.cost)));
            ids.extend_from_slice(&members);
        }
        Ok(RivalCoverageReference {
            shape,
            rivals,
            rows,
            ids,
        })
    }

    /// Mixed CE plus strength * mean(max(0, -log P(S) + log P_ref(S))).
    /// S is fixed by the reference, not reselected after the candidate changes. A tie has zero subgradient.
    pub fn guarded_rival_cross_entropy_mean_gradient(
        &self,
        labels: &[i64],
        recipe: RivalCrossEntropy,
        reference: &RivalCoverageReference,
        guard: RivalCoverageGuard,
    ) -> PureResult<(RivalCrossEntropyReport, Tensor)> {
        let shape = self.shape();
        let n = validate_labels(self, labels, CrossEntropyConfig::default())?;
        if shape != reference.shape
            || recipe.rivals != reference.rivals
            || n != reference.rows.len()
            || reference
                .rows
                .iter()
                .any(|&(i, gold, _)| labels[i] != gold as i64)
        {
            return Err(TensorError::InvalidValue {
                label: "coverage reference must match rows, labels and rival count",
            });
        }
        let input = finite_row_major(self, "rival_coverage_logits")?;
        let (mut report, mut gradient) = input.rival_cross_entropy_mean_gradient(labels, recipe)?;
        if !guard.enabled() {
            return Ok((report, gradient));
        }
        let mut summary = RivalCoverageGuardReport {
            strength: guard.0,
            unguarded_loss: report.loss,
            active_targets: 0,
            mean_reference_coverage_nll: 0.,
            mean_candidate_coverage_nll: 0.,
            mean_violation: 0.,
        };
        let weight = guard.0 / n as f64;
        for (&(i, _, baseline), members) in reference
            .rows
            .iter()
            .zip(reference.ids.chunks_exact(recipe.rivals + 1))
        {
            let start = i * shape.1;
            let row = &input.data()[start..start + shape.1];
            summary.mean_reference_coverage_nll += baseline;
            if let Some(split) = Split::of(row, members) {
                summary.mean_candidate_coverage_nll += split.cost;
                if split.cost > baseline {
                    summary.active_targets += 1;
                    summary.mean_violation += split.cost - baseline;
                    split.add_gradient(
                        row,
                        members,
                        &mut gradient.data_mut()[start..start + shape.1],
                        weight,
                    )?;
                }
            }
        }
        summary.mean_reference_coverage_nll /= n as f64;
        summary.mean_candidate_coverage_nll /= n as f64;
        summary.mean_violation /= n as f64;
        if summary.active_targets != 0 {
            report.loss = checked_f32(
                f64::from(report.loss) + guard.0 * summary.mean_violation,
                "rival_coverage_loss",
            )?;
        }
        report.coverage_guard = Some(summary);
        Ok((report, gradient))
    }
}

#[cfg(test)]
mod tests;
