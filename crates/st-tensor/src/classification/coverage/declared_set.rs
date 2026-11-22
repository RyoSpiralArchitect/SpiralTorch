//! Factor unsmoothed target CE into -log P(target | S) and -log P(S).
use super::*;
use crate::classification::competition::RestrictedRow;
#[cfg(test)]
mod tests;

/// Both means use all non-ignored targets. For undeclared active rows, S is the vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct DeclaredSetNll {
    pub target_tokens: usize,
    pub contrasted_tokens: usize,
    pub within_nll: f64,
    pub coverage_nll: f64,
    /// Sum of the two components, not a subtraction of rounded scalar losses.
    pub full_nll: f64,
}

struct Batch<'a> {
    input: Tensor,
    labels: &'a [i64],
    plan: &'a [DeclaredRivalRow],
    n: usize,
}

fn members(r: &DeclaredRivalRow, scratch: &mut Vec<usize>) {
    let at = r.rivals.partition_point(|&id| id < r.target);
    scratch.clear();
    scratch.extend_from_slice(&r.rivals[..at]);
    scratch.push(r.target);
    scratch.extend_from_slice(&r.rivals[at..]);
}

impl<'a> Batch<'a> {
    fn new(input: &Tensor, labels: &'a [i64], plan: &'a [DeclaredRivalRow]) -> PureResult<Self> {
        let (rows, classes) = input.shape();
        DeclaredRivalRow::validate_plan(plan, rows, classes, 1.)?;
        let n = validate_labels(input, labels, CrossEntropyConfig::default())?;
        if plan
            .iter()
            .any(|r| usize::try_from(labels[r.position]) != Ok(r.target))
        {
            return Err(TensorError::InvalidValue {
                label: "declared set differs from its active label",
            });
        }
        Ok(Self {
            input: finite_row_major(input, "declared_set_logits")?,
            labels,
            plan,
            n,
        })
    }

    fn components(
        &self,
        mut within_seed: Option<&mut [f32]>,
        mut keep: impl FnMut(&'a DeclaredRivalRow, Split),
    ) -> PureResult<DeclaredSetNll> {
        let classes = self.input.shape().1;
        let (mut within, mut coverage) = (0., 0.);
        let mut plan = self.plan.iter().peekable();
        let mut inside = Vec::new();
        for (i, (row, &label)) in self
            .input
            .data()
            .chunks_exact(classes)
            .zip(self.labels)
            .enumerate()
        {
            if label == -100 {
                continue;
            }
            let gold = label as usize;
            if let Some(r) = plan
                .next_if(|r| r.position == i)
                .filter(|r| r.rivals.len() < classes - 1)
            {
                members(r, &mut inside);
                let split = Split::of(row, &inside).expect("proper subset, validated class IDs");
                let best = r
                    .rivals
                    .iter()
                    .map(|&id| f64::from(row[id]))
                    .fold(f64::NEG_INFINITY, f64::max);
                let part = RestrictedRow::new(row, gold, &r.rivals, best);
                within += part.loss;
                coverage += split.cost;
                if let Some(grad) = within_seed.as_deref_mut() {
                    part.blend(
                        row,
                        gold,
                        &r.rivals,
                        &mut grad[i * classes..(i + 1) * classes],
                        self.n,
                        1.,
                    )?;
                }
                keep(r, split);
            } else {
                let p = RowPartition::new(row);
                within += (p.maximum - f64::from(row[gold])) + p.log_partition();
            }
        }
        let within_nll = within / self.n as f64;
        let coverage_nll = coverage / self.n as f64;
        Ok(DeclaredSetNll {
            target_tokens: self.n,
            contrasted_tokens: self.plan.len(),
            within_nll,
            coverage_nll,
            full_nll: within_nll + coverage_nll,
        })
    }
}

impl Tensor {
    /// Measure the two components without constructing a logit gradient.
    pub fn declared_set_nll(
        &self,
        labels: &[i64],
        plan: &[DeclaredRivalRow],
    ) -> PureResult<DeclaredSetNll> {
        Batch::new(self, labels, plan)?.components(None, |_, _| {})
    }

    /// Project the within-set VJP, then reuse its buffer for the directly computed
    /// coverage VJP. In particular, this never subtracts two nearly identical f32 seeds.
    pub fn declared_set_nll_capture_within<T, E: From<TensorError>>(
        &self,
        labels: &[i64],
        plan: &[DeclaredRivalRow],
        project_within: impl FnOnce(&Tensor) -> Result<T, E>,
    ) -> Result<(DeclaredSetNll, Tensor, T), E> {
        let b = Batch::new(self, labels, plan)?;
        let (rows, classes) = b.input.shape();
        let mut seed = if plan.len() == b.n && plan.iter().all(|r| r.rivals.len() < classes - 1) {
            Tensor::zeros(rows, classes)?
        } else {
            b.input.cross_entropy_with_logits_backward(
                labels,
                CrossEntropyConfig::default(),
                &Tensor::from_vec(1, 1, vec![1.])?,
            )?
        };
        let mut splits = Vec::with_capacity(plan.len());
        let report = b.components(Some(seed.data_mut()), |r, split| splits.push((r, split)))?;
        let within = project_within(&seed)?;
        seed.data_mut().fill(0.);
        let mut inside = Vec::new();
        for (r, split) in splits {
            members(r, &mut inside);
            let range = r.position * classes..(r.position + 1) * classes;
            split.add_gradient(
                &b.input.data()[range.clone()],
                &inside,
                &mut seed.data_mut()[range],
                1. / b.n as f64,
            )?;
        }
        Ok((report, seed, within))
    }
}
