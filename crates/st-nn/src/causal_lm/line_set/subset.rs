//! Conditional marginal likelihood of a nonempty proper subset of a terminated menu.
use super::ranking::{costs, paths, winner_index, Path};
use super::readout::PrefixPass;
use super::*;

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmLineSubsetReport {
    pub accepted_indices: Vec<usize>,
    pub stop_id: usize,
    pub target_counts: Vec<usize>,
    pub decoder_lanes: usize,
    pub prefix_nodes: usize,
    pub winner_index: usize,
    pub line_nlls: Vec<f64>,
    pub menu_probabilities: Vec<f64>,
    pub menu_set_nll: f64,
    pub accepted_set_nll: f64,
    pub rejected_probability_given_menu: f64,
    pub conditional_nll: f64,
    pub loss: f32,
}

pub(in crate::causal_lm) struct LineSubset {
    support: LineSet,
    paths: Vec<Path>,
    shared: usize,
    accepted: Vec<usize>,
}

#[derive(Clone, Copy)]
struct Coeff {
    ce: Term,
    fork: Term,
}
impl Default for Coeff {
    fn default() -> Self {
        Self {
            ce: Term::ZERO,
            fork: Term::ZERO,
        }
    }
}

impl LineSubset {
    fn new(support: LineSet, lines: &[Vec<usize>], accepted: &[usize]) -> Self {
        let (shared, paths) = paths(&support, lines);
        Self {
            support,
            paths,
            shared,
            accepted: accepted.to_vec(),
        }
    }

    fn summarize(
        &self,
        pass: &PrefixPass<'_>,
    ) -> CausalLmResult<(CausalLmLineSubsetReport, Vec<Term>)> {
        let costs = costs(&self.support, &self.paths, self.shared, pass);
        let (mut yes, mut no) = (Group::default(), Group::default());
        let winner = winner_index(&costs.suffix);
        for (i, &cost) in costs.conditional.iter().enumerate() {
            if self.accepted.binary_search(&i).is_ok() {
                yes.add(cost.neg());
            } else {
                no.add(cost.neg());
            }
        }
        let event = Outcome::new(yes, no);
        let conditional_nll = -event.yes.rounded();
        let loss = conditional_nll as f32;
        if !loss.is_finite() {
            return Err("complete-line subset objective exceeds finite f32 loss".into());
        }
        Ok((
            CausalLmLineSubsetReport {
                accepted_indices: self.accepted.clone(),
                stop_id: self.support.stop,
                target_counts: self.support.counts.clone(),
                decoder_lanes: self.support.leaves.len(),
                prefix_nodes: self.support.nodes.len(),
                winner_index: winner,
                line_nlls: costs.line,
                menu_probabilities: costs
                    .conditional
                    .iter()
                    .map(|t| (-t.rounded()).exp())
                    .collect(),
                menu_set_nll: -pass.outcomes[0].yes.rounded(),
                accepted_set_nll: -pass.outcomes[0].yes.plus(event.yes).rounded(),
                rejected_probability_given_menu: event.no.expect("proper subset").rounded().exp(),
                conditional_nll,
                loss,
            },
            costs.conditional,
        ))
    }

    pub(in crate::causal_lm) fn score(
        &self,
        logits: &Tensor,
    ) -> CausalLmResult<CausalLmLineSubsetReport> {
        Ok(self
            .summarize(&PrefixPass::new(&self.support, logits, true)?)?
            .0)
    }

    pub(in crate::causal_lm) fn loss_gradient(
        &self,
        logits: &Tensor,
    ) -> CausalLmResult<(CausalLmLineSubsetReport, Tensor)> {
        let pass = PrefixPass::new(&self.support, logits, true)?;
        let (report, scores) = self.summarize(&pass)?;
        let negatives: Vec<_> = (0..scores.len())
            .filter(|i| self.accepted.binary_search(i).is_err())
            .collect();
        let best = |ids: &[usize]| {
            ids.iter()
                .map(|&i| scores[i])
                .reduce(|a, b| if a.gap(b) <= 0. { a } else { b })
                .unwrap()
        };
        let (best_a, best_n) = (best(&self.accepted), best(&negatives));
        let mut signed = vec![BTreeMap::<usize, Coeff>::new(); self.support.nodes.len()];
        let mut den = Term::ZERO;
        // dL = P(N|M) E_{a|A,n|N}[d NLL(a) - d NLL(n)]. Factor out the
        // common normalizer. No two accepted aliases are contrasted with one another.
        for &a in &self.accepted {
            for &n in &negatives {
                let w = best_a
                    .plus(scores[a].neg())
                    .plus(best_n.plus(scores[n].neg()))
                    .exp();
                den = den.plus(w);
                let (left, right) = (&self.paths[a], &self.paths[n]);
                let fork = left
                    .iter()
                    .zip(right)
                    .position(|(p, q)| p != q)
                    .expect("distinct terminated lines");
                for (path, weight) in [(left, w), (right, w.neg())] {
                    let (i, k) = path[fork];
                    let entry = signed[i].entry(k).or_default();
                    entry.fork = entry.fork.plus(weight.neg());
                    for &(i, k) in &path[fork + 1..] {
                        let entry = signed[i].entry(k).or_default();
                        entry.ce = entry.ce.plus(weight);
                    }
                }
            }
        }
        let scale = report.rejected_probability_given_menu / den.rounded();
        let (rows, cols) = logits.shape();
        let mut seed = vec![0.; rows * cols];
        for (i, terms) in signed.iter().enumerate() {
            if !terms.is_empty() && scale != 0. {
                seed_row(
                    pass.row(i),
                    terms,
                    scale,
                    &mut seed[i * cols..(i + 1) * cols],
                );
            }
        }
        Ok((report, Tensor::from_vec(rows, cols, seed)?))
    }
}

fn seed_row(z: &[f32], terms: &BTreeMap<usize, Coeff>, scale: f64, dst: &mut [f32]) {
    let total_ce = terms.values().fold(Term::ZERO, |v, t| v.plus(t.ce));
    if total_ce.rounded() == 0. {
        for (&k, t) in terms {
            dst[k] = (scale * t.fork.plus(t.ce.neg()).rounded()) as f32;
        }
        return;
    }
    let peak = Term::logit(z.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    let mut outside = Term::ZERO;
    let mut entries = terms.iter().peekable();
    let mut edges = Vec::with_capacity(terms.len());
    for (k, &logit) in z.iter().enumerate() {
        let e = Term::logit(logit).plus(peak.neg()).exp();
        if entries.peek().is_some_and(|(&id, _)| id == k) {
            edges.push((k, *entries.next().unwrap().1, e));
        } else {
            outside = outside.plus(e);
        }
    }
    let den = edges
        .iter()
        .fold(outside, |s, (_, _, e)| s.plus(*e))
        .rounded();
    let scaled = scale / den;
    for (g, &v) in dst.iter_mut().zip(z) {
        *g = (scaled * total_ce.rounded() * (f64::from(v) - peak.rounded()).exp()) as f32;
    }
    // The row derivative is fork_k + p_k * total_ce - ce_k. Sum its
    // numerator by disjoint events instead of subtracting rounded near-ones.
    for &(k, t, e) in &edges {
        let other = t.fork.plus(t.ce.neg());
        let mut num = outside.times(other).plus(e.times(t.fork));
        for &(j, s, f) in &edges {
            if j != k {
                num = num.plus(e.times(s.ce)).plus(f.times(other));
            }
        }
        dst[k] = (scaled * num.rounded()) as f32;
    }
}

impl NativeCausalLm {
    pub fn validate_line_subset(count: usize, accepted: &[usize]) -> CausalLmResult<()> {
        if !(2..=MAX_LINE_MENU_CANDIDATES).contains(&count)
            || accepted.is_empty()
            || accepted.len() >= count
            || accepted.last().is_some_and(|&k| k >= count)
            || accepted.windows(2).any(|p| p[0] >= p[1])
        {
            return Err(
                "accepted indices must be a sorted distinct nonempty proper subset of the menu"
                    .into(),
            );
        }
        Ok(())
    }

    fn prepare_line_subset(
        &self,
        prompt: &[usize],
        candidates: &[Vec<usize>],
        accepted: &[usize],
        stop: usize,
        training: bool,
    ) -> CausalLmResult<(PreparedCausalWindow, LineSubset)> {
        Self::validate_line_subset(candidates.len(), accepted)?;
        let (batch, support) = self.prepare_line_set(prompt, candidates, stop, training)?;
        Ok((batch, LineSubset::new(support, candidates, accepted)))
    }

    /// -log P(accepted | menu), including each line's stop, with no preferred alias.
    pub fn line_subset_loss(
        &self,
        prompt: &[usize],
        candidates: &[Vec<usize>],
        accepted_indices: &[usize],
        stop_id: usize,
    ) -> CausalLmResult<CausalLmLineSubsetReport> {
        let (batch, plan) =
            self.prepare_line_subset(prompt, candidates, accepted_indices, stop_id, false)?;
        let h = self.decode_hidden(&batch.input, batch.shape)?;
        plan.score(&self.causal_logits(h, &batch)?)
    }

    pub fn geometry_line_subset_gradients(
        &self,
        prompt: &[usize],
        candidates: &[Vec<usize>],
        accepted_indices: &[usize],
        stop_id: usize,
    ) -> CausalLmResult<NativeCausalLmAdaptationGradients> {
        let (mut batch, plan) =
            self.prepare_line_subset(prompt, candidates, accepted_indices, stop_id, true)?;
        batch.competition = Some(objective::PreparedCompetition::LineSubset(plan));
        self.geometry_gradients_for_competition(&batch)
    }
}

#[cfg(test)]
#[path = "subset_tests.rs"]
mod tests;
