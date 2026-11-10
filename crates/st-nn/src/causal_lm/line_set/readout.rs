//! One physical row per prefix. A success partition includes the full suffix
//! probability, so its edge probabilities are conditional on eventual acceptance.
use super::*;
use crate::causal_lm::sum::Sum;
use std::borrow::Cow;

pub(super) struct PrefixPass<'a> {
    logits: Cow<'a, Tensor>,
    pub(super) outcomes: Vec<Outcome>,
    pub(super) rows: Vec<Row>,
}

#[derive(Clone, Copy, Default)]
pub(super) struct Row {
    pub(super) accepted: Group,
    pub(super) total: Group,
    outside: f64,
}

#[derive(Clone, Copy)]
pub(super) struct RankingSeed<'a> {
    pub gold_tokens: &'a [Option<usize>],
    pub mix: f64,
    pub gold_ce: f64,
}

impl<'a> PrefixPass<'a> {
    pub(super) fn new(set: &LineSet, logits: &'a Tensor, keep_rows: bool) -> CausalLmResult<Self> {
        let (count, cols) = logits.shape();
        if count != set.nodes.len() || logits.data().iter().any(|v| !v.is_finite()) {
            return Err("unexpected or nonfinite terminated-line logits".into());
        }
        let logits = if logits.layout() == st_tensor::Layout::RowMajor {
            Cow::Borrowed(logits)
        } else {
            Cow::Owned(logits.to_layout(st_tensor::Layout::RowMajor)?)
        };
        let mut outcomes = vec![Outcome::STOP; count];
        let mut rows = if keep_rows {
            vec![Row::default(); count]
        } else {
            Vec::new()
        };
        for (i, node) in set.nodes.iter().enumerate().rev() {
            let z = &logits.data()[i * cols..(i + 1) * cols];
            let mut outside = Group::default();
            let mut allowed = node.edges.keys().copied().peekable();
            for (k, &v) in z.iter().enumerate() {
                if allowed.peek() == Some(&k) {
                    allowed.next();
                } else {
                    outside.add(Term::logit(v));
                }
            }
            let (mut yes, mut no) = (Group::default(), outside);
            for (&k, &child) in &node.edges {
                let value = Term::logit(z[k]);
                let s = child.map_or(Outcome::STOP, |j| outcomes[j]);
                yes.add(value.plus(s.yes));
                if let Some(failure) = s.no {
                    no.add(value.plus(failure));
                }
            }
            outcomes[i] = Outcome::new(yes, no);
            if keep_rows {
                let mut total = outside;
                for &k in node.edges.keys() {
                    total.add(Term::logit(z[k]));
                }
                rows[i] = Row {
                    accepted: yes,
                    total,
                    outside: outside.mass_in(total),
                };
            }
        }
        Ok(Self {
            logits,
            outcomes,
            rows,
        })
    }

    pub(super) fn row(&self, i: usize) -> &[f32] {
        let cols = self.logits.shape().1;
        &self.logits.data()[i * cols..(i + 1) * cols]
    }

    pub(super) fn seed(
        &self,
        set: &LineSet,
        ranking: Option<RankingSeed<'_>>,
        gradient: &mut [f32],
    ) {
        let cols = self.logits.shape().1;
        let mut reach = vec![0.; set.nodes.len()];
        reach[0] = 1.;
        let mut earlier_rivals = Sum::default();
        for (i, node) in set.nodes.iter().enumerate() {
            let row = self.rows[i];
            let z = self.row(i);
            let dst = &mut gradient[i * cols..(i + 1) * cols];
            let r = reach[i];
            let on_gold = ranking.and_then(|s| s.gold_tokens[i].map(|token| (s, token)));
            if on_gold.is_none() && (r == 0. || ranking.is_some_and(|s| s.mix == 0.)) {
                continue;
            }
            let mut edges = Vec::with_capacity(node.edges.len());
            for (&k, &child) in &node.edges {
                let value = Term::logit(z[k]);
                let success = child.map_or(Term::ZERO, |j| self.outcomes[j].yes);
                let p = row.total.share(value);
                let q = row.accepted.share(value.plus(success));
                if let Some(j) = child {
                    reach[j] = r * q;
                }
                edges.push((k, p, q, success));
            }
            if let Some((recipe, t)) = on_gold {
                // dC = (1-r)(p-e_t) + r(q-e_t). Accumulate 1-r from rivals
                // that actually diverged earlier, never by subtracting r from 1.
                let ce = recipe.gold_ce + recipe.mix * earlier_rivals.value();
                let mut other_p = Sum::default();
                other_p.add(row.outside);
                let mut diverging = Sum::default();
                if ce != 0. {
                    for (g, &v) in dst.iter_mut().zip(z) {
                        *g = (ce * row.total.share(Term::logit(v))) as f32;
                    }
                }
                for &(k, p, q, _) in &edges {
                    if k != t {
                        other_p.add(p);
                        diverging.add(r * q);
                        dst[k] = (ce * p + recipe.mix * (r * q)) as f32;
                    }
                }
                dst[t] = (-ce * other_p.value() - recipe.mix * diverging.value()) as f32;
                earlier_rivals.merge(&diverging);
            } else {
                // Off the gold path dC = -dS. For set NLL S, sum suffix
                // differences directly instead of subtracting two rounded softmaxes.
                let scale = r * ranking.map_or(1., |s| -s.mix);
                for (g, &v) in dst.iter_mut().zip(z) {
                    *g = (scale * row.total.share(Term::logit(v))) as f32;
                }
                for &(k, p, q, s) in &edges {
                    let mut d = Sum::default();
                    d.add(-q * row.outside);
                    for &(_, p2, q2, s2) in &edges {
                        let gap = s.gap(s2);
                        if gap < 0. {
                            d.add(-p * q2 * gap.exp_m1());
                        } else if gap > 0. {
                            d.add(q * p2 * (-gap).exp_m1());
                        }
                    }
                    dst[k] = (scale * d.value()) as f32;
                }
            }
        }
    }
}
