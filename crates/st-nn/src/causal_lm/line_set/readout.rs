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

pub(super) fn outside_group(node: &Node, z: &[f32]) -> CausalLmResult<Group> {
    if node
        .edges
        .last_key_value()
        .is_some_and(|(&k, _)| k >= z.len())
    {
        return Err("terminated-line edge outside vocabulary".into());
    }
    let mut outside = Group::default();
    let mut allowed = node.edges.keys().copied().peekable();
    for (k, &v) in z.iter().enumerate() {
        if !v.is_finite() {
            return Err("nonfinite terminated-line logits".into());
        }
        if allowed.peek() == Some(&k) {
            allowed.next();
        } else {
            outside.add(Term::logit(v));
        }
    }
    Ok(outside)
}

// Both the dense gradient path and streaming inference reduce nonedges first,
// followed by allowed edges in token-ID order. Keep the two-part terms intact.
pub(super) fn reduce_row(
    node: &Node,
    outside: Group,
    outcomes: &[Outcome],
    values: impl Iterator<Item = f32>,
    keep: bool,
) -> (Outcome, Row) {
    let (mut yes, mut no, mut total) = (Group::default(), outside, outside);
    for (&child, v) in node.edges.values().zip(values) {
        let value = Term::logit(v);
        let s = child.map_or(Outcome::STOP, |j| outcomes[j]);
        yes.add(value.plus(s.yes));
        if let Some(failure) = s.no {
            no.add(value.plus(failure));
        }
        if keep {
            total.add(value);
        }
    }
    let row = if keep {
        Row {
            accepted: yes,
            total,
            outside: outside.mass_in(total),
        }
    } else {
        Row::default()
    };
    (Outcome::new(yes, no), row)
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
        if count != set.nodes.len() {
            return Err("unexpected terminated-line logit shape".into());
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
            let (outcome, row) = reduce_row(
                node,
                outside_group(node, z)?,
                &outcomes,
                node.edges.keys().map(|&k| z[k]),
                keep_rows,
            );
            outcomes[i] = outcome;
            if keep_rows {
                rows[i] = row;
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
        let mut cursor = SeedCursor::new(set.nodes.len());
        for (i, dst) in gradient.chunks_exact_mut(cols).enumerate() {
            cursor.write(
                set,
                &self.outcomes,
                self.rows[i],
                ranking,
                i,
                self.row(i),
                dst,
            );
        }
    }
}

pub(super) struct SeedCursor {
    reach: Vec<f64>,
    earlier_rivals: Sum,
}

impl SeedCursor {
    pub(super) fn new(rows: usize) -> Self {
        let mut reach = vec![0.; rows];
        reach[0] = 1.;
        Self {
            reach,
            earlier_rivals: Sum::default(),
        }
    }

    /// Rows arrive once in parent-before-child order, with a zeroed destination.
    pub(super) fn write(
        &mut self,
        set: &LineSet,
        outcomes: &[Outcome],
        row: Row,
        ranking: Option<RankingSeed<'_>>,
        i: usize,
        z: &[f32],
        dst: &mut [f32],
    ) {
        let node = &set.nodes[i];
        let r = self.reach[i];
        let on_gold = ranking.and_then(|s| s.gold_tokens[i].map(|token| (s, token)));
        if on_gold.is_none() && (r == 0. || ranking.is_some_and(|s| s.mix == 0.)) {
            return;
        }
        let mut edges = Vec::with_capacity(node.edges.len());
        for (&k, &child) in &node.edges {
            let value = Term::logit(z[k]);
            let success = child.map_or(Term::ZERO, |j| outcomes[j].yes);
            let p = row.total.share(value);
            let q = row.accepted.share(value.plus(success));
            if let Some(j) = child {
                self.reach[j] = r * q;
            }
            edges.push((k, p, q, success));
        }
        if let Some((recipe, t)) = on_gold {
            // dC = (1-r)(p-e_t) + r(q-e_t). Accumulate 1-r from rivals
            // that actually diverged earlier, never by subtracting r from 1.
            let ce = recipe.gold_ce + recipe.mix * self.earlier_rivals.value();
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
            self.earlier_rivals.merge(&diverging);
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

pub(super) trait PrefixStats {
    fn outcomes(&self) -> &[Outcome];
    fn rows(&self) -> &[Row];
    fn edge(&self, i: usize, k: usize) -> f32;
}

impl PrefixStats for PrefixPass<'_> {
    fn outcomes(&self) -> &[Outcome] {
        &self.outcomes
    }
    fn rows(&self) -> &[Row] {
        &self.rows
    }
    fn edge(&self, i: usize, k: usize) -> f32 {
        self.row(i)[k]
    }
}
