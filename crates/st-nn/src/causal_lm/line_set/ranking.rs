use super::readout::{PrefixPass, RankingSeed};
use super::*;
use crate::causal_lm::sum::Sum;

pub(super) type Path = Vec<(usize, usize)>;

pub(super) fn paths(support: &LineSet, lines: &[Vec<usize>]) -> (usize, Vec<Path>) {
    let shared = lines[0]
        .iter()
        .zip(lines.last().unwrap())
        .take_while(|(a, b)| a == b)
        .count();
    let paths = lines
        .iter()
        .map(|line| {
            let mut i = 0;
            let mut path = Vec::with_capacity(line.len() + 1);
            for &k in line {
                path.push((i, k));
                i = support.nodes[i].edges[&k].expect("admitted content edge");
            }
            path.push((i, support.stop));
            path
        })
        .collect();
    (shared, paths)
}

pub(super) struct Costs {
    pub line: Vec<f64>,
    pub suffix: Vec<Term>,
    pub conditional: Vec<Term>,
}

pub(super) fn winner_index(suffixes: &[Term]) -> usize {
    (1..suffixes.len()).fold(0, |best, i| {
        if suffixes[i].gap(suffixes[best]) < 0. {
            i
        } else {
            best
        }
    })
}

pub(super) fn costs(
    support: &LineSet,
    paths: &[Path],
    shared: usize,
    pass: &PrefixPass<'_>,
) -> Costs {
    costs_with_tokens(support, paths, shared, pass, |_, _| {})
}

pub(super) fn costs_with_tokens(
    support: &LineSet,
    paths: &[Path],
    shared: usize,
    pass: &PrefixPass<'_>,
    mut visit: impl FnMut(usize, f64),
) -> Costs {
    let (mut lines, mut suffixes, mut conditional) = (Vec::new(), Vec::new(), Vec::new());
    for (line, path) in paths.iter().enumerate() {
        let (mut prefix, mut suffix, mut cost) = (Term::ZERO, Term::ZERO, Term::ZERO);
        for (depth, &(i, k)) in path.iter().enumerate() {
            let value = Term::logit(pass.row(i)[k]);
            let s = support.nodes[i].edges[&k].map_or(Term::ZERO, |j| pass.outcomes[j].yes);
            let row = pass.rows[i];
            let token_nll = row.total.nll(value);
            visit(line, token_nll.rounded());
            if depth < shared {
                prefix = prefix.plus(token_nll);
            } else {
                suffix = suffix.plus(token_nll);
            }
            cost = cost.plus(row.accepted.nll(value.plus(s)));
        }
        lines.push(prefix.plus(suffix).rounded());
        suffixes.push(suffix);
        conditional.push(cost);
    }
    Costs {
        line: lines,
        suffix: suffixes,
        conditional,
    }
}

pub(in crate::causal_lm) struct LineMenu {
    support: LineSet,
    paths: Vec<Vec<(usize, usize)>>,
    shared: usize,
    gold: usize,
    mix: f64,
}

impl LineMenu {
    pub(in crate::causal_lm) fn new(
        support: LineSet,
        lines: &[Vec<usize>],
        gold: usize,
        mix: f64,
    ) -> Self {
        let (shared, paths) = paths(&support, lines);
        Self {
            support,
            paths,
            shared,
            gold,
            mix,
        }
    }

    fn report(&self, pass: &PrefixPass<'_>) -> CausalLmResult<CausalLmLineMenuReport> {
        let Costs {
            line: lines,
            suffix: suffixes,
            conditional,
        } = costs(&self.support, &self.paths, self.shared, pass);
        let winner = winner_index(&suffixes);
        let mut rivals = Sum::default();
        let mut probabilities = Vec::with_capacity(conditional.len());
        for (i, &cost) in conditional.iter().enumerate() {
            let q = (-cost.rounded()).exp();
            probabilities.push(q);
            if i != self.gold {
                rivals.add(q);
            }
        }
        let menu_nll = conditional[self.gold].rounded();
        let gold_mean_nll = lines[self.gold] / self.support.counts[self.gold] as f64;
        let loss = ((1. - self.mix) * gold_mean_nll + self.mix * menu_nll) as f32;
        if !loss.is_finite() {
            return Err("complete-line objective exceeds finite f32 loss".into());
        }
        Ok(CausalLmLineMenuReport {
            gold_index: self.gold,
            winner_index: winner,
            shared_content_tokens: self.shared,
            stop_id: self.support.stop,
            target_counts: self.support.counts.clone(),
            line_nlls: lines,
            suffix_nlls: suffixes.into_iter().map(Term::rounded).collect(),
            menu_probabilities: probabilities,
            rival_mass: rivals.value(),
            gold_mean_nll,
            menu_nll,
            mix: self.mix,
            prefix_nodes: self.support.nodes.len(),
            decoder_lanes: self.support.leaves.len(),
            loss,
        })
    }

    pub(in crate::causal_lm) fn score(
        &self,
        logits: &Tensor,
    ) -> CausalLmResult<CausalLmLineMenuReport> {
        self.report(&PrefixPass::new(&self.support, logits, true)?)
    }

    pub(in crate::causal_lm) fn loss_gradient(
        &self,
        logits: &Tensor,
    ) -> CausalLmResult<(CausalLmLineMenuReport, Tensor)> {
        let pass = PrefixPass::new(&self.support, logits, true)?;
        let report = self.report(&pass)?;
        let (rows, cols) = logits.shape();
        let mut gold_tokens = vec![None; rows];
        for &(i, k) in &self.paths[self.gold] {
            gold_tokens[i] = Some(k);
        }
        let recipe = RankingSeed {
            gold_tokens: &gold_tokens,
            mix: self.mix,
            gold_ce: (1. - self.mix) / self.paths[self.gold].len() as f64,
        };
        let mut seed = vec![0.; logits.data().len()];
        pass.seed(&self.support, Some(recipe), &mut seed);
        Ok((report, Tensor::from_vec(rows, cols, seed)?))
    }
}

#[cfg(test)]
#[path = "ranking_tests.rs"]
mod tests;
