//! Label-free readout of complete, distinct terminated alternatives.
use super::*;

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmLineScores {
    pub stop_id: usize,
    pub target_counts: Vec<usize>,
    pub prefix_nodes: usize,
    pub decoder_lanes: usize,
    pub shared_content_tokens: usize,
    /// First in input order on an exact tie. Compared before rounding away a shared cost.
    pub winner_index: usize,
    pub line_nlls: Vec<f64>,
    /// Negative log probabilities conditioned on membership of the supplied menu.
    pub conditional_nlls: Vec<f64>,
    /// Full-vocabulary costs, one for each content token and then exactly one stop.
    pub token_nlls: Vec<Vec<f64>>,
    pub menu_set_nll: f64,
}

impl LineSet {
    fn scores(&self, lines: &[Vec<usize>], logits: &Tensor) -> CausalLmResult<CausalLmLineScores> {
        let pass = readout::PrefixPass::new(self, logits, true)?;
        let (shared, paths) = ranking::paths(self, lines);
        let mut tokens: Vec<_> = self
            .counts
            .iter()
            .map(|&len| Vec::with_capacity(len))
            .collect();
        let costs = ranking::costs_with_tokens(self, &paths, shared, &pass, |line, value| {
            tokens[line].push(value);
        });
        Ok(CausalLmLineScores {
            stop_id: self.stop,
            target_counts: self.counts.clone(),
            prefix_nodes: self.nodes.len(),
            decoder_lanes: self.leaves.len(),
            shared_content_tokens: shared,
            winner_index: ranking::winner_index(&costs.suffix),
            line_nlls: costs.line,
            conditional_nlls: costs.conditional.into_iter().map(Term::rounded).collect(),
            token_nlls: tokens,
            menu_set_nll: -pass.outcomes[0].yes.rounded(),
        })
    }
}

impl NativeCausalLm {
    /// Score a sorted, nonempty menu without choosing a gold answer or training objective.
    pub fn score_lines(
        &self,
        prompt: &[usize],
        candidates: &[Vec<usize>],
        stop_id: usize,
    ) -> CausalLmResult<CausalLmLineScores> {
        let (batch, plan) = self.prepare_line_set(prompt, candidates, stop_id, false)?;
        let hidden = self.decode_hidden(&batch.input, batch.shape)?;
        plan.scores(candidates, &self.causal_logits(hidden, &batch)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unlabelled_winner_survives_huge_shared_cost_and_conditional_underflow() {
        let lines = [vec![1, 2], vec![1, 3]];
        let set = LineSet::new(&lines, 0);
        let logits = Tensor::from_vec(
            4,
            4,
            [[0., -1e30, 0., 0.], [0., 0., -1200., 0.], [0.; 4], [0.; 4]].concat(),
        )
        .unwrap();
        let r = set.scores(&lines, &logits).unwrap();
        assert_eq!(r.line_nlls[0], r.line_nlls[1]);
        assert_eq!(r.winner_index, 1);
        assert_eq!(r.shared_content_tokens, 1);
        assert_eq!(r.token_nlls[0][0], r.token_nlls[1][0]);
        assert_eq!(r.token_nlls[0].len(), 3);
        assert!((r.conditional_nlls[0] - 1200.).abs() < 1e-10);
        assert_eq!((-r.conditional_nlls[0]).exp(), 0.);
        let tie = set.scores(&lines, &Tensor::zeros(4, 4).unwrap()).unwrap();
        assert_eq!(tie.winner_index, 0);
        assert!((tie.conditional_nlls[0] - 2f64.ln()).abs() < 1e-14);
    }

    #[test]
    fn singleton_charges_the_stop_without_a_gold_or_a_f32_loss_cast() {
        let lines = [vec![1]];
        let set = LineSet::new(&lines, 0);
        let logits =
            Tensor::from_vec(2, 2, vec![f32::MAX, -f32::MAX, -f32::MAX, f32::MAX]).unwrap();
        let r = set.scores(&lines, &logits).unwrap();
        assert_eq!(r.target_counts, [2]);
        assert_eq!(r.shared_content_tokens, 1);
        assert_eq!(r.winner_index, 0);
        assert_eq!(r.conditional_nlls, [0.]);
        assert_eq!(r.token_nlls[0], [2. * f64::from(f32::MAX); 2]);
        assert_eq!(r.line_nlls, [4. * f64::from(f32::MAX)]);
        assert_eq!(r.line_nlls[0], r.menu_set_nll);
    }
}
