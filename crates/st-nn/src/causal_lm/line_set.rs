//! Unrestricted probability of emitting any one of a set of terminated alternatives.
use super::*;
use std::collections::BTreeMap;

mod partition;
mod scores;
pub use scores::CausalLmLineScores;
mod subset;
pub use subset::CausalLmLineSubsetReport;
pub(super) use subset::LineSubset;
mod ranking;
pub(super) use ranking::LineMenu;
mod readout;
use partition::{Group, Outcome, Term};

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmLineSetReport {
    pub stop_id: usize,
    pub target_counts: Vec<usize>,
    /// Actual vocabulary readout rows, including the empty answer prefix.
    pub prefix_nodes: usize,
    /// Actual decoder windows: candidates which are not strict prefixes of other candidates.
    pub decoder_lanes: usize,
    /// Negative log of total accepted probability, without a token or candidate-count divisor.
    pub set_nll: f64,
    pub loss: f32,
}

struct Node {
    lane: usize,
    depth: usize,
    // None denotes the terminating stop. Children are allocated after their parents.
    edges: BTreeMap<usize, Option<usize>>,
}

pub(super) struct LineSet {
    nodes: Vec<Node>,
    leaves: Vec<usize>,
    counts: Vec<usize>,
    stop: usize,
}

impl LineSet {
    fn new(lines: &[Vec<usize>], stop: usize) -> Self {
        let mut nodes = vec![Node {
            lane: 0,
            depth: 0,
            edges: BTreeMap::new(),
        }];
        let mut leaves = Vec::new();
        let mut counts = Vec::with_capacity(lines.len());
        for (j, line) in lines.iter().enumerate() {
            let mut u = 0;
            for (i, &k) in line.iter().enumerate() {
                u = match nodes[u].edges.get(&k) {
                    Some(Some(v)) => *v,
                    _ => {
                        let next = nodes.len();
                        nodes[u].edges.insert(k, Some(next));
                        nodes.push(Node {
                            lane: leaves.len(),
                            depth: i + 1,
                            edges: BTreeMap::new(),
                        });
                        next
                    }
                };
            }
            nodes[u].edges.insert(stop, None);
            counts.push(line.len() + 1);
            // Lexicographic order makes a prefix chain contiguous. Its first
            // maximal descendant supplies the canonical hidden row for every new node.
            if !lines.get(j + 1).is_some_and(|next| next.starts_with(line)) {
                leaves.push(j);
            }
        }
        Self {
            nodes,
            leaves,
            counts,
            stop,
        }
    }

    fn evaluate(
        &self,
        logits: &Tensor,
        gradient: Option<&mut [f32]>,
    ) -> CausalLmResult<CausalLmLineSetReport> {
        let pass = readout::PrefixPass::new(self, logits, gradient.is_some())?;
        let set_nll = -pass.outcomes[0].yes.rounded();
        let loss = set_nll as f32;
        if !loss.is_finite() {
            return Err("terminated-set objective exceeds finite f32 loss".into());
        }
        if let Some(dst) = gradient {
            pass.seed(self, None, dst);
        }
        Ok(CausalLmLineSetReport {
            stop_id: self.stop,
            target_counts: self.counts.clone(),
            prefix_nodes: self.nodes.len(),
            decoder_lanes: self.leaves.len(),
            set_nll,
            loss,
        })
    }

    pub(super) fn loss_gradient(
        &self,
        logits: &Tensor,
    ) -> CausalLmResult<(CausalLmLineSetReport, Tensor)> {
        let (rows, cols) = logits.shape();
        let mut seed = vec![0.; logits.data().len()];
        let report = self.evaluate(logits, Some(&mut seed))?;
        Ok((report, Tensor::from_vec(rows, cols, seed)?))
    }
}

impl NativeCausalLm {
    /// Admit length-only lower bounds before copying token arrays. Exact decoder
    /// lanes and readout rows are admitted after inspecting the bounded contents.
    pub fn line_set_shape(
        &self,
        prompt_len: usize,
        lengths: &[usize],
        stop_id: usize,
        training: bool,
    ) -> CausalLmResult<[usize; 2]> {
        let (shape, _) = self.terminated_shape(prompt_len, lengths, stop_id)?;
        let mut sorted = lengths.to_vec();
        sorted.sort_unstable();
        let (mut min_lanes, mut distinct) = (0, 0);
        for group in sorted.chunk_by(|a, b| a == b) {
            min_lanes = min_lanes.max(group.len());
            distinct += 1;
        }
        // Each line has a distinct terminal node; every missing depth on the
        // longest line also needs at least one node, in addition to the root.
        let min_nodes = 1 + lengths.len() + (sorted.last().unwrap() - distinct);
        self.admit_terminated_work([min_lanes, shape[1]], min_nodes, training)?;
        Ok(shape)
    }

    pub(super) fn prepare_line_set(
        &self,
        prompt: &[usize],
        candidates: &[Vec<usize>],
        stop: usize,
        training: bool,
    ) -> CausalLmResult<(PreparedCausalWindow, LineSet)> {
        if candidates.len() > MAX_LINE_MENU_CANDIDATES {
            return Err("too many complete-line candidates".into());
        }
        let lens = candidates.iter().map(Vec::len).collect::<Vec<_>>();
        let shape = self.line_set_shape(prompt.len(), &lens, stop, training)?;
        self.validate_terminated_contents(prompt, candidates, stop)?;
        let plan = LineSet::new(candidates, stop);
        let physical = [plan.leaves.len(), shape[1]];
        self.admit_terminated_work(physical, plan.nodes.len(), training)?;
        let mut ids = Vec::with_capacity(physical[0] * physical[1]);
        for &j in &plan.leaves {
            let end = ids.len() + physical[1];
            ids.extend_from_slice(prompt);
            ids.extend_from_slice(&candidates[j]);
            ids.resize(end, stop);
        }
        // Unshifted labels select one real hidden state per trie node, in node
        // order. The set seed, not the representative label, scores every edge
        // (including an interior stop). The normal pullback scatters to these rows.
        let mut labels = vec![-100; ids.len()];
        for node in &plan.nodes {
            let i = node.lane * physical[1] + prompt.len() + node.depth;
            labels[i] = ids[i] as i64;
        }
        let mut batch =
            self.prepare_objective(&ids, physical, Some(&labels), CausalLmObjective::default())?;
        debug_assert_eq!(batch.target_tokens, plan.nodes.len());
        batch.target_tokens = plan.counts.iter().sum();
        batch.window_shape = shape;
        Ok((batch, plan))
    }

    /// Absolute NLL of a disjunction of distinct terminated strings; no gold index.
    pub fn line_set_loss(
        &self,
        prompt: &[usize],
        candidates: &[Vec<usize>],
        stop_id: usize,
    ) -> CausalLmResult<CausalLmLineSetReport> {
        let (batch, plan) = self.prepare_line_set(prompt, candidates, stop_id, false)?;
        let hidden = self.decode_hidden(&batch.input, batch.shape)?;
        plan.evaluate(&self.causal_logits(hidden, &batch)?, None)
    }

    /// Geometry-only VJP of the absolute terminated-set NLL. Base and LoRA are frozen.
    pub fn geometry_line_set_gradients(
        &self,
        prompt: &[usize],
        candidates: &[Vec<usize>],
        stop_id: usize,
    ) -> CausalLmResult<NativeCausalLmAdaptationGradients> {
        let (mut batch, plan) = self.prepare_line_set(prompt, candidates, stop_id, true)?;
        batch.competition = Some(objective::PreparedCompetition::LineSet(plan));
        self.geometry_gradients_for_competition(&batch)
    }
}

#[cfg(test)]
mod tests;
