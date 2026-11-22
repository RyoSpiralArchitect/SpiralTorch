//! Feed the same terminated-line scorer from one cached prompt and one decode per content prefix.
use super::*;

#[derive(Debug, Serialize)]
pub struct CausalLmCachedLineScores {
    pub scores: CausalLmLineScores,
    pub cached_prompt_tokens: usize,
    pub decoded_content_tokens: usize,
    pub decoder_chunks: usize,
    /// Persistent K/V + geometry f32 values in all live forks, excluding the supplied root and temporaries.
    pub peak_fork_cache_values: usize,
}

struct Chunk {
    ids: Vec<usize>,
    nodes: Vec<usize>,
    children: Vec<Chunk>,
}

impl LineSet {
    fn children(&self, i: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.nodes[i]
            .edges
            .iter()
            .filter_map(|(&id, &node)| node.map(|j| (id, j)))
    }

    // Only real forks recurse (depth <= number of candidates). An interior stop still
    // gets scored, but does not interrupt a chain with one continuing content edge.
    fn chunks(&self, parent: usize) -> Vec<Chunk> {
        self.children(parent)
            .map(|mut edge| {
                let (mut ids, mut nodes) = (Vec::new(), Vec::new());
                loop {
                    let (id, i) = edge;
                    ids.push(id);
                    nodes.push(i);
                    let mut next = self.children(i);
                    match (next.next(), next.next()) {
                        (Some(child), None) => edge = child,
                        _ => {
                            return Chunk {
                                ids,
                                nodes,
                                children: self.chunks(i),
                            }
                        }
                    }
                }
            })
            .collect()
    }
}

#[derive(Default)]
struct Work {
    chunks: usize,
    content: usize,
    peak: usize,
}

impl Chunk {
    fn admit(
        &self,
        root: &NativeCausalLmContext,
        start: usize,
        live: usize,
        work: &mut Work,
    ) -> CausalLmResult<()> {
        let end = start
            .checked_add(self.ids.len())
            .ok_or("cached line position overflow")?;
        let model = root.decoder();
        let budget = root.max_cache_values();
        model.cached_query_count(end, self.ids.len(), budget)?;
        let live = live
            .checked_add(model.context_cache_values(end, budget)?)
            .filter(|&v| v <= budget)
            .ok_or("live complete-line forks exceed cache value budget")?;
        work.chunks += 1;
        work.content += self.ids.len();
        work.peak = work.peak.max(live);
        for child in &self.children {
            child.admit(root, end, live, work)?;
        }
        Ok(())
    }

    fn decode(
        &self,
        parent: &NativeCausalLmContext,
        dst: &mut [f32],
        vocab: usize,
    ) -> CausalLmResult<()> {
        let (next, logits) = parent.scored_suffix(&self.ids)?;
        for (&node, row) in self.nodes.iter().zip(logits.data().chunks_exact(vocab)) {
            dst[node * vocab..(node + 1) * vocab].copy_from_slice(row);
        }
        drop(logits);
        for child in &self.children {
            child.decode(&next, dst, vocab)?;
        }
        Ok(())
    }
}

impl NativeCausalLmContext {
    /// Length-only admission, before extracting foreign token arrays. No padded prompt lanes.
    /// The exact radix-tree shapes and depth-first live cache are checked once IDs are known.
    pub fn admit_line_lengths(&self, lengths: &[usize], stop: usize) -> CausalLmResult<()> {
        let m = self.decoder();
        let (width, _) = m.terminated_lengths(self.token_ids().len(), lengths, stop)?;
        m.cached_input_count(width - 1, self.max_cache_values())?;
        m.logit_count(1 + lengths.iter().copied().max().unwrap().max(lengths.len()))?;
        Ok(())
    }

    /// Score each sorted, nonempty content sequence plus exactly one stop. No root mutation,
    /// full-prompt replay, sampling, or candidate-only vocabulary normalization.
    pub fn score_lines(
        &self,
        lines: &[Vec<usize>],
        stop: usize,
    ) -> CausalLmResult<CausalLmCachedLineScores> {
        if lines.len() > MAX_LINE_MENU_CANDIDATES {
            return Err("too many complete-line candidates".into());
        }
        self.admit_line_lengths(&lines.iter().map(Vec::len).collect::<Vec<_>>(), stop)?;
        let m = self.decoder();
        m.validate_terminated_contents(self.token_ids(), lines, stop)?;
        let set = LineSet::new(lines, stop);
        let count = m.logit_count(set.nodes.len())?;
        let chunks = set.chunks(0);
        let prompt = self.token_ids().len();
        let mut work = Work::default();
        for c in &chunks {
            c.admit(self, prompt, 0, &mut work)?;
        }
        let cols = m.config.vocabulary;
        let mut rows = Vec::new();
        rows.try_reserve_exact(count)?;
        rows.resize(count, 0.);
        rows[..cols].copy_from_slice(self.logit_values());
        for c in &chunks {
            c.decode(self, &mut rows, cols)?;
        }
        let logits = Tensor::from_vec(set.nodes.len(), cols, rows)?;
        Ok(CausalLmCachedLineScores {
            scores: set.scores(lines, &logits)?,
            cached_prompt_tokens: prompt,
            decoded_content_tokens: work.content,
            decoder_chunks: work.chunks,
            peak_fork_cache_values: work.peak,
        })
    }
}
