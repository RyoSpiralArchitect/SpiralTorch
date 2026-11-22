//! Feed the same terminated-line scorer from one cached prompt and one decode per content prefix.
use super::stream::EdgeRows;
use super::*;
use crate::causal_lm::context::KvPrefix;

#[derive(Debug, Serialize)]
pub struct CausalLmCachedLineScores {
    pub scores: CausalLmLineScores,
    pub cached_prompt_tokens: usize,
    pub decoded_content_tokens: usize,
    pub decoder_chunks: usize,
    /// Persistent K/V + geometry f32 values in all live forks, excluding the supplied root and temporaries.
    pub peak_fork_cache_values: usize,
    /// Full-vocabulary values consumed, including the supplied prompt's already computed row.
    pub scanned_logit_values: usize,
    /// Largest newly projected output tile, excluding the supplied root and kernel temporaries.
    pub peak_projected_logit_values: usize,
    pub readout_batches: usize,
    /// One raw f32 logit per unique content edge and one terminating edge per line.
    pub retained_edge_logits: usize,
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

struct Stream<'a> {
    rows: EdgeRows<'a>,
    tile_rows: usize,
    scanned: usize,
    peak_tile: usize,
    batches: usize,
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
        root: &NativeCausalLmContext,
        prefix: KvPrefix<'_>,
        stream: &mut Stream<'_>,
    ) -> CausalLmResult<()> {
        let next =
            root.visit_scored_suffix(prefix, &self.ids, stream.tile_rows, |start, tile| {
                let (n, vocab) = tile.shape();
                stream.scanned += tile.data().len();
                stream.peak_tile = stream.peak_tile.max(tile.data().len());
                stream.batches += 1;
                for (&node, row) in self.nodes[start..start + n]
                    .iter()
                    .zip(tile.data().chunks_exact(vocab))
                {
                    stream.rows.capture(node, row)?;
                }
                Ok(())
            })?;
        for child in &self.children {
            child.decode(root, next.prefix(), stream)?;
        }
        Ok(())
    }
}

impl NativeCausalLmContext {
    /// A tile budget may lower peak projection size, never raise the model's cumulative work limit.
    pub fn line_readout_tile_rows(&self, max_tile_values: Option<usize>) -> CausalLmResult<usize> {
        let m = self.decoder();
        let vocab = m.config.vocabulary;
        let limit = m.limits.max_logit_elements;
        let cap = max_tile_values.unwrap_or_else(|| (256 * 1024).max(vocab).min(limit));
        if !(vocab..=limit).contains(&cap) {
            return Err("complete-line tile budget must fit one vocabulary row and not exceed the model logit limit".into());
        }
        Ok(cap / vocab)
    }

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
        self.score_lines_with_tile_limit(lines, stop, None)
    }

    /// Same terminated-line scoring; optionally bound a projected tile in f32 values.
    pub fn score_lines_with_tile_limit(
        &self,
        lines: &[Vec<usize>],
        stop: usize,
        max_tile_values: Option<usize>,
    ) -> CausalLmResult<CausalLmCachedLineScores> {
        let tile_rows = self.line_readout_tile_rows(max_tile_values)?;
        if lines.len() > MAX_LINE_MENU_CANDIDATES {
            return Err("too many complete-line candidates".into());
        }
        self.admit_line_lengths(&lines.iter().map(Vec::len).collect::<Vec<_>>(), stop)?;
        let m = self.decoder();
        m.validate_terminated_contents(self.token_ids(), lines, stop)?;
        let set = LineSet::new(lines, stop);
        let expected = m.logit_count(set.nodes.len())?;
        let chunks = set.chunks(0);
        let prompt = self.token_ids().len();
        let mut work = Work::default();
        for c in &chunks {
            c.admit(self, prompt, 0, &mut work)?;
        }
        let mut stream = Stream {
            rows: EdgeRows::new(&set),
            tile_rows,
            scanned: self.logit_values().len(),
            peak_tile: 0,
            batches: 0,
        };
        stream.rows.capture(0, self.logit_values())?;
        for c in &chunks {
            c.decode(self, self.kv_prefix(), &mut stream)?;
        }
        debug_assert_eq!(stream.scanned, expected);
        Ok(CausalLmCachedLineScores {
            retained_edge_logits: stream.rows.edge_count(),
            scores: stream.rows.finish(lines)?,
            scanned_logit_values: stream.scanned,
            readout_batches: stream.batches,
            peak_projected_logit_values: stream.peak_tile,
            cached_prompt_tokens: prompt,
            decoded_content_tokens: work.content,
            decoder_chunks: work.chunks,
            peak_fork_cache_values: work.peak,
        })
    }
}
