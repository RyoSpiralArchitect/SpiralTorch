//! Compile source-span supervision for a single, already encoded causal document.
//! No tokenizer, padding, special-token insertion or model vocabulary is assumed.
use serde::Serialize;

mod completion;
pub use completion::CausalCompletion;

/// Inputs use one consistent source coordinate system and half-open intervals.
/// Runs are [token_start, token_end, source_start, source_end]. A strict text
/// client can decode each run and compare it with that exact source slice.
#[derive(Debug, Serialize)]
pub struct CausalTargetAlignment {
    token_ids: Vec<u32>,
    labels: Vec<i64>,
    runs: Vec<[u32; 4]>,
}

impl CausalTargetAlignment {
    pub const MAX_SOURCE_UNITS: u32 = 1 << 20;
    pub const MAX_TOKENS: usize = 1 << 20;
    pub const MAX_TARGET_SPANS: usize = 1 << 16;

    /// Also used by foreign bindings before they visit individual elements.
    pub fn admit_counts(
        length: u32,
        tokens: usize,
        offsets: usize,
        spans: usize,
    ) -> Result<(), &'static str> {
        if !(1..=Self::MAX_SOURCE_UNITS).contains(&length) {
            return Err("source length outside causal alignment budget");
        }
        if !(2..=Self::MAX_TOKENS).contains(&tokens) || tokens != offsets {
            return Err("alignment requires bounded token IDs and one offset per token");
        }
        if !(1..=Self::MAX_TARGET_SPANS.min(length as usize)).contains(&spans) {
            return Err("supervised span count outside causal alignment budget");
        }
        Ok(())
    }

    pub fn new(
        length: u32,
        token_ids: Vec<u32>,
        offsets: &[[u32; 2]],
        targets: &[[u32; 2]],
    ) -> Result<Self, &'static str> {
        Self::admit_counts(length, token_ids.len(), offsets.len(), targets.len())?;
        let mut spans: Vec<[u32; 2]> = Vec::with_capacity(targets.len());
        for &[start, end] in targets {
            if start >= end || end > length {
                return Err("supervised spans must be nonempty and inside the source");
            }
            if let Some(previous) = spans.last_mut() {
                if start < previous[1] {
                    return Err("supervised spans must be ordered and nonoverlapping");
                }
                if start == previous[1] {
                    previous[1] = end;
                    continue;
                }
            }
            spans.push([start, end]);
        }
        let mut labels = Vec::with_capacity(token_ids.len());
        let mut runs: Vec<[u32; 4]> = Vec::new();
        let mut span = 0;
        let mut previous = [0, 0];
        let mut was_scored = None;
        for (index, (&id, &[start, end])) in token_ids.iter().zip(offsets).enumerate() {
            // Multiple byte-level tokens can share a Unicode scalar's source
            // interval. Allow monotone overlap, but never missing source units.
            if start >= end
                || end > length
                || start < previous[0]
                || end < previous[1]
                || start > previous[1]
            {
                return Err(
                    "token offsets must cover the source in order without gaps or empty tokens",
                );
            }
            while span < spans.len() && spans[span][1] <= start {
                span += 1;
            }
            let scored = if let Some(&[left, right]) = spans.get(span) {
                if left < end && (start < left || end > right) {
                    return Err("token crosses a supervision boundary; do not split or relabel it");
                }
                left < end
            } else {
                false
            };
            if index == 0 && scored {
                return Err("the first token has no causal predecessor; provide a context token");
            }
            labels.push(if scored { i64::from(id) } else { -100 });
            if was_scored != Some(scored) {
                runs.push([index as u32, index as u32 + 1, start, end]);
                was_scored = Some(scored);
            } else if let Some(run) = runs.last_mut() {
                run[1] = index as u32 + 1;
                run[3] = end;
            }
            previous = [start, end];
        }
        if previous[1] != length {
            return Err("token offsets omit the end of the source");
        }
        Ok(Self {
            token_ids,
            labels,
            runs,
        })
    }

    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }
    pub fn labels(&self) -> &[i64] {
        &self.labels
    }
    pub fn runs(&self) -> &[[u32; 4]] {
        &self.runs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_source_ownership_including_adjacent_spans_and_shared_offsets() {
        let a = CausalTargetAlignment::new(
            7,
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            &[
                [0, 1],
                [1, 2],
                [2, 3],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 6],
                [6, 7],
            ],
            &[[1, 2], [2, 4], [6, 7]],
        )
        .unwrap();
        assert_eq!(a.labels(), &[-100, 2, 3, 4, 5, -100, -100, 8]);
        assert_eq!(
            a.runs(),
            &[[0, 1, 0, 1], [1, 5, 1, 4], [5, 7, 4, 6], [7, 8, 6, 7]]
        );
        let merged =
            CausalTargetAlignment::new(4, vec![0, u32::MAX], &[[0, 1], [1, 4]], &[[1, 2], [2, 4]])
                .unwrap();
        assert_eq!(merged.labels(), &[-100, i64::from(u32::MAX)]);
    }

    #[test]
    fn rejects_incomplete_sources_ambiguous_tokens_and_invalid_targets() {
        let plain = [[0, 1], [1, 2], [2, 3], [3, 4]];
        for spans in [
            vec![],
            vec![[1, 1]],
            vec![[2, 5]],
            vec![[0, 1]],
            vec![[2, 4], [1, 2]],
            vec![[1, 3], [2, 4]],
        ] {
            assert!(CausalTargetAlignment::new(4, vec![1; 4], &plain, &spans).is_err());
        }
        for offsets in [
            vec![[1, 2], [2, 4]],
            vec![[0, 1], [2, 4]],
            vec![[0, 1], [1, 3]],
            vec![[0, 1], [1, 5]],
            vec![[0, 0], [0, 4]],
            vec![[0, 2], [1, 1]],
            vec![[0, 3], [1, 2], [2, 4]],
            vec![[0, 1], [1, 2], [0, 2], [2, 4]],
            vec![[0, 1], [1, 4]],
            vec![[0, 1], [1, 3], [2, 4]],
        ] {
            assert!(
                CausalTargetAlignment::new(4, vec![1; offsets.len()], &offsets, &[[2, 3]]).is_err()
            );
        }
        assert!(CausalTargetAlignment::admit_counts(4, usize::MAX, usize::MAX, 1).is_err());
        assert!(CausalTargetAlignment::admit_counts(4, 2, 1, 1).is_err());
    }

    #[test]
    fn agrees_with_exhaustive_small_partitions_without_leaking_prompt_units() {
        let length = 6;
        for cuts in 0..1u32 << (length - 1) {
            let mut offsets = Vec::new();
            let mut start = 0;
            for end in 1..=length {
                if end == length || cuts & (1 << (end - 1)) != 0 {
                    offsets.push([start, end]);
                    start = end;
                }
            }
            for mask in 1..1u32 << length {
                let spans: Vec<_> = (0..length)
                    .filter(|&i| mask & (1 << i) != 0)
                    .map(|i| [i, i + 1])
                    .collect();
                let mut expected = Vec::new();
                let mut valid = offsets.len() >= 2 && mask & 1 == 0;
                for (index, &[s, e]) in offsets.iter().enumerate() {
                    let supervised = mask & (1 << s) != 0;
                    valid &= (s..e).all(|t| (mask & (1 << t) != 0) == supervised);
                    expected.push(if supervised { index as i64 } else { -100 });
                }
                let plan = CausalTargetAlignment::new(
                    length,
                    (0..offsets.len() as u32).collect(),
                    &offsets,
                    &spans,
                );
                assert_eq!(plan.is_ok(), valid, "{offsets:?} {spans:?}");
                if let Ok(plan) = plan {
                    assert_eq!(plan.labels(), expected);
                }
            }
        }
    }
}
