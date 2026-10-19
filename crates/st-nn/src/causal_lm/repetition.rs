//! Native objective v1: canonical history matching, native deterministic proposals.
use super::*;
use st_core::runtime::zspace_repetition_unlikelihood::{
    admit_repetition_unlikelihood_request, select_repetition_unlikelihood_candidates,
    ZSpaceRepetitionUnlikelihoodCandidateSource as Source,
    ZSpaceRepetitionUnlikelihoodConfig as Config, ZSpaceRepetitionUnlikelihoodRequest as Request,
    ZSpaceRepetitionUnlikelihoodSequence as Sequence,
    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_MATERIALIZED_PLAN_BYTES as MAX_BYTES,
    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SEQUENCES as MAX_SEQUENCES,
    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_TOKENS_PER_SEQUENCE as MAX_SEQUENCE,
    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_TOTAL_TOKENS as MAX_TOKENS,
    ZSPACE_REPETITION_UNLIKELIHOOD_PROBABILITY_EPSILON as EPSILON,
};
use st_tensor::UnlikelihoodPosition;

/// Version binds native logit-descending / token-ID-ascending top-k and separate
/// target-count CE versus active-position unlikelihood microbatch normalization.
/// Proposals are stop-gradient; the historical HF v3 plan/torch tie rule is not used.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CausalLmRepetition {
    NativeV1 { config: Config },
}

impl CausalLmRepetition {
    pub fn config(self) -> Config {
        match self {
            Self::NativeV1 { config } => config,
        }
    }
    pub fn validate(self) -> CausalLmResult<()> {
        self.config().validate()?;
        Ok(())
    }
    fn top_k(self) -> Option<usize> {
        match self.config().candidate_source {
            Source::PriorContinuation { .. } => None,
            Source::ModelTopkHistory { proposal_top_k }
            | Source::ModelTopkPeriodic { proposal_top_k } => Some(proposal_top_k),
        }
    }
    pub(super) fn admit_shape(self, shape: [usize; 2], vocabulary: usize) -> CausalLmResult<()> {
        self.validate()?;
        if shape[0] == 0 || shape[0] > MAX_SEQUENCES || shape[1] > MAX_SEQUENCE {
            return Err("native repetition window exceeds sequence limits".into());
        }
        let tokens = product(&shape, MAX_TOKENS, "repetition tokens")?;
        if let Some(k) = self.top_k() {
            if k > vocabulary {
                return Err("native repetition top-k exceeds vocabulary".into());
            }
            product(
                &[tokens, k],
                MAX_BYTES as usize / 17,
                "repetition proposal storage",
            )?;
        }
        Ok(())
    }
}

pub(super) struct RepetitionWindow {
    request: Request,
}

impl RepetitionWindow {
    pub fn prepare(
        recipe: CausalLmRepetition,
        windows: &[usize],
        shape: [usize; 2],
        labels: Option<&[i64]>,
        vocabulary: usize,
    ) -> CausalLmResult<Option<Self>> {
        recipe.admit_shape(shape, vocabulary)?;
        if recipe.config().strength == 0. {
            return Ok(None);
        }
        if labels.is_some_and(|labels| {
            labels
                .iter()
                .zip(windows)
                .any(|(&label, &id)| label != -100 && label != id as i64)
        }) {
            return Err("repetition labels must equal input IDs or -100".into());
        }
        let sequences = windows
            .chunks_exact(shape[1])
            .enumerate()
            .map(|(b, row)| {
                let label_mask: Vec<_> = (0..row.len())
                    .map(|t| t > 0 && labels.is_none_or(|l| l[b * shape[1] + t] != -100))
                    .collect();
                let proposal_token_ids = recipe.top_k().map(|k| {
                    label_mask
                        .iter()
                        .map(|&active| {
                            if active {
                                (0..k as u64).collect()
                            } else {
                                Vec::new()
                            }
                        })
                        .collect()
                });
                Sequence {
                    token_ids: row.iter().map(|&v| v as u64).collect(),
                    token_mask: vec![true; row.len()],
                    label_mask,
                    proposal_token_ids,
                }
            })
            .collect();
        let request = Request {
            config: recipe.config(),
            sequences,
        };
        admit_repetition_unlikelihood_request(&request)?;
        Ok(Some(Self { request }))
    }

    pub fn loss_gradient(
        mut self,
        logits: &Tensor,
        selected_rows: Option<&[usize]>,
        steps: usize,
    ) -> CausalLmResult<(f32, Tensor, usize)> {
        let index = |flat: usize| -> CausalLmResult<usize> {
            match selected_rows {
                Some(rows) => rows
                    .binary_search(&flat)
                    .map_err(|_| "unscored repetition row".into()),
                None => Ok(flat),
            }
        };
        let classes = logits.shape().1;
        let mut ranks = Vec::new();
        for (b, sequence) in self.request.sequences.iter_mut().enumerate() {
            if let Some(proposals) = &mut sequence.proposal_token_ids {
                for (t, ids) in proposals.iter_mut().enumerate().skip(1) {
                    if ids.is_empty() {
                        continue;
                    }
                    let row = index(b * steps + t - 1)?;
                    let values = &logits.data()[row * classes..(row + 1) * classes];
                    if values.iter().any(|v| !v.is_finite()) {
                        return Err("nonfinite repetition logits".into());
                    }
                    let compare = |&a: &usize, &b: &usize| {
                        values[b].partial_cmp(&values[a]).unwrap().then(a.cmp(&b))
                    };
                    ranks.clear();
                    ranks.extend(0..classes);
                    if ids.len() < classes {
                        ranks.select_nth_unstable_by(ids.len(), compare);
                    }
                    ranks.truncate(ids.len());
                    ranks.sort_unstable_by(compare);
                    for (id, &rank) in ids.iter_mut().zip(&ranks) {
                        *id = rank as u64;
                    }
                }
            }
        }
        let (positions, _) = select_repetition_unlikelihood_candidates(&self.request)?;
        let positions = positions
            .into_iter()
            .map(|p| {
                Ok(UnlikelihoodPosition {
                    row: index(p.sequence_index * steps + p.prediction_index)?,
                    candidates: p.candidates.iter().map(|c| c.token_id as usize).collect(),
                })
            })
            .collect::<CausalLmResult<Vec<_>>>()?;
        let (loss, gradient) = logits.unlikelihood_with_logits(&positions, EPSILON)?;
        Ok((loss, gradient, positions.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_repetition_ties_use_lowest_ids_and_periodic_rows_use_shifted_mask() {
        let recipe = |source| CausalLmRepetition::NativeV1 {
            config: Config {
                strength: 1.,
                candidate_source: source,
                context_window: 16,
                max_candidates_per_position: 4,
            },
        };
        let logits = Tensor::zeros(1, 4).unwrap();
        for (k, expected) in [(1, 0), (2, 1)] {
            let window = RepetitionWindow::prepare(
                recipe(Source::ModelTopkHistory { proposal_top_k: k }),
                &[1, 2],
                [1, 2],
                None,
                4,
            )
            .unwrap()
            .unwrap();
            let (loss, grad, positions) = window.loss_gradient(&logits, None, 1).unwrap();
            assert_eq!(positions, expected);
            if expected == 1 {
                assert!((loss - (4f32 / 3.).ln()).abs() < 1e-7);
                assert_eq!(grad.data()[1], 0.25);
            } else {
                assert_eq!(loss, 0.);
                assert_eq!(grad.data(), &[0.; 4]);
            }
        }
        let tokens = [1, 2, 1, 2, 1, 3];
        let labels = [-100, -100, -100, -100, -100, 3];
        let window = RepetitionWindow::prepare(
            recipe(Source::ModelTopkPeriodic { proposal_top_k: 4 }),
            &tokens,
            [1, 6],
            Some(&labels),
            4,
        )
        .unwrap()
        .unwrap();
        let (loss, grad, positions) = window.loss_gradient(&logits, Some(&[4]), 5).unwrap();
        assert_eq!(positions, 1);
        assert!((loss - (4f32 / 3.).ln()).abs() < 1e-7);
        assert_eq!(grad.data()[2], 0.25);
    }
}
