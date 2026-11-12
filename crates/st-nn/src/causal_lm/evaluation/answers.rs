//! Paired descriptive scores for contiguous runs of supervised targets.
use super::*;
use crate::causal_lm::sum::Sum;
use st_tensor::ClassScoreGroup;

/// Bound the opt-in answer-by-variant result independently of corpus size.
pub const MAX_ANSWER_SCORE_CELLS: usize = 65_536;

#[derive(Debug, Serialize)]
pub struct CausalLmAnswerScore {
    pub nll: f64,
    pub top1_correct: u64,
}

#[derive(Debug, Serialize)]
pub struct CausalLmScoredAnswer {
    pub document: usize,
    /// Half-open target ID positions in the original document, not in a window.
    pub start: usize,
    pub end: usize,
    pub scores: Vec<CausalLmAnswerScore>,
}

#[derive(Debug, Serialize)]
pub struct CausalLmAnswerComparison {
    schema: &'static str,
    comparison: CausalLmEvaluationComparison,
    document_count: usize,
    scored_documents: usize,
    answers: Vec<CausalLmScoredAnswer>,
}

#[derive(Debug, Default, Serialize)]
pub struct CausalLmAnswerContrast {
    pub reference: usize,
    pub candidate: usize,
    pub answer_count: usize,
    pub target_tokens: u64,
    pub improved: usize,
    pub tied: usize,
    pub regressed: usize,
    /// Candidate minus reference. Both use the f64 scored rows, not rounded batch means.
    pub token_weighted_delta_nll: f64,
    pub answer_weighted_delta_nll: f64,
    pub reference_top1_correct: u64,
    pub candidate_top1_correct: u64,
    pub reference_all_correct_answers: usize,
    pub candidate_all_correct_answers: usize,
}

impl CausalLmAnswerComparison {
    pub fn comparison(&self) -> &CausalLmEvaluationComparison {
        &self.comparison
    }
    pub fn answers(&self) -> &[CausalLmScoredAnswer] {
        &self.answers
    }
    pub fn document_count(&self) -> usize {
        self.document_count
    }
    pub fn scored_documents(&self) -> usize {
        self.scored_documents
    }

    pub fn contrast(
        &self,
        reference: usize,
        candidate: usize,
    ) -> CausalLmResult<CausalLmAnswerContrast> {
        let n = self.comparison.evaluations.len();
        if reference >= n || candidate >= n || reference == candidate {
            return Err("answer contrast requires two distinct in-range variant indices".into());
        }
        let mut result = CausalLmAnswerContrast {
            reference,
            candidate,
            answer_count: self.answers.len(),
            ..Default::default()
        };
        let mut tokens = Sum::default();
        let mut equal = Sum::default();
        for answer in &self.answers {
            let count = (answer.end - answer.start) as u64;
            let a = &answer.scores[reference];
            let b = &answer.scores[candidate];
            let delta = b.nll - a.nll;
            if delta < 0. {
                result.improved += 1;
            } else if delta > 0. {
                result.regressed += 1;
            } else {
                result.tied += 1;
            }
            tokens.add(delta * count as f64);
            equal.add(delta);
            result.target_tokens += count;
            result.reference_top1_correct += a.top1_correct;
            result.candidate_top1_correct += b.top1_correct;
            result.reference_all_correct_answers += usize::from(a.top1_correct == count);
            result.candidate_all_correct_answers += usize::from(b.top1_correct == count);
        }
        result.token_weighted_delta_nll = tokens.mean(result.target_tokens);
        result.answer_weighted_delta_nll = equal.mean(result.answer_count as u64);
        Ok(result)
    }
}

impl NativeCausalLm {
    /// Without explicit labels, every document with at least two IDs is one answer.
    /// With labels, exact run-count admission follows bounded value extraction.
    pub fn answer_comparison_plan(
        models: &[&Self],
        lengths: &[usize],
        config: CausalLmEvaluationConfig,
        has_labels: bool,
    ) -> CausalLmResult<CausalDocumentWindows> {
        Self::comparison_prefix_layers(models)?;
        if lengths.len() > MAX_DOCUMENTS {
            return Err("document count exceeds budget".into());
        }
        if !has_labels
            && lengths.iter().filter(|&&n| n >= 2).count() > MAX_ANSWER_SCORE_CELLS / models.len()
        {
            return Err("answer-by-variant score budget exceeded".into());
        }
        models[0].corpus_plan(models, lengths, config, has_labels)
    }

    /// Score each contiguous run of non-ignored within-document targets, even
    /// across batch/window boundaries. Retains the ordinary aggregate receipts.
    pub fn evaluate_variants_by_answer(
        models: &[&Self],
        documents: &[Vec<usize>],
        labels: Option<&[Vec<i64>]>,
        config: CausalLmEvaluationConfig,
    ) -> CausalLmResult<CausalLmAnswerComparison> {
        Self::evaluate_variants_by_answer_with_prefix_cap(
            models,
            documents,
            labels,
            config,
            usize::MAX,
        )
    }

    pub(crate) fn evaluate_variants_by_answer_with_prefix_cap(
        models: &[&Self],
        documents: &[Vec<usize>],
        labels: Option<&[Vec<i64>]>,
        config: CausalLmEvaluationConfig,
        prefix_cap: usize,
    ) -> CausalLmResult<CausalLmAnswerComparison> {
        let shared_prefix_layers = Self::comparison_prefix_layers(models)?.min(prefix_cap);
        let mut detail = AnswerCollector::default();
        let evaluations = models[0].evaluate_corpus(
            models,
            shared_prefix_layers,
            documents,
            labels,
            config,
            Some(&mut detail),
        )?;
        detail.finish(
            documents.len(),
            CausalLmEvaluationComparison::new(shared_prefix_layers, evaluations),
        )
    }
}

#[derive(Default)]
pub(super) struct AnswerCollector {
    spans: Vec<(usize, usize, usize)>,
    cells: Vec<ClassScoreGroup>,
    owners: usize,
    cursor: usize,
    consumed: usize,
    first: usize,
    lengths: Vec<usize>,
}

impl AnswerCollector {
    pub(super) fn prepare(
        &mut self,
        lengths: &[usize],
        labels: Option<&[Vec<i64>]>,
        owners: usize,
    ) -> CausalLmResult<()> {
        let mut push = |doc, start, end| -> CausalLmResult<()> {
            if self.spans.len() >= MAX_ANSWER_SCORE_CELLS / owners {
                return Err("answer-by-variant score budget exceeded".into());
            }
            self.spans.push((doc, start, end));
            Ok(())
        };
        for (doc, &n) in lengths.iter().enumerate() {
            if n < 2 {
                continue;
            }
            if let Some(labels) = labels {
                let row = &labels[doc];
                let mut i = 1;
                while i < n {
                    if row[i] == -100 {
                        i += 1;
                        continue;
                    }
                    let start = i;
                    while i < n && row[i] != -100 {
                        i += 1;
                    }
                    push(doc, start, i)?;
                }
            } else {
                push(doc, 1, n)?;
            }
        }
        self.owners = owners;
        self.cells
            .resize(self.spans.len() * owners, ClassScoreGroup::default());
        Ok(())
    }

    /// Covering windows and selected readout rows both preserve document order.
    /// Split this batch's selected rows at answer boundaries without allocating
    /// a document map or retaining logits for the rest of the corpus.
    pub(super) fn begin_batch(&mut self, mut targets: usize) -> CausalLmResult<()> {
        self.first = self.cursor;
        self.lengths.clear();
        while targets > 0 {
            let &(_, start, end) = self
                .spans
                .get(self.cursor)
                .ok_or("unexpected selected evaluation row")?;
            let length = end - start;
            let take = targets.min(length - self.consumed);
            self.lengths.push(take);
            self.consumed += take;
            targets -= take;
            if self.consumed == length {
                self.cursor += 1;
                self.consumed = 0;
            }
        }
        Ok(())
    }

    pub(super) fn score(
        &mut self,
        owner: usize,
        logits: Tensor,
        labels: &[i64],
    ) -> CausalLmResult<f32> {
        let (mean, groups) = logits.grouped_class_scores(labels, &self.lengths)?;
        for (i, g) in groups.into_iter().enumerate() {
            if g.active != self.lengths[i] {
                return Err("selected evaluation rows contain ignored labels".into());
            }
            let cell = &mut self.cells[(self.first + i) * self.owners + owner];
            cell.active += g.active;
            cell.nll_sum += g.nll_sum;
            cell.top1_correct += g.top1_correct;
        }
        Ok(mean)
    }

    fn finish(
        self,
        document_count: usize,
        comparison: CausalLmEvaluationComparison,
    ) -> CausalLmResult<CausalLmAnswerComparison> {
        if self.cursor != self.spans.len() || self.consumed != 0 || self.spans.is_empty() {
            return Err("incomplete evaluation answer coverage".into());
        }
        let mut answers = Vec::with_capacity(self.spans.len());
        let mut scored_documents = 0;
        let mut previous = None;
        for ((document, start, end), cells) in self
            .spans
            .into_iter()
            .zip(self.cells.chunks_exact(self.owners))
        {
            let mut scores = Vec::with_capacity(self.owners);
            for c in cells {
                if c.active != end - start || c.top1_correct > c.active {
                    return Err("incomplete evaluation answer scores".into());
                }
                scores.push(CausalLmAnswerScore {
                    nll: c.nll_sum / c.active as f64,
                    top1_correct: c.top1_correct as u64,
                });
            }
            if previous.replace(document) != Some(document) {
                scored_documents += 1;
            }
            answers.push(CausalLmScoredAnswer {
                document,
                start,
                end,
                scores,
            });
        }
        Ok(CausalLmAnswerComparison {
            schema: "spiraltorch.native_causal_lm_answer_comparison.v1",
            comparison,
            document_count,
            scored_documents,
            answers,
        })
    }
}
