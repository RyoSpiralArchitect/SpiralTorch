//! Read-only corpus NLL. Rust owns coverage, batching and target-weighted reduction.
use super::*;
use crate::language_windows::{CausalDocumentWindow, CausalDocumentWindows, MAX_DOCUMENTS};
use sha2::{Digest, Sha256};
use std::ops::Range;

pub const MAX_EVALUATION_TOKENS: usize = 16_777_216;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CausalLmEvaluationConfig {
    /// Includes the final target ID; the decoder sees at most context_length - 1.
    pub context_length: u32,
    pub microbatch_windows: usize,
}

impl CausalLmEvaluationConfig {
    pub fn validate(self) -> CausalLmResult<()> {
        if self.context_length < 2
            || self.microbatch_windows == 0
            || self.microbatch_windows > 1_048_576
        {
            return Err("invalid native evaluation context or batch count".into());
        }
        Ok(())
    }
}

/// Mean unsmoothed NLL, not the training objective or an average of batch means.
/// Perplexity is absent only when exp(nll) would overflow f64.
#[derive(Clone, Debug, Serialize)]
pub struct CausalLmEvaluation {
    pub schema: &'static str,
    pub base_model_identity: String,
    pub geometry_artifact_id: Option<String>,
    pub readout_accumulator: CausalLmReadoutAccumulator,
    pub data_id: String,
    pub plan_id: String,
    pub config: CausalLmEvaluationConfig,
    pub window_count: usize,
    pub skipped_windows: usize,
    pub microbatches: usize,
    pub target_tokens: u64,
    pub ignored_targets: u64,
    pub nll: f64,
    pub perplexity: Option<f64>,
}

fn groups(windows: &[CausalDocumentWindow], maximum: usize) -> Vec<Range<usize>> {
    let mut result = Vec::new();
    let mut start = 0;
    while start < windows.len() {
        let width = windows[start].length;
        let count = windows[start..]
            .iter()
            .take(maximum)
            .take_while(|w| w.length == width)
            .count();
        result.push(start..start + count);
        start += count;
    }
    result
}

impl NativeCausalLm {
    /// Bounds before foreign token extraction. Explicit labels defer only the
    /// exact selected-logit count, never the context/geometry work budgets.
    pub fn evaluation_plan(
        &self,
        lengths: &[usize],
        config: CausalLmEvaluationConfig,
        has_labels: bool,
    ) -> CausalLmResult<CausalDocumentWindows> {
        config.validate()?;
        if config.context_length as usize - 1 > self.config.positions {
            return Err("evaluation context exceeds model position capacity".into());
        }
        if lengths.len() > MAX_DOCUMENTS {
            return Err("document count exceeds budget".into());
        }
        lengths
            .iter()
            .try_fold(0usize, |sum, &n| sum.checked_add(n))
            .filter(|&n| n <= MAX_EVALUATION_TOKENS)
            .ok_or("evaluation token count exceeds budget")?;
        let plan = CausalDocumentWindows::covering_targets(
            lengths.iter().map(|&n| n as u32).collect(),
            config.context_length,
        )?;
        if plan.windows().is_empty() {
            return Err("evaluation requires a within-document next-token target".into());
        }
        for group in groups(plan.windows(), config.microbatch_windows) {
            self.loss_objective_input_count(
                [group.len(), plan.windows()[group.start].length as usize],
                has_labels,
            )?;
        }
        Ok(plan)
    }

    /// Score each document's positions 1..len once, including short tails.
    /// Context resets per window and never crosses a document boundary. Labels
    /// are unshifted, -100 masks loss only, and zero-target windows are counted
    /// but not scored. An entirely masked corpus rejects before decoding.
    pub fn evaluate_documents(
        &self,
        documents: &[Vec<usize>],
        labels: Option<&[Vec<i64>]>,
        config: CausalLmEvaluationConfig,
    ) -> CausalLmResult<CausalLmEvaluation> {
        let lengths: Vec<_> = documents.iter().map(Vec::len).collect();
        let plan = self.evaluation_plan(&lengths, config, labels.is_some())?;
        if documents
            .iter()
            .flatten()
            .any(|&id| id >= self.config.vocabulary)
        {
            return Err("invalid evaluation document token ID".into());
        }
        if let Some(labels) = labels {
            if labels.len() != documents.len()
                || labels.iter().zip(&lengths).any(|(row, &n)| row.len() != n)
                || labels
                    .iter()
                    .flatten()
                    .any(|&v| v != -100 && (v < 0 || v as u64 >= self.config.vocabulary as u64))
            {
                return Err("invalid evaluation document labels".into());
            }
        }
        let windows = plan.windows();
        let selected: Vec<usize> = windows
            .iter()
            .map(|w| {
                labels.map_or(w.length as usize - 1, |labels| {
                    labels[w.document as usize][w.start as usize + 1..(w.start + w.length) as usize]
                        .iter()
                        .filter(|&&v| v != -100)
                        .count()
                })
            })
            .collect();
        let target_tokens = selected.iter().map(|&n| n as u64).sum::<u64>();
        if target_tokens == 0 {
            return Err("evaluation has no selected targets".into());
        }
        let groups = groups(windows, config.microbatch_windows);
        // Admit every selected readout before evaluating the first batch.
        for group in &groups {
            let active = selected[group.clone()].iter().sum::<usize>();
            if active > 0 {
                self.logit_count(active)?;
            }
        }
        let plan_id = plan.plan_id()?;
        let mut hash = Sha256::new();
        hash.update(b"spiraltorch.native_causal_evaluation_data.v1\0");
        hash.update(plan_id.as_bytes());
        hash.update([u8::from(labels.is_some())]);
        for (index, document) in documents.iter().enumerate() {
            hash.update((document.len() as u64).to_le_bytes());
            for &id in document {
                hash.update((id as u64).to_le_bytes());
            }
            if let Some(labels) = labels {
                for &label in &labels[index] {
                    hash.update(label.to_le_bytes());
                }
            }
        }
        let mut total_nll = 0.;
        let mut microbatches = 0;
        for group in groups {
            let targets = selected[group.clone()].iter().sum::<usize>();
            if targets == 0 {
                continue;
            }
            let width = windows[group.start].length as usize;
            let mut ids = Vec::with_capacity(group.len() * width);
            let mut batch_labels = labels.map(|_| Vec::with_capacity(group.len() * width));
            for w in &windows[group.clone()] {
                let span = w.start as usize..(w.start + w.length) as usize;
                ids.extend_from_slice(&documents[w.document as usize][span.clone()]);
                if let Some(targets) = &mut batch_labels {
                    targets.extend_from_slice(&labels.unwrap()[w.document as usize][span]);
                }
            }
            let loss = self.next_token_loss_with_objective(
                &ids,
                [group.len(), width],
                batch_labels.as_deref(),
                CausalLmObjective::default(),
            )?;
            if !loss.is_finite() {
                return Err("evaluation produced non-finite NLL".into());
            }
            total_nll += f64::from(loss) * targets as f64;
            microbatches += 1;
        }
        let nll = total_nll / target_tokens as f64;
        let perplexity = nll.exp();
        Ok(CausalLmEvaluation {
            schema: "spiraltorch.native_causal_lm_evaluation.v1",
            base_model_identity: self.base_id.clone(),
            geometry_artifact_id: self.geometry_artifact_id().map(str::to_owned),
            readout_accumulator: self.readout_accumulator(),
            data_id: format!("sha256:{:x}", hash.finalize()),
            plan_id,
            config,
            window_count: windows.len(),
            skipped_windows: selected.iter().filter(|&&n| n == 0).count(),
            microbatches,
            target_tokens,
            ignored_targets: windows.iter().map(|w| u64::from(w.length - 1)).sum::<u64>()
                - target_tokens,
            nll,
            perplexity: perplexity.is_finite().then_some(perplexity),
        })
    }
}
