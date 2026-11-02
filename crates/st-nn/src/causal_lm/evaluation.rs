//! Read-only corpus NLL. Rust owns coverage, batching and target-weighted reduction.
use super::*;
use crate::language_windows::{CausalDocumentWindow, CausalDocumentWindows, MAX_DOCUMENTS};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::ops::Range;

pub const MAX_EVALUATION_TOKENS: usize = 16_777_216;
pub const MAX_EVALUATION_VARIANTS: usize = 16;

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_artifact_id: Option<String>,
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

/// Ordered independent scores. The count is frozen decoder layers executed once
/// per active microbatch, not a timing measurement or a persistent activation cache.
#[derive(Debug, Serialize)]
pub struct CausalLmEvaluationComparison {
    pub schema: &'static str,
    pub shared_prefix_layers: usize,
    pub evaluations: Vec<CausalLmEvaluation>,
}

fn groups(
    windows: &[CausalDocumentWindow],
    maximum: usize,
) -> impl Iterator<Item = Range<usize>> + '_ {
    let mut start = 0;
    std::iter::from_fn(move || {
        let width = windows.get(start)?.length;
        let count = windows[start..]
            .iter()
            .take(maximum)
            .take_while(|w| w.length == width)
            .count();
        let range = start..start + count;
        start += count;
        Some(range)
    })
}

fn unique_shapes(
    windows: &[CausalDocumentWindow],
    maximum: usize,
) -> impl Iterator<Item = [usize; 2]> + '_ {
    let mut seen = HashSet::new();
    let mut previous = None;
    groups(windows, maximum).filter_map(move |g| {
        let shape = [g.len(), windows[g.start].length as usize];
        if previous.replace(shape) == Some(shape) {
            return None;
        }
        seen.insert(shape).then_some(shape)
    })
}

impl NativeCausalLm {
    /// Per-corpus input allowance, accounting for all owners in a comparison.
    pub fn evaluation_token_limit(owners: usize) -> CausalLmResult<usize> {
        if !(1..=MAX_EVALUATION_VARIANTS).contains(&owners) {
            return Err("invalid evaluation owner count".into());
        }
        Ok(MAX_EVALUATION_TOKENS / owners)
    }

    /// Bounds before foreign token extraction. Explicit labels defer only the
    /// exact selected-logit count, never the context/geometry work budgets.
    pub fn evaluation_plan(
        &self,
        lengths: &[usize],
        config: CausalLmEvaluationConfig,
        has_labels: bool,
    ) -> CausalLmResult<CausalDocumentWindows> {
        self.corpus_plan(&[self], lengths, config, has_labels)
    }

    fn corpus_plan(
        &self,
        models: &[&Self],
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
        let token_limit = Self::evaluation_token_limit(models.len())?;
        lengths
            .iter()
            .try_fold(0usize, |sum, &n| sum.checked_add(n))
            .filter(|&n| n <= token_limit)
            .ok_or("evaluation token count exceeds budget")?;
        let plan = CausalDocumentWindows::covering_targets(
            lengths.iter().map(|&n| n as u32).collect(),
            config.context_length,
        )?;
        if plan.windows().is_empty() {
            return Err("evaluation requires a within-document next-token target".into());
        }
        for shape in unique_shapes(plan.windows(), config.microbatch_windows) {
            for model in models {
                model.loss_objective_input_count(shape, has_labels)?;
            }
        }
        Ok(plan)
    }

    /// Admit a roster before consuming a foreign or streamed corpus.
    pub fn comparison_prefix_layers(models: &[&Self]) -> CausalLmResult<usize> {
        if !(2..=MAX_EVALUATION_VARIANTS).contains(&models.len()) {
            return Err("native comparison requires 2 to 16 variants".into());
        }
        let first = models[0];
        let mut depth = first.config.layers;
        for model in models {
            if model.base_id != first.base_id
                || model.readout_accumulator != first.readout_accumulator
            {
                return Err("comparison requires the same frozen base and readout policy".into());
            }
            if model.geometry.is_some() {
                depth = depth.min(model.first_geometry_site()?.0);
            }
            if let Some(adapter) = &model.lora {
                depth = depth.min(adapter.layer());
            }
        }
        Ok(depth)
    }

    /// Foreign-input admission checks *all* variants before extracting any tokens.
    pub fn comparison_plan(
        models: &[&Self],
        lengths: &[usize],
        config: CausalLmEvaluationConfig,
        has_labels: bool,
    ) -> CausalLmResult<CausalDocumentWindows> {
        Self::comparison_prefix_layers(models)?;
        models[0].corpus_plan(models, lengths, config, has_labels)
    }

    /// Score immutable snapshots on identical windows. Only blocks strictly before
    /// every geometry and LoRA site can be shared. Each variant's suffix, masking,
    /// final normalization and readout are its own; no tensors survive a microbatch.
    pub fn evaluate_variants(
        models: &[&Self],
        documents: &[Vec<usize>],
        labels: Option<&[Vec<i64>]>,
        config: CausalLmEvaluationConfig,
    ) -> CausalLmResult<CausalLmEvaluationComparison> {
        let shared_prefix_layers = Self::comparison_prefix_layers(models)?;
        let evaluations =
            models[0].evaluate_corpus(models, shared_prefix_layers, documents, labels, config)?;
        Ok(CausalLmEvaluationComparison {
            schema: "spiraltorch.native_causal_lm_evaluation_comparison.v1",
            shared_prefix_layers,
            evaluations,
        })
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
        self.evaluate_corpus(&[self], 0, documents, labels, config)?
            .pop()
            .ok_or_else(|| "missing evaluation result".into())
    }

    fn evaluate_corpus(
        &self,
        models: &[&Self],
        shared: usize,
        documents: &[Vec<usize>],
        labels: Option<&[Vec<i64>]>,
        config: CausalLmEvaluationConfig,
    ) -> CausalLmResult<Vec<CausalLmEvaluation>> {
        if documents.len() > MAX_DOCUMENTS {
            return Err("document count exceeds budget".into());
        }
        let lengths: Vec<_> = documents.iter().map(Vec::len).collect();
        let plan = self.corpus_plan(models, &lengths, config, labels.is_some())?;
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
        // A fixed owner's readout size is monotonic in selected rows, including masked tails.
        let maximum_active = groups(windows, config.microbatch_windows)
            .map(|g| selected[g].iter().sum::<usize>())
            .max()
            .unwrap_or(0);
        for model in models {
            model.logit_count(maximum_active)?;
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
        let mut totals = vec![0.; models.len()];
        let mut microbatches = 0;
        for group in groups(windows, config.microbatch_windows) {
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
            let prepared = self.prepare_objective(
                &ids,
                [group.len(), width],
                batch_labels.as_deref(),
                CausalLmObjective::default(),
            )?;
            let prefix = self.forward_blocks(
                self.embed(&prepared.input, prepared.shape)?,
                prepared.shape,
                0..shared,
            )?;
            // Transfer ownership to the last suffix, including the ordinary single-model path.
            for ((model, total), start) in models
                .iter()
                .zip(&mut totals)
                .zip(std::iter::repeat_n(prefix, models.len()))
            {
                let hidden =
                    model.forward_blocks(start, prepared.shape, shared..model.config.layers)?;
                let (_, loss) = model.causal_readout(hidden, &prepared)?;
                if !loss.is_finite() {
                    return Err("evaluation produced non-finite NLL".into());
                }
                *total += f64::from(loss) * targets as f64;
            }
            microbatches += 1;
        }
        let data_id = format!("sha256:{:x}", hash.finalize());
        let skipped_windows = selected.iter().filter(|&&n| n == 0).count();
        let ignored_targets =
            windows.iter().map(|w| u64::from(w.length - 1)).sum::<u64>() - target_tokens;
        Ok(models
            .iter()
            .zip(totals)
            .map(|(model, total_nll)| {
                let nll = total_nll / target_tokens as f64;
                let perplexity = nll.exp();
                CausalLmEvaluation {
                    schema: "spiraltorch.native_causal_lm_evaluation.v1",
                    base_model_identity: self.base_id.clone(),
                    geometry_artifact_id: model.geometry_artifact_id().map(str::to_owned),
                    lora_artifact_id: model.lora_artifact_id().map(str::to_owned),
                    readout_accumulator: self.readout_accumulator(),
                    data_id: data_id.clone(),
                    plan_id: plan_id.clone(),
                    config,
                    window_count: windows.len(),
                    skipped_windows,
                    microbatches,
                    target_tokens,
                    ignored_targets,
                    nll,
                    perplexity: perplexity.is_finite().then_some(perplexity),
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod shape_tests {
    use super::*;

    #[test]
    fn admission_checks_distinct_shapes_in_first_seen_order() {
        let mut lengths = vec![4; 1025];
        lengths.push(3);
        lengths.extend([4; 8]);
        let plan = CausalDocumentWindows::covering_targets(lengths, 4).unwrap();
        assert_eq!(
            unique_shapes(plan.windows(), 8).collect::<Vec<_>>(),
            [[8, 4], [1, 4], [1, 3]]
        );
    }
}
