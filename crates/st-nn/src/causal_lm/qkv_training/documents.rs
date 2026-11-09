//! One selected-token mean and atomic update over ordered, unpadded documents.
use super::*;
use crate::language_windows::{equal_width_groups, CausalDocumentWindow};
use std::ops::Range;

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmQkvDocumentStep {
    pub steps: u32,
    pub documents: usize,
    pub microbatches: usize,
    pub target_tokens: usize,
    pub objective_loss: f32,
    pub label_smoothing: f64,
    /// Raw Q/K/V derivative norms of the whole logical mean, before clipping.
    pub gradient_l2: [f64; 3],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gradient_clip: Option<CausalLmQkvGradientClip>,
}

impl NativeCausalLmQkvAdamW {
    pub const MAX_CAUSAL_DOCUMENTS: usize = 4096;

    /// Structural admission before clients copy IDs or labels. Exact sparse
    /// output bounds follow label validation, before the first forward pass.
    pub fn document_input_count(
        &self,
        lengths: &[usize],
        max_rows: usize,
        has_labels: bool,
    ) -> CausalLmResult<usize> {
        self.causal_document_groups(lengths, max_rows, has_labels)
            .map(|(_, count)| count)
    }

    fn causal_document_groups(
        &self,
        lengths: &[usize],
        max_rows: usize,
        has_labels: bool,
    ) -> CausalLmResult<(Vec<Range<usize>>, usize)> {
        if lengths.is_empty() || lengths.len() > Self::MAX_CAUSAL_DOCUMENTS || max_rows == 0 {
            return Err("invalid QKV causal document roster or microbatch limit".into());
        }
        let mut total = 0usize;
        let mut contexts = 0usize;
        let mut windows = Vec::with_capacity(lengths.len());
        for (index, &length) in lengths.iter().enumerate() {
            let steps = length
                .checked_sub(1)
                .filter(|&n| n > 0)
                .ok_or("QKV causal document needs at least two IDs")?;
            if steps > self.model.config.positions {
                return Err("QKV causal document exceeds position capacity".into());
            }
            total = total
                .checked_add(length)
                .ok_or("QKV document token count overflow")?;
            contexts = contexts
                .checked_add(steps)
                .ok_or("QKV document context overflow")?;
            windows.push(CausalDocumentWindow {
                document: index as u32,
                start: 0,
                length: u32::try_from(length)?,
            });
        }
        product(
            &[
                contexts,
                3,
                self.model.config.width,
                self.model.config.width,
            ],
            1 << 30,
            "logical QKV causal gradient work",
        )?;
        let groups: Vec<_> = equal_width_groups(&windows, max_rows).collect();
        if groups.len() > 1024 {
            return Err("QKV causal microbatch count exceeds limit".into());
        }
        for group in &groups {
            self.loss_input_count([group.len(), lengths[group.start]], has_labels)?;
        }
        Ok((groups, total))
    }

    /// Selected-token weighted CE and derivative over adjacent equal-length groups.
    /// Every document must have a shifted target; -100 masks loss, not context.
    /// No padding, reordering, concatenation across documents or truncation occurs.
    /// Preserve max_rows and order for repeatable floating-point accumulation.
    pub fn document_loss_gradients(
        &self,
        documents: &[&[usize]],
        labels: Option<&[&[i64]]>,
        max_rows: usize,
        label_smoothing: f64,
    ) -> CausalLmResult<(f32, Tensor, usize)> {
        let (loss, gradient, targets, _) =
            self.causal_document_gradient(documents, labels, max_rows, label_smoothing)?;
        Ok((loss, gradient, targets))
    }

    fn causal_document_gradient(
        &self,
        documents: &[&[usize]],
        labels: Option<&[&[i64]]>,
        max_rows: usize,
        label_smoothing: f64,
    ) -> CausalLmResult<(f32, Tensor, usize, usize)> {
        let objective = CausalLmObjective {
            label_smoothing,
            ..Default::default()
        };
        objective.validate()?;
        if documents.is_empty()
            || documents.len() > Self::MAX_CAUSAL_DOCUMENTS
            || labels.is_some_and(|rows| rows.len() != documents.len())
        {
            return Err("invalid QKV causal document or label roster".into());
        }
        let lengths = documents.iter().map(|row| row.len()).collect::<Vec<_>>();
        let (groups, _) = self.causal_document_groups(&lengths, max_rows, labels.is_some())?;
        // Validate every ID/label and each group's actual sparse logit demand
        // before decoding any group. Temporary prepared arrays are not retained.
        let mut targets = Vec::with_capacity(documents.len());
        for (index, ids) in documents.iter().enumerate() {
            targets.push(
                self.model
                    .prepare_objective(
                        ids,
                        [1, ids.len()],
                        labels.map(|rows| rows[index]),
                        objective,
                    )?
                    .target_tokens,
            );
        }
        for group in &groups {
            self.model
                .logit_count(targets[group.clone()].iter().sum())?;
        }
        let total_targets = targets.iter().sum();
        let microbatches = groups.len();
        let mut mean = batching::WeightedGradientMean::new(self.model.config.width);
        for group in groups {
            let ids = documents[group.clone()].concat();
            let batch_labels = labels.map(|rows| rows[group.clone()].concat());
            let (loss, gradient, count) = self.causal_loss_gradients(
                &ids,
                [group.len(), lengths[group.start]],
                batch_labels.as_deref(),
                label_smoothing,
            )?;
            mean.add(loss, &gradient, count);
        }
        let (loss, gradient) = mean.finish()?;
        self.validate_derivative(loss, &gradient)?;
        Ok((loss, gradient, total_targets, microbatches))
    }

    /// Clip once and commit once, only after all document groups succeed.
    pub fn step_documents(
        &mut self,
        documents: &[&[usize]],
        labels: Option<&[&[i64]]>,
        max_rows: usize,
        label_smoothing: f64,
        step: AdamWStep,
    ) -> CausalLmResult<CausalLmQkvDocumentStep> {
        let (objective_loss, gradient, target_tokens, microbatches) =
            self.causal_document_gradient(documents, labels, max_rows, label_smoothing)?;
        let (steps, gradient_l2, gradient_clip) = self.apply_gradient(gradient, step)?;
        Ok(CausalLmQkvDocumentStep {
            steps,
            documents: documents.len(),
            microbatches,
            target_tokens,
            objective_loss,
            label_smoothing,
            gradient_l2,
            gradient_clip,
        })
    }
}
