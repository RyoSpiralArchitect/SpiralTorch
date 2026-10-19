//! Causal label transport and admission; shared tensor kernels own CE and VJP.
use super::*;

/// Mean CE over selected shifted labels, with uniform label smoothing.
/// Unshifted labels use -100 to exclude loss terms, never attention positions.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CausalLmObjective {
    pub label_smoothing: f64,
}

impl CausalLmObjective {
    pub fn validate(self) -> CausalLmResult<()> {
        self.kernel_config().validate()?;
        Ok(())
    }

    pub(super) fn kernel_config(self) -> CrossEntropyConfig {
        CrossEntropyConfig {
            label_smoothing: self.label_smoothing,
            ..CrossEntropyConfig::default()
        }
    }
}

pub(super) struct PreparedCausalWindow {
    pub input: Vec<usize>,
    pub labels: Vec<i64>,
    pub shape: [usize; 2],
    pub window_shape: [usize; 2],
    pub target_tokens: usize,
    pub selected_rows: Option<Vec<usize>>,
    pub objective: CausalLmObjective,
}

impl NativeCausalLm {
    /// Bound all arrays before a binding copies tokens or labels.
    pub fn loss_input_count(&self, shape: [usize; 2]) -> CausalLmResult<usize> {
        self.loss_objective_input_count(shape, false)
    }

    /// With labels, admit context first; exact output budget follows bounded
    /// label extraction, before decoding. At least one logit row must fit.
    pub fn loss_objective_input_count(
        &self,
        shape: [usize; 2],
        has_labels: bool,
    ) -> CausalLmResult<usize> {
        let steps = shape[1]
            .checked_sub(1)
            .filter(|&n| n > 0)
            .ok_or("next-token loss needs at least two IDs per window")?;
        if has_labels {
            self.context_input_count([shape[0], steps])?;
            self.logit_count(1)?;
        } else {
            self.input_count([shape[0], steps], false)?;
        }
        product(&shape, usize::MAX, "window IDs")
    }

    pub(super) fn prepare_objective(
        &self,
        windows: &[usize],
        shape: [usize; 2],
        labels: Option<&[i64]>,
        objective: CausalLmObjective,
    ) -> CausalLmResult<PreparedCausalWindow> {
        objective.validate()?;
        let count = self.loss_objective_input_count(shape, labels.is_some())?;
        if windows.len() != count || windows.iter().any(|&id| id >= self.config.vocabulary) {
            return Err("invalid next-token window length or token ID".into());
        }
        if let Some(labels) = labels {
            if labels.len() != count
                || labels.iter().any(|&v| {
                    v != -100 && usize::try_from(v).map_or(true, |v| v >= self.config.vocabulary)
                })
            {
                return Err(
                    "causal labels must match the window and be vocabulary IDs or -100".into(),
                );
            }
        }
        let steps = shape[1] - 1;
        let mut input = Vec::with_capacity(shape[0] * steps);
        let mut shifted = Vec::with_capacity(shape[0] * steps);
        for (row, window) in windows.chunks_exact(shape[1]).enumerate() {
            input.extend_from_slice(&window[..steps]);
            if let Some(labels) = labels {
                let start = row * shape[1];
                shifted.extend_from_slice(&labels[start + 1..start + shape[1]]);
            } else {
                shifted.extend(window[1..].iter().map(|&v| v as i64));
            }
        }
        let target_tokens = shifted.iter().filter(|&&v| v != -100).count();
        if target_tokens == 0 {
            return Err("causal loss requires at least one non-ignored shifted target".into());
        }
        self.logit_count(target_tokens)?;
        let selected_rows = if target_tokens == shifted.len() {
            None
        } else {
            let rows = shifted
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| (v != -100).then_some(i))
                .collect();
            shifted.retain(|&v| v != -100);
            Some(rows)
        };
        Ok(PreparedCausalWindow {
            input,
            labels: shifted,
            shape: [shape[0], steps],
            window_shape: shape,
            target_tokens,
            selected_rows,
            objective,
        })
    }

    /// Labels are unshifted `[B,T]`, independent of input IDs; each row's first
    /// label is validated but not scored. No padding or attention mask inferred.
    pub fn next_token_loss_with_objective(
        &self,
        windows: &[usize],
        shape: [usize; 2],
        labels: Option<&[i64]>,
        objective: CausalLmObjective,
    ) -> CausalLmResult<f32> {
        let prepared = self.prepare_objective(windows, shape, labels, objective)?;
        let hidden = self.decode_hidden(&prepared.input, prepared.shape)?;
        let selected = match &prepared.selected_rows {
            Some(rows) => hidden.gather_rows(rows)?,
            None => hidden,
        };
        let normalized = self
            .final_norm
            .forward(&selected, self.config.norm_epsilon)?;
        Ok(self
            .project_readout(&normalized)?
            .cross_entropy_with_logits(&prepared.labels, objective.kernel_config())?
            .data()[0])
    }
}
