//! Teacher-forced QKV learning through the existing causal objective and decoder VJP.
use super::*;

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmQkvWindowStep {
    pub steps: u32,
    pub window_shape: [usize; 2],
    pub target_tokens: usize,
    pub objective_loss: f32,
    pub label_smoothing: f64,
    /// Raw Q/K/V derivative norms, before optional global clipping.
    pub gradient_l2: [f64; 3],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gradient_clip: Option<CausalLmQkvGradientClip>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmQkvRivalWindowStep {
    pub steps: u32,
    pub window_shape: [usize; 2],
    pub competition: RivalCrossEntropyReport,
    /// Raw Q/K/V derivative norms, before optional global clipping.
    pub gradient_l2: [f64; 3],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gradient_clip: Option<CausalLmQkvGradientClip>,
}

impl NativeCausalLmQkvAdamW {
    /// Admit context/checkpoints before copying foreign arrays. With sparse labels,
    /// the exact selected-logit bound is checked by prepare_objective before decoding.
    pub fn loss_input_count(&self, shape: [usize; 2], has_labels: bool) -> CausalLmResult<usize> {
        let count = self.model.loss_objective_input_count(shape, has_labels)?;
        let rows = product(&[shape[0], shape[1] - 1], usize::MAX, "QKV causal context")?;
        self.admit_gradient_rows(rows)?;
        Ok(count)
    }

    /// Admit the rival recipe before a foreign client copies window IDs.
    pub fn rival_loss_input_count(
        &self,
        shape: [usize; 2],
        has_labels: bool,
        recipe: RivalCrossEntropy,
    ) -> CausalLmResult<usize> {
        recipe.validate(self.model.config.vocabulary)?;
        self.loss_input_count(shape, has_labels)
    }

    /// Mean CE and QKV derivative over selected shifted target tokens, not sequences.
    /// Labels have unshifted `[B,T]` layout; -100 masks loss, never attention/context.
    /// No padding, EOS, truncation or repetition objective is inferred.
    pub fn causal_loss_gradients(
        &self,
        windows: &[usize],
        shape: [usize; 2],
        labels: Option<&[i64]>,
        label_smoothing: f64,
    ) -> CausalLmResult<(f32, Tensor, usize)> {
        let (loss, gradient, target_tokens, _) =
            self.causal_gradients(windows, shape, labels, label_smoothing, None)?;
        Ok((loss, gradient, target_tokens))
    }

    /// Top-rival CE is mixed with ordinary CE over the selected shifted targets.
    /// Only QKV receives the derivative; attached adapters stay frozen.
    pub fn rival_loss_gradients(
        &self,
        windows: &[usize],
        shape: [usize; 2],
        labels: Option<&[i64]>,
        recipe: RivalCrossEntropy,
    ) -> CausalLmResult<(RivalCrossEntropyReport, Tensor)> {
        let (_, gradient, _, report) =
            self.causal_gradients(windows, shape, labels, 0., Some(recipe))?;
        Ok((report.ok_or("missing QKV rival report")?, gradient))
    }

    fn causal_gradients(
        &self,
        windows: &[usize],
        shape: [usize; 2],
        labels: Option<&[i64]>,
        label_smoothing: f64,
        recipe: Option<RivalCrossEntropy>,
    ) -> CausalLmResult<(f32, Tensor, usize, Option<RivalCrossEntropyReport>)> {
        let objective = CausalLmObjective {
            label_smoothing,
            ..Default::default()
        };
        objective.validate()?;
        match recipe {
            Some(recipe) => {
                self.rival_loss_input_count(shape, labels.is_some(), recipe)?;
            }
            None => {
                self.loss_input_count(shape, labels.is_some())?;
            }
        }
        let m = &self.model;
        let mut prepared = m.prepare_objective(windows, shape, labels, objective)?;
        prepared.competition = recipe.map(Into::into);
        let readout = m.training_readout(
            &prepared,
            m.embed(&prepared.input, prepared.shape)?,
            0,
            self.layer,
            None,
        )?;
        let competition = match readout.competition {
            Some(objective::CompetitionReport::Top(report)) => Some(report),
            None => None,
            _ => return Err("unexpected QKV competition report".into()),
        };
        drop(readout.logits);
        let mut gradient = None;
        m.decoder_parameter_pullback(
            readout.checkpoints,
            readout.hidden_rows,
            readout.selected,
            prepared.shape,
            prepared.selected_rows.as_deref(),
            readout.d_normalized,
            self.layer,
            learning::DecoderTarget::QkvWeight(&mut gradient),
            None,
        )?;
        let gradient = gradient.ok_or("unreached QKV causal derivative")?;
        self.validate_derivative(readout.loss, &gradient)?;
        Ok((readout.loss, gradient, prepared.target_tokens, competition))
    }

    /// One atomic optimizer update over the selected-token mean of this window batch.
    pub fn step_windows(
        &mut self,
        windows: &[usize],
        shape: [usize; 2],
        labels: Option<&[i64]>,
        label_smoothing: f64,
        step: AdamWStep,
    ) -> CausalLmResult<CausalLmQkvWindowStep> {
        let (objective_loss, gradient, target_tokens) =
            self.causal_loss_gradients(windows, shape, labels, label_smoothing)?;
        let (steps, gradient_l2, gradient_clip) = self.apply_gradient(gradient, step)?;
        Ok(CausalLmQkvWindowStep {
            steps,
            window_shape: shape,
            target_tokens,
            objective_loss,
            label_smoothing,
            gradient_l2,
            gradient_clip,
        })
    }

    /// One atomic QKV update using the existing Rust top-rival objective.
    pub fn step_rival_windows(
        &mut self,
        windows: &[usize],
        shape: [usize; 2],
        labels: Option<&[i64]>,
        recipe: RivalCrossEntropy,
        step: AdamWStep,
    ) -> CausalLmResult<CausalLmQkvRivalWindowStep> {
        let (competition, gradient) = self.rival_loss_gradients(windows, shape, labels, recipe)?;
        let (steps, gradient_l2, gradient_clip) = self.apply_gradient(gradient, step)?;
        Ok(CausalLmQkvRivalWindowStep {
            steps,
            window_shape: shape,
            competition,
            gradient_l2,
            gradient_clip,
        })
    }
}
