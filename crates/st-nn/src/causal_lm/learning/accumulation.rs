//! Token-weighted microbatch reduction. No tapes, logits or token windows retained.
use super::*;
use crate::geometry_runtime::{AdamWStep, NativeGeometryAdamW};

const MAX_TARGET_TOKENS: u64 = 1 << 53;
type Means = BTreeMap<String, BTreeMap<String, Vec<f64>>>;

/// A bounded, sealed gradient mean for one frozen model/geometry snapshot.
/// Each microbatch contributes its non-ignored shifted targets, not one loss vote.
#[derive(Debug)]
pub struct NativeCausalLmAccumulator {
    base_id: String,
    readout_accumulator: CausalLmReadoutAccumulator,
    geometry_id: String,
    objective: CausalLmObjective,
    means: Means,
    loss: f64,
    target_tokens: u64,
    microbatches: u64,
    repetition_means: Option<Means>,
    repetition_loss: f64,
    repetition_positions: u64,
}

impl NativeCausalLm {
    pub fn geometry_accumulator(&self) -> CausalLmResult<NativeCausalLmAccumulator> {
        self.geometry_accumulator_with_objective(CausalLmObjective::default())
    }

    pub fn geometry_accumulator_with_objective(
        &self,
        objective: CausalLmObjective,
    ) -> CausalLmResult<NativeCausalLmAccumulator> {
        self.validate_objective_shape([1, 2], objective)?;
        self.geometry_gradient_input_count([1, 2])?;
        let artifact = self.geometry_artifact().unwrap();
        let means: Means = artifact
            .record()
            .placements
            .iter()
            .map(|p| {
                let parameters = p
                    .tensors
                    .iter()
                    .filter(|t| t.role == GeometryAdapterTensorRole::Parameter)
                    .map(|t| (t.name.clone(), vec![0.; t.values.len()]))
                    .collect();
                (p.path.clone(), parameters)
            })
            .collect();
        Ok(NativeCausalLmAccumulator {
            base_id: self.base_id.clone(),
            readout_accumulator: self.readout_accumulator,
            geometry_id: artifact.artifact_id().into(),
            objective,
            repetition_means: objective
                .repetition
                .filter(|r| r.config().strength > 0.)
                .map(|_| means.clone()),
            means,
            loss: 0.,
            target_tokens: 0,
            microbatches: 0,
            repetition_loss: 0.,
            repetition_positions: 0,
        })
    }
}

impl NativeCausalLmAccumulator {
    pub fn base_model_identity(&self) -> &str {
        &self.base_id
    }
    pub fn readout_accumulator(&self) -> CausalLmReadoutAccumulator {
        self.readout_accumulator
    }
    pub fn geometry_artifact_id(&self) -> &str {
        &self.geometry_id
    }
    pub fn target_tokens(&self) -> u64 {
        self.target_tokens
    }
    pub fn microbatches(&self) -> u64 {
        self.microbatches
    }
    pub fn objective(&self) -> CausalLmObjective {
        self.objective
    }
    pub fn loss(&self) -> Option<f64> {
        (self.target_tokens > 0).then_some(
            self.loss
                + self
                    .objective
                    .repetition
                    .map_or(0., |r| r.config().strength * self.repetition_loss),
        )
    }
    pub fn causal_loss(&self) -> Option<f64> {
        (self.target_tokens > 0).then_some(self.loss)
    }
    pub fn repetition_loss(&self) -> Option<f64> {
        (self.target_tokens > 0).then_some(self.repetition_loss)
    }
    pub fn repetition_positions(&self) -> u64 {
        self.repetition_positions
    }

    /// Identity check can run before a binding copies tokens from foreign memory.
    pub fn validate_model(&self, model: &NativeCausalLm) -> CausalLmResult<()> {
        if model.base_model_identity() != self.base_id
            || model.readout_accumulator != self.readout_accumulator
            || model.geometry_artifact_id() != Some(self.geometry_id.as_str())
        {
            return Err("LM accumulation belongs to another model or geometry snapshot".into());
        }
        Ok(())
    }

    fn next_counts(&self, shape: [usize; 2], active: usize) -> CausalLmResult<(u64, u64, u64)> {
        let capacity = shape[1]
            .checked_sub(1)
            .and_then(|steps| shape[0].checked_mul(steps))
            .filter(|&n| n > 0)
            .ok_or("LM accumulation requires nonempty shifted targets")?;
        if active == 0 || active > capacity {
            return Err("invalid non-ignored LM target count".into());
        }
        let targets = u64::try_from(active)?;
        let total = self
            .target_tokens
            .checked_add(targets)
            .filter(|&n| n <= MAX_TARGET_TOKENS)
            .ok_or("LM accumulation exceeds exact target-count budget")?;
        let microbatches = self
            .microbatches
            .checked_add(1)
            .ok_or("LM accumulation microbatch count exhausted")?;
        Ok((targets, total, microbatches))
    }

    /// Check the cumulative counter and per-window budgets before foreign ID extraction.
    pub fn input_count(&self, model: &NativeCausalLm, shape: [usize; 2]) -> CausalLmResult<usize> {
        self.objective_input_count(model, shape, false)
    }

    pub fn objective_input_count(
        &self,
        model: &NativeCausalLm,
        shape: [usize; 2],
        has_labels: bool,
    ) -> CausalLmResult<usize> {
        self.validate_model(model)?;
        // Actual selected-target count is known after bounded label extraction.
        self.next_counts(shape, 1)?;
        if let Some(recipe) = self.objective.repetition {
            recipe.admit_shape(shape, model.config.vocabulary)?;
        }
        model.geometry_objective_input_count(shape, has_labels)
    }

    /// Compute and reduce one window without exporting parameter lists to Python.
    /// Failure leaves the accumulated means and counts unchanged.
    pub fn accumulate_loss(
        &mut self,
        model: &NativeCausalLm,
        windows: &[usize],
        shape: [usize; 2],
    ) -> CausalLmResult<f32> {
        self.accumulate_loss_with_labels(model, windows, shape, None)
    }

    pub fn accumulate_loss_with_labels(
        &mut self,
        model: &NativeCausalLm,
        windows: &[usize],
        shape: [usize; 2],
        labels: Option<&[i64]>,
    ) -> CausalLmResult<f32> {
        self.objective_input_count(model, shape, labels.is_some())?;
        let prepared = model.prepare_objective(windows, shape, labels, self.objective)?;
        self.next_counts(shape, prepared.target_tokens)?;
        let batch = model.geometry_loss_gradients_prepared(prepared)?;
        self.add(&batch)?;
        Ok(batch.loss())
    }

    /// Reduce a sealed native loss batch, validating everything before commit.
    /// f64 online means avoid f32 sum overflow and retain tiny contributions.
    pub fn add(&mut self, batch: &NativeCausalLmGradients) -> CausalLmResult<()> {
        if batch.base_id != self.base_id
            || batch.readout_accumulator != self.readout_accumulator
            || batch.geometry_id != self.geometry_id
            || batch.objective != self.objective
        {
            return Err("LM accumulation cannot mix model, geometry or objective snapshots".into());
        }
        let (targets, total, microbatches) = self.next_counts(batch.shape, batch.target_tokens)?;
        let primary = batch
            .causal_parameters
            .as_ref()
            .unwrap_or(&batch.parameters);
        if !batch.loss.is_finite()
            || !batch.causal_loss.is_finite()
            || primary.len() != self.means.len()
        {
            return Err("invalid LM accumulation loss or parameter set".into());
        }
        let ratio = targets as f64 / total as f64;
        let loss = self.loss + (f64::from(batch.causal_loss) - self.loss) * ratio;
        let mut means = self.means.clone();
        for (path, parameters) in &mut means {
            let incoming = primary.get(path).ok_or("missing LM gradient placement")?;
            if incoming.len() != parameters.len() {
                return Err("LM accumulation parameter set mismatch".into());
            }
            for (name, values) in parameters {
                let gradient = incoming.get(name).ok_or("missing LM parameter gradient")?;
                if gradient.len() != values.len() || gradient.iter().any(|v| !v.is_finite()) {
                    return Err("invalid LM accumulation gradient shape or values".into());
                }
                for (mean, &value) in values.iter_mut().zip(gradient) {
                    *mean += (f64::from(value) - *mean) * ratio;
                }
            }
        }
        let mut repetition_means = self.repetition_means.clone();
        let mut repetition_loss = self.repetition_loss;
        let mut repetition_positions = self.repetition_positions;
        if let Some(auxiliary) = &batch.repetition {
            if auxiliary.active_positions == 0
                || auxiliary.active_positions > batch.target_tokens
                || !auxiliary.loss.is_finite()
            {
                return Err("invalid repetition accumulation batch".into());
            }
            let count = auxiliary.active_positions as u64;
            repetition_positions = repetition_positions
                .checked_add(count)
                .filter(|&n| n <= MAX_TARGET_TOKENS)
                .ok_or("repetition accumulation exceeds count budget")?;
            let ratio = count as f64 / repetition_positions as f64;
            repetition_loss += (f64::from(auxiliary.loss) - repetition_loss) * ratio;
            let means = repetition_means
                .as_mut()
                .ok_or("unexpected repetition gradients")?;
            if means.len() != auxiliary.parameters.len() {
                return Err("repetition placement mismatch".into());
            }
            for (path, group) in means {
                let incoming = auxiliary
                    .parameters
                    .get(path)
                    .ok_or("missing repetition placement")?;
                if group.len() != incoming.len() {
                    return Err("repetition parameter mismatch".into());
                }
                for (name, values) in group {
                    let gradients = incoming.get(name).ok_or("missing repetition parameter")?;
                    if values.len() != gradients.len() || gradients.iter().any(|v| !v.is_finite()) {
                        return Err("invalid repetition gradient".into());
                    }
                    for (mean, &value) in values.iter_mut().zip(gradients) {
                        *mean += (f64::from(value) - *mean) * ratio;
                    }
                }
            }
        }
        self.means = means;
        self.loss = loss;
        self.target_tokens = total;
        self.microbatches = microbatches;
        self.repetition_means = repetition_means;
        self.repetition_loss = repetition_loss;
        self.repetition_positions = repetition_positions;
        Ok(())
    }

    /// Detached f32 means, rounded once at the update/inspection boundary.
    /// No fictitious rectangular shape is assigned to variable-length windows.
    pub fn parameter_gradients(&self) -> CausalLmResult<Parameters> {
        if self.target_tokens == 0 {
            return Err("cannot use empty LM accumulation".into());
        }
        self.means
            .iter()
            .map(|(path, parameters)| {
                let parameters = parameters
                    .iter()
                    .map(|(name, values)| {
                        let values: Vec<f32> = values
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| match &self.repetition_means {
                                Some(m) => {
                                    (v + self.objective.repetition.unwrap().config().strength
                                        * m[path][name][i])
                                        as f32
                                }
                                None => v as f32,
                            })
                            .collect();
                        if values.iter().any(|v| !v.is_finite()) {
                            return Err("LM accumulation mean is not finite f32".into());
                        }
                        Ok((name.clone(), values))
                    })
                    .collect::<CausalLmResult<_>>()?;
                Ok((path.clone(), parameters))
            })
            .collect()
    }

    /// One optimizer update for the combined mean; this accumulator is not reset.
    pub fn step_sgd(&self, model: &NativeCausalLm, rate: f32) -> CausalLmResult<NativeCausalLm> {
        self.validate_model(model)?;
        model.with_geometry_sgd(
            &self.base_id,
            self.readout_accumulator,
            &self.geometry_id,
            &self.parameter_gradients()?,
            rate,
        )
    }

    /// Clipping and moments run once, after reduction, via the shared owner.
    pub fn step_adamw(
        &self,
        model: &NativeCausalLm,
        optimizer: &NativeGeometryAdamW,
        step: AdamWStep,
    ) -> CausalLmResult<(NativeCausalLm, NativeGeometryAdamW)> {
        self.validate_model(model)?;
        model.with_geometry_adamw(
            &self.base_id,
            self.readout_accumulator,
            &self.geometry_id,
            &self.parameter_gradients()?,
            optimizer,
            step,
        )
    }
}

#[cfg(test)]
mod tests;
