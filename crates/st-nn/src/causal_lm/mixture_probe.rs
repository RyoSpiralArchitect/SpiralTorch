//! A fixed-primal diagnostic for the last attention-output geometry VJP.
use super::*;
use std::collections::BTreeMap;

/// Bound diagnostic result storage in addition to ordinary model workspace limits.
pub const MAX_MIXTURE_PROBE_ELEMENTS: usize = 1_048_576;

#[derive(Debug)]
pub struct CausalLmMixturePath {
    pub head_backward: CausalLmReadoutAccumulator,
    /// Active row at each boundary, in reverse-pass order.
    pub boundaries: Vec<(&'static str, Vec<f32>)>,
    pub parameters: BTreeMap<String, Vec<f32>>,
}

#[derive(Debug)]
pub struct CausalLmMixtureObjectiveTrace {
    pub objective: &'static str,
    pub loss: f32,
    pub d_logits: Vec<f32>,
    pub paths: Vec<CausalLmMixturePath>,
}

#[derive(Debug)]
pub struct CausalLmMixtureProbe {
    pub base_model_identity: String,
    pub lora_artifact_id: Option<String>,
    pub geometry_artifact_id: String,
    pub site: String,
    pub prompt_tokens: usize,
    pub target_id: usize,
    pub rival_id: usize,
    pub mix: f64,
    /// One f32 forward logit vector shared by all seeds and both head VJPs.
    pub logits: Vec<f32>,
    pub objectives: Vec<CausalLmMixtureObjectiveTrace>,
}

fn result_count(v: usize, w: usize, i: usize, params: u128) -> CausalLmResult<()> {
    let total = 4 * v as u128 + 6 * (5 * w as u128 + 2 * i as u128 + params);
    if total > MAX_MIXTURE_PROBE_ELEMENTS as u128 {
        return Err("mixture pullback probe result exceeds element budget".into());
    }
    Ok(())
}

fn active_row(t: &Tensor, rows: usize) -> CausalLmResult<Vec<f32>> {
    let (n, cols) = t.shape();
    if n != rows || rows == 0 || cols == 0 {
        return Err("unexpected mixture probe boundary shape".into());
    }
    let (before, last) = t.data().split_at((rows - 1) * cols);
    if before.iter().any(|&v| v != 0.) || last.iter().any(|v| !v.is_finite()) {
        return Err("invalid mixture boundary: nonzero earlier row or nonfinite adjoint".into());
    }
    Ok(last.to_vec())
}

impl NativeCausalLm {
    /// Admit owner, context and aggregate result sizes before a binding copies IDs.
    pub fn mixture_probe_input_count(
        &self,
        steps: usize,
        target: usize,
        rival: usize,
        mix: f64,
    ) -> CausalLmResult<usize> {
        let c = &self.config;
        if self.readout_accumulator != CausalLmReadoutAccumulator::F32
            || steps == 0
            || target == rival
            || target >= c.vocabulary
            || rival >= c.vocabulary
            || !(0.0..=1.0).contains(&mix)
        {
            return Err("mixture probe needs an f32 model, nonempty prompt, distinct vocabulary IDs and mix in [0,1]".into());
        }
        self.validate_adaptation_mode(CausalLmAdaptationMode::Geometry)?;
        let artifact = self.geometry_artifact().ok_or("missing probe geometry")?;
        let placements = &artifact.record().placements;
        let path = format!("transformer.h.{}.attn", c.layers - 1);
        if placements.len() != 1 || placements[0].path != path {
            return Err("mixture probe requires a single final attention-output geometry".into());
        }
        self.adaptation_labeled_input_count(
            [
                1,
                steps.checked_add(1).ok_or("probe prompt length overflow")?,
            ],
            true,
        )?;
        let params = placements[0]
            .tensors
            .iter()
            .map(|x| x.values.len() as u128)
            .sum();
        result_count(c.vocabulary, c.width, c.intermediate, params)?;
        Ok(steps)
    }

    /// A single f32 full-prefix forward. For each seed, widen only the
    /// transpose product through the vocabulary head, never the forward logits,
    /// other VJPs or returned f32 storage. No training state is advanced.
    pub fn last_attention_mixture_probe(
        &self,
        prompt: &[usize],
        target: usize,
        rival: usize,
        mix: f64,
    ) -> CausalLmResult<CausalLmMixtureProbe> {
        let steps = self.mixture_probe_input_count(prompt.len(), target, rival, mix)?;
        let shape = [1, steps];
        let index = self.config.layers - 1;
        let hidden = self.forward_blocks(self.embed(prompt, shape)?, shape, 0..index)?;
        let block = self.run_block(index, hidden, shape, decoder::BlockPass::BackwardReplay)?;
        let cache = block
            .cache
            .ok_or("probe block did not retain its forward")?;
        let geom = cache
            .attention_geometry
            .as_ref()
            .ok_or("missing cached probe geometry")?;
        let selected = block.output.gather_rows(&[steps - 1])?;
        let normalized = self
            .final_norm
            .forward(&selected, self.config.norm_epsilon)?;
        let logits = self.project_readout(&normalized)?;
        let plan = [DeclaredRivalRow {
            position: 0,
            target,
            rivals: vec![rival],
        }];
        let labels = [target as i64];
        let mut objectives = Vec::with_capacity(3);
        for (objective, weight) in [("ce", None), ("declared", Some(1.)), ("mixed", Some(mix))] {
            let (loss, seed) = match weight {
                None => {
                    learning::ordinary_loss_seed(&logits, &labels, CrossEntropyConfig::default())?
                }
                Some(w) => {
                    let (r, g) =
                        logits.declared_rival_cross_entropy_mean_gradient(&labels, &plan, w)?;
                    (r.loss, g)
                }
            };
            let mut paths = Vec::with_capacity(2);
            for head_backward in [
                CausalLmReadoutAccumulator::F32,
                CausalLmReadoutAccumulator::F64,
            ] {
                let head = learning::frozen_pullback(
                    &self.head,
                    &self.backward_head,
                    &seed,
                    head_backward.backward_precision(),
                )?;
                let final_input = selected.layer_norm_input_backward(
                    &self.final_norm.gamma,
                    &head,
                    self.config.norm_epsilon,
                )?;
                let gradient = if steps == 1 {
                    final_input.clone()
                } else {
                    final_input.scatter_add_rows(&[steps - 1], steps)?
                };
                let mut boundaries = vec![
                    ("head_input", active_row(&head, 1)?),
                    ("final_norm_input", active_row(&final_input, 1)?),
                ];
                let mlp_branch = self.blocks[index].mlp_input_pullback(
                    &cache.pre_gelu,
                    &cache.after_attention,
                    &gradient,
                    self.config.norm_epsilon,
                    |name, adjoint| {
                        boundaries.push((name, active_row(adjoint, steps)?));
                        Ok(())
                    },
                )?;
                let incoming = gradient.add_with_backend(&mlp_branch, TensorUtilBackend::Cpu)?;
                boundaries.push(("attention_output_upstream", active_row(&incoming, steps)?));
                let parameters = geom.vjp_parameters(incoming.data())?;
                paths.push(CausalLmMixturePath {
                    head_backward,
                    boundaries,
                    parameters,
                });
            }
            objectives.push(CausalLmMixtureObjectiveTrace {
                objective,
                loss,
                d_logits: active_row(&seed, 1)?,
                paths,
            });
        }
        let geom = self.geometry_artifact().ok_or("missing probe geometry")?;
        Ok(CausalLmMixtureProbe {
            base_model_identity: self.base_model_identity().to_owned(),
            lora_artifact_id: self.lora_artifact_id().map(str::to_owned),
            geometry_artifact_id: geom.artifact_id().to_owned(),
            site: geom.record().placements[0].path.clone(),
            prompt_tokens: steps,
            target_id: target,
            rival_id: rival,
            mix,
            logits: logits.data().to_vec(),
            objectives,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn output_budget_counts_seeds_six_paths_and_all_returned_rows() {
        assert!(result_count(50257, 768, 3072, 1536).is_ok());
        assert!(result_count(MAX_MIXTURE_PROBE_ELEMENTS / 4, 1, 1, 1).is_err());
        assert!(result_count(1, 1, 1, MAX_MIXTURE_PROBE_ELEMENTS as u128 / 6).is_err());
        assert!(active_row(&Tensor::from_vec(2, 1, vec![1., 0.]).unwrap(), 2).is_err());
    }
}
