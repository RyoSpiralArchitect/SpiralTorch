//! Exact first-order LM-loss pullback through a frozen decoder into geometry.
//! Checkpoint block boundaries; replay only one block's internal tape at a time.
use super::*;
use crate::geometry_artifact::GeometryAdapterTensorRole;
use crate::geometry_runtime::NativeGeometryBatch;
use st_kernel_contracts::attention::attention_vjp_reference;
use std::collections::BTreeMap;

type Parameters = BTreeMap<String, BTreeMap<String, Vec<f32>>>;

mod accumulation;
pub use accumulation::NativeCausalLmAccumulator;

/// Sealed gradients from one model/artifact snapshot, not a caller-authored update.
#[derive(Debug)]
pub struct NativeCausalLmGradients {
    base_id: String,
    readout_accumulator: CausalLmReadoutAccumulator,
    geometry_id: String,
    shape: [usize; 2],
    target_tokens: usize,
    objective: CausalLmObjective,
    loss: f32,
    causal_loss: f32,
    parameters: Parameters,
    causal_parameters: Option<Parameters>,
    repetition: Option<RepetitionGradients>,
}

#[derive(Debug)]
struct RepetitionGradients {
    loss: f32,
    active_positions: usize,
    parameters: Parameters,
}

impl NativeCausalLmGradients {
    pub fn loss(&self) -> f32 {
        self.loss
    }
    pub fn causal_loss(&self) -> f32 {
        self.causal_loss
    }
    pub fn repetition_loss(&self) -> f32 {
        self.repetition.as_ref().map_or(0., |r| r.loss)
    }
    pub fn repetition_positions(&self) -> usize {
        self.repetition.as_ref().map_or(0, |r| r.active_positions)
    }
    pub fn base_model_identity(&self) -> &str {
        &self.base_id
    }
    pub fn readout_accumulator(&self) -> CausalLmReadoutAccumulator {
        self.readout_accumulator
    }
    pub fn geometry_artifact_id(&self) -> &str {
        &self.geometry_id
    }
    pub fn window_shape(&self) -> [usize; 2] {
        self.shape
    }
    pub fn target_tokens(&self) -> usize {
        self.target_tokens
    }
    pub fn objective(&self) -> CausalLmObjective {
        self.objective
    }
    pub fn parameters(&self) -> &Parameters {
        &self.parameters
    }
}

fn frozen_pullback(
    weight: &PackedB,
    cache: &OnceLock<Result<PackedB, String>>,
    upstream: &Tensor,
    accumulator: CausalLmReadoutAccumulator,
) -> CausalLmResult<Tensor> {
    // Serialize construction, not just publication: concurrent first calls must
    // not each allocate a vocabulary-sized transpose. Frozen layout errors are
    // deterministic, so they can be retained alongside successful initialization.
    let weight = cache
        .get_or_init(|| weight.transposed().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| e.clone())?;
    Ok(match accumulator {
        CausalLmReadoutAccumulator::F32 => {
            upstream.matmul_prepacked_with_backend(weight, MatmulBackend::CpuSimd)?
        }
        CausalLmReadoutAccumulator::F64 => upstream.matmul_prepacked_wide(weight)?,
    })
}

impl Dense {
    fn backward_input(&self, upstream: &Tensor) -> CausalLmResult<Tensor> {
        frozen_pullback(
            &self.weight,
            &self.backward_weight,
            upstream,
            CausalLmReadoutAccumulator::F32,
        )
    }
}

fn geometry_pullback(
    path: String,
    batch: Option<NativeGeometryBatch>,
    upstream: Tensor,
    parameters: &mut Parameters,
    first_site: &str,
) -> CausalLmResult<Option<Tensor>> {
    if let Some(batch) = batch {
        let (input, gradient) = if path == first_site {
            (None, batch.vjp_parameters(upstream.data())?)
        } else {
            let gradient = batch.vjp(upstream.data())?;
            (Some(gradient.input), gradient.parameters)
        };
        if parameters.insert(path, gradient).is_some() {
            return Err("duplicate geometry gradient placement".into());
        }
        return Ok(input);
    }
    Ok(Some(upstream))
}

impl NativeCausalLm {
    fn first_geometry_site(&self) -> CausalLmResult<(usize, String)> {
        let runtime = self
            .geometry
            .as_ref()
            .ok_or("native LM has no attached geometry")?;
        for layer in 0..self.config.layers {
            for suffix in [".attn", ".mlp", ""] {
                let path = format!("transformer.h.{layer}{suffix}");
                if runtime
                    .artifact()
                    .record()
                    .placements
                    .iter()
                    .any(|p| p.path == path)
                {
                    return Ok((layer, path));
                }
            }
        }
        Err("native LM has no participating geometry site".into())
    }

    /// Admit a complete labeled window before a binding extracts any IDs.
    /// Existing forward/geometry limits apply to each replay; checkpoints have
    /// an additional aggregate element bound. This is not a peak-RSS guarantee.
    pub fn geometry_gradient_input_count(&self, shape: [usize; 2]) -> CausalLmResult<usize> {
        self.geometry_objective_input_count(shape, false)
    }

    pub fn geometry_objective_input_count(
        &self,
        shape: [usize; 2],
        has_labels: bool,
    ) -> CausalLmResult<usize> {
        let runtime = self
            .geometry
            .as_ref()
            .ok_or("native LM has no attached geometry")?;
        if !runtime.artifact().record().placements.iter().any(|p| {
            p.tensors
                .iter()
                .any(|t| t.role == GeometryAdapterTensorRole::Parameter && t.requires_grad)
        }) {
            return Err("attached geometry has no trainable parameters".into());
        }
        let count = self.loss_objective_input_count(shape, has_labels)?;
        let steps = shape[1] - 1;
        let (first_layer, _) = self.first_geometry_site()?;
        product(
            &[
                self.config.layers - first_layer + 1,
                shape[0],
                steps,
                self.config.width,
            ],
            self.limits.max_training_checkpoint_elements,
            "training checkpoints",
        )?;
        Ok(count)
    }

    /// Mean next-token NLL and all placement parameter gradients, with base
    /// weights frozen. Gradients include frozen parameters for inspection; SGD
    /// respects each parameter's requires_grad flag and never updates buffers.
    pub fn geometry_loss_gradients(
        &self,
        windows: &[usize],
        window_shape: [usize; 2],
    ) -> CausalLmResult<NativeCausalLmGradients> {
        self.geometry_loss_gradients_with_objective(
            windows,
            window_shape,
            None,
            CausalLmObjective::default(),
        )
    }

    pub fn geometry_loss_gradients_with_objective(
        &self,
        windows: &[usize],
        window_shape: [usize; 2],
        labels: Option<&[i64]>,
        objective: CausalLmObjective,
    ) -> CausalLmResult<NativeCausalLmGradients> {
        objective.validate()?;
        self.geometry_objective_input_count(window_shape, labels.is_some())?;
        let prepared = self.prepare_objective(windows, window_shape, labels, objective)?;
        self.geometry_loss_gradients_prepared(prepared)
    }

    fn geometry_loss_gradients_prepared(
        &self,
        prepared: PreparedCausalWindow,
    ) -> CausalLmResult<NativeCausalLmGradients> {
        let PreparedCausalWindow {
            input: ids,
            labels,
            shape,
            window_shape,
            target_tokens,
            selected_rows,
            objective,
            repetition,
        } = prepared;
        let runtime = self.geometry.as_ref().unwrap();
        let c = &self.config;
        let (first_layer, _) = self.first_geometry_site()?;
        let mut checkpoints = Vec::with_capacity(c.layers - first_layer + 1);
        let mut hidden = self.embed(&ids, shape)?;
        for index in 0..c.layers {
            if index >= first_layer {
                checkpoints.push(hidden.clone());
            }
            hidden = self.run_block(index, hidden, shape, false)?.output;
        }
        checkpoints.push(hidden.clone());
        let selected = match &selected_rows {
            Some(rows) => hidden.gather_rows(rows)?,
            None => hidden.clone(),
        };
        let normalized = self.final_norm.forward(&selected, c.norm_epsilon)?;
        let logits = self.project_readout(&normalized)?;
        let causal_loss = logits
            .cross_entropy_with_logits(&labels, objective.kernel_config())?
            .data()[0];
        let seed = Tensor::from_vec(1, 1, vec![1.])?;
        let d_logits =
            logits.cross_entropy_with_logits_backward(&labels, objective.kernel_config(), &seed)?;
        let d_normalized = frozen_pullback(
            &self.head,
            &self.backward_head,
            &d_logits,
            self.readout_accumulator,
        )?;
        drop(d_logits);
        let repetition = match repetition {
            Some(window) => {
                let (loss, gradient, active_positions) =
                    window.loss_gradient(&logits, selected_rows.as_deref(), shape[1])?;
                if active_positions == 0 {
                    None
                } else {
                    let d_normalized = frozen_pullback(
                        &self.head,
                        &self.backward_head,
                        &gradient,
                        self.readout_accumulator,
                    )?;
                    Some((loss, active_positions, d_normalized))
                }
            }
            None => None,
        };
        let hidden_rows = hidden.shape().0;
        drop((logits, normalized, hidden));
        let auxiliary_checkpoints = repetition.as_ref().map(|_| checkpoints.clone());
        let auxiliary_selected = repetition.as_ref().map(|_| selected.clone());
        let causal_parameters = self.parameter_pullback(
            checkpoints,
            hidden_rows,
            selected,
            shape,
            selected_rows.as_deref(),
            d_normalized,
        )?;
        let repetition = match repetition {
            Some((loss, active_positions, gradient)) => Some(RepetitionGradients {
                loss,
                active_positions,
                parameters: self.parameter_pullback(
                    auxiliary_checkpoints.unwrap(),
                    hidden_rows,
                    auxiliary_selected.unwrap(),
                    shape,
                    selected_rows.as_deref(),
                    gradient,
                )?,
            }),
            None => None,
        };
        let mut parameters = causal_parameters;
        let causal_parameters = repetition.as_ref().map(|_| parameters.clone());
        let mut loss = causal_loss;
        if let Some(auxiliary) = &repetition {
            let strength = objective.repetition.unwrap().config().strength;
            loss = (f64::from(causal_loss) + strength * f64::from(auxiliary.loss)) as f32;
            for (path, group) in &mut parameters {
                for (name, values) in group {
                    for (value, &other) in values.iter_mut().zip(&auxiliary.parameters[path][name])
                    {
                        *value = (f64::from(*value) + strength * f64::from(other)) as f32;
                        if !value.is_finite() {
                            return Err("nonfinite combined repetition gradient".into());
                        }
                    }
                }
            }
        }
        if !loss.is_finite() {
            return Err("nonfinite combined repetition loss".into());
        }
        Ok(NativeCausalLmGradients {
            base_id: self.base_id.clone(),
            readout_accumulator: self.readout_accumulator,
            geometry_id: runtime.artifact().artifact_id().to_owned(),
            shape: window_shape,
            target_tokens,
            objective,
            loss,
            causal_loss,
            parameters,
            causal_parameters,
            repetition,
        })
    }

    fn parameter_pullback(
        &self,
        mut checkpoints: Vec<Tensor>,
        hidden_rows: usize,
        selected: Tensor,
        shape: [usize; 2],
        selected_rows: Option<&[usize]>,
        d_normalized: Tensor,
    ) -> CausalLmResult<Parameters> {
        let runtime = self.geometry.as_ref().unwrap();
        let c = &self.config;
        let (first_layer, first_site) = self.first_geometry_site()?;
        let d_selected = selected.layer_norm_input_backward(
            &self.final_norm.gamma,
            &d_normalized,
            c.norm_epsilon,
        )?;
        drop((selected, d_normalized));
        let mut gradient = match &selected_rows {
            Some(rows) => d_selected.scatter_add_rows(rows, hidden_rows)?,
            None => d_selected,
        };
        checkpoints.pop();
        let mut parameters = BTreeMap::new();
        for index in (first_layer..c.layers).rev() {
            let input = checkpoints.pop().unwrap();
            let cache = self
                .run_block(index, input.clone(), shape, true)?
                .cache
                .unwrap();
            let block = &self.blocks[index];
            let path = format!("transformer.h.{index}");
            let Some(residual) = geometry_pullback(
                path.clone(),
                cache.block_geometry,
                gradient,
                &mut parameters,
                &first_site,
            )?
            else {
                break;
            };
            let Some(mlp) = geometry_pullback(
                format!("{path}.mlp"),
                cache.mlp_geometry,
                residual.clone(),
                &mut parameters,
                &first_site,
            )?
            else {
                break;
            };
            let d_gelu = block.contract.backward_input(&mlp)?;
            let d_expand = cache
                .pre_gelu
                .gelu_backward_with_backend(&d_gelu, TensorUtilBackend::Cpu)?;
            let d_norm = block.expand.backward_input(&d_expand)?;
            let d_residual = cache.after_attention.layer_norm_input_backward(
                &block.second_norm.gamma,
                &d_norm,
                c.norm_epsilon,
            )?;
            let residual = residual.add_with_backend(&d_residual, TensorUtilBackend::Cpu)?;
            let Some(attention) = geometry_pullback(
                format!("{path}.attn"),
                cache.attention_geometry,
                residual.clone(),
                &mut parameters,
                &first_site,
            )?
            else {
                break;
            };
            let d_merge = block.output.backward_input(&attention)?;
            let [d_heads] = self.split_heads::<1>(&d_merge, shape)?;
            let d_qkv = attention_vjp_reference(
                self.attention_spec(index, shape)?,
                &cache.qkv[0],
                &cache.qkv[1],
                &cache.qkv[2],
                None,
                None,
                &d_heads,
            )?;
            let d_fused = self.merge_heads(
                [
                    d_qkv.query.as_slice(),
                    d_qkv.key.as_slice(),
                    d_qkv.value.as_slice(),
                ],
                shape,
            )?;
            let d_norm = block.qkv.backward_input(&d_fused)?;
            let d_input = input.layer_norm_input_backward(
                &block.first_norm.gamma,
                &d_norm,
                c.norm_epsilon,
            )?;
            gradient = residual.add_with_backend(&d_input, TensorUtilBackend::Cpu)?;
        }
        if parameters.len() != runtime.artifact().record().placements.len() {
            return Err("not all native geometry placements received a loss gradient".into());
        }
        Ok(parameters)
    }

    /// Atomic stateless SGD across every trainable placement. Return a new
    /// model sharing frozen base storage; the original model remains unchanged.
    pub fn step_geometry_sgd(
        &self,
        gradients: &NativeCausalLmGradients,
        learning_rate: f32,
    ) -> CausalLmResult<Self> {
        self.with_geometry_sgd(
            &gradients.base_id,
            gradients.readout_accumulator,
            &gradients.geometry_id,
            &gradients.parameters,
            learning_rate,
        )
    }

    fn with_geometry_sgd(
        &self,
        base_id: &str,
        accumulator: CausalLmReadoutAccumulator,
        geometry_id: &str,
        parameters: &Parameters,
        learning_rate: f32,
    ) -> CausalLmResult<Self> {
        if base_id != self.base_id || accumulator != self.readout_accumulator {
            return Err(
                "LM gradients belong to a different base model or readout accumulator".into(),
            );
        }
        let runtime = self
            .geometry
            .as_ref()
            .ok_or("native LM has no attached geometry")?;
        let geometry = runtime.with_gradients_sgd(geometry_id, parameters, learning_rate)?;
        let mut next = self.clone();
        next.geometry = Some(Arc::new(geometry));
        Ok(next)
    }

    /// Geometry-only AdamW with shared native clipping, moments and clocks.
    /// Both returned owners commit together; the original pair and base stay unchanged.
    pub fn step_geometry_adamw(
        &self,
        gradients: &NativeCausalLmGradients,
        optimizer: &crate::geometry_runtime::NativeGeometryAdamW,
        step: crate::geometry_runtime::AdamWStep,
    ) -> CausalLmResult<(Self, crate::geometry_runtime::NativeGeometryAdamW)> {
        self.with_geometry_adamw(
            &gradients.base_id,
            gradients.readout_accumulator,
            &gradients.geometry_id,
            &gradients.parameters,
            optimizer,
            step,
        )
    }

    fn with_geometry_adamw(
        &self,
        base_id: &str,
        accumulator: CausalLmReadoutAccumulator,
        geometry_id: &str,
        parameters: &Parameters,
        optimizer: &crate::geometry_runtime::NativeGeometryAdamW,
        step: crate::geometry_runtime::AdamWStep,
    ) -> CausalLmResult<(Self, crate::geometry_runtime::NativeGeometryAdamW)> {
        if base_id != self.base_id || accumulator != self.readout_accumulator {
            return Err(
                "LM gradients belong to a different base model or readout accumulator".into(),
            );
        }
        let runtime = self
            .geometry
            .as_ref()
            .ok_or("native LM has no attached geometry")?;
        let (geometry, optimizer) =
            runtime.with_gradients_adamw(geometry_id, parameters, optimizer, step)?;
        let mut next = self.clone();
        next.geometry = Some(Arc::new(geometry));
        Ok((next, optimizer))
    }
}
