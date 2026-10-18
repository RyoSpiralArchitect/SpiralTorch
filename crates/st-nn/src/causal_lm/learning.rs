//! Exact first-order LM-loss pullback through a frozen decoder into geometry.
//! Checkpoint block boundaries; replay only one block's internal tape at a time.
use super::*;
use crate::geometry_artifact::GeometryAdapterTensorRole;
use crate::geometry_runtime::NativeGeometryBatch;
use st_kernel_contracts::attention::attention_vjp_reference;
use std::collections::BTreeMap;

type Parameters = BTreeMap<String, BTreeMap<String, Vec<f32>>>;

/// Sealed gradients from one model/artifact snapshot, not a caller-authored update.
#[derive(Debug)]
pub struct NativeCausalLmGradients {
    base_id: String,
    geometry_id: String,
    shape: [usize; 2],
    loss: f32,
    parameters: Parameters,
}

impl NativeCausalLmGradients {
    pub fn loss(&self) -> f32 {
        self.loss
    }
    pub fn base_model_identity(&self) -> &str {
        &self.base_id
    }
    pub fn geometry_artifact_id(&self) -> &str {
        &self.geometry_id
    }
    pub fn window_shape(&self) -> [usize; 2] {
        self.shape
    }
    pub fn parameters(&self) -> &Parameters {
        &self.parameters
    }
}

fn frozen_pullback(
    weight: &PackedB,
    cache: &OnceLock<Result<PackedB, String>>,
    upstream: &Tensor,
) -> CausalLmResult<Tensor> {
    // Serialize construction, not just publication: concurrent first calls must
    // not each allocate a vocabulary-sized transpose. Frozen layout errors are
    // deterministic, so they can be retained alongside successful initialization.
    let weight = cache
        .get_or_init(|| weight.transposed().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| e.clone())?;
    Ok(upstream.matmul_prepacked_with_backend(weight, MatmulBackend::CpuSimd)?)
}

impl Dense {
    fn backward_input(&self, upstream: &Tensor) -> CausalLmResult<Tensor> {
        frozen_pullback(&self.weight, &self.backward_weight, upstream)
    }
}

fn geometry_pullback(
    path: String,
    batch: Option<NativeGeometryBatch>,
    upstream: Tensor,
    parameters: &mut Parameters,
) -> CausalLmResult<Tensor> {
    if let Some(batch) = batch {
        let gradient = batch.vjp(upstream.data())?;
        if parameters.insert(path, gradient.parameters).is_some() {
            return Err("duplicate geometry gradient placement".into());
        }
        return Ok(gradient.input);
    }
    Ok(upstream)
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
        let steps = shape[1]
            .checked_sub(1)
            .filter(|&n| n > 0)
            .ok_or("geometry loss needs at least two IDs per window")?;
        self.input_count([shape[0], steps], false)?;
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
        product(&shape, usize::MAX, "window IDs")
    }

    /// Mean next-token NLL and all placement parameter gradients, with base
    /// weights frozen. Gradients include frozen parameters for inspection; SGD
    /// respects each parameter's requires_grad flag and never updates buffers.
    pub fn geometry_loss_gradients(
        &self,
        windows: &[usize],
        window_shape: [usize; 2],
    ) -> CausalLmResult<NativeCausalLmGradients> {
        self.geometry_gradient_input_count(window_shape)?;
        let (ids, labels, shape) = self.shift_windows(windows, window_shape)?;
        let runtime = self.geometry.as_ref().unwrap();
        let c = &self.config;
        let (first_layer, first_site) = self.first_geometry_site()?;
        let mut checkpoints = Vec::with_capacity(c.layers - first_layer + 1);
        let mut hidden = self.embed(&ids, shape)?;
        for index in 0..c.layers {
            if index >= first_layer {
                checkpoints.push(hidden.clone());
            }
            hidden = self.run_block(index, hidden, shape, false)?.output;
        }
        checkpoints.push(hidden.clone());
        let normalized = self.final_norm.forward(&hidden, c.norm_epsilon)?;
        let logits =
            normalized.matmul_prepacked_with_backend(&self.head, MatmulBackend::CpuSimd)?;
        let loss = logits
            .cross_entropy_with_logits(&labels, CrossEntropyConfig::default())?
            .data()[0];
        let seed = Tensor::from_vec(1, 1, vec![1.])?;
        let d_logits = logits.cross_entropy_with_logits_backward(
            &labels,
            CrossEntropyConfig::default(),
            &seed,
        )?;
        let d_normalized = frozen_pullback(&self.head, &self.backward_head, &d_logits)?;
        let mut gradient = hidden.layer_norm_input_backward(
            &self.final_norm.gamma,
            &d_normalized,
            c.norm_epsilon,
        )?;
        drop((hidden, normalized, logits, d_logits, d_normalized));
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
            let residual = geometry_pullback(
                path.clone(),
                cache.block_geometry,
                gradient,
                &mut parameters,
            )?;
            if path == first_site {
                break;
            }
            let mlp = geometry_pullback(
                format!("{path}.mlp"),
                cache.mlp_geometry,
                residual.clone(),
                &mut parameters,
            )?;
            if format!("{path}.mlp") == first_site {
                break;
            }
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
            let attention = geometry_pullback(
                format!("{path}.attn"),
                cache.attention_geometry,
                residual.clone(),
                &mut parameters,
            )?;
            if format!("{path}.attn") == first_site {
                break;
            }
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
        Ok(NativeCausalLmGradients {
            base_id: self.base_id.clone(),
            geometry_id: runtime.artifact().artifact_id().to_owned(),
            shape: window_shape,
            loss,
            parameters,
        })
    }

    /// Atomic stateless SGD across every trainable placement. Return a new
    /// model sharing frozen base storage; the original model remains unchanged.
    pub fn step_geometry_sgd(
        &self,
        gradients: &NativeCausalLmGradients,
        learning_rate: f32,
    ) -> CausalLmResult<Self> {
        if gradients.base_id != self.base_id {
            return Err("LM gradients belong to a different base model".into());
        }
        let runtime = self
            .geometry
            .as_ref()
            .ok_or("native LM has no attached geometry")?;
        let geometry = runtime.with_gradients_sgd(
            &gradients.geometry_id,
            &gradients.parameters,
            learning_rate,
        )?;
        let mut next = self.clone();
        next.geometry = Some(Arc::new(geometry));
        Ok(next)
    }
}
