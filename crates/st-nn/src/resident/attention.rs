//! Snapshot lowering of existing projection parameters. QKV is fused once on
//! the host; inference stays frozen and training owns separate resident updates.

use super::*;
use crate::Linear;
use st_kernel_contracts::attention::{AttentionMask, AttentionSpec};

#[cfg(feature = "wgpu")]
mod training;
#[cfg(feature = "wgpu")]
pub use training::{ResidentAttentionForward, ResidentAttentionTraining, ResidentAttentionVjp};

/// Self-attention inference plan for `[batch, sequence, input_features]`.
/// Parameter updates require rebuilding the plan. Bias inputs are independent
/// runtime tensors; this plan neither manufactures geometry nor drops metadata
/// from a richer attention model under the guise of a complete forward.
#[derive(Clone, Debug)]
pub struct AttentionInferencePlan {
    qkv: InferencePlan,
    output: InferencePlan,
    spec: AttentionSpec,
}

impl AttentionInferencePlan {
    /// Projections are ordered query, key, value, output. Weights use SpiralTorch
    /// Linear's `[input_features, output_features]` convention.
    pub fn from_linears(
        input: NdLayout,
        heads: usize,
        mask: AttentionMask,
        projections: [&Linear; 4],
    ) -> Result<Self, InferenceError> {
        Self::from_parameters(
            input,
            heads,
            mask,
            projections.map(|layer| (layer.weight().value(), layer.bias().value())),
        )
    }

    /// Freeze imported or module-owned weight/bias pairs. Each bias is `[1, out]`.
    /// Q/K/V must share their projected width; output width may differ from input.
    pub fn from_parameters(
        input: NdLayout,
        heads: usize,
        mask: AttentionMask,
        projections: [(&Tensor, &Tensor); 4],
    ) -> Result<Self, InferenceError> {
        if input.rank() != 3 || input.is_empty() || !input.is_contiguous() || input.offset() != 0 {
            return Err(InferenceError::Attention(
                "expected a nonempty canonical [B,T,I] layout",
            ));
        }
        let [batch, sequence, inner] = <[usize; 3]>::try_from(input.shape()).unwrap();
        let width = projections[0].0.shape().1;
        if heads == 0 || width == 0 || !width.is_multiple_of(heads) {
            return Err(InferenceError::Attention(
                "QKV width must be divisible by nonzero heads",
            ));
        }
        let shape = [batch, heads, sequence, width / heads];
        let spec = AttentionSpec::new(&shape, &shape, &shape, 1. / (shape[3] as f32).sqrt(), mask)?;
        let mut frozen = Vec::with_capacity(4);
        for (index, (weight, bias)) in projections.into_iter().enumerate() {
            let expected_inner = if index == 3 { width } else { inner };
            if weight.shape().0 != expected_inner || (index < 3 && weight.shape().1 != width) {
                return Err(InferenceError::Shape(index));
            }
            let layout = NdLayout::contiguous(&[batch, sequence, expected_inner])?;
            // Reuse the established row-major conversion, validation and COW
            // freezing rules rather than reading live foreign parameter aliases.
            let plan = InferencePlan::from_operations(
                layout,
                vec![InferenceOp::Linear {
                    weight: weight.clone(),
                    bias: bias.clone(),
                }],
            )?;
            frozen.push(plan);
        }
        let fused_width = width.checked_mul(3).ok_or(NdLayoutError::Overflow)?;
        let mut weights = vec![
            0.;
            inner
                .checked_mul(fused_width)
                .ok_or(NdLayoutError::Overflow)?
        ];
        let mut bias = Vec::with_capacity(fused_width);
        for (projection, plan) in frozen[..3].iter().enumerate() {
            let stage = &plan.stages[0];
            for row in 0..inner {
                let start = row * fused_width + projection * width;
                weights[start..start + width]
                    .copy_from_slice(&stage.weight.data()[row * width..(row + 1) * width]);
            }
            bias.extend_from_slice(stage.bias.data());
        }
        let qkv = InferencePlan::from_operations(
            input,
            vec![InferenceOp::Linear {
                weight: Tensor::from_vec(inner, fused_width, weights)?,
                bias: Tensor::from_vec(1, fused_width, bias)?,
            }],
        )?;
        Ok(Self {
            qkv,
            output: frozen.pop().unwrap(),
            spec,
        })
    }

    pub fn input_layout(&self) -> &NdLayout {
        self.qkv.input_layout()
    }
    pub fn output_layout(&self) -> &NdLayout {
        self.output.output_layout()
    }
    pub fn attention_spec(&self) -> AttentionSpec {
        self.spec
    }

    #[cfg(feature = "wgpu")]
    pub fn compile_wgpu(
        &self,
        runtime: st_backend_wgpu::runtime::WgpuRuntime,
    ) -> Result<ResidentAttentionBlock, InferenceError> {
        self.compile_wgpu_with_options(
            runtime,
            Default::default(),
            st_backend_wgpu::resident_matmul::MatmulKernel::Scalar,
            Default::default(),
        )
    }

    /// Choose the existing resident matmul implementation for both projections.
    /// Attention, geometry, ownership and numerical guards are unchanged.
    #[cfg(feature = "wgpu")]
    pub fn compile_wgpu_with_options(
        &self,
        runtime: st_backend_wgpu::runtime::WgpuRuntime,
        tile: st_backend_wgpu::resident_matmul::MatmulTile,
        kernel: st_backend_wgpu::resident_matmul::MatmulKernel,
        accumulation: st_backend_wgpu::resident_matmul::MatmulAccumulation,
    ) -> Result<ResidentAttentionBlock, InferenceError> {
        require_uncommitted_route()?;
        st_backend_wgpu::resident_tensor::attention::validate_limits(
            self.spec,
            &runtime.context().device().limits(),
        )?;
        Ok(ResidentAttentionBlock {
            qkv: self.qkv.compile_graph_wgpu_with_options(
                runtime.clone(),
                tile,
                kernel,
                accumulation,
            )?,
            output: self.output.compile_graph_wgpu_with_options(
                runtime,
                tile,
                kernel,
                accumulation,
            )?,
            spec: self.spec,
        })
    }
}

#[cfg(feature = "wgpu")]
pub struct ResidentAttentionBlock {
    qkv: st_backend_wgpu::resident_graph::ResidentGraph,
    output: st_backend_wgpu::resident_graph::ResidentGraph,
    spec: AttentionSpec,
}

#[cfg(feature = "wgpu")]
impl ResidentAttentionBlock {
    pub fn input_layout(&self) -> &NdLayout {
        self.qkv.input_layout()
    }
    pub fn output_layout(&self) -> &NdLayout {
        self.output.output_layout()
    }
    pub fn tensor_device(&self) -> &st_backend_wgpu::resident_tensor::TensorDevice {
        self.qkv.tensor_device()
    }

    /// Fused QKV projection -> head views -> attention -> output projection.
    /// No host activation transfer or CPU fallback occurs. Attention reads head
    /// views directly; head merging still copies values on GPU. Graph outputs
    /// are owned immutable versions.
    /// This is several submissions, not a claimed single-dispatch fused kernel.
    pub fn forward(
        &mut self,
        input: &st_backend_wgpu::resident_tensor::ResidentTensor,
        z_bias: Option<&st_backend_wgpu::resident_tensor::ResidentTensor>,
        pair_bias: Option<&st_backend_wgpu::resident_tensor::ResidentTensor>,
    ) -> Result<st_backend_wgpu::resident_tensor::ResidentTensor, InferenceError> {
        self.forward_impl::<false>(input, z_bias, pair_bias)
    }

    /// Opt into writing attention directly in head-concatenated order, avoiding
    /// the intermediate head-merge copy. Same values, shapes and guards as
    /// [Self::forward]. This is not always faster; select it using measurements
    /// for the actual shape/device rather than a universal performance policy.
    pub fn forward_merged_heads(
        &mut self,
        input: &st_backend_wgpu::resident_tensor::ResidentTensor,
        z_bias: Option<&st_backend_wgpu::resident_tensor::ResidentTensor>,
        pair_bias: Option<&st_backend_wgpu::resident_tensor::ResidentTensor>,
    ) -> Result<st_backend_wgpu::resident_tensor::ResidentTensor, InferenceError> {
        self.forward_impl::<true>(input, z_bias, pair_bias)
    }

    fn forward_impl<const MERGED_HEADS: bool>(
        &mut self,
        input: &st_backend_wgpu::resident_tensor::ResidentTensor,
        z_bias: Option<&st_backend_wgpu::resident_tensor::ResidentTensor>,
        pair_bias: Option<&st_backend_wgpu::resident_tensor::ResidentTensor>,
    ) -> Result<st_backend_wgpu::resident_tensor::ResidentTensor, InferenceError> {
        require_uncommitted_route()?;
        if input.layout().shape() != self.input_layout().shape() {
            return Err(InferenceError::Attention(
                "input shape differs from the frozen plan",
            ));
        }
        self.spec.validate_bias_shapes(
            z_bias.map(|b| b.layout().shape()),
            pair_bias.map(|b| b.layout().shape()),
        )?;
        for tensor in [Some(input), z_bias, pair_bias].into_iter().flatten() {
            if !tensor
                .device()
                .runtime()
                .context()
                .shares_handles_with(self.tensor_device().runtime().context())
            {
                return Err(st_backend_wgpu::resident_tensor::TensorError::DeviceMismatch.into());
            }
        }
        let [batch, heads, sequence, dim] = self.spec.query_shape();
        let projected = self
            .qkv
            .forward_tensor(input)?
            .reshape(&[batch, sequence, 3, heads, dim])?;
        let head = |index| projected.select(2, index)?.permute(&[0, 2, 1, 3]);
        let query = head(0)?;
        let keys = head(1)?;
        let values = head(2)?;
        let merged = if MERGED_HEADS {
            query.scaled_dot_attention_merged_heads(
                &keys,
                &values,
                self.spec.scale(),
                self.spec.mask(),
                z_bias,
                pair_bias,
            )?
        } else {
            query
                .scaled_dot_attention(
                    &keys,
                    &values,
                    self.spec.scale(),
                    self.spec.mask(),
                    z_bias,
                    pair_bias,
                )?
                .permute(&[0, 2, 1, 3])?
                .contiguous()?
                .reshape(&[batch, sequence, heads * dim])?
        };
        Ok(self.output.forward_tensor(&merged)?)
    }
}

#[cfg(test)]
mod tests;
