// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Fused scaled dot-product attention with online softmax normalisation.
//!
//! The planner emits a single WGSL compute kernel that evaluates Q·Kᵀ, applies
//! bias/mask terms, performs a numerically-stable softmax reduction, and mixes
//! the values — all without materialising intermediate logits. The online
//! softmax fusion keeps the kernel memory-bound rather than launch-bound.

use std::path::PathBuf;

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    BufferBindingType, ComputePipeline, ComputePipelineDescriptor, Device, PipelineLayout,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderStages,
};

use crate::{util::read_wgsl, ShaderLoadError};

const TEMPLATE_FILE: &str = "fused_attention_online.wgsl";

/// Mask controlling optional inputs consumed by the fused kernel.
pub const FLAG_USE_Z_BIAS: u32 = 1 << 0;
/// Mask flag enabling per-query/key attention bias.
pub const FLAG_USE_ATTN_BIAS: u32 = 1 << 1;

/// Runtime parameters consumed by the fused attention shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Params {
    pub contexts: u32,
    pub sequence: u32,
    pub head_dim: u32,
    pub flags: u32,
    pub scale: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

impl Params {
    /// Cast the params into a raw byte slice suitable for uniform uploads.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

/// Supported accumulator precision for the fused kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccumulatorPrecision {
    /// Pure 32-bit float accumulation.
    F32,
    /// Half precision inputs accumulated into f32.
    F16AccF32,
}

impl Default for AccumulatorPrecision {
    fn default() -> Self {
        Self::F32
    }
}

/// Errors emitted while constructing a fused attention kernel plan.
#[derive(Debug, Error)]
pub enum PlanError {
    /// Invalid tiling or kernel geometry provided by the caller.
    #[error("invalid fused attention tile configuration: {0}")]
    InvalidTile(&'static str),
    /// Workgroup sizes outside the portable range (1..=256) are rejected.
    #[error("workgroup size {size} is unsupported; must be within 1..=256")]
    InvalidWorkgroup { size: u32 },
    /// Head dimensions larger than the templated scratch buffers cannot run.
    #[error("head dimension {head_dim} exceeds templated maximum {max_head_dim}")]
    HeadDimExceeded { head_dim: u32, max_head_dim: u32 },
    /// Accumulator precision that the kernel does not yet support.
    #[error("dtype {dtype:?} is not implemented for fused attention")]
    UnsupportedDType { dtype: AccumulatorPrecision },
    /// Wrapper around shader loading failures.
    #[error(transparent)]
    Shader(#[from] ShaderLoadError),
}

/// DSL entry point that describes how to specialise the fused kernel.
#[derive(Debug, Clone)]
pub struct Plan {
    workgroup_size: u32,
    max_head_dim: u32,
    tile_queries: u32,
    _tile_keys: u32,
    accumulator: AccumulatorPrecision,
}

impl Plan {
    /// Start building a fused attention plan with conservative defaults.
    pub fn new() -> Self {
        Self {
            workgroup_size: 64,
            max_head_dim: 256,
            tile_queries: 1,
            _tile_keys: 1,
            accumulator: AccumulatorPrecision::F32,
        }
    }

    /// Configure the tile geometry. `tm` controls queries per workgroup, `tn`
    /// controls key staging (currently informational), and `tk` sets the
    /// maximum head dimension baked into the shader.
    pub fn tile(mut self, tm: u32, tn: u32, tk: u32) -> Self {
        self.tile_queries = tm;
        self._tile_keys = tn;
        self.max_head_dim = tk.max(1);
        self
    }

    /// Override the workgroup size used to cooperatively reduce dot products.
    pub fn workgroup_size(mut self, size: u32) -> Self {
        self.workgroup_size = size;
        self
    }

    /// Select the accumulator precision used by the fused kernel.
    pub fn dtype(mut self, precision: AccumulatorPrecision) -> Self {
        self.accumulator = precision;
        self
    }

    /// Materialise the WGSL shader and compute pipeline for the described plan.
    pub fn build(
        self,
        device: &Device,
        shader_dir: impl Into<PathBuf>,
    ) -> Result<Kernel, PlanError> {
        self.validate()?;

        let shader_source = self.generate_shader(shader_dir.into())?;
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("st.backend.fused_attention"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("st.backend.fused_attention.bind_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("st.backend.fused_attention.pipeline_layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("st.backend.fused_attention.pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "main",
        });

        Ok(Kernel {
            bind_layout,
            pipeline_layout,
            pipeline,
            workgroup_size: self.workgroup_size,
            max_head_dim: self.max_head_dim,
        })
    }

    fn validate(&self) -> Result<(), PlanError> {
        if self.tile_queries != 1 {
            return Err(PlanError::InvalidTile(
                "current fused kernel requires tile(tm) == 1 (one query per workgroup)",
            ));
        }
        if !(1..=256).contains(&self.workgroup_size) {
            return Err(PlanError::InvalidWorkgroup {
                size: self.workgroup_size,
            });
        }
        if self.max_head_dim == 0 {
            return Err(PlanError::InvalidTile("tile(tk) must be positive"));
        }
        if self.accumulator != AccumulatorPrecision::F32 {
            return Err(PlanError::UnsupportedDType {
                dtype: self.accumulator,
            });
        }
        Ok(())
    }

    fn generate_shader(&self, shader_dir: PathBuf) -> Result<String, PlanError> {
        let template = read_wgsl(shader_dir, TEMPLATE_FILE)?;
        Ok(template
            .replace("{WORKGROUP_SIZE}", &self.workgroup_size.to_string())
            .replace("{MAX_HEAD_DIM}", &self.max_head_dim.to_string()))
    }
}

/// Finalised fused attention kernel and associated metadata.
#[derive(Debug)]
pub struct Kernel {
    bind_layout: BindGroupLayout,
    pipeline_layout: PipelineLayout,
    pipeline: ComputePipeline,
    workgroup_size: u32,
    max_head_dim: u32,
}

impl Kernel {
    /// Accessor for the bind group layout.
    pub fn bind_layout(&self) -> &BindGroupLayout {
        &self.bind_layout
    }

    /// Accessor for the pipeline layout used during dispatch construction.
    pub fn pipeline_layout(&self) -> &PipelineLayout {
        &self.pipeline_layout
    }

    /// Underlying compute pipeline handle.
    pub fn pipeline(&self) -> &ComputePipeline {
        &self.pipeline
    }

    /// Maximum head dimension baked into the current shader variant.
    pub fn max_head_dim(&self) -> u32 {
        self.max_head_dim
    }

    /// Cooperative workgroup width (threads per query) used by the kernel.
    pub fn workgroup_size(&self) -> u32 {
        self.workgroup_size
    }

    /// Validate that the requested head dimension fits inside the kernel.
    pub fn ensure_head_dim(&self, head_dim: u32) -> Result<(), PlanError> {
        if head_dim > self.max_head_dim {
            return Err(PlanError::HeadDimExceeded {
                head_dim,
                max_head_dim: self.max_head_dim,
            });
        }
        Ok(())
    }

    /// Compute dispatch dimensions for the provided context and sequence size.
    pub fn dispatch_dims(&self, contexts: u32, sequence: u32) -> (u32, u32, u32) {
        (sequence, contexts, 1)
    }
}

/// Convenience constructor mirroring the DSL-style entry point discussed in the
/// planner notes. Example: `fused_attention().tile(1, 64, 256).build(...)`.
pub fn fused_attention() -> Plan {
    Plan::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn params_layout_is_32_bytes() {
        assert_eq!(std::mem::size_of::<Params>(), 32);
    }

    #[test]
    fn rejects_multiple_query_tiles() {
        let shader_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders");
        let plan = Plan::new().tile(2, 1, 128);
        let err = plan.build(&dummy_device(), shader_dir).unwrap_err();
        assert!(matches!(err, PlanError::InvalidTile(_)));
    }

    #[test]
    fn rejects_unsupported_dtype() {
        let shader_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders");
        let err = Plan::new()
            .dtype(AccumulatorPrecision::F16AccF32)
            .build(&dummy_device(), shader_dir)
            .unwrap_err();
        assert!(matches!(err, PlanError::UnsupportedDType { .. }));
    }

    fn dummy_device() -> Device {
        // Create a minimal headless device for validation-only tests. The
        // fallback adapter is requested so CI environments without a hardware
        // GPU can still instantiate the pipeline layout builders.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(async {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: None,
                    force_fallback_adapter: true,
                })
                .expect("no WGPU adapter available")
        });
        pollster::block_on(async move {
            adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .expect("failed to request dummy device")
        })
    }
}
