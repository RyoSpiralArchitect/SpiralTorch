// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! WGPU softmax kernels with optional subgroup acceleration.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferUsages, CommandBuffer, CommandEncoder,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, Device,
    PipelineLayoutDescriptor, Queue, ShaderStages,
};

use crate::{util::device_supports_subgroup, ShaderCache, ShaderLoadError};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct Params {
    pub rows: u32,
    pub cols: u32,
    pub in_stride: u32,
    pub out_stride: u32,
    pub chimera_tile: u32,
    pub chimera_stripes: u32,
    pub flags: u32,
    pub mask_stride: u32,
}

impl Params {
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

#[derive(Debug)]
pub struct Pipelines {
    pub bind_layout: BindGroupLayout,
    pub workgroup: Arc<ComputePipeline>,
    pub subgroup: Option<Arc<ComputePipeline>>,
}

impl Pipelines {
    /// Pick the best available pipeline depending on subgroup support.
    pub fn best(&self) -> &ComputePipeline {
        self.subgroup.as_deref().unwrap_or(self.workgroup.as_ref())
    }
}

pub struct Builder<'a> {
    device: &'a Device,
    cache: ShaderCache,
    supports_subgroup: bool,
}

impl<'a> Builder<'a> {
    pub fn new(device: &'a Device, shader_dir: impl Into<PathBuf>) -> Self {
        Self::with_cache(device, ShaderCache::new(shader_dir))
    }

    pub fn with_cache(device: &'a Device, cache: ShaderCache) -> Self {
        Self {
            device,
            cache,
            supports_subgroup: false,
        }
    }

    pub fn supports_subgroup(mut self, supports: bool) -> Self {
        self.supports_subgroup = supports;
        self
    }

    pub fn cache_mut(&self) -> &ShaderCache {
        &self.cache
    }

    pub fn into_cache(self) -> ShaderCache {
        self.cache
    }

    pub fn build(self) -> Result<(Pipelines, ShaderCache), ShaderLoadError> {
        let supports_subgroup = self.supports_subgroup && device_supports_subgroup(self.device);
        let bind_layout = self
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("st.backend.softmax.bind_layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("st.backend.softmax.pipeline_layout"),
                bind_group_layouts: &[&bind_layout],
                push_constant_ranges: &[],
            });

        self.cache
            .prefetch(["softmax_workgroup.wgsl", "softmax_subgroup.wgsl"])?;

        let workgroup = self.cache.load_compute_pipeline_with_layout(
            self.device,
            "softmax_workgroup.wgsl",
            "st.softmax.workgroup",
            "main_cs",
            Some(&pipeline_layout),
        )?;

        let subgroup = supports_subgroup
            .then(|| {
                self.cache.load_compute_pipeline_with_layout(
                    self.device,
                    "softmax_subgroup.wgsl",
                    "st.softmax.subgroup",
                    "main_cs",
                    Some(&pipeline_layout),
                )
            })
            .transpose()?;

        Ok((
            Pipelines {
                bind_layout,
                workgroup,
                subgroup,
            },
            self.cache,
        ))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Dispatch {
    pub rows: u32,
}

impl Dispatch {
    pub fn workgroups(&self) -> (u32, u32, u32) {
        (self.rows, 1, 1)
    }
}

/// Buffers required to execute the softmax kernels.
pub struct DispatchArgs<'a> {
    /// Buffer containing the input values.
    pub values: &'a Buffer,
    /// Buffer receiving the primary output (probabilities or hardmax mask).
    pub output: &'a Buffer,
    /// Uniform buffer describing the tensor layout and execution flags.
    pub params: &'a Buffer,
    /// Optional buffer that stores the hardmax mask when requested.
    pub mask: Option<&'a Buffer>,
}

impl<'a> DispatchArgs<'a> {
    fn mask_binding(&self) -> &Buffer {
        self.mask.unwrap_or(self.output)
    }
}

/// Variant of the softmax pipeline to execute.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PipelineVariant {
    /// Let the backend pick the most optimised pipeline available.
    #[default]
    Best,
    /// Force the workgroup-based implementation.
    Workgroup,
    /// Force the subgroup-accelerated implementation when available.
    Subgroup,
}

impl PipelineVariant {
    fn resolve<'a>(self, pipelines: &'a Pipelines) -> (&'a ComputePipeline, PipelineVariant) {
        match self {
            PipelineVariant::Workgroup => {
                (pipelines.workgroup.as_ref(), PipelineVariant::Workgroup)
            }
            PipelineVariant::Subgroup => pipelines
                .subgroup
                .as_deref()
                .map(|pipeline| (pipeline, PipelineVariant::Subgroup))
                .unwrap_or_else(|| (pipelines.workgroup.as_ref(), PipelineVariant::Workgroup)),
            PipelineVariant::Best => pipelines
                .subgroup
                .as_deref()
                .map(|pipeline| (pipeline, PipelineVariant::Subgroup))
                .unwrap_or_else(|| (pipelines.workgroup.as_ref(), PipelineVariant::Workgroup)),
        }
    }

    fn bind_group_label(self) -> &'static str {
        match self {
            PipelineVariant::Workgroup => "st.softmax.bind_group.workgroup",
            PipelineVariant::Subgroup => "st.softmax.bind_group.subgroup",
            PipelineVariant::Best => "st.softmax.bind_group.auto",
        }
    }

    fn pass_label(self) -> &'static str {
        match self {
            PipelineVariant::Workgroup => "st.softmax.pass.workgroup",
            PipelineVariant::Subgroup => "st.softmax.pass.subgroup",
            PipelineVariant::Best => "st.softmax.pass.auto",
        }
    }
}

/// Encode the softmax dispatch into an existing command encoder.
pub fn encode_into(
    device: &Device,
    encoder: &mut CommandEncoder,
    pipelines: &Pipelines,
    args: &DispatchArgs<'_>,
    dispatch: Dispatch,
    variant: PipelineVariant,
) -> bool {
    let (workgroups_x, workgroups_y, workgroups_z) = dispatch.workgroups();
    if workgroups_x == 0 || workgroups_y == 0 || workgroups_z == 0 {
        return false;
    }

    let (pipeline, actual_variant) = variant.resolve(pipelines);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some(actual_variant.bind_group_label()),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: args.values.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: args.output.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: args.params.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: args.mask_binding().as_entire_binding(),
            },
        ],
    });

    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some(actual_variant.pass_label()),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
    true
}

/// Encode the softmax dispatch into a new command buffer.
pub fn encode(
    device: &Device,
    pipelines: &Pipelines,
    args: &DispatchArgs<'_>,
    dispatch: Dispatch,
    variant: PipelineVariant,
) -> Option<CommandBuffer> {
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("st.softmax.encoder"),
    });

    if encode_into(device, &mut encoder, pipelines, args, dispatch, variant) {
        Some(encoder.finish())
    } else {
        None
    }
}

/// Dispatch the softmax kernels directly, submitting a command buffer to the queue.
pub fn dispatch(
    device: &Device,
    queue: &Queue,
    pipelines: &Pipelines,
    args: &DispatchArgs<'_>,
    dispatch: Dispatch,
    variant: PipelineVariant,
) -> bool {
    if let Some(cmd) = encode(device, pipelines, args, dispatch, variant) {
        queue.submit(Some(cmd));
        true
    } else {
        false
    }
}

pub fn upload_params(device: &Device, queue: &Queue, params: &Params) -> Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("st.backend.softmax.params"),
        size: std::mem::size_of::<Params>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, params.as_bytes());
    buffer
}

pub fn create_pipelines(
    device: &Device,
    shader_dir: impl AsRef<Path>,
    supports_subgroup: bool,
) -> Result<Pipelines, ShaderLoadError> {
    let supports_subgroup = supports_subgroup && device_supports_subgroup(device);
    let (pipelines, _) = Builder::new(device, shader_dir.as_ref().to_path_buf())
        .supports_subgroup(supports_subgroup)
        .build()?;
    Ok(pipelines)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_layout_is_32_bytes() {
        assert_eq!(std::mem::size_of::<Params>(), 32);
    }

    #[test]
    fn dispatch_matches_rows() {
        let dispatch = Dispatch { rows: 42 };
        assert_eq!(dispatch.workgroups(), (42, 1, 1));
    }
}
