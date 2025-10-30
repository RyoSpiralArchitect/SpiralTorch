// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! WGPU softmax kernels with optional subgroup acceleration.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, Buffer,
    BufferUsages, ComputePipeline, Device, PipelineLayoutDescriptor, Queue, ShaderStages,
};

use crate::{ShaderCache, ShaderLoadError};

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
    pub _pad: u32,
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

        let subgroup = self
            .supports_subgroup
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
