// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Helper utilities for loading and dispatching the fused GELU backward shaders.
//!
//! The WGSL implementation ships in two passes: a fused backward pass that
//! generates `gZ`, updates the residual gradient, and accumulates bias partials
//! per workgroup tile, followed by a column-wise reduction that finalises `db`.

use std::cmp;
use std::path::{Path, PathBuf};

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, Buffer,
    BufferUsages, ComputePipeline, Device, PipelineLayoutDescriptor, Queue, ShaderStages,
};

use crate::{ShaderCache, ShaderLoadError};

/// Default workgroup geometry matching the WGSL overrides.
pub const DEFAULT_WG_ROWS: u32 = 16;
pub const DEFAULT_WG_COLS: u32 = 16;
pub const DEFAULT_REDUCE_WG: u32 = 256;

/// Parameters describing the tile geometry for the fused and reduction passes.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub wg_rows: u32,
    pub wg_cols: u32,
    pub reduce_wg: u32,
}

impl Geometry {
    /// Create a new geometry description, falling back to the WGSL defaults when zero.
    pub fn new(wg_rows: u32, wg_cols: u32, reduce_wg: u32) -> Self {
        Self {
            wg_rows: if wg_rows == 0 {
                DEFAULT_WG_ROWS
            } else {
                wg_rows
            },
            wg_cols: if wg_cols == 0 {
                DEFAULT_WG_COLS
            } else {
                wg_cols
            },
            reduce_wg: if reduce_wg == 0 {
                DEFAULT_REDUCE_WG
            } else {
                reduce_wg
            },
        }
    }

    /// Compute the number of tiles required to cover a `[B, O]` matrix.
    pub fn tiles(&self, batch: u32, cols: u32) -> (u32, u32) {
        (ceil_div(cols, self.wg_cols), ceil_div(batch, self.wg_rows))
    }

    /// Return the dispatch dimensions for the fused kernel.
    pub fn fused_dispatch(&self, batch: u32, cols: u32) -> (u32, u32, u32) {
        let (tiles_x, tiles_y) = self.tiles(batch, cols);
        (tiles_x, tiles_y, 1)
    }

    /// Return the dispatch dimensions for the reduction kernel.
    pub fn reduce_dispatch(&self, cols: u32) -> (u32, u32, u32) {
        (ceil_div(cols, self.reduce_wg), 1, 1)
    }

    /// Compute the number of partial elements produced by the fused pass.
    pub fn partial_len(&self, batch: u32, cols: u32) -> u64 {
        let (tiles_x, tiles_y) = self.tiles(batch, cols);
        u64::from(tiles_x) * u64::from(tiles_y) * u64::from(self.wg_cols)
    }

    /// Select tile geometry tuned for the provided [`Device`]'s limits.
    pub fn autotune(device: &Device) -> Self {
        Self::autotune_with_limits(&device.limits())
    }

    /// Select tile geometry tuned for the provided [`wgpu::Limits`].
    pub fn autotune_with_limits(limits: &wgpu::Limits) -> Self {
        const TILE_CANDIDATES: &[(u32, u32)] = &[
            (64, 4),
            (48, 8),
            (32, 8),
            (24, 12),
            (16, 16),
            (12, 24),
            (8, 32),
            (4, 64),
        ];
        const REDUCE_CANDIDATES: &[u32] = &[512, 384, 256, 192, 128, 96, 64, 48, 32, 16];

        let mut geometry = Self::default();
        let mut assigned = false;
        for &(cols, rows) in TILE_CANDIDATES {
            if cols <= limits.max_compute_workgroup_size_x
                && rows <= limits.max_compute_workgroup_size_y
                && cols.saturating_mul(rows) <= limits.max_compute_invocations_per_workgroup
            {
                geometry.wg_cols = cols;
                geometry.wg_rows = rows;
                assigned = true;
                break;
            }
        }

        let max_wg_x = limits.max_compute_workgroup_size_x.max(1);
        let max_inv = limits.max_compute_invocations_per_workgroup.max(1);

        if !assigned {
            // The default tile might violate device limits (e.g. very small workgroup sizes).
            // Clamp the tile to the maximum supported dimensions while keeping the product
            // under the invocation bound.
            let mut cols = cmp::min(DEFAULT_WG_COLS, max_wg_x);
            cols = cmp::min(cols, max_inv);
            cols = cols.max(1);
            let mut rows = max_inv / cols;
            rows = cmp::min(rows.max(1), limits.max_compute_workgroup_size_y.max(1));
            geometry.wg_cols = cols;
            geometry.wg_rows = rows;
        }

        for &candidate in REDUCE_CANDIDATES {
            if candidate <= max_inv {
                geometry.reduce_wg = candidate;
                return geometry;
            }
        }

        geometry.reduce_wg = cmp::min(DEFAULT_REDUCE_WG, max_inv).max(1);
        geometry
    }
}

impl Default for Geometry {
    fn default() -> Self {
        Self {
            wg_rows: DEFAULT_WG_ROWS,
            wg_cols: DEFAULT_WG_COLS,
            reduce_wg: DEFAULT_REDUCE_WG,
        }
    }
}

fn ceil_div(lhs: u32, rhs: u32) -> u32 {
    assert!(rhs > 0, "division by zero in ceil_div");
    (lhs + rhs - 1) / rhs
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct FusedUniforms {
    pub batch: u32,
    pub cols: u32,
    pub stride: u32,
    pub num_wg_x: u32,
    pub num_wg_y: u32,
    pub add_residual: u32,
    _padding: [u32; 2],
}

impl FusedUniforms {
    pub fn new(
        batch: u32,
        cols: u32,
        stride: u32,
        num_wg_x: u32,
        num_wg_y: u32,
        add_residual: bool,
    ) -> Self {
        Self {
            batch,
            cols,
            stride,
            num_wg_x,
            num_wg_y,
            add_residual: add_residual as u32,
            _padding: [0, 0],
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct ReduceUniforms {
    pub cols: u32,
    pub num_wg_x: u32,
    pub num_wg_y: u32,
    _padding: u32,
}

impl ReduceUniforms {
    pub fn new(cols: u32, num_wg_x: u32, num_wg_y: u32) -> Self {
        Self {
            cols,
            num_wg_x,
            num_wg_y,
            _padding: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

#[derive(Debug)]
pub struct Pipelines {
    pub fused_bind_layout: BindGroupLayout,
    pub reduce_bind_layout: BindGroupLayout,
    pub fused: ComputePipeline,
    pub reduce: ComputePipeline,
    pub geometry: Geometry,
}

impl Pipelines {
    pub fn fused_dispatch<'a>(&'a self) -> &'a ComputePipeline {
        &self.fused
    }

    pub fn reduce_dispatch<'a>(&'a self) -> &'a ComputePipeline {
        &self.reduce
    }

    pub fn geometry(&self) -> Geometry {
        self.geometry
    }
}

pub struct Builder<'a> {
    device: &'a Device,
    cache: ShaderCache,
}

impl<'a> Builder<'a> {
    pub fn new(device: &'a Device, shader_dir: impl Into<PathBuf>) -> Self {
        Self::with_cache(device, ShaderCache::new(shader_dir))
    }

    pub fn with_cache(device: &'a Device, cache: ShaderCache) -> Self {
        Self { device, cache }
    }

    pub fn cache_mut(&mut self) -> &mut ShaderCache {
        &mut self.cache
    }

    pub fn into_cache(self) -> ShaderCache {
        self.cache
    }

    pub fn build(self) -> Result<(Pipelines, ShaderCache), ShaderLoadError> {
        let device = self.device;
        let geometry = Geometry::autotune(device);
        self.build_with_geometry(geometry)
    }

    pub fn build_with_geometry(
        mut self,
        geometry: Geometry,
    ) -> Result<(Pipelines, ShaderCache), ShaderLoadError> {
        let fused_bind_layout = self
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("st.backend.gelu_back.fused.bind_layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                    BindGroupLayoutEntry {
                        binding: 4,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 5,
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

        let reduce_bind_layout = self
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("st.backend.gelu_back.reduce.bind_layout"),
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

        let fused_layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("st.backend.gelu_back.fused.pipeline_layout"),
                bind_group_layouts: &[&fused_bind_layout],
                push_constant_ranges: &[],
            });

        let reduce_layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("st.backend.gelu_back.reduce.pipeline_layout"),
                bind_group_layouts: &[&reduce_bind_layout],
                push_constant_ranges: &[],
            });

        self.cache
            .prefetch(["fused_gelu_back.wgsl", "reduce_db.wgsl"])?;

        let fused_label = format!(
            "st.backend.gelu_back.fused.rows{}_cols{}",
            geometry.wg_rows, geometry.wg_cols
        );
        let reduce_label = format!(
            "st.backend.gelu_back.reduce.cols{}_wg{}",
            geometry.wg_cols, geometry.reduce_wg
        );

        let fused = self.cache.load_compute_pipeline_with_layout_and_overrides(
            self.device,
            "fused_gelu_back.wgsl",
            &fused_label,
            "main",
            Some(&fused_layout),
            &[("WG_ROWS", geometry.wg_rows), ("WG_COLS", geometry.wg_cols)],
        )?;

        let reduce = self.cache.load_compute_pipeline_with_layout_and_overrides(
            self.device,
            "reduce_db.wgsl",
            &reduce_label,
            "reduce",
            Some(&reduce_layout),
            &[
                ("WG_COLS", geometry.wg_cols),
                ("REDUCE_WG", geometry.reduce_wg),
            ],
        )?;

        Ok((
            Pipelines {
                fused_bind_layout,
                reduce_bind_layout,
                fused,
                reduce,
                geometry,
            },
            self.cache,
        ))
    }
}

pub fn create_pipelines(
    device: &Device,
    shader_dir: impl AsRef<Path>,
) -> Result<Pipelines, ShaderLoadError> {
    let (pipelines, _) = Builder::new(device, shader_dir.as_ref().to_path_buf()).build()?;
    Ok(pipelines)
}

pub fn create_pipelines_with_geometry(
    device: &Device,
    shader_dir: impl AsRef<Path>,
    geometry: Geometry,
) -> Result<Pipelines, ShaderLoadError> {
    let (pipelines, _) =
        Builder::new(device, shader_dir.as_ref().to_path_buf()).build_with_geometry(geometry)?;
    Ok(pipelines)
}

pub fn upload_fused_uniforms(device: &Device, queue: &Queue, uniforms: &FusedUniforms) -> Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("st.backend.gelu_back.fused.uniforms"),
        size: std::mem::size_of::<FusedUniforms>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, uniforms.as_bytes());
    buffer
}

pub fn upload_reduce_uniforms(device: &Device, queue: &Queue, uniforms: &ReduceUniforms) -> Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("st.backend.gelu_back.reduce.uniforms"),
        size: std::mem::size_of::<ReduceUniforms>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, uniforms.as_bytes());
    buffer
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::Limits;

    #[test]
    fn fused_uniform_size() {
        assert_eq!(std::mem::size_of::<FusedUniforms>(), 32);
    }

    #[test]
    fn reduce_uniform_size() {
        assert_eq!(std::mem::size_of::<ReduceUniforms>(), 16);
    }

    #[test]
    fn geometry_defaults_match_overrides() {
        let geom = Geometry::default();
        assert_eq!(geom.wg_rows, DEFAULT_WG_ROWS);
        assert_eq!(geom.wg_cols, DEFAULT_WG_COLS);
        assert_eq!(geom.reduce_wg, DEFAULT_REDUCE_WG);
    }

    #[test]
    fn tiles_and_dispatch_match_reference() {
        let geom = Geometry::default();
        let batch = 33;
        let cols = 65;
        let (tiles_x, tiles_y) = geom.tiles(batch, cols);
        assert_eq!(tiles_x, 5);
        assert_eq!(tiles_y, 3);
        assert_eq!(geom.fused_dispatch(batch, cols), (tiles_x, tiles_y, 1));
        assert_eq!(geom.reduce_dispatch(cols), (1, 1, 1));
        assert_eq!(
            geom.partial_len(batch, cols),
            5 * 3 * u64::from(DEFAULT_WG_COLS)
        );
    }

    #[test]
    fn geometry_autotune_prefers_wide_tiles_when_supported() {
        let mut limits = Limits::downlevel_defaults();
        limits.max_compute_workgroup_size_x = 64;
        limits.max_compute_workgroup_size_y = 8;
        limits.max_compute_invocations_per_workgroup = 512;
        let geom = Geometry::autotune_with_limits(&limits);
        assert_eq!(geom.wg_cols, 64);
        assert_eq!(geom.wg_rows, 4);
        assert!(geom.reduce_wg >= 256);
    }

    #[test]
    fn geometry_autotune_clamps_to_tiny_limits() {
        let mut limits = Limits::downlevel_defaults();
        limits.max_compute_workgroup_size_x = 8;
        limits.max_compute_workgroup_size_y = 4;
        limits.max_compute_invocations_per_workgroup = 32;
        let geom = Geometry::autotune_with_limits(&limits);
        assert!(geom.wg_cols <= 8);
        assert!(geom.wg_rows <= 4);
        assert!(geom.wg_cols * geom.wg_rows <= 32);
        assert!(geom.reduce_wg <= 8);
    }
}
