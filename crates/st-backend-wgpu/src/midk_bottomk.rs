// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Pipeline assembly helpers for MidK/BottomK compaction kernels.

use std::path::PathBuf;

use bytemuck::{Pod, Zeroable};
use wgpu::{
    util::DeviceExt, BindGroupDescriptor, BindGroupEntry, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, Device, Queue,
};

use crate::{ShaderCache, ShaderLoadError};

#[derive(Debug)]
pub struct Pipelines {
    pub scan_tiles: ComputePipeline,
    pub row_prefix: ComputePipeline,
    pub apply_fallback: ComputePipeline,
    pub apply_subgroup: Option<ComputePipeline>,
    pub apply_subgroup_v2: Option<ComputePipeline>,
}

impl Pipelines {
    /// Pick the best available subgroup variant, falling back to the portable pipeline.
    pub fn best_subgroup(&self) -> Option<&ComputePipeline> {
        self.apply_subgroup_v2
            .as_ref()
            .or(self.apply_subgroup.as_ref())
    }
}

/// Selects which compaction flavour should be emitted by the shaders.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    /// MidK selects the middle band of values between two thresholds.
    MidK,
    /// BottomK selects the lowest band of values beneath a threshold.
    BottomK,
}

impl Kind {
    fn as_uniform(self) -> u32 {
        match self {
            Kind::MidK => 0,
            Kind::BottomK => 1,
        }
    }
}

/// Strategy used to choose the apply stage pipeline.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ApplyStrategy {
    /// Pick the best available subgroup pipeline, falling back when unavailable.
    #[default]
    Auto,
    /// Force the portable workgroup fallback implementation.
    Fallback,
    /// Prefer the legacy subgroup implementation when present.
    Subgroup,
    /// Prefer the enhanced subgroup implementation when present.
    SubgroupV2,
}

/// Arguments required to enqueue a MidK/BottomK compaction pass.
pub struct DispatchArgs<'a> {
    /// Number of rows to process.
    pub rows: u32,
    /// Number of columns within each row.
    pub cols: u32,
    /// Stride between rows in the input/output buffers.
    pub row_stride: u32,
    /// Whether to run the kernels in MidK or BottomK mode.
    pub kind: Kind,
    /// Buffer containing the input values that will be compacted.
    pub values: &'a wgpu::Buffer,
    /// Buffer containing the predicate mask (non-zero entries are kept).
    pub mask: &'a wgpu::Buffer,
    /// Buffer that receives the per-row output counts written by the scan stage.
    pub out_positions: &'a wgpu::Buffer,
    /// Buffer that receives the compacted values.
    pub out_values: &'a wgpu::Buffer,
    /// Scratch buffer that stores per-tile prefix totals.
    pub prefix: &'a wgpu::Buffer,
}

impl<'a> DispatchArgs<'a> {
    fn tiles_x(&self) -> u32 {
        tiles_for_cols(self.cols)
    }
}

/// Dispatch the MidK/BottomK compaction pipeline family.
///
/// The helper encodes the three stages required by the compaction kernels
/// (tile scan, row prefix, apply) and submits them to the provided queue.
/// The caller is responsible for ensuring the buffers have the correct size:
///
/// * `values`: `rows * row_stride` elements.
/// * `mask`: `rows * row_stride` entries, interpreted as `u32` flags.
/// * `out_positions`: `rows` atomic counters (written as `u32`).
/// * `out_values`: `rows * row_stride` elements to receive compacted values.
/// * `prefix`: `rows * tiles_x` entries (where `tiles_x = ceil(cols / 256)`).
pub fn dispatch(
    device: &Device,
    queue: &Queue,
    pipelines: &Pipelines,
    args: DispatchArgs<'_>,
    strategy: ApplyStrategy,
) {
    let tiles_x = args.tiles_x();
    if args.rows == 0 || tiles_x == 0 {
        return;
    }

    let params = ParamsUniform {
        rows: args.rows,
        cols: args.cols,
        row_stride: args.row_stride,
        kind: args.kind.as_uniform(),
        tiles_x,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.midk_bottomk.compaction.params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let select_apply = |strategy: ApplyStrategy| match strategy {
        ApplyStrategy::Fallback => &pipelines.apply_fallback,
        ApplyStrategy::Subgroup => pipelines
            .apply_subgroup
            .as_ref()
            .unwrap_or(&pipelines.apply_fallback),
        ApplyStrategy::SubgroupV2 => pipelines
            .apply_subgroup_v2
            .as_ref()
            .or(pipelines.apply_subgroup.as_ref())
            .unwrap_or(&pipelines.apply_fallback),
        ApplyStrategy::Auto => pipelines
            .best_subgroup()
            .unwrap_or(&pipelines.apply_fallback),
    };

    let apply_pipeline = select_apply(strategy);

    let encoder_label = match args.kind {
        Kind::MidK => "st.midk.compaction",
        Kind::BottomK => "st.bottomk.compaction",
    };

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some(encoder_label),
    });

    encode_stage(
        device,
        &mut encoder,
        &pipelines.scan_tiles,
        &params_buffer,
        &args,
        tiles_x,
        "scan",
    );
    encode_stage(
        device,
        &mut encoder,
        &pipelines.row_prefix,
        &params_buffer,
        &args,
        tiles_x,
        "row_prefix",
    );
    encode_stage(
        device,
        &mut encoder,
        apply_pipeline,
        &params_buffer,
        &args,
        tiles_x,
        "apply",
    );

    queue.submit(Some(encoder.finish()));
}

fn encode_stage(
    device: &Device,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &ComputePipeline,
    params: &wgpu::Buffer,
    args: &DispatchArgs<'_>,
    tiles_x: u32,
    stage: &str,
) {
    let layout = pipeline.get_bind_group_layout(0);
    let bind_group_label = match stage {
        "scan" => "st.midk_bottomk.bind_group.scan",
        "row_prefix" => "st.midk_bottomk.bind_group.row_prefix",
        "apply" => "st.midk_bottomk.bind_group.apply",
        _ => "st.midk_bottomk.bind_group",
    };
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some(bind_group_label),
        layout: &layout,
        entries: &bind_group_entries(args, params),
    });

    let pass_label = match stage {
        "scan" => "st.midk_bottomk.pass.scan",
        "row_prefix" => "st.midk_bottomk.pass.row_prefix",
        "apply" => "st.midk_bottomk.pass.apply",
        _ => "st.midk_bottomk.pass",
    };
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some(pass_label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(tiles_x, args.rows, 1);
}

fn bind_group_entries<'a>(
    args: &'a DispatchArgs<'a>,
    params: &'a wgpu::Buffer,
) -> [BindGroupEntry<'a>; 6] {
    [
        BindGroupEntry {
            binding: 0,
            resource: args.values.as_entire_binding(),
        },
        BindGroupEntry {
            binding: 1,
            resource: args.mask.as_entire_binding(),
        },
        BindGroupEntry {
            binding: 2,
            resource: args.out_positions.as_entire_binding(),
        },
        BindGroupEntry {
            binding: 3,
            resource: args.out_values.as_entire_binding(),
        },
        BindGroupEntry {
            binding: 4,
            resource: params.as_entire_binding(),
        },
        BindGroupEntry {
            binding: 5,
            resource: args.prefix.as_entire_binding(),
        },
    ]
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ParamsUniform {
    rows: u32,
    cols: u32,
    row_stride: u32,
    kind: u32,
    tiles_x: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

fn tiles_for_cols(cols: u32) -> u32 {
    if cols == 0 {
        0
    } else {
        (cols + 255) / 256
    }
}

#[cfg(test)]
mod tests {
    use super::tiles_for_cols;

    #[test]
    fn tiles_for_cols_rounds_up() {
        assert_eq!(tiles_for_cols(1), 1);
        assert_eq!(tiles_for_cols(256), 1);
        assert_eq!(tiles_for_cols(257), 2);
    }

    #[test]
    fn tiles_for_cols_handles_zero() {
        assert_eq!(tiles_for_cols(0), 0);
    }
}

pub struct Builder<'a> {
    device: &'a Device,
    cache: ShaderCache,
    supports_subgroup: bool,
    include_subgroup_v1: bool,
    include_subgroup_v2: bool,
}

impl<'a> Builder<'a> {
    /// Create a builder rooted at the provided shader directory.
    pub fn new(device: &'a Device, shader_dir: impl Into<PathBuf>) -> Self {
        Self::with_cache(device, ShaderCache::new(shader_dir))
    }

    /// Create a builder from an existing cache instance.
    pub fn with_cache(device: &'a Device, cache: ShaderCache) -> Self {
        Self {
            device,
            cache,
            supports_subgroup: false,
            include_subgroup_v1: false,
            include_subgroup_v2: false,
        }
    }

    /// Toggle support for subgroup pipelines.
    pub fn supports_subgroup(mut self, supports: bool) -> Self {
        self.supports_subgroup = supports;
        self
    }

    /// Request the legacy subgroup compaction path.
    pub fn include_subgroup(mut self) -> Self {
        self.include_subgroup_v1 = true;
        self
    }

    /// Request the enhanced subgroup compaction path introduced in v1.8.5.
    pub fn include_subgroup_v2(mut self) -> Self {
        self.include_subgroup_v2 = true;
        self
    }

    /// Borrow the underlying cache for prefetching or manual shader control.
    pub fn cache_mut(&mut self) -> &mut ShaderCache {
        &mut self.cache
    }

    /// Consume the builder and return the cache without building pipelines.
    pub fn into_cache(self) -> ShaderCache {
        self.cache
    }

    fn assemble(&mut self) -> Result<Pipelines, ShaderLoadError> {
        let scan_tiles = self.cache.load_compute_pipeline(
            self.device,
            "midk_bottomk_compaction.wgsl",
            "midk_compact_scan_tiles",
            "midk_compact_scan_tiles",
        )?;

        let row_prefix = self.cache.load_compute_pipeline(
            self.device,
            "midk_bottomk_compaction.wgsl",
            "midk_compact_row_prefix",
            "midk_compact_row_prefix",
        )?;

        let apply_fallback = self.cache.load_compute_pipeline(
            self.device,
            "midk_bottomk_compaction.wgsl",
            "midk_compact_apply",
            "midk_compact_apply",
        )?;

        let apply_subgroup = (self.supports_subgroup && self.include_subgroup_v1)
            .then(|| {
                self.cache.load_compute_pipeline(
                    self.device,
                    "midk_bottomk_compaction.wgsl",
                    "midk_compact_apply_sg",
                    "midk_compact_apply_sg",
                )
            })
            .transpose()?;

        let apply_subgroup_v2 = (self.supports_subgroup && self.include_subgroup_v2)
            .then(|| {
                self.cache.load_compute_pipeline(
                    self.device,
                    "midk_bottomk_compaction.wgsl",
                    "midk_compact_apply_sg2",
                    "midk_compact_apply_sg2",
                )
            })
            .transpose()?;

        Ok(Pipelines {
            scan_tiles,
            row_prefix,
            apply_fallback,
            apply_subgroup,
            apply_subgroup_v2,
        })
    }

    /// Build the requested pipelines.
    pub fn build(mut self) -> Result<Pipelines, ShaderLoadError> {
        self.assemble()
    }

    /// Build pipelines while returning the [`ShaderCache`] for reuse.
    pub fn build_with_cache(mut self) -> Result<(Pipelines, ShaderCache), ShaderLoadError> {
        let pipelines = self.assemble()?;
        Ok((pipelines, self.cache))
    }
}

pub fn create_pipelines(
    device: &Device,
    shader_dir: &str,
    supports_subgroup: bool,
    include_v1: bool,
    include_v2: bool,
) -> Result<Pipelines, ShaderLoadError> {
    let builder = Builder::new(device, shader_dir).supports_subgroup(supports_subgroup);
    let builder = if include_v1 {
        builder.include_subgroup()
    } else {
        builder
    };
    let builder = if include_v2 {
        builder.include_subgroup_v2()
    } else {
        builder
    };
    builder.build()
}
