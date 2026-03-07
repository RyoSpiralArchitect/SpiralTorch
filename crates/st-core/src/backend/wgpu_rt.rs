//! WGPU runtime dispatch helpers for Rank-K kernels.
//!
//! This module is intentionally small and feature-gated behind `wgpu-rt`.
//! Callers must install a [`WgpuCtx`] once, then dispatch kernels by passing
//! concrete WGPU buffers.
//!
//! Note: `st-core::backend::wgpu_exec` currently exposes a plan-only executor
//! surface (for parity with HIP/CUDA stubs). The plan-only dispatch entry points
//! in this module return an error that tells callers to use the buffer-based
//! APIs below.

#![cfg(all(feature = "wgpu", feature = "wgpu-rt"))]

use crate::ops::compaction::CompactionPlan;
use crate::ops::rank_entry::RankPlan;
use once_cell::sync::OnceCell;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub struct WgpuCtx {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    topk_heap_pl: OnceCell<wgpu::ComputePipeline>,
    topk_heap_sgintrin_pl: OnceCell<wgpu::ComputePipeline>,
    topk_bit_pl: OnceCell<wgpu::ComputePipeline>,
    topk_wg_pl: OnceCell<wgpu::ComputePipeline>,
    layout_topk: OnceCell<wgpu::BindGroupLayout>,
    compact_scan_pl: OnceCell<wgpu::ComputePipeline>,
    compact_row_prefix_pl: OnceCell<wgpu::ComputePipeline>,
    compact_apply_sg_pl: OnceCell<wgpu::ComputePipeline>,
    compact_apply_pl: OnceCell<wgpu::ComputePipeline>,
    layout_compaction: OnceCell<wgpu::BindGroupLayout>,
}

impl WgpuCtx {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            device,
            queue,
            topk_heap_pl: OnceCell::new(),
            topk_heap_sgintrin_pl: OnceCell::new(),
            topk_bit_pl: OnceCell::new(),
            topk_wg_pl: OnceCell::new(),
            layout_topk: OnceCell::new(),
            compact_scan_pl: OnceCell::new(),
            compact_row_prefix_pl: OnceCell::new(),
            compact_apply_sg_pl: OnceCell::new(),
            compact_apply_pl: OnceCell::new(),
            layout_compaction: OnceCell::new(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TopkParams {
    rows: u32,
    cols: u32,
    k: u32,
    row_stride: u32,
    k_lane: u32,
    tile_cols: u32,
    radix: u32,
    segments: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CompactionParams {
    rows: u32,
    cols: u32,
    row_stride: u32,
    kind: u32,
    tiles_x: u32,
    _pad: u32,
    _pad2: u32,
    _pad3: u32,
}

static CTX: OnceCell<Arc<WgpuCtx>> = OnceCell::new();

pub fn install_ctx(ctx: Arc<WgpuCtx>) {
    let _ = CTX.set(ctx);
}

fn ctx() -> Result<Arc<WgpuCtx>, String> {
    CTX.get()
        .cloned()
        .ok_or_else(|| "WGPU ctx not installed".to_string())
}

const WGSL_RANKK: &str = include_str!("wgpu_kernels_rankk.wgsl");

fn module(ctx: &WgpuCtx, src: &'static str) -> wgpu::ShaderModule {
    ctx.device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("st.rankk.wgsl"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        })
}

fn bge_storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bge_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn ensure_layout_topk(ctx: &WgpuCtx) -> &wgpu::BindGroupLayout {
    ctx.layout_topk.get_or_init(|| {
        ctx.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("st.rankk.layout.topk"),
                entries: &[
                    bge_storage(0, true),
                    bge_storage(1, false),
                    bge_storage(2, false),
                    bge_uniform(3),
                ],
            })
    })
}

fn ensure_layout_compaction(ctx: &WgpuCtx) -> &wgpu::BindGroupLayout {
    ctx.layout_compaction.get_or_init(|| {
        ctx.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("st.rankk.layout.compaction"),
                entries: &[
                    bge_storage(0, true),
                    bge_storage(1, true),
                    bge_storage(2, false),
                    bge_storage(3, false),
                    bge_storage(4, false),
                    bge_uniform(5),
                    bge_storage(6, false),
                ],
            })
    })
}

fn pl(ctx: &WgpuCtx, entry: &'static str, layout: &wgpu::BindGroupLayout) -> wgpu::ComputePipeline {
    let module = module(ctx, WGSL_RANKK);
    let pl_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("st.rankk.pipeline_layout"),
            bind_group_layouts: &[layout],
            push_constant_ranges: &[],
        });
    ctx.device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&pl_layout),
            module: &module,
            entry_point: entry,
        })
}

fn ensure_topk_heap_pl(ctx: &WgpuCtx) -> &wgpu::ComputePipeline {
    ctx.topk_heap_pl
        .get_or_init(|| pl(ctx, "topk_subgroups_heap_1ce", ensure_layout_topk(ctx)))
}

fn ensure_topk_heap_sgintrin_pl(ctx: &WgpuCtx) -> &wgpu::ComputePipeline {
    ctx.topk_heap_sgintrin_pl.get_or_init(|| {
        pl(
            ctx,
            "topk_subgroups_heap_sgintrin_1ce",
            ensure_layout_topk(ctx),
        )
    })
}

fn ensure_topk_bit_pl(ctx: &WgpuCtx) -> &wgpu::ComputePipeline {
    ctx.topk_bit_pl
        .get_or_init(|| pl(ctx, "topk_subgroups_bitonic_1ce", ensure_layout_topk(ctx)))
}

fn ensure_topk_wg_pl(ctx: &WgpuCtx) -> &wgpu::ComputePipeline {
    ctx.topk_wg_pl
        .get_or_init(|| pl(ctx, "topk_workgroup_1ce", ensure_layout_topk(ctx)))
}

fn ensure_compact_scan_pl(ctx: &WgpuCtx) -> &wgpu::ComputePipeline {
    ctx.compact_scan_pl
        .get_or_init(|| pl(ctx, "compact_scan_tiles", ensure_layout_compaction(ctx)))
}

fn ensure_compact_row_prefix_pl(ctx: &WgpuCtx) -> &wgpu::ComputePipeline {
    ctx.compact_row_prefix_pl
        .get_or_init(|| pl(ctx, "compact_row_prefix", ensure_layout_compaction(ctx)))
}

fn ensure_compact_apply_pl(ctx: &WgpuCtx) -> &wgpu::ComputePipeline {
    ctx.compact_apply_pl
        .get_or_init(|| pl(ctx, "compact_apply", ensure_layout_compaction(ctx)))
}

fn ensure_compact_apply_sg_pl(ctx: &WgpuCtx) -> &wgpu::ComputePipeline {
    ctx.compact_apply_sg_pl
        .get_or_init(|| pl(ctx, "compact_apply_sg", ensure_layout_compaction(ctx)))
}

fn prefer_compact_apply_sg() -> bool {
    std::env::var("ST_COMPACTION_APPLY")
        .ok()
        .map(|value| value.to_ascii_lowercase() != "fallback")
        .unwrap_or(true)
}

fn compaction_tiles_x(cols: u32) -> u32 {
    (cols.saturating_add(255) / 256).max(1)
}

fn validate_compaction_shape(rows: u32, cols: u32, row_stride: u32) -> Result<u32, String> {
    if rows == 0 || cols == 0 {
        return Err("wgpu_rt: compaction rows/cols must be non-zero".into());
    }
    if row_stride < cols {
        return Err(format!(
            "wgpu_rt: compaction row_stride ({row_stride}) must be >= cols ({cols})"
        ));
    }
    Ok(compaction_tiles_x(cols))
}

/// Plan-only entry point used by `st-core::backend::wgpu_exec`.
///
/// The rank entry / executor trait currently does not carry buffers. Use
/// [`dispatch_topk_1ce_buffers`] instead.
pub fn dispatch_topk_1ce(_plan: &RankPlan) -> Result<(), String> {
    Err("wgpu_rt: plan-only dispatch is not wired; call dispatch_topk_1ce_buffers(...)".into())
}

/// Execute TopK 1CE on WGPU using the precompiled WGSL kernels.
pub fn dispatch_topk_1ce_buffers(
    rows: u32,
    cols: u32,
    k: u32,
    row_stride: u32,
    k_lane: u32,
    x: &wgpu::Buffer,
    out_vals: &wgpu::Buffer,
    out_idx: &wgpu::Buffer,
) -> Result<(), String> {
    let ctx = ctx()?;

    let mut tile_cols = cols;
    let mut radix_hint: u32 = if k.is_power_of_two() { 4 } else { 2 };
    let mut segments_hint: u32 = if cols > 131_072 {
        4
    } else if cols > 32_768 {
        2
    } else {
        1
    };

    // WGPU 0.19 does not expose subgroup capability flags on stable yet.
    // Keep the runtime conservative by default (workgroup path).
    let has_sub = false;
    let mut algo_hint: u8 = 0;
    let mut ctile_hint: u32 = 0;
    if let Some(ch) = crate::backend::wgpu_heuristics::choose_topk(rows, cols, k, has_sub) {
        algo_hint = ch.algo_topk;
        ctile_hint = ch.ctile;
        if ch.tile_cols != 0 {
            tile_cols = ch.tile_cols;
        }
        radix_hint = ch.radix.max(1);
        segments_hint = ch.segments.max(1);
    }
    if ctile_hint != 0 {
        tile_cols = ctile_hint;
    }

    let params = TopkParams {
        rows,
        cols,
        k,
        row_stride,
        k_lane,
        tile_cols,
        radix: radix_hint,
        segments: segments_hint,
    };
    let ub = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("st.rankk.params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let layout = ensure_layout_topk(&ctx);
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("st.rankk.bg.topk"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_vals.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_idx.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: ub.as_entire_binding(),
            },
        ],
    });

    let sgc_hint = if has_sub { 8 } else { 1 };
    let prefer_intrin = std::env::var("ST_USE_SG_INTRIN").ok().as_deref() == Some("1");
    let prefer_heap_default = (k <= 32 && cols >= 2048) || (k <= 16 && sgc_hint >= 8);
    let prefer_heap = match algo_hint {
        1 => true,
        2 => false,
        _ => prefer_heap_default,
    };

    let pipeline = if has_sub {
        if prefer_heap {
            if prefer_intrin {
                ensure_topk_heap_sgintrin_pl(&ctx)
            } else {
                ensure_topk_heap_pl(&ctx)
            }
        } else {
            ensure_topk_bit_pl(&ctx)
        }
    } else {
        ensure_topk_wg_pl(&ctx)
    };

    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("st.rankk.enc.topk"),
        });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.rankk.pass.topk"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(1, rows.max(1), 1);
    }
    ctx.queue.submit(Some(enc.finish()));
    Ok(())
}

fn dispatch_compaction_buffers(
    rows: u32,
    cols: u32,
    row_stride: u32,
    kind: u32,
    x: &wgpu::Buffer,
    mask: &wgpu::Buffer,
    out_counts: &wgpu::Buffer,
    out_vals: &wgpu::Buffer,
    out_idx: &wgpu::Buffer,
    encoder_label: &'static str,
) -> Result<(), String> {
    let ctx = ctx()?;
    let tiles_x = validate_compaction_shape(rows, cols, row_stride)?;
    let params = CompactionParams {
        rows,
        cols,
        row_stride,
        kind,
        tiles_x,
        _pad: 0,
        _pad2: 0,
        _pad3: 0,
    };
    let ub = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("st.rankk.compaction.params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let prefix = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("st.rankk.compaction.prefix"),
        size: rows as u64 * tiles_x as u64 * std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let layout = ensure_layout_compaction(&ctx);
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("st.rankk.bg.compaction"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: mask.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_counts.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_vals.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: out_idx.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: ub.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: prefix.as_entire_binding(),
            },
        ],
    });

    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(encoder_label),
        });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.rankk.pass.compaction.scan"),
            timestamp_writes: None,
        });
        pass.set_pipeline(ensure_compact_scan_pl(&ctx));
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(tiles_x, rows, 1);
    }
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.rankk.pass.compaction.prefix"),
            timestamp_writes: None,
        });
        pass.set_pipeline(ensure_compact_row_prefix_pl(&ctx));
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(1, rows, 1);
    }
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.rankk.pass.compaction.apply"),
            timestamp_writes: None,
        });
        let apply_pipeline = if prefer_compact_apply_sg() {
            ensure_compact_apply_sg_pl(&ctx)
        } else {
            ensure_compact_apply_pl(&ctx)
        };
        pass.set_pipeline(apply_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(tiles_x, rows, 1);
    }
    ctx.queue.submit(Some(enc.finish()));
    Ok(())
}

/// Execute threshold compaction on WGPU.
///
/// The current shader contract consumes an explicit `mask` buffer and emits
/// per-row counts plus packed values and indices. Only the first
/// `counts[row]` values of each row are valid. `kind` is kept for API parity
/// but the current WGSL path is driven entirely by `mask`.
pub fn dispatch_compaction_1ce_buffers(
    rows: u32,
    cols: u32,
    row_stride: u32,
    kind: u32,
    x: &wgpu::Buffer,
    mask: &wgpu::Buffer,
    out_counts: &wgpu::Buffer,
    out_vals: &wgpu::Buffer,
    out_idx: &wgpu::Buffer,
) -> Result<(), String> {
    dispatch_compaction_buffers(
        rows,
        cols,
        row_stride,
        kind,
        x,
        mask,
        out_counts,
        out_vals,
        out_idx,
        "st.rankk.enc.compaction.1ce",
    )
}

/// Execute the two-stage threshold compaction surface on WGPU.
///
/// The current runtime reuses the same scan/prefix/apply kernel suite as the
/// 1CE path and keeps the separate entry point for higher-level API parity.
pub fn dispatch_compaction_2ce_buffers(
    rows: u32,
    cols: u32,
    row_stride: u32,
    kind: u32,
    x: &wgpu::Buffer,
    mask: &wgpu::Buffer,
    out_counts: &wgpu::Buffer,
    out_vals: &wgpu::Buffer,
    out_idx: &wgpu::Buffer,
) -> Result<(), String> {
    dispatch_compaction_buffers(
        rows,
        cols,
        row_stride,
        kind,
        x,
        mask,
        out_counts,
        out_vals,
        out_idx,
        "st.rankk.enc.compaction.2ce",
    )
}

/// Plan-only compaction entry point kept for parity with the older executor surface.
pub fn dispatch_compaction_1ce(_plan: &CompactionPlan, _kind: u32) -> Result<(), String> {
    Err("wgpu_rt: plan-only compaction dispatch is not wired; call dispatch_compaction_1ce_buffers(...)".into())
}

/// Plan-only compaction entry point kept for parity with the older executor surface.
pub fn dispatch_compaction_2ce(_plan: &CompactionPlan, _kind: u32) -> Result<(), String> {
    Err("wgpu_rt: plan-only compaction dispatch is not wired; call dispatch_compaction_2ce_buffers(...)".into())
}

#[cfg(test)]
mod tests {
    use super::{compaction_tiles_x, validate_compaction_shape};

    #[test]
    fn compaction_tiles_are_fixed_256_columns() {
        assert_eq!(compaction_tiles_x(1), 1);
        assert_eq!(compaction_tiles_x(256), 1);
        assert_eq!(compaction_tiles_x(257), 2);
    }

    #[test]
    fn compaction_shape_rejects_invalid_dense_rows() {
        assert!(validate_compaction_shape(0, 4, 4).is_err());
        assert!(validate_compaction_shape(4, 0, 4).is_err());
        assert!(validate_compaction_shape(1, 8, 4).is_err());
        assert_eq!(validate_compaction_shape(2, 8, 8).unwrap(), 1);
    }
}
