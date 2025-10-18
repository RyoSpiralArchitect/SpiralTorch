// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Minimal driver for WGSL keep‑k kernels (subgroup/workgroup).
//! This is a scaffold; integrate with your existing WGPU backend device/queue management.

use wgpu::*;

use crate::util::{load_compute_pipeline, ShaderLoadError};

pub enum MergeKind {
    Bitonic = 0,
    Shared = 1,
    Warp = 2,
}

pub struct Pipelines {
    pub keepk_subgroup: Option<ComputePipeline>,
    pub keepk_workgroup: ComputePipeline,
}

pub fn create_pipelines(
    device: &Device,
    shader_dir: &str,
    supports_subgroup: bool,
) -> Result<Pipelines, ShaderLoadError> {
    let keepk_workgroup = load_compute_pipeline(
        device,
        shader_dir,
        "topk_keepk_workgroup.wgsl",
        "keepk_workgroup",
        "main_cs",
    )?;

    let keepk_subgroup = if supports_subgroup {
        Some(load_compute_pipeline(
            device,
            shader_dir,
            "topk_keepk_subgroup.wgsl",
            "keepk_subgroup",
            "main_cs",
        )?)
    } else {
        None
    };

    Ok(Pipelines {
        keepk_subgroup,
        keepk_workgroup,
    })
}
