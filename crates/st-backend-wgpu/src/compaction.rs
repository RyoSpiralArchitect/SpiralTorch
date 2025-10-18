// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use wgpu::{ComputePipeline, Device};

use crate::util::{load_compute_pipeline, ShaderLoadError};

use crate::util::{load_compute_pipeline, ShaderLoadError};

pub struct CompactionPipelines {
    pub p_1ce: ComputePipeline,
}

pub fn create(device: &Device, shader_dir: &str) -> Result<CompactionPipelines, ShaderLoadError> {
    let pipeline = load_compute_pipeline(
        device,
        shader_dir,
        "wgpu_compaction_1ce.wgsl",
        "compaction_1ce",
        "main_cs",
    )?;
    Ok(CompactionPipelines { p_1ce: pipeline })
}
