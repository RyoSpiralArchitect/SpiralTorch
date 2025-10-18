// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use wgpu::{ComputePipeline, Device};

use crate::util::{load_compute_pipeline, ShaderLoadError};

pub struct Compaction2CE {
    pub p_scan: ComputePipeline,
    pub p_apply: ComputePipeline,
}

pub fn create(device: &Device, shader_dir: &str) -> Result<Compaction2CE, ShaderLoadError> {
    let p_scan = load_compute_pipeline(
        device,
        shader_dir,
        "wgpu_compaction_scan_pass.wgsl",
        "compaction_scan_pass",
        "main_cs",
    )?;
    let p_apply = load_compute_pipeline(
        device,
        shader_dir,
        "wgpu_compaction_apply_pass.wgsl",
        "compaction_apply_pass",
        "main_cs",
    )?;
    Ok(Compaction2CE { p_scan, p_apply })
}
