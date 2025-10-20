// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use wgpu::{ComputePipeline, Device};

use crate::util::{load_compute_pipeline, ShaderLoadError};

/// Builds the volumetric ray marching pipeline used to composite NeRF samples.
pub fn create_raymarch_pipeline(
    device: &Device,
    shader_dir: &str,
) -> Result<ComputePipeline, ShaderLoadError> {
    load_compute_pipeline(
        device,
        shader_dir,
        "nerf_raymarch.wgsl",
        "nerf_raymarch",
        "main",
    )
}

/// Builds the sampling utility pipeline that expands rays into stratified sample positions.
pub fn create_sampling_pipeline(
    device: &Device,
    shader_dir: &str,
) -> Result<ComputePipeline, ShaderLoadError> {
    load_compute_pipeline(
        device,
        shader_dir,
        "nerf_volume_utils.wgsl",
        "nerf_volume_utils",
        "main",
    )
}
