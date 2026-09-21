// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use wgpu::{ComputePipeline, Device};

use crate::util::{create_inline_module, load_compute_pipeline, read_wgsl, ShaderLoadError};

mod resident;
pub use resident::{NerfError, NerfRay, NerfSamples, RaySampling, ResidentNerf, ResidentRays};

/// Builds the volumetric ray marching pipeline used to composite NeRF samples.
pub fn create_raymarch_pipeline(
    device: &Device,
    shader_dir: &str,
) -> Result<ComputePipeline, ShaderLoadError> {
    let source = read_wgsl(shader_dir, "nerf_raymarch.wgsl")?;
    let module = create_inline_module(
        device,
        "nerf_raymarch",
        crate::shader_sources::expand_nerf_raymarch_source(&source),
    )?;
    Ok(
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nerf_raymarch"),
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
        }),
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
