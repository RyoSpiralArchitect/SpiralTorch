// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use wgpu::*;

use crate::util::{load_compute_pipeline, ShaderLoadError};

/// Loads the ND indexer shader that materialises strided indices and segment ids.
pub fn create_pipeline(
    device: &Device,
    shader_dir: &str,
) -> Result<ComputePipeline, ShaderLoadError> {
    load_compute_pipeline(device, shader_dir, "nd_indexer.wgsl", "nd_indexer", "main")
}
