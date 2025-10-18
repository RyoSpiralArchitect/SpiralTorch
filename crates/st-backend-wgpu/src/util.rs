// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::{
    fs,
    path::{Path, PathBuf},
};

use thiserror::Error;
use wgpu::{
    ComputePipeline, ComputePipelineDescriptor, Device, ShaderModuleDescriptor, ShaderSource,
};

/// Errors that may occur when loading WGSL shaders into a WGPU pipeline.
#[derive(Debug, Error)]
pub enum ShaderLoadError {
    /// The WGSL source file could not be read from disk.
    #[error("failed to read WGSL shader '{path}'")]
    Io {
        /// The underlying IO error raised by the filesystem.
        #[source]
        source: std::io::Error,
        /// The path that could not be read.
        path: PathBuf,
    },
}

/// Read a WGSL shader relative to `shader_dir` and return the source string.
pub fn read_wgsl(shader_dir: impl AsRef<Path>, file: &str) -> Result<String, ShaderLoadError> {
    let path = shader_dir.as_ref().join(file);
    fs::read_to_string(&path).map_err(|source| ShaderLoadError::Io { source, path })
}

/// Load a WGSL shader and build a compute pipeline for the provided entry point.
///
/// The helper keeps module creation consistent across SpiralTorch's WGPU kernels
/// and eliminates repetitive boilerplate from individual pipeline constructors.
pub fn load_compute_pipeline(
    device: &Device,
    shader_dir: impl AsRef<Path>,
    file: &str,
    label: &str,
    entry_point: &str,
) -> Result<ComputePipeline, ShaderLoadError> {
    let source = read_wgsl(shader_dir, file)?;
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some(label),
        source: ShaderSource::Wgsl(source.into()),
    });
    Ok(device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point,
    }))
}
