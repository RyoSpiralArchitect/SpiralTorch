// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::{
    collections::HashMap,
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

/// Lightweight cache for WGSL shader sources stored on disk.
///
/// Many of SpiralTorch's compute pipelines are composed of multiple entry
/// points that live in separate shader files. When bootstrapping the backend
/// we often need to read several of these files in quick succession. The cache
/// keeps the decoded UTF-8 source in memory so subsequent pipeline
/// construction avoids touching the filesystem again.
#[derive(Debug, Default)]
pub struct ShaderCache {
    shader_dir: PathBuf,
    sources: HashMap<PathBuf, String>,
}

impl ShaderCache {
    /// Create a cache rooted at `shader_dir`.
    pub fn new(shader_dir: impl Into<PathBuf>) -> Self {
        Self {
            shader_dir: shader_dir.into(),
            sources: HashMap::new(),
        }
    }

    /// Return the absolute directory containing cached shaders.
    pub fn shader_dir(&self) -> &Path {
        &self.shader_dir
    }

    /// Drop all cached shader sources.
    pub fn clear(&mut self) {
        self.sources.clear();
    }

    /// Retrieve the WGSL source for `file`, reading it from disk if necessary.
    pub fn source(&mut self, file: &str) -> Result<&str, ShaderLoadError> {
        let path = self.shader_dir.join(file);
        if !self.sources.contains_key(&path) {
            let source = fs::read_to_string(&path).map_err(|source| ShaderLoadError::Io {
                source,
                path: path.clone(),
            })?;
            self.sources.insert(path.clone(), source);
        }
        Ok(self
            .sources
            .get(&path)
            .expect("shader cache entry missing")
            .as_str())
    }

    /// Load a compute pipeline by file name using the cached shader source.
    pub fn load_compute_pipeline(
        &mut self,
        device: &Device,
        file: &str,
        label: &str,
        entry_point: &str,
    ) -> Result<ComputePipeline, ShaderLoadError> {
        let source = self.source(file)?.to_owned();
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
    load_compute_pipeline_with_layout(device, shader_dir, file, label, entry_point, None)
}

/// Variant of [`load_compute_pipeline`] that allows the caller to specify a
/// pipeline layout.
pub fn load_compute_pipeline_with_layout(
    device: &Device,
    shader_dir: impl AsRef<Path>,
    file: &str,
    label: &str,
    entry_point: &str,
    layout: Option<&wgpu::PipelineLayout>,
) -> Result<ComputePipeline, ShaderLoadError> {
    let source = read_wgsl(shader_dir, file)?;
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some(label),
        source: ShaderSource::Wgsl(source.into()),
    });
    Ok(device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(label),
        layout,
        module: &module,
        entry_point,
    }))
}
