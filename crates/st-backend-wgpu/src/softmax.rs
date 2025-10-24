// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Row-wise softmax pipelines for the WGPU backend.
//!
//! The module exposes a builder that loads the subgroup-optimised WGSL kernel
//! when the device advertises `wgpu::Features::SUBGROUPS`. A future
//! workgroup-only fallback can plug into the same surface area without forcing
//! callers to juggle shader file names.

use std::path::PathBuf;

use wgpu::{ComputePipeline, Device, Features};

use crate::{ShaderCache, ShaderLoadError};

/// Compute pipelines required to execute a row-wise softmax.
#[derive(Debug)]
pub struct Pipelines {
    /// Subgroup-optimised pipeline. Present only when the device exposes the
    /// subgroup feature and the caller requested it.
    pub subgroup: Option<ComputePipeline>,
}

impl Pipelines {
    /// Returns the best available pipeline for execution. Prefers the subgroup
    /// variant when it exists, otherwise falls back to the workgroup path
    /// (currently unimplemented and represented by `None`).
    pub fn best(&self) -> Option<&ComputePipeline> {
        self.subgroup.as_ref()
    }
}

/// Builder that wires shader loading and pipeline creation.
pub struct Builder<'a> {
    device: &'a Device,
    cache: ShaderCache,
    supports_subgroup: bool,
}

impl<'a> Builder<'a> {
    /// Create a builder that reads WGSL files from `shader_dir`.
    pub fn new(device: &'a Device, shader_dir: impl Into<PathBuf>) -> Self {
        Self::with_cache(device, ShaderCache::new(shader_dir))
    }

    /// Create a builder from an existing [`ShaderCache`].
    pub fn with_cache(device: &'a Device, cache: ShaderCache) -> Self {
        Self {
            device,
            cache,
            supports_subgroup: false,
        }
    }

    /// Toggle subgroup support on the builder.
    pub fn supports_subgroup(mut self, supports_subgroup: bool) -> Self {
        self.supports_subgroup = supports_subgroup;
        self
    }

    /// Inspect the underlying cache for eager shader preloads.
    pub fn cache_mut(&mut self) -> &mut ShaderCache {
        &mut self.cache
    }

    /// Consume the builder without constructing pipelines, returning the cache.
    pub fn into_cache(self) -> ShaderCache {
        self.cache
    }

    fn assemble(&mut self) -> Result<Pipelines, ShaderLoadError> {
        let subgroup = self
            .supports_subgroup
            .then(|| {
                self.cache.load_compute_pipeline(
                    self.device,
                    "softmax_row_subgroup.wgsl",
                    "softmax_row_subgroup",
                    "main_cs",
                )
            })
            .transpose()?;

        Ok(Pipelines { subgroup })
    }

    /// Build the requested pipelines.
    pub fn build(mut self) -> Result<Pipelines, ShaderLoadError> {
        self.assemble()
    }

    /// Build pipelines while returning the cache for reuse.
    pub fn build_with_cache(mut self) -> Result<(Pipelines, ShaderCache), ShaderLoadError> {
        let pipelines = self.assemble()?;
        Ok((pipelines, self.cache))
    }
}

/// Convenience helper that creates the softmax pipelines using feature flags
/// advertised by the device.
pub fn create_pipelines(
    device: &Device,
    shader_dir: &str,
    features: Features,
) -> Result<Pipelines, ShaderLoadError> {
    Builder::new(device, shader_dir)
        .supports_subgroup(features.contains(Features::SUBGROUPS))
        .build()
}
