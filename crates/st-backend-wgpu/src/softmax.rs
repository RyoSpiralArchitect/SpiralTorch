// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Row-wise softmax pipelines optimised with subgroup operations.
//! The builder mirrors the style used by the keep-k kernels and only
//! materialises subgroup variants when the device exposes the relevant
//! WGPU features. The shader itself performs a fused max-reduce,
//! exponentiation and normalisation in a single pass without temporary
//! buffers, matching the fusion requirements outlined for SpiralTorch's
//! Level 2 optimisation roadmap.

use std::path::PathBuf;

use wgpu::{ComputePipeline, Device};

use crate::{ShaderCache, ShaderLoadError};

/// Collection of compute pipelines implementing row-wise softmax.
#[derive(Debug)]
pub struct Pipelines {
    /// Subgroup-accelerated implementation. Present when the target device
    /// advertises subgroup operations via `wgpu::Features::SUBGROUPS`.
    pub row_softmax_subgroup: Option<ComputePipeline>,
}

impl Pipelines {
    /// Return the subgroup accelerated pipeline if it was constructed.
    pub fn subgroup(&self) -> Option<&ComputePipeline> {
        self.row_softmax_subgroup.as_ref()
    }
}

/// Builder for the row-wise softmax pipelines.
pub struct Builder<'a> {
    device: &'a Device,
    cache: ShaderCache,
    supports_subgroup: bool,
}

impl<'a> Builder<'a> {
    /// Create a builder rooted at the provided shader directory.
    pub fn new(device: &'a Device, shader_dir: impl Into<PathBuf>) -> Self {
        Self::with_cache(device, ShaderCache::new(shader_dir))
    }

    /// Create a builder from an existing cache instance.
    pub fn with_cache(device: &'a Device, cache: ShaderCache) -> Self {
        Self {
            device,
            cache,
            supports_subgroup: false,
        }
    }

    /// Toggle subgroup pipeline generation.
    pub fn supports_subgroup(mut self, supports_subgroup: bool) -> Self {
        self.supports_subgroup = supports_subgroup;
        self
    }

    /// Borrow the underlying cache for custom shader loading sequences.
    pub fn cache_mut(&mut self) -> &mut ShaderCache {
        &mut self.cache
    }

    /// Consume the builder and return the cache without constructing pipelines.
    pub fn into_cache(self) -> ShaderCache {
        self.cache
    }

    fn assemble(&mut self) -> Result<Pipelines, ShaderLoadError> {
        let row_softmax_subgroup = self
            .supports_subgroup
            .then(|| {
                self.cache.load_compute_pipeline(
                    self.device,
                    "row_softmax_subgroup.wgsl",
                    "row_softmax_subgroup",
                    "main_cs",
                )
            })
            .transpose()?;

        Ok(Pipelines {
            row_softmax_subgroup,
        })
    }

    /// Consume the builder, loading the requested shaders and producing pipeline handles.
    pub fn build(mut self) -> Result<Pipelines, ShaderLoadError> {
        self.assemble()
    }

    /// Build pipelines while returning the underlying [`ShaderCache`] for reuse.
    pub fn build_with_cache(mut self) -> Result<(Pipelines, ShaderCache), ShaderLoadError> {
        let pipelines = self.assemble()?;
        Ok((pipelines, self.cache))
    }
}

/// Convenience helper to build the softmax pipelines in one call.
pub fn create_pipelines(
    device: &Device,
    shader_dir: &str,
    supports_subgroup: bool,
) -> Result<Pipelines, ShaderLoadError> {
    Builder::new(device, shader_dir)
        .supports_subgroup(supports_subgroup)
        .build()
}

// Tests are omitted intentionally: instantiating a `wgpu::Device` in CI without a
// hardware adapter is brittle and distracts from the primary goal of providing the
// shader plumbing. Integration coverage should live in higher level crates that own
// the actual backend context.
