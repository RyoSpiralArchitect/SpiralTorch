// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Pipeline assembly helpers for MidK/BottomK compaction kernels.

use std::path::PathBuf;

use wgpu::{ComputePipeline, Device};

use crate::{ShaderCache, ShaderLoadError};

#[derive(Debug)]
pub struct Pipelines {
    pub scan_tiles: ComputePipeline,
    pub row_prefix: ComputePipeline,
    pub apply_fallback: ComputePipeline,
    pub apply_subgroup: Option<ComputePipeline>,
    pub apply_subgroup_v2: Option<ComputePipeline>,
}

impl Pipelines {
    /// Pick the best available subgroup variant, falling back to the portable pipeline.
    pub fn best_subgroup(&self) -> Option<&ComputePipeline> {
        self.apply_subgroup_v2
            .as_ref()
            .or(self.apply_subgroup.as_ref())
    }
}

pub struct Builder<'a> {
    device: &'a Device,
    cache: ShaderCache,
    supports_subgroup: bool,
    include_subgroup_v1: bool,
    include_subgroup_v2: bool,
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
            include_subgroup_v1: false,
            include_subgroup_v2: false,
        }
    }

    /// Toggle support for subgroup pipelines.
    pub fn supports_subgroup(mut self, supports: bool) -> Self {
        self.supports_subgroup = supports;
        self
    }

    /// Request the legacy subgroup compaction path.
    pub fn include_subgroup(mut self) -> Self {
        self.include_subgroup_v1 = true;
        self
    }

    /// Request the enhanced subgroup compaction path introduced in v1.8.5.
    pub fn include_subgroup_v2(mut self) -> Self {
        self.include_subgroup_v2 = true;
        self
    }

    /// Borrow the underlying cache for prefetching or manual shader control.
    pub fn cache_mut(&mut self) -> &mut ShaderCache {
        &mut self.cache
    }

    /// Consume the builder and return the cache without building pipelines.
    pub fn into_cache(self) -> ShaderCache {
        self.cache
    }

    fn assemble(&mut self) -> Result<Pipelines, ShaderLoadError> {
        let scan_tiles = self.cache.load_compute_pipeline(
            self.device,
            "midk_bottomk_compaction.wgsl",
            "midk_compact_scan_tiles",
            "midk_compact_scan_tiles",
        )?;

        let row_prefix = self.cache.load_compute_pipeline(
            self.device,
            "midk_bottomk_compaction.wgsl",
            "midk_compact_row_prefix",
            "midk_compact_row_prefix",
        )?;

        let apply_fallback = self.cache.load_compute_pipeline(
            self.device,
            "midk_bottomk_compaction.wgsl",
            "midk_compact_apply",
            "midk_compact_apply",
        )?;

        let apply_subgroup = (self.supports_subgroup && self.include_subgroup_v1)
            .then(|| {
                self.cache.load_compute_pipeline(
                    self.device,
                    "midk_bottomk_compaction.wgsl",
                    "midk_compact_apply_sg",
                    "midk_compact_apply_sg",
                )
            })
            .transpose()?;

        let apply_subgroup_v2 = (self.supports_subgroup && self.include_subgroup_v2)
            .then(|| {
                self.cache.load_compute_pipeline(
                    self.device,
                    "midk_bottomk_compaction.wgsl",
                    "midk_compact_apply_sg2",
                    "midk_compact_apply_sg2",
                )
            })
            .transpose()?;

        Ok(Pipelines {
            scan_tiles,
            row_prefix,
            apply_fallback,
            apply_subgroup,
            apply_subgroup_v2,
        })
    }

    /// Build the requested pipelines.
    pub fn build(mut self) -> Result<Pipelines, ShaderLoadError> {
        self.assemble()
    }

    /// Build pipelines while returning the [`ShaderCache`] for reuse.
    pub fn build_with_cache(mut self) -> Result<(Pipelines, ShaderCache), ShaderLoadError> {
        let pipelines = self.assemble()?;
        Ok((pipelines, self.cache))
    }
}

pub fn create_pipelines(
    device: &Device,
    shader_dir: &str,
    supports_subgroup: bool,
    include_v1: bool,
    include_v2: bool,
) -> Result<Pipelines, ShaderLoadError> {
    let builder = Builder::new(device, shader_dir).supports_subgroup(supports_subgroup);
    let builder = if include_v1 {
        builder.include_subgroup()
    } else {
        builder
    };
    let builder = if include_v2 {
        builder.include_subgroup_v2()
    } else {
        builder
    };
    builder.build()
}
