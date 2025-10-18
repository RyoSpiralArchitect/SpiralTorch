// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Minimal driver for WGSL keep‑k kernels (subgroup/workgroup).
//! This is a scaffold; integrate with your existing WGPU backend device/queue management.

use std::path::PathBuf;

use wgpu::*;

use crate::{ShaderCache, ShaderLoadError};

pub enum MergeKind {
    Bitonic = 0,
    Shared = 1,
    Warp = 2,
}

pub struct Pipelines {
    pub keepk_subgroup: Option<ComputePipeline>,
    pub keepk_workgroup: ComputePipeline,
    pub keepk_subgroup_1ce: Option<ComputePipeline>,
    pub keepk_subgroup_1ce_large: Option<ComputePipeline>,
}

impl Pipelines {
    /// Convenience accessor to select the best subgroup kernel based on the
    /// desired merge strategy and availability of specialised variants.
    pub fn subgroup_for_merge(
        &self,
        merge: MergeKind,
        prefer_large: bool,
    ) -> Option<&ComputePipeline> {
        match merge {
            MergeKind::Bitonic | MergeKind::Shared | MergeKind::Warp => {
                if prefer_large {
                    self.keepk_subgroup_1ce_large
                        .as_ref()
                        .or_else(|| self.keepk_subgroup_1ce.as_ref())
                        .or(self.keepk_subgroup.as_ref())
                } else {
                    self.keepk_subgroup_1ce
                        .as_ref()
                        .or_else(|| self.keepk_subgroup_1ce_large.as_ref())
                        .or(self.keepk_subgroup.as_ref())
                }
            }
        }
    }
}

/// Builder that loads the keep-k shader family and constructs the desired
/// pipelines.
pub struct Builder<'a> {
    device: &'a Device,
    cache: ShaderCache,
    supports_subgroup: bool,
    include_1ce: bool,
    include_large_1ce: bool,
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
            include_1ce: false,
            include_large_1ce: false,
        }
    }

    /// Toggle subgroup pipeline generation.
    pub fn supports_subgroup(mut self, supports_subgroup: bool) -> Self {
        self.supports_subgroup = supports_subgroup;
        self
    }

    /// Borrow the underlying cache for manual control.
    pub fn cache_mut(&mut self) -> &mut ShaderCache {
        &mut self.cache
    }

    /// Consume the builder and return the cache without constructing pipelines.
    pub fn into_cache(self) -> ShaderCache {
        self.cache
    }

    /// Include the subgroup 1CE variant of the keep-k kernel.
    pub fn include_subgroup_1ce(mut self) -> Self {
        self.include_1ce = true;
        self
    }

    /// Include the large subgroup 1CE shader in addition to the standard one.
    pub fn include_subgroup_1ce_large(mut self) -> Self {
        self.include_1ce = true;
        self.include_large_1ce = true;
        self
    }

    /// Consume the builder, load the requested shaders and produce pipeline handles.
    pub fn build(mut self) -> Result<Pipelines, ShaderLoadError> {
        let keepk_workgroup = self.cache.load_compute_pipeline(
            self.device,
            "topk_keepk_workgroup.wgsl",
            "keepk_workgroup",
            "main_cs",
        )?;

        let keepk_subgroup = if self.supports_subgroup {
            Some(self.cache.load_compute_pipeline(
                self.device,
                "topk_keepk_subgroup.wgsl",
                "keepk_subgroup",
                "main_cs",
            )?)
        } else {
            None
        };

        let keepk_subgroup_1ce = if self.include_1ce {
            Some(self.cache.load_compute_pipeline(
                self.device,
                "topk_keepk_subgroup_1ce.wgsl",
                "keepk_subgroup_1ce",
                "main_cs",
            )?)
        } else {
            None
        };

        let keepk_subgroup_1ce_large = if self.include_large_1ce {
            Some(self.cache.load_compute_pipeline(
                self.device,
                "topk_keepk_subgroup_1ce_large.wgsl",
                "keepk_subgroup_1ce_large",
                "main_cs",
            )?)
        } else {
            None
        };

        Ok(Pipelines {
            keepk_subgroup,
            keepk_workgroup,
            keepk_subgroup_1ce,
            keepk_subgroup_1ce_large,
        })
    }
}

pub fn create_pipelines(
    device: &Device,
    shader_dir: &str,
    supports_subgroup: bool,
) -> Result<Pipelines, ShaderLoadError> {
    Builder::new(device, shader_dir)
        .supports_subgroup(supports_subgroup)
        .build()
}
