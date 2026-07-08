// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Static WGPU kernel catalog and shape-level selection reports.
//!
//! These helpers deliberately avoid creating a `wgpu::Device`.  They expose the
//! kernel family that the runtime would attempt to assemble so higher layers can
//! trace planner choices, prepare Redis/telemetry payloads, and explain WGPU
//! fallback behaviour without needing a GPU during report generation.

const SOFTMAX_BINDINGS: &[&str] = &[
    "values:read_storage",
    "output:write_storage",
    "params:uniform",
    "mask:write_storage",
];
const TOPK_BINDINGS: &[&str] = &["values:read_storage", "out_values:write_storage"];
const COMPACTION_BINDINGS: &[&str] = &[
    "values:read_storage",
    "mask:read_storage",
    "out_positions:write_storage",
    "out_values:write_storage",
    "prefix:scratch_storage",
];
const ATTENTION_BINDINGS: &[&str] = &[
    "queries:read_storage",
    "keys:read_storage",
    "values:read_storage",
    "z_bias:read_storage_optional",
    "attn_bias:read_storage_optional",
    "output:write_storage",
    "params:uniform",
];
const TRANSFORM_BINDINGS: &[&str] = &["image:storage", "params:uniform"];
const NERF_BINDINGS: &[&str] = &["rays:read_storage", "volume:read_storage", "output:write_storage"];
const REDUCE_BINDINGS: &[&str] = &["input:read_storage", "output:write_storage"];

const STAGE_SOFTMAX: &[&str] = &["reduce_and_normalize"];
const STAGE_TOPK: &[&str] = &["keepk"];
const STAGE_MIDK: &[&str] = &["scan_tiles", "row_prefix", "apply", "middlemax_optional"];
const STAGE_COMPACTION_1CE: &[&str] = &["compact"];
const STAGE_COMPACTION_2CE: &[&str] = &["scan", "apply"];
const STAGE_ATTENTION: &[&str] = &["qk", "bias", "online_softmax", "value_mix"];
const STAGE_GELU_BACK: &[&str] = &["fused_backward", "reduce"];
const STAGE_TRANSFORM: &[&str] = &["sample", "write"];
const STAGE_NERF: &[&str] = &["raymarch"];
const STAGE_INDEXER: &[&str] = &["index"];
const STAGE_REDUCE: &[&str] = &["reduce"];

/// Static descriptor for a WGPU shader/pipeline variant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelDescriptor {
    pub name: &'static str,
    pub family: &'static str,
    pub operation: &'static str,
    pub shader: &'static str,
    pub entry_point: &'static str,
    pub pipeline_label: &'static str,
    pub variant: &'static str,
    pub subgroup: bool,
    pub portable: bool,
    pub stages: &'static [&'static str],
    pub bindings: &'static [&'static str],
    pub notes: &'static str,
}

/// Compact dispatch geometry used by report surfaces.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DispatchGeometry {
    pub workgroups: (u32, u32, u32),
    pub tiles_x: u32,
    pub row_stride: u32,
    pub empty: bool,
}

/// FFT/fractional hints carried through rank plans into WGPU traces.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FftKernelHints {
    pub tile_cols: u32,
    pub radix: u32,
    pub segments: u32,
}

/// Rank-family operation for WGPU kernel reports.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RankKernelKind {
    TopK,
    MidK,
    BottomK,
}

impl RankKernelKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TopK => "topk",
            Self::MidK => "midk",
            Self::BottomK => "bottomk",
        }
    }
}

/// Shape and planner knobs needed to explain rank-k WGPU kernel selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RankKernelRequest {
    pub kind: RankKernelKind,
    pub rows: u32,
    pub cols: u32,
    pub k: u32,
    pub subgroup: bool,
    pub use_two_stage: bool,
    pub fft_tile: u32,
    pub fft_radix: u32,
    pub fft_segments: u32,
    pub compaction_tile: u32,
}

/// Rank-k kernel report derived without allocating WGPU resources.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RankKernelReport {
    pub request: RankKernelRequest,
    pub primary: &'static KernelDescriptor,
    pub fallback: Option<&'static KernelDescriptor>,
    pub dispatch: DispatchGeometry,
    pub fft: FftKernelHints,
    pub stages: &'static [&'static str],
}

/// Softmax-family request for static WGPU kernel reports.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SoftmaxKernelRequest {
    pub rows: u32,
    pub cols: u32,
    pub subgroup: bool,
    pub hardmax: bool,
    pub mask: bool,
}

/// Softmax/hardmax report derived without allocating WGPU resources.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SoftmaxKernelReport {
    pub request: SoftmaxKernelRequest,
    pub primary: &'static KernelDescriptor,
    pub fallback: Option<&'static KernelDescriptor>,
    pub dispatch: DispatchGeometry,
    pub flags: u32,
    pub stages: &'static [&'static str],
}

const KERNEL_CATALOG: &[KernelDescriptor] = &[
    KernelDescriptor {
        name: "softmax_workgroup",
        family: "softmax",
        operation: "row_softmax",
        shader: "softmax_workgroup.wgsl",
        entry_point: "main_cs",
        pipeline_label: "st.softmax.workgroup",
        variant: "workgroup",
        subgroup: false,
        portable: true,
        stages: STAGE_SOFTMAX,
        bindings: SOFTMAX_BINDINGS,
        notes: "portable row-wise softmax fallback",
    },
    KernelDescriptor {
        name: "softmax_subgroup",
        family: "softmax",
        operation: "row_softmax",
        shader: "softmax_subgroup.wgsl",
        entry_point: "main_cs",
        pipeline_label: "st.softmax.subgroup",
        variant: "subgroup",
        subgroup: true,
        portable: false,
        stages: STAGE_SOFTMAX,
        bindings: SOFTMAX_BINDINGS,
        notes: "subgroup-accelerated row-wise softmax",
    },
    KernelDescriptor {
        name: "softmax_row_subgroup",
        family: "softmax",
        operation: "row_softmax",
        shader: "softmax_row_subgroup.wgsl",
        entry_point: "main_cs",
        pipeline_label: "st.softmax.row_subgroup",
        variant: "row_subgroup",
        subgroup: true,
        portable: false,
        stages: STAGE_SOFTMAX,
        bindings: SOFTMAX_BINDINGS,
        notes: "alternate row-subgroup softmax shader used by tensor utilities",
    },
    KernelDescriptor {
        name: "softmax_spiral_consensus",
        family: "softmax",
        operation: "zspace_softmax",
        shader: "softmax_spiral_consensus.wgsl",
        entry_point: "main_cs",
        pipeline_label: "st.softmax.spiral_consensus",
        variant: "spiral_consensus",
        subgroup: false,
        portable: true,
        stages: STAGE_SOFTMAX,
        bindings: SOFTMAX_BINDINGS,
        notes: "Z-space consensus softmax experimental shader",
    },
    KernelDescriptor {
        name: "softmax_zspace_projection",
        family: "softmax",
        operation: "zspace_projection",
        shader: "softmax_zspace_projection.wgsl",
        entry_point: "main_cs",
        pipeline_label: "st.softmax.zspace_projection",
        variant: "zspace_projection",
        subgroup: false,
        portable: true,
        stages: STAGE_SOFTMAX,
        bindings: SOFTMAX_BINDINGS,
        notes: "Z-space projection helper for softmax telemetry experiments",
    },
    KernelDescriptor {
        name: "topk_keepk_workgroup",
        family: "rank",
        operation: "topk",
        shader: "topk_keepk_workgroup.wgsl",
        entry_point: "main_cs",
        pipeline_label: "keepk_workgroup",
        variant: "workgroup",
        subgroup: false,
        portable: true,
        stages: STAGE_TOPK,
        bindings: TOPK_BINDINGS,
        notes: "portable TopK keep-k fallback",
    },
    KernelDescriptor {
        name: "topk_keepk_subgroup",
        family: "rank",
        operation: "topk",
        shader: "topk_keepk_subgroup.wgsl",
        entry_point: "main_cs",
        pipeline_label: "keepk_subgroup",
        variant: "subgroup",
        subgroup: true,
        portable: false,
        stages: STAGE_TOPK,
        bindings: TOPK_BINDINGS,
        notes: "subgroup TopK keep-k kernel",
    },
    KernelDescriptor {
        name: "topk_keepk_subgroup_1ce",
        family: "rank",
        operation: "topk",
        shader: "topk_keepk_subgroup_1ce.wgsl",
        entry_point: "main_cs",
        pipeline_label: "keepk_subgroup_1ce",
        variant: "subgroup_1ce",
        subgroup: true,
        portable: false,
        stages: STAGE_TOPK,
        bindings: TOPK_BINDINGS,
        notes: "single-pass subgroup TopK keep-k kernel",
    },
    KernelDescriptor {
        name: "topk_keepk_subgroup_1ce_large",
        family: "rank",
        operation: "topk",
        shader: "topk_keepk_subgroup_1ce_large.wgsl",
        entry_point: "main_cs",
        pipeline_label: "keepk_subgroup_1ce_large",
        variant: "subgroup_1ce_large",
        subgroup: true,
        portable: false,
        stages: STAGE_TOPK,
        bindings: TOPK_BINDINGS,
        notes: "large-row single-pass subgroup TopK keep-k kernel",
    },
    KernelDescriptor {
        name: "midk_bottomk_scan_tiles",
        family: "rank",
        operation: "midk_bottomk",
        shader: "midk_bottomk_compaction.wgsl",
        entry_point: "scan_tiles",
        pipeline_label: "st.midk_bottomk.scan_tiles",
        variant: "scan",
        subgroup: false,
        portable: true,
        stages: STAGE_MIDK,
        bindings: COMPACTION_BINDINGS,
        notes: "tile scan stage for MidK/BottomK compaction",
    },
    KernelDescriptor {
        name: "midk_bottomk_row_prefix",
        family: "rank",
        operation: "midk_bottomk",
        shader: "midk_bottomk_compaction.wgsl",
        entry_point: "row_prefix",
        pipeline_label: "st.midk_bottomk.row_prefix",
        variant: "row_prefix",
        subgroup: false,
        portable: true,
        stages: STAGE_MIDK,
        bindings: COMPACTION_BINDINGS,
        notes: "row prefix stage for MidK/BottomK compaction",
    },
    KernelDescriptor {
        name: "midk_bottomk_apply_fallback",
        family: "rank",
        operation: "midk_bottomk",
        shader: "midk_bottomk_compaction.wgsl",
        entry_point: "apply_fallback",
        pipeline_label: "st.midk_bottomk.apply_fallback",
        variant: "fallback",
        subgroup: false,
        portable: true,
        stages: STAGE_MIDK,
        bindings: COMPACTION_BINDINGS,
        notes: "portable apply stage for MidK/BottomK compaction",
    },
    KernelDescriptor {
        name: "midk_bottomk_apply_subgroup",
        family: "rank",
        operation: "midk_bottomk",
        shader: "midk_bottomk_compaction.wgsl",
        entry_point: "apply_subgroup",
        pipeline_label: "st.midk_bottomk.apply_subgroup",
        variant: "subgroup",
        subgroup: true,
        portable: false,
        stages: STAGE_MIDK,
        bindings: COMPACTION_BINDINGS,
        notes: "legacy subgroup apply stage for MidK/BottomK compaction",
    },
    KernelDescriptor {
        name: "midk_bottomk_apply_subgroup_v2",
        family: "rank",
        operation: "midk_bottomk",
        shader: "midk_bottomk_compaction.wgsl",
        entry_point: "apply_subgroup_v2",
        pipeline_label: "st.midk_bottomk.apply_subgroup_v2",
        variant: "subgroup_v2",
        subgroup: true,
        portable: false,
        stages: STAGE_MIDK,
        bindings: COMPACTION_BINDINGS,
        notes: "enhanced subgroup apply stage for MidK/BottomK compaction",
    },
    KernelDescriptor {
        name: "midk_bottomk_middlemax",
        family: "rank",
        operation: "midk_bottomk",
        shader: "midk_bottomk_compaction.wgsl",
        entry_point: "middlemax",
        pipeline_label: "st.midk_bottomk.middlemax",
        variant: "middlemax",
        subgroup: false,
        portable: true,
        stages: STAGE_MIDK,
        bindings: COMPACTION_BINDINGS,
        notes: "optional middle-band maximum reduction",
    },
    KernelDescriptor {
        name: "wgpu_compaction_1ce",
        family: "compaction",
        operation: "compact",
        shader: "wgpu_compaction_1ce.wgsl",
        entry_point: "main_cs",
        pipeline_label: "compaction_1ce",
        variant: "1ce",
        subgroup: false,
        portable: true,
        stages: STAGE_COMPACTION_1CE,
        bindings: COMPACTION_BINDINGS,
        notes: "single-command encoder compaction helper",
    },
    KernelDescriptor {
        name: "wgpu_compaction_scan_pass",
        family: "compaction",
        operation: "compact",
        shader: "wgpu_compaction_scan_pass.wgsl",
        entry_point: "main_cs",
        pipeline_label: "compaction_scan_pass",
        variant: "2ce_scan",
        subgroup: false,
        portable: true,
        stages: STAGE_COMPACTION_2CE,
        bindings: COMPACTION_BINDINGS,
        notes: "two-command encoder compaction scan pass",
    },
    KernelDescriptor {
        name: "wgpu_compaction_apply_pass",
        family: "compaction",
        operation: "compact",
        shader: "wgpu_compaction_apply_pass.wgsl",
        entry_point: "main_cs",
        pipeline_label: "compaction_apply_pass",
        variant: "2ce_apply",
        subgroup: false,
        portable: true,
        stages: STAGE_COMPACTION_2CE,
        bindings: COMPACTION_BINDINGS,
        notes: "two-command encoder compaction apply pass",
    },
    KernelDescriptor {
        name: "fused_attention_online",
        family: "attention",
        operation: "attention",
        shader: "fused_attention_online.wgsl",
        entry_point: "main_cs",
        pipeline_label: "st.backend.fused_attention",
        variant: "online_softmax",
        subgroup: false,
        portable: true,
        stages: STAGE_ATTENTION,
        bindings: ATTENTION_BINDINGS,
        notes: "fused scaled dot-product attention with online softmax",
    },
    KernelDescriptor {
        name: "fused_gelu_back",
        family: "activation",
        operation: "gelu_backward",
        shader: "fused_gelu_back.wgsl",
        entry_point: "main_cs",
        pipeline_label: "st.gelu_back.fused",
        variant: "fused",
        subgroup: false,
        portable: true,
        stages: STAGE_GELU_BACK,
        bindings: REDUCE_BINDINGS,
        notes: "fused GELU backward helper",
    },
    KernelDescriptor {
        name: "reduce_db",
        family: "reduction",
        operation: "reduce",
        shader: "reduce_db.wgsl",
        entry_point: "main_cs",
        pipeline_label: "st.reduce_db",
        variant: "workgroup",
        subgroup: false,
        portable: true,
        stages: STAGE_REDUCE,
        bindings: REDUCE_BINDINGS,
        notes: "database-style reduction helper",
    },
    KernelDescriptor {
        name: "nd_indexer",
        family: "indexing",
        operation: "index",
        shader: "nd_indexer.wgsl",
        entry_point: "main_cs",
        pipeline_label: "st.nd_indexer",
        variant: "workgroup",
        subgroup: false,
        portable: true,
        stages: STAGE_INDEXER,
        bindings: REDUCE_BINDINGS,
        notes: "N-dimensional indexing helper",
    },
    KernelDescriptor {
        name: "nerf_raymarch",
        family: "vision",
        operation: "nerf_raymarch",
        shader: "nerf_raymarch.wgsl",
        entry_point: "main_cs",
        pipeline_label: "st.nerf.raymarch",
        variant: "raymarch",
        subgroup: false,
        portable: true,
        stages: STAGE_NERF,
        bindings: NERF_BINDINGS,
        notes: "NeRF ray marching shader",
    },
    KernelDescriptor {
        name: "resize",
        family: "transform",
        operation: "image_resize",
        shader: "transforms/resize.wgsl",
        entry_point: "main_cs",
        pipeline_label: "st.transform.resize",
        variant: "bilinear",
        subgroup: false,
        portable: true,
        stages: STAGE_TRANSFORM,
        bindings: TRANSFORM_BINDINGS,
        notes: "image resize transform",
    },
    KernelDescriptor {
        name: "center_crop",
        family: "transform",
        operation: "image_crop",
        shader: "transforms/center_crop.wgsl",
        entry_point: "main_cs",
        pipeline_label: "st.transform.center_crop",
        variant: "center_crop",
        subgroup: false,
        portable: true,
        stages: STAGE_TRANSFORM,
        bindings: TRANSFORM_BINDINGS,
        notes: "image center crop transform",
    },
    KernelDescriptor {
        name: "horizontal_flip",
        family: "transform",
        operation: "image_flip",
        shader: "transforms/horizontal_flip.wgsl",
        entry_point: "main_cs",
        pipeline_label: "st.transform.horizontal_flip",
        variant: "horizontal",
        subgroup: false,
        portable: true,
        stages: STAGE_TRANSFORM,
        bindings: TRANSFORM_BINDINGS,
        notes: "image horizontal flip transform",
    },
    KernelDescriptor {
        name: "color_jitter",
        family: "transform",
        operation: "image_color_jitter",
        shader: "transforms/color_jitter.wgsl",
        entry_point: "main_cs",
        pipeline_label: "st.transform.color_jitter",
        variant: "jitter",
        subgroup: false,
        portable: true,
        stages: STAGE_TRANSFORM,
        bindings: TRANSFORM_BINDINGS,
        notes: "image color jitter transform",
    },
];

/// Return the static WGPU kernel catalog.
#[must_use]
pub fn kernel_catalog() -> &'static [KernelDescriptor] {
    KERNEL_CATALOG
}

/// Find a kernel descriptor by name, shader filename, or pipeline label.
#[must_use]
pub fn kernel_descriptor(name: &str) -> Option<&'static KernelDescriptor> {
    KERNEL_CATALOG.iter().find(|descriptor| {
        descriptor.name == name
            || descriptor.shader == name
            || descriptor.pipeline_label == name
            || descriptor.shader.rsplit('/').next() == Some(name)
    })
}

fn descriptor(name: &str) -> &'static KernelDescriptor {
    kernel_descriptor(name).expect("static WGPU kernel catalog is missing a required descriptor")
}

fn tiles_for_cols(cols: u32) -> u32 {
    cols.saturating_add(255) / 256
}

fn default_dispatch(rows: u32, cols: u32) -> DispatchGeometry {
    DispatchGeometry {
        workgroups: if rows == 0 || cols == 0 {
            (0, 0, 0)
        } else {
            (rows, 1, 1)
        },
        tiles_x: tiles_for_cols(cols),
        row_stride: cols,
        empty: rows == 0 || cols == 0,
    }
}

/// Describe WGPU rank-k kernel selection for a planner-like request.
#[must_use]
pub fn rank_kernel_report(request: RankKernelRequest) -> RankKernelReport {
    let fallback = match request.kind {
        RankKernelKind::TopK => Some(descriptor("topk_keepk_workgroup")),
        RankKernelKind::MidK | RankKernelKind::BottomK => {
            Some(descriptor("midk_bottomk_apply_fallback"))
        }
    };
    let primary = match request.kind {
        RankKernelKind::TopK if request.subgroup && request.k > 1024 => {
            descriptor("topk_keepk_subgroup_1ce_large")
        }
        RankKernelKind::TopK if request.subgroup && request.use_two_stage => {
            descriptor("topk_keepk_subgroup")
        }
        RankKernelKind::TopK if request.subgroup => descriptor("topk_keepk_subgroup_1ce"),
        RankKernelKind::TopK => descriptor("topk_keepk_workgroup"),
        RankKernelKind::MidK | RankKernelKind::BottomK if request.subgroup && request.use_two_stage => {
            descriptor("midk_bottomk_apply_subgroup_v2")
        }
        RankKernelKind::MidK | RankKernelKind::BottomK if request.subgroup => {
            descriptor("midk_bottomk_apply_subgroup")
        }
        RankKernelKind::MidK | RankKernelKind::BottomK => {
            descriptor("midk_bottomk_apply_fallback")
        }
    };

    let tiles_x = tiles_for_cols(request.cols);
    let dispatch = match request.kind {
        RankKernelKind::TopK => default_dispatch(request.rows, request.cols),
        RankKernelKind::MidK | RankKernelKind::BottomK => DispatchGeometry {
            workgroups: if request.rows == 0 || request.cols == 0 {
                (0, 0, 0)
            } else {
                (tiles_x, request.rows, 1)
            },
            tiles_x,
            row_stride: request.cols,
            empty: request.rows == 0 || request.cols == 0,
        },
    };

    RankKernelReport {
        request,
        primary,
        fallback: fallback.filter(|candidate| candidate.name != primary.name),
        dispatch,
        fft: FftKernelHints {
            tile_cols: request.fft_tile.max(request.cols.max(1)),
            radix: request.fft_radix.max(1),
            segments: request.fft_segments.max(1),
        },
        stages: match request.kind {
            RankKernelKind::TopK => STAGE_TOPK,
            RankKernelKind::MidK | RankKernelKind::BottomK => STAGE_MIDK,
        },
    }
}

/// Describe WGPU softmax/hardmax kernel selection for a shape.
#[must_use]
pub fn softmax_kernel_report(request: SoftmaxKernelRequest) -> SoftmaxKernelReport {
    let primary = if request.subgroup {
        descriptor("softmax_subgroup")
    } else {
        descriptor("softmax_workgroup")
    };
    let fallback = request
        .subgroup
        .then(|| descriptor("softmax_workgroup"))
        .filter(|candidate| candidate.name != primary.name);
    let mut flags = 0u32;
    if request.hardmax {
        flags |= 1 << 1;
    }
    if request.mask {
        flags |= 1 << 2;
    }
    SoftmaxKernelReport {
        request,
        primary,
        fallback,
        dispatch: default_dispatch(request.rows, request.cols),
        flags,
        stages: STAGE_SOFTMAX,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_contains_core_wgpu_families() {
        let names: Vec<_> = kernel_catalog().iter().map(|item| item.name).collect();
        assert!(names.contains(&"softmax_workgroup"));
        assert!(names.contains(&"topk_keepk_workgroup"));
        assert!(names.contains(&"midk_bottomk_apply_subgroup_v2"));
        assert!(names.contains(&"fused_attention_online"));
    }

    #[test]
    fn descriptor_matches_shader_basename() {
        let descriptor = kernel_descriptor("softmax_workgroup.wgsl").unwrap();
        assert_eq!(descriptor.name, "softmax_workgroup");
        assert!(descriptor.portable);
    }

    #[test]
    fn rank_report_prefers_subgroup_v2_for_two_stage_bottomk() {
        let report = rank_kernel_report(RankKernelRequest {
            kind: RankKernelKind::BottomK,
            rows: 8,
            cols: 1024,
            k: 32,
            subgroup: true,
            use_two_stage: true,
            fft_tile: 2048,
            fft_radix: 4,
            fft_segments: 2,
            compaction_tile: 512,
        });
        assert_eq!(report.primary.name, "midk_bottomk_apply_subgroup_v2");
        assert_eq!(report.dispatch.workgroups, (4, 8, 1));
        assert_eq!(report.fft.tile_cols, 2048);
    }

    #[test]
    fn softmax_report_sets_hardmax_mask_flags() {
        let report = softmax_kernel_report(SoftmaxKernelRequest {
            rows: 4,
            cols: 16,
            subgroup: true,
            hardmax: true,
            mask: true,
        });
        assert_eq!(report.primary.name, "softmax_subgroup");
        assert_eq!(report.fallback.unwrap().name, "softmax_workgroup");
        assert_eq!(report.flags, (1 << 1) | (1 << 2));
    }
}
