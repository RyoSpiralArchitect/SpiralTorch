// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Standard entry for TopK/MidK/BottomK across all backends.
//! This module plans the run (heuristics) and exposes a single surface the backends can implement.
//
//! Typical flow:
//!   let plan = plan_rank(RankKind::TopK, rows, cols, k, caps);
//!   execute_rank(&plan, backend_impl, tensors);
use crate::backend::device_caps::{DeviceCaps, DeviceCapsError};
use crate::backend::execution_plan::{AcceleratorFallback, ExecutionConfig};
use crate::backend::rank_directives::RankDirectiveError;
#[cfg(feature = "kdsl")]
use crate::backend::rank_directives::RankDirectives;
use crate::backend::spiralk_fft::{SpiralKFftPlan, SpiralKFftPlanError};
use crate::backend::unison::{self, Choice, RankKind};
use serde::Serialize;
use thiserror::Error;

pub const RANK_PLAN_CONTRACT_VERSION: &str = "spiraltorch.rank_plan.v1";
pub const RANK_PLAN_KIND: &str = "spiraltorch.rank_plan";
pub const RANK_PLAN_SEMANTIC_OWNER: &str = "st-core::ops::rank_entry";
pub const RANK_PLAN_SEMANTIC_BACKEND: &str = "rust";

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum RankPlanError {
    #[error("rank-plan dimension '{field}' must be positive")]
    ZeroDimension { field: &'static str },
    #[error("rank-plan k={k} exceeds cols={cols}")]
    KExceedsCols { k: u32, cols: u32 },
    #[error("rank-plan {field} element count does not fit this target")]
    ElementCountOverflow { field: &'static str },
    #[error("rank-plan choice '{field}' must be positive")]
    ZeroChoiceField { field: &'static str },
    #[error("rank-plan choice '{field}'={value} exceeds device limit {limit}")]
    ChoiceExceedsDeviceLimit {
        field: &'static str,
        value: u32,
        limit: u32,
    },
    #[error("rank-plan workgroup={workgroup} must be aligned to device lane_width={lane_width}")]
    WorkgroupNotLaneAligned { workgroup: u32, lane_width: u32 },
    #[error("rank-plan choice '{field}'={value} is invalid; expected {expected}")]
    InvalidChoiceValue {
        field: &'static str,
        value: u32,
        expected: &'static str,
    },
    #[error(
        "rank-plan choice subgroup={choice_subgroup} disagrees with device subgroup={device_subgroup}"
    )]
    SubgroupMismatch {
        choice_subgroup: bool,
        device_subgroup: bool,
    },
    #[error(
        "SpiralK overrides disagree on two-stage execution: u2={use_two_stage}, {mode_field}={mode}"
    )]
    ConflictingTwoStageOverrides {
        use_two_stage: bool,
        mode_field: &'static str,
        mode: u8,
    },
    #[error(transparent)]
    InvalidDeviceCaps(#[from] DeviceCapsError),
}

impl From<RankDirectiveError> for RankPlanError {
    fn from(error: RankDirectiveError) -> Self {
        match error {
            RankDirectiveError::InvalidValue {
                field,
                value,
                expected,
            } => Self::InvalidChoiceValue {
                field,
                value: u32::from(value),
                expected,
            },
            RankDirectiveError::ConflictingTwoStage {
                use_two_stage,
                mode_field,
                mode,
            } => Self::ConflictingTwoStageOverrides {
                use_two_stage,
                mode_field,
                mode,
            },
        }
    }
}

impl From<SpiralKFftPlanError> for RankPlanError {
    fn from(error: SpiralKFftPlanError) -> Self {
        match error {
            SpiralKFftPlanError::UnsupportedRadix { radix } => Self::InvalidChoiceValue {
                field: "fft_radix",
                value: radix,
                expected: "2 or 4",
            },
            SpiralKFftPlanError::TileTooSmall { tile_cols } => Self::InvalidChoiceValue {
                field: "fft_tile",
                value: tile_cols,
                expected: "a power of two greater than or equal to 2",
            },
            SpiralKFftPlanError::TileNotPowerOfTwo { tile_cols } => Self::InvalidChoiceValue {
                field: "fft_tile",
                value: tile_cols,
                expected: "a power of two greater than or equal to 2",
            },
            SpiralKFftPlanError::TileTooLarge { tile_cols } => Self::InvalidChoiceValue {
                field: "fft_tile",
                value: tile_cols,
                expected: "a power of two no greater than 1048576",
            },
            SpiralKFftPlanError::InvalidSegments { segments } => Self::InvalidChoiceValue {
                field: "fft_segments",
                value: segments,
                expected: "a value from 1 through 4",
            },
        }
    }
}

/// Explicit choice overrides accepted by the Rust semantic core.
///
/// Optional clients may construct this transport type, but validation and
/// interpretation remain owned by [`RankPlan::try_with_choice_overrides`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RankPlanChoiceOverrides {
    pub use_two_stage: Option<bool>,
    pub workgroup: Option<u32>,
    pub lanes: Option<u32>,
    pub channel_stride: Option<u32>,
    pub merge_kind: Option<u32>,
    pub merge_detail: Option<u32>,
    pub compaction_tile: Option<u32>,
    pub fft_tile: Option<u32>,
    pub fft_radix: Option<u32>,
    pub fft_segments: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct RankPlanDeviceCapsSnapshot {
    pub backend: &'static str,
    pub subgroup: bool,
    pub lane_width: u32,
    pub max_workgroup: u32,
    pub shared_mem_per_workgroup: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct RankPlanLatencyWindowSnapshot {
    pub target: u32,
    pub lower: u32,
    pub upper: u32,
    pub min_lane: u32,
    pub max_lane: u32,
    pub slack: u32,
    pub stride: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct RankPlanChoiceSnapshot {
    pub use_two_stage: bool,
    pub workgroup: u32,
    pub lanes: u32,
    pub channel_stride: u32,
    pub merge_kind: u32,
    pub merge_detail: u32,
    pub tile: u32,
    pub compaction_tile: u32,
    pub subgroup: bool,
    pub fft_tile: u32,
    pub fft_radix: u32,
    pub fft_segments: u32,
    pub latency_window: Option<RankPlanLatencyWindowSnapshot>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct RankPlanExecutionSnapshot {
    pub accelerator_fallback: &'static str,
    pub tensor_util_wgpu_min_values: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct RankPlanSnapshot {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub rank_kind: &'static str,
    pub rows: u32,
    pub cols: u32,
    pub k: u32,
    pub input_elements: u64,
    pub output_elements: u64,
    pub device_caps: RankPlanDeviceCapsSnapshot,
    pub choice: RankPlanChoiceSnapshot,
    pub execution: RankPlanExecutionSnapshot,
}

#[derive(Clone, Debug)]
pub struct RankPlan {
    pub kind: RankKind,
    pub rows: u32,
    pub cols: u32,
    pub k: u32,
    pub device_caps: DeviceCaps,
    pub choice: Choice,
    pub execution_config: ExecutionConfig,
}

impl RankPlan {
    /// Revalidates the captured problem, capability, and planner-choice contract.
    pub fn validate(&self) -> Result<(), RankPlanError> {
        validate_rank_request(self.rows, self.cols, self.k, self.device_caps)?;
        validate_rank_choice(self.kind, self.choice, self.device_caps)
    }

    /// Applies explicit planner overrides without silently clamping invalid values.
    pub fn try_with_choice_overrides(
        &self,
        overrides: RankPlanChoiceOverrides,
    ) -> Result<Self, RankPlanError> {
        for (field, value) in [
            ("workgroup", overrides.workgroup),
            ("lanes", overrides.lanes),
            ("compaction_tile", overrides.compaction_tile),
            ("fft_tile", overrides.fft_tile),
            ("fft_radix", overrides.fft_radix),
            ("fft_segments", overrides.fft_segments),
        ] {
            if value == Some(0) {
                return Err(RankPlanError::ZeroChoiceField { field });
            }
        }

        let mut updated = self.clone();
        if let Some(value) = overrides.use_two_stage {
            updated.choice.use_2ce = value;
        }
        if let Some(value) = overrides.workgroup {
            updated.choice.wg = value;
        }
        if let Some(value) = overrides.lanes {
            updated.choice.kl = value;
        }
        if let Some(value) = overrides.channel_stride {
            updated.choice.ch = value;
        }
        if let Some(value) = overrides.merge_kind {
            updated.choice.mk = value;
        }
        if let Some(value) = overrides.merge_detail {
            updated.choice.mkd = value;
        }
        if let Some(value) = overrides.compaction_tile {
            updated.choice.ctile = value;
        }
        if let Some(value) = overrides.fft_tile {
            updated.choice.fft_tile = value;
        }
        if let Some(value) = overrides.fft_radix {
            updated.choice.fft_radix = value;
        }
        if let Some(value) = overrides.fft_segments {
            updated.choice.fft_segments = value;
        }

        if overrides.lanes.is_some() || overrides.compaction_tile.is_some() {
            // This window is derived from both values; retaining it after a
            // hard rewrite would make the audit snapshot claim stale bounds.
            updated.choice.latency_window = None;
        }
        updated.validate()?;
        Ok(updated)
    }

    /// Builds the canonical SpiralK evaluation context from this validated plan.
    #[cfg(feature = "kdsl")]
    pub fn spiralk_context(&self) -> Result<st_kdsl::Ctx, RankPlanError> {
        self.validate()?;
        let subgroup_capacity = if self.choice.subgroup {
            self.choice.kl
        } else {
            1
        };
        let kernel_capacity = if self.k <= 1_024 {
            1
        } else if self.k <= 16_384 {
            2
        } else {
            3
        };
        let fft = self.fft_plan()?;
        Ok(st_kdsl::Ctx {
            r: self.rows,
            c: self.cols,
            k: self.k,
            sg: self.choice.subgroup,
            sgc: subgroup_capacity,
            kc: kernel_capacity,
            tile_cols: fft.tile_cols(),
            radix: fft.radix(),
            segments: fft.segments(),
        })
    }

    /// Interprets a SpiralK hard result through the Rust-owned rank contract.
    #[cfg(feature = "kdsl")]
    pub fn try_with_spiralk_hard(&self, hard: &st_kdsl::Hard) -> Result<Self, RankPlanError> {
        let directives = RankDirectives::try_from_codes(
            hard.algo.unwrap_or(0),
            hard.midk.unwrap_or(0),
            hard.bottomk.unwrap_or(0),
        )?;
        let resolved = directives.resolve(self.kind, hard.use_2ce)?;

        self.try_with_choice_overrides(RankPlanChoiceOverrides {
            use_two_stage: resolved.use_two_stage,
            workgroup: hard.wg,
            lanes: hard.kl,
            channel_stride: hard.ch,
            merge_kind: resolved.merge_kind,
            merge_detail: resolved.merge_detail,
            compaction_tile: hard.ctile,
            fft_tile: hard.tile_cols,
            fft_radix: hard.radix,
            fft_segments: hard.segments,
        })
    }

    /// Returns a serialization-safe audit snapshot shared by language clients.
    pub fn snapshot(&self) -> RankPlanSnapshot {
        let caps = self.device_caps;
        let choice = self.choice;
        RankPlanSnapshot {
            kind: RANK_PLAN_KIND,
            contract_version: RANK_PLAN_CONTRACT_VERSION,
            semantic_owner: RANK_PLAN_SEMANTIC_OWNER,
            semantic_backend: RANK_PLAN_SEMANTIC_BACKEND,
            rank_kind: self.kind.as_str(),
            rows: self.rows,
            cols: self.cols,
            k: self.k,
            input_elements: u64::from(self.rows) * u64::from(self.cols),
            output_elements: u64::from(self.rows) * u64::from(self.k),
            device_caps: RankPlanDeviceCapsSnapshot {
                backend: caps.backend.as_str(),
                subgroup: caps.subgroup,
                lane_width: caps.lane_width,
                max_workgroup: caps.max_workgroup,
                shared_mem_per_workgroup: caps.shared_mem_per_workgroup,
            },
            choice: RankPlanChoiceSnapshot {
                use_two_stage: choice.use_2ce,
                workgroup: choice.wg,
                lanes: choice.kl,
                channel_stride: choice.ch,
                merge_kind: choice.mk,
                merge_detail: choice.mkd,
                tile: choice.tile,
                compaction_tile: choice.ctile,
                subgroup: choice.subgroup,
                fft_tile: choice.fft_tile,
                fft_radix: choice.fft_radix,
                fft_segments: choice.fft_segments,
                latency_window: choice
                    .latency_window
                    .map(|window| RankPlanLatencyWindowSnapshot {
                        target: window.target,
                        lower: window.lower,
                        upper: window.upper,
                        min_lane: window.min_lane,
                        max_lane: window.max_lane,
                        slack: window.slack,
                        stride: window.stride,
                    }),
            },
            execution: RankPlanExecutionSnapshot {
                accelerator_fallback: self.execution_config.accelerator_fallback.as_str(),
                tensor_util_wgpu_min_values: self.execution_config.tensor_util_wgpu_min_values,
            },
        }
    }

    /// Build a SpiralK FFT plan that mirrors the heuristic choice associated with this rank plan.
    pub fn fft_plan(&self) -> Result<SpiralKFftPlan, RankPlanError> {
        SpiralKFftPlan::try_new(
            self.choice.fft_radix,
            self.choice.fft_tile,
            self.choice.fft_segments,
            self.choice.subgroup,
        )
        .map_err(Into::into)
    }

    /// Emit a WGSL kernel derived from the underlying FFT plan.
    pub fn fft_wgsl(&self) -> Result<String, RankPlanError> {
        Ok(self.fft_plan()?.emit_wgsl())
    }

    /// Emit the SpiralK hint associated with the FFT plan so DSL consumers can record the decision.
    pub fn fft_spiralk_hint(&self) -> Result<String, RankPlanError> {
        Ok(self.fft_plan()?.emit_spiralk_hint())
    }

    /// Returns the fallback contract captured when this plan was created.
    pub const fn accelerator_fallback(&self) -> AcceleratorFallback {
        self.execution_config.accelerator_fallback
    }
}

pub fn plan_rank(kind: RankKind, rows: u32, cols: u32, k: u32, caps: DeviceCaps) -> RankPlan {
    try_plan_rank(kind, rows, cols, k, caps).expect("rank-plan request must be valid")
}

/// Validates and plans rank-k while capturing the process execution contract.
pub fn try_plan_rank(
    kind: RankKind,
    rows: u32,
    cols: u32,
    k: u32,
    caps: DeviceCaps,
) -> Result<RankPlan, RankPlanError> {
    try_plan_rank_with_config(kind, rows, cols, k, caps, ExecutionConfig::from_env())
}

/// Plans rank-k with an explicit execution contract instead of consulting process state.
pub fn plan_rank_with_config(
    kind: RankKind,
    rows: u32,
    cols: u32,
    k: u32,
    caps: DeviceCaps,
    execution_config: ExecutionConfig,
) -> RankPlan {
    try_plan_rank_with_config(kind, rows, cols, k, caps, execution_config)
        .expect("rank-plan request must be valid")
}

/// Plans rank-k only after validating shape and device capability invariants.
pub fn try_plan_rank_with_config(
    kind: RankKind,
    rows: u32,
    cols: u32,
    k: u32,
    caps: DeviceCaps,
    execution_config: ExecutionConfig,
) -> Result<RankPlan, RankPlanError> {
    validate_rank_request(rows, cols, k, caps)?;
    let choice = unison::choose_unified_rank(rows, cols, k, caps, kind);
    let plan = RankPlan {
        kind,
        rows,
        cols,
        k,
        device_caps: caps,
        choice,
        execution_config,
    };
    plan.validate()?;
    Ok(plan)
}

fn validate_rank_request(
    rows: u32,
    cols: u32,
    k: u32,
    caps: DeviceCaps,
) -> Result<(), RankPlanError> {
    for (field, value) in [("rows", rows), ("cols", cols), ("k", k)] {
        if value == 0 {
            return Err(RankPlanError::ZeroDimension { field });
        }
    }
    if k > cols {
        return Err(RankPlanError::KExceedsCols { k, cols });
    }
    caps.validate()?;
    let rows = usize::try_from(rows)
        .map_err(|_| RankPlanError::ElementCountOverflow { field: "input" })?;
    let cols = usize::try_from(cols)
        .map_err(|_| RankPlanError::ElementCountOverflow { field: "input" })?;
    let k =
        usize::try_from(k).map_err(|_| RankPlanError::ElementCountOverflow { field: "output" })?;
    rows.checked_mul(cols)
        .ok_or(RankPlanError::ElementCountOverflow { field: "input" })?;
    rows.checked_mul(k)
        .ok_or(RankPlanError::ElementCountOverflow { field: "output" })?;
    Ok(())
}

fn validate_rank_choice(
    kind: RankKind,
    choice: Choice,
    caps: DeviceCaps,
) -> Result<(), RankPlanError> {
    for (field, value) in [
        ("workgroup", choice.wg),
        ("lanes", choice.kl),
        ("tile", choice.tile),
        ("fft_tile", choice.fft_tile),
        ("fft_radix", choice.fft_radix),
        ("fft_segments", choice.fft_segments),
    ] {
        if value == 0 {
            return Err(RankPlanError::ZeroChoiceField { field });
        }
    }
    if matches!(kind, RankKind::MidK | RankKind::BottomK) && choice.ctile == 0 {
        return Err(RankPlanError::ZeroChoiceField {
            field: "compaction_tile",
        });
    }
    if choice.wg > caps.max_workgroup {
        return Err(RankPlanError::ChoiceExceedsDeviceLimit {
            field: "workgroup",
            value: choice.wg,
            limit: caps.max_workgroup,
        });
    }
    if !choice.wg.is_multiple_of(caps.lane_width) {
        return Err(RankPlanError::WorkgroupNotLaneAligned {
            workgroup: choice.wg,
            lane_width: caps.lane_width,
        });
    }
    if choice.mk > 2 {
        return Err(RankPlanError::InvalidChoiceValue {
            field: "merge_kind",
            value: choice.mk,
            expected: "0 (bitonic), 1 (shared), or 2 (warp)",
        });
    }
    if choice.mkd > 5 {
        return Err(RankPlanError::InvalidChoiceValue {
            field: "merge_detail",
            value: choice.mkd,
            expected: "a value from 0 through 5",
        });
    }
    SpiralKFftPlan::try_new(
        choice.fft_radix,
        choice.fft_tile,
        choice.fft_segments,
        choice.subgroup,
    )?;
    if choice.subgroup != caps.subgroup {
        return Err(RankPlanError::SubgroupMismatch {
            choice_subgroup: choice.subgroup,
            device_subgroup: caps.subgroup,
        });
    }
    Ok(())
}

/// Trait that a backend implements to execute rank-k with a given plan.
pub trait RankKExecutor {
    type Error;
    fn launch_topk(&self, plan: &RankPlan) -> Result<(), Self::Error>;
    fn launch_midk(&self, plan: &RankPlan) -> Result<(), Self::Error>;
    fn launch_bottomk(&self, plan: &RankPlan) -> Result<(), Self::Error>;
}

/// Helper to dispatch by kind.
pub fn execute_rank<E: RankKExecutor>(exec: &E, plan: &RankPlan) -> Result<(), E::Error> {
    match plan.kind {
        RankKind::TopK => exec.launch_topk(plan),
        RankKind::MidK => exec.launch_midk(plan),
        RankKind::BottomK => exec.launch_bottomk(plan),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_plan_carries_subgroup_hint() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let plan = plan_rank(RankKind::TopK, 128, 8192, 64, caps);
        let fft = plan.fft_plan().expect("validated rank FFT plan");
        assert!(fft.tile_cols() >= 1024);
        assert!(matches!(fft.radix(), 2 | 4));
        assert!(fft.segments() >= 1);
        assert!(fft.subgroup());
    }

    #[test]
    fn fft_helpers_emit_artifacts() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let plan = plan_rank(RankKind::TopK, 64, 4096, 32, caps);
        let wgsl = plan.fft_wgsl().expect("validated WGSL plan");
        assert!(wgsl.contains("@compute"));
        let hint = plan.fft_spiralk_hint().expect("validated SpiralK FFT hint");
        assert!(hint.contains("tile_cols"));
    }

    #[test]
    fn explicit_execution_config_is_captured_by_the_plan() {
        let config = ExecutionConfig::new(AcceleratorFallback::Forbid, 4096);
        let plan = plan_rank_with_config(
            RankKind::MidK,
            4,
            128,
            8,
            DeviceCaps::wgpu(32, true, 256),
            config,
        );
        assert_eq!(plan.execution_config, config);
        assert_eq!(plan.accelerator_fallback(), AcceleratorFallback::Forbid);
    }

    #[test]
    fn checked_planner_rejects_invalid_problem_shapes() {
        let caps = DeviceCaps::cpu();
        assert!(matches!(
            try_plan_rank(RankKind::TopK, 0, 8, 2, caps),
            Err(RankPlanError::ZeroDimension { field: "rows" })
        ));
        assert!(matches!(
            try_plan_rank(RankKind::TopK, 2, 8, 9, caps),
            Err(RankPlanError::KExceedsCols { k: 9, cols: 8 })
        ));
    }

    #[test]
    fn checked_planner_rejects_invalid_capabilities() {
        let caps = DeviceCaps {
            backend: crate::backend::device_caps::BackendKind::Wgpu,
            subgroup: true,
            lane_width: 64,
            max_workgroup: 32,
            shared_mem_per_workgroup: None,
        };
        assert!(matches!(
            try_plan_rank(RankKind::TopK, 2, 8, 2, caps),
            Err(RankPlanError::InvalidDeviceCaps(
                DeviceCapsError::LaneWidthExceedsMaxWorkgroup {
                    lane_width: 64,
                    max_workgroup: 32,
                }
            ))
        ));
    }

    #[test]
    fn checked_planner_respects_small_device_workgroup_limits() {
        let plan = try_plan_rank(RankKind::TopK, 2, 128, 8, DeviceCaps::wgpu(32, true, 32))
            .expect("small but valid WGPU capability contract");
        assert_eq!(plan.choice.wg, 32);
        plan.validate().expect("planner output remains valid");
    }

    #[test]
    fn explicit_choice_overrides_fail_closed() {
        let plan = try_plan_rank(RankKind::MidK, 4, 128, 8, DeviceCaps::wgpu(32, true, 256))
            .expect("valid baseline");
        assert!(matches!(
            plan.try_with_choice_overrides(RankPlanChoiceOverrides {
                workgroup: Some(0),
                ..RankPlanChoiceOverrides::default()
            }),
            Err(RankPlanError::ZeroChoiceField { field: "workgroup" })
        ));
        assert!(matches!(
            plan.try_with_choice_overrides(RankPlanChoiceOverrides {
                workgroup: Some(48),
                ..RankPlanChoiceOverrides::default()
            }),
            Err(RankPlanError::WorkgroupNotLaneAligned {
                workgroup: 48,
                lane_width: 32,
            })
        ));
        assert!(matches!(
            plan.try_with_choice_overrides(RankPlanChoiceOverrides {
                fft_radix: Some(3),
                ..RankPlanChoiceOverrides::default()
            }),
            Err(RankPlanError::InvalidChoiceValue {
                field: "fft_radix",
                value: 3,
                ..
            })
        ));
    }

    #[cfg(feature = "kdsl")]
    #[test]
    fn spiralk_hard_overrides_use_rank_specific_rust_semantics() {
        let top = try_plan_rank(
            RankKind::TopK,
            64,
            4_096,
            32,
            DeviceCaps::wgpu(32, true, 256),
        )
        .expect("valid top-k baseline");
        let top = top
            .try_with_spiralk_hard(&st_kdsl::Hard {
                algo: Some(2),
                tile_cols: Some(2_048),
                radix: Some(2),
                segments: Some(2),
                ..st_kdsl::Hard::default()
            })
            .expect("valid SpiralK override");
        assert_eq!((top.choice.mk, top.choice.mkd), (0, 3));
        assert_eq!(top.choice.fft_tile, 2_048);
        assert_eq!(top.spiralk_context().expect("context").tile_cols, 2_048);

        let bottom = try_plan_rank(
            RankKind::BottomK,
            256,
            65_536,
            1_024,
            DeviceCaps::wgpu(32, true, 256),
        )
        .expect("valid bottom-k baseline");
        let bottom = bottom
            .try_with_spiralk_hard(&st_kdsl::Hard {
                bottomk: Some(2),
                ..st_kdsl::Hard::default()
            })
            .expect("valid bottom-k mode override");
        assert!(bottom.choice.use_2ce);
    }

    #[cfg(feature = "kdsl")]
    #[test]
    fn spiralk_hard_overrides_reject_ambiguous_or_invalid_values() {
        let plan = try_plan_rank(
            RankKind::BottomK,
            256,
            65_536,
            1_024,
            DeviceCaps::wgpu(32, true, 256),
        )
        .expect("valid baseline");
        assert!(matches!(
            plan.try_with_spiralk_hard(&st_kdsl::Hard {
                use_2ce: Some(false),
                bottomk: Some(2),
                ..st_kdsl::Hard::default()
            }),
            Err(RankPlanError::ConflictingTwoStageOverrides {
                use_two_stage: false,
                mode_field: "bottomk",
                mode: 2,
            })
        ));
        assert!(matches!(
            plan.try_with_spiralk_hard(&st_kdsl::Hard {
                radix: Some(3),
                ..st_kdsl::Hard::default()
            }),
            Err(RankPlanError::InvalidChoiceValue {
                field: "fft_radix",
                value: 3,
                ..
            })
        ));
    }

    #[test]
    fn snapshot_carries_the_complete_rust_owned_contract() {
        let config = ExecutionConfig::new(AcceleratorFallback::Forbid, 4_096);
        let plan = try_plan_rank_with_config(
            RankKind::MidK,
            4,
            128,
            8,
            DeviceCaps::wgpu(32, true, 256),
            config,
        )
        .expect("valid rank plan");
        let snapshot = plan.snapshot();

        assert_eq!(snapshot.kind, RANK_PLAN_KIND);
        assert_eq!(snapshot.contract_version, RANK_PLAN_CONTRACT_VERSION);
        assert_eq!(snapshot.semantic_owner, RANK_PLAN_SEMANTIC_OWNER);
        assert_eq!(snapshot.semantic_backend, RANK_PLAN_SEMANTIC_BACKEND);
        assert_eq!(snapshot.rank_kind, "midk");
        assert_eq!(snapshot.input_elements, 512);
        assert_eq!(snapshot.output_elements, 32);
        assert_eq!(snapshot.device_caps.backend, "wgpu");
        assert_eq!(snapshot.execution.accelerator_fallback, "forbid");
        assert_eq!(snapshot.execution.tensor_util_wgpu_min_values, 4_096);
        assert!(snapshot.choice.workgroup > 0);
        assert!(snapshot.choice.latency_window.is_some());
        plan.validate().expect("captured plan remains valid");
    }
}
