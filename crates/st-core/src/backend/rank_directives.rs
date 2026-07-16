// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Typed semantics for rank-k planner directives.
//!
//! KDSL, persisted tuning choices, and generated tables all transport compact
//! integer codes. This module is the only place where those codes acquire
//! runtime meaning; callers must not independently reconstruct the mapping.

use serde::{Deserialize, Serialize};
use std::str::FromStr;
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RankKind {
    #[serde(rename = "topk", alias = "top_k")]
    TopK,
    #[serde(rename = "midk", alias = "mid_k")]
    MidK,
    #[serde(rename = "bottomk", alias = "bottom_k")]
    BottomK,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum RankKindParseError {
    #[error("unknown rank kind '{value}', expected 'topk', 'midk', or 'bottomk'")]
    Unknown { value: String },
}

impl RankKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TopK => "topk",
            Self::MidK => "midk",
            Self::BottomK => "bottomk",
        }
    }
}

impl FromStr for RankKind {
    type Err = RankKindParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "topk" | "top_k" => Ok(Self::TopK),
            "midk" | "mid_k" => Ok(Self::MidK),
            "bottomk" | "bottom_k" => Ok(Self::BottomK),
            _ => Err(RankKindParseError::Unknown {
                value: value.to_owned(),
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TopKAlgorithm {
    #[default]
    Auto,
    Heap,
    Bitonic,
    KWay,
}

impl TopKAlgorithm {
    pub const fn code(self) -> u8 {
        match self {
            Self::Auto => 0,
            Self::Heap => 1,
            Self::Bitonic => 2,
            Self::KWay => 3,
        }
    }

    pub const fn merge_strategy(self) -> Option<(u32, u32)> {
        match self {
            Self::Auto => None,
            Self::Heap => Some((1, 1)),
            Self::Bitonic => Some((0, 3)),
            Self::KWay => Some((1, 2)),
        }
    }
}

impl TryFrom<u8> for TopKAlgorithm {
    type Error = RankDirectiveError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Auto),
            1 => Ok(Self::Heap),
            2 => Ok(Self::Bitonic),
            3 => Ok(Self::KWay),
            _ => Err(RankDirectiveError::InvalidValue {
                field: "algo",
                value,
                expected: "0 (auto), 1 (heap), 2 (bitonic), or 3 (k-way)",
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionMode {
    #[default]
    Auto,
    OneStage,
    TwoStage,
}

impl CompactionMode {
    pub const fn code(self) -> u8 {
        match self {
            Self::Auto => 0,
            Self::OneStage => 1,
            Self::TwoStage => 2,
        }
    }

    pub const fn use_two_stage(self) -> Option<bool> {
        match self {
            Self::Auto => None,
            Self::OneStage => Some(false),
            Self::TwoStage => Some(true),
        }
    }

    fn try_from_field(field: &'static str, value: u8) -> Result<Self, RankDirectiveError> {
        match value {
            0 => Ok(Self::Auto),
            1 => Ok(Self::OneStage),
            2 => Ok(Self::TwoStage),
            _ => Err(RankDirectiveError::InvalidValue {
                field,
                value,
                expected: "0 (auto), 1 (one-stage), or 2 (two-stage)",
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum RankDirectiveError {
    #[error("rank directive '{field}'={value} is invalid; expected {expected}")]
    InvalidValue {
        field: &'static str,
        value: u8,
        expected: &'static str,
    },
    #[error(
        "rank directives disagree on two-stage execution: use_2ce={use_two_stage}, {mode_field}={mode}"
    )]
    ConflictingTwoStage {
        use_two_stage: bool,
        mode_field: &'static str,
        mode: u8,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RankDirectiveResolution {
    pub use_two_stage: Option<bool>,
    pub merge_kind: Option<u32>,
    pub merge_detail: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankDirectives {
    pub topk_algorithm: TopKAlgorithm,
    pub midk_mode: CompactionMode,
    pub bottomk_mode: CompactionMode,
}

impl RankDirectives {
    pub fn try_from_codes(
        topk_algorithm: u8,
        midk_mode: u8,
        bottomk_mode: u8,
    ) -> Result<Self, RankDirectiveError> {
        Ok(Self {
            topk_algorithm: TopKAlgorithm::try_from(topk_algorithm)?,
            midk_mode: CompactionMode::try_from_field("midk", midk_mode)?,
            bottomk_mode: CompactionMode::try_from_field("bottomk", bottomk_mode)?,
        })
    }

    pub fn resolve(
        self,
        kind: RankKind,
        explicit_two_stage: Option<bool>,
    ) -> Result<RankDirectiveResolution, RankDirectiveError> {
        let (mode_field, mode) = match kind {
            RankKind::TopK => (None, CompactionMode::Auto),
            RankKind::MidK => (Some("midk"), self.midk_mode),
            RankKind::BottomK => (Some("bottomk"), self.bottomk_mode),
        };
        let mode_two_stage = mode.use_two_stage();
        if let (Some(explicit), Some(from_mode), Some(mode_field)) =
            (explicit_two_stage, mode_two_stage, mode_field)
        {
            if explicit != from_mode {
                return Err(RankDirectiveError::ConflictingTwoStage {
                    use_two_stage: explicit,
                    mode_field,
                    mode: mode.code(),
                });
            }
        }

        let (merge_kind, merge_detail) = if kind == RankKind::TopK {
            self.topk_algorithm
                .merge_strategy()
                .map_or((None, None), |(kind, detail)| (Some(kind), Some(detail)))
        } else {
            (None, None)
        };

        Ok(RankDirectiveResolution {
            use_two_stage: mode_two_stage.or(explicit_two_stage),
            merge_kind,
            merge_detail,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_algorithms_resolve_to_unified_merge_fields() {
        let directives = RankDirectives::try_from_codes(2, 0, 0).expect("valid codes");
        let resolved = directives
            .resolve(RankKind::TopK, None)
            .expect("compatible directives");
        assert_eq!(
            (resolved.merge_kind, resolved.merge_detail),
            (Some(0), Some(3))
        );
    }

    #[test]
    fn rank_specific_mode_controls_two_stage_execution() {
        let directives = RankDirectives::try_from_codes(0, 1, 2).expect("valid codes");
        assert_eq!(
            directives
                .resolve(RankKind::MidK, None)
                .unwrap()
                .use_two_stage,
            Some(false)
        );
        assert_eq!(
            directives
                .resolve(RankKind::BottomK, None)
                .unwrap()
                .use_two_stage,
            Some(true)
        );
    }

    #[test]
    fn conflicting_directives_fail_closed() {
        let directives = RankDirectives::try_from_codes(0, 0, 2).expect("valid codes");
        assert!(matches!(
            directives.resolve(RankKind::BottomK, Some(false)),
            Err(RankDirectiveError::ConflictingTwoStage {
                use_two_stage: false,
                mode_field: "bottomk",
                mode: 2,
            })
        ));
    }

    #[test]
    fn rank_kind_wire_names_match_the_runtime_contract() {
        assert_eq!(serde_json::to_string(&RankKind::TopK).unwrap(), r#""topk""#);
        assert_eq!(
            serde_json::from_str::<RankKind>(r#""bottom_k""#).unwrap(),
            RankKind::BottomK
        );
    }
}
