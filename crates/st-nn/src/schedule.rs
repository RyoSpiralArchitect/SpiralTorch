// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::execution::current_tensor_util_backend_for_values;
use crate::plan::RankPlanner;
use crate::{PureResult, Tensor};
use st_core::backend::unison_heuristics::Choice;
use st_core::heur::free_energy::{BandEnergy as CoreBandEnergy, FreeEnergyError};
use st_core::ops::rank_entry::{RankPlan, RankPlanError};
use st_core::ops::zspace_round::{self, RoundtableBand, RoundtableError, SpectralFeatureSample};
use st_tensor::TensorError;
use std::collections::{BTreeSet, HashMap};
use thiserror::Error;

/// Failure to map an Autopilot control into an executable roundtable choice.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum KnobOverrideError {
    #[error("roundtable override key {key:?} is empty")]
    EmptyKey { key: String },
    #[error("roundtable override '{key}' has an empty value")]
    EmptyValue { key: String },
    #[error("roundtable override '{key}' names an unsupported knob")]
    UnsupportedKnob { key: String },
    #[error("roundtable override '{key}' has invalid value {value:?}")]
    InvalidValue { key: String, value: String },
    #[error("roundtable override '{key}' assigns one band knob more than once")]
    DuplicateAssignment { key: String },
    #[error("roundtable override produced an invalid {band} rank plan: {source}")]
    InvalidPlan {
        band: &'static str,
        #[source]
        source: RankPlanError,
    },
}

/// Configuration used to derive the A/B/C roundtable schedule.
#[derive(Debug, Clone, Copy)]
pub struct RoundtableConfig {
    pub top_k: u32,
    pub mid_k: u32,
    pub bottom_k: u32,
    pub here_tolerance: f32,
    #[cfg(feature = "psychoid")]
    pub psychoid_enabled: bool,
    #[cfg(feature = "psychoid")]
    pub psychoid_log: bool,
    #[cfg(feature = "psi")]
    pub psi_enabled: bool,
    #[cfg(feature = "collapse")]
    pub collapse_enabled: bool,
}

impl Default for RoundtableConfig {
    fn default() -> Self {
        Self {
            top_k: 8,
            mid_k: 8,
            bottom_k: 8,
            here_tolerance: 1e-5,
            #[cfg(feature = "psychoid")]
            psychoid_enabled: false,
            #[cfg(feature = "psychoid")]
            psychoid_log: false,
            #[cfg(feature = "psi")]
            psi_enabled: false,
            #[cfg(feature = "collapse")]
            collapse_enabled: false,
        }
    }
}

impl RoundtableConfig {
    /// Overrides the TopK count.
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = top_k;
        self
    }

    /// Overrides the MidK count.
    pub fn with_mid_k(mut self, mid_k: u32) -> Self {
        self.mid_k = mid_k;
        self
    }

    /// Overrides the BottomK count.
    pub fn with_bottom_k(mut self, bottom_k: u32) -> Self {
        self.bottom_k = bottom_k;
        self
    }

    /// Overrides the tolerance that decides which entries stay in-place.
    pub fn with_here_tolerance(mut self, tol: f32) -> Self {
        self.here_tolerance = tol.max(0.0);
        self
    }

    #[cfg(feature = "psychoid")]
    pub fn enable_psychoid(mut self) -> Self {
        self.psychoid_enabled = true;
        self
    }

    #[cfg(feature = "psychoid")]
    pub fn enable_psychoid_with_log(mut self) -> Self {
        self.psychoid_enabled = true;
        self.psychoid_log = true;
        self
    }

    #[cfg(feature = "psi")]
    pub fn enable_psi(mut self) -> Self {
        self.psi_enabled = true;
        self
    }

    #[cfg(feature = "collapse")]
    pub fn enable_collapse(mut self) -> Self {
        self.collapse_enabled = true;
        #[cfg(feature = "psi")]
        {
            if !self.psi_enabled {
                self.psi_enabled = true;
            }
        }
        self
    }
}

/// Roundtable schedule that binds TopK (Above), MidK (Here), and BottomK
/// (Beneath) ranks to gradient bands.
#[derive(Debug, Clone)]
pub struct RoundtableSchedule {
    above: RankPlan,
    here: RankPlan,
    beneath: RankPlan,
    here_tolerance: f32,
    #[cfg(feature = "psychoid")]
    psychoid_enabled: bool,
    #[cfg(feature = "psychoid")]
    psychoid_log: bool,
    #[cfg(feature = "psi")]
    psi_enabled: bool,
    #[cfg(feature = "collapse")]
    collapse_enabled: bool,
}

impl RoundtableSchedule {
    /// Builds a schedule for the provided output shape.
    pub fn new(planner: &RankPlanner, rows: u32, cols: u32, config: RoundtableConfig) -> Self {
        let limit = cols.max(1);
        let clamp_k = |k: u32| -> u32 {
            let capped = k.max(1).min(limit);
            debug_assert!(capped >= 1 && capped <= limit);
            capped
        };
        let above = planner.topk(rows, cols, clamp_k(config.top_k));
        let here = planner.midk(rows, cols, clamp_k(config.mid_k));
        let beneath = planner.bottomk(rows, cols, clamp_k(config.bottom_k));
        Self {
            above,
            here,
            beneath,
            here_tolerance: config.here_tolerance,
            #[cfg(feature = "psychoid")]
            psychoid_enabled: config.psychoid_enabled,
            #[cfg(feature = "psychoid")]
            psychoid_log: config.psychoid_log,
            #[cfg(feature = "psi")]
            psi_enabled: config.psi_enabled,
            #[cfg(feature = "collapse")]
            collapse_enabled: config.collapse_enabled && config.psi_enabled,
        }
    }

    /// Applies knob overrides produced by the Autopilot runtime.
    ///
    /// Keys may optionally be scoped to a specific band using prefixes such as
    /// `"above."`, `"here."`, or `"beneath."`. Unscoped keys are applied to all
    /// bands. Invalid override sets leave the schedule unchanged.
    pub fn apply_knob_overrides(&mut self, overrides: &HashMap<String, String>) {
        let _ = self.try_apply_knob_overrides(overrides);
    }

    /// Applies a complete override set transactionally. Every key and value
    /// must map to an executable choice before any schedule field is changed.
    pub fn try_apply_knob_overrides(
        &mut self,
        overrides: &HashMap<String, String>,
    ) -> Result<usize, KnobOverrideError> {
        let mut next = self.clone();
        let mut assignments = BTreeSet::new();
        let mut applied = 0usize;
        let mut ordered = overrides.iter().collect::<Vec<_>>();
        ordered.sort_by(|left, right| left.0.cmp(right.0));
        for (raw_key, raw_value) in ordered {
            let value = raw_value.trim();
            if value.is_empty() {
                return Err(KnobOverrideError::EmptyValue {
                    key: raw_key.clone(),
                });
            }

            let key_lower = raw_key.trim().to_ascii_lowercase();
            if key_lower.is_empty() {
                return Err(KnobOverrideError::EmptyKey {
                    key: raw_key.clone(),
                });
            }
            let (targets, knob_key) = if let Some(rest) = key_lower.strip_prefix("above.") {
                (OverrideTarget::specific(OverrideBand::Above), rest)
            } else if let Some(rest) = key_lower.strip_prefix("here.") {
                (OverrideTarget::specific(OverrideBand::Here), rest)
            } else if let Some(rest) = key_lower.strip_prefix("beneath.") {
                (OverrideTarget::specific(OverrideBand::Beneath), rest)
            } else {
                (OverrideTarget::all(), key_lower.as_str())
            };

            let canonical = canonical_knob_key(knob_key);
            if canonical.is_empty() {
                return Err(KnobOverrideError::EmptyKey {
                    key: raw_key.clone(),
                });
            }
            let knob = normalized_knob_key(&canonical).ok_or_else(|| {
                KnobOverrideError::UnsupportedKnob {
                    key: raw_key.clone(),
                }
            })?;

            for band in targets.into_iter() {
                if !assignments.insert((band, knob)) {
                    return Err(KnobOverrideError::DuplicateAssignment {
                        key: raw_key.clone(),
                    });
                }
                match band {
                    OverrideBand::Above => {
                        apply_choice_override(&mut next.above.choice, knob, value, raw_key)?
                    }
                    OverrideBand::Here => {
                        apply_choice_override(&mut next.here.choice, knob, value, raw_key)?
                    }
                    OverrideBand::Beneath => {
                        apply_choice_override(&mut next.beneath.choice, knob, value, raw_key)?
                    }
                }
                applied = applied.saturating_add(1);
            }
        }
        for (band, plan) in [
            ("above", &next.above),
            ("here", &next.here),
            ("beneath", &next.beneath),
        ] {
            plan.validate()
                .map_err(|source| KnobOverrideError::InvalidPlan { band, source })?;
        }
        *self = next;
        Ok(applied)
    }

    /// Returns the TopK plan (Above band).
    pub fn above(&self) -> &RankPlan {
        &self.above
    }

    /// Returns the MidK plan (Here band).
    pub fn here(&self) -> &RankPlan {
        &self.here
    }

    /// Returns the BottomK plan (Beneath band).
    pub fn beneath(&self) -> &RankPlan {
        &self.beneath
    }

    /// Splits a gradient tensor into Above/Here/Beneath bands.
    pub fn split(&self, gradient: &Tensor) -> PureResult<GradientBands> {
        let len = gradient.data().len();
        let assignment = match zspace_round::classify_roundtable(
            gradient.data(),
            &self.above,
            &self.here,
            &self.beneath,
            self.here_tolerance,
        ) {
            Ok(assign) => assign,
            Err(RoundtableError::EmptyGradient) => {
                return Err(TensorError::DataLength {
                    expected: 1,
                    got: 0,
                })
            }
        };

        let (rows, cols) = gradient.shape();
        let mut above_data = vec![0.0f32; len];
        let mut here_data = vec![0.0f32; len];
        let mut beneath_data = vec![0.0f32; len];
        for (idx, value) in gradient.data().iter().enumerate() {
            match assignment.band(idx) {
                RoundtableBand::Above => above_data[idx] = *value,
                RoundtableBand::Here => here_data[idx] = *value,
                RoundtableBand::Beneath => beneath_data[idx] = *value,
            }
        }

        Ok(GradientBands {
            above: Tensor::from_vec(rows, cols, above_data)?,
            here: Tensor::from_vec(rows, cols, here_data)?,
            beneath: Tensor::from_vec(rows, cols, beneath_data)?,
        })
    }

    /// Returns the absolute sum of each band without allocating tensors.
    pub fn band_energy(&self, gradient: &Tensor) -> PureResult<BandEnergy> {
        let assignment = zspace_round::classify_roundtable(
            gradient.data(),
            &self.above,
            &self.here,
            &self.beneath,
            self.here_tolerance,
        )
        .map_err(|err| match err {
            RoundtableError::EmptyGradient => TensorError::DataLength {
                expected: 1,
                got: 0,
            },
        })?;
        let (above, here, beneath) = assignment.energy(gradient.data());
        let spectral = zspace_round::roundtable_spectral_features(
            gradient.data(),
            &self.above,
            &self.here,
            &self.beneath,
        );
        Ok(BandEnergy::new(above, here, beneath).with_spectral(spectral))
    }

    #[cfg(feature = "psychoid")]
    pub fn psychoid_enabled(&self) -> bool {
        self.psychoid_enabled
    }

    #[cfg(feature = "psychoid")]
    pub fn psychoid_log(&self) -> bool {
        self.psychoid_log
    }

    #[cfg(feature = "psi")]
    pub fn psi_enabled(&self) -> bool {
        self.psi_enabled
    }

    #[cfg(feature = "psi")]
    pub fn psi_hint(&self) -> st_core::telemetry::psi::PsiAutomationHint {
        use st_core::telemetry::psi::PsiAutomationHint;
        let depth = self.above.k + self.here.k + self.beneath.k;
        let band_focus =
            self.above.k as f32 * 1.4 + self.here.k as f32 + self.beneath.k as f32 * 0.6;
        let drift_weight = if depth == 0 {
            0.2
        } else {
            (self.here.k as f32 / depth as f32 * 0.35 + 0.12).clamp(0.1, 0.45)
        };
        PsiAutomationHint {
            above: self.above.k,
            here: self.here.k,
            beneath: self.beneath.k,
            band_focus,
            drift_weight,
        }
    }

    #[cfg(feature = "collapse")]
    pub fn collapse_enabled(&self) -> bool {
        self.collapse_enabled
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum OverrideBand {
    Above,
    Here,
    Beneath,
}

struct OverrideTarget(Vec<OverrideBand>);

impl OverrideTarget {
    fn all() -> Self {
        Self(vec![
            OverrideBand::Above,
            OverrideBand::Here,
            OverrideBand::Beneath,
        ])
    }

    fn specific(band: OverrideBand) -> Self {
        Self(vec![band])
    }
}

impl IntoIterator for OverrideTarget {
    type Item = OverrideBand;
    type IntoIter = std::vec::IntoIter<OverrideBand>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

fn canonical_knob_key(key: &str) -> String {
    key.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
}

fn normalized_knob_key(key: &str) -> Option<&'static str> {
    match key {
        "wg" | "workgroup" => Some("wg"),
        "kl" | "lanes" => Some("kl"),
        "ch" | "channelstride" => Some("ch"),
        "mk" | "mergekind" => Some("mk"),
        "mkd" | "mergedetail" => Some("mkd"),
        "tile" | "topktile" | "toptile" => Some("tile"),
        "ctile" | "compactiontile" | "midtile" | "bottomtile" => Some("ctile"),
        "subgroup" => Some("subgroup"),
        "ffttile" | "ffttilecols" | "fftcolumns" => Some("ffttile"),
        "fftradix" => Some("fftradix"),
        "fftsegments" | "fftseg" => Some("fftsegments"),
        _ => None,
    }
}

fn apply_choice_override(
    choice: &mut Choice,
    key: &str,
    value: &str,
    raw_key: &str,
) -> Result<(), KnobOverrideError> {
    let invalid = || KnobOverrideError::InvalidValue {
        key: raw_key.to_string(),
        value: value.to_string(),
    };
    match key {
        "wg" => {
            choice.wg = parse_positive_u32(value).ok_or_else(invalid)?;
        }
        "kl" => {
            choice.kl = parse_positive_u32(value).ok_or_else(invalid)?;
        }
        "ch" => {
            choice.ch = parse_u32(value).ok_or_else(invalid)?;
        }
        "mk" => {
            choice.mk = parse_merge_kind(value)
                .filter(|parsed| *parsed <= 2)
                .ok_or_else(invalid)?;
        }
        "mkd" => {
            choice.mkd = parse_merge_detail(value)
                .filter(|parsed| *parsed <= 5)
                .ok_or_else(invalid)?;
        }
        "tile" => {
            choice.tile = parse_positive_u32(value).ok_or_else(invalid)?;
        }
        "ctile" => {
            choice.ctile = parse_positive_u32(value).ok_or_else(invalid)?;
        }
        "subgroup" => {
            choice.subgroup = parse_bool(value).ok_or_else(invalid)?;
        }
        "ffttile" => {
            choice.fft_tile = parse_positive_u32(value).ok_or_else(invalid)?;
        }
        "fftradix" => match parse_u32(value) {
            Some(parsed @ (2 | 4)) => choice.fft_radix = parsed,
            _ => return Err(invalid()),
        },
        "fftsegments" => {
            choice.fft_segments = parse_positive_u32(value).ok_or_else(invalid)?;
        }
        _ => unreachable!("knob aliases are normalized before application"),
    }
    Ok(())
}

fn parse_u32(value: &str) -> Option<u32> {
    value.trim().parse::<u32>().ok()
}

fn parse_positive_u32(value: &str) -> Option<u32> {
    parse_u32(value).filter(|value| *value > 0)
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn parse_merge_kind(value: &str) -> Option<u32> {
    match value.trim().to_ascii_lowercase().as_str() {
        "bitonic" => Some(0),
        "shared" => Some(1),
        "warp" => Some(2),
        other => other.parse::<u32>().ok(),
    }
}

fn parse_merge_detail(value: &str) -> Option<u32> {
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" => Some(0),
        "heap" => Some(1),
        "kway" => Some(2),
        "bitonic" => Some(3),
        "warpheap" | "warp_heap" => Some(4),
        "warpbitonic" | "warp_bitonic" => Some(5),
        other => other.parse::<u32>().ok(),
    }
}

/// Gradient components that mirror the roundtable decisions.
#[derive(Debug, Clone)]
pub struct GradientBands {
    above: Tensor,
    here: Tensor,
    beneath: Tensor,
}

impl GradientBands {
    /// Builds a gradient triplet from explicit Above/Here/Beneath tensors.
    pub fn from_components(above: Tensor, here: Tensor, beneath: Tensor) -> PureResult<Self> {
        let shape = above.shape();
        if here.shape() != shape {
            return Err(TensorError::ShapeMismatch {
                left: here.shape(),
                right: shape,
            });
        }
        if beneath.shape() != shape {
            return Err(TensorError::ShapeMismatch {
                left: beneath.shape(),
                right: shape,
            });
        }
        Ok(Self {
            above,
            here,
            beneath,
        })
    }

    /// Returns the gradient for the Above (TopK/A) band.
    pub fn above(&self) -> &Tensor {
        &self.above
    }

    /// Returns the gradient for the Here (MidK/B) band.
    pub fn here(&self) -> &Tensor {
        &self.here
    }

    /// Returns the gradient for the Beneath (BottomK/C) band.
    pub fn beneath(&self) -> &Tensor {
        &self.beneath
    }

    /// Returns an iterator over every band.
    pub fn iter(&self) -> [&Tensor; 3] {
        [&self.above, &self.here, &self.beneath]
    }

    /// Returns every band paired with its semantic roundtable label.
    pub fn iter_labeled(&self) -> [(RoundtableBand, &Tensor); 3] {
        [
            (RoundtableBand::Above, &self.above),
            (RoundtableBand::Here, &self.here),
            (RoundtableBand::Beneath, &self.beneath),
        ]
    }

    /// Combines all bands back into a single tensor.
    pub fn combine(&self) -> PureResult<Tensor> {
        let (rows, cols) = self.above.shape();
        let mut data = Vec::with_capacity(rows * cols);
        for idx in 0..(rows * cols) {
            let value = self.above.data()[idx] + self.here.data()[idx] + self.beneath.data()[idx];
            data.push(value);
        }
        Tensor::from_vec(rows, cols, data)
    }

    /// Stacks the Above/Here/Beneath tensors along a new depth axis represented
    /// by contiguous row blocks.
    pub fn stack_depth(&self) -> PureResult<Tensor> {
        let (rows, cols) = self.above.shape();
        let mut data = Vec::with_capacity(rows * cols * 3);
        data.extend_from_slice(self.above.data());
        data.extend_from_slice(self.here.data());
        data.extend_from_slice(self.beneath.data());
        Tensor::from_vec(rows * 3, cols, data)
    }

    /// Applies per-band scaling factors through the tensor utility backend.
    pub fn scale_inplace(&mut self, above: f32, here: f32, beneath: f32) -> PureResult<()> {
        let above_scaled = self.above.scale_with_backend(
            above,
            current_tensor_util_backend_for_values(self.above.data().len()),
        )?;
        let here_scaled = self.here.scale_with_backend(
            here,
            current_tensor_util_backend_for_values(self.here.data().len()),
        )?;
        let beneath_scaled = self.beneath.scale_with_backend(
            beneath,
            current_tensor_util_backend_for_values(self.beneath.data().len()),
        )?;
        self.above = above_scaled;
        self.here = here_scaled;
        self.beneath = beneath_scaled;
        Ok(())
    }
}

/// Aggregate magnitude per roundtable band.
#[derive(Debug, Clone, Copy)]
pub struct BandEnergy {
    pub above: f32,
    pub here: f32,
    pub beneath: f32,
    pub drift: f32,
    pub spectral: SpectralFeatureSample,
}

impl BandEnergy {
    pub fn new(above: f32, here: f32, beneath: f32) -> Self {
        Self {
            above,
            here,
            beneath,
            drift: 0.0,
            spectral: SpectralFeatureSample::default(),
        }
    }

    pub fn with_drift(mut self, drift: f32) -> Self {
        self.drift = drift;
        self
    }

    pub fn with_spectral(mut self, spectral: SpectralFeatureSample) -> Self {
        self.spectral = spectral;
        self
    }

    pub fn spectral_focus(&self) -> f32 {
        self.spectral.sheet_confidence.clamp(0.0, 1.0)
    }

    pub fn spectral_curvature(&self) -> f32 {
        (self.spectral.curvature / 4.0).clamp(0.0, 1.0)
    }

    pub fn spectral_stability(&self) -> f32 {
        (self.spectral.spin.abs() * (1.0 - self.spectral_curvature())).clamp(0.0, 1.0)
    }

    /// Returns the L1 norm of the band magnitudes.
    pub fn l1(&self) -> f32 {
        self.free_energy_band().l1()
    }

    /// Projects the roundtable signal into the canonical Rust free-energy type.
    pub fn free_energy_band(&self) -> CoreBandEnergy {
        CoreBandEnergy {
            above: self.above,
            here: self.here,
            beneath: self.beneath,
        }
    }

    /// Strictly normalises finite, non-negative band energy while preserving
    /// the higher-layer drift and spectral annotations.
    pub fn try_norm(self) -> Result<Self, FreeEnergyError> {
        let normalized = self.free_energy_band().try_norm()?;
        Ok(Self {
            above: normalized.above,
            here: normalized.here,
            beneath: normalized.beneath,
            drift: self.drift,
            spectral: self.spectral,
        })
    }

    /// Compatibility normalisation through the canonical Rust primitive.
    /// Guarded training paths validate first and may use [`Self::try_norm`].
    pub fn norm(self) -> Self {
        let normalized = self.free_energy_band().norm();
        Self {
            above: normalized.above,
            here: normalized.here,
            beneath: normalized.beneath,
            drift: self.drift,
            spectral: self.spectral,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo_schedule() -> RoundtableSchedule {
        let planner = RankPlanner::new(st_core::backend::device_caps::DeviceCaps::wgpu(
            32, true, 256,
        ));
        RoundtableSchedule::new(&planner, 8, 16, RoundtableConfig::default())
    }

    fn choice_witness(schedule: &RoundtableSchedule) -> [(u32, u32, u32, bool); 3] {
        [
            (
                schedule.above().choice.wg,
                schedule.above().choice.tile,
                schedule.above().choice.fft_radix,
                schedule.above().choice.subgroup,
            ),
            (
                schedule.here().choice.wg,
                schedule.here().choice.tile,
                schedule.here().choice.fft_radix,
                schedule.here().choice.subgroup,
            ),
            (
                schedule.beneath().choice.wg,
                schedule.beneath().choice.tile,
                schedule.beneath().choice.fft_radix,
                schedule.beneath().choice.subgroup,
            ),
        ]
    }

    #[test]
    fn guarded_knob_overrides_apply_every_declared_target() {
        let mut schedule = demo_schedule();
        let applied = schedule
            .try_apply_knob_overrides(&HashMap::from([
                ("wg".to_string(), "64".to_string()),
                ("here.tile".to_string(), "42".to_string()),
                ("beneath.fftradix".to_string(), "2".to_string()),
            ]))
            .expect("valid overrides");
        assert_eq!(applied, 5);
        assert_eq!(schedule.above().choice.wg, 64);
        assert_eq!(schedule.here().choice.wg, 64);
        assert_eq!(schedule.beneath().choice.wg, 64);
        assert_eq!(schedule.here().choice.tile, 42);
        assert_eq!(schedule.beneath().choice.fft_radix, 2);
    }

    #[test]
    fn guarded_knob_overrides_are_transactional_on_invalid_values() {
        let mut schedule = demo_schedule();
        let before = choice_witness(&schedule);
        let error = schedule
            .try_apply_knob_overrides(&HashMap::from([
                ("wg".to_string(), "64".to_string()),
                ("here.tile".to_string(), "0".to_string()),
            ]))
            .expect_err("zero tile must fail");
        assert!(matches!(error, KnobOverrideError::InvalidValue { .. }));
        assert_eq!(choice_witness(&schedule), before);
    }

    #[test]
    fn guarded_knob_overrides_reject_non_executable_rank_plans() {
        let mut schedule = demo_schedule();
        let before = choice_witness(&schedule);
        let error = schedule
            .try_apply_knob_overrides(&HashMap::from([("wg".to_string(), "48".to_string())]))
            .expect_err("workgroups must remain lane aligned");
        assert!(matches!(error, KnobOverrideError::InvalidPlan { .. }));
        assert_eq!(choice_witness(&schedule), before);
    }

    #[test]
    fn guarded_knob_overrides_reject_unknown_or_duplicate_assignments() {
        let mut schedule = demo_schedule();
        let before = choice_witness(&schedule);
        let unknown = schedule
            .try_apply_knob_overrides(&HashMap::from([("unknown".to_string(), "1".to_string())]))
            .expect_err("unknown knob must fail");
        assert!(matches!(unknown, KnobOverrideError::UnsupportedKnob { .. }));
        let duplicate = schedule
            .try_apply_knob_overrides(&HashMap::from([
                ("wg".to_string(), "64".to_string()),
                ("workgroup".to_string(), "128".to_string()),
            ]))
            .expect_err("aliases must not assign the same knob twice");
        assert!(matches!(
            duplicate,
            KnobOverrideError::DuplicateAssignment { .. }
        ));
        assert_eq!(choice_witness(&schedule), before);
    }

    #[test]
    fn schedule_splits_gradients() {
        let planner = RankPlanner::new(st_core::backend::device_caps::DeviceCaps::wgpu(
            32, true, 256,
        ));
        let schedule = RoundtableSchedule::new(&planner, 1, 8, RoundtableConfig::default());
        let gradient =
            Tensor::from_vec(1, 8, vec![-0.1, 0.2, -0.05, 0.9, -1.2, 0.0, 0.3, -0.4]).unwrap();
        let bands = schedule.split(&gradient).unwrap();
        let recombined = bands.combine().unwrap();
        assert_eq!(gradient, recombined);
        let energy = bands.above().squared_l2_norm()
            + bands.here().squared_l2_norm()
            + bands.beneath().squared_l2_norm();
        assert!(energy > 0.0);
    }

    #[test]
    fn band_energy_matches_split() {
        let planner = RankPlanner::new(st_core::backend::device_caps::DeviceCaps::wgpu(
            32, true, 256,
        ));
        let schedule = RoundtableSchedule::new(&planner, 1, 6, RoundtableConfig::default());
        let gradient = Tensor::from_vec(1, 6, vec![0.5, -0.2, 0.1, -0.4, 0.9, 0.0]).unwrap();
        let energy = schedule.band_energy(&gradient).unwrap();
        let bands = schedule.split(&gradient).unwrap();
        let recon = bands.combine().unwrap();
        assert_eq!(gradient, recon);
        let above = bands.above().data().iter().map(|v| v.abs()).sum::<f32>();
        let here = bands.here().data().iter().map(|v| v.abs()).sum::<f32>();
        let beneath = bands.beneath().data().iter().map(|v| v.abs()).sum::<f32>();
        assert!((energy.above - above).abs() < 1e-6);
        assert!((energy.here - here).abs() < 1e-6);
        assert!((energy.beneath - beneath).abs() < 1e-6);
        assert!(energy.spectral.energy > 0.0);
        assert!((0.0..=1.0).contains(&energy.spectral.sheet_confidence));
    }

    #[test]
    fn gradient_bands_scale_inplace_is_fallible_and_commit_safe() {
        let above = Tensor::from_vec(1, 2, vec![1.0, -2.0]).unwrap();
        let here = Tensor::from_vec(1, 2, vec![f32::MAX, 0.5]).unwrap();
        let beneath = Tensor::from_vec(1, 2, vec![3.0, -4.0]).unwrap();
        let mut bands =
            GradientBands::from_components(above.clone(), here.clone(), beneath.clone()).unwrap();

        let err = bands
            .scale_inplace(2.0, 2.0, 0.5)
            .expect_err("overflowed band scaling should fail");
        match err {
            TensorError::NonFiniteValue { label, value } => {
                assert_eq!(label, "scale_output");
                assert!(value.is_infinite());
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert_eq!(bands.above().data(), above.data());
        assert_eq!(bands.here().data(), here.data());
        assert_eq!(bands.beneath().data(), beneath.data());

        bands.scale_inplace(2.0, 0.25, 0.5).unwrap();
        assert_eq!(bands.above().data(), &[2.0, -4.0]);
        assert_eq!(bands.here().data(), &[f32::MAX * 0.25, 0.125]);
        assert_eq!(bands.beneath().data(), &[1.5, -2.0]);
    }

    #[test]
    fn band_energy_normalization_delegates_to_strict_core_contract() {
        let spectral = SpectralFeatureSample {
            energy: 0.7,
            ..SpectralFeatureSample::default()
        };
        let normalized = BandEnergy::new(6.0, 3.0, 1.0)
            .with_drift(0.2)
            .with_spectral(spectral)
            .try_norm()
            .expect("valid non-negative bands");
        assert!((normalized.above - 0.6).abs() < 1e-6);
        assert!((normalized.here - 0.3).abs() < 1e-6);
        assert!((normalized.beneath - 0.1).abs() < 1e-6);
        assert_eq!(normalized.drift, 0.2);
        assert_eq!(normalized.spectral, spectral);

        let invalid = BandEnergy::new(-1.0, 1.0, 1.0);
        assert!(matches!(
            invalid.try_norm(),
            Err(FreeEnergyError::Negative { .. })
        ));
        let fallback = invalid.norm();
        assert!((fallback.above + fallback.here + fallback.beneath - 1.0).abs() < 1e-6);
    }

    #[cfg(feature = "psi")]
    #[test]
    fn psi_flags_follow_config() {
        let planner = RankPlanner::new(st_core::backend::device_caps::DeviceCaps::wgpu(
            32, true, 256,
        ));
        let cfg = RoundtableConfig::default().enable_psi();
        let schedule = RoundtableSchedule::new(&planner, 1, 4, cfg);
        assert!(schedule.psi_enabled());
        let hint = schedule.psi_hint();
        assert!(hint.depth() > 0);
    }

    #[cfg(feature = "psychoid")]
    #[test]
    fn psychoid_flags_follow_config() {
        let planner = RankPlanner::new(st_core::backend::device_caps::DeviceCaps::wgpu(
            32, true, 256,
        ));
        let cfg = RoundtableConfig::default().enable_psychoid_with_log();
        let schedule = RoundtableSchedule::new(&planner, 1, 4, cfg);
        assert!(schedule.psychoid_enabled());
        assert!(schedule.psychoid_log());
    }

    #[cfg(feature = "collapse")]
    #[test]
    fn collapse_requires_psi() {
        let planner = RankPlanner::new(st_core::backend::device_caps::DeviceCaps::wgpu(
            32, true, 256,
        ));
        let cfg = RoundtableConfig::default().enable_collapse();
        let schedule = RoundtableSchedule::new(&planner, 1, 4, cfg);
        assert!(schedule.psi_enabled());
        assert!(schedule.collapse_enabled());
    }
}
