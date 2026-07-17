// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical GNN roundtable influence semantics.
//!
//! A caller supplies non-negative band energy, the active schedule occupancy,
//! and spectral observations. Rust validates that evidence and owns every
//! multiplier used by graph message passing. Bindings may transport the
//! observation and payload, but must not reconstruct these formulas.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ops::zspace_round::RoundtableBand;

pub const GNN_ROUNDTABLE_INFLUENCE_KIND: &str = "spiraltorch.gnn_roundtable_influence";
pub const GNN_ROUNDTABLE_INFLUENCE_CONTRACT_VERSION: &str =
    "spiraltorch.gnn_roundtable_influence.v1";
pub const GNN_ROUNDTABLE_INFLUENCE_SEMANTIC_OWNER: &str = "st-core::inference::gnn_roundtable";
pub const GNN_ROUNDTABLE_INFLUENCE_SEMANTIC_BACKEND: &str = "rust";
pub const GNN_ROUNDTABLE_INFLUENCE_FORMULA: &str =
    "bary=band_energy/sum(band_energy);occupancy=mix(bary,band_sizes/depth,spectral);multipliers=clamp(direction,curvature,stability);drift_bias=clamp(tanh(drift),direction,curvature)";
pub const GNN_ROUNDTABLE_MAX_HISTORY_SIGNALS: usize = 4096;

const MAX_EXACT_INTEGER: u64 = 9_007_199_254_740_991;
const MAX_NATIVE_MAGNITUDE: f64 = f32::MAX as f64;
const ENERGY_FLOOR: f64 = 1.0e-6;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GnnRoundtableBandEnergyObservation {
    pub above: f64,
    pub here: f64,
    pub beneath: f64,
    pub drift: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GnnRoundtableBandSizes {
    pub above: u32,
    pub here: u32,
    pub beneath: u32,
}

impl GnnRoundtableBandSizes {
    pub fn depth(self) -> u64 {
        u64::from(self.above) + u64::from(self.here) + u64::from(self.beneath)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GnnRoundtableSpectralObservation {
    pub sheet_index: u64,
    pub sheet_confidence: f64,
    pub curvature: f64,
    pub spin: f64,
    pub energy: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GnnRoundtableSignalObservation {
    pub band_energy: GnnRoundtableBandEnergyObservation,
    pub band_sizes: GnnRoundtableBandSizes,
    pub spectral: GnnRoundtableSpectralObservation,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct GnnRoundtableInfluencePayload {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub influence_formula: &'static str,
    pub multipliers: [f64; 3],
    pub drift_bias: f64,
}

impl GnnRoundtableInfluencePayload {
    pub fn scale_for_step(self, step_index: u64) -> f64 {
        let scale = match step_index {
            0 => self.multipliers[1],
            1 => self.multipliers[0] * self.drift_bias,
            _ => {
                let mut base = self.multipliers[2] / self.drift_bias.max(0.5);
                if step_index > 2 {
                    let decay_steps = (step_index - 2).min(8) as f64;
                    base *= (1.0 - decay_steps * 0.05).max(0.6);
                }
                base
            }
        };
        scale.clamp(0.25, 2.0)
    }

    pub fn band_pass_scale_for_step(
        self,
        band: RoundtableBand,
        step_index: u64,
        intensity: f64,
    ) -> Result<f64, GnnRoundtableInfluenceError> {
        require_range("band_pass.intensity", intensity, 0.0, 1.0)?;
        let average =
            ((self.multipliers[0] + self.multipliers[1] + self.multipliers[2]) / 3.0).max(0.35);
        let aligned = match band {
            RoundtableBand::Above => self.multipliers[0],
            RoundtableBand::Here => self.multipliers[1],
            RoundtableBand::Beneath => self.multipliers[2],
        };
        let alignment = (aligned / average).clamp(0.8, 1.35);
        let focus = intensity * alignment;
        let scale = match band {
            RoundtableBand::Above => match step_index {
                0 => 1.0 - 0.10 * focus,
                1 => 1.0 + 0.24 * focus,
                _ => 1.0 - 0.14 * focus,
            },
            RoundtableBand::Here => match step_index {
                0 => 1.0 + 0.20 * focus,
                1 => 1.0 - 0.07 * focus,
                _ => 1.0 - 0.10 * focus,
            },
            RoundtableBand::Beneath => match step_index {
                0 => 1.0 - 0.12 * focus,
                1 => 1.0 - 0.08 * focus,
                _ => {
                    let tail_steps = step_index.saturating_sub(2).min(3) as f64;
                    let tail_bias = (1.0 - tail_steps * 0.06).clamp(0.82, 1.0);
                    1.0 + 0.24 * focus * tail_bias
                }
            },
        };
        require_derived("band_pass.scale", scale.clamp(0.7, 1.35))
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum GnnRoundtableInfluenceError {
    #[error("GNN roundtable field '{field}' must be finite")]
    NonFinite { field: &'static str },
    #[error("GNN roundtable field '{field}' must be non-negative")]
    Negative { field: &'static str },
    #[error("GNN roundtable field '{field}' must be positive")]
    NonPositive { field: &'static str },
    #[error("GNN roundtable field '{field}' must be in [{min}, {max}]")]
    OutOfRange {
        field: &'static str,
        min: &'static str,
        max: &'static str,
    },
    #[error("GNN roundtable field '{field}' exceeds the portable integer domain")]
    UnsafeInteger { field: &'static str },
    #[error("derived GNN roundtable field '{field}' must be finite")]
    NonFiniteDerived { field: &'static str },
}

pub fn derive_gnn_roundtable_influence(
    observation: GnnRoundtableSignalObservation,
) -> Result<GnnRoundtableInfluencePayload, GnnRoundtableInfluenceError> {
    validate_observation(observation)?;

    let energy = observation.band_energy;
    let total = require_derived(
        "band_energy.total",
        energy.above + energy.here + energy.beneath,
    )?
    .max(ENERGY_FLOOR);
    let bary = [
        (energy.above / total).clamp(0.0, 1.0),
        (energy.here / total).clamp(0.0, 1.0),
        (energy.beneath / total).clamp(0.0, 1.0),
    ];
    let sizes = observation.band_sizes;
    let depth = sizes.depth() as f64;
    let share = [
        f64::from(sizes.above) / depth,
        f64::from(sizes.here) / depth,
        f64::from(sizes.beneath) / depth,
    ];
    let spectral = observation.spectral;
    let spectral_focus = spectral.sheet_confidence;
    let spectral_curvature = (spectral.curvature / 4.0).clamp(0.0, 1.0);
    let spectral_stability = (spectral.spin.abs() * (1.0 - spectral_curvature)).clamp(0.0, 1.0);
    let bary_asymmetry = (bary[0] - bary[2]).abs().clamp(0.0, 1.0);
    let energy_reliance = (0.5 + 0.35 * spectral_focus + 0.25 * bary_asymmetry).clamp(0.55, 0.9);
    let schedule_reliance = 1.0 - energy_reliance;
    let occupancy = [
        energy_reliance * bary[0] + schedule_reliance * share[0],
        energy_reliance * bary[1] + schedule_reliance * share[1],
        energy_reliance * bary[2] + schedule_reliance * share[2],
    ];
    let schedule_asymmetry = (occupancy[0] - occupancy[2]).clamp(-1.0, 1.0);
    let direction =
        (0.6 * schedule_asymmetry + 0.4 * spectral_focus * spectral.spin).clamp(-1.0, 1.0);
    let curvature_pull = (spectral_focus * spectral_curvature).clamp(0.0, 1.0);
    let edge_bias = (bary[0].max(bary[2]) - bary[1]).max(0.0);
    let edge_emphasis = (0.5 * edge_bias + 0.5 * direction.abs() * spectral_focus).clamp(0.0, 1.0);
    let edge_lift = (1.0 + 0.18 * edge_emphasis).clamp(1.0, 1.35);
    let center_damping = (1.0 - 0.24 * edge_emphasis).clamp(0.72, 1.0);
    let stability_lift = (0.92 + 0.16 * spectral_stability + 0.10 * edge_bias).clamp(0.85, 1.2);
    let neutrality = ((occupancy[1] + spectral_stability + curvature_pull) / 3.0).clamp(0.0, 1.0);
    let directional_relaxation = (1.0 - curvature_pull).clamp(0.65, 1.0);
    let mut multipliers = [
        (0.72 + 0.62 * occupancy[0])
            * (1.0 + 0.26 * direction.max(0.0) + 0.12 * edge_bias)
            * directional_relaxation,
        (0.78 + 0.52 * occupancy[1])
            * (1.0 + 0.18 * curvature_pull + 0.12 * neutrality)
            * (1.0 + 0.07 * (1.0 - spectral_focus))
            * center_damping,
        (0.72 + 0.62 * occupancy[2])
            * (1.0 + 0.26 * (-direction).max(0.0) + 0.12 * edge_bias)
            * directional_relaxation,
    ];
    multipliers[0] *= edge_lift;
    multipliers[2] *= edge_lift;
    if direction > 0.0 {
        multipliers[0] *= stability_lift;
    } else if direction < 0.0 {
        multipliers[2] *= stability_lift;
    } else {
        multipliers[1] *= stability_lift;
    }
    for (index, field) in [
        "multipliers.above",
        "multipliers.here",
        "multipliers.beneath",
    ]
    .into_iter()
    .enumerate()
    {
        multipliers[index] = require_derived(field, multipliers[index].clamp(0.35, 1.75))?;
    }
    let directional_bias = (0.55 * energy.drift.tanh() + 0.45 * direction).clamp(-1.0, 1.0);
    let drift_bias = require_derived(
        "drift_bias",
        ((1.0 + 0.22 * directional_bias)
            * (1.0 + 0.08 * edge_emphasis)
            * (1.0 - 0.12 * curvature_pull)
            * (1.0 - 0.06 * neutrality * spectral_stability))
            .clamp(0.55, 1.45),
    )?;

    Ok(GnnRoundtableInfluencePayload {
        kind: GNN_ROUNDTABLE_INFLUENCE_KIND,
        contract_version: GNN_ROUNDTABLE_INFLUENCE_CONTRACT_VERSION,
        semantic_owner: GNN_ROUNDTABLE_INFLUENCE_SEMANTIC_OWNER,
        semantic_backend: GNN_ROUNDTABLE_INFLUENCE_SEMANTIC_BACKEND,
        influence_formula: GNN_ROUNDTABLE_INFLUENCE_FORMULA,
        multipliers,
        drift_bias,
    })
}

pub fn validate_gnn_roundtable_signal(
    observation: GnnRoundtableSignalObservation,
) -> Result<(), GnnRoundtableInfluenceError> {
    validate_observation(observation)
}

fn validate_observation(
    observation: GnnRoundtableSignalObservation,
) -> Result<(), GnnRoundtableInfluenceError> {
    let energy = observation.band_energy;
    for (field, value) in [
        ("band_energy.above", energy.above),
        ("band_energy.here", energy.here),
        ("band_energy.beneath", energy.beneath),
        ("spectral.energy", observation.spectral.energy),
    ] {
        require_non_negative(field, value)?;
        require_native_magnitude(field, value)?;
    }
    require_native_magnitude("band_energy.drift", energy.drift.abs())?;
    require_range(
        "spectral.sheet_confidence",
        observation.spectral.sheet_confidence,
        0.0,
        1.0,
    )?;
    require_range(
        "spectral.curvature",
        observation.spectral.curvature,
        0.0,
        4.0,
    )?;
    require_range("spectral.spin", observation.spectral.spin, -1.0, 1.0)?;
    if observation.spectral.sheet_index > MAX_EXACT_INTEGER {
        return Err(GnnRoundtableInfluenceError::UnsafeInteger {
            field: "spectral.sheet_index",
        });
    }
    for (field, value) in [
        ("band_sizes.above", observation.band_sizes.above),
        ("band_sizes.here", observation.band_sizes.here),
        ("band_sizes.beneath", observation.band_sizes.beneath),
    ] {
        if value == 0 {
            return Err(GnnRoundtableInfluenceError::NonPositive { field });
        }
    }
    Ok(())
}

fn require_finite(field: &'static str, value: f64) -> Result<f64, GnnRoundtableInfluenceError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GnnRoundtableInfluenceError::NonFinite { field })
    }
}

fn require_non_negative(
    field: &'static str,
    value: f64,
) -> Result<f64, GnnRoundtableInfluenceError> {
    require_finite(field, value)?;
    if value >= 0.0 {
        Ok(value)
    } else {
        Err(GnnRoundtableInfluenceError::Negative { field })
    }
}

fn require_native_magnitude(
    field: &'static str,
    value: f64,
) -> Result<f64, GnnRoundtableInfluenceError> {
    require_finite(field, value)?;
    if value <= MAX_NATIVE_MAGNITUDE {
        Ok(value)
    } else {
        Err(GnnRoundtableInfluenceError::OutOfRange {
            field,
            min: "-f32::MAX",
            max: "f32::MAX",
        })
    }
}

fn require_range(
    field: &'static str,
    value: f64,
    min: f64,
    max: f64,
) -> Result<f64, GnnRoundtableInfluenceError> {
    require_finite(field, value)?;
    if (min..=max).contains(&value) {
        Ok(value)
    } else {
        Err(GnnRoundtableInfluenceError::OutOfRange {
            field,
            min: "contract minimum",
            max: "contract maximum",
        })
    }
}

fn require_derived(field: &'static str, value: f64) -> Result<f64, GnnRoundtableInfluenceError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GnnRoundtableInfluenceError::NonFiniteDerived { field })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation() -> GnnRoundtableSignalObservation {
        GnnRoundtableSignalObservation {
            band_energy: GnnRoundtableBandEnergyObservation {
                above: 1.4,
                here: 0.45,
                beneath: 0.2,
                drift: 0.35,
            },
            band_sizes: GnnRoundtableBandSizes {
                above: 4,
                here: 2,
                beneath: 2,
            },
            spectral: GnnRoundtableSpectralObservation {
                sheet_index: 1,
                sheet_confidence: 0.9,
                curvature: 0.6,
                spin: 0.85,
                energy: 0.72,
            },
        }
    }

    #[test]
    fn derives_directional_influence_and_step_scales() {
        let influence = derive_gnn_roundtable_influence(observation()).unwrap();
        assert_eq!(influence.semantic_backend, "rust");
        assert!(influence.multipliers[0] > influence.multipliers[2]);
        assert!(influence.scale_for_step(1) > influence.scale_for_step(2));
        assert!(
            influence
                .band_pass_scale_for_step(RoundtableBand::Above, 1, 0.75)
                .unwrap()
                > 1.0
        );
    }

    #[test]
    fn rejects_invalid_signal_evidence() {
        let mut request = observation();
        request.band_energy.above = f64::NAN;
        assert!(matches!(
            derive_gnn_roundtable_influence(request),
            Err(GnnRoundtableInfluenceError::NonFinite {
                field: "band_energy.above"
            })
        ));

        let mut request = observation();
        request.band_energy.beneath = -0.1;
        assert!(matches!(
            derive_gnn_roundtable_influence(request),
            Err(GnnRoundtableInfluenceError::Negative {
                field: "band_energy.beneath"
            })
        ));

        let mut request = observation();
        request.band_sizes.here = 0;
        assert!(matches!(
            derive_gnn_roundtable_influence(request),
            Err(GnnRoundtableInfluenceError::NonPositive {
                field: "band_sizes.here"
            })
        ));

        let mut request = observation();
        request.spectral.sheet_confidence = 1.1;
        assert!(matches!(
            derive_gnn_roundtable_influence(request),
            Err(GnnRoundtableInfluenceError::OutOfRange {
                field: "spectral.sheet_confidence",
                ..
            })
        ));
    }

    #[test]
    fn rejects_non_finite_band_pass_intensity() {
        let influence = derive_gnn_roundtable_influence(observation()).unwrap();
        assert!(matches!(
            influence.band_pass_scale_for_step(RoundtableBand::Here, 0, f64::NAN),
            Err(GnnRoundtableInfluenceError::NonFinite {
                field: "band_pass.intensity"
            })
        ));
    }
}
