// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Canonical variational free-energy semantics for runtime plan evaluation.
//!
//! Runtime clients provide dimensional observations and explicit reference
//! scales. This module normalises those observations, interprets roundtable
//! band energy as a probability distribution, and evaluates
//!
//! `F(q) = E_observed + (E_q[V] - E_prior[V]) + temperature * KL(q || prior)`.
//!
//! Rust owns every validation, normalisation, and reduction step. Python and
//! WASM bindings may transport requests and reports, but must not reconstruct
//! this equation.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const FREE_ENERGY_CONTRACT_VERSION: &str = "spiraltorch.variational_free_energy.v1";
pub const FREE_ENERGY_KIND: &str = "spiraltorch.variational_free_energy";
pub const FREE_ENERGY_SEMANTIC_OWNER: &str = "st-core::heur::free_energy";
pub const FREE_ENERGY_SEMANTIC_BACKEND: &str = "rust";
pub const FREE_ENERGY_FORMULA: &str =
    "F(q)=E_observed+(E_q[V]-E_prior[V])+temperature*KL(q||prior)";
pub const FREE_ENERGY_ACCEPTANCE_RULE: &str =
    "P(accept)=1/(1+exp(F_candidate-F_neutral)),F_neutral=0";
pub const FREE_ENERGY_BAND_ZERO_MASS_THRESHOLD: f64 = f32::EPSILON as f64;

const BAND_COUNT: f64 = 3.0;
const UNIFORM_BAND_PROBABILITY: f64 = 1.0 / BAND_COUNT;
const DERIVED_TOLERANCE: f64 = 64.0 * f64::EPSILON;

/// Aggregate non-negative energy observed in the three roundtable bands.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BandEnergy {
    pub above: f32,
    pub here: f32,
    pub beneath: f32,
}

impl BandEnergy {
    /// Returns the L1 magnitude across all bands.
    pub fn l1(&self) -> f32 {
        self.above.abs() + self.here.abs() + self.beneath.abs()
    }

    /// Strict finite L1 magnitude used by guarded semantic routes.
    pub fn try_l1(self) -> Result<f64, FreeEnergyError> {
        let values = [
            ("band.above", f64::from(self.above)),
            ("band.here", f64::from(self.here)),
            ("band.beneath", f64::from(self.beneath)),
        ];
        for (field, value) in values {
            require_non_negative(field, value)?;
        }
        checked_sum("band.total", values.into_iter().map(|(_, value)| value))
    }

    /// Strictly converts finite, non-negative energies into probabilities.
    /// A zero-mass observation has no preferred band and becomes uniform.
    pub fn try_probabilities(self) -> Result<[f64; 3], FreeEnergyError> {
        let total = self.try_l1()?;
        if total <= FREE_ENERGY_BAND_ZERO_MASS_THRESHOLD {
            return Ok([UNIFORM_BAND_PROBABILITY; 3]);
        }
        Ok([
            checked_div("band.above_probability", f64::from(self.above), total)?,
            checked_div("band.here_probability", f64::from(self.here), total)?,
            checked_div("band.beneath_probability", f64::from(self.beneath), total)?,
        ])
    }

    /// Strictly normalises finite, non-negative band energies.
    pub fn try_norm(self) -> Result<Self, FreeEnergyError> {
        let probabilities = self.try_probabilities()?;
        Ok(Self {
            above: probabilities[0] as f32,
            here: probabilities[1] as f32,
            beneath: probabilities[2] as f32,
        })
    }

    /// Compatibility normalisation that falls back to an even split for an
    /// invalid observation. Guarded runtime paths should prefer [`Self::try_norm`].
    pub fn norm(self) -> Self {
        self.try_norm().unwrap_or_else(|_| Self::uniform())
    }

    pub const fn uniform() -> Self {
        Self {
            above: 1.0 / 3.0,
            here: 1.0 / 3.0,
            beneath: 1.0 / 3.0,
        }
    }
}

/// Strictly positive reference distribution used by the KL term.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct BandPrior {
    pub above: f64,
    pub here: f64,
    pub beneath: f64,
}

impl Default for BandPrior {
    fn default() -> Self {
        Self {
            above: UNIFORM_BAND_PROBABILITY,
            here: UNIFORM_BAND_PROBABILITY,
            beneath: UNIFORM_BAND_PROBABILITY,
        }
    }
}

impl BandPrior {
    pub fn try_probabilities(self) -> Result<[f64; 3], FreeEnergyError> {
        let values = [
            ("prior.above", self.above),
            ("prior.here", self.here),
            ("prior.beneath", self.beneath),
        ];
        for (field, value) in values {
            require_positive(field, value)?;
        }
        let total = checked_sum("prior.total", values.into_iter().map(|(_, value)| value))?;
        Ok([
            checked_div("prior.above_probability", self.above, total)?,
            checked_div("prior.here_probability", self.here, total)?,
            checked_div("prior.beneath_probability", self.beneath, total)?,
        ])
    }
}

/// Energy assigned to occupying each roundtable band.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct BandPotentials {
    pub above: f64,
    pub here: f64,
    pub beneath: f64,
}

impl Default for BandPotentials {
    fn default() -> Self {
        Self {
            above: -0.1,
            here: 0.0,
            beneath: 0.3,
        }
    }
}

impl BandPotentials {
    fn validate(self) -> Result<(), FreeEnergyError> {
        require_finite("band_potential.above", self.above)?;
        require_finite("band_potential.here", self.here)?;
        require_finite("band_potential.beneath", self.beneath)?;
        Ok(())
    }

    const fn as_array(self) -> [f64; 3] {
        [self.above, self.here, self.beneath]
    }
}

/// Reference scales and component weights for dimensionless evaluation.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct FreeEnergyConfig {
    loss_scale: f64,
    step_time_scale_ms: f64,
    memory_scale_mb: f64,
    retry_scale: f64,
    observation_entropy_scale: f64,
    loss_weight: f64,
    speed_weight: f64,
    memory_weight: f64,
    retry_weight: f64,
    uncertainty_weight: f64,
    external_penalty_weight: f64,
    temperature: f64,
    prior: BandPrior,
    band_potentials: BandPotentials,
}

impl Default for FreeEnergyConfig {
    fn default() -> Self {
        Self {
            loss_scale: 1.0,
            step_time_scale_ms: 10.0,
            memory_scale_mb: 1024.0,
            retry_scale: 1.0,
            observation_entropy_scale: 1.0,
            loss_weight: 1.0,
            speed_weight: 0.5,
            memory_weight: 0.3,
            retry_weight: 0.2,
            uncertainty_weight: 0.0,
            external_penalty_weight: 1.0,
            temperature: 0.1,
            prior: BandPrior::default(),
            band_potentials: BandPotentials::default(),
        }
    }
}

impl FreeEnergyConfig {
    pub fn with_loss_scale(mut self, loss_scale: f64) -> Result<Self, FreeEnergyError> {
        self.loss_scale = loss_scale;
        self.validate()?;
        Ok(self)
    }

    pub fn with_resource_scales(
        mut self,
        step_time_scale_ms: f64,
        memory_scale_mb: f64,
        retry_scale: f64,
    ) -> Result<Self, FreeEnergyError> {
        self.step_time_scale_ms = step_time_scale_ms;
        self.memory_scale_mb = memory_scale_mb;
        self.retry_scale = retry_scale;
        self.validate()?;
        Ok(self)
    }

    pub fn with_observation_entropy_scale(
        mut self,
        observation_entropy_scale: f64,
    ) -> Result<Self, FreeEnergyError> {
        self.observation_entropy_scale = observation_entropy_scale;
        self.validate()?;
        Ok(self)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_component_weights(
        mut self,
        loss_weight: f64,
        speed_weight: f64,
        memory_weight: f64,
        retry_weight: f64,
        uncertainty_weight: f64,
        external_penalty_weight: f64,
    ) -> Result<Self, FreeEnergyError> {
        self.loss_weight = loss_weight;
        self.speed_weight = speed_weight;
        self.memory_weight = memory_weight;
        self.retry_weight = retry_weight;
        self.uncertainty_weight = uncertainty_weight;
        self.external_penalty_weight = external_penalty_weight;
        self.validate()?;
        Ok(self)
    }

    pub fn with_temperature(mut self, temperature: f64) -> Result<Self, FreeEnergyError> {
        self.temperature = temperature;
        self.validate()?;
        Ok(self)
    }

    pub fn with_prior(mut self, prior: BandPrior) -> Result<Self, FreeEnergyError> {
        self.prior = prior;
        self.validate()?;
        Ok(self)
    }

    pub fn with_band_potentials(
        mut self,
        band_potentials: BandPotentials,
    ) -> Result<Self, FreeEnergyError> {
        self.band_potentials = band_potentials;
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), FreeEnergyError> {
        for (field, value) in [
            ("loss_scale", self.loss_scale),
            ("step_time_scale_ms", self.step_time_scale_ms),
            ("memory_scale_mb", self.memory_scale_mb),
            ("retry_scale", self.retry_scale),
            ("observation_entropy_scale", self.observation_entropy_scale),
        ] {
            require_positive(field, value)?;
        }
        for (field, value) in [
            ("loss_weight", self.loss_weight),
            ("speed_weight", self.speed_weight),
            ("memory_weight", self.memory_weight),
            ("retry_weight", self.retry_weight),
            ("uncertainty_weight", self.uncertainty_weight),
            ("external_penalty_weight", self.external_penalty_weight),
            ("temperature", self.temperature),
        ] {
            require_non_negative(field, value)?;
        }
        self.prior.try_probabilities()?;
        self.band_potentials.validate()?;
        Ok(())
    }

    pub const fn loss_scale(&self) -> f64 {
        self.loss_scale
    }

    pub const fn step_time_scale_ms(&self) -> f64 {
        self.step_time_scale_ms
    }

    pub const fn memory_scale_mb(&self) -> f64 {
        self.memory_scale_mb
    }

    pub const fn retry_scale(&self) -> f64 {
        self.retry_scale
    }

    pub const fn observation_entropy_scale(&self) -> f64 {
        self.observation_entropy_scale
    }

    pub const fn loss_weight(&self) -> f64 {
        self.loss_weight
    }

    pub const fn speed_weight(&self) -> f64 {
        self.speed_weight
    }

    pub const fn memory_weight(&self) -> f64 {
        self.memory_weight
    }

    pub const fn retry_weight(&self) -> f64 {
        self.retry_weight
    }

    pub const fn uncertainty_weight(&self) -> f64 {
        self.uncertainty_weight
    }

    pub const fn external_penalty_weight(&self) -> f64 {
        self.external_penalty_weight
    }

    pub const fn temperature(&self) -> f64 {
        self.temperature
    }

    pub const fn prior(&self) -> BandPrior {
        self.prior
    }

    pub const fn band_potentials(&self) -> BandPotentials {
        self.band_potentials
    }
}

/// Dimensional observations supplied by a runtime step.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct FreeEnergyObservation {
    #[serde(alias = "loss_before")]
    pub reference_loss: f64,
    #[serde(alias = "loss_after")]
    pub candidate_loss: f64,
    pub step_time_ms: f64,
    pub memory_mb: f64,
    pub retry_rate: f64,
    pub observation_entropy: f64,
    pub external_penalty: f64,
    pub band: BandEnergy,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct FreeEnergyRequest {
    pub observation: FreeEnergyObservation,
    pub config: FreeEnergyConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct NormalizedFreeEnergyObservation {
    pub loss_delta: f64,
    pub step_time: f64,
    pub memory: f64,
    pub retry: f64,
    pub observation_entropy: f64,
    pub external_penalty: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct BandDistributionAudit {
    pub status: &'static str,
    pub raw_total: f64,
    pub zero_mass_threshold: f64,
    pub above: f64,
    pub here: f64,
    pub beneath: f64,
    pub prior_above: f64,
    pub prior_here: f64,
    pub prior_beneath: f64,
    pub entropy: f64,
    pub normalized_entropy: f64,
    pub cross_entropy: f64,
    pub kl_divergence: f64,
    pub variational_identity_residual: f64,
    pub dominant_band: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct FreeEnergyComponents {
    pub loss: f64,
    pub speed: f64,
    pub memory: f64,
    pub retry: f64,
    pub uncertainty: f64,
    pub external_penalty: f64,
    pub observed_energy: f64,
    pub band_potential_expectation: f64,
    pub prior_band_potential: f64,
    pub band_potential: f64,
    pub relative_entropy: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FreeEnergyReport {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub formula: &'static str,
    pub acceptance_rule: &'static str,
    pub config: FreeEnergyConfig,
    pub observation: FreeEnergyObservation,
    pub normalized: NormalizedFreeEnergyObservation,
    pub distribution: BandDistributionAudit,
    pub components: FreeEnergyComponents,
    pub free_energy: f64,
    pub utility: f64,
    pub acceptance_probability: f64,
    pub component_sum_residual: f64,
}

/// Evaluate the canonical, dimensionless variational free-energy contract.
pub fn evaluate_free_energy(
    request: FreeEnergyRequest,
) -> Result<FreeEnergyReport, FreeEnergyError> {
    request.config.validate()?;
    validate_observation(&request.observation)?;

    let config = request.config;
    let observation = request.observation;
    let prior = config.prior.try_probabilities()?;
    let band_total = observation.band.try_l1()?;
    let zero_band_mass = band_total <= FREE_ENERGY_BAND_ZERO_MASS_THRESHOLD;
    let probabilities = if zero_band_mass {
        prior
    } else {
        observation.band.try_probabilities()?
    };
    let loss_delta = checked_add(
        "loss_delta",
        observation.candidate_loss,
        -observation.reference_loss,
    )?;
    let normalized = NormalizedFreeEnergyObservation {
        loss_delta: checked_div("normalized.loss_delta", loss_delta, config.loss_scale)?,
        step_time: checked_div(
            "normalized.step_time",
            observation.step_time_ms,
            config.step_time_scale_ms,
        )?,
        memory: checked_div(
            "normalized.memory",
            observation.memory_mb,
            config.memory_scale_mb,
        )?,
        retry: checked_div(
            "normalized.retry",
            observation.retry_rate,
            config.retry_scale,
        )?,
        observation_entropy: checked_div(
            "normalized.observation_entropy",
            observation.observation_entropy,
            config.observation_entropy_scale,
        )?,
        external_penalty: observation.external_penalty,
    };

    let loss = checked_mul("component.loss", config.loss_weight, normalized.loss_delta)?;
    let speed = checked_mul("component.speed", config.speed_weight, normalized.step_time)?;
    let memory = checked_mul("component.memory", config.memory_weight, normalized.memory)?;
    let retry = checked_mul("component.retry", config.retry_weight, normalized.retry)?;
    let uncertainty = checked_mul(
        "component.uncertainty",
        config.uncertainty_weight,
        normalized.observation_entropy,
    )?;
    let external_penalty = checked_mul(
        "component.external_penalty",
        config.external_penalty_weight,
        normalized.external_penalty,
    )?;
    let observed_energy = checked_sum(
        "component.observed_energy",
        [loss, speed, memory, retry, uncertainty, external_penalty],
    )?;

    let potentials = config.band_potentials.as_array();
    let band_potential_expectation = checked_sum(
        "component.band_potential_expectation",
        probabilities
            .into_iter()
            .zip(potentials)
            .map(|(probability, potential)| probability * potential),
    )?;
    let prior_band_potential = checked_sum(
        "component.prior_band_potential",
        prior
            .into_iter()
            .zip(potentials)
            .map(|(probability, potential)| probability * potential),
    )?;
    let band_potential = checked_add(
        "component.band_potential",
        band_potential_expectation,
        -prior_band_potential,
    )?;

    let entropy = shannon_entropy(probabilities)?;
    let cross_entropy = cross_entropy(probabilities, prior)?;
    let raw_kl = checked_add("distribution.kl_divergence", cross_entropy, -entropy)?;
    let kl_divergence = if raw_kl >= -DERIVED_TOLERANCE {
        raw_kl.max(0.0)
    } else {
        return Err(FreeEnergyError::NegativeDerived {
            field: "distribution.kl_divergence",
            value: raw_kl,
        });
    };
    let relative_entropy = checked_mul(
        "component.relative_entropy",
        config.temperature,
        kl_divergence,
    )?;
    let free_energy = checked_sum(
        "free_energy",
        [observed_energy, band_potential, relative_entropy],
    )?;
    let utility = require_derived_finite("utility", -free_energy)?;
    let acceptance_probability =
        require_derived_finite("acceptance_probability", stable_sigmoid(utility))?;
    let component_sum = checked_sum(
        "component_sum",
        [observed_energy, band_potential, relative_entropy],
    )?;

    Ok(FreeEnergyReport {
        kind: FREE_ENERGY_KIND,
        contract_version: FREE_ENERGY_CONTRACT_VERSION,
        semantic_owner: FREE_ENERGY_SEMANTIC_OWNER,
        semantic_backend: FREE_ENERGY_SEMANTIC_BACKEND,
        formula: FREE_ENERGY_FORMULA,
        acceptance_rule: FREE_ENERGY_ACCEPTANCE_RULE,
        config,
        observation,
        normalized,
        distribution: BandDistributionAudit {
            status: if zero_band_mass {
                "prior_zero_mass"
            } else {
                "normalized"
            },
            raw_total: band_total,
            zero_mass_threshold: FREE_ENERGY_BAND_ZERO_MASS_THRESHOLD,
            above: probabilities[0],
            here: probabilities[1],
            beneath: probabilities[2],
            prior_above: prior[0],
            prior_here: prior[1],
            prior_beneath: prior[2],
            entropy,
            normalized_entropy: checked_div(
                "distribution.normalized_entropy",
                entropy,
                BAND_COUNT.ln(),
            )?,
            cross_entropy,
            kl_divergence,
            variational_identity_residual: require_derived_finite(
                "distribution.variational_identity_residual",
                (kl_divergence - (cross_entropy - entropy)).abs(),
            )?,
            dominant_band: dominant_band(probabilities),
        },
        components: FreeEnergyComponents {
            loss,
            speed,
            memory,
            retry,
            uncertainty,
            external_penalty,
            observed_energy,
            band_potential_expectation,
            prior_band_potential,
            band_potential,
            relative_entropy,
        },
        free_energy,
        utility,
        acceptance_probability,
        component_sum_residual: require_derived_finite(
            "component_sum_residual",
            (free_energy - component_sum).abs(),
        )?,
    })
}

fn validate_observation(observation: &FreeEnergyObservation) -> Result<(), FreeEnergyError> {
    require_finite("reference_loss", observation.reference_loss)?;
    require_finite("candidate_loss", observation.candidate_loss)?;
    for (field, value) in [
        ("step_time_ms", observation.step_time_ms),
        ("memory_mb", observation.memory_mb),
        ("retry_rate", observation.retry_rate),
        ("observation_entropy", observation.observation_entropy),
        ("external_penalty", observation.external_penalty),
    ] {
        require_non_negative(field, value)?;
    }
    observation.band.try_probabilities()?;
    Ok(())
}

fn shannon_entropy(probabilities: [f64; 3]) -> Result<f64, FreeEnergyError> {
    checked_sum(
        "distribution.entropy",
        probabilities.into_iter().map(|probability| {
            if probability > 0.0 {
                -probability * probability.ln()
            } else {
                0.0
            }
        }),
    )
}

fn cross_entropy(probabilities: [f64; 3], prior: [f64; 3]) -> Result<f64, FreeEnergyError> {
    checked_sum(
        "distribution.cross_entropy",
        probabilities
            .into_iter()
            .zip(prior)
            .map(|(probability, prior)| -probability * prior.ln()),
    )
}

fn dominant_band(probabilities: [f64; 3]) -> &'static str {
    if probabilities[0] >= probabilities[1] && probabilities[0] >= probabilities[2] {
        "above"
    } else if probabilities[1] >= probabilities[2] {
        "here"
    } else {
        "beneath"
    }
}

fn stable_sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn require_finite(field: &'static str, value: f64) -> Result<f64, FreeEnergyError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(FreeEnergyError::NonFinite { field, value })
    }
}

fn require_positive(field: &'static str, value: f64) -> Result<f64, FreeEnergyError> {
    require_finite(field, value)?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err(FreeEnergyError::NonPositive { field, value })
    }
}

fn require_non_negative(field: &'static str, value: f64) -> Result<f64, FreeEnergyError> {
    require_finite(field, value)?;
    if value >= 0.0 {
        Ok(value)
    } else {
        Err(FreeEnergyError::Negative { field, value })
    }
}

fn require_derived_finite(field: &'static str, value: f64) -> Result<f64, FreeEnergyError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(FreeEnergyError::NonFiniteDerived { field, value })
    }
}

fn checked_add(field: &'static str, left: f64, right: f64) -> Result<f64, FreeEnergyError> {
    require_derived_finite(field, left + right)
}

fn checked_mul(field: &'static str, left: f64, right: f64) -> Result<f64, FreeEnergyError> {
    require_derived_finite(field, left * right)
}

fn checked_div(
    field: &'static str,
    numerator: f64,
    denominator: f64,
) -> Result<f64, FreeEnergyError> {
    require_derived_finite(field, numerator / denominator)
}

fn checked_sum(
    field: &'static str,
    values: impl IntoIterator<Item = f64>,
) -> Result<f64, FreeEnergyError> {
    values
        .into_iter()
        .try_fold(0.0, |sum, value| checked_add(field, sum, value))
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum FreeEnergyError {
    #[error("free-energy field '{field}' must be finite, got {value}")]
    NonFinite { field: &'static str, value: f64 },
    #[error("free-energy field '{field}' must be positive, got {value}")]
    NonPositive { field: &'static str, value: f64 },
    #[error("free-energy field '{field}' must be non-negative, got {value}")]
    Negative { field: &'static str, value: f64 },
    #[error("derived free-energy field '{field}' must be finite, got {value}")]
    NonFiniteDerived { field: &'static str, value: f64 },
    #[error("derived free-energy field '{field}' must be non-negative, got {value}")]
    NegativeDerived { field: &'static str, value: f64 },
    #[error("free-energy score cannot be represented as f32")]
    ScoreOutOfRange,
}

/// Backward-compatible context for the original scalar ranking helper.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FeCtx {
    pub loss_before: f32,
    pub loss_after: f32,
    pub step_ms: f32,
    pub mem_mb: f32,
    pub retry: f32,
    pub band: BandEnergy,
    /// Exogenous uncertainty estimate. This is not the Shannon entropy of the
    /// band distribution; that quantity is derived by the canonical evaluator.
    pub entropy: f32,
}

/// Compatibility scalar used by older rank callers.
pub fn score_with_free_energy(ctx: &FeCtx, beta: f32) -> f32 {
    try_score_with_free_energy(ctx, beta).unwrap_or(f32::NEG_INFINITY)
}

/// Evaluate the compatibility scalar through the canonical Rust contract.
/// `beta` remains the weight of exogenous uncertainty for source compatibility.
pub fn try_score_with_free_energy(ctx: &FeCtx, beta: f32) -> Result<f32, FreeEnergyError> {
    require_non_negative("beta", f64::from(beta))?;
    let config = FreeEnergyConfig::default()
        .with_resource_scales(1.0, 1.0, 1.0)?
        .with_component_weights(1.0, 0.0025, 0.001, 0.5, f64::from(beta), 0.0)?
        .with_temperature(0.0)?;
    let report = evaluate_free_energy(FreeEnergyRequest {
        observation: FreeEnergyObservation {
            reference_loss: f64::from(ctx.loss_before),
            candidate_loss: f64::from(ctx.loss_after),
            step_time_ms: f64::from(ctx.step_ms),
            memory_mb: f64::from(ctx.mem_mb),
            retry_rate: f64::from(ctx.retry),
            observation_entropy: f64::from(ctx.entropy),
            external_penalty: 0.0,
            band: ctx.band,
        },
        config,
    })?;
    // The original scalar used an uncentred band heuristic. Remove the prior
    // gauge restored by the canonical report so existing thresholds stay exact.
    let compatibility_utility = checked_add(
        "compatibility.utility",
        report.utility,
        -report.components.prior_band_potential,
    )?;
    if compatibility_utility.abs() > f64::from(f32::MAX) {
        return Err(FreeEnergyError::ScoreOutOfRange);
    }
    Ok(compatibility_utility as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn observation() -> FreeEnergyObservation {
        FreeEnergyObservation {
            reference_loss: 0.8,
            candidate_loss: 0.5,
            step_time_ms: 12.0,
            memory_mb: 256.0,
            retry_rate: 0.05,
            observation_entropy: 0.4,
            external_penalty: 0.1,
            band: BandEnergy {
                above: 0.6,
                here: 0.3,
                beneath: 0.1,
            },
        }
    }

    #[test]
    fn report_satisfies_variational_decomposition() {
        let report = evaluate_free_energy(FreeEnergyRequest {
            observation: observation(),
            config: FreeEnergyConfig::default(),
        })
        .expect("valid free-energy request");

        assert_eq!(report.kind, FREE_ENERGY_KIND);
        assert_eq!(report.contract_version, FREE_ENERGY_CONTRACT_VERSION);
        assert_eq!(report.semantic_owner, FREE_ENERGY_SEMANTIC_OWNER);
        assert_eq!(report.semantic_backend, FREE_ENERGY_SEMANTIC_BACKEND);
        assert_eq!(report.formula, FREE_ENERGY_FORMULA);
        assert_eq!(report.acceptance_rule, FREE_ENERGY_ACCEPTANCE_RULE);
        assert_abs_diff_eq!(
            report.free_energy,
            report.components.observed_energy
                + report.components.band_potential
                + report.components.relative_entropy,
            epsilon = 1e-14
        );
        assert!(report.component_sum_residual <= f64::EPSILON);
        assert!(report.distribution.variational_identity_residual <= 1e-14);
        assert_abs_diff_eq!(report.utility, -report.free_energy, epsilon = f64::EPSILON);
        assert!((0.0..=1.0).contains(&report.acceptance_probability));
    }

    #[test]
    fn dimensionless_scales_are_unit_invariant() {
        let base = evaluate_free_energy(FreeEnergyRequest {
            observation: observation(),
            config: FreeEnergyConfig::default(),
        })
        .expect("base request");
        let mut scaled_observation = observation();
        scaled_observation.step_time_ms *= 1000.0;
        scaled_observation.memory_mb *= 1024.0;
        let scaled_config = FreeEnergyConfig::default()
            .with_resource_scales(10_000.0, 1024.0 * 1024.0, 1.0)
            .expect("scaled units");
        let scaled = evaluate_free_energy(FreeEnergyRequest {
            observation: scaled_observation,
            config: scaled_config,
        })
        .expect("scaled request");

        assert_abs_diff_eq!(base.free_energy, scaled.free_energy, epsilon = 1e-12);
    }

    #[test]
    fn kl_is_zero_when_observation_matches_prior() {
        let prior = BandPrior {
            above: 0.5,
            here: 0.3,
            beneath: 0.2,
        };
        let config = FreeEnergyConfig::default()
            .with_prior(prior)
            .expect("valid prior");
        let report = evaluate_free_energy(FreeEnergyRequest {
            observation: FreeEnergyObservation {
                band: BandEnergy {
                    above: 5.0,
                    here: 3.0,
                    beneath: 2.0,
                },
                ..FreeEnergyObservation::default()
            },
            config,
        })
        .expect("matching distribution");

        assert_abs_diff_eq!(report.distribution.kl_divergence, 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(report.components.relative_entropy, 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(report.components.band_potential, 0.0, epsilon = 1e-14);
    }

    #[test]
    fn zero_band_mass_uses_the_prior_and_is_audited() {
        let prior = BandPrior {
            above: 0.5,
            here: 0.3,
            beneath: 0.2,
        };
        let config = FreeEnergyConfig::default()
            .with_prior(prior)
            .expect("valid prior");
        let report = evaluate_free_energy(FreeEnergyRequest {
            config,
            ..FreeEnergyRequest::default()
        })
        .expect("zero mass is valid");
        assert_eq!(report.distribution.status, "prior_zero_mass");
        assert_abs_diff_eq!(report.distribution.above, 0.5, epsilon = f64::EPSILON);
        assert_abs_diff_eq!(report.distribution.here, 0.3, epsilon = f64::EPSILON);
        assert_abs_diff_eq!(report.distribution.beneath, 0.2, epsilon = f64::EPSILON);
        assert_abs_diff_eq!(report.distribution.kl_divergence, 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(report.components.band_potential, 0.0, epsilon = 1e-14);
        assert_eq!(report.distribution.raw_total, 0.0);
        assert_eq!(
            report.distribution.zero_mass_threshold,
            FREE_ENERGY_BAND_ZERO_MASS_THRESHOLD
        );
    }

    #[test]
    fn tiny_band_noise_does_not_move_the_prior() {
        let report = evaluate_free_energy(FreeEnergyRequest {
            observation: FreeEnergyObservation {
                band: BandEnergy {
                    above: f32::EPSILON / 4.0,
                    here: f32::EPSILON / 8.0,
                    beneath: 0.0,
                },
                ..FreeEnergyObservation::default()
            },
            config: FreeEnergyConfig::default(),
        })
        .expect("tiny finite energy is valid");

        assert_eq!(report.distribution.status, "prior_zero_mass");
        assert_abs_diff_eq!(
            report.distribution.above,
            UNIFORM_BAND_PROBABILITY,
            epsilon = f64::EPSILON
        );
        assert_abs_diff_eq!(report.components.band_potential, 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(report.distribution.kl_divergence, 0.0, epsilon = 1e-14);
    }

    #[test]
    fn component_weights_change_only_their_named_component() {
        let without_speed = FreeEnergyConfig::default()
            .with_component_weights(1.0, 0.0, 0.3, 0.2, 0.0, 1.0)
            .expect("valid weights");
        let with_speed = FreeEnergyConfig::default();
        let left = evaluate_free_energy(FreeEnergyRequest {
            observation: observation(),
            config: without_speed,
        })
        .expect("without speed");
        let right = evaluate_free_energy(FreeEnergyRequest {
            observation: observation(),
            config: with_speed,
        })
        .expect("with speed");

        assert_abs_diff_eq!(left.components.speed, 0.0, epsilon = f64::EPSILON);
        assert!(right.components.speed > 0.0);
        assert_abs_diff_eq!(
            right.free_energy - left.free_energy,
            right.components.speed,
            epsilon = 1e-14
        );
    }

    #[test]
    fn higher_exogenous_uncertainty_is_penalised_by_compatibility_api() {
        let ctx = FeCtx {
            loss_before: 0.5,
            loss_after: 0.25,
            entropy: 1.2,
            step_ms: 10.0,
            mem_mb: 128.0,
            retry: 0.0,
            band: BandEnergy {
                above: 0.6,
                here: 0.3,
                beneath: 0.1,
            },
        };
        let no_uncertainty_cost = score_with_free_energy(&ctx, 0.0);
        let uncertainty_cost = score_with_free_energy(&ctx, 0.8);
        assert!(uncertainty_cost < no_uncertainty_cost);
    }

    #[test]
    fn compatibility_scalar_matches_the_original_valid_input_formula() {
        let ctx = FeCtx {
            loss_before: 0.5,
            loss_after: 0.25,
            step_ms: 10.0,
            mem_mb: 128.0,
            retry: 0.2,
            band: BandEnergy {
                above: 0.6,
                here: 0.3,
                beneath: 0.1,
            },
            entropy: 1.2,
        };
        let beta = 0.8;
        let band = ctx.band.try_norm().expect("valid band");
        let expected = -(ctx.loss_after - ctx.loss_before)
            - beta * ctx.entropy
            - 0.0025 * ctx.step_ms
            - 0.001 * ctx.mem_mb
            - 0.5 * ctx.retry
            + 0.2 * (band.above - band.beneath)
            - 0.1 * (1.0 - band.here);

        assert!((score_with_free_energy(&ctx, beta) - expected).abs() < 1.0e-6);
    }

    #[test]
    fn lower_above_potential_rewards_novel_band_mass() {
        let base = FeCtx {
            loss_before: 0.4,
            loss_after: 0.3,
            entropy: 0.4,
            step_ms: 12.0,
            mem_mb: 64.0,
            retry: 0.0,
            band: BandEnergy {
                above: 0.2,
                here: 0.6,
                beneath: 0.2,
            },
        };
        let novel = FeCtx {
            band: BandEnergy {
                above: 0.6,
                here: 0.3,
                beneath: 0.1,
            },
            ..base
        };
        assert!(score_with_free_energy(&novel, 0.3) > score_with_free_energy(&base, 0.3));
    }

    #[test]
    fn strict_paths_reject_invalid_inputs_and_config() {
        let error = evaluate_free_energy(FreeEnergyRequest {
            observation: FreeEnergyObservation {
                memory_mb: f64::NAN,
                ..FreeEnergyObservation::default()
            },
            config: FreeEnergyConfig::default(),
        })
        .expect_err("non-finite observation must fail");
        assert!(matches!(error, FreeEnergyError::NonFinite { .. }));

        let invalid_band = BandEnergy {
            above: -0.1,
            here: 0.5,
            beneath: 0.6,
        };
        assert!(matches!(
            invalid_band.try_norm(),
            Err(FreeEnergyError::Negative { .. })
        ));
        assert_eq!(invalid_band.norm(), BandEnergy::uniform());

        let invalid_prior = FreeEnergyConfig::default().with_prior(BandPrior {
            above: 0.0,
            here: 0.5,
            beneath: 0.5,
        });
        assert!(matches!(
            invalid_prior,
            Err(FreeEnergyError::NonPositive { .. })
        ));
        assert!(matches!(
            try_score_with_free_energy(&FeCtx::default(), -0.1),
            Err(FreeEnergyError::Negative { field: "beta", .. })
        ));
    }

    #[test]
    fn serde_rejects_unknown_semantic_knobs() {
        let error = serde_json::from_value::<FreeEnergyRequest>(serde_json::json!({
            "config": {"mystery_weight": 1.0}
        }))
        .expect_err("unknown config must not be ignored");
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn legacy_loss_aliases_emit_only_canonical_report_fields() {
        let request = serde_json::from_value::<FreeEnergyRequest>(serde_json::json!({
            "observation": {
                "loss_before": 0.8,
                "loss_after": 0.5
            }
        }))
        .expect("legacy loss aliases remain accepted at the transport boundary");
        let report = evaluate_free_energy(request).expect("aliased request is valid");
        let payload = serde_json::to_value(report).expect("report serializes");
        let observation = payload
            .get("observation")
            .and_then(serde_json::Value::as_object)
            .expect("canonical observation object");

        assert_eq!(
            observation.get("reference_loss"),
            Some(&serde_json::json!(0.8))
        );
        assert_eq!(
            observation.get("candidate_loss"),
            Some(&serde_json::json!(0.5))
        );
        assert!(!observation.contains_key("loss_before"));
        assert!(!observation.contains_key("loss_after"));
    }
}
