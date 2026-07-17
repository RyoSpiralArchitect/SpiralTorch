// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Probabilistic lane selection and distributed lane consensus.
//!
//! Rust owns the probability and consensus semantics in this module. External
//! stores and collectives may contribute votes, but they cannot clamp, repair,
//! or reinterpret an invalid distribution.

use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;

use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use st_tensor::{emit_tensor_op, emit_tensor_op_meta};

use super::wire::canonical_u64;

pub const LANE_MIN: i32 = 1;
pub const LANE_MAX: i32 = 4096;
pub const LANE_DISTRIBUTION_MAX_SUPPORT: usize = LANE_MAX as usize;
pub const LANE_DISTRIBUTION_MAX_SUPPORT_WIRE: u64 = LANE_MAX as u64;

pub const LANE_SAMPLE_KIND: &str = "spiraltorch.distributed_lane_sample";
pub const LANE_SAMPLE_CONTRACT_VERSION: &str = "spiraltorch.distributed_lane_sample.v1";
pub const LANE_CONSENSUS_REPORT_KIND: &str = "spiraltorch.lane_consensus_report";
pub const LANE_CONSENSUS_REPORT_CONTRACT_VERSION: &str = "spiraltorch.lane_consensus_report.v1";
pub const LANE_PROBABILITY_SEMANTIC_OWNER: &str = "st-core::distributed::prob_params";
pub const LANE_PROBABILITY_SEMANTIC_BACKEND: &str = "rust";
pub const LANE_SAMPLE_RNG_ALGORITHM: &str = "splitmix64_u53_v1";

const DISTRIBUTION_DIGEST_DOMAIN: &[u8] = b"spiraltorch.lane_distribution.v1\0";
const CONSENSUS_OUTPUT_DIGEST_DOMAIN: &[u8] = b"spiraltorch.lane_consensus.output.v1\0";
const PROBABILITY_SUM_BASE_TOLERANCE: f64 = 1.0e-12;
const PROBABILITY_SUM_EPSILON_MULTIPLIER: f64 = 8.0;
const DEFAULT_REDIS_SAMPLE_LIMIT: usize = 16;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LaneParams {
    pub lane: i32,
}

impl LaneParams {
    pub fn try_new(lane: i32) -> Result<Self, LaneProbabilityError> {
        let params = Self { lane };
        params.validate()?;
        Ok(params)
    }

    pub fn validate(self) -> Result<(), LaneProbabilityError> {
        if !(LANE_MIN..=LANE_MAX).contains(&self.lane) {
            return Err(LaneProbabilityError::LaneOutOfRange { lane: self.lane });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LaneProbability {
    pub params: LaneParams,
    pub probability: f64,
}

/// A non-empty, unique-support categorical distribution over valid lanes.
///
/// Construction and deserialization validate every probability, sort support by
/// lane, and normalize accepted mass to the Rust-owned representation. Raw
/// weights should enter through [`LaneDistribution::from_weights`], which uses
/// max-scaled normalization to avoid finite-weight overflow.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LaneDistribution {
    entries: Vec<LaneProbability>,
}

impl<'de> Deserialize<'de> for LaneDistribution {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            entries: Vec<LaneProbability>,
        }

        let wire = Wire::deserialize(deserializer)?;
        Self::from_probabilities(wire.entries).map_err(serde::de::Error::custom)
    }
}

impl LaneDistribution {
    pub fn from_probabilities(
        mut entries: Vec<LaneProbability>,
    ) -> Result<Self, LaneProbabilityError> {
        validate_support_size(entries.len())?;
        let mut seen = HashSet::with_capacity(entries.len());
        for (position, entry) in entries.iter().enumerate() {
            entry.params.validate()?;
            if !seen.insert(entry.params.lane) {
                return Err(LaneProbabilityError::DuplicateLane {
                    lane: entry.params.lane,
                });
            }
            validate_positive_finite("probability", position, entry.probability)?;
        }
        entries.sort_by_key(|entry| entry.params.lane);

        let mut mass = 0.0;
        let mut compensation = 0.0;
        for entry in &entries {
            compensated_add(
                &mut mass,
                &mut compensation,
                entry.probability,
                "probability_mass",
            )?;
        }
        let tolerance = probability_sum_tolerance(entries.len());
        if (mass - 1.0).abs() > tolerance {
            return Err(LaneProbabilityError::InvalidProbabilityMass { mass, tolerance });
        }
        for (position, entry) in entries.iter_mut().enumerate() {
            entry.probability /= mass;
            validate_positive_finite("normalized_probability", position, entry.probability)?;
        }
        Ok(Self { entries })
    }

    pub fn from_weights<I>(entries: I) -> Result<Self, LaneProbabilityError>
    where
        I: IntoIterator<Item = (LaneParams, f64)>,
    {
        let mut entries = entries.into_iter().collect::<Vec<_>>();
        validate_support_size(entries.len())?;
        let mut seen = HashSet::with_capacity(entries.len());
        let mut max_weight = 0.0f64;
        for (position, (params, weight)) in entries.iter().enumerate() {
            params.validate()?;
            if !seen.insert(params.lane) {
                return Err(LaneProbabilityError::DuplicateLane { lane: params.lane });
            }
            validate_positive_finite("weight", position, *weight)?;
            max_weight = max_weight.max(*weight);
        }
        entries.sort_by_key(|(params, _)| params.lane);

        let mut scaled_sum = 0.0;
        let mut compensation = 0.0;
        let mut scaled = Vec::with_capacity(entries.len());
        for (position, (_, weight)) in entries.iter().enumerate() {
            let value = weight / max_weight;
            validate_positive_finite("scaled_weight", position, value)?;
            compensated_add(
                &mut scaled_sum,
                &mut compensation,
                value,
                "scaled_weight_sum",
            )?;
            scaled.push(value);
        }
        validate_positive_finite("scaled_weight_sum", 0, scaled_sum)?;

        let probabilities = entries
            .into_iter()
            .zip(scaled)
            .enumerate()
            .map(|(position, ((params, _), scaled_weight))| {
                let probability = scaled_weight / scaled_sum;
                validate_positive_finite("normalized_probability", position, probability)?;
                Ok(LaneProbability {
                    params,
                    probability,
                })
            })
            .collect::<Result<Vec<_>, LaneProbabilityError>>()?;
        Self::from_probabilities(probabilities)
    }

    pub fn entries(&self) -> &[LaneProbability] {
        &self.entries
    }

    pub fn mass(&self) -> f64 {
        let mut mass = 0.0;
        let mut compensation = 0.0;
        for entry in &self.entries {
            let adjusted = entry.probability - compensation;
            let next = mass + adjusted;
            compensation = (next - mass) - adjusted;
            mass = next;
        }
        mass
    }

    pub fn entropy(&self) -> Result<f64, LaneProbabilityError> {
        let mut entropy = 0.0;
        let mut compensation = 0.0;
        for entry in &self.entries {
            let term = -entry.probability * entry.probability.ln();
            compensated_add(&mut entropy, &mut compensation, term, "entropy")?;
        }
        Ok(entropy)
    }

    pub fn effective_sample_size(&self) -> Result<f64, LaneProbabilityError> {
        let mut sum_squares = 0.0;
        let mut compensation = 0.0;
        for entry in &self.entries {
            compensated_add(
                &mut sum_squares,
                &mut compensation,
                entry.probability * entry.probability,
                "probability_square_sum",
            )?;
        }
        let effective = 1.0 / sum_squares;
        validate_positive_finite("effective_sample_size", 0, effective)?;
        Ok(effective)
    }

    pub fn sha256(&self) -> String {
        distribution_sha256(&self.entries)
    }

    pub fn sample(&self, seed: u64) -> Result<LaneSample, LaneProbabilityError> {
        let draw = canonical_lane_draw(seed);
        let selected = self.selected_index_for_draw(draw)?;
        let selected_entry = self.entries[selected];
        let sample = LaneSample {
            kind: LANE_SAMPLE_KIND.to_owned(),
            contract_version: LANE_SAMPLE_CONTRACT_VERSION.to_owned(),
            semantic_owner: LANE_PROBABILITY_SEMANTIC_OWNER.to_owned(),
            semantic_backend: LANE_PROBABILITY_SEMANTIC_BACKEND.to_owned(),
            rng_algorithm: LANE_SAMPLE_RNG_ALGORITHM.to_owned(),
            seed,
            support_size: u64::try_from(self.entries.len()).map_err(|_| {
                LaneProbabilityError::CountOverflow {
                    field: "support_size",
                }
            })?,
            support_index: u64::try_from(selected).map_err(|_| {
                LaneProbabilityError::CountOverflow {
                    field: "support_index",
                }
            })?,
            draw,
            params: selected_entry.params,
            probability: selected_entry.probability,
            distribution_sha256: self.sha256(),
            committed: true,
        };
        self.validate_sample(&sample)?;
        emit_tensor_op("distributed_lane_sample", &[self.entries.len()], &[1]);
        emit_tensor_op_meta("distributed_lane_sample", || {
            serde_json::to_value(&sample).expect("validated lane sample must serialize")
        });
        Ok(sample)
    }

    /// Validates a sample against this exact distribution, including its draw.
    pub fn validate_sample(&self, sample: &LaneSample) -> Result<(), LaneProbabilityError> {
        sample.validate()?;
        let support_size =
            u64::try_from(self.entries.len()).map_err(|_| LaneProbabilityError::CountOverflow {
                field: "support_size",
            })?;
        if sample.support_size != support_size {
            return Err(invalid_report(
                "support_size",
                format!(
                    "sample support {} does not match distribution support {support_size}",
                    sample.support_size
                ),
            ));
        }
        if sample.distribution_sha256 != self.sha256() {
            return Err(invalid_report(
                "distribution_sha256",
                "sample does not commit to this lane distribution",
            ));
        }
        let expected_index = self.selected_index_for_draw(sample.draw)?;
        let actual_index = usize::try_from(sample.support_index).map_err(|_| {
            invalid_report("support_index", "sample index does not fit this runtime")
        })?;
        if actual_index != expected_index {
            return Err(invalid_report(
                "support_index",
                format!("draw selects support index {expected_index}, got {actual_index}"),
            ));
        }
        let expected = self.entries[expected_index];
        if sample.params != expected.params
            || sample.probability.to_bits() != expected.probability.to_bits()
        {
            return Err(invalid_report(
                "params",
                "sample payload does not match the selected distribution entry",
            ));
        }
        Ok(())
    }

    /// Validates a consensus outcome against this exact distribution.
    pub fn validate_consensus_outcome(
        &self,
        outcome: &LaneConsensusOutcome,
    ) -> Result<(), LaneProbabilityError> {
        outcome.report.validate()?;
        let expected = build_lane_consensus_outcome(self, outcome.report.policy)?;
        if outcome.params != expected.params
            || !lane_consensus_reports_exactly_equal(&outcome.report, &expected.report)
        {
            return Err(invalid_report(
                "outcome",
                "consensus outcome does not match this distribution and policy",
            ));
        }
        Ok(())
    }

    fn selected_index_for_draw(&self, draw: f64) -> Result<usize, LaneProbabilityError> {
        if !draw.is_finite() || !(0.0..1.0).contains(&draw) {
            return Err(invalid_report("draw", "must be finite in [0, 1)"));
        }
        let threshold = draw * self.mass();
        let mut cumulative = 0.0;
        let mut selected = self.entries.len() - 1;
        for (index, entry) in self.entries.iter().enumerate() {
            cumulative = checked_add(
                "sample_cumulative_probability",
                index,
                cumulative,
                entry.probability,
            )?;
            if threshold < cumulative {
                selected = index;
                break;
            }
        }
        Ok(selected)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LaneSample {
    pub kind: String,
    pub contract_version: String,
    pub semantic_owner: String,
    pub semantic_backend: String,
    pub rng_algorithm: String,
    #[serde(with = "canonical_u64")]
    pub seed: u64,
    #[serde(with = "canonical_u64")]
    pub support_size: u64,
    #[serde(with = "canonical_u64")]
    pub support_index: u64,
    pub draw: f64,
    pub params: LaneParams,
    pub probability: f64,
    pub distribution_sha256: String,
    pub committed: bool,
}

impl LaneSample {
    pub fn validate(&self) -> Result<(), LaneProbabilityError> {
        validate_identity(
            &self.kind,
            LANE_SAMPLE_KIND,
            &self.contract_version,
            LANE_SAMPLE_CONTRACT_VERSION,
            &self.semantic_owner,
            &self.semantic_backend,
        )?;
        self.params.validate()?;
        if self.support_size == 0 || self.support_size > LANE_DISTRIBUTION_MAX_SUPPORT_WIRE {
            return Err(invalid_report(
                "support_size",
                "must be within the supported lane distribution size",
            ));
        }
        if self.support_index >= self.support_size {
            return Err(invalid_report(
                "support_index",
                "must identify an entry within non-empty support",
            ));
        }
        if !self.draw.is_finite() || !(0.0..1.0).contains(&self.draw) {
            return Err(invalid_report("draw", "must be finite in [0, 1)"));
        }
        if self.rng_algorithm != LANE_SAMPLE_RNG_ALGORITHM {
            return Err(invalid_report(
                "rng_algorithm",
                format!("must be {LANE_SAMPLE_RNG_ALGORITHM}"),
            ));
        }
        let expected_draw = canonical_lane_draw(self.seed);
        if self.draw.to_bits() != expected_draw.to_bits() {
            return Err(invalid_report(
                "draw",
                "does not match the canonical Rust draw for seed",
            ));
        }
        if !self.probability.is_finite() || !(0.0..=1.0).contains(&self.probability) {
            return Err(invalid_report("probability", "must be finite in (0, 1]"));
        }
        if self.probability <= 0.0 {
            return Err(invalid_report("probability", "must be finite in (0, 1]"));
        }
        if !valid_sha256(&self.distribution_sha256) {
            return Err(invalid_report(
                "distribution_sha256",
                "must be a 64-digit lowercase SHA-256",
            ));
        }
        if !self.committed {
            return Err(invalid_report(
                "committed",
                "lane samples must describe a committed draw",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LaneConsensusPolicy {
    /// Normalize by observed mass, then round the positive-lane mean half up.
    WeightedMean,
    /// Select the first ascending lane whose cumulative mass reaches one half.
    WeightedMedian,
}

impl LaneConsensusPolicy {
    pub fn parse(value: &str) -> Result<Self, LaneProbabilityError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "mean" | "weighted_mean" => Ok(Self::WeightedMean),
            "median" | "weighted_median" => Ok(Self::WeightedMedian),
            _ => Err(LaneProbabilityError::UnknownPolicy {
                value: value.to_owned(),
            }),
        }
    }

    fn wire_name(self) -> &'static str {
        match self {
            Self::WeightedMean => "weighted_mean",
            Self::WeightedMedian => "weighted_median",
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LaneConsensusReport {
    pub kind: String,
    pub contract_version: String,
    pub semantic_owner: String,
    pub semantic_backend: String,
    pub policy: LaneConsensusPolicy,
    #[serde(with = "canonical_u64")]
    pub support_size: u64,
    pub probability_mass: f64,
    pub entropy: f64,
    pub effective_sample_size: f64,
    pub selected_lane: i32,
    pub input_sha256: String,
    pub output_sha256: String,
    pub committed: bool,
}

impl LaneConsensusReport {
    pub fn validate(&self) -> Result<(), LaneProbabilityError> {
        validate_identity(
            &self.kind,
            LANE_CONSENSUS_REPORT_KIND,
            &self.contract_version,
            LANE_CONSENSUS_REPORT_CONTRACT_VERSION,
            &self.semantic_owner,
            &self.semantic_backend,
        )?;
        if self.support_size == 0 || self.support_size > LANE_DISTRIBUTION_MAX_SUPPORT_WIRE {
            return Err(invalid_report(
                "support_size",
                "must be within the supported lane distribution size",
            ));
        }
        let support_size = self.support_size as f64;
        let tolerance = probability_sum_tolerance(self.support_size as usize);
        if !self.probability_mass.is_finite() || (self.probability_mass - 1.0).abs() > tolerance {
            return Err(invalid_report(
                "probability_mass",
                "must be finite and sum to one within contract tolerance",
            ));
        }
        let max_entropy = support_size.ln();
        if !self.entropy.is_finite()
            || self.entropy < -tolerance
            || self.entropy > max_entropy + tolerance
        {
            return Err(invalid_report(
                "entropy",
                "must be within the support's Shannon entropy bounds",
            ));
        }
        if !self.effective_sample_size.is_finite()
            || self.effective_sample_size < 1.0 - tolerance
            || self.effective_sample_size > support_size + tolerance
        {
            return Err(invalid_report(
                "effective_sample_size",
                "must be within [1, support_size]",
            ));
        }
        let params = LaneParams::try_new(self.selected_lane)?;
        for (field, digest) in [
            ("input_sha256", self.input_sha256.as_str()),
            ("output_sha256", self.output_sha256.as_str()),
        ] {
            if !valid_sha256(digest) {
                return Err(invalid_report(
                    field,
                    "must be a 64-digit lowercase SHA-256",
                ));
            }
        }
        let expected_output =
            consensus_output_sha256(self.policy, params, self.input_sha256.as_str());
        if self.output_sha256 != expected_output {
            return Err(invalid_report(
                "output_sha256",
                "does not bind the policy, selected lane, and input commitment",
            ));
        }
        if !self.committed {
            return Err(invalid_report(
                "committed",
                "lane consensus reports must describe a committed decision",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LaneConsensusOutcome {
    pub params: LaneParams,
    pub report: LaneConsensusReport,
}

/// Resolves one validated lane distribution under a named deterministic policy.
pub fn consensus_lane_params(
    distribution: &LaneDistribution,
    policy: LaneConsensusPolicy,
) -> Result<LaneConsensusOutcome, LaneProbabilityError> {
    let outcome = build_lane_consensus_outcome(distribution, policy)?;
    distribution.validate_consensus_outcome(&outcome)?;
    emit_tensor_op(
        "distributed_lane_consensus",
        &[distribution.entries.len()],
        &[1],
    );
    emit_tensor_op_meta("distributed_lane_consensus", || {
        serde_json::to_value(&outcome.report)
            .expect("validated lane consensus report must serialize")
    });
    Ok(outcome)
}

fn build_lane_consensus_outcome(
    distribution: &LaneDistribution,
    policy: LaneConsensusPolicy,
) -> Result<LaneConsensusOutcome, LaneProbabilityError> {
    let selected_lane = match policy {
        LaneConsensusPolicy::WeightedMean => {
            let mut weighted_sum = 0.0;
            let mut compensation = 0.0;
            for entry in &distribution.entries {
                compensated_add(
                    &mut weighted_sum,
                    &mut compensation,
                    f64::from(entry.params.lane) * entry.probability,
                    "weighted_lane_sum",
                )?;
            }
            let mean = weighted_sum / distribution.mass();
            if mean < f64::from(LANE_MIN) || mean > f64::from(LANE_MAX) {
                return Err(LaneProbabilityError::DerivedLaneOutOfRange { value: mean });
            }
            mean.round() as i32
        }
        LaneConsensusPolicy::WeightedMedian => {
            let mut entries = distribution.entries.clone();
            entries.sort_by_key(|entry| entry.params.lane);
            let midpoint = distribution.mass() * 0.5;
            let mut cumulative = 0.0;
            let mut selected = entries
                .last()
                .ok_or(LaneProbabilityError::EmptySupport)?
                .params
                .lane;
            for (position, entry) in entries.iter().enumerate() {
                cumulative = checked_add(
                    "weighted_median_cumulative_probability",
                    position,
                    cumulative,
                    entry.probability,
                )?;
                if cumulative >= midpoint {
                    selected = entry.params.lane;
                    break;
                }
            }
            selected
        }
    };
    let params = LaneParams::try_new(selected_lane)?;
    let input_sha256 = distribution.sha256();
    let output_sha256 = consensus_output_sha256(policy, params, &input_sha256);
    let report = LaneConsensusReport {
        kind: LANE_CONSENSUS_REPORT_KIND.to_owned(),
        contract_version: LANE_CONSENSUS_REPORT_CONTRACT_VERSION.to_owned(),
        semantic_owner: LANE_PROBABILITY_SEMANTIC_OWNER.to_owned(),
        semantic_backend: LANE_PROBABILITY_SEMANTIC_BACKEND.to_owned(),
        policy,
        support_size: u64::try_from(distribution.entries.len()).map_err(|_| {
            LaneProbabilityError::CountOverflow {
                field: "support_size",
            }
        })?,
        probability_mass: distribution.mass(),
        entropy: distribution.entropy()?,
        effective_sample_size: distribution.effective_sample_size()?,
        selected_lane,
        input_sha256,
        output_sha256,
        committed: true,
    };
    report.validate()?;
    Ok(LaneConsensusOutcome { params, report })
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct LaneConsensusRuntimeConfig {
    pub policy: LaneConsensusPolicy,
    pub include_redis: bool,
    pub include_hip: bool,
    pub redis_sample_limit: usize,
}

impl Default for LaneConsensusRuntimeConfig {
    fn default() -> Self {
        Self {
            policy: LaneConsensusPolicy::WeightedMean,
            include_redis: false,
            include_hip: false,
            redis_sample_limit: DEFAULT_REDIS_SAMPLE_LIMIT,
        }
    }
}

impl LaneConsensusRuntimeConfig {
    pub fn validate(self) -> Result<(), LaneProbabilityError> {
        if self.redis_sample_limit == 0 || self.redis_sample_limit > LANE_DISTRIBUTION_MAX_SUPPORT {
            return Err(LaneProbabilityError::InvalidRuntimeConfig {
                field: "redis_sample_limit",
                message: format!(
                    "must be within 1..={LANE_DISTRIBUTION_MAX_SUPPORT}, got {}",
                    self.redis_sample_limit
                ),
            });
        }
        Ok(())
    }

    /// Explicitly opts into the historical environment-driven runtime policy.
    pub fn from_env() -> Result<Self, LaneProbabilityError> {
        let policy = match std::env::var("SPIRAL_UNISON_AGG") {
            Ok(value) => LaneConsensusPolicy::parse(&value)?,
            Err(std::env::VarError::NotPresent) => LaneConsensusPolicy::WeightedMean,
            Err(error) => {
                return Err(LaneProbabilityError::RuntimeEnvironment {
                    field: "SPIRAL_UNISON_AGG",
                    message: error.to_string(),
                })
            }
        };
        let include_redis = std::env::var_os("REDIS_URL").is_some();
        let include_hip = cfg!(all(feature = "hip", feature = "hip-real"));
        let redis_sample_limit = match std::env::var("SPIRAL_UNISON_REDIS_SAMPLES") {
            Ok(value) => value.parse::<usize>().map_err(|error| {
                LaneProbabilityError::RuntimeEnvironment {
                    field: "SPIRAL_UNISON_REDIS_SAMPLES",
                    message: error.to_string(),
                }
            })?,
            Err(std::env::VarError::NotPresent) => DEFAULT_REDIS_SAMPLE_LIMIT,
            Err(error) => {
                return Err(LaneProbabilityError::RuntimeEnvironment {
                    field: "SPIRAL_UNISON_REDIS_SAMPLES",
                    message: error.to_string(),
                })
            }
        };
        let config = Self {
            policy,
            include_redis,
            include_hip,
            redis_sample_limit,
        };
        config.validate()?;
        Ok(config)
    }
}

/// Gathers explicitly enabled runtime votes and resolves them through the same
/// pure Rust distribution contract used by in-process callers.
pub fn consensus_runtime_lane_params(
    local: LaneParams,
    config: LaneConsensusRuntimeConfig,
) -> Result<LaneConsensusOutcome, LaneProbabilityError> {
    local.validate()?;
    config.validate()?;

    let (mut votes, local_votes, hip_votes) = if config.include_hip {
        let gathered = gather_hip_lane_votes(local)?;
        let count = gathered.len();
        (gathered, 0usize, count)
    } else {
        (vec![(local, 1.0)], 1usize, 0usize)
    };
    let redis_votes = if config.include_redis {
        let gathered = fetch_redis_lane_votes(config.redis_sample_limit)?;
        let count = gathered.len();
        votes.extend(gathered);
        count
    } else {
        0
    };

    let raw_vote_count = votes.len();
    let distribution = distribution_from_votes(votes)?;
    let outcome = consensus_lane_params(&distribution, config.policy)?;
    emit_tensor_op_meta("distributed_lane_consensus_runtime", || {
        serde_json::json!({
            "kind": "st_core_distributed_lane_consensus_runtime",
            "semantic_owner": LANE_PROBABILITY_SEMANTIC_OWNER,
            "semantic_backend": LANE_PROBABILITY_SEMANTIC_BACKEND,
            "control_backend": "rust_host",
            "local_votes": local_votes,
            "redis_votes": redis_votes,
            "hip_votes": hip_votes,
            "raw_vote_count": raw_vote_count,
            "support_size": distribution.entries.len(),
            "redis_enabled": config.include_redis,
            "hip_enabled": config.include_hip,
            "consensus_output_sha256": outcome.report.output_sha256,
            "committed": true,
        })
    });
    Ok(outcome)
}

fn distribution_from_votes(
    mut votes: Vec<(LaneParams, f64)>,
) -> Result<LaneDistribution, LaneProbabilityError> {
    if votes.is_empty() {
        return Err(LaneProbabilityError::EmptySupport);
    }
    let mut max_weight = 0.0f64;
    for (position, (params, weight)) in votes.iter().enumerate() {
        params.validate()?;
        validate_positive_finite("vote_weight", position, *weight)?;
        max_weight = max_weight.max(*weight);
    }
    votes.sort_by(|(left_params, left_weight), (right_params, right_weight)| {
        left_params
            .lane
            .cmp(&right_params.lane)
            .then_with(|| left_weight.total_cmp(right_weight))
    });

    let mut aggregated = BTreeMap::<i32, f64>::new();
    for (position, (params, weight)) in votes.into_iter().enumerate() {
        let scaled_weight = weight / max_weight;
        validate_positive_finite("scaled_vote_weight", position, scaled_weight)?;
        let current = aggregated.entry(params.lane).or_insert(0.0);
        *current = checked_add(
            "aggregated_scaled_vote_weight",
            position,
            *current,
            scaled_weight,
        )?;
    }
    LaneDistribution::from_weights(
        aggregated
            .into_iter()
            .map(|(lane, weight)| (LaneParams { lane }, weight)),
    )
}

#[cfg(feature = "kv-redis")]
fn fetch_redis_lane_votes(
    sample_limit: usize,
) -> Result<Vec<(LaneParams, f64)>, LaneProbabilityError> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RedisLaneVote {
        lane: i32,
        #[serde(default = "unit_weight", alias = "probability", alias = "prob")]
        weight: f64,
    }

    fn unit_weight() -> f64 {
        1.0
    }

    let url =
        std::env::var("REDIS_URL").map_err(|error| LaneProbabilityError::RuntimeEnvironment {
            field: "REDIS_URL",
            message: error.to_string(),
        })?;
    let start = isize::try_from(sample_limit)
        .map_err(|_| LaneProbabilityError::CountOverflow {
            field: "redis_sample_limit",
        })?
        .checked_neg()
        .ok_or(LaneProbabilityError::CountOverflow {
            field: "redis_sample_limit",
        })?;
    let samples = st_kv::redis_lrange(&url, "spiral:heur:lparams", start, -1).map_err(|error| {
        LaneProbabilityError::ProviderFailure {
            provider: "redis",
            message: error.to_string(),
        }
    })?;
    if samples.is_empty() {
        return Err(LaneProbabilityError::EmptyProvider { provider: "redis" });
    }
    samples
        .into_iter()
        .enumerate()
        .map(|(position, sample)| {
            let vote = serde_json::from_str::<RedisLaneVote>(&sample).map_err(|error| {
                LaneProbabilityError::InvalidProviderPayload {
                    provider: "redis",
                    position,
                    message: error.to_string(),
                }
            })?;
            let params = LaneParams::try_new(vote.lane)?;
            validate_positive_finite("redis_vote_weight", position, vote.weight)?;
            Ok((params, vote.weight))
        })
        .collect()
}

#[cfg(not(feature = "kv-redis"))]
fn fetch_redis_lane_votes(
    _sample_limit: usize,
) -> Result<Vec<(LaneParams, f64)>, LaneProbabilityError> {
    Err(LaneProbabilityError::ProviderUnavailable { provider: "redis" })
}

#[cfg(all(feature = "hip", feature = "hip-real"))]
fn gather_hip_lane_votes(
    local: LaneParams,
) -> Result<Vec<(LaneParams, f64)>, LaneProbabilityError> {
    use st_backend_hip::rccl_comm::init_rccl_from_env;
    use st_backend_hip::real::{
        allgather_u64_dev, free, malloc, memcpy_d2h_async, memcpy_h2d_async, stream_synchronize,
        HipPtr, HipStream,
    };

    struct HipBuffer(HipPtr);

    impl HipBuffer {
        fn new(bytes: usize) -> Result<Self, LaneProbabilityError> {
            malloc(bytes)
                .map(Self)
                .map_err(|error| LaneProbabilityError::ProviderFailure {
                    provider: "hip_rccl",
                    message: error.to_string(),
                })
        }

        fn as_ptr(&self) -> HipPtr {
            self.0
        }

        fn release(mut self) -> Result<(), LaneProbabilityError> {
            free(self.0).map_err(|error| LaneProbabilityError::ProviderFailure {
                provider: "hip_rccl",
                message: error.to_string(),
            })?;
            self.0 = std::ptr::null_mut();
            Ok(())
        }
    }

    impl Drop for HipBuffer {
        fn drop(&mut self) {
            if !self.0.is_null() {
                let _ = free(self.0);
            }
        }
    }

    let comm = init_rccl_from_env().map_err(|error| LaneProbabilityError::ProviderFailure {
        provider: "hip_rccl",
        message: error.to_string(),
    })?;
    if comm.world <= 0 || comm.rank < 0 || comm.rank >= comm.world {
        return Err(LaneProbabilityError::ProviderFailure {
            provider: "hip_rccl",
            message: format!(
                "invalid topology rank {} within world {}",
                comm.rank, comm.world
            ),
        });
    }
    let world = usize::try_from(comm.world).map_err(|_| LaneProbabilityError::CountOverflow {
        field: "hip_world_size",
    })?;
    let element_bytes = std::mem::size_of::<u64>();
    let receive_bytes =
        element_bytes
            .checked_mul(world)
            .ok_or(LaneProbabilityError::CountOverflow {
                field: "hip_receive_bytes",
            })?;
    let stream = HipStream::create().map_err(|error| LaneProbabilityError::ProviderFailure {
        provider: "hip_rccl",
        message: error.to_string(),
    })?;
    let send = HipBuffer::new(element_bytes)?;
    let receive = HipBuffer::new(receive_bytes)?;
    let lane = u64::try_from(local.lane)
        .map_err(|_| LaneProbabilityError::LaneOutOfRange { lane: local.lane })?;
    let mut gathered = vec![0u64; world];

    unsafe {
        memcpy_h2d_async(
            send.as_ptr(),
            (&lane as *const u64).cast::<u8>(),
            element_bytes,
            &stream,
        )
        .map_err(|error| LaneProbabilityError::ProviderFailure {
            provider: "hip_rccl",
            message: error.to_string(),
        })?;
    }
    allgather_u64_dev(comm.comm, &stream, send.as_ptr(), receive.as_ptr(), 1).map_err(|error| {
        LaneProbabilityError::ProviderFailure {
            provider: "hip_rccl",
            message: error.to_string(),
        }
    })?;
    unsafe {
        memcpy_d2h_async(
            gathered.as_mut_ptr().cast::<u8>(),
            receive.as_ptr(),
            receive_bytes,
            &stream,
        )
        .map_err(|error| LaneProbabilityError::ProviderFailure {
            provider: "hip_rccl",
            message: error.to_string(),
        })?;
    }
    stream_synchronize(&stream).map_err(|error| LaneProbabilityError::ProviderFailure {
        provider: "hip_rccl",
        message: error.to_string(),
    })?;
    send.release()?;
    receive.release()?;

    gathered
        .into_iter()
        .map(|lane| {
            let lane = i32::try_from(lane).map_err(|_| LaneProbabilityError::ProviderFailure {
                provider: "hip_rccl",
                message: format!("gathered lane {lane} does not fit i32"),
            })?;
            Ok((LaneParams::try_new(lane)?, 1.0))
        })
        .collect()
}

#[cfg(not(all(feature = "hip", feature = "hip-real")))]
fn gather_hip_lane_votes(
    _local: LaneParams,
) -> Result<Vec<(LaneParams, f64)>, LaneProbabilityError> {
    Err(LaneProbabilityError::ProviderUnavailable {
        provider: "hip_rccl",
    })
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum LaneProbabilityError {
    #[error("lane distribution must contain at least one entry")]
    EmptySupport,
    #[error("lane distribution support {got} exceeds maximum {max}")]
    SupportTooLarge { got: usize, max: usize },
    #[error("lane {lane} must be within {LANE_MIN}..={LANE_MAX}")]
    LaneOutOfRange { lane: i32 },
    #[error("lane distribution contains duplicate lane {lane}")]
    DuplicateLane { lane: i32 },
    #[error("{field} at position {position} must be finite and positive, got {value}")]
    InvalidPositiveValue {
        field: &'static str,
        position: usize,
        value: f64,
    },
    #[error("lane probability mass {mass} differs from one by more than {tolerance}")]
    InvalidProbabilityMass { mass: f64, tolerance: f64 },
    #[error("derived {field} at position {position} is non-finite: {value}")]
    NonFiniteDerived {
        field: &'static str,
        position: usize,
        value: f64,
    },
    #[error("derived weighted lane {value} must be within {LANE_MIN}..={LANE_MAX}")]
    DerivedLaneOutOfRange { value: f64 },
    #[error("unknown lane consensus policy {value:?}")]
    UnknownPolicy { value: String },
    #[error("lane probability count overflow at {field}")]
    CountOverflow { field: &'static str },
    #[error("invalid lane consensus runtime config at {field}: {message}")]
    InvalidRuntimeConfig {
        field: &'static str,
        message: String,
    },
    #[error("invalid runtime environment {field}: {message}")]
    RuntimeEnvironment {
        field: &'static str,
        message: String,
    },
    #[error("lane proposal provider {provider} is unavailable in this build")]
    ProviderUnavailable { provider: &'static str },
    #[error("lane proposal provider {provider} returned no votes")]
    EmptyProvider { provider: &'static str },
    #[error("lane proposal provider {provider} failed: {message}")]
    ProviderFailure {
        provider: &'static str,
        message: String,
    },
    #[error("invalid {provider} lane vote at position {position}: {message}")]
    InvalidProviderPayload {
        provider: &'static str,
        position: usize,
        message: String,
    },
    #[error("invalid lane probability report at {field}: {message}")]
    InvalidReport { field: String, message: String },
}

pub fn probability_sum_tolerance(value_count: usize) -> f64 {
    PROBABILITY_SUM_BASE_TOLERANCE
        + value_count as f64 * f64::EPSILON * PROBABILITY_SUM_EPSILON_MULTIPLIER
}

/// Stable seed-to-unit-interval mapping used by the v1 sampling contract.
pub fn canonical_lane_draw(seed: u64) -> f64 {
    let mut value = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    let mantissa = value >> 11;
    mantissa as f64 * (1.0 / ((1u64 << 53) as f64))
}

fn validate_support_size(size: usize) -> Result<(), LaneProbabilityError> {
    if size == 0 {
        return Err(LaneProbabilityError::EmptySupport);
    }
    if size > LANE_DISTRIBUTION_MAX_SUPPORT {
        return Err(LaneProbabilityError::SupportTooLarge {
            got: size,
            max: LANE_DISTRIBUTION_MAX_SUPPORT,
        });
    }
    Ok(())
}

fn validate_positive_finite(
    field: &'static str,
    position: usize,
    value: f64,
) -> Result<(), LaneProbabilityError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(LaneProbabilityError::InvalidPositiveValue {
            field,
            position,
            value,
        });
    }
    Ok(())
}

fn checked_add(
    field: &'static str,
    position: usize,
    left: f64,
    right: f64,
) -> Result<f64, LaneProbabilityError> {
    let value = left + right;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(LaneProbabilityError::NonFiniteDerived {
            field,
            position,
            value,
        })
    }
}

fn compensated_add(
    sum: &mut f64,
    compensation: &mut f64,
    value: f64,
    field: &'static str,
) -> Result<(), LaneProbabilityError> {
    let adjusted = value - *compensation;
    let next = *sum + adjusted;
    if !next.is_finite() {
        return Err(LaneProbabilityError::NonFiniteDerived {
            field,
            position: 0,
            value: next,
        });
    }
    *compensation = (next - *sum) - adjusted;
    *sum = next;
    Ok(())
}

fn distribution_sha256(entries: &[LaneProbability]) -> String {
    let mut digest = Sha256::new();
    digest.update(DISTRIBUTION_DIGEST_DOMAIN);
    digest.update((entries.len() as u64).to_le_bytes());
    for entry in entries {
        digest.update(entry.params.lane.to_le_bytes());
        digest.update(entry.probability.to_bits().to_le_bytes());
    }
    hex_digest(&digest.finalize())
}

fn consensus_output_sha256(
    policy: LaneConsensusPolicy,
    params: LaneParams,
    input_sha256: &str,
) -> String {
    let mut digest = Sha256::new();
    digest.update(CONSENSUS_OUTPUT_DIGEST_DOMAIN);
    digest.update(policy.wire_name().as_bytes());
    digest.update([0]);
    digest.update(params.lane.to_le_bytes());
    digest.update(input_sha256.as_bytes());
    hex_digest(&digest.finalize())
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn lane_consensus_reports_exactly_equal(
    left: &LaneConsensusReport,
    right: &LaneConsensusReport,
) -> bool {
    left == right
        && left.probability_mass.to_bits() == right.probability_mass.to_bits()
        && left.entropy.to_bits() == right.entropy.to_bits()
        && left.effective_sample_size.to_bits() == right.effective_sample_size.to_bits()
}

fn validate_identity(
    kind: &str,
    expected_kind: &str,
    version: &str,
    expected_version: &str,
    owner: &str,
    backend: &str,
) -> Result<(), LaneProbabilityError> {
    for (field, actual, expected) in [
        ("kind", kind, expected_kind),
        ("contract_version", version, expected_version),
        ("semantic_owner", owner, LANE_PROBABILITY_SEMANTIC_OWNER),
        (
            "semantic_backend",
            backend,
            LANE_PROBABILITY_SEMANTIC_BACKEND,
        ),
    ] {
        if actual != expected {
            return Err(invalid_report(
                field,
                format!("must be {expected}, got {actual}"),
            ));
        }
    }
    Ok(())
}

fn invalid_report(field: impl Into<String>, message: impl Into<String>) -> LaneProbabilityError {
    LaneProbabilityError::InvalidReport {
        field: field.into(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn distribution(entries: &[(i32, f64)]) -> LaneDistribution {
        LaneDistribution::from_weights(
            entries
                .iter()
                .map(|&(lane, weight)| (LaneParams::try_new(lane).unwrap(), weight)),
        )
        .unwrap()
    }

    #[test]
    fn lane_params_reject_out_of_contract_values_instead_of_clamping() {
        assert_eq!(
            LaneParams::try_new(0),
            Err(LaneProbabilityError::LaneOutOfRange { lane: 0 })
        );
        assert_eq!(
            LaneParams::try_new(LANE_MAX + 1),
            Err(LaneProbabilityError::LaneOutOfRange { lane: LANE_MAX + 1 })
        );
    }

    #[test]
    fn distribution_normalizes_extreme_finite_weights_without_overflow() {
        let distribution = LaneDistribution::from_weights([
            (LaneParams::try_new(8).unwrap(), f64::MAX),
            (LaneParams::try_new(16).unwrap(), f64::MAX / 2.0),
        ])
        .unwrap();

        assert!((distribution.mass() - 1.0).abs() <= probability_sum_tolerance(2));
        assert!((distribution.entries()[0].probability - 2.0 / 3.0).abs() < 1.0e-15);
        assert!((distribution.entries()[1].probability - 1.0 / 3.0).abs() < 1.0e-15);
    }

    #[test]
    fn imported_probability_mass_is_canonicalized_before_consensus() {
        let scale = 1.0 + 5.0e-13;
        let distribution = LaneDistribution::from_probabilities(vec![
            LaneProbability {
                params: LaneParams::try_new(1).unwrap(),
                probability: (0.5 + 2.0e-13) * scale,
            },
            LaneProbability {
                params: LaneParams::try_new(2).unwrap(),
                probability: (0.5 - 2.0e-13) * scale,
            },
        ])
        .unwrap();
        let deterministic = LaneDistribution::from_probabilities(vec![LaneProbability {
            params: LaneParams::try_new(8).unwrap(),
            probability: 1.0 + 5.0e-13,
        }])
        .unwrap();

        assert!((distribution.mass() - 1.0).abs() <= f64::EPSILON);
        assert_eq!(
            consensus_lane_params(&distribution, LaneConsensusPolicy::WeightedMean)
                .unwrap()
                .params
                .lane,
            1
        );
        assert_eq!(
            deterministic.entries()[0].probability.to_bits(),
            1.0f64.to_bits()
        );
        deterministic.sample(7).unwrap();
    }

    #[test]
    fn lane_support_order_is_canonical_across_sampling_and_consensus() {
        let first = LaneDistribution::from_weights([
            (
                LaneParams::try_new(16).unwrap(),
                f64::MAX * f64::EPSILON / 2.0,
            ),
            (LaneParams::try_new(4).unwrap(), f64::MAX),
            (LaneParams::try_new(8).unwrap(), f64::MAX * f64::EPSILON),
        ])
        .unwrap();
        let second = LaneDistribution::from_weights([
            (LaneParams::try_new(8).unwrap(), f64::MAX * f64::EPSILON),
            (
                LaneParams::try_new(16).unwrap(),
                f64::MAX * f64::EPSILON / 2.0,
            ),
            (LaneParams::try_new(4).unwrap(), f64::MAX),
        ])
        .unwrap();

        assert_eq!(first, second);
        assert_eq!(first.sha256(), second.sha256());
        assert_eq!(first.sample(42).unwrap(), second.sample(42).unwrap());
        assert_eq!(
            consensus_lane_params(&first, LaneConsensusPolicy::WeightedMean).unwrap(),
            consensus_lane_params(&second, LaneConsensusPolicy::WeightedMean).unwrap()
        );
    }

    #[test]
    fn distribution_rejects_duplicate_non_finite_zero_and_invalid_mass() {
        assert!(matches!(
            LaneDistribution::from_weights([
                (LaneParams::try_new(8).unwrap(), 1.0),
                (LaneParams::try_new(8).unwrap(), 2.0),
            ]),
            Err(LaneProbabilityError::DuplicateLane { lane: 8 })
        ));
        assert!(matches!(
            LaneDistribution::from_weights([(LaneParams::try_new(8).unwrap(), f64::NAN)]),
            Err(LaneProbabilityError::InvalidPositiveValue { .. })
        ));
        assert!(matches!(
            LaneDistribution::from_weights([(LaneParams::try_new(8).unwrap(), 0.0)]),
            Err(LaneProbabilityError::InvalidPositiveValue { .. })
        ));
        assert!(matches!(
            LaneDistribution::from_probabilities(vec![LaneProbability {
                params: LaneParams::try_new(8).unwrap(),
                probability: 0.5,
            }]),
            Err(LaneProbabilityError::InvalidProbabilityMass { .. })
        ));
    }

    #[test]
    fn lane_sampling_is_seeded_reproducible_and_uses_string_u64_wire_values() {
        let distribution = distribution(&[(4, 1.0), (8, 3.0), (16, 2.0)]);
        let first = distribution.sample(u64::MAX).unwrap();
        let second = distribution.sample(u64::MAX).unwrap();

        assert_eq!(first, second);
        first.validate().unwrap();
        distribution.validate_sample(&first).unwrap();
        let mut wrong_entry = first.clone();
        wrong_entry.params = LaneParams::try_new(64).unwrap();
        assert!(distribution.validate_sample(&wrong_entry).is_err());
        let mut wrong_seed = first.clone();
        wrong_seed.seed -= 1;
        assert!(distribution.validate_sample(&wrong_seed).is_err());
        let mut wrong_draw = first.clone();
        wrong_draw.draw = if first.support_index == 0 { 0.999 } else { 0.0 };
        assert!(distribution.validate_sample(&wrong_draw).is_err());
        let wire = serde_json::to_value(&first).unwrap();
        assert_eq!(wire["seed"], serde_json::json!(u64::MAX.to_string()));
        assert!(wire["support_index"].is_string());
        assert!(wire["support_size"].is_string());
        assert_eq!(wire["rng_algorithm"], LANE_SAMPLE_RNG_ALGORITHM);
        assert_eq!(serde_json::from_value::<LaneSample>(wire).unwrap(), first);

        let mut numeric_seed = serde_json::to_value(&first).unwrap();
        numeric_seed["seed"] = serde_json::json!(u64::MAX);
        assert!(serde_json::from_value::<LaneSample>(numeric_seed).is_err());
        let mut wrong_algorithm = first.clone();
        wrong_algorithm.rng_algorithm = "runtime_default".to_owned();
        assert!(wrong_algorithm.validate().is_err());
        let mut oversized_support = first.clone();
        oversized_support.support_size = LANE_DISTRIBUTION_MAX_SUPPORT_WIRE + 1;
        assert!(matches!(
            oversized_support.validate(),
            Err(LaneProbabilityError::InvalidReport { field, .. }) if field == "support_size"
        ));
        let mut invalid_index = first.clone();
        invalid_index.support_index = invalid_index.support_size;
        assert!(matches!(
            invalid_index.validate(),
            Err(LaneProbabilityError::InvalidReport { field, .. }) if field == "support_index"
        ));
    }

    #[test]
    fn canonical_lane_draw_has_fixed_cross_runtime_vectors() {
        assert_eq!(canonical_lane_draw(0).to_bits(), 0x3fec_4415_072f_63b9);
        assert_eq!(canonical_lane_draw(1).to_bits(), 0x3fe2_2145_bd91_204b);
        assert_eq!(
            canonical_lane_draw(u64::MAX).to_bits(),
            0x3fec_9b2e_2ee3_6ca5
        );
    }

    #[test]
    fn weighted_mean_and_median_have_distinct_deterministic_semantics() {
        let distribution = distribution(&[(1, 0.2), (10, 0.3), (100, 0.5)]);

        let mean = consensus_lane_params(&distribution, LaneConsensusPolicy::WeightedMean).unwrap();
        let median =
            consensus_lane_params(&distribution, LaneConsensusPolicy::WeightedMedian).unwrap();

        assert_eq!(mean.params.lane, 53);
        assert_eq!(median.params.lane, 10);
        mean.report.validate().unwrap();
        median.report.validate().unwrap();
        assert_ne!(mean.report.output_sha256, median.report.output_sha256);
    }

    #[test]
    fn consensus_report_rejects_tampering_unknown_fields_and_numeric_u64_wire() {
        let outcome = consensus_lane_params(
            &distribution(&[(8, 1.0), (16, 1.0)]),
            LaneConsensusPolicy::WeightedMean,
        )
        .unwrap();
        let mut report = outcome.report.clone();
        report.selected_lane = 0;
        assert!(report.validate().is_err());
        let mut valid_lane_tamper = outcome.report.clone();
        valid_lane_tamper.selected_lane = 9;
        assert!(valid_lane_tamper.validate().is_err());
        let mut outcome_tamper = outcome.clone();
        outcome_tamper.params = LaneParams::try_new(9).unwrap();
        assert!(LaneDistribution::from_weights([
            (LaneParams::try_new(8).unwrap(), 1.0),
            (LaneParams::try_new(16).unwrap(), 1.0),
        ])
        .unwrap()
        .validate_consensus_outcome(&outcome_tamper)
        .is_err());
        let deterministic = distribution(&[(8, 1.0)]);
        let mut signed_zero_tamper =
            consensus_lane_params(&deterministic, LaneConsensusPolicy::WeightedMean).unwrap();
        assert_eq!(signed_zero_tamper.report.entropy, 0.0);
        signed_zero_tamper.report.entropy =
            f64::from_bits(signed_zero_tamper.report.entropy.to_bits() ^ (1u64 << 63));
        assert!(deterministic
            .validate_consensus_outcome(&signed_zero_tamper)
            .is_err());

        let mut wire = serde_json::to_value(&outcome.report).unwrap();
        wire.as_object_mut()
            .unwrap()
            .insert("commander".to_owned(), serde_json::json!("python"));
        assert!(serde_json::from_value::<LaneConsensusReport>(wire).is_err());

        let mut numeric = serde_json::to_value(&outcome.report).unwrap();
        numeric["support_size"] = serde_json::json!(2);
        assert!(serde_json::from_value::<LaneConsensusReport>(numeric).is_err());
    }

    #[test]
    fn runtime_consensus_is_local_only_by_default_and_rejects_unavailable_sources() {
        let outcome = consensus_runtime_lane_params(
            LaneParams::try_new(32).unwrap(),
            LaneConsensusRuntimeConfig::default(),
        )
        .unwrap();
        assert_eq!(outcome.params.lane, 32);

        #[cfg(not(feature = "kv-redis"))]
        assert_eq!(
            consensus_runtime_lane_params(
                LaneParams::try_new(32).unwrap(),
                LaneConsensusRuntimeConfig {
                    include_redis: true,
                    ..LaneConsensusRuntimeConfig::default()
                },
            ),
            Err(LaneProbabilityError::ProviderUnavailable { provider: "redis" })
        );

        #[cfg(not(all(feature = "hip", feature = "hip-real")))]
        assert_eq!(
            consensus_runtime_lane_params(
                LaneParams::try_new(32).unwrap(),
                LaneConsensusRuntimeConfig {
                    include_hip: true,
                    ..LaneConsensusRuntimeConfig::default()
                },
            ),
            Err(LaneProbabilityError::ProviderUnavailable {
                provider: "hip_rccl"
            })
        );
    }

    #[test]
    fn runtime_vote_aggregation_scales_before_combining_duplicate_lanes() {
        let distribution = distribution_from_votes(vec![
            (LaneParams::try_new(8).unwrap(), f64::MAX),
            (LaneParams::try_new(16).unwrap(), f64::MAX / 2.0),
            (LaneParams::try_new(8).unwrap(), f64::MAX),
        ])
        .unwrap();

        assert_eq!(distribution.entries()[0].params.lane, 8);
        assert!((distribution.entries()[0].probability - 0.8).abs() < 1.0e-15);
        assert_eq!(distribution.entries()[1].params.lane, 16);
        assert!((distribution.entries()[1].probability - 0.2).abs() < 1.0e-15);
    }

    #[test]
    fn consensus_emits_rust_owned_probability_report() {
        let _lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let outcome = consensus_lane_params(
            &distribution(&[(8, 1.0), (32, 3.0)]),
            LaneConsensusPolicy::WeightedMedian,
        )
        .unwrap();
        st_tensor::set_thread_meta_observer(previous);

        let events = events.lock().unwrap();
        let meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "distributed_lane_consensus"
                    && data["kind"] == LANE_CONSENSUS_REPORT_KIND
            })
            .expect("distributed lane consensus report");
        assert_eq!(meta.1["semantic_backend"], "rust");
        assert_eq!(meta.1["selected_lane"], outcome.params.lane);
        assert_eq!(meta.1["committed"], true);
    }
}
