// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical evidence aggregation for Z-space optimizer studies.
//!
//! Clients collect matched run-card contrasts, while this module owns the
//! balanced cross-corpus validation and corpus-equal-weight summaries.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use thiserror::Error;

pub const ZSPACE_POLARITY_EVIDENCE_CONTRACT_VERSION: &str =
    "spiraltorch.zspace_polarity_evidence.v1";
pub const ZSPACE_POLARITY_EVIDENCE_KIND: &str = "spiraltorch.zspace_polarity_evidence";
pub const ZSPACE_POLARITY_EVIDENCE_SEMANTIC_OWNER: &str = "st-core::runtime::zspace_evidence";
pub const ZSPACE_POLARITY_EVIDENCE_SEMANTIC_BACKEND: &str = "rust";
pub const ZSPACE_POLARITY_EVIDENCE_AGGREGATION_RULE: &str =
    "validate balanced (corpus_id,seed) rows; summarize seeds within corpus; summarize corpus means with equal corpus weight";
pub const ZSPACE_POLARITY_EVIDENCE_CONTRAST_RULE: &str =
    "polarity_effect=complement_shape_effect-dose_normalized_shape_effect";
pub const ZSPACE_POLARITY_EVIDENCE_MIN_BOUNDED_CORPORA: usize = 3;
pub const ZSPACE_POLARITY_EVIDENCE_MIN_BOUNDED_SEEDS: usize = 3;
pub const ZSPACE_POLARITY_EVIDENCE_MAX_ROWS: usize = 100_000;
pub const ZSPACE_POLARITY_EVIDENCE_MAX_SAFE_SEED: u64 = 9_007_199_254_740_991;

const CONTRAST_RELATIVE_TOLERANCE: f64 = 1.0e-12;
const CONTRAST_ABSOLUTE_TOLERANCE: f64 = 1.0e-15;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpacePolarityEvidenceRow {
    pub corpus_id: String,
    pub seed: u64,
    pub dose_normalized_shape_effect: f64,
    pub complement_shape_effect: f64,
    pub polarity_effect: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpacePolarityEvidenceRequest {
    pub protocol_id: String,
    pub runtime_identity_id: String,
    pub trajectory_id: String,
    pub trajectory_policy_id: String,
    pub control_sequence_id: String,
    pub nominal_schedule_sequence_id: String,
    pub rows: Vec<ZSpacePolarityEvidenceRow>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ZSpacePolarityEvidenceDirection {
    LeftArmBetter,
    RightArmBetter,
    NoObservedDifference,
    Mixed,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpacePolaritySeedContrast {
    pub seed: u64,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpacePolarityCorpusContrastSummary {
    pub corpus_id: String,
    pub seed_count: usize,
    pub mean: f64,
    pub left_arm_win_count: usize,
    pub right_arm_win_count: usize,
    pub tie_count: usize,
    pub values: Vec<ZSpacePolaritySeedContrast>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpacePolarityContrastSummary {
    pub left_arm: &'static str,
    pub right_arm: &'static str,
    pub lower_is_better: bool,
    pub observation_count: usize,
    pub corpus_count: usize,
    pub seed_count_per_corpus: usize,
    pub pooled_seed_mean: f64,
    pub corpus_equal_weight_mean: f64,
    pub corpus_mean_minimum: f64,
    pub corpus_mean_maximum: f64,
    pub corpus_mean_population_standard_deviation: f64,
    pub seed_left_arm_win_count: usize,
    pub seed_right_arm_win_count: usize,
    pub seed_tie_count: usize,
    pub corpus_left_arm_win_count: usize,
    pub corpus_right_arm_win_count: usize,
    pub corpus_tie_count: usize,
    pub bounded_trend_ready: bool,
    pub bounded_trend_direction: ZSpacePolarityEvidenceDirection,
    pub corpora: Vec<ZSpacePolarityCorpusContrastSummary>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpacePolarityEvidenceReport {
    pub contract_version: &'static str,
    pub kind: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub evidence_validated: bool,
    pub evidence_id: String,
    pub status: &'static str,
    pub request: ZSpacePolarityEvidenceRequest,
    pub aggregation_rule: &'static str,
    pub contrast_rule: &'static str,
    pub corpus_count: usize,
    pub seed_count_per_corpus: usize,
    pub observation_count: usize,
    pub corpus_ids: Vec<String>,
    pub seeds: Vec<u64>,
    pub evidence_scope: &'static str,
    pub contrasts: BTreeMap<String, ZSpacePolarityContrastSummary>,
    pub bounded_polarity_improvement_observed: bool,
    pub bounded_baseline_improvement_observed: bool,
    pub efficacy_claim_ready: bool,
    pub evidence_boundary: &'static str,
    pub efficacy_claim_requirements: &'static str,
}

#[derive(Debug, Error, PartialEq)]
pub enum ZSpacePolarityEvidenceError {
    #[error("polarity evidence rows must not be empty")]
    EmptyRows,
    #[error("polarity evidence row count {count} exceeds maximum {maximum}")]
    RowLimit { count: usize, maximum: usize },
    #[error("{field} must be a lowercase sha256 identity")]
    InvalidIdentity { field: String },
    #[error("row {index} seed {seed} exceeds the cross-client maximum {maximum}")]
    SeedLimit {
        index: usize,
        seed: u64,
        maximum: u64,
    },
    #[error("row {index} field {field} must be finite, got {value}")]
    NonFiniteContrast {
        index: usize,
        field: &'static str,
        value: f64,
    },
    #[error("aggregate field {field} must be finite, got {value}")]
    NonFiniteAggregate { field: String, value: f64 },
    #[error(
        "row {index} violates polarity contrast identity: observed {observed}, expected {expected}, tolerance {tolerance}"
    )]
    ContrastInvariant {
        index: usize,
        observed: f64,
        expected: f64,
        tolerance: f64,
    },
    #[error("duplicate polarity evidence row for corpus {corpus_id} seed {seed}")]
    DuplicateRow { corpus_id: String, seed: u64 },
    #[error("corpus {corpus_id} has seed set {actual:?}, expected {expected:?}")]
    UnbalancedSeeds {
        corpus_id: String,
        actual: Vec<u64>,
        expected: Vec<u64>,
    },
    #[error("malformed polarity evidence report: {message}")]
    MalformedReport { message: String },
}

pub fn summarize_zspace_polarity_evidence(
    mut request: ZSpacePolarityEvidenceRequest,
) -> Result<ZSpacePolarityEvidenceReport, ZSpacePolarityEvidenceError> {
    validate_request_identities(&request)?;
    if request.rows.is_empty() {
        return Err(ZSpacePolarityEvidenceError::EmptyRows);
    }
    if request.rows.len() > ZSPACE_POLARITY_EVIDENCE_MAX_ROWS {
        return Err(ZSpacePolarityEvidenceError::RowLimit {
            count: request.rows.len(),
            maximum: ZSPACE_POLARITY_EVIDENCE_MAX_ROWS,
        });
    }
    validate_rows(&mut request.rows)?;
    request.rows.sort_by(|left, right| {
        (left.corpus_id.as_str(), left.seed).cmp(&(right.corpus_id.as_str(), right.seed))
    });

    let mut seeds_by_corpus: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
    let mut seen = BTreeSet::new();
    for row in &request.rows {
        if !seen.insert((row.corpus_id.clone(), row.seed)) {
            return Err(ZSpacePolarityEvidenceError::DuplicateRow {
                corpus_id: row.corpus_id.clone(),
                seed: row.seed,
            });
        }
        seeds_by_corpus
            .entry(row.corpus_id.clone())
            .or_default()
            .insert(row.seed);
    }
    let expected_seeds = seeds_by_corpus
        .values()
        .next()
        .expect("non-empty rows produce at least one corpus")
        .iter()
        .copied()
        .collect::<Vec<_>>();
    for (corpus_id, seeds) in &seeds_by_corpus {
        let actual = seeds.iter().copied().collect::<Vec<_>>();
        if actual != expected_seeds {
            return Err(ZSpacePolarityEvidenceError::UnbalancedSeeds {
                corpus_id: corpus_id.clone(),
                actual,
                expected: expected_seeds.clone(),
            });
        }
    }

    let corpus_count = seeds_by_corpus.len();
    let seed_count_per_corpus = expected_seeds.len();
    let bounded_ready = corpus_count >= ZSPACE_POLARITY_EVIDENCE_MIN_BOUNDED_CORPORA
        && seed_count_per_corpus >= ZSPACE_POLARITY_EVIDENCE_MIN_BOUNDED_SEEDS;
    let mut contrasts = BTreeMap::new();
    contrasts.insert(
        "dose_normalized_shape_effect".to_owned(),
        contrast_summary(
            &request.rows,
            "dose_normalized",
            "observe",
            bounded_ready,
            |row| row.dose_normalized_shape_effect,
        )?,
    );
    contrasts.insert(
        "complement_shape_effect".to_owned(),
        contrast_summary(
            &request.rows,
            "dose_preserving_complement",
            "observe",
            bounded_ready,
            |row| row.complement_shape_effect,
        )?,
    );
    contrasts.insert(
        "polarity_effect".to_owned(),
        contrast_summary(
            &request.rows,
            "dose_preserving_complement",
            "dose_normalized",
            bounded_ready,
            |row| row.polarity_effect,
        )?,
    );
    let polarity = contrasts
        .get("polarity_effect")
        .expect("canonical polarity contrast exists");
    let baseline = contrasts
        .get("complement_shape_effect")
        .expect("canonical baseline contrast exists");
    let evidence_id = polarity_evidence_id(&request)?;

    Ok(ZSpacePolarityEvidenceReport {
        contract_version: ZSPACE_POLARITY_EVIDENCE_CONTRACT_VERSION,
        kind: ZSPACE_POLARITY_EVIDENCE_KIND,
        semantic_owner: ZSPACE_POLARITY_EVIDENCE_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_POLARITY_EVIDENCE_SEMANTIC_BACKEND,
        evidence_validated: true,
        evidence_id,
        status: "ready",
        corpus_count,
        seed_count_per_corpus,
        observation_count: request.rows.len(),
        corpus_ids: seeds_by_corpus.keys().cloned().collect(),
        seeds: expected_seeds,
        evidence_scope: if bounded_ready {
            "multi_corpus_multi_seed_dose_matched_polarity_ablation"
        } else {
            "cross_corpus_polarity_diagnostic"
        },
        bounded_polarity_improvement_observed: bounded_ready
            && polarity.bounded_trend_direction
                == ZSpacePolarityEvidenceDirection::LeftArmBetter,
        bounded_baseline_improvement_observed: bounded_ready
            && baseline.bounded_trend_direction
                == ZSpacePolarityEvidenceDirection::LeftArmBetter,
        efficacy_claim_ready: false,
        evidence_boundary: "balanced multi-corpus matched contrasts support only a bounded corpus-level trend; corpus means are equally weighted and do not establish statistical significance or general model superiority",
        efficacy_claim_requirements: "a prespecified, adequately powered multi-model evaluation with independent held-out quality and stability metrics remains required",
        aggregation_rule: ZSPACE_POLARITY_EVIDENCE_AGGREGATION_RULE,
        contrast_rule: ZSPACE_POLARITY_EVIDENCE_CONTRAST_RULE,
        contrasts,
        request,
    })
}

pub fn validate_zspace_polarity_evidence_value(
    report: serde_json::Value,
) -> Result<ZSpacePolarityEvidenceReport, ZSpacePolarityEvidenceError> {
    let request_value = report.get("request").cloned().ok_or_else(|| {
        ZSpacePolarityEvidenceError::MalformedReport {
            message: "missing request".to_owned(),
        }
    })?;
    let request = serde_json::from_value(request_value).map_err(|error| {
        ZSpacePolarityEvidenceError::MalformedReport {
            message: error.to_string(),
        }
    })?;
    let canonical = summarize_zspace_polarity_evidence(request)?;
    let canonical_value = serde_json::to_value(&canonical).map_err(|error| {
        ZSpacePolarityEvidenceError::MalformedReport {
            message: error.to_string(),
        }
    })?;
    if report != canonical_value {
        return Err(ZSpacePolarityEvidenceError::MalformedReport {
            message: "report does not match the canonical Rust evidence aggregate".to_owned(),
        });
    }
    Ok(canonical)
}

fn validate_request_identities(
    request: &ZSpacePolarityEvidenceRequest,
) -> Result<(), ZSpacePolarityEvidenceError> {
    for (field, value) in [
        ("protocol_id", request.protocol_id.as_str()),
        ("runtime_identity_id", request.runtime_identity_id.as_str()),
        ("trajectory_id", request.trajectory_id.as_str()),
        (
            "trajectory_policy_id",
            request.trajectory_policy_id.as_str(),
        ),
        ("control_sequence_id", request.control_sequence_id.as_str()),
        (
            "nominal_schedule_sequence_id",
            request.nominal_schedule_sequence_id.as_str(),
        ),
    ] {
        require_sha256_id(field, value)?;
    }
    Ok(())
}

fn validate_rows(
    rows: &mut [ZSpacePolarityEvidenceRow],
) -> Result<(), ZSpacePolarityEvidenceError> {
    for (index, row) in rows.iter_mut().enumerate() {
        require_sha256_id(&format!("rows[{index}].corpus_id"), &row.corpus_id)?;
        if row.seed > ZSPACE_POLARITY_EVIDENCE_MAX_SAFE_SEED {
            return Err(ZSpacePolarityEvidenceError::SeedLimit {
                index,
                seed: row.seed,
                maximum: ZSPACE_POLARITY_EVIDENCE_MAX_SAFE_SEED,
            });
        }
        for (field, value) in [
            (
                "dose_normalized_shape_effect",
                row.dose_normalized_shape_effect,
            ),
            ("complement_shape_effect", row.complement_shape_effect),
            ("polarity_effect", row.polarity_effect),
        ] {
            if !value.is_finite() {
                return Err(ZSpacePolarityEvidenceError::NonFiniteContrast {
                    index,
                    field,
                    value,
                });
            }
        }
        let expected = row.complement_shape_effect - row.dose_normalized_shape_effect;
        if !expected.is_finite() {
            return Err(ZSpacePolarityEvidenceError::NonFiniteContrast {
                index,
                field: "derived_polarity_effect",
                value: expected,
            });
        }
        let scale = row.polarity_effect.abs().max(expected.abs());
        let tolerance = CONTRAST_ABSOLUTE_TOLERANCE.max(CONTRAST_RELATIVE_TOLERANCE * scale);
        let observed_direction = row.polarity_effect.partial_cmp(&0.0);
        let expected_direction = expected.partial_cmp(&0.0);
        if observed_direction != expected_direction
            || (row.polarity_effect - expected).abs() > tolerance
        {
            return Err(ZSpacePolarityEvidenceError::ContrastInvariant {
                index,
                observed: row.polarity_effect,
                expected,
                tolerance,
            });
        }
        row.polarity_effect = expected;
    }
    Ok(())
}

fn require_sha256_id(field: &str, value: &str) -> Result<(), ZSpacePolarityEvidenceError> {
    let valid = value.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    });
    if valid {
        Ok(())
    } else {
        Err(ZSpacePolarityEvidenceError::InvalidIdentity {
            field: field.to_owned(),
        })
    }
}

fn contrast_summary<F>(
    rows: &[ZSpacePolarityEvidenceRow],
    left_arm: &'static str,
    right_arm: &'static str,
    bounded_ready: bool,
    value: F,
) -> Result<ZSpacePolarityContrastSummary, ZSpacePolarityEvidenceError>
where
    F: Fn(&ZSpacePolarityEvidenceRow) -> f64,
{
    let seed_values = rows.iter().map(&value).collect::<Vec<_>>();
    let mut values_by_corpus: BTreeMap<&str, Vec<ZSpacePolaritySeedContrast>> = BTreeMap::new();
    for (row, contrast) in rows.iter().zip(&seed_values) {
        values_by_corpus
            .entry(&row.corpus_id)
            .or_default()
            .push(ZSpacePolaritySeedContrast {
                seed: row.seed,
                value: *contrast,
            });
    }
    let mut corpora = Vec::with_capacity(values_by_corpus.len());
    for (corpus_id, mut values) in values_by_corpus {
        values.sort_by_key(|item| item.seed);
        let corpus_values = values.iter().map(|item| item.value).collect::<Vec<_>>();
        let mean = finite_mean(
            &corpus_values,
            &format!("{left_arm}_vs_{right_arm}.corpus[{corpus_id}].mean"),
        )?;
        corpora.push(ZSpacePolarityCorpusContrastSummary {
            corpus_id: corpus_id.to_owned(),
            seed_count: values.len(),
            mean,
            left_arm_win_count: values.iter().filter(|item| item.value < 0.0).count(),
            right_arm_win_count: values.iter().filter(|item| item.value > 0.0).count(),
            tie_count: values.iter().filter(|item| item.value == 0.0).count(),
            values,
        });
    }
    let corpus_means = corpora.iter().map(|corpus| corpus.mean).collect::<Vec<_>>();
    let pooled_seed_mean = finite_mean(
        &seed_values,
        &format!("{left_arm}_vs_{right_arm}.pooled_seed_mean"),
    )?;
    let corpus_equal_weight_mean = finite_mean(
        &corpus_means,
        &format!("{left_arm}_vs_{right_arm}.corpus_equal_weight_mean"),
    )?;
    let corpus_mean_population_standard_deviation = finite_population_standard_deviation(
        &corpus_means,
        corpus_equal_weight_mean,
        &format!("{left_arm}_vs_{right_arm}.corpus_mean_population_standard_deviation"),
    )?;
    let direction = evidence_direction(&corpus_means);
    let bounded_trend_ready = bounded_ready && direction != ZSpacePolarityEvidenceDirection::Mixed;

    Ok(ZSpacePolarityContrastSummary {
        left_arm,
        right_arm,
        lower_is_better: true,
        observation_count: rows.len(),
        corpus_count: corpora.len(),
        seed_count_per_corpus: corpora[0].seed_count,
        pooled_seed_mean,
        corpus_equal_weight_mean,
        corpus_mean_minimum: corpus_means.iter().copied().fold(f64::INFINITY, f64::min),
        corpus_mean_maximum: corpus_means
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
        corpus_mean_population_standard_deviation,
        seed_left_arm_win_count: seed_values.iter().filter(|value| **value < 0.0).count(),
        seed_right_arm_win_count: seed_values.iter().filter(|value| **value > 0.0).count(),
        seed_tie_count: seed_values.iter().filter(|value| **value == 0.0).count(),
        corpus_left_arm_win_count: corpus_means.iter().filter(|value| **value < 0.0).count(),
        corpus_right_arm_win_count: corpus_means.iter().filter(|value| **value > 0.0).count(),
        corpus_tie_count: corpus_means.iter().filter(|value| **value == 0.0).count(),
        bounded_trend_ready,
        bounded_trend_direction: direction,
        corpora,
    })
}

fn finite_mean(values: &[f64], field: &str) -> Result<f64, ZSpacePolarityEvidenceError> {
    let count = values.len() as f64;
    if values.iter().all(|value| *value == values[0]) {
        return Ok(values[0]);
    }
    // Power-of-two scaling is exact for normal operands and bounds every prefix.
    let divisor = values.len().next_power_of_two() as f64;
    let mut scaled_values = Vec::with_capacity(values.len());
    let mut division_underflows = Vec::new();
    for value in values {
        let scaled = value / divisor;
        if *value != 0.0 && !scaled.is_normal() {
            division_underflows.push(*value);
        } else {
            scaled_values.push(scaled);
        }
    }
    let scaled_mean = expansion_sum(scaled_values.iter().copied()) * (divisor / count);
    let underflow_mean = expansion_sum(division_underflows.iter().copied()) / count;
    let mean = refine_scaled_mean(
        &scaled_values,
        &division_underflows,
        divisor,
        values.len(),
        expansion_sum([scaled_mean, underflow_mean]),
    );
    if mean.is_finite() {
        Ok(mean)
    } else {
        Err(ZSpacePolarityEvidenceError::NonFiniteAggregate {
            field: field.to_owned(),
            value: mean,
        })
    }
}

fn finite_population_standard_deviation(
    values: &[f64],
    mean: f64,
    field: &str,
) -> Result<f64, ZSpacePolarityEvidenceError> {
    let squared_deviation_sum = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>();
    let standard_deviation = if squared_deviation_sum.is_finite() {
        (squared_deviation_sum / values.len() as f64).sqrt()
    } else {
        let scale = values
            .iter()
            .map(|value| value.abs())
            .fold(mean.abs(), f64::max);
        if scale == 0.0 {
            0.0
        } else {
            let normalized_mean = mean / scale;
            let normalized_variance = compensated_sum(values.iter().map(|value| {
                let deviation = value / scale - normalized_mean;
                deviation * deviation
            })) / values.len() as f64;
            scale * normalized_variance.clamp(0.0, 1.0).sqrt()
        }
    };
    if standard_deviation.is_finite() {
        Ok(standard_deviation)
    } else {
        Err(ZSpacePolarityEvidenceError::NonFiniteAggregate {
            field: field.to_owned(),
            value: standard_deviation,
        })
    }
}

fn compensated_sum<I>(values: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    let mut sum = 0.0;
    let mut compensation = 0.0;
    for value in values {
        let updated = sum + value;
        if sum.abs() >= value.abs() {
            compensation += (sum - updated) + value;
        } else {
            compensation += (value - updated) + sum;
        }
        sum = updated;
    }
    sum + compensation
}

fn expansion_sum<I>(values: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    // Keep a non-overlapping expansion, then apply the final half-even correction.
    let mut partials = Vec::<f64>::new();
    for mut value in values {
        let partial_count = partials.len();
        let mut retained = 0;
        for index in 0..partial_count {
            let mut partial = partials[index];
            if value.abs() < partial.abs() {
                std::mem::swap(&mut value, &mut partial);
            }
            let high = value + partial;
            let low = partial - (high - value);
            if low != 0.0 {
                partials[retained] = low;
                retained += 1;
            }
            value = high;
        }
        partials.truncate(retained);
        if value != 0.0 {
            partials.push(value);
        }
    }

    let Some(mut high) = partials.pop() else {
        return 0.0;
    };
    let mut low = 0.0;
    while let Some(partial) = partials.pop() {
        let previous = high;
        high += partial;
        low = partial - (high - previous);
        if low != 0.0 {
            break;
        }
    }
    if partials
        .last()
        .is_some_and(|partial| (low < 0.0 && *partial < 0.0) || (low > 0.0 && *partial > 0.0))
    {
        let doubled = low * 2.0;
        let rounded = high + doubled;
        if doubled == rounded - high {
            high = rounded;
        }
    }
    high
}

fn refine_scaled_mean(
    scaled_values: &[f64],
    division_underflows: &[f64],
    divisor: f64,
    count: usize,
    mut mean: f64,
) -> f64 {
    let count_f64 = count as f64;
    for _ in 0..2 {
        let scaled_mean_term = mean / divisor;
        if mean != 0.0 && !scaled_mean_term.is_normal() {
            break;
        }
        // Interleave each observation with the candidate contribution so residual
        // accumulation cannot recreate the large same-sign prefix we scaled away.
        let residual_terms = scaled_values
            .iter()
            .flat_map(|value| [*value, -scaled_mean_term])
            .chain(std::iter::repeat_n(
                -scaled_mean_term,
                count - scaled_values.len(),
            ));
        let scaled_residual = expansion_sum(residual_terms) * (divisor / count_f64);
        let underflow_residual = expansion_sum(division_underflows.iter().copied()) / count_f64;
        let correction = expansion_sum([scaled_residual, underflow_residual]);
        let refined = expansion_sum([mean, correction]);
        if refined == mean {
            break;
        }
        mean = refined;
    }
    mean
}

fn evidence_direction(values: &[f64]) -> ZSpacePolarityEvidenceDirection {
    if values.iter().all(|value| *value < 0.0) {
        ZSpacePolarityEvidenceDirection::LeftArmBetter
    } else if values.iter().all(|value| *value > 0.0) {
        ZSpacePolarityEvidenceDirection::RightArmBetter
    } else if values.iter().all(|value| *value == 0.0) {
        ZSpacePolarityEvidenceDirection::NoObservedDifference
    } else {
        ZSpacePolarityEvidenceDirection::Mixed
    }
}

fn polarity_evidence_id(
    request: &ZSpacePolarityEvidenceRequest,
) -> Result<String, ZSpacePolarityEvidenceError> {
    let encoded = serde_json::to_vec(&(ZSPACE_POLARITY_EVIDENCE_CONTRACT_VERSION, request))
        .map_err(|error| ZSpacePolarityEvidenceError::MalformedReport {
            message: error.to_string(),
        })?;
    let digest = Sha256::digest(encoded);
    let mut hex = String::with_capacity(64);
    for byte in digest {
        write!(&mut hex, "{byte:02x}").map_err(|error| {
            ZSpacePolarityEvidenceError::MalformedReport {
                message: error.to_string(),
            }
        })?;
    }
    Ok(format!("sha256:{hex}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn id(value: char) -> String {
        format!("sha256:{}", value.to_string().repeat(64))
    }

    fn request(corpus_count: usize, seeds: &[u64]) -> ZSpacePolarityEvidenceRequest {
        let mut rows = Vec::new();
        for corpus in 0..corpus_count {
            for (index, seed) in seeds.iter().copied().enumerate() {
                let normalized = 0.002 + corpus as f64 * 0.0001 + index as f64 * 0.00001;
                let complement = -0.001 - corpus as f64 * 0.0001 - index as f64 * 0.00001;
                rows.push(ZSpacePolarityEvidenceRow {
                    corpus_id: id(char::from(b'a' + corpus as u8)),
                    seed,
                    dose_normalized_shape_effect: normalized,
                    complement_shape_effect: complement,
                    polarity_effect: complement - normalized,
                });
            }
        }
        rows.reverse();
        ZSpacePolarityEvidenceRequest {
            protocol_id: id('1'),
            runtime_identity_id: id('2'),
            trajectory_id: id('3'),
            trajectory_policy_id: id('4'),
            control_sequence_id: id('5'),
            nominal_schedule_sequence_id: id('6'),
            rows,
        }
    }

    #[test]
    fn balanced_three_by_three_evidence_uses_corpus_level_direction() {
        let report = summarize_zspace_polarity_evidence(request(3, &[13, 17, 23]))
            .expect("balanced evidence");

        assert!(report.evidence_validated);
        assert_eq!(report.corpus_count, 3);
        assert_eq!(report.seed_count_per_corpus, 3);
        assert_eq!(report.observation_count, 9);
        assert_eq!(
            report.evidence_scope,
            "multi_corpus_multi_seed_dose_matched_polarity_ablation"
        );
        assert!(report.bounded_polarity_improvement_observed);
        assert!(report.bounded_baseline_improvement_observed);
        assert!(!report.efficacy_claim_ready);
        let polarity = &report.contrasts["polarity_effect"];
        assert!(polarity.bounded_trend_ready);
        assert_eq!(
            polarity.bounded_trend_direction,
            ZSpacePolarityEvidenceDirection::LeftArmBetter
        );
        assert_eq!(polarity.corpus_left_arm_win_count, 3);
        assert_eq!(polarity.seed_left_arm_win_count, 9);
        assert!(report.evidence_id.starts_with("sha256:"));
        assert!(report.request.rows.windows(2).all(|rows| (
            rows[0].corpus_id.as_str(),
            rows[0].seed
        ) < (
            rows[1].corpus_id.as_str(),
            rows[1].seed
        )));
    }

    #[test]
    fn one_or_two_corpora_remain_diagnostic() {
        let report = summarize_zspace_polarity_evidence(request(2, &[13, 17, 23]))
            .expect("diagnostic evidence");

        assert_eq!(report.evidence_scope, "cross_corpus_polarity_diagnostic");
        assert!(!report.contrasts["polarity_effect"].bounded_trend_ready);
        assert!(!report.bounded_polarity_improvement_observed);
    }

    #[test]
    fn mixed_corpus_directions_are_not_trend_ready() {
        let mut request = request(3, &[13, 17, 23]);
        let mixed_corpus = id('c');
        for row in request
            .rows
            .iter_mut()
            .filter(|row| row.corpus_id == mixed_corpus)
        {
            row.complement_shape_effect = 0.004;
            row.polarity_effect = row.complement_shape_effect - row.dose_normalized_shape_effect;
        }

        let report = summarize_zspace_polarity_evidence(request).expect("mixed evidence");
        let polarity = &report.contrasts["polarity_effect"];

        assert_eq!(
            polarity.bounded_trend_direction,
            ZSpacePolarityEvidenceDirection::Mixed
        );
        assert!(!polarity.bounded_trend_ready);
        assert!(!report.bounded_polarity_improvement_observed);
    }

    #[test]
    fn duplicate_and_unbalanced_rows_fail_closed() {
        let mut duplicate = request(2, &[13, 17, 23]);
        duplicate.rows.push(duplicate.rows[0].clone());
        assert!(matches!(
            summarize_zspace_polarity_evidence(duplicate),
            Err(ZSpacePolarityEvidenceError::DuplicateRow { .. })
        ));

        let mut unbalanced = request(2, &[13, 17, 23]);
        unbalanced.rows.pop();
        assert!(matches!(
            summarize_zspace_polarity_evidence(unbalanced),
            Err(ZSpacePolarityEvidenceError::UnbalancedSeeds { .. })
        ));
    }

    #[test]
    fn contrast_algebra_is_an_enforced_invariant() {
        let mut invalid = request(1, &[13]);
        invalid.rows[0].polarity_effect = 0.0;

        assert!(matches!(
            summarize_zspace_polarity_evidence(invalid),
            Err(ZSpacePolarityEvidenceError::ContrastInvariant { .. })
        ));
    }

    #[test]
    fn tiny_contrast_cannot_reverse_the_validated_direction() {
        let mut invalid = request(1, &[13]);
        invalid.rows[0].dose_normalized_shape_effect = 4.0e-13;
        invalid.rows[0].complement_shape_effect = -4.0e-13;
        invalid.rows[0].polarity_effect = 1.0e-13;

        assert!(matches!(
            summarize_zspace_polarity_evidence(invalid),
            Err(ZSpacePolarityEvidenceError::ContrastInvariant { .. })
        ));
    }

    #[test]
    fn absolute_tolerance_floor_cannot_reverse_the_validated_direction() {
        let mut invalid = request(3, &[13, 17, 23]);
        for row in &mut invalid.rows {
            row.dose_normalized_shape_effect = 4.0e-16;
            row.complement_shape_effect = -4.0e-16;
            row.polarity_effect = 1.0e-16;
        }

        assert!(matches!(
            summarize_zspace_polarity_evidence(invalid),
            Err(ZSpacePolarityEvidenceError::ContrastInvariant { .. })
        ));
    }

    #[test]
    fn absolute_tolerance_floor_cannot_create_a_direction_from_zero() {
        let mut invalid = request(1, &[13]);
        invalid.rows[0].dose_normalized_shape_effect = 4.0e-16;
        invalid.rows[0].complement_shape_effect = 4.0e-16;
        invalid.rows[0].polarity_effect = 1.0e-16;

        assert!(matches!(
            summarize_zspace_polarity_evidence(invalid),
            Err(ZSpacePolarityEvidenceError::ContrastInvariant { .. })
        ));
    }

    #[test]
    fn large_finite_contrasts_use_overflow_safe_aggregates() {
        let mut request = request(3, &[13, 17, 23]);
        for row in &mut request.rows {
            row.dose_normalized_shape_effect = 5.0e307;
            row.complement_shape_effect = -5.0e307;
            row.polarity_effect = -1.0e308;
        }

        let report = summarize_zspace_polarity_evidence(request).expect("finite evidence");
        for summary in report.contrasts.values() {
            assert!(summary.pooled_seed_mean.is_finite());
            assert!(summary.corpus_equal_weight_mean.is_finite());
            assert!(summary.corpus_mean_minimum.is_finite());
            assert!(summary.corpus_mean_maximum.is_finite());
            assert!(summary
                .corpus_mean_population_standard_deviation
                .is_finite());
            assert!(summary.corpora.iter().all(|corpus| corpus.mean.is_finite()));
        }
        assert_eq!(
            report.contrasts["dose_normalized_shape_effect"].corpus_equal_weight_mean,
            5.0e307
        );
        assert_eq!(
            report.contrasts["polarity_effect"].corpus_equal_weight_mean,
            -1.0e308
        );
        serde_json::to_value(report).expect("finite evidence serializes as JSON numbers");
    }

    #[test]
    fn finite_means_preserve_large_cancellation_residuals() {
        let mut request = request(3, &[13, 17, 23]);
        for row in &mut request.rows {
            let value = match row.seed {
                13 => -1.0e308,
                17 => 1.0e-16,
                23 => 1.0e308,
                _ => unreachable!("fixture contains only the requested seeds"),
            };
            row.dose_normalized_shape_effect = value;
            row.complement_shape_effect = value;
            row.polarity_effect = 0.0;
        }

        let report = summarize_zspace_polarity_evidence(request).expect("finite evidence");
        let normalized = &report.contrasts["dose_normalized_shape_effect"];
        assert!((normalized.corpus_equal_weight_mean - 1.0e-16 / 3.0).abs() < 1.0e-30);
        assert_eq!(normalized.corpus_right_arm_win_count, 3);
        assert_eq!(
            normalized.bounded_trend_direction,
            ZSpacePolarityEvidenceDirection::RightArmBetter
        );
    }

    #[test]
    fn finite_means_preserve_representable_extremes() {
        let smallest_subnormal = f64::from_bits(1);
        let next_down_maximum = f64::from_bits(f64::MAX.to_bits() - 1);
        assert_eq!(
            finite_mean(
                &[
                    smallest_subnormal,
                    smallest_subnormal * 2.0,
                    smallest_subnormal * 3.0,
                ],
                "smallest_subnormal",
            )
            .expect("subnormal mean"),
            smallest_subnormal * 2.0
        );
        assert_eq!(
            finite_mean(&[f64::MAX; 3], "largest_finite").expect("maximum finite mean"),
            f64::MAX
        );
        assert_eq!(
            finite_mean(
                &[f64::MAX, f64::MAX, next_down_maximum],
                "mixed_largest_finite",
            )
            .expect("mixed maximum finite mean"),
            f64::MAX
        );
    }

    #[test]
    fn finite_means_preserve_subnormal_residuals_after_cancellation() {
        let smallest_subnormal = f64::from_bits(1);
        let mut values = vec![
            f64::MAX,
            f64::MAX,
            -f64::MAX,
            -f64::MAX,
            smallest_subnormal * 9.0,
            smallest_subnormal * -4.0,
        ];

        let mean = finite_mean(&values, "subnormal_residual").expect("subnormal residual mean");
        assert_eq!(mean, smallest_subnormal);

        values.reverse();
        assert_eq!(
            finite_mean(&values, "reversed_subnormal_residual")
                .expect("reversed subnormal residual mean"),
            mean
        );
    }

    #[test]
    fn finite_means_preserve_ulp_residuals_after_prefix_overflow() {
        let next_down_maximum = f64::from_bits(f64::MAX.to_bits() - 1);
        let mut values = vec![f64::MAX; 11];
        values.extend(std::iter::repeat_n(-f64::MAX, 10));
        values.extend([-next_down_maximum, -1.0e290]);

        let mean = finite_mean(&values, "ulp_residual").expect("finite residual mean");
        let expected = ((f64::MAX - next_down_maximum) - 1.0e290) / values.len() as f64;
        assert!(mean > 0.0);
        assert!((mean / expected - 1.0).abs() < 1.0e-14);

        values.reverse();
        let reversed =
            finite_mean(&values, "reversed_ulp_residual").expect("reversed finite residual mean");
        assert_eq!(reversed, mean);
    }

    #[test]
    fn tolerated_row_error_is_canonicalized_before_aggregation() {
        let mut request = request(3, &[13, 17, 23]);
        for row in &mut request.rows {
            let (expected, supplied) = match row.seed {
                13 => (4.0e-16, 1.0e-15),
                17 | 23 => (-4.0e-16, -1.0e-17),
                _ => unreachable!("fixture contains only the requested seeds"),
            };
            row.dose_normalized_shape_effect = 0.0;
            row.complement_shape_effect = expected;
            row.polarity_effect = supplied;
        }

        let report = summarize_zspace_polarity_evidence(request).expect("canonical evidence");
        let polarity = &report.contrasts["polarity_effect"];
        assert_eq!(
            polarity.bounded_trend_direction,
            ZSpacePolarityEvidenceDirection::LeftArmBetter
        );
        assert_eq!(polarity.corpus_left_arm_win_count, 3);
        assert!(report.request.rows.iter().all(|row| row.polarity_effect
            == row.complement_shape_effect - row.dose_normalized_shape_effect));
    }

    #[test]
    fn derived_contrast_overflow_fails_closed() {
        let mut invalid = request(1, &[13]);
        invalid.rows[0].dose_normalized_shape_effect = f64::MAX;
        invalid.rows[0].complement_shape_effect = -f64::MAX;
        invalid.rows[0].polarity_effect = -f64::MAX;

        assert!(matches!(
            summarize_zspace_polarity_evidence(invalid),
            Err(ZSpacePolarityEvidenceError::NonFiniteContrast {
                field: "derived_polarity_effect",
                ..
            })
        ));
    }

    #[test]
    fn serialized_evidence_is_recomputed_and_tampering_is_rejected() {
        let report = summarize_zspace_polarity_evidence(request(3, &[13, 17, 23]))
            .expect("canonical evidence");
        let encoded = serde_json::to_value(&report).expect("serialized evidence");
        assert_eq!(
            validate_zspace_polarity_evidence_value(encoded.clone()).expect("validated evidence"),
            report
        );

        let mut tampered = encoded;
        tampered["contrasts"]["polarity_effect"]["corpus_equal_weight_mean"] = json!(0.0);
        assert!(matches!(
            validate_zspace_polarity_evidence_value(tampered),
            Err(ZSpacePolarityEvidenceError::MalformedReport { .. })
        ));
    }
}
