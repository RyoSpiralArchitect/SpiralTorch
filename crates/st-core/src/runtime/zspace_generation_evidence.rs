// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical token-sequence evidence for held-out Z-space generation probes.
//!
//! Model clients own inference and tokenization. This module owns the bounded,
//! tokenizer-agnostic repetition and diversity semantics applied to continuation
//! token IDs, together with canonical ordering, identity, and validation.

use super::zspace_periodicity::{
    longest_periodic_suffix as shared_longest_periodic_suffix, PeriodicSuffix,
    ZSPACE_PERIODIC_SUFFIX_MAX_PERIOD, ZSPACE_PERIODIC_SUFFIX_MIN_REPETITIONS,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use thiserror::Error;

pub const ZSPACE_GENERATION_EVIDENCE_CONTRACT_VERSION: &str =
    "spiraltorch.zspace_generation_evidence.v1";
pub const ZSPACE_GENERATION_EVIDENCE_KIND: &str = "spiraltorch.zspace_generation_evidence";
pub const ZSPACE_GENERATION_EVIDENCE_SEMANTIC_OWNER: &str =
    "st-core::runtime::zspace_generation_evidence";
pub const ZSPACE_GENERATION_EVIDENCE_SEMANTIC_BACKEND: &str = "rust";
pub const ZSPACE_GENERATION_EVIDENCE_METRIC_RULE: &str =
    "continuation token IDs only; sample-local n-gram distinct/repeated-occurrence ratios for orders 1..4; adjacent equal-token ratio; longest trailing periodic suffix with period<=16 and >=3 repetitions";
pub const ZSPACE_GENERATION_EVIDENCE_LOOP_SCORE_RULE: &str =
    "loop_score=trigram_repetition_ratio+consecutive_repetition_ratio+periodic_suffix_repeated_token_ratio; unavailable ratios contribute zero";
pub const ZSPACE_GENERATION_EVIDENCE_MAX_SAMPLES: usize = 10_000;
pub const ZSPACE_GENERATION_EVIDENCE_MAX_TOKENS_PER_SAMPLE: usize = 16_384;
pub const ZSPACE_GENERATION_EVIDENCE_MAX_TOTAL_TOKENS: usize = 1_000_000;
pub const ZSPACE_GENERATION_EVIDENCE_MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
pub const ZSPACE_GENERATION_EVIDENCE_NGRAM_ORDERS: [usize; 4] = [1, 2, 3, 4];
pub const ZSPACE_GENERATION_EVIDENCE_PERIODIC_SUFFIX_MAX_PERIOD: usize =
    ZSPACE_PERIODIC_SUFFIX_MAX_PERIOD;
pub const ZSPACE_GENERATION_EVIDENCE_PERIODIC_SUFFIX_MIN_REPETITIONS: usize =
    ZSPACE_PERIODIC_SUFFIX_MIN_REPETITIONS;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceGenerationEvidenceSample {
    pub prompt_id: String,
    pub seed: u64,
    pub continuation_token_ids: Vec<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceGenerationEvidenceRequest {
    pub protocol_id: String,
    pub runtime_identity_id: String,
    pub model_artifact_id: String,
    pub prompt_set_id: String,
    pub decoding_config_id: String,
    pub samples: Vec<ZSpaceGenerationEvidenceSample>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceGenerationNgramEvidence {
    pub order: usize,
    pub possible_count: usize,
    pub unique_count: usize,
    pub repeated_occurrence_count: usize,
    pub maximum_occurrence_count: usize,
    pub distinct_ratio: Option<f64>,
    pub repetition_ratio: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceGenerationSampleEvidence {
    pub prompt_id: String,
    pub seed: u64,
    pub token_count: usize,
    pub empty_continuation: bool,
    pub adjacent_transition_count: usize,
    pub consecutive_repeated_token_count: usize,
    pub consecutive_repetition_ratio: Option<f64>,
    pub periodic_loop_detected: bool,
    pub periodic_suffix_period: Option<usize>,
    pub periodic_suffix_token_count: usize,
    pub periodic_suffix_repeated_token_count: usize,
    pub periodic_suffix_repetition_count: usize,
    pub periodic_suffix_repeated_token_ratio: Option<f64>,
    pub loop_score: f64,
    pub ngrams: Vec<ZSpaceGenerationNgramEvidence>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceGenerationAggregateEvidence {
    pub sample_count: usize,
    pub nonempty_sample_count: usize,
    pub empty_sample_count: usize,
    pub total_token_count: usize,
    pub minimum_token_count: usize,
    pub maximum_token_count: usize,
    pub adjacent_transition_count: usize,
    pub consecutive_repeated_token_count: usize,
    pub consecutive_repetition_ratio: Option<f64>,
    pub periodic_loop_sample_count: usize,
    pub periodic_loop_sample_ratio: f64,
    pub periodic_suffix_repeated_token_count: usize,
    pub periodic_suffix_repeated_token_ratio: Option<f64>,
    pub sample_mean_loop_score: f64,
    pub maximum_loop_score: f64,
    pub ngrams: Vec<ZSpaceGenerationNgramEvidence>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceGenerationEvidenceReport {
    pub contract_version: &'static str,
    pub kind: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub evidence_validated: bool,
    pub evidence_id: String,
    pub status: &'static str,
    pub request: ZSpaceGenerationEvidenceRequest,
    pub metric_rule: &'static str,
    pub loop_score_rule: &'static str,
    pub ngram_orders: [usize; 4],
    pub periodic_suffix_max_period: usize,
    pub periodic_suffix_min_repetitions: usize,
    pub sample_count: usize,
    pub evidence_scope: &'static str,
    pub samples: Vec<ZSpaceGenerationSampleEvidence>,
    pub aggregate: ZSpaceGenerationAggregateEvidence,
    pub efficacy_claim_ready: bool,
    pub evidence_boundary: &'static str,
    pub efficacy_claim_requirements: &'static str,
}

#[derive(Debug, Error, PartialEq)]
pub enum ZSpaceGenerationEvidenceError {
    #[error("generation evidence samples must not be empty")]
    EmptySamples,
    #[error("generation evidence sample count {count} exceeds maximum {maximum}")]
    SampleLimit { count: usize, maximum: usize },
    #[error("{field} must be a lowercase sha256 identity")]
    InvalidIdentity { field: String },
    #[error("sample {index} seed {seed} exceeds the cross-client maximum {maximum}")]
    SeedLimit {
        index: usize,
        seed: u64,
        maximum: u64,
    },
    #[error("sample {index} token count {count} exceeds per-sample maximum {maximum}")]
    SampleTokenLimit {
        index: usize,
        count: usize,
        maximum: usize,
    },
    #[error("generation evidence total token count {count} exceeds maximum {maximum}")]
    TotalTokenLimit { count: usize, maximum: usize },
    #[error(
        "sample {sample_index} token {token_index} ID {token_id} exceeds the cross-client maximum {maximum}"
    )]
    TokenIdLimit {
        sample_index: usize,
        token_index: usize,
        token_id: u64,
        maximum: u64,
    },
    #[error("duplicate generation evidence sample for prompt {prompt_id} seed {seed}")]
    DuplicateSample { prompt_id: String, seed: u64 },
    #[error("generation evidence token count overflow")]
    TokenCountOverflow,
    #[error("malformed generation evidence report: {message}")]
    MalformedReport { message: String },
}

pub fn summarize_zspace_generation_evidence(
    mut request: ZSpaceGenerationEvidenceRequest,
) -> Result<ZSpaceGenerationEvidenceReport, ZSpaceGenerationEvidenceError> {
    validate_request_identities(&request)?;
    if request.samples.is_empty() {
        return Err(ZSpaceGenerationEvidenceError::EmptySamples);
    }
    if request.samples.len() > ZSPACE_GENERATION_EVIDENCE_MAX_SAMPLES {
        return Err(ZSpaceGenerationEvidenceError::SampleLimit {
            count: request.samples.len(),
            maximum: ZSPACE_GENERATION_EVIDENCE_MAX_SAMPLES,
        });
    }
    validate_samples(&request.samples)?;
    request.samples.sort_by(|left, right| {
        (left.prompt_id.as_str(), left.seed).cmp(&(right.prompt_id.as_str(), right.seed))
    });
    reject_duplicate_samples(&request.samples)?;

    let samples = request
        .samples
        .iter()
        .map(summarize_sample)
        .collect::<Vec<_>>();
    let aggregate = summarize_aggregate(&samples);
    let evidence_id = generation_evidence_id(&request)?;

    Ok(ZSpaceGenerationEvidenceReport {
        contract_version: ZSPACE_GENERATION_EVIDENCE_CONTRACT_VERSION,
        kind: ZSPACE_GENERATION_EVIDENCE_KIND,
        semantic_owner: ZSPACE_GENERATION_EVIDENCE_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_GENERATION_EVIDENCE_SEMANTIC_BACKEND,
        evidence_validated: true,
        evidence_id,
        status: "ready",
        metric_rule: ZSPACE_GENERATION_EVIDENCE_METRIC_RULE,
        loop_score_rule: ZSPACE_GENERATION_EVIDENCE_LOOP_SCORE_RULE,
        ngram_orders: ZSPACE_GENERATION_EVIDENCE_NGRAM_ORDERS,
        periodic_suffix_max_period: ZSPACE_GENERATION_EVIDENCE_PERIODIC_SUFFIX_MAX_PERIOD,
        periodic_suffix_min_repetitions:
            ZSPACE_GENERATION_EVIDENCE_PERIODIC_SUFFIX_MIN_REPETITIONS,
        sample_count: samples.len(),
        evidence_scope: "held_out_generation_token_observation",
        samples,
        aggregate,
        efficacy_claim_ready: false,
        evidence_boundary: "token-sequence repetition and diversity indicators are independent held-out observations; they do not measure semantic quality or establish model or training superiority",
        efficacy_claim_requirements: "prespecified matched multi-model and multi-corpus comparisons with held-out loss, stability, and semantic-quality review remain required",
        request,
    })
}

pub fn validate_zspace_generation_evidence_value(
    report: serde_json::Value,
) -> Result<ZSpaceGenerationEvidenceReport, ZSpaceGenerationEvidenceError> {
    let request_value = report.get("request").cloned().ok_or_else(|| {
        ZSpaceGenerationEvidenceError::MalformedReport {
            message: "missing request".to_owned(),
        }
    })?;
    let request = serde_json::from_value(request_value).map_err(|error| {
        ZSpaceGenerationEvidenceError::MalformedReport {
            message: error.to_string(),
        }
    })?;
    let canonical = summarize_zspace_generation_evidence(request)?;
    let canonical_value = serde_json::to_value(&canonical).map_err(|error| {
        ZSpaceGenerationEvidenceError::MalformedReport {
            message: error.to_string(),
        }
    })?;
    if !super::canonical_json::values_equivalent(&report, &canonical_value) {
        return Err(ZSpaceGenerationEvidenceError::MalformedReport {
            message: "report does not match the canonical Rust generation evidence".to_owned(),
        });
    }
    Ok(canonical)
}

fn validate_request_identities(
    request: &ZSpaceGenerationEvidenceRequest,
) -> Result<(), ZSpaceGenerationEvidenceError> {
    for (field, value) in [
        ("protocol_id", request.protocol_id.as_str()),
        ("runtime_identity_id", request.runtime_identity_id.as_str()),
        ("model_artifact_id", request.model_artifact_id.as_str()),
        ("prompt_set_id", request.prompt_set_id.as_str()),
        ("decoding_config_id", request.decoding_config_id.as_str()),
    ] {
        require_sha256_id(field, value)?;
    }
    Ok(())
}

fn validate_samples(
    samples: &[ZSpaceGenerationEvidenceSample],
) -> Result<(), ZSpaceGenerationEvidenceError> {
    let mut total_token_count = 0usize;
    for (sample_index, sample) in samples.iter().enumerate() {
        require_sha256_id(
            &format!("samples[{sample_index}].prompt_id"),
            &sample.prompt_id,
        )?;
        if sample.seed > ZSPACE_GENERATION_EVIDENCE_MAX_SAFE_INTEGER {
            return Err(ZSpaceGenerationEvidenceError::SeedLimit {
                index: sample_index,
                seed: sample.seed,
                maximum: ZSPACE_GENERATION_EVIDENCE_MAX_SAFE_INTEGER,
            });
        }
        let sample_token_count = sample.continuation_token_ids.len();
        if sample_token_count > ZSPACE_GENERATION_EVIDENCE_MAX_TOKENS_PER_SAMPLE {
            return Err(ZSpaceGenerationEvidenceError::SampleTokenLimit {
                index: sample_index,
                count: sample_token_count,
                maximum: ZSPACE_GENERATION_EVIDENCE_MAX_TOKENS_PER_SAMPLE,
            });
        }
        total_token_count = total_token_count
            .checked_add(sample_token_count)
            .ok_or(ZSpaceGenerationEvidenceError::TokenCountOverflow)?;
        if total_token_count > ZSPACE_GENERATION_EVIDENCE_MAX_TOTAL_TOKENS {
            return Err(ZSpaceGenerationEvidenceError::TotalTokenLimit {
                count: total_token_count,
                maximum: ZSPACE_GENERATION_EVIDENCE_MAX_TOTAL_TOKENS,
            });
        }
        for (token_index, &token_id) in sample.continuation_token_ids.iter().enumerate() {
            if token_id > ZSPACE_GENERATION_EVIDENCE_MAX_SAFE_INTEGER {
                return Err(ZSpaceGenerationEvidenceError::TokenIdLimit {
                    sample_index,
                    token_index,
                    token_id,
                    maximum: ZSPACE_GENERATION_EVIDENCE_MAX_SAFE_INTEGER,
                });
            }
        }
    }
    Ok(())
}

fn reject_duplicate_samples(
    samples: &[ZSpaceGenerationEvidenceSample],
) -> Result<(), ZSpaceGenerationEvidenceError> {
    let mut seen = BTreeSet::new();
    for sample in samples {
        if !seen.insert((sample.prompt_id.clone(), sample.seed)) {
            return Err(ZSpaceGenerationEvidenceError::DuplicateSample {
                prompt_id: sample.prompt_id.clone(),
                seed: sample.seed,
            });
        }
    }
    Ok(())
}

fn require_sha256_id(field: &str, value: &str) -> Result<(), ZSpaceGenerationEvidenceError> {
    let valid = value.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    });
    if valid {
        Ok(())
    } else {
        Err(ZSpaceGenerationEvidenceError::InvalidIdentity {
            field: field.to_owned(),
        })
    }
}

fn summarize_sample(sample: &ZSpaceGenerationEvidenceSample) -> ZSpaceGenerationSampleEvidence {
    let tokens = &sample.continuation_token_ids;
    let token_count = tokens.len();
    let adjacent_transition_count = token_count.saturating_sub(1);
    let consecutive_repeated_token_count =
        tokens.windows(2).filter(|pair| pair[0] == pair[1]).count();
    let periodic = longest_periodic_suffix(tokens);
    let ngrams = ZSPACE_GENERATION_EVIDENCE_NGRAM_ORDERS
        .into_iter()
        .map(|order| summarize_ngrams(tokens, order))
        .collect::<Vec<_>>();
    let consecutive_repetition_ratio =
        ratio(consecutive_repeated_token_count, adjacent_transition_count);
    let periodic_suffix_repeated_token_ratio = ratio(
        periodic.map_or(0, |value| value.repeated_token_count),
        token_count,
    );
    let loop_score = ngrams[2].repetition_ratio.unwrap_or(0.0)
        + consecutive_repetition_ratio.unwrap_or(0.0)
        + periodic_suffix_repeated_token_ratio.unwrap_or(0.0);
    ZSpaceGenerationSampleEvidence {
        prompt_id: sample.prompt_id.clone(),
        seed: sample.seed,
        token_count,
        empty_continuation: tokens.is_empty(),
        adjacent_transition_count,
        consecutive_repeated_token_count,
        consecutive_repetition_ratio,
        periodic_loop_detected: periodic.is_some(),
        periodic_suffix_period: periodic.map(|value| value.period),
        periodic_suffix_token_count: periodic.map_or(0, |value| value.token_count),
        periodic_suffix_repeated_token_count: periodic
            .map_or(0, |value| value.repeated_token_count),
        periodic_suffix_repetition_count: periodic.map_or(0, |value| value.repetition_count),
        periodic_suffix_repeated_token_ratio,
        loop_score,
        ngrams,
    }
}

fn summarize_ngrams(tokens: &[u64], order: usize) -> ZSpaceGenerationNgramEvidence {
    let possible_count = tokens.len().saturating_sub(order.saturating_sub(1));
    let mut counts = BTreeMap::new();
    for ngram in tokens.windows(order) {
        *counts.entry(ngram).or_insert(0usize) += 1;
    }
    let unique_count = counts.len();
    let repeated_occurrence_count = possible_count - unique_count;
    ZSpaceGenerationNgramEvidence {
        order,
        possible_count,
        unique_count,
        repeated_occurrence_count,
        maximum_occurrence_count: counts.values().copied().max().unwrap_or(0),
        distinct_ratio: ratio(unique_count, possible_count),
        repetition_ratio: ratio(repeated_occurrence_count, possible_count),
    }
}

fn longest_periodic_suffix(tokens: &[u64]) -> Option<PeriodicSuffix> {
    shared_longest_periodic_suffix(
        tokens,
        ZSPACE_GENERATION_EVIDENCE_PERIODIC_SUFFIX_MAX_PERIOD,
        ZSPACE_GENERATION_EVIDENCE_PERIODIC_SUFFIX_MIN_REPETITIONS,
    )
}

fn summarize_aggregate(
    samples: &[ZSpaceGenerationSampleEvidence],
) -> ZSpaceGenerationAggregateEvidence {
    let sample_count = samples.len();
    let nonempty_sample_count = samples
        .iter()
        .filter(|sample| !sample.empty_continuation)
        .count();
    let total_token_count = samples.iter().map(|sample| sample.token_count).sum();
    let adjacent_transition_count = samples
        .iter()
        .map(|sample| sample.adjacent_transition_count)
        .sum();
    let consecutive_repeated_token_count = samples
        .iter()
        .map(|sample| sample.consecutive_repeated_token_count)
        .sum();
    let periodic_loop_sample_count = samples
        .iter()
        .filter(|sample| sample.periodic_loop_detected)
        .count();
    let periodic_suffix_repeated_token_count = samples
        .iter()
        .map(|sample| sample.periodic_suffix_repeated_token_count)
        .sum();
    let ngrams = ZSPACE_GENERATION_EVIDENCE_NGRAM_ORDERS
        .into_iter()
        .map(|order| {
            let values = samples
                .iter()
                .map(|sample| &sample.ngrams[order - 1])
                .collect::<Vec<_>>();
            let possible_count = values.iter().map(|value| value.possible_count).sum();
            let unique_count = values.iter().map(|value| value.unique_count).sum();
            let repeated_occurrence_count = values
                .iter()
                .map(|value| value.repeated_occurrence_count)
                .sum();
            let maximum_occurrence_count = values
                .iter()
                .map(|value| value.maximum_occurrence_count)
                .max()
                .unwrap_or(0);
            ZSpaceGenerationNgramEvidence {
                order,
                possible_count,
                unique_count,
                repeated_occurrence_count,
                maximum_occurrence_count,
                distinct_ratio: ratio(unique_count, possible_count),
                repetition_ratio: ratio(repeated_occurrence_count, possible_count),
            }
        })
        .collect();
    ZSpaceGenerationAggregateEvidence {
        sample_count,
        nonempty_sample_count,
        empty_sample_count: sample_count - nonempty_sample_count,
        total_token_count,
        minimum_token_count: samples
            .iter()
            .map(|sample| sample.token_count)
            .min()
            .unwrap_or(0),
        maximum_token_count: samples
            .iter()
            .map(|sample| sample.token_count)
            .max()
            .unwrap_or(0),
        adjacent_transition_count,
        consecutive_repeated_token_count,
        consecutive_repetition_ratio: ratio(
            consecutive_repeated_token_count,
            adjacent_transition_count,
        ),
        periodic_loop_sample_count,
        periodic_loop_sample_ratio: periodic_loop_sample_count as f64 / sample_count as f64,
        periodic_suffix_repeated_token_count,
        periodic_suffix_repeated_token_ratio: ratio(
            periodic_suffix_repeated_token_count,
            total_token_count,
        ),
        sample_mean_loop_score: samples.iter().map(|sample| sample.loop_score).sum::<f64>()
            / sample_count as f64,
        maximum_loop_score: samples
            .iter()
            .map(|sample| sample.loop_score)
            .fold(0.0, f64::max),
        ngrams,
    }
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

fn generation_evidence_id(
    request: &ZSpaceGenerationEvidenceRequest,
) -> Result<String, ZSpaceGenerationEvidenceError> {
    let encoded = serde_json::to_vec(&(ZSPACE_GENERATION_EVIDENCE_CONTRACT_VERSION, request))
        .map_err(|error| ZSpaceGenerationEvidenceError::MalformedReport {
            message: error.to_string(),
        })?;
    let digest = Sha256::digest(encoded);
    let mut output = String::with_capacity(71);
    output.push_str("sha256:");
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn identity(character: char) -> String {
        format!("sha256:{}", character.to_string().repeat(64))
    }

    fn request(samples: Vec<ZSpaceGenerationEvidenceSample>) -> ZSpaceGenerationEvidenceRequest {
        ZSpaceGenerationEvidenceRequest {
            protocol_id: identity('a'),
            runtime_identity_id: identity('b'),
            model_artifact_id: identity('c'),
            prompt_set_id: identity('d'),
            decoding_config_id: identity('e'),
            samples,
        }
    }

    fn sample(
        prompt_character: char,
        seed: u64,
        continuation_token_ids: &[u64],
    ) -> ZSpaceGenerationEvidenceSample {
        ZSpaceGenerationEvidenceSample {
            prompt_id: identity(prompt_character),
            seed,
            continuation_token_ids: continuation_token_ids.to_vec(),
        }
    }

    #[test]
    fn summarizes_repetition_diversity_and_periodic_suffixes() {
        let report = summarize_zspace_generation_evidence(request(vec![
            sample('1', 13, &[1, 2, 1, 2, 1, 2]),
            sample('2', 13, &[7, 8, 9]),
        ]))
        .expect("generation evidence");

        assert_eq!(report.sample_count, 2);
        assert_eq!(
            report.evidence_scope,
            "held_out_generation_token_observation"
        );
        assert!(!report.efficacy_claim_ready);
        assert!(report.evidence_id.starts_with("sha256:"));
        let repeated = &report.samples[0];
        assert!(repeated.periodic_loop_detected);
        assert_eq!(repeated.periodic_suffix_period, Some(2));
        assert_eq!(repeated.periodic_suffix_token_count, 6);
        assert_eq!(repeated.periodic_suffix_repeated_token_count, 4);
        assert_eq!(repeated.periodic_suffix_repetition_count, 3);
        assert_eq!(repeated.ngrams[0].possible_count, 6);
        assert_eq!(repeated.ngrams[0].unique_count, 2);
        assert_eq!(repeated.ngrams[0].repeated_occurrence_count, 4);
        assert_eq!(repeated.ngrams[1].possible_count, 5);
        assert_eq!(repeated.ngrams[1].unique_count, 2);
        assert_eq!(repeated.ngrams[1].repeated_occurrence_count, 3);
        assert_eq!(repeated.ngrams[1].maximum_occurrence_count, 3);
        assert!((repeated.loop_score - (0.5 + 4.0 / 6.0)).abs() < 1.0e-12);
        assert_eq!(report.aggregate.total_token_count, 9);
        assert_eq!(report.aggregate.periodic_loop_sample_count, 1);
        assert_eq!(report.aggregate.periodic_loop_sample_ratio, 0.5);
        assert_eq!(report.aggregate.ngrams[1].possible_count, 7);
        assert_eq!(report.aggregate.ngrams[1].unique_count, 4);
        assert_eq!(report.aggregate.ngrams[1].repeated_occurrence_count, 3);
        assert_eq!(report.aggregate.maximum_loop_score, repeated.loop_score);
    }

    #[test]
    fn records_empty_continuations_instead_of_discarding_them() {
        let report = summarize_zspace_generation_evidence(request(vec![sample('1', 13, &[])]))
            .expect("empty continuation evidence");
        let sample = &report.samples[0];

        assert!(sample.empty_continuation);
        assert_eq!(sample.consecutive_repetition_ratio, None);
        assert_eq!(sample.periodic_suffix_repeated_token_ratio, None);
        assert!(sample
            .ngrams
            .iter()
            .all(|ngram| ngram.distinct_ratio.is_none()));
        assert_eq!(report.aggregate.empty_sample_count, 1);
        assert_eq!(report.aggregate.periodic_loop_sample_ratio, 0.0);
        assert_eq!(report.aggregate.periodic_suffix_repeated_token_ratio, None);
    }

    #[test]
    fn selects_the_longest_trailing_periodic_run() {
        let report = summarize_zspace_generation_evidence(request(vec![sample(
            '1',
            13,
            &[9, 1, 2, 1, 2, 1, 2],
        )]))
        .expect("periodic suffix evidence");
        let sample = &report.samples[0];

        assert_eq!(sample.periodic_suffix_period, Some(2));
        assert_eq!(sample.periodic_suffix_token_count, 6);
        assert_eq!(sample.periodic_suffix_repeated_token_count, 4);
    }

    #[test]
    fn repeated_single_tokens_use_the_smallest_period() {
        let report =
            summarize_zspace_generation_evidence(request(vec![sample('1', 13, &[5, 5, 5, 5])]))
                .expect("single-token loop evidence");
        let sample = &report.samples[0];

        assert_eq!(sample.periodic_suffix_period, Some(1));
        assert_eq!(sample.periodic_suffix_token_count, 4);
        assert_eq!(sample.periodic_suffix_repeated_token_count, 3);
        assert_eq!(sample.consecutive_repeated_token_count, 3);
        assert_eq!(sample.consecutive_repetition_ratio, Some(1.0));
    }

    #[test]
    fn canonicalizes_sample_order_before_identity() {
        let first = summarize_zspace_generation_evidence(request(vec![
            sample('2', 17, &[2]),
            sample('1', 13, &[1]),
        ]))
        .expect("first evidence");
        let second = summarize_zspace_generation_evidence(request(vec![
            sample('1', 13, &[1]),
            sample('2', 17, &[2]),
        ]))
        .expect("second evidence");

        assert_eq!(first, second);
    }

    #[test]
    fn rejects_duplicate_prompt_seed_samples() {
        let duplicate = request(vec![sample('1', 13, &[1]), sample('1', 13, &[2])]);

        assert!(matches!(
            summarize_zspace_generation_evidence(duplicate),
            Err(ZSpaceGenerationEvidenceError::DuplicateSample { .. })
        ));
    }

    #[test]
    fn rejects_invalid_identity_and_unsafe_integer_inputs() {
        let mut invalid_identity = request(vec![sample('1', 13, &[1])]);
        invalid_identity.model_artifact_id = "model".to_owned();
        assert!(matches!(
            summarize_zspace_generation_evidence(invalid_identity),
            Err(ZSpaceGenerationEvidenceError::InvalidIdentity { .. })
        ));

        let unsafe_token = request(vec![sample(
            '1',
            13,
            &[ZSPACE_GENERATION_EVIDENCE_MAX_SAFE_INTEGER + 1],
        )]);
        assert!(matches!(
            summarize_zspace_generation_evidence(unsafe_token),
            Err(ZSpaceGenerationEvidenceError::TokenIdLimit { .. })
        ));
    }

    #[test]
    fn rejects_oversized_samples() {
        let tokens = vec![0; ZSPACE_GENERATION_EVIDENCE_MAX_TOKENS_PER_SAMPLE + 1];
        let oversized = request(vec![sample('1', 13, &tokens)]);

        assert!(matches!(
            summarize_zspace_generation_evidence(oversized),
            Err(ZSpaceGenerationEvidenceError::SampleTokenLimit { .. })
        ));
    }

    #[test]
    fn validator_recomputes_and_rejects_tampering() {
        let report = summarize_zspace_generation_evidence(request(vec![sample(
            '1',
            13,
            &[1, 2, 1, 2, 1, 2],
        )]))
        .expect("generation evidence");
        let encoded = serde_json::to_value(&report).expect("encoded evidence");
        assert_eq!(
            validate_zspace_generation_evidence_value(encoded.clone()).expect("validated evidence"),
            report
        );

        let mut tampered = encoded;
        tampered["aggregate"]["periodic_loop_sample_count"] = json!(0);
        assert!(matches!(
            validate_zspace_generation_evidence_value(tampered),
            Err(ZSpaceGenerationEvidenceError::MalformedReport { .. })
        ));
    }

    #[test]
    fn validator_accepts_javascript_integer_spelling_for_nested_float_fields() {
        let report = summarize_zspace_generation_evidence(request(vec![sample(
            '1',
            13,
            &[1, 2, 1, 2, 1, 2],
        )]))
        .expect("generation evidence");
        let mut stored = serde_json::to_value(&report).expect("serialized evidence");
        stored["samples"][0]["consecutive_repetition_ratio"] = json!(0);
        stored["aggregate"]["periodic_loop_sample_ratio"] = json!(1);

        assert_eq!(
            validate_zspace_generation_evidence_value(stored)
                .expect("JavaScript numeric spelling is equivalent"),
            report
        );
    }
}
