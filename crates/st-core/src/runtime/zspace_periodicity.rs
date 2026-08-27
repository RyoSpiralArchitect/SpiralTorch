// SPDX-License-Identifier: AGPL-3.0-or-later

//! Shared periodic-suffix semantics for Z-space training, inference, and evidence.
//!
//! The allocation-free kernel remains on the training and generation hot paths.
//! [`analyze_zspace_periodicity`] adds the bounded, content-addressed contract
//! used by direct Rust callers and language bindings.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt::Write;
use thiserror::Error;

/// Contract version for persisted periodicity reports.
pub const ZSPACE_PERIODICITY_CONTRACT_VERSION: &str = "spiraltorch.zspace_periodicity.v1";
/// Stable kind discriminator for persisted periodicity reports.
pub const ZSPACE_PERIODICITY_KIND: &str = "spiraltorch.zspace_periodicity";
/// Rust module that owns periodic-suffix semantics.
pub const ZSPACE_PERIODICITY_SEMANTIC_OWNER: &str = "st-core::runtime::zspace_periodicity";
/// Backend that computed and validated the report.
pub const ZSPACE_PERIODICITY_SEMANTIC_BACKEND: &str = "rust";
/// Default maximum period shared by training and generation evidence.
pub const ZSPACE_PERIODIC_SUFFIX_MAX_PERIOD: usize = 16;
/// Default minimum full-period span shared by training and generation evidence.
pub const ZSPACE_PERIODIC_SUFFIX_MIN_REPETITIONS: usize = 3;
/// Maximum number of observed plus appended tokens accepted by the public contract.
pub const ZSPACE_PERIODICITY_MAX_TOKENS: usize = 1_000_000;
/// Maximum configurable suffix period accepted by the public contract.
pub const ZSPACE_PERIODICITY_MAX_PERIOD: usize = 4_096;
/// Maximum configurable repetition threshold accepted by the public contract.
pub const ZSPACE_PERIODICITY_MAX_MINIMUM_REPETITIONS: usize = 1_000_000;
/// Maximum conservative token-comparison budget accepted by the public contract.
pub const ZSPACE_PERIODICITY_MAX_COMPARISON_WORK: usize = 16_777_216;
/// Largest integer transported exactly by every supported JSON/JavaScript client.
pub const ZSPACE_PERIODICITY_MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
/// Maximum JSON-compatible bytes admitted before request/report deserialization.
pub const ZSPACE_PERIODICITY_MAX_INGRESS_BYTES: u64 = 32 * 1_024 * 1_024;
/// Maximum JSON-compatible values admitted before request/report deserialization.
pub const ZSPACE_PERIODICITY_MAX_INGRESS_NODES: u64 = 1_000_128;
/// Maximum JSON-compatible nesting admitted before request/report deserialization.
pub const ZSPACE_PERIODICITY_MAX_INGRESS_DEPTH: u32 = 8;
/// Canonical periodic-suffix and tie-breaking semantics.
pub const ZSPACE_PERIODICITY_RULE: &str =
    "trailing periodic run with period<=maximum_period and token_count>=period*minimum_repetitions; the run may start mid-cycle; select by maximum repeated_token_count, then token_count, then smaller period";
/// Content identity rule for the canonical request.
pub const ZSPACE_PERIODICITY_ANALYSIS_ID_RULE: &str =
    "sha256 of UTF-8 JSON for [contract_version, canonical request] with no insignificant whitespace";
/// Interpretation boundary carried by every report.
pub const ZSPACE_PERIODICITY_EVIDENCE_BOUNDARY: &str =
    "periodicity is a structural token-ID suffix observation; it does not measure semantic quality, prove a generation loop will continue, or establish model or training superiority";

/// Bounded configuration for the public periodicity contract.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ZSpacePeriodicityConfig {
    /// Largest candidate period inspected by the suffix kernel.
    pub maximum_period: usize,
    /// Minimum full-period span required for a match.
    pub minimum_repetitions: usize,
}

impl Default for ZSpacePeriodicityConfig {
    fn default() -> Self {
        Self {
            maximum_period: ZSPACE_PERIODIC_SUFFIX_MAX_PERIOD,
            minimum_repetitions: ZSPACE_PERIODIC_SUFFIX_MIN_REPETITIONS,
        }
    }
}

/// Token sequence analyzed by the public periodicity contract.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpacePeriodicityRequest {
    /// Observed token IDs in temporal order.
    pub token_ids: Vec<u64>,
    /// Optional proposal evaluated without allocating a second token vector.
    #[serde(default)]
    pub appended_token_id: Option<u64>,
    /// Period and repetition bounds. Missing JSON configuration uses shared defaults.
    #[serde(default)]
    pub config: ZSpacePeriodicityConfig,
}

/// Longest periodic suffix selected by the shared kernel.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PeriodicSuffix {
    /// Number of tokens in one repeated period.
    pub period: usize,
    /// Total number of tokens in the selected trailing suffix.
    pub token_count: usize,
    /// Tokens after the first period that repeat an earlier token.
    pub repeated_token_count: usize,
    /// Number of complete periods contained in the selected run.
    pub repetition_count: usize,
}

/// Canonical, replayable result of one bounded periodicity analysis.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpacePeriodicityReport {
    /// Version of the persisted report contract.
    pub contract_version: &'static str,
    /// Stable report kind.
    pub kind: &'static str,
    /// Rust module that owns the analysis semantics.
    pub semantic_owner: &'static str,
    /// Backend that performed the analysis.
    pub semantic_backend: &'static str,
    /// Whether the request passed all bounded Rust validation.
    pub analysis_validated: bool,
    /// Content identity of the canonical request and contract version.
    pub analysis_id: String,
    /// Readiness of this structural analysis.
    pub status: &'static str,
    /// Canonical request used to reproduce the report.
    pub request: ZSpacePeriodicityRequest,
    /// Human-readable periodic-suffix selection rule.
    pub rule: &'static str,
    /// Human-readable content-identity rule.
    pub analysis_id_rule: &'static str,
    /// Whether the analysis includes an appended proposal token.
    pub analysis_scope: &'static str,
    /// Number of observed tokens before any proposal is appended.
    pub input_token_count: usize,
    /// Number of observed plus appended tokens inspected by the kernel.
    pub effective_token_count: usize,
    /// Number of candidate periods considered under the configured bounds.
    pub candidate_period_count: usize,
    /// Conservative upper bound on token comparisons for this request.
    pub comparison_work_upper_bound: usize,
    /// Whether a suffix met the configured period and repetition bounds.
    pub periodic_loop_detected: bool,
    /// Longest selected periodic suffix, if one exists.
    pub periodic_suffix: Option<PeriodicSuffix>,
    /// Selected suffix tokens divided by effective tokens.
    pub periodic_suffix_token_ratio: Option<f64>,
    /// Repeated suffix tokens divided by effective tokens.
    pub periodic_suffix_repeated_token_ratio: Option<f64>,
    /// Always false because structural periodicity alone is not efficacy evidence.
    pub efficacy_claim_ready: bool,
    /// Interpretation boundary that must accompany the structural result.
    pub evidence_boundary: &'static str,
}

/// Validation failures for the public periodicity contract.
#[derive(Debug, Error, PartialEq)]
pub enum ZSpacePeriodicityError {
    #[error("periodicity maximum_period {value} must be in 1..={maximum}")]
    MaximumPeriod { value: usize, maximum: usize },
    #[error("periodicity minimum_repetitions {value} must be in 2..={maximum}")]
    MinimumRepetitions { value: usize, maximum: usize },
    #[error("periodicity effective token count overflow")]
    TokenCountOverflow,
    #[error("periodicity token count {count} exceeds maximum {maximum}")]
    TokenLimit { count: usize, maximum: usize },
    #[error("periodicity token {index} ID {token_id} exceeds cross-client maximum {maximum}")]
    TokenIdLimit {
        index: usize,
        token_id: u64,
        maximum: u64,
    },
    #[error("periodicity comparison work {work} exceeds maximum {maximum}")]
    ComparisonWorkLimit { work: usize, maximum: usize },
    #[error("malformed periodicity report: {message}")]
    MalformedReport { message: String },
}

/// Analyze a bounded token sequence through the shared periodic-suffix kernel.
pub fn analyze_zspace_periodicity(
    request: ZSpacePeriodicityRequest,
) -> Result<ZSpacePeriodicityReport, ZSpacePeriodicityError> {
    validate_config(request.config)?;
    let input_token_count = request.token_ids.len();
    let effective_token_count = input_token_count
        .checked_add(usize::from(request.appended_token_id.is_some()))
        .ok_or(ZSpacePeriodicityError::TokenCountOverflow)?;
    if effective_token_count > ZSPACE_PERIODICITY_MAX_TOKENS {
        return Err(ZSpacePeriodicityError::TokenLimit {
            count: effective_token_count,
            maximum: ZSPACE_PERIODICITY_MAX_TOKENS,
        });
    }
    for (index, &token_id) in request
        .token_ids
        .iter()
        .chain(request.appended_token_id.iter())
        .enumerate()
    {
        if token_id > ZSPACE_PERIODICITY_MAX_SAFE_INTEGER {
            return Err(ZSpacePeriodicityError::TokenIdLimit {
                index,
                token_id,
                maximum: ZSPACE_PERIODICITY_MAX_SAFE_INTEGER,
            });
        }
    }
    let candidate_period_count = request
        .config
        .maximum_period
        .min(effective_token_count / request.config.minimum_repetitions);
    let comparison_work_upper_bound = effective_token_count
        .checked_mul(candidate_period_count)
        .ok_or(ZSpacePeriodicityError::TokenCountOverflow)?;
    if comparison_work_upper_bound > ZSPACE_PERIODICITY_MAX_COMPARISON_WORK {
        return Err(ZSpacePeriodicityError::ComparisonWorkLimit {
            work: comparison_work_upper_bound,
            maximum: ZSPACE_PERIODICITY_MAX_COMPARISON_WORK,
        });
    }

    let periodic_suffix = match request.appended_token_id {
        Some(appended_token_id) => longest_periodic_suffix_with_appended_token(
            &request.token_ids,
            appended_token_id,
            request.config.maximum_period,
            request.config.minimum_repetitions,
        ),
        None => longest_periodic_suffix(
            &request.token_ids,
            request.config.maximum_period,
            request.config.minimum_repetitions,
        ),
    };
    let periodic_suffix_token_count = periodic_suffix.map_or(0, |suffix| suffix.token_count);
    let periodic_suffix_repeated_token_count =
        periodic_suffix.map_or(0, |suffix| suffix.repeated_token_count);
    let analysis_id = periodicity_analysis_id(&request)?;

    Ok(ZSpacePeriodicityReport {
        contract_version: ZSPACE_PERIODICITY_CONTRACT_VERSION,
        kind: ZSPACE_PERIODICITY_KIND,
        semantic_owner: ZSPACE_PERIODICITY_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_PERIODICITY_SEMANTIC_BACKEND,
        analysis_validated: true,
        analysis_id,
        status: "ready",
        rule: ZSPACE_PERIODICITY_RULE,
        analysis_id_rule: ZSPACE_PERIODICITY_ANALYSIS_ID_RULE,
        analysis_scope: if request.appended_token_id.is_some() {
            "observed_sequence_with_appended_token"
        } else {
            "observed_sequence"
        },
        input_token_count,
        effective_token_count,
        candidate_period_count,
        comparison_work_upper_bound,
        periodic_loop_detected: periodic_suffix.is_some(),
        periodic_suffix,
        periodic_suffix_token_ratio: ratio(periodic_suffix_token_count, effective_token_count),
        periodic_suffix_repeated_token_ratio: ratio(
            periodic_suffix_repeated_token_count,
            effective_token_count,
        ),
        efficacy_claim_ready: false,
        evidence_boundary: ZSPACE_PERIODICITY_EVIDENCE_BOUNDARY,
        request,
    })
}

/// Recompute a persisted report and reject any non-canonical field.
pub fn validate_zspace_periodicity_value(
    report: serde_json::Value,
) -> Result<ZSpacePeriodicityReport, ZSpacePeriodicityError> {
    let request = report
        .get("request")
        .cloned()
        .ok_or_else(|| malformed("missing request"))?;
    let request = serde_json::from_value(request).map_err(|error| malformed(error.to_string()))?;
    let canonical = analyze_zspace_periodicity(request)?;
    let canonical_value =
        serde_json::to_value(&canonical).map_err(|error| malformed(error.to_string()))?;
    if !super::canonical_json::values_equivalent(&report, &canonical_value) {
        return Err(malformed(
            "report does not match the canonical Rust periodicity analysis",
        ));
    }
    Ok(canonical)
}

fn validate_config(config: ZSpacePeriodicityConfig) -> Result<(), ZSpacePeriodicityError> {
    if config.maximum_period == 0 || config.maximum_period > ZSPACE_PERIODICITY_MAX_PERIOD {
        return Err(ZSpacePeriodicityError::MaximumPeriod {
            value: config.maximum_period,
            maximum: ZSPACE_PERIODICITY_MAX_PERIOD,
        });
    }
    if config.minimum_repetitions < 2
        || config.minimum_repetitions > ZSPACE_PERIODICITY_MAX_MINIMUM_REPETITIONS
    {
        return Err(ZSpacePeriodicityError::MinimumRepetitions {
            value: config.minimum_repetitions,
            maximum: ZSPACE_PERIODICITY_MAX_MINIMUM_REPETITIONS,
        });
    }
    Ok(())
}

fn periodicity_analysis_id(
    request: &ZSpacePeriodicityRequest,
) -> Result<String, ZSpacePeriodicityError> {
    let encoded = serde_json::to_vec(&(ZSPACE_PERIODICITY_CONTRACT_VERSION, request))
        .map_err(|error| malformed(error.to_string()))?;
    let digest = Sha256::digest(encoded);
    let mut output = String::with_capacity(71);
    output.push_str("sha256:");
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    Ok(output)
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

fn malformed(message: impl Into<String>) -> ZSpacePeriodicityError {
    ZSpacePeriodicityError::MalformedReport {
        message: message.into(),
    }
}

pub(crate) fn longest_periodic_suffix(
    tokens: &[u64],
    maximum_period: usize,
    minimum_repetitions: usize,
) -> Option<PeriodicSuffix> {
    longest_periodic_suffix_by(
        tokens.len(),
        |index| tokens[index],
        maximum_period,
        minimum_repetitions,
    )
}

pub(crate) fn longest_periodic_suffix_with_appended_token(
    tokens: &[u64],
    appended_token: u64,
    maximum_period: usize,
    minimum_repetitions: usize,
) -> Option<PeriodicSuffix> {
    longest_periodic_suffix_by(
        tokens.len() + 1,
        |index| {
            if index == tokens.len() {
                appended_token
            } else {
                tokens[index]
            }
        },
        maximum_period,
        minimum_repetitions,
    )
}

fn longest_periodic_suffix_by(
    token_count: usize,
    token_at: impl Fn(usize) -> u64,
    maximum_period: usize,
    minimum_repetitions: usize,
) -> Option<PeriodicSuffix> {
    if maximum_period == 0 || minimum_repetitions < 2 {
        return None;
    }
    let maximum_period = maximum_period.min(token_count / minimum_repetitions);
    let mut best: Option<PeriodicSuffix> = None;
    for period in 1..=maximum_period {
        let mut repeated_token_count = 0usize;
        while repeated_token_count + period < token_count
            && token_at(token_count - 1 - repeated_token_count)
                == token_at(token_count - 1 - repeated_token_count - period)
        {
            repeated_token_count += 1;
        }
        let token_count = repeated_token_count + period;
        if token_count < period * minimum_repetitions {
            continue;
        }
        let candidate = PeriodicSuffix {
            period,
            token_count,
            repeated_token_count,
            repetition_count: token_count / period,
        };
        let replace = best.is_none_or(|current| {
            (
                candidate.repeated_token_count,
                candidate.token_count,
                usize::MAX - candidate.period,
            ) > (
                current.repeated_token_count,
                current.token_count,
                usize::MAX - current.period,
            )
        });
        if replace {
            best = Some(candidate);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_longest_bounded_periodic_suffix() {
        assert_eq!(
            longest_periodic_suffix(&[9, 1, 2, 1, 2, 1, 2], 16, 3),
            Some(PeriodicSuffix {
                period: 2,
                token_count: 6,
                repeated_token_count: 4,
                repetition_count: 3,
            })
        );
    }

    #[test]
    fn respects_period_and_repetition_bounds() {
        assert_eq!(longest_periodic_suffix(&[1, 2, 1, 2], 16, 3), None);
        assert_eq!(longest_periodic_suffix(&[1, 2, 1, 2, 1, 2], 1, 3), None);
        assert_eq!(
            longest_periodic_suffix(&[7, 7, 7], 16, 3).unwrap().period,
            1
        );
    }

    #[test]
    fn periodic_run_may_begin_mid_cycle_without_overstating_repetition_count() {
        let suffix = longest_periodic_suffix(&[1, 2, 1, 2, 1, 2, 1], 16, 3)
            .expect("seven-token periodic run");

        assert_eq!(suffix.period, 2);
        assert_eq!(suffix.token_count, 7);
        assert_eq!(suffix.repeated_token_count, 5);
        assert_eq!(suffix.repetition_count, 3);
        assert!(ZSPACE_PERIODICITY_RULE.contains("may start mid-cycle"));
    }

    #[test]
    fn appended_token_uses_the_same_periodicity_semantics_without_allocating() {
        assert_eq!(
            longest_periodic_suffix_with_appended_token(&[9, 1, 2, 1, 2, 1], 2, 16, 3),
            longest_periodic_suffix(&[9, 1, 2, 1, 2, 1, 2], 16, 3)
        );
        assert_eq!(
            longest_periodic_suffix_with_appended_token(&[1, 2, 1, 2, 1], 3, 16, 3),
            None
        );
    }

    #[test]
    fn appended_token_kernel_matches_allocated_reference_exhaustively() {
        for prefix_len in 0..=8 {
            for bits in 0usize..(1usize << prefix_len) {
                let prefix = (0..prefix_len)
                    .map(|index| ((bits >> index) & 1) as u64)
                    .collect::<Vec<_>>();
                for appended_token in [0, 1] {
                    let mut allocated = prefix.clone();
                    allocated.push(appended_token);
                    for maximum_period in 1..=4 {
                        for minimum_repetitions in 2..=4 {
                            assert_eq!(
                                longest_periodic_suffix_with_appended_token(
                                    &prefix,
                                    appended_token,
                                    maximum_period,
                                    minimum_repetitions,
                                ),
                                longest_periodic_suffix(
                                    &allocated,
                                    maximum_period,
                                    minimum_repetitions,
                                ),
                                "prefix={prefix:?} appended={appended_token} period={maximum_period} repetitions={minimum_repetitions}",
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn public_report_uses_defaults_and_matches_the_appended_kernel() {
        let request: ZSpacePeriodicityRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [9, 1, 2, 1, 2, 1],
            "appended_token_id": 2
        }))
        .expect("defaulted request");
        let report = analyze_zspace_periodicity(request.clone()).expect("periodicity report");

        assert_eq!(request.config, ZSpacePeriodicityConfig::default());
        assert_eq!(
            report.analysis_scope,
            "observed_sequence_with_appended_token"
        );
        assert_eq!(report.input_token_count, 6);
        assert_eq!(report.effective_token_count, 7);
        assert!(report.periodic_loop_detected);
        assert_eq!(
            report.periodic_suffix,
            longest_periodic_suffix_with_appended_token(&[9, 1, 2, 1, 2, 1], 2, 16, 3)
        );
        assert_eq!(report.periodic_suffix_token_ratio, Some(6.0 / 7.0));
        assert_eq!(report.periodic_suffix_repeated_token_ratio, Some(4.0 / 7.0));
        assert!(report.analysis_id.starts_with("sha256:"));
    }

    #[test]
    fn public_report_fails_closed_on_unsafe_tokens_and_excessive_work() {
        let unsafe_token = analyze_zspace_periodicity(ZSpacePeriodicityRequest {
            token_ids: vec![ZSPACE_PERIODICITY_MAX_SAFE_INTEGER + 1],
            appended_token_id: None,
            config: ZSpacePeriodicityConfig::default(),
        })
        .expect_err("unsafe token must fail");
        assert!(matches!(
            unsafe_token,
            ZSpacePeriodicityError::TokenIdLimit { index: 0, .. }
        ));

        let excessive_work = analyze_zspace_periodicity(ZSpacePeriodicityRequest {
            token_ids: vec![1; 8_192],
            appended_token_id: None,
            config: ZSpacePeriodicityConfig {
                maximum_period: ZSPACE_PERIODICITY_MAX_PERIOD,
                minimum_repetitions: 2,
            },
        })
        .expect_err("excessive work must fail");
        assert!(matches!(
            excessive_work,
            ZSpacePeriodicityError::ComparisonWorkLimit { .. }
        ));

        let unsafe_appended = analyze_zspace_periodicity(ZSpacePeriodicityRequest {
            token_ids: vec![1, 2],
            appended_token_id: Some(ZSPACE_PERIODICITY_MAX_SAFE_INTEGER + 1),
            config: ZSpacePeriodicityConfig::default(),
        })
        .expect_err("unsafe appended token must fail");
        assert!(matches!(
            unsafe_appended,
            ZSpacePeriodicityError::TokenIdLimit { index: 2, .. }
        ));
    }

    #[test]
    fn public_report_defines_empty_and_request_identity_boundaries() {
        let empty = analyze_zspace_periodicity(ZSpacePeriodicityRequest {
            token_ids: Vec::new(),
            appended_token_id: None,
            config: ZSpacePeriodicityConfig::default(),
        })
        .expect("empty analysis");
        assert_eq!(empty.effective_token_count, 0);
        assert_eq!(empty.candidate_period_count, 0);
        assert_eq!(empty.periodic_suffix, None);
        assert_eq!(empty.periodic_suffix_token_ratio, None);
        assert_eq!(empty.periodic_suffix_repeated_token_ratio, None);

        let observed = analyze_zspace_periodicity(ZSpacePeriodicityRequest {
            token_ids: vec![1, 2, 1, 2, 1, 2],
            appended_token_id: None,
            config: ZSpacePeriodicityConfig::default(),
        })
        .expect("observed analysis");
        let configured = analyze_zspace_periodicity(ZSpacePeriodicityRequest {
            token_ids: vec![1, 2, 1, 2, 1, 2],
            appended_token_id: None,
            config: ZSpacePeriodicityConfig {
                maximum_period: 8,
                minimum_repetitions: 3,
            },
        })
        .expect("configured analysis");
        assert_eq!(observed.periodic_suffix, configured.periodic_suffix);
        assert_ne!(observed.analysis_id, configured.analysis_id);
    }

    #[test]
    fn validator_recomputes_and_rejects_tampering() {
        let report = analyze_zspace_periodicity(ZSpacePeriodicityRequest {
            token_ids: vec![1, 2, 1, 2, 1, 2],
            appended_token_id: None,
            config: ZSpacePeriodicityConfig::default(),
        })
        .expect("periodicity report");
        let mut value = serde_json::to_value(&report).expect("serialized report");
        assert_eq!(
            validate_zspace_periodicity_value(value.clone()).expect("canonical report"),
            report
        );

        value["periodic_loop_detected"] = serde_json::Value::Bool(false);
        let error = validate_zspace_periodicity_value(value).expect_err("tamper must fail");
        assert!(matches!(
            error,
            ZSpacePeriodicityError::MalformedReport { .. }
        ));
    }

    #[test]
    fn validator_accepts_javascript_json_integer_spelling_for_exact_ratios() {
        let full_periodic = analyze_zspace_periodicity(ZSpacePeriodicityRequest {
            token_ids: vec![1, 1, 1],
            appended_token_id: None,
            config: ZSpacePeriodicityConfig::default(),
        })
        .expect("fully periodic report");
        let mut stored = serde_json::to_value(&full_periodic).expect("serialized report");
        stored["periodic_suffix_token_ratio"] = serde_json::json!(1);
        assert_eq!(
            validate_zspace_periodicity_value(stored).expect("JavaScript 1 is numeric 1.0"),
            full_periodic
        );

        let non_periodic = analyze_zspace_periodicity(ZSpacePeriodicityRequest {
            token_ids: vec![1, 2, 3],
            appended_token_id: None,
            config: ZSpacePeriodicityConfig::default(),
        })
        .expect("non-periodic report");
        let mut stored = serde_json::to_value(&non_periodic).expect("serialized report");
        stored["periodic_suffix_token_ratio"] = serde_json::json!(0);
        stored["periodic_suffix_repeated_token_ratio"] = serde_json::json!(0);
        assert_eq!(
            validate_zspace_periodicity_value(stored).expect("JavaScript 0 is numeric 0.0"),
            non_periodic
        );

        let mut tampered = serde_json::to_value(&non_periodic).expect("serialized report");
        tampered["periodic_suffix_token_ratio"] = serde_json::json!(1);
        let error = validate_zspace_periodicity_value(tampered)
            .expect_err("numeric normalization must not accept a changed ratio");
        assert!(matches!(
            error,
            ZSpacePeriodicityError::MalformedReport { .. }
        ));
    }
}
