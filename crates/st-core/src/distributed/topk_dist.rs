// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Strict deterministic TopK merge semantics for distributed candidate shards.

use std::collections::HashMap;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use st_tensor::{emit_tensor_op, emit_tensor_op_meta};

use super::wire::canonical_u64;

pub const TOPK_MERGE_REPORT_KIND: &str = "spiraltorch.topk_merge_report";
pub const TOPK_MERGE_REPORT_CONTRACT_VERSION: &str = "spiraltorch.topk_merge_report.v1";
pub const TOPK_MERGE_SEMANTIC_OWNER: &str = "st-core::distributed::topk_dist";
pub const TOPK_MERGE_SEMANTIC_BACKEND: &str = "rust";

const TOPK_INPUT_DIGEST_DOMAIN: &[u8] = b"spiraltorch.topk_merge.input.v1\0";
const TOPK_OUTPUT_DIGEST_DOMAIN: &[u8] = b"spiraltorch.topk_merge.output.v1\0";

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TopKShard<T> {
    pub vals: Vec<T>,
    /// Non-negative global candidate identities, unique across merged shards.
    pub idxs: Vec<i32>,
}

impl<T> TopKShard<T> {
    pub fn new(vals: Vec<T>, idxs: Vec<i32>) -> Self {
        Self { vals, idxs }
    }

    pub fn len(&self) -> usize {
        self.vals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vals.is_empty()
    }
}

impl TopKShard<f32> {
    pub fn try_new(vals: Vec<f32>, idxs: Vec<i32>) -> Result<Self, TopKMergeError> {
        let shard = Self { vals, idxs };
        shard.validate(0)?;
        Ok(shard)
    }

    pub fn validate(&self, shard: usize) -> Result<(), TopKMergeError> {
        if self.vals.len() != self.idxs.len() {
            return Err(TopKMergeError::ShapeMismatch {
                shard,
                values: self.vals.len(),
                indices: self.idxs.len(),
            });
        }
        let mut seen = HashMap::with_capacity(self.idxs.len());
        for (position, (&value, &index)) in self.vals.iter().zip(&self.idxs).enumerate() {
            if !value.is_finite() {
                return Err(TopKMergeError::NonFiniteValue {
                    shard,
                    position,
                    value,
                });
            }
            if index < 0 {
                return Err(TopKMergeError::NegativeIndex {
                    shard,
                    position,
                    index,
                });
            }
            if let Some(first_position) = seen.insert(index, position) {
                return Err(TopKMergeError::DuplicateIndex {
                    index,
                    first_shard: shard,
                    first_position,
                    shard,
                    position,
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TopKMergeReport {
    pub kind: String,
    pub contract_version: String,
    pub semantic_owner: String,
    pub semantic_backend: String,
    #[serde(with = "canonical_u64")]
    pub input_shards: u64,
    #[serde(with = "canonical_u64")]
    pub input_candidates: u64,
    #[serde(with = "canonical_u64")]
    pub requested_k: u64,
    #[serde(with = "canonical_u64")]
    pub output_candidates: u64,
    pub requested_backend: String,
    pub selected_backend: String,
    pub control_backend: String,
    pub merge_kernel: String,
    pub ordering: String,
    pub input_l1_energy: f64,
    pub output_l1_energy: f64,
    pub retained_l1_ratio: f64,
    pub top_score: f32,
    pub cutoff_score: f32,
    pub input_sha256: String,
    pub output_sha256: String,
    pub committed: bool,
}

impl TopKMergeReport {
    pub fn validate(&self) -> Result<(), TopKMergeError> {
        for (field, actual, expected) in [
            ("kind", self.kind.as_str(), TOPK_MERGE_REPORT_KIND),
            (
                "contract_version",
                self.contract_version.as_str(),
                TOPK_MERGE_REPORT_CONTRACT_VERSION,
            ),
            (
                "semantic_owner",
                self.semantic_owner.as_str(),
                TOPK_MERGE_SEMANTIC_OWNER,
            ),
            (
                "semantic_backend",
                self.semantic_backend.as_str(),
                TOPK_MERGE_SEMANTIC_BACKEND,
            ),
        ] {
            if actual != expected {
                return Err(invalid_report(
                    field,
                    format!("must be {expected}, got {actual}"),
                ));
            }
        }
        if self.input_shards == 0 {
            return Err(invalid_report(
                "input_shards",
                "TopK merge requires at least one shard",
            ));
        }
        if self.requested_k == 0 {
            return Err(invalid_report(
                "requested_k",
                "TopK merge requires k greater than zero",
            ));
        }
        if self.requested_k > self.input_candidates || self.output_candidates != self.requested_k {
            return Err(invalid_report(
                "output_candidates",
                "output must contain exactly k values drawn from the input candidates",
            ));
        }
        for (field, actual, expected) in [
            ("requested_backend", self.requested_backend.as_str(), "cpu"),
            ("selected_backend", self.selected_backend.as_str(), "cpu"),
            (
                "control_backend",
                self.control_backend.as_str(),
                "rust_host",
            ),
            (
                "merge_kernel",
                self.merge_kernel.as_str(),
                "finite_pair_sort_truncate",
            ),
            ("ordering", self.ordering.as_str(), "score_desc_index_asc"),
        ] {
            if actual != expected {
                return Err(invalid_report(
                    field,
                    format!("must be {expected}, got {actual}"),
                ));
            }
        }
        for (field, value) in [
            ("input_l1_energy", self.input_l1_energy),
            ("output_l1_energy", self.output_l1_energy),
            ("retained_l1_ratio", self.retained_l1_ratio),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(invalid_report(field, "must be finite and non-negative"));
            }
        }
        if !self.top_score.is_finite() || !self.cutoff_score.is_finite() {
            return Err(invalid_report(
                "top_score",
                "top and cutoff scores must be finite",
            ));
        }
        if self.top_score < self.cutoff_score {
            return Err(invalid_report(
                "cutoff_score",
                "must not exceed the top score",
            ));
        }
        let energy_tolerance = self.input_l1_energy.max(1.0) * 1.0e-12;
        if self.output_l1_energy > self.input_l1_energy + energy_tolerance {
            return Err(invalid_report(
                "output_l1_energy",
                "selected candidates cannot contain more L1 energy than all inputs",
            ));
        }
        let expected_ratio = if self.input_l1_energy > 0.0 {
            self.output_l1_energy / self.input_l1_energy
        } else {
            1.0
        };
        if (self.retained_l1_ratio - expected_ratio).abs() > 1.0e-12 {
            return Err(invalid_report(
                "retained_l1_ratio",
                "does not match output_l1_energy / input_l1_energy",
            ));
        }
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
        if !self.committed {
            return Err(invalid_report(
                "committed",
                "TopK merge reports must describe a committed selection",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TopKMergeOutcome {
    pub shard: TopKShard<f32>,
    pub report: TopKMergeReport,
}

impl TopKMergeOutcome {
    /// Verifies that the output shard matches every claim that can be checked
    /// without replaying the original input shards.
    pub fn validate_committed_output(&self) -> Result<(), TopKMergeError> {
        self.report.validate()?;
        self.shard.validate(0)?;
        let requested_k =
            usize::try_from(self.report.requested_k).map_err(|_| TopKMergeError::CountOverflow)?;
        if self.shard.len() != requested_k {
            return Err(invalid_report(
                "output_candidates",
                "output shard must contain exactly the committed k candidates",
            ));
        }
        for position in 1..self.shard.len() {
            let left_value = self.shard.vals[position - 1];
            let left_index = self.shard.idxs[position - 1];
            let right_value = self.shard.vals[position];
            let right_index = self.shard.idxs[position];
            if left_value < right_value || (left_value == right_value && left_index > right_index) {
                return Err(invalid_report(
                    "ordering",
                    "output shard must use score-descending, index-ascending order",
                ));
            }
        }
        let output_l1_energy =
            self.shard
                .vals
                .iter()
                .enumerate()
                .try_fold(0.0, |sum, (position, value)| {
                    checked_metric_add("output_l1_energy", position, sum, f64::from(*value).abs())
                })?;
        if self.report.output_candidates
            != u64::try_from(self.shard.len()).map_err(|_| TopKMergeError::CountOverflow)?
            || self.report.output_l1_energy.to_bits() != output_l1_energy.to_bits()
            || self.report.top_score.to_bits() != self.shard.vals[0].to_bits()
            || self.report.cutoff_score.to_bits() != self.shard.vals[requested_k - 1].to_bits()
        {
            return Err(invalid_report(
                "output_metrics",
                "output shard does not match its committed counts, energy, or boundary scores",
            ));
        }
        let expected_output =
            topk_output_sha256(&self.shard, requested_k, self.report.input_sha256.as_str())?;
        if self.report.output_sha256 != expected_output {
            return Err(invalid_report(
                "output_sha256",
                "does not bind the output shard, k, and input commitment",
            ));
        }
        Ok(())
    }

    pub fn validate_against(
        &self,
        shards: &[TopKShard<f32>],
        k: usize,
    ) -> Result<(), TopKMergeError> {
        self.validate_committed_output()?;
        let expected = build_topk_merge(shards, k)?;
        if !topk_shards_exactly_equal(&self.shard, &expected.shard)
            || !topk_reports_exactly_equal(&self.report, &expected.report)
        {
            return Err(invalid_report(
                "outcome",
                "TopK output does not match the committed shards, k, and canonical ordering",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum TopKMergeError {
    #[error("TopK merge requires at least one shard")]
    EmptyShards,
    #[error("TopK merge requires k greater than zero")]
    ZeroK,
    #[error("TopK merge requested k={k} from only {candidates} candidates")]
    KExceedsCandidates { k: usize, candidates: usize },
    #[error("TopK shard {shard} has {values} values but {indices} indices")]
    ShapeMismatch {
        shard: usize,
        values: usize,
        indices: usize,
    },
    #[error("TopK shard {shard} value {position} is non-finite: {value}")]
    NonFiniteValue {
        shard: usize,
        position: usize,
        value: f32,
    },
    #[error("TopK shard {shard} index {position} is negative: {index}")]
    NegativeIndex {
        shard: usize,
        position: usize,
        index: i32,
    },
    #[error(
        "TopK index {index} is ambiguous: first at shard {first_shard} position {first_position}, then shard {shard} position {position}"
    )]
    DuplicateIndex {
        index: i32,
        first_shard: usize,
        first_position: usize,
        shard: usize,
        position: usize,
    },
    #[error("TopK candidate count overflow")]
    CountOverflow,
    #[error("TopK metric {field} became non-finite at position {position}: {value}")]
    NonFiniteMetric {
        field: &'static str,
        position: usize,
        value: f64,
    },
    #[error("invalid TopK merge report at {field}: {message}")]
    InvalidReport { field: String, message: String },
}

/// Merges any number of candidate shards into an exact, canonical global TopK.
pub fn merge_topk_shards_f32(
    shards: &[TopKShard<f32>],
    k: usize,
) -> Result<TopKMergeOutcome, TopKMergeError> {
    let outcome = build_topk_merge(shards, k)?;
    outcome.validate_against(shards, k)?;
    let input_candidates = usize::try_from(outcome.report.input_candidates)
        .map_err(|_| TopKMergeError::CountOverflow)?;
    emit_tensor_op("distributed_topk_merge", &[input_candidates, 2], &[k, 2]);
    emit_tensor_op_meta("distributed_topk_merge", || {
        serde_json::to_value(&outcome.report).expect("validated TopK report must serialize")
    });
    Ok(outcome)
}

pub fn merge_two_shards_f32(
    left: &TopKShard<f32>,
    right: &TopKShard<f32>,
    k: usize,
) -> Result<TopKMergeOutcome, TopKMergeError> {
    merge_topk_shards_f32(&[left.clone(), right.clone()], k)
}

pub fn select_topk_shard_f32(
    shard: &TopKShard<f32>,
    k: usize,
) -> Result<TopKMergeOutcome, TopKMergeError> {
    merge_topk_shards_f32(std::slice::from_ref(shard), k)
}

fn build_topk_merge(
    shards: &[TopKShard<f32>],
    k: usize,
) -> Result<TopKMergeOutcome, TopKMergeError> {
    if shards.is_empty() {
        return Err(TopKMergeError::EmptyShards);
    }
    if k == 0 {
        return Err(TopKMergeError::ZeroK);
    }

    let mut candidate_count = 0usize;
    let mut input_l1_energy = 0.0f64;
    let mut locations = HashMap::<i32, (usize, usize)>::new();
    let mut candidates = Vec::<(f32, i32)>::new();
    for (shard_index, shard) in shards.iter().enumerate() {
        shard.validate(shard_index)?;
        candidate_count = candidate_count
            .checked_add(shard.len())
            .ok_or(TopKMergeError::CountOverflow)?;
        candidates.reserve(shard.len());
        for (position, (&value, &index)) in shard.vals.iter().zip(&shard.idxs).enumerate() {
            if let Some(&(first_shard, first_position)) = locations.get(&index) {
                return Err(TopKMergeError::DuplicateIndex {
                    index,
                    first_shard,
                    first_position,
                    shard: shard_index,
                    position,
                });
            }
            locations.insert(index, (shard_index, position));
            input_l1_energy = checked_metric_add(
                "input_l1_energy",
                position,
                input_l1_energy,
                f64::from(value).abs(),
            )?;
            candidates.push((value, index));
        }
    }
    if k > candidate_count {
        return Err(TopKMergeError::KExceedsCandidates {
            k,
            candidates: candidate_count,
        });
    }

    candidates.sort_by(|left, right| {
        let score_order = if left.0 == right.0 {
            std::cmp::Ordering::Equal
        } else {
            right.0.total_cmp(&left.0)
        };
        score_order.then_with(|| left.1.cmp(&right.1))
    });
    let selected = candidates.into_iter().take(k).collect::<Vec<_>>();
    let vals = selected.iter().map(|&(value, _)| value).collect::<Vec<_>>();
    let idxs = selected.iter().map(|&(_, index)| index).collect::<Vec<_>>();
    let shard = TopKShard { vals, idxs };
    shard.validate(0)?;
    let output_l1_energy =
        shard
            .vals
            .iter()
            .enumerate()
            .try_fold(0.0, |sum, (position, value)| {
                checked_metric_add("output_l1_energy", position, sum, f64::from(*value).abs())
            })?;
    let retained_l1_ratio = if input_l1_energy > 0.0 {
        output_l1_energy / input_l1_energy
    } else {
        1.0
    };
    if !retained_l1_ratio.is_finite() {
        return Err(TopKMergeError::NonFiniteMetric {
            field: "retained_l1_ratio",
            position: 0,
            value: retained_l1_ratio,
        });
    }
    let input_sha256 = topk_input_sha256(shards)?;
    let output_sha256 = topk_output_sha256(&shard, k, &input_sha256)?;
    let report = TopKMergeReport {
        kind: TOPK_MERGE_REPORT_KIND.to_owned(),
        contract_version: TOPK_MERGE_REPORT_CONTRACT_VERSION.to_owned(),
        semantic_owner: TOPK_MERGE_SEMANTIC_OWNER.to_owned(),
        semantic_backend: TOPK_MERGE_SEMANTIC_BACKEND.to_owned(),
        input_shards: u64::try_from(shards.len()).map_err(|_| TopKMergeError::CountOverflow)?,
        input_candidates: u64::try_from(candidate_count)
            .map_err(|_| TopKMergeError::CountOverflow)?,
        requested_k: u64::try_from(k).map_err(|_| TopKMergeError::CountOverflow)?,
        output_candidates: u64::try_from(shard.len()).map_err(|_| TopKMergeError::CountOverflow)?,
        requested_backend: "cpu".to_owned(),
        selected_backend: "cpu".to_owned(),
        control_backend: "rust_host".to_owned(),
        merge_kernel: "finite_pair_sort_truncate".to_owned(),
        ordering: "score_desc_index_asc".to_owned(),
        input_l1_energy,
        output_l1_energy,
        retained_l1_ratio,
        top_score: shard.vals[0],
        cutoff_score: shard.vals[k - 1],
        input_sha256,
        output_sha256,
        committed: true,
    };
    report.validate()?;
    Ok(TopKMergeOutcome { shard, report })
}

fn checked_metric_add(
    field: &'static str,
    position: usize,
    left: f64,
    right: f64,
) -> Result<f64, TopKMergeError> {
    let value = left + right;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(TopKMergeError::NonFiniteMetric {
            field,
            position,
            value,
        })
    }
}

fn topk_input_sha256(shards: &[TopKShard<f32>]) -> Result<String, TopKMergeError> {
    let mut digest = Sha256::new();
    digest.update(TOPK_INPUT_DIGEST_DOMAIN);
    digest.update(
        u64::try_from(shards.len())
            .map_err(|_| TopKMergeError::CountOverflow)?
            .to_le_bytes(),
    );
    for shard in shards {
        digest.update(
            u64::try_from(shard.len())
                .map_err(|_| TopKMergeError::CountOverflow)?
                .to_le_bytes(),
        );
        for (&value, &index) in shard.vals.iter().zip(&shard.idxs) {
            digest.update(value.to_bits().to_le_bytes());
            digest.update(index.to_le_bytes());
        }
    }
    Ok(hex_digest(&digest.finalize()))
}

fn topk_output_sha256(
    shard: &TopKShard<f32>,
    k: usize,
    input_sha256: &str,
) -> Result<String, TopKMergeError> {
    let mut digest = Sha256::new();
    digest.update(TOPK_OUTPUT_DIGEST_DOMAIN);
    digest.update(input_sha256.as_bytes());
    digest.update(
        u64::try_from(k)
            .map_err(|_| TopKMergeError::CountOverflow)?
            .to_le_bytes(),
    );
    for (&value, &index) in shard.vals.iter().zip(&shard.idxs) {
        digest.update(value.to_bits().to_le_bytes());
        digest.update(index.to_le_bytes());
    }
    Ok(hex_digest(&digest.finalize()))
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

pub(super) fn topk_shards_exactly_equal(left: &TopKShard<f32>, right: &TopKShard<f32>) -> bool {
    left.idxs == right.idxs
        && left.vals.len() == right.vals.len()
        && left
            .vals
            .iter()
            .zip(&right.vals)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn topk_reports_exactly_equal(left: &TopKMergeReport, right: &TopKMergeReport) -> bool {
    left == right
        && left.input_l1_energy.to_bits() == right.input_l1_energy.to_bits()
        && left.output_l1_energy.to_bits() == right.output_l1_energy.to_bits()
        && left.retained_l1_ratio.to_bits() == right.retained_l1_ratio.to_bits()
        && left.top_score.to_bits() == right.top_score.to_bits()
        && left.cutoff_score.to_bits() == right.cutoff_score.to_bits()
}

fn invalid_report(field: impl Into<String>, message: impl Into<String>) -> TopKMergeError {
    TopKMergeError::InvalidReport {
        field: field.into(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn shard(values: &[(f32, i32)]) -> TopKShard<f32> {
        TopKShard::new(
            values.iter().map(|&(value, _)| value).collect(),
            values.iter().map(|&(_, index)| index).collect(),
        )
    }

    #[test]
    fn merge_is_exact_deterministic_and_emits_rust_owned_report() {
        let _lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let left = shard(&[(0.5, 5), (0.2, 2), (-0.1, 1)]);
        let right = shard(&[(0.7, 7), (0.4, 4), (-0.2, 0)]);
        let merged = merge_two_shards_f32(&left, &right, 3).unwrap();
        st_tensor::set_thread_meta_observer(previous);

        assert_eq!(merged.shard, shard(&[(0.7, 7), (0.5, 5), (0.4, 4)]));
        merged.validate_against(&[left, right], 3).unwrap();
        let events = events.lock().unwrap();
        let report = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "distributed_topk_merge" && data["kind"] == TOPK_MERGE_REPORT_KIND
            })
            .expect("distributed TopK merge report");
        assert_eq!(report.1["semantic_backend"], "rust");
        assert_eq!(report.1["output_candidates"], "3");
        assert_eq!(report.1["committed"], true);
    }

    #[test]
    fn merge_rejects_shape_mismatch_non_finite_negative_and_duplicate_indices() {
        let mismatch = TopKShard::new(vec![1.0], vec![1, 2]);
        assert!(matches!(
            merge_topk_shards_f32(&[mismatch], 1),
            Err(TopKMergeError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            merge_topk_shards_f32(&[shard(&[(f32::NAN, 1)])], 1),
            Err(TopKMergeError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            merge_topk_shards_f32(&[shard(&[(1.0, -1)])], 1),
            Err(TopKMergeError::NegativeIndex { .. })
        ));
        assert!(matches!(
            merge_topk_shards_f32(&[shard(&[(1.0, 3)]), shard(&[(2.0, 3)])], 1),
            Err(TopKMergeError::DuplicateIndex { index: 3, .. })
        ));
    }

    #[test]
    fn merge_requires_nonempty_shards_positive_k_and_enough_candidates() {
        assert_eq!(
            merge_topk_shards_f32(&[], 1),
            Err(TopKMergeError::EmptyShards)
        );
        assert_eq!(
            merge_topk_shards_f32(&[shard(&[(1.0, 1)])], 0),
            Err(TopKMergeError::ZeroK)
        );
        assert_eq!(
            merge_topk_shards_f32(&[shard(&[(1.0, 1)])], 2),
            Err(TopKMergeError::KExceedsCandidates {
                k: 2,
                candidates: 1
            })
        );
    }

    #[test]
    fn merge_uses_numeric_score_then_index_tie_break_including_signed_zero() {
        let merged = merge_topk_shards_f32(
            &[
                shard(&[(0.0, 7), (-0.0, 3), (1.0, 9)]),
                shard(&[(0.0, 1), (-1.0, 0)]),
            ],
            4,
        )
        .unwrap();

        assert_eq!(merged.shard.idxs, vec![9, 1, 3, 7]);
        assert_eq!(merged.shard.vals[1].to_bits(), 0.0f32.to_bits());
        assert_eq!(merged.shard.vals[2].to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn merge_supports_more_than_two_shards_and_finite_extreme_energy() {
        let shards = [
            shard(&[(f32::MAX, 0)]),
            shard(&[(f32::MAX / 2.0, 1)]),
            shard(&[(-f32::MAX, 2)]),
        ];
        let merged = merge_topk_shards_f32(&shards, 2).unwrap();

        assert_eq!(merged.shard.idxs, vec![0, 1]);
        assert!(merged.report.input_l1_energy.is_finite());
        assert!(merged.report.output_l1_energy.is_finite());
        assert!(merged.report.retained_l1_ratio <= 1.0);
    }

    #[test]
    fn report_wire_is_strict_and_outcome_validation_detects_tampering() {
        let shards = [shard(&[(3.0, 3), (1.0, 1)]), shard(&[(2.0, 2)])];
        let outcome = merge_topk_shards_f32(&shards, 2).unwrap();
        outcome.validate_committed_output().unwrap();
        let wire = serde_json::to_value(&outcome.report).unwrap();
        assert_eq!(wire["input_shards"], "2");
        assert_eq!(wire["requested_k"], "2");
        assert_eq!(
            serde_json::from_value::<TopKMergeReport>(wire.clone()).unwrap(),
            outcome.report
        );

        let mut unknown = wire.clone();
        unknown
            .as_object_mut()
            .unwrap()
            .insert("commander".to_owned(), serde_json::json!("python"));
        assert!(serde_json::from_value::<TopKMergeReport>(unknown).is_err());
        let mut numeric = wire;
        numeric["requested_k"] = serde_json::json!(2);
        assert!(serde_json::from_value::<TopKMergeReport>(numeric).is_err());

        let mut tampered = outcome;
        tampered.shard.idxs.swap(0, 1);
        assert!(tampered.validate_committed_output().is_err());
        assert!(tampered.validate_against(&shards, 2).is_err());

        let zero_shards = [shard(&[(0.0, 1)]), shard(&[(-0.0, 2)])];
        let mut signed_zero_tamper = merge_topk_shards_f32(&zero_shards, 2).unwrap();
        signed_zero_tamper.shard.vals[0] = -0.0;
        assert!(signed_zero_tamper
            .validate_against(&zero_shards, 2)
            .is_err());
    }
}
