// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Auditable multi-rank TopK staging.
//!
//! HIP/RCCL is a transport here, not an alternate source of TopK semantics.
//! Every gathered candidate set is committed by `topk_dist`'s Rust exact merge.
//! Input indices are global candidate identities; rank-local namespaces must be
//! globalized by the caller and are rejected if they collide across ranks.

use serde::{Deserialize, Serialize};
use st_tensor::{emit_tensor_op, emit_tensor_op_meta};

use super::topk_dist::{
    merge_topk_shards_f32, select_topk_shard_f32, topk_shards_exactly_equal, TopKMergeError,
    TopKMergeOutcome, TopKShard,
};
use super::wire::canonical_u64;

pub const TOPK_STAGE_REPORT_KIND: &str = "spiraltorch.topk_stage_report";
pub const TOPK_STAGE_REPORT_CONTRACT_VERSION: &str = "spiraltorch.topk_stage_report.v1";
pub const TOPK_STAGE_SEMANTIC_OWNER: &str = "st-core::distributed::topk3_stage";
pub const TOPK_STAGE_SEMANTIC_BACKEND: &str = "rust";

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StagePlan {
    pub k_local: usize,
    pub k_merge: usize,
    pub nodes: usize,
}

impl StagePlan {
    pub fn validate(self) -> Result<(), TopKStageError> {
        if self.k_local == 0 || self.k_merge == 0 {
            return Err(TopKStageError::ZeroK);
        }
        if self.nodes == 0 {
            return Err(TopKStageError::ZeroRanks);
        }
        if self.k_local != self.k_merge {
            return Err(TopKStageError::InvalidPlan {
                message: format!(
                    "exact local keep-k {} must equal final k {}",
                    self.k_local, self.k_merge
                ),
            });
        }
        Ok(())
    }
}

/// Exact global TopK only needs each rank's exact local TopK of the same size.
pub fn stage_plan(k: usize, nodes: usize) -> Result<StagePlan, TopKStageError> {
    let plan = StagePlan {
        k_local: k,
        k_merge: k,
        nodes,
    };
    plan.validate()?;
    Ok(plan)
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DistCtx {
    pub nranks: usize,
    pub rank: usize,
    pub use_hip: bool,
    /// Legacy magic kernel IDs are retained only to reject them explicitly.
    pub merge_kind: Option<u32>,
}

impl DistCtx {
    pub fn validate(self) -> Result<(), TopKStageError> {
        if self.nranks == 0 {
            return Err(TopKStageError::ZeroRanks);
        }
        if self.rank >= self.nranks {
            return Err(TopKStageError::RankOutOfRange {
                rank: self.rank,
                nranks: self.nranks,
            });
        }
        if let Some(kernel) = self.merge_kind {
            return Err(TopKStageError::UncertifiedLegacyKernel { kernel });
        }
        if self.nranks > 1 && !self.use_hip {
            return Err(TopKStageError::DistributedTransportRequired {
                nranks: self.nranks,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TopKStageReport {
    pub kind: String,
    pub contract_version: String,
    pub semantic_owner: String,
    pub semantic_backend: String,
    #[serde(with = "canonical_u64")]
    pub rank: u64,
    #[serde(with = "canonical_u64")]
    pub world_size: u64,
    #[serde(with = "canonical_u64")]
    pub requested_k: u64,
    #[serde(with = "canonical_u64")]
    pub local_input_candidates: u64,
    #[serde(with = "canonical_u64")]
    pub local_selected_candidates: u64,
    #[serde(with = "canonical_u64")]
    pub global_candidates: u64,
    #[serde(with = "canonical_u64")]
    pub output_candidates: u64,
    pub requested_transport_backend: String,
    pub transport_backend: String,
    pub merge_backend: String,
    pub local_input_sha256: String,
    pub local_output_sha256: String,
    pub global_input_sha256: String,
    pub output_sha256: String,
    pub committed: bool,
}

impl TopKStageReport {
    pub fn validate(&self) -> Result<(), TopKStageError> {
        for (field, actual, expected) in [
            ("kind", self.kind.as_str(), TOPK_STAGE_REPORT_KIND),
            (
                "contract_version",
                self.contract_version.as_str(),
                TOPK_STAGE_REPORT_CONTRACT_VERSION,
            ),
            (
                "semantic_owner",
                self.semantic_owner.as_str(),
                TOPK_STAGE_SEMANTIC_OWNER,
            ),
            (
                "semantic_backend",
                self.semantic_backend.as_str(),
                TOPK_STAGE_SEMANTIC_BACKEND,
            ),
        ] {
            if actual != expected {
                return Err(invalid_report(
                    field,
                    format!("must be {expected}, got {actual}"),
                ));
            }
        }
        if self.world_size == 0 || self.rank >= self.world_size {
            return Err(invalid_report(
                "rank",
                "must identify a participant within a non-empty world",
            ));
        }
        if self.requested_k == 0
            || self.local_input_candidates < self.requested_k
            || self.local_selected_candidates != self.requested_k
            || self.output_candidates != self.requested_k
        {
            return Err(invalid_report(
                "requested_k",
                "local and global selections must commit exactly k candidates",
            ));
        }
        let expected_global =
            self.world_size
                .checked_mul(self.requested_k)
                .ok_or(TopKStageError::CountOverflow {
                    field: "global_candidates",
                })?;
        if self.global_candidates != expected_global {
            return Err(invalid_report(
                "global_candidates",
                "must equal world_size multiplied by requested_k",
            ));
        }
        if !matches!(
            self.requested_transport_backend.as_str(),
            "none" | "hip_rccl"
        ) {
            return Err(invalid_report(
                "requested_transport_backend",
                "must be none or hip_rccl",
            ));
        }
        if self.world_size > 1 && self.requested_transport_backend != "hip_rccl" {
            return Err(invalid_report(
                "requested_transport_backend",
                "multi-rank TopK requires an explicit hip_rccl request",
            ));
        }
        let expected_transport = if self.world_size == 1 {
            "none"
        } else {
            "hip_rccl_u64_allgather"
        };
        if self.transport_backend != expected_transport {
            return Err(invalid_report(
                "transport_backend",
                format!("must be {expected_transport} for this topology"),
            ));
        }
        if self.merge_backend != "rust_cpu_exact_topk" {
            return Err(invalid_report(
                "merge_backend",
                "must be rust_cpu_exact_topk",
            ));
        }
        for (field, digest) in [
            ("local_input_sha256", self.local_input_sha256.as_str()),
            ("local_output_sha256", self.local_output_sha256.as_str()),
            ("global_input_sha256", self.global_input_sha256.as_str()),
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
                "TopK stage reports must describe a committed exact merge",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TopKStageOutcome {
    pub local_selection: TopKMergeOutcome,
    pub gathered_shards: Vec<TopKShard<f32>>,
    pub global_merge: TopKMergeOutcome,
    pub report: TopKStageReport,
}

impl TopKStageOutcome {
    pub fn shard(&self) -> &TopKShard<f32> {
        &self.global_merge.shard
    }

    pub fn validate(&self) -> Result<(), TopKStageError> {
        self.report.validate()?;
        self.local_selection
            .validate_committed_output()
            .map_err(TopKStageError::Merge)?;
        self.global_merge
            .validate_against(
                &self.gathered_shards,
                usize::try_from(self.report.requested_k).map_err(|_| {
                    TopKStageError::CountOverflow {
                        field: "requested_k",
                    }
                })?,
            )
            .map_err(TopKStageError::Merge)?;
        if self.gathered_shards.len()
            != usize::try_from(self.report.world_size).map_err(|_| {
                TopKStageError::CountOverflow {
                    field: "world_size",
                }
            })?
        {
            return Err(invalid_report(
                "gathered_shards",
                "must contain exactly one shard per rank",
            ));
        }
        let expected_k = usize::try_from(self.report.requested_k).map_err(|_| {
            TopKStageError::CountOverflow {
                field: "requested_k",
            }
        })?;
        if self
            .gathered_shards
            .iter()
            .any(|shard| shard.len() != expected_k)
        {
            return Err(invalid_report(
                "gathered_shards",
                "every rank must contribute exactly k local candidates",
            ));
        }
        let rank = usize::try_from(self.report.rank)
            .map_err(|_| TopKStageError::CountOverflow { field: "rank" })?;
        if !topk_shards_exactly_equal(&self.local_selection.shard, &self.gathered_shards[rank]) {
            return Err(invalid_report(
                "local_selection",
                "local selection must be the exact gathered contribution for this rank",
            ));
        }
        if self.report.local_input_sha256 != self.local_selection.report.input_sha256
            || self.report.local_output_sha256 != self.local_selection.report.output_sha256
            || self.report.global_input_sha256 != self.global_merge.report.input_sha256
            || self.report.output_sha256 != self.global_merge.report.output_sha256
        {
            return Err(invalid_report(
                "output_sha256",
                "stage commitments must match the local and global TopK reports",
            ));
        }
        if self.report.local_selected_candidates != self.local_selection.report.output_candidates
            || self.report.global_candidates != self.global_merge.report.input_candidates
            || self.report.output_candidates != self.global_merge.report.output_candidates
            || self.report.requested_k != self.local_selection.report.requested_k
            || self.report.requested_k != self.global_merge.report.requested_k
        {
            return Err(invalid_report(
                "output_candidates",
                "stage counts must match the local and global TopK reports",
            ));
        }
        Ok(())
    }

    /// Replays the local selection and the complete gathered global merge.
    pub fn validate_against_local(&self, local: &TopKShard<f32>) -> Result<(), TopKStageError> {
        self.validate()?;
        let requested_k = usize::try_from(self.report.requested_k).map_err(|_| {
            TopKStageError::CountOverflow {
                field: "requested_k",
            }
        })?;
        self.local_selection
            .validate_against(std::slice::from_ref(local), requested_k)
            .map_err(TopKStageError::Merge)?;
        if self.report.local_input_candidates
            != u64::try_from(local.len()).map_err(|_| TopKStageError::CountOverflow {
                field: "local_input_candidates",
            })?
        {
            return Err(invalid_report(
                "local_input_candidates",
                "stage report does not match the replayed local input",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum TopKStageError {
    #[error("TopK stage requires k greater than zero")]
    ZeroK,
    #[error("TopK stage requires at least one rank")]
    ZeroRanks,
    #[error("TopK stage rank {rank} must be within world size {nranks}")]
    RankOutOfRange { rank: usize, nranks: usize },
    #[error("invalid TopK stage plan: {message}")]
    InvalidPlan { message: String },
    #[error("TopK stage requires a distributed transport for {nranks} ranks")]
    DistributedTransportRequired { nranks: usize },
    #[error("legacy HIP TopK kernel id {kernel} is not parity-certified")]
    UncertifiedLegacyKernel { kernel: u32 },
    #[error("HIP/RCCL TopK transport is unavailable in this build")]
    HipUnavailable,
    #[error("HIP/RCCL TopK transport failed: {message}")]
    HipFailure { message: String },
    #[error("HIP/RCCL topology mismatch: expected rank {expected_rank}/{expected_world}, got {actual_rank}/{actual_world}")]
    HipTopologyMismatch {
        expected_rank: usize,
        expected_world: usize,
        actual_rank: i32,
        actual_world: i32,
    },
    #[error("TopK stage count overflow at {field}")]
    CountOverflow { field: &'static str },
    #[error(transparent)]
    Merge(#[from] TopKMergeError),
    #[error("invalid TopK stage report at {field}: {message}")]
    InvalidReport { field: String, message: String },
}

pub fn run_topk3_stage(
    ctx: &DistCtx,
    local: TopKShard<f32>,
    k: usize,
) -> Result<TopKStageOutcome, TopKStageError> {
    ctx.validate()?;
    let plan = stage_plan(k, ctx.nranks)?;
    let local_input_candidates = local.len();
    let local_selection = select_topk_shard_f32(&local, plan.k_local)?;

    let transport_backend;
    let gathered_shards = if ctx.nranks == 1 {
        transport_backend = "none";
        vec![local_selection.shard.clone()]
    } else {
        transport_backend = "hip_rccl_u64_allgather";
        gather_topk_shards_hip(ctx, &local_selection.shard)?
    };
    let global_merge = merge_topk_shards_f32(&gathered_shards, plan.k_merge)?;
    let report = TopKStageReport {
        kind: TOPK_STAGE_REPORT_KIND.to_owned(),
        contract_version: TOPK_STAGE_REPORT_CONTRACT_VERSION.to_owned(),
        semantic_owner: TOPK_STAGE_SEMANTIC_OWNER.to_owned(),
        semantic_backend: TOPK_STAGE_SEMANTIC_BACKEND.to_owned(),
        rank: u64::try_from(ctx.rank)
            .map_err(|_| TopKStageError::CountOverflow { field: "rank" })?,
        world_size: u64::try_from(ctx.nranks).map_err(|_| TopKStageError::CountOverflow {
            field: "world_size",
        })?,
        requested_k: u64::try_from(k).map_err(|_| TopKStageError::CountOverflow {
            field: "requested_k",
        })?,
        local_input_candidates: u64::try_from(local_input_candidates).map_err(|_| {
            TopKStageError::CountOverflow {
                field: "local_input_candidates",
            }
        })?,
        local_selected_candidates: u64::try_from(local_selection.shard.len()).map_err(|_| {
            TopKStageError::CountOverflow {
                field: "local_selected_candidates",
            }
        })?,
        global_candidates: gathered_shards.iter().try_fold(0u64, |total, shard| {
            let count = u64::try_from(shard.len()).map_err(|_| TopKStageError::CountOverflow {
                field: "global_candidates",
            })?;
            total
                .checked_add(count)
                .ok_or(TopKStageError::CountOverflow {
                    field: "global_candidates",
                })
        })?,
        output_candidates: u64::try_from(global_merge.shard.len()).map_err(|_| {
            TopKStageError::CountOverflow {
                field: "output_candidates",
            }
        })?,
        requested_transport_backend: if ctx.use_hip { "hip_rccl" } else { "none" }.to_owned(),
        transport_backend: transport_backend.to_owned(),
        merge_backend: "rust_cpu_exact_topk".to_owned(),
        local_input_sha256: local_selection.report.input_sha256.clone(),
        local_output_sha256: local_selection.report.output_sha256.clone(),
        global_input_sha256: global_merge.report.input_sha256.clone(),
        output_sha256: global_merge.report.output_sha256.clone(),
        committed: true,
    };
    let outcome = TopKStageOutcome {
        local_selection,
        gathered_shards,
        global_merge,
        report,
    };
    outcome.validate_against_local(&local)?;
    emit_tensor_op(
        "distributed_topk_stage",
        &[local_input_candidates, ctx.nranks],
        &[k, 2],
    );
    emit_tensor_op_meta("distributed_topk_stage", || {
        serde_json::to_value(&outcome.report).expect("validated TopK stage report must serialize")
    });
    Ok(outcome)
}

#[cfg(all(feature = "hip", feature = "hip-real"))]
fn gather_topk_shards_hip(
    ctx: &DistCtx,
    local: &TopKShard<f32>,
) -> Result<Vec<TopKShard<f32>>, TopKStageError> {
    use st_backend_hip::rccl_comm::init_rccl_from_env;
    use st_backend_hip::real::{
        allgather_u64_dev, free, malloc, memcpy_d2h_async, memcpy_h2d_async, stream_synchronize,
        HipPtr, HipStream,
    };

    struct HipBuffer(HipPtr);

    impl HipBuffer {
        fn new(bytes: usize) -> Result<Self, TopKStageError> {
            malloc(bytes)
                .map(Self)
                .map_err(|error| TopKStageError::HipFailure {
                    message: error.to_string(),
                })
        }

        fn as_ptr(&self) -> HipPtr {
            self.0
        }

        fn release(mut self) -> Result<(), TopKStageError> {
            free(self.0).map_err(|error| TopKStageError::HipFailure {
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

    let comm = init_rccl_from_env().map_err(|error| TopKStageError::HipFailure {
        message: error.to_string(),
    })?;
    let expected_rank =
        i32::try_from(ctx.rank).map_err(|_| TopKStageError::CountOverflow { field: "rank" })?;
    let expected_world = i32::try_from(ctx.nranks).map_err(|_| TopKStageError::CountOverflow {
        field: "world_size",
    })?;
    if comm.rank != expected_rank || comm.world != expected_world {
        return Err(TopKStageError::HipTopologyMismatch {
            expected_rank: ctx.rank,
            expected_world: ctx.nranks,
            actual_rank: comm.rank,
            actual_world: comm.world,
        });
    }

    let local_packed = local
        .vals
        .iter()
        .zip(&local.idxs)
        .map(|(&value, &index)| (u64::from(value.to_bits()) << 32) | u64::from(index as u32))
        .collect::<Vec<_>>();
    let element_bytes = std::mem::size_of::<u64>();
    let send_bytes =
        element_bytes
            .checked_mul(local_packed.len())
            .ok_or(TopKStageError::CountOverflow {
                field: "hip_send_bytes",
            })?;
    let gathered_count =
        local_packed
            .len()
            .checked_mul(ctx.nranks)
            .ok_or(TopKStageError::CountOverflow {
                field: "hip_gathered_candidates",
            })?;
    let receive_bytes =
        element_bytes
            .checked_mul(gathered_count)
            .ok_or(TopKStageError::CountOverflow {
                field: "hip_receive_bytes",
            })?;
    let stream = HipStream::create().map_err(|error| TopKStageError::HipFailure {
        message: error.to_string(),
    })?;
    let send = HipBuffer::new(send_bytes)?;
    let receive = HipBuffer::new(receive_bytes)?;
    let mut gathered = vec![0u64; gathered_count];
    unsafe {
        memcpy_h2d_async(
            send.as_ptr(),
            local_packed.as_ptr().cast::<u8>(),
            send_bytes,
            &stream,
        )
        .map_err(|error| TopKStageError::HipFailure {
            message: error.to_string(),
        })?;
    }
    allgather_u64_dev(
        comm.comm,
        &stream,
        send.as_ptr(),
        receive.as_ptr(),
        local_packed.len(),
    )
    .map_err(|error| TopKStageError::HipFailure {
        message: error.to_string(),
    })?;
    unsafe {
        memcpy_d2h_async(
            gathered.as_mut_ptr().cast::<u8>(),
            receive.as_ptr(),
            receive_bytes,
            &stream,
        )
        .map_err(|error| TopKStageError::HipFailure {
            message: error.to_string(),
        })?;
    }
    stream_synchronize(&stream).map_err(|error| TopKStageError::HipFailure {
        message: error.to_string(),
    })?;
    send.release()?;
    receive.release()?;

    gathered
        .chunks_exact(local.len())
        .map(decode_packed_shard)
        .collect()
}

#[cfg(not(all(feature = "hip", feature = "hip-real")))]
fn gather_topk_shards_hip(
    _ctx: &DistCtx,
    _local: &TopKShard<f32>,
) -> Result<Vec<TopKShard<f32>>, TopKStageError> {
    Err(TopKStageError::HipUnavailable)
}

#[cfg(any(all(feature = "hip", feature = "hip-real"), test))]
fn decode_packed_shard(packed: &[u64]) -> Result<TopKShard<f32>, TopKStageError> {
    let vals = packed
        .iter()
        .map(|value| f32::from_bits((value >> 32) as u32))
        .collect::<Vec<_>>();
    let idxs = packed
        .iter()
        .map(|value| (*value & u64::from(u32::MAX)) as u32 as i32)
        .collect::<Vec<_>>();
    TopKShard::try_new(vals, idxs).map_err(TopKStageError::Merge)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn invalid_report(field: impl Into<String>, message: impl Into<String>) -> TopKStageError {
    TopKStageError::InvalidReport {
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

    fn local_context() -> DistCtx {
        DistCtx {
            nranks: 1,
            rank: 0,
            use_hip: false,
            merge_kind: None,
        }
    }

    #[test]
    fn stage_plan_is_exact_and_rejects_empty_topology() {
        assert_eq!(
            stage_plan(32, 4).unwrap(),
            StagePlan {
                k_local: 32,
                k_merge: 32,
                nodes: 4,
            }
        );
        assert_eq!(stage_plan(0, 1), Err(TopKStageError::ZeroK));
        assert_eq!(stage_plan(1, 0), Err(TopKStageError::ZeroRanks));
        assert!(matches!(
            StagePlan {
                k_local: 2,
                k_merge: 1,
                nodes: 1,
            }
            .validate(),
            Err(TopKStageError::InvalidPlan { .. })
        ));
    }

    #[test]
    fn context_rejects_invalid_rank_missing_transport_and_legacy_kernel() {
        assert_eq!(
            DistCtx {
                nranks: 2,
                rank: 2,
                use_hip: true,
                merge_kind: None,
            }
            .validate(),
            Err(TopKStageError::RankOutOfRange { rank: 2, nranks: 2 })
        );
        assert_eq!(
            DistCtx {
                nranks: 2,
                rank: 0,
                use_hip: false,
                merge_kind: None,
            }
            .validate(),
            Err(TopKStageError::DistributedTransportRequired { nranks: 2 })
        );
        assert_eq!(
            DistCtx {
                nranks: 1,
                rank: 0,
                use_hip: false,
                merge_kind: Some(2),
            }
            .validate(),
            Err(TopKStageError::UncertifiedLegacyKernel { kernel: 2 })
        );
    }

    #[test]
    fn single_rank_stage_uses_same_exact_local_and_global_contract() {
        let outcome =
            run_topk3_stage(&local_context(), shard(&[(0.1, 1), (0.9, 9), (0.4, 4)]), 2).unwrap();

        assert_eq!(outcome.shard(), &shard(&[(0.9, 9), (0.4, 4)]));
        assert_eq!(outcome.report.transport_backend, "none");
        assert_eq!(outcome.report.requested_transport_backend, "none");
        assert_eq!(outcome.report.merge_backend, "rust_cpu_exact_topk");
        outcome.validate().unwrap();
    }

    #[test]
    fn distributed_stage_never_duplicates_local_data_as_a_fallback() {
        let context = DistCtx {
            nranks: 2,
            rank: 0,
            use_hip: true,
            merge_kind: None,
        };
        #[cfg(not(all(feature = "hip", feature = "hip-real")))]
        assert_eq!(
            run_topk3_stage(&context, shard(&[(1.0, 1)]), 1),
            Err(TopKStageError::HipUnavailable)
        );
    }

    #[test]
    fn single_rank_records_hip_request_without_claiming_transport_work() {
        let context = DistCtx {
            use_hip: true,
            ..local_context()
        };
        let outcome = run_topk3_stage(&context, shard(&[(1.0, 1)]), 1).unwrap();

        assert_eq!(outcome.report.requested_transport_backend, "hip_rccl");
        assert_eq!(outcome.report.transport_backend, "none");
        outcome.validate().unwrap();
    }

    #[test]
    fn packed_transport_roundtrip_preserves_float_bits_and_indices() {
        let original = shard(&[(0.0, 0), (-0.0, 7), (f32::MAX, i32::MAX)]);
        let packed = original
            .vals
            .iter()
            .zip(&original.idxs)
            .map(|(&value, &index)| (u64::from(value.to_bits()) << 32) | u64::from(index as u32))
            .collect::<Vec<_>>();
        let decoded = decode_packed_shard(&packed).unwrap();

        assert_eq!(decoded, original);
        assert_eq!(decoded.vals[1].to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn stage_report_uses_strict_string_u64_wire_and_detects_tampering() {
        let outcome = run_topk3_stage(&local_context(), shard(&[(0.2, 2), (0.8, 8)]), 1).unwrap();
        let wire = serde_json::to_value(&outcome.report).unwrap();
        assert_eq!(wire["world_size"], "1");
        assert_eq!(wire["requested_k"], "1");
        assert_eq!(
            serde_json::from_value::<TopKStageReport>(wire.clone()).unwrap(),
            outcome.report
        );

        let mut numeric = wire.clone();
        numeric["world_size"] = serde_json::json!(1);
        assert!(serde_json::from_value::<TopKStageReport>(numeric).is_err());
        let mut unknown = wire;
        unknown
            .as_object_mut()
            .unwrap()
            .insert("commander".to_owned(), serde_json::json!("python"));
        assert!(serde_json::from_value::<TopKStageReport>(unknown).is_err());

        let mut tampered = outcome;
        tampered.report.output_sha256 = "0".repeat(64);
        assert!(tampered.validate().is_err());
    }

    #[test]
    fn stage_validation_binds_local_selection_to_the_rank_contribution() {
        let mut outcome =
            run_topk3_stage(&local_context(), shard(&[(0.2, 2), (0.8, 8)]), 1).unwrap();
        let replacement = select_topk_shard_f32(&shard(&[(0.9, 9)]), 1).unwrap();
        outcome.report.local_input_sha256 = replacement.report.input_sha256.clone();
        outcome.report.local_output_sha256 = replacement.report.output_sha256.clone();
        outcome.local_selection = replacement;

        assert!(outcome.validate().is_err());
    }

    #[test]
    fn stage_replay_binds_the_original_local_candidates() {
        let local = shard(&[(0.2, 2), (0.8, 8)]);
        let outcome = run_topk3_stage(&local_context(), local.clone(), 1).unwrap();
        outcome.validate_against_local(&local).unwrap();

        assert!(outcome
            .validate_against_local(&shard(&[(0.2, 2), (0.9, 8)]))
            .is_err());
    }

    #[test]
    fn stage_emits_transport_and_exact_merge_evidence() {
        let _lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        run_topk3_stage(&local_context(), shard(&[(0.3, 3), (0.7, 7)]), 1).unwrap();
        st_tensor::set_thread_meta_observer(previous);

        let events = events.lock().unwrap();
        let report = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "distributed_topk_stage" && data["kind"] == TOPK_STAGE_REPORT_KIND
            })
            .expect("TopK stage report");
        assert_eq!(report.1["transport_backend"], "none");
        assert_eq!(report.1["requested_transport_backend"], "none");
        assert_eq!(report.1["merge_backend"], "rust_cpu_exact_topk");
        assert_eq!(report.1["committed"], true);
    }
}
