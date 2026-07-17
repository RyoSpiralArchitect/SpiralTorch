// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Transactional in-memory collectives with Rust-owned execution evidence.

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex, MutexGuard};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use st_tensor::{
    emit_tensor_op, emit_tensor_op_meta, tensor_op_meta_observer_installed, Tensor, TensorError,
};

use super::wire::canonical_u64;
use crate::backend::execution::current_tensor_util_route;

pub const COLLECTIVE_EXECUTION_REPORT_KIND: &str = "spiraltorch.collective_execution_report";
pub const COLLECTIVE_EXECUTION_REPORT_CONTRACT_VERSION: &str =
    "spiraltorch.collective_execution_report.v1";
pub const COLLECTIVE_EXECUTION_REPORT_SEMANTIC_OWNER: &str = "st-core::distributed::collective";
pub const COLLECTIVE_EXECUTION_REPORT_SEMANTIC_BACKEND: &str = "rust";

const ALL_REDUCE_INPUT_DIGEST_DOMAIN: &[u8] = b"spiraltorch.collective.all_reduce_sum.input.v1\0";
const ALL_REDUCE_OUTPUT_DIGEST_DOMAIN: &[u8] = b"spiraltorch.collective.all_reduce_sum.output.v1\0";
const BROADCAST_INPUT_DIGEST_DOMAIN: &[u8] = b"spiraltorch.collective.broadcast.input.v1\0";
const BROADCAST_OUTPUT_DIGEST_DOMAIN: &[u8] = b"spiraltorch.collective.broadcast.output.v1\0";

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CollectiveOperation {
    AllReduceSum,
    Broadcast,
}

/// Versioned report for one successfully committed collective operation.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CollectiveExecutionReport {
    pub kind: String,
    pub contract_version: String,
    pub semantic_owner: String,
    pub semantic_backend: String,
    pub operation: CollectiveOperation,
    #[serde(with = "canonical_u64")]
    pub participants: u64,
    #[serde(with = "canonical_u64")]
    pub width: u64,
    #[serde(with = "canonical_u64")]
    pub input_values: u64,
    #[serde(with = "canonical_u64")]
    pub output_values: u64,
    pub requested_backend: String,
    pub selected_backend: String,
    pub control_backend: String,
    pub numeric_kernel: String,
    pub actual_numeric_backend_source: String,
    pub input_sha256: String,
    pub output_sha256: String,
    pub committed: bool,
}

impl CollectiveExecutionReport {
    pub fn validate(&self) -> Result<(), CollectiveError> {
        for (field, actual, expected) in [
            ("kind", self.kind.as_str(), COLLECTIVE_EXECUTION_REPORT_KIND),
            (
                "contract_version",
                self.contract_version.as_str(),
                COLLECTIVE_EXECUTION_REPORT_CONTRACT_VERSION,
            ),
            (
                "semantic_owner",
                self.semantic_owner.as_str(),
                COLLECTIVE_EXECUTION_REPORT_SEMANTIC_OWNER,
            ),
            (
                "semantic_backend",
                self.semantic_backend.as_str(),
                COLLECTIVE_EXECUTION_REPORT_SEMANTIC_BACKEND,
            ),
        ] {
            if actual != expected {
                return Err(invalid_report(
                    field,
                    format!("must be {expected}, got {actual}"),
                ));
            }
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
                "successful execution reports must describe a committed operation",
            ));
        }

        let distributed_values = self
            .participants
            .checked_mul(self.width)
            .ok_or_else(|| invalid_report("output_values", "participant-width product overflow"))?;
        match self.operation {
            CollectiveOperation::AllReduceSum => {
                if self.participants == 0 {
                    return Err(invalid_report(
                        "participants",
                        "all-reduce requires at least one participant",
                    ));
                }
                if self.input_values != distributed_values
                    || self.output_values != distributed_values
                {
                    return Err(invalid_report(
                        "input_values",
                        "all-reduce counts must equal participants multiplied by width",
                    ));
                }
                if self.control_backend != "cpu_vec_buffers" {
                    return Err(invalid_report(
                        "control_backend",
                        "all-reduce control must remain cpu_vec_buffers",
                    ));
                }
                if self.width == 0 {
                    if self.requested_backend != "none"
                        || self.selected_backend != "none"
                        || self.numeric_kernel != "none"
                        || self.actual_numeric_backend_source != "none"
                    {
                        return Err(invalid_report(
                            "numeric_kernel",
                            "zero-width all-reduce must use the explicit none route",
                        ));
                    }
                } else {
                    validate_tensor_util_route(&self.requested_backend, &self.selected_backend)?;
                    if self.numeric_kernel != "tensor_sum_axis0"
                        || self.actual_numeric_backend_source != "sum_axis0_tensor_op_meta"
                    {
                        return Err(invalid_report(
                            "numeric_kernel",
                            "all-reduce must identify the Tensor sum_axis0 evidence source",
                        ));
                    }
                }
            }
            CollectiveOperation::Broadcast => {
                if self.input_values != self.width || self.output_values != distributed_values {
                    return Err(invalid_report(
                        "input_values",
                        "broadcast input is one root row and output is participants multiplied by width",
                    ));
                }
                if self.requested_backend != "cpu"
                    || self.selected_backend != "cpu"
                    || self.control_backend != "cpu_vec_buffers"
                    || self.numeric_kernel != "host_slice_clone"
                    || self.actual_numeric_backend_source != "host_slice_clone"
                {
                    return Err(invalid_report(
                        "selected_backend",
                        "broadcast must remain an explicit host slice clone",
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum CollectiveError {
    #[error("all-reduce requires at least one participant")]
    EmptyParticipants,
    #[error(
        "collective participant {participant} has width {got}, expected the canonical width {expected}"
    )]
    ShapeMismatch {
        participant: usize,
        expected: usize,
        got: usize,
    },
    #[error(
        "{operation} participant {participant} contained non-finite value at index {index}: {value}"
    )]
    NonFiniteInput {
        operation: &'static str,
        participant: usize,
        index: usize,
        value: f32,
    },
    #[error("all-reduce sum became non-finite at index {index}: {value}")]
    NonFiniteReduction { index: usize, value: f32 },
    #[error("gradient batches must contain at least one value")]
    EmptyGradient,
    #[error("collective value count overflow for {operation}")]
    CountOverflow { operation: &'static str },
    #[error("collective tensor operation {operation} failed: {source}")]
    TensorOperation {
        operation: &'static str,
        #[source]
        source: TensorError,
    },
    #[error("invalid collective execution report at {field}: {message}")]
    InvalidReport { field: String, message: String },
}

/// In-memory collective communication fabric with transactional mutation.
#[derive(Clone, Default, Debug)]
pub struct CollectiveArena {
    buffer: Arc<Mutex<VecDeque<Vec<f32>>>>,
}

#[derive(Debug)]
pub(super) struct GradientBatchLease {
    arena: CollectiveArena,
    batches: Option<Vec<Vec<f32>>>,
}

impl GradientBatchLease {
    pub(super) fn batches(&self) -> &[Vec<f32>] {
        self.batches.as_deref().unwrap_or(&[])
    }

    pub(super) fn is_empty(&self) -> bool {
        self.batches().is_empty()
    }

    pub(super) fn commit(mut self) {
        self.batches.take();
    }
}

impl Drop for GradientBatchLease {
    fn drop(&mut self) {
        if let Some(batches) = self.batches.take() {
            self.arena.restore_gradient_batches(batches);
        }
    }
}

impl CollectiveArena {
    pub fn new() -> Self {
        Self {
            buffer: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    fn lock_buffer(&self) -> MutexGuard<'_, VecDeque<Vec<f32>>> {
        match self.buffer.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                let guard = poisoned.into_inner();
                self.buffer.clear_poison();
                guard
            }
        }
    }

    /// Sums equal-width participant rows and commits the result to every row.
    pub fn all_reduce_sum(
        &self,
        tensors: &mut [Vec<f32>],
    ) -> Result<CollectiveExecutionReport, CollectiveError> {
        let Some(first) = tensors.first() else {
            return Err(CollectiveError::EmptyParticipants);
        };
        let width = first.len();
        for (participant, tensor) in tensors.iter().enumerate() {
            if tensor.len() != width {
                return Err(CollectiveError::ShapeMismatch {
                    participant,
                    expected: width,
                    got: tensor.len(),
                });
            }
            validate_finite_input("all_reduce_sum", participant, tensor)?;
        }
        let input_sha256 = rows_sha256(ALL_REDUCE_INPUT_DIGEST_DOMAIN, tensors);
        let total_values =
            tensors
                .len()
                .checked_mul(width)
                .ok_or(CollectiveError::CountOverflow {
                    operation: "all_reduce_sum",
                })?;

        let (outputs, requested_backend, selected_backend, numeric_kernel, evidence_source) =
            if width == 0 {
                (
                    vec![Vec::new(); tensors.len()],
                    "none",
                    "none",
                    "none",
                    "none",
                )
            } else {
                let route = current_tensor_util_route(total_values);
                let data = tensors
                    .iter()
                    .flat_map(|tensor| tensor.iter().copied())
                    .collect();
                let tensor = Tensor::from_vec(tensors.len(), width, data).map_err(|source| {
                    CollectiveError::TensorOperation {
                        operation: "all_reduce_input",
                        source,
                    }
                })?;
                let reduced = tensor
                    .try_sum_axis0_with_backend(route.selected_backend)
                    .map_err(|source| map_reduction_error(tensors, width, source))?;
                for (index, value) in reduced.iter().copied().enumerate() {
                    if !value.is_finite() {
                        return Err(CollectiveError::NonFiniteReduction { index, value });
                    }
                }
                (
                    vec![reduced; tensors.len()],
                    route.requested_backend_label(),
                    route.selected_backend_label(),
                    "tensor_sum_axis0",
                    "sum_axis0_tensor_op_meta",
                )
            };

        let output_sha256 = rows_sha256(ALL_REDUCE_OUTPUT_DIGEST_DOMAIN, &outputs);
        let report = build_report(
            CollectiveOperation::AllReduceSum,
            tensors.len(),
            width,
            total_values,
            total_values,
            requested_backend,
            selected_backend,
            "cpu_vec_buffers",
            numeric_kernel,
            evidence_source,
            input_sha256,
            output_sha256,
        )?;
        for (target, output) in tensors.iter_mut().zip(outputs) {
            *target = output;
        }
        emit_collective_report("collective_all_reduce_sum", &report);
        Ok(report)
    }

    /// Replaces every peer with the finite root payload after scratch preparation.
    pub fn broadcast(
        &self,
        root: &[f32],
        peers: &mut [Vec<f32>],
    ) -> Result<CollectiveExecutionReport, CollectiveError> {
        validate_finite_input("broadcast", 0, root)?;
        let output_values =
            peers
                .len()
                .checked_mul(root.len())
                .ok_or(CollectiveError::CountOverflow {
                    operation: "broadcast",
                })?;
        let outputs = vec![root.to_vec(); peers.len()];
        let report = build_report(
            CollectiveOperation::Broadcast,
            peers.len(),
            root.len(),
            root.len(),
            output_values,
            "cpu",
            "cpu",
            "cpu_vec_buffers",
            "host_slice_clone",
            "host_slice_clone",
            slice_sha256(BROADCAST_INPUT_DIGEST_DOMAIN, root),
            rows_sha256(BROADCAST_OUTPUT_DIGEST_DOMAIN, &outputs),
        )?;
        for (peer, output) in peers.iter_mut().zip(outputs) {
            *peer = output;
        }
        emit_collective_report("collective_broadcast", &report);
        Ok(report)
    }

    /// Stores one validated gradient batch for a later transactional merge.
    pub fn enqueue_gradient(&self, gradient: &[f32]) -> Result<(), CollectiveError> {
        if gradient.is_empty() {
            return Err(CollectiveError::EmptyGradient);
        }
        validate_finite_input("enqueue_gradient", 0, gradient)?;
        self.lock_buffer().push_back(gradient.to_vec());
        Ok(())
    }

    /// Flattens all pending batches into `out` and commits the drain atomically.
    pub fn drain_gradients(&self, out: &mut Vec<f32>) -> Result<(), CollectiveError> {
        let lease = self.lease_gradient_batches();
        let mut next = out.clone();
        for (batch, gradient) in lease.batches().iter().enumerate() {
            if gradient.is_empty() {
                return Err(CollectiveError::EmptyGradient);
            }
            validate_finite_input("drain_gradients", batch, gradient)?;
            next.extend_from_slice(gradient);
        }
        *out = next;
        lease.commit();
        Ok(())
    }

    pub(super) fn lease_gradient_batches(&self) -> GradientBatchLease {
        let batches = self.lock_buffer().drain(..).collect();
        GradientBatchLease {
            arena: self.clone(),
            batches: Some(batches),
        }
    }

    pub(super) fn restore_gradient_batches(&self, gradients: Vec<Vec<f32>>) {
        if gradients.is_empty() {
            return;
        }
        let mut guard = self.lock_buffer();
        let mut restored = VecDeque::from(gradients);
        restored.append(&mut guard);
        *guard = restored;
    }

    pub(super) fn queued_gradient_batches(&self) -> usize {
        self.lock_buffer().len()
    }
}

#[allow(clippy::too_many_arguments)]
fn build_report(
    operation: CollectiveOperation,
    participants: usize,
    width: usize,
    input_values: usize,
    output_values: usize,
    requested_backend: &str,
    selected_backend: &str,
    control_backend: &str,
    numeric_kernel: &str,
    evidence_source: &str,
    input_sha256: String,
    output_sha256: String,
) -> Result<CollectiveExecutionReport, CollectiveError> {
    let report = CollectiveExecutionReport {
        kind: COLLECTIVE_EXECUTION_REPORT_KIND.to_owned(),
        contract_version: COLLECTIVE_EXECUTION_REPORT_CONTRACT_VERSION.to_owned(),
        semantic_owner: COLLECTIVE_EXECUTION_REPORT_SEMANTIC_OWNER.to_owned(),
        semantic_backend: COLLECTIVE_EXECUTION_REPORT_SEMANTIC_BACKEND.to_owned(),
        operation,
        participants: count_u64("participants", participants)?,
        width: count_u64("width", width)?,
        input_values: count_u64("input_values", input_values)?,
        output_values: count_u64("output_values", output_values)?,
        requested_backend: requested_backend.to_owned(),
        selected_backend: selected_backend.to_owned(),
        control_backend: control_backend.to_owned(),
        numeric_kernel: numeric_kernel.to_owned(),
        actual_numeric_backend_source: evidence_source.to_owned(),
        input_sha256,
        output_sha256,
        committed: true,
    };
    report.validate()?;
    Ok(report)
}

fn emit_collective_report(op_name: &'static str, report: &CollectiveExecutionReport) {
    let input_participants = match report.operation {
        CollectiveOperation::AllReduceSum => report.participants,
        CollectiveOperation::Broadcast => 1,
    };
    emit_tensor_op(
        op_name,
        &[
            usize_from_count(input_participants).max(1),
            usize_from_count(report.width).max(1),
        ],
        &[
            usize_from_count(report.participants).max(1),
            usize_from_count(report.width).max(1),
        ],
    );
    if tensor_op_meta_observer_installed() {
        emit_tensor_op_meta(op_name, || {
            let mut value = serde_json::to_value(report)
                .expect("CollectiveExecutionReport serialization cannot fail");
            if let Some(object) = value.as_object_mut() {
                object.insert(
                    "backend".to_owned(),
                    serde_json::Value::String("cpu".to_owned()),
                );
                object.insert(
                    "numeric_backend_semantics".to_owned(),
                    serde_json::Value::String("selected_route_or_explicit_host".to_owned()),
                );
            }
            value
        });
    }
}

fn validate_tensor_util_route(requested: &str, selected: &str) -> Result<(), CollectiveError> {
    if matches!(
        (requested, selected),
        ("auto", "auto") | ("cpu", "cpu") | ("wgpu", "cpu") | ("wgpu", "wgpu")
    ) {
        Ok(())
    } else {
        Err(invalid_report(
            "selected_backend",
            format!(
                "requested/selected route {requested}/{selected} cannot be produced by the Rust-owned tensor utility planner"
            ),
        ))
    }
}

fn validate_finite_input(
    operation: &'static str,
    participant: usize,
    values: &[f32],
) -> Result<(), CollectiveError> {
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(CollectiveError::NonFiniteInput {
                operation,
                participant,
                index,
                value,
            });
        }
    }
    Ok(())
}

fn map_reduction_error(rows: &[Vec<f32>], width: usize, source: TensorError) -> CollectiveError {
    if matches!(source, TensorError::NonFiniteValue { .. }) {
        for index in 0..width {
            let mut sum = 0.0f32;
            for row in rows {
                sum += row[index];
                if !sum.is_finite() {
                    return CollectiveError::NonFiniteReduction { index, value: sum };
                }
            }
        }
    }
    CollectiveError::TensorOperation {
        operation: "all_reduce_sum",
        source,
    }
}

fn rows_sha256(domain: &[u8], rows: &[Vec<f32>]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hash_u64(&mut hasher, count_u64_infallible(rows.len()));
    for row in rows {
        hash_f32_slice(&mut hasher, row);
    }
    finish_sha256(hasher)
}

fn slice_sha256(domain: &[u8], values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hash_f32_slice(&mut hasher, values);
    finish_sha256(hasher)
}

fn hash_f32_slice(hasher: &mut Sha256, values: &[f32]) {
    hash_u64(hasher, count_u64_infallible(values.len()));
    for value in values {
        hasher.update(value.to_bits().to_be_bytes());
    }
}

fn hash_u64(hasher: &mut Sha256, value: u64) {
    hasher.update(value.to_be_bytes());
}

fn finish_sha256(hasher: Sha256) -> String {
    let digest = hasher.finalize();
    let mut hexadecimal = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut hexadecimal, "{byte:02x}").expect("writing to a String cannot fail");
    }
    hexadecimal
}

fn valid_sha256(digest: &str) -> bool {
    digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn count_u64(operation: &'static str, value: usize) -> Result<u64, CollectiveError> {
    u64::try_from(value).map_err(|_| CollectiveError::CountOverflow { operation })
}

fn count_u64_infallible(value: usize) -> u64 {
    u64::try_from(value).expect("supported Rust targets have at most 64-bit usize")
}

fn usize_from_count(value: u64) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

fn invalid_report(field: &str, message: impl ToString) -> CollectiveError {
    CollectiveError::InvalidReport {
        field: field.to_owned(),
        message: message.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn all_reduce_sums_values_and_returns_a_valid_route_report() {
        let arena = CollectiveArena::new();
        let mut tensors = vec![vec![1.0, 2.0], vec![0.5, -1.0]];
        let before = tensors.clone();
        let report = arena.all_reduce_sum(&mut tensors).unwrap();

        assert_eq!(tensors, vec![vec![1.5, 1.0], vec![1.5, 1.0]]);
        assert_eq!(report.operation, CollectiveOperation::AllReduceSum);
        assert_eq!(report.participants, 2);
        assert_eq!(report.width, 2);
        assert_eq!(report.requested_backend, "auto");
        assert_eq!(report.selected_backend, "auto");
        assert_eq!(
            report.input_sha256,
            rows_sha256(ALL_REDUCE_INPUT_DIGEST_DOMAIN, &before)
        );
        assert_eq!(
            report.output_sha256,
            rows_sha256(ALL_REDUCE_OUTPUT_DIGEST_DOMAIN, &tensors)
        );
        report.validate().unwrap();

        let wire = serde_json::to_value(&report).unwrap();
        assert_eq!(wire["participants"], serde_json::json!("2"));
        assert_eq!(wire["input_values"], serde_json::json!("4"));
        assert_eq!(
            serde_json::from_value::<CollectiveExecutionReport>(wire.clone()).unwrap(),
            report
        );

        let mut numeric_count = wire.clone();
        numeric_count["participants"] = serde_json::json!(2);
        assert!(serde_json::from_value::<CollectiveExecutionReport>(numeric_count).is_err());

        let mut unknown_field = wire;
        unknown_field["python_route"] = serde_json::json!("exploratory");
        assert!(serde_json::from_value::<CollectiveExecutionReport>(unknown_field).is_err());

        let mut invalid_count = report.clone();
        invalid_count.output_values += 1;
        assert!(matches!(
            invalid_count.validate(),
            Err(CollectiveError::InvalidReport { .. })
        ));

        let mut invalid_route = report.clone();
        invalid_route.selected_backend = "python".to_owned();
        assert!(matches!(
            invalid_route.validate(),
            Err(CollectiveError::InvalidReport { .. })
        ));

        let mut impossible_route = report.clone();
        impossible_route.requested_backend = "cpu".to_owned();
        impossible_route.selected_backend = "wgpu".to_owned();
        assert!(matches!(
            impossible_route.validate(),
            Err(CollectiveError::InvalidReport { .. })
        ));

        let mut invalid_digest = report;
        invalid_digest.output_sha256.make_ascii_uppercase();
        assert!(matches!(
            invalid_digest.validate(),
            Err(CollectiveError::InvalidReport { .. })
        ));
    }

    #[test]
    fn all_reduce_rejects_empty_or_mismatched_participants_without_mutation() {
        let arena = CollectiveArena::new();
        let mut empty = Vec::<Vec<f32>>::new();
        assert_eq!(
            arena.all_reduce_sum(&mut empty).unwrap_err(),
            CollectiveError::EmptyParticipants
        );

        for mut tensors in [
            vec![vec![1.0, 2.0], vec![3.0]],
            vec![vec![1.0], vec![2.0, 3.0]],
        ] {
            let before = tensors.clone();
            assert!(matches!(
                arena.all_reduce_sum(&mut tensors),
                Err(CollectiveError::ShapeMismatch { .. })
            ));
            assert_eq!(tensors, before);
        }
    }

    #[test]
    fn all_reduce_rejects_non_finite_and_overflow_without_mutation() {
        let arena = CollectiveArena::new();
        let mut non_finite = vec![vec![1.0, f32::NAN], vec![2.0, 3.0]];
        let before = non_finite.clone();
        assert!(matches!(
            arena.all_reduce_sum(&mut non_finite),
            Err(CollectiveError::NonFiniteInput { .. })
        ));
        assert_eq!(non_finite[0][0], before[0][0]);
        assert!(non_finite[0][1].is_nan());
        assert_eq!(non_finite[1], before[1]);

        let mut overflow = vec![vec![f32::MAX], vec![f32::MAX]];
        let before = overflow.clone();
        assert!(matches!(
            arena.all_reduce_sum(&mut overflow),
            Err(CollectiveError::NonFiniteReduction { .. })
                | Err(CollectiveError::TensorOperation { .. })
        ));
        assert_eq!(overflow, before);
    }

    #[test]
    fn zero_width_all_reduce_is_explicit_and_committed() {
        let arena = CollectiveArena::new();
        let mut tensors = vec![Vec::new(), Vec::new()];
        let report = arena.all_reduce_sum(&mut tensors).unwrap();

        assert_eq!(report.requested_backend, "none");
        assert_eq!(report.selected_backend, "none");
        assert_eq!(report.input_values, 0);
        assert_eq!(tensors, vec![Vec::<f32>::new(), Vec::new()]);
    }

    #[test]
    fn broadcast_clones_root_and_rejects_non_finite_without_mutation() {
        let arena = CollectiveArena::new();
        let mut peers = vec![vec![9.0], vec![8.0, 7.0]];
        let report = arena.broadcast(&[1.0, 2.0, 3.0], &mut peers).unwrap();
        assert_eq!(peers, vec![vec![1.0, 2.0, 3.0]; 2]);
        assert_eq!(report.operation, CollectiveOperation::Broadcast);
        assert_eq!(report.selected_backend, "cpu");
        report.validate().unwrap();

        let before = peers.clone();
        assert!(matches!(
            arena.broadcast(&[1.0, f32::NAN], &mut peers),
            Err(CollectiveError::NonFiniteInput { .. })
        ));
        assert_eq!(peers, before);
    }

    #[test]
    fn collective_report_metadata_is_emitted_after_commit() {
        let _lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));
        let arena = CollectiveArena::new();
        let mut tensors = vec![vec![1.0], vec![2.0]];
        let report = arena.all_reduce_sum(&mut tensors).unwrap();
        st_tensor::set_thread_meta_observer(previous);

        let events = events.lock().unwrap();
        let event = events
            .iter()
            .find(|(op_name, _)| *op_name == "collective_all_reduce_sum")
            .expect("collective all-reduce report event");
        assert_eq!(event.1["output_sha256"], report.output_sha256);
        assert_eq!(event.1["committed"], true);
        assert_eq!(event.1["participants"], "2");
        assert_eq!(event.1["backend"], "cpu");
        assert_eq!(event.1["selected_backend"], "auto");
        assert_eq!(
            event.1["actual_numeric_backend_source"],
            "sum_axis0_tensor_op_meta"
        );
    }

    #[test]
    fn gradient_queue_validates_and_commits_flat_drains() {
        let arena = CollectiveArena::new();
        assert_eq!(
            arena.enqueue_gradient(&[]).unwrap_err(),
            CollectiveError::EmptyGradient
        );
        assert!(matches!(
            arena.enqueue_gradient(&[f32::NAN]),
            Err(CollectiveError::NonFiniteInput { .. })
        ));
        arena.enqueue_gradient(&[1.0, 2.0]).unwrap();
        arena.enqueue_gradient(&[3.0]).unwrap();
        assert_eq!(arena.queued_gradient_batches(), 2);
        let mut out = vec![-1.0];
        arena.drain_gradients(&mut out).unwrap();
        assert_eq!(out, vec![-1.0, 1.0, 2.0, 3.0]);
        assert_eq!(arena.queued_gradient_batches(), 0);
    }

    #[test]
    fn dropped_gradient_lease_restores_before_concurrent_enqueues() {
        let arena = CollectiveArena::new();
        arena.enqueue_gradient(&[1.0, 2.0]).unwrap();
        let lease = arena.lease_gradient_batches();
        let concurrent_arena = arena.clone();
        std::thread::spawn(move || concurrent_arena.enqueue_gradient(&[3.0, 4.0]).unwrap())
            .join()
            .unwrap();
        drop(lease);

        let lease = arena.lease_gradient_batches();
        assert_eq!(lease.batches(), &[vec![1.0, 2.0], vec![3.0, 4.0]]);
        lease.commit();
        assert_eq!(arena.queued_gradient_batches(), 0);
    }

    #[test]
    fn panicking_gradient_lease_restores_batches() {
        let arena = CollectiveArena::new();
        arena.enqueue_gradient(&[1.0, 2.0]).unwrap();
        let lease = arena.lease_gradient_batches();

        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _lease = lease;
            panic!("abort asynchronous gradient merge");
        }));
        assert!(outcome.is_err());

        arena.enqueue_gradient(&[3.0, 4.0]).unwrap();
        let lease = arena.lease_gradient_batches();
        assert_eq!(lease.batches(), &[vec![1.0, 2.0], vec![3.0, 4.0]]);
        lease.commit();
        assert_eq!(arena.queued_gradient_batches(), 0);
    }

    #[test]
    fn failed_flat_drain_preserves_output_and_queue() {
        let arena = CollectiveArena::new();
        arena.restore_gradient_batches(vec![vec![1.0], vec![f32::NAN]]);
        let mut out = vec![9.0];
        assert!(matches!(
            arena.drain_gradients(&mut out),
            Err(CollectiveError::NonFiniteInput { .. })
        ));
        assert_eq!(out, vec![9.0]);

        let lease = arena.lease_gradient_batches();
        assert_eq!(lease.batches()[0], vec![1.0]);
        assert!(lease.batches()[1][0].is_nan());
        lease.commit();
    }

    #[test]
    fn gradient_queue_recovers_after_mutex_poison() {
        let arena = CollectiveArena::new();
        let buffer = arena.buffer.clone();
        let _ = std::thread::spawn(move || {
            let _guard = buffer.lock().unwrap();
            panic!("poison collective queue");
        })
        .join();

        arena.enqueue_gradient(&[1.0, 2.0]).unwrap();
        let lease = arena.lease_gradient_batches();
        assert_eq!(lease.batches(), &[vec![1.0, 2.0]]);
        lease.commit();
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn wgpu_selected_all_reduce_matches_cpu() {
        use crate::backend::device_caps::DeviceCaps;
        use crate::backend::execution::{
            push_backend_policy, AcceleratorFallback, BackendPolicy, ExecutionConfig,
        };

        let mut cpu = vec![vec![1.0, -2.0], vec![0.5, 3.0]];
        let cpu_policy = BackendPolicy::from_device_caps_with_config(
            DeviceCaps::cpu(),
            ExecutionConfig::default(),
        );
        let cpu_report = {
            let _guard = push_backend_policy(cpu_policy);
            CollectiveArena::new().all_reduce_sum(&mut cpu).unwrap()
        };

        let mut wgpu = vec![vec![1.0, -2.0], vec![0.5, 3.0]];
        let wgpu_policy = BackendPolicy::from_device_caps_with_config(
            DeviceCaps::wgpu(32, true, 256),
            ExecutionConfig::new(AcceleratorFallback::Allow, 1),
        );
        let wgpu_report = {
            let _guard = push_backend_policy(wgpu_policy);
            CollectiveArena::new().all_reduce_sum(&mut wgpu).unwrap()
        };

        assert_eq!(cpu, wgpu);
        assert_eq!(cpu_report.selected_backend, "cpu");
        assert_eq!(wgpu_report.requested_backend, "wgpu");
        assert_eq!(wgpu_report.selected_backend, "wgpu");
        assert_eq!(cpu_report.output_sha256, wgpu_report.output_sha256);
    }
}
