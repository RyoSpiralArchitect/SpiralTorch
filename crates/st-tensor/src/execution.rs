// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Tensor-level application of an already resolved execution contract.

use crate::execution_capability::{
    TensorExecutionBackend, TensorExecutionComponent, TensorExecutionWorkload,
    TensorExecutionWorkloadKey,
};
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::fmt;
use std::marker::PhantomData;
use std::rc::Rc;

pub use spiral_config::execution::AcceleratorFallback;

pub const TENSOR_EXECUTION_RECEIPT_KIND: &str = "spiraltorch.tensor_execution_receipt";
pub const TENSOR_EXECUTION_RECEIPT_CONTRACT_VERSION: &str =
    "spiraltorch.tensor_execution_receipt.v1";
pub const TENSOR_EXECUTION_RECEIPT_SEMANTIC_OWNER: &str = "st-tensor::execution";

thread_local! {
    static ACTIVE_ACCELERATOR_FALLBACK: Cell<Option<AcceleratorFallback>> = const { Cell::new(None) };
    static ACTIVE_EXECUTION_PLAN: Cell<Option<TensorExecutionPlanBinding>> = const { Cell::new(None) };
}

/// Stable outcome vocabulary for one completed tensor dispatch.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorExecutionRouteStatus {
    Direct,
    AutoResolved,
    CpuThreshold,
    RuntimeFallback,
    NoOp,
}

impl TensorExecutionRouteStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::AutoResolved => "auto_resolved",
            Self::CpuThreshold => "cpu_threshold",
            Self::RuntimeFallback => "runtime_fallback",
            Self::NoOp => "no_op",
        }
    }
}

/// Concrete kernel family that produced a completed tensor output.
///
/// Policy routing uses [`TensorExecutionBackend::Wgpu`], while the current
/// implementation is the more precise `wgpu_dense` kernel family. Keeping
/// both labels in the receipt prevents transport clients from reconstructing
/// implementation telemetry from a canonical route backend.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorExecutionKernelBackend {
    Cpu,
    CpuSimd,
    Naive,
    Faer,
    WgpuDense,
    Hip,
}

impl TensorExecutionKernelBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::CpuSimd => "cpu_simd",
            Self::Naive => "naive",
            Self::Faer => "faer",
            Self::WgpuDense => "wgpu_dense",
            Self::Hip => "hip",
        }
    }

    const fn for_executed_backend(backend: TensorExecutionBackend) -> Option<Self> {
        match backend {
            TensorExecutionBackend::Auto => None,
            TensorExecutionBackend::Cpu => Some(Self::Cpu),
            TensorExecutionBackend::CpuSimd => Some(Self::CpuSimd),
            TensorExecutionBackend::Naive => Some(Self::Naive),
            TensorExecutionBackend::Faer => Some(Self::Faer),
            TensorExecutionBackend::Wgpu => Some(Self::WgpuDense),
            TensorExecutionBackend::Hip => Some(Self::Hip),
        }
    }

    const fn executed_backend(self) -> TensorExecutionBackend {
        match self {
            Self::Cpu => TensorExecutionBackend::Cpu,
            Self::CpuSimd => TensorExecutionBackend::CpuSimd,
            Self::Naive => TensorExecutionBackend::Naive,
            Self::Faer => TensorExecutionBackend::Faer,
            Self::WgpuDense => TensorExecutionBackend::Wgpu,
            Self::Hip => TensorExecutionBackend::Hip,
        }
    }
}

/// Stable reason vocabulary for a successful fallback completion.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorExecutionFallbackReason {
    RuntimeUnavailable,
}

/// Typed evidence that an accelerator request completed on another backend.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorExecutionFallback {
    pub from: TensorExecutionBackend,
    pub to: TensorExecutionBackend,
    pub reason: TensorExecutionFallbackReason,
}

impl TensorExecutionFallback {
    pub const fn runtime_unavailable(
        from: TensorExecutionBackend,
        to: TensorExecutionBackend,
    ) -> Self {
        Self {
            from,
            to,
            reason: TensorExecutionFallbackReason::RuntimeUnavailable,
        }
    }
}

/// Rust-owned proof for one logical tensor operation that returned successfully.
///
/// A receipt is emitted only after accelerator readback and output validation.
/// A committed-plan hash binds the completion to the policy active on the same
/// thread; direct Rust calls remain valid without such a commitment.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorExecutionReceipt {
    pub kind: String,
    pub contract_version: String,
    pub semantic_owner: String,
    pub component: TensorExecutionComponent,
    pub operation: String,
    pub workload: TensorExecutionWorkload,
    pub requested_backend: TensorExecutionBackend,
    pub selected_backend: TensorExecutionBackend,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executed_backend: Option<TensorExecutionBackend>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel_backend: Option<TensorExecutionKernelBackend>,
    pub route_status: TensorExecutionRouteStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback: Option<TensorExecutionFallback>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_execution_plan_output_sha256: Option<String>,
}

impl TensorExecutionReceipt {
    pub fn validate(&self) -> Result<(), TensorExecutionContractError> {
        if self.kind != TENSOR_EXECUTION_RECEIPT_KIND {
            return Err(TensorExecutionContractError::InvalidReceipt {
                field: "kind",
                message: "unsupported receipt kind",
            });
        }
        if self.contract_version != TENSOR_EXECUTION_RECEIPT_CONTRACT_VERSION {
            return Err(TensorExecutionContractError::InvalidReceipt {
                field: "contract_version",
                message: "unsupported receipt contract version",
            });
        }
        if self.semantic_owner != TENSOR_EXECUTION_RECEIPT_SEMANTIC_OWNER {
            return Err(TensorExecutionContractError::InvalidReceipt {
                field: "semantic_owner",
                message: "receipt semantics are not owned by st-tensor",
            });
        }
        if !valid_operation_name(&self.operation) {
            return Err(TensorExecutionContractError::InvalidReceipt {
                field: "operation",
                message: "operation must be a non-empty snake-case identifier",
            });
        }
        if self.workload.component() != self.component {
            return Err(TensorExecutionContractError::InvalidReceipt {
                field: "workload",
                message: "workload component does not match the receipt component",
            });
        }
        if !operation_matches_workload(&self.operation, self.workload) {
            return Err(TensorExecutionContractError::InvalidReceipt {
                field: "operation",
                message: "operation does not match the declared workload",
            });
        }
        self.require_supported_backend("requested_backend", self.requested_backend)?;
        self.require_supported_backend("selected_backend", self.selected_backend)?;
        if let Some(executed) = self.executed_backend {
            self.require_supported_backend("executed_backend", executed)?;
        }
        match (self.executed_backend, self.kernel_backend) {
            (Some(executed), Some(kernel)) if kernel.executed_backend() == executed => {}
            (Some(_), Some(_)) => {
                return Err(TensorExecutionContractError::InvalidReceipt {
                    field: "kernel_backend",
                    message: "kernel backend does not implement the executed route backend",
                });
            }
            (Some(_), None) => {
                return Err(TensorExecutionContractError::InvalidReceipt {
                    field: "kernel_backend",
                    message: "completed execution requires a concrete kernel backend",
                });
            }
            (None, Some(_)) => {
                return Err(TensorExecutionContractError::InvalidReceipt {
                    field: "kernel_backend",
                    message: "a no-op cannot claim a concrete kernel backend",
                });
            }
            (None, None) => {}
        }
        if let Some(commitment) = self.runtime_execution_plan_output_sha256.as_deref() {
            if !valid_sha256_hex(commitment) {
                return Err(TensorExecutionContractError::InvalidReceipt {
                    field: "runtime_execution_plan_output_sha256",
                    message: "commitment must be 64 lowercase hexadecimal characters",
                });
            }
        }
        let output_is_empty = self.workload.has_empty_output();
        if output_is_empty != (self.route_status == TensorExecutionRouteStatus::NoOp) {
            return Err(invalid_route(
                "empty outputs must be no-op and no-op receipts must have empty outputs",
            ));
        }

        match self.route_status {
            TensorExecutionRouteStatus::Direct => {
                self.require_no_fallback()?;
                self.require_executed(self.selected_backend)?;
                if self.requested_backend != self.selected_backend
                    || self.selected_backend == TensorExecutionBackend::Auto
                {
                    return Err(invalid_route(
                        "direct completion requires one explicit backend throughout",
                    ));
                }
            }
            TensorExecutionRouteStatus::AutoResolved => {
                self.require_no_fallback()?;
                self.require_executed(self.selected_backend)?;
                if self.requested_backend != TensorExecutionBackend::Auto
                    || self.selected_backend == TensorExecutionBackend::Auto
                {
                    return Err(invalid_route(
                        "auto resolution must select and execute one concrete backend",
                    ));
                }
            }
            TensorExecutionRouteStatus::CpuThreshold => {
                self.require_no_fallback()?;
                self.require_executed(TensorExecutionBackend::Cpu)?;
                if self.requested_backend != TensorExecutionBackend::Wgpu
                    || self.selected_backend != TensorExecutionBackend::Cpu
                    || self.runtime_execution_plan_output_sha256.is_none()
                {
                    return Err(invalid_route(
                        "threshold routing requires a committed WGPU plan and CPU selection",
                    ));
                }
            }
            TensorExecutionRouteStatus::RuntimeFallback => {
                let executed = self.executed_backend.ok_or_else(|| {
                    invalid_route("runtime fallback requires an executed backend")
                })?;
                let fallback = self
                    .fallback
                    .ok_or_else(|| invalid_route("runtime fallback requires typed evidence"))?;
                if self.requested_backend != self.selected_backend
                    || self.selected_backend == TensorExecutionBackend::Auto
                    || executed == self.selected_backend
                    || fallback.from != self.selected_backend
                    || fallback.to != executed
                    || !is_accelerator_backend(self.selected_backend)
                    || !is_host_backend(executed)
                {
                    return Err(invalid_route(
                        "runtime fallback must form one accelerator-to-host route",
                    ));
                }
            }
            TensorExecutionRouteStatus::NoOp => {
                self.require_no_fallback()?;
                if self.executed_backend.is_some() {
                    return Err(invalid_route("a no-op cannot claim an executed backend"));
                }
                let direct_route = self.requested_backend == self.selected_backend;
                let threshold_route = self.requested_backend == TensorExecutionBackend::Wgpu
                    && self.selected_backend == TensorExecutionBackend::Cpu
                    && self.runtime_execution_plan_output_sha256.is_some();
                if !direct_route && !threshold_route {
                    return Err(invalid_route(
                        "a no-op route must be direct or a committed CPU threshold decision",
                    ));
                }
            }
        }
        Ok(())
    }

    fn require_no_fallback(&self) -> Result<(), TensorExecutionContractError> {
        if self.fallback.is_some() {
            return Err(invalid_route("route must not include fallback evidence"));
        }
        Ok(())
    }

    fn require_supported_backend(
        &self,
        field: &'static str,
        backend: TensorExecutionBackend,
    ) -> Result<(), TensorExecutionContractError> {
        if !backend.supports_component(self.component) {
            return Err(TensorExecutionContractError::InvalidReceipt {
                field,
                message: "backend is not implemented for the declared component",
            });
        }
        Ok(())
    }

    fn require_executed(
        &self,
        expected: TensorExecutionBackend,
    ) -> Result<(), TensorExecutionContractError> {
        if self.executed_backend != Some(expected) {
            return Err(invalid_route("executed backend does not match the route"));
        }
        Ok(())
    }
}

/// Committed tensor policy projected from the Rust runtime execution plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorExecutionPlanBinding {
    output_sha256: [u8; 32],
    backends: [TensorExecutionBackend; 6],
    workloads: [Option<TensorExecutionWorkload>; TensorExecutionWorkloadKey::COUNT],
    accelerator_fallback: AcceleratorFallback,
    tensor_util_wgpu_min_values: usize,
}

impl TensorExecutionPlanBinding {
    pub const fn new(
        output_sha256: [u8; 32],
        backends: [TensorExecutionBackend; 6],
        accelerator_fallback: AcceleratorFallback,
        tensor_util_wgpu_min_values: usize,
    ) -> Self {
        Self {
            output_sha256,
            backends,
            workloads: [None; TensorExecutionWorkloadKey::COUNT],
            accelerator_fallback,
            tensor_util_wgpu_min_values,
        }
    }

    /// Creates a binding that additionally constrains declared component workloads.
    ///
    /// Workloads are indexed by their Rust-owned operation identity. Missing
    /// operation kinds remain deliberately unbound for deferred operation-time
    /// capability checks, while duplicate operation kinds fail closed.
    pub fn try_new_with_workloads(
        output_sha256: [u8; 32],
        backends: [TensorExecutionBackend; 6],
        workloads: impl IntoIterator<Item = TensorExecutionWorkload>,
        accelerator_fallback: AcceleratorFallback,
        tensor_util_wgpu_min_values: usize,
    ) -> Result<Self, TensorExecutionContractError> {
        let mut binding = Self::new(
            output_sha256,
            backends,
            accelerator_fallback,
            tensor_util_wgpu_min_values,
        );
        for workload in workloads {
            let component = workload.component();
            let key = workload.key();
            let slot = &mut binding.workloads[key.index()];
            if slot.replace(workload).is_some() {
                return Err(TensorExecutionContractError::DuplicatePlanWorkload { component, key });
            }
        }
        Ok(binding)
    }

    pub const fn output_sha256(self) -> [u8; 32] {
        self.output_sha256
    }

    pub const fn backend_for(self, component: TensorExecutionComponent) -> TensorExecutionBackend {
        self.backends[component.index()]
    }

    /// Returns the committed workload for the actual operation kind, when declared.
    pub const fn planned_workload_for(
        self,
        actual: TensorExecutionWorkload,
    ) -> Option<TensorExecutionWorkload> {
        self.workloads[actual.key().index()]
    }

    /// Whether this plan closes over an exact workload set for `component`.
    pub fn binds_component(self, component: TensorExecutionComponent) -> bool {
        self.workloads
            .into_iter()
            .flatten()
            .any(|workload| workload.component() == component)
    }

    /// Validate that a self-consistent receipt was authorized by this binding.
    ///
    /// This reapplies the same workload, backend, threshold, and fallback
    /// contract used before dispatch. A receipt cannot authorize itself merely
    /// by carrying a syntactically valid plan commitment.
    pub fn validate_receipt(
        self,
        receipt: &TensorExecutionReceipt,
    ) -> Result<(), TensorExecutionContractError> {
        receipt.validate()?;

        let planned_commitment = sha256_hex(self.output_sha256);
        let Some(receipt_commitment) = receipt.runtime_execution_plan_output_sha256.as_deref()
        else {
            return Err(TensorExecutionContractError::ReceiptPlanCommitmentMissing);
        };
        if receipt_commitment != planned_commitment {
            return Err(
                TensorExecutionContractError::ReceiptPlanCommitmentMismatch {
                    planned: planned_commitment,
                    receipt: receipt_commitment.to_owned(),
                },
            );
        }

        let component = receipt.component;
        if let Some(planned_workload) = self.planned_workload_for(receipt.workload) {
            if receipt.workload != planned_workload {
                return Err(TensorExecutionContractError::ReceiptPlanWorkloadMismatch {
                    component,
                    planned: planned_workload,
                    receipt: receipt.workload,
                });
            }
        } else if self.binds_component(component) {
            return Err(
                TensorExecutionContractError::ReceiptPlanWorkloadNotAuthorized {
                    component,
                    receipt: receipt.workload,
                },
            );
        }

        let planned_backend = self.backend_for(component);
        if receipt.requested_backend != planned_backend {
            return Err(TensorExecutionContractError::ReceiptPlanBackendMismatch {
                field: "requested_backend",
                component,
                planned: planned_backend,
                receipt: receipt.requested_backend,
            });
        }

        let threshold_cpu_route = self.uses_tensor_util_cpu_threshold(receipt.workload);
        let planned_selected_backend = self.selected_backend_for(receipt.workload);
        if threshold_cpu_route {
            if receipt.selected_backend != TensorExecutionBackend::Cpu {
                return Err(TensorExecutionContractError::ReceiptPlanBackendMismatch {
                    field: "selected_backend",
                    component,
                    planned: TensorExecutionBackend::Cpu,
                    receipt: receipt.selected_backend,
                });
            }
        } else if planned_selected_backend != TensorExecutionBackend::Auto
            && receipt.selected_backend != planned_selected_backend
        {
            return Err(TensorExecutionContractError::ReceiptPlanBackendMismatch {
                field: "selected_backend",
                component,
                planned: planned_selected_backend,
                receipt: receipt.selected_backend,
            });
        }

        let expected_route = if receipt.workload.has_empty_output() {
            PlannedReceiptRoute::NoOp
        } else if threshold_cpu_route {
            PlannedReceiptRoute::CpuThreshold
        } else if planned_backend == TensorExecutionBackend::Auto {
            PlannedReceiptRoute::AutoResolved
        } else if self.accelerator_fallback.allows_fallback() {
            PlannedReceiptRoute::DirectOrRuntimeFallback
        } else {
            PlannedReceiptRoute::Direct
        };
        if !expected_route.accepts(receipt.route_status) {
            return Err(TensorExecutionContractError::ReceiptPlanRouteMismatch {
                component,
                expected: expected_route.as_str(),
                receipt: receipt.route_status,
            });
        }

        Ok(())
    }

    fn uses_tensor_util_cpu_threshold(self, workload: TensorExecutionWorkload) -> bool {
        workload.component() == TensorExecutionComponent::TensorUtil
            && self.backend_for(TensorExecutionComponent::TensorUtil)
                == TensorExecutionBackend::Wgpu
            && workload.output_values_saturating() < self.tensor_util_wgpu_min_values
            && self.accelerator_fallback.allows_fallback()
    }

    fn selected_backend_for(self, workload: TensorExecutionWorkload) -> TensorExecutionBackend {
        if self.uses_tensor_util_cpu_threshold(workload) {
            TensorExecutionBackend::Cpu
        } else {
            self.backend_for(workload.component())
        }
    }

    pub const fn accelerator_fallback(self) -> AcceleratorFallback {
        self.accelerator_fallback
    }

    pub const fn tensor_util_wgpu_min_values(self) -> usize {
        self.tensor_util_wgpu_min_values
    }
}

/// RAII guard that restores the previous tensor fallback contract.
#[derive(Debug)]
pub struct AcceleratorFallbackGuard {
    previous: Option<AcceleratorFallback>,
    _not_send: PhantomData<Rc<()>>,
}

impl Drop for AcceleratorFallbackGuard {
    fn drop(&mut self) {
        ACTIVE_ACCELERATOR_FALLBACK.with(|slot| slot.set(self.previous));
    }
}

/// RAII guard that restores the previous committed execution-plan binding.
#[derive(Debug)]
pub struct TensorExecutionPlanGuard {
    previous: Option<TensorExecutionPlanBinding>,
    _not_send: PhantomData<Rc<()>>,
}

impl Drop for TensorExecutionPlanGuard {
    fn drop(&mut self) {
        ACTIVE_EXECUTION_PLAN.with(|slot| slot.set(self.previous));
    }
}

/// Installs a resolved fallback contract for tensor operations on this thread.
pub fn push_accelerator_fallback(fallback: AcceleratorFallback) -> AcceleratorFallbackGuard {
    let previous = ACTIVE_ACCELERATOR_FALLBACK.with(|slot| slot.replace(Some(fallback)));
    AcceleratorFallbackGuard {
        previous,
        _not_send: PhantomData,
    }
}

/// Binds a committed execution plan to tensor dispatches on this thread.
pub fn push_execution_plan_binding(
    binding: TensorExecutionPlanBinding,
) -> TensorExecutionPlanGuard {
    let previous = ACTIVE_EXECUTION_PLAN.with(|slot| slot.replace(Some(binding)));
    TensorExecutionPlanGuard {
        previous,
        _not_send: PhantomData,
    }
}

/// Returns the active contract, preserving direct-call compatibility outside a policy scope.
pub fn current_accelerator_fallback() -> AcceleratorFallback {
    ACTIVE_EXECUTION_PLAN
        .with(Cell::get)
        .map(TensorExecutionPlanBinding::accelerator_fallback)
        .or_else(|| ACTIVE_ACCELERATOR_FALLBACK.with(Cell::get))
        .unwrap_or_else(AcceleratorFallback::from_env)
}

/// Returns the committed tensor execution binding active on this thread.
pub fn current_execution_plan_binding() -> Option<TensorExecutionPlanBinding> {
    ACTIVE_EXECUTION_PLAN.with(Cell::get)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PreparedRoute {
    Direct,
    CpuThreshold,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PlannedReceiptRoute {
    Direct,
    DirectOrRuntimeFallback,
    AutoResolved,
    CpuThreshold,
    NoOp,
}

impl PlannedReceiptRoute {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::DirectOrRuntimeFallback => "direct or runtime_fallback",
            Self::AutoResolved => "auto_resolved",
            Self::CpuThreshold => "cpu_threshold",
            Self::NoOp => "no_op",
        }
    }

    const fn accepts(self, actual: TensorExecutionRouteStatus) -> bool {
        match self {
            Self::Direct => matches!(actual, TensorExecutionRouteStatus::Direct),
            Self::DirectOrRuntimeFallback => matches!(
                actual,
                TensorExecutionRouteStatus::Direct | TensorExecutionRouteStatus::RuntimeFallback
            ),
            Self::AutoResolved => matches!(actual, TensorExecutionRouteStatus::AutoResolved),
            Self::CpuThreshold => matches!(actual, TensorExecutionRouteStatus::CpuThreshold),
            Self::NoOp => matches!(actual, TensorExecutionRouteStatus::NoOp),
        }
    }
}

/// Validated request that may be completed into a receipt after output checks pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedTensorExecution {
    component: TensorExecutionComponent,
    operation: &'static str,
    workload: TensorExecutionWorkload,
    requested_backend: TensorExecutionBackend,
    selected_backend: TensorExecutionBackend,
    route: PreparedRoute,
    fallback: AcceleratorFallback,
    runtime_execution_plan_output_sha256: Option<[u8; 32]>,
}

/// Validate one operation request against the committed policy before dispatch.
pub fn prepare_tensor_execution(
    workload: TensorExecutionWorkload,
    operation: &'static str,
    selected_backend: TensorExecutionBackend,
) -> Result<PreparedTensorExecution, TensorExecutionContractError> {
    if !valid_operation_name(operation) {
        return Err(TensorExecutionContractError::InvalidOperation(operation));
    }
    let component = workload.component();
    if !operation_matches_workload(operation, workload) {
        return Err(TensorExecutionContractError::InvalidOperation(operation));
    }
    if !selected_backend.supports_component(component) {
        return Err(TensorExecutionContractError::UnsupportedBackend {
            component,
            backend: selected_backend,
        });
    }
    let fallback = current_accelerator_fallback();
    let Some(binding) = current_execution_plan_binding() else {
        return Ok(PreparedTensorExecution {
            component,
            operation,
            workload,
            requested_backend: selected_backend,
            selected_backend,
            route: PreparedRoute::Direct,
            fallback,
            runtime_execution_plan_output_sha256: None,
        });
    };

    let planned_backend = binding.backend_for(component);
    if let Some(planned_workload) = binding.planned_workload_for(workload) {
        if workload != planned_workload {
            return Err(TensorExecutionContractError::PlanWorkloadMismatch {
                component,
                planned: planned_workload,
                actual: workload,
            });
        }
    } else if binding.binds_component(component) {
        return Err(TensorExecutionContractError::PlanWorkloadNotAuthorized {
            component,
            actual: workload,
        });
    }
    let threshold_cpu_route = binding.uses_tensor_util_cpu_threshold(workload);
    let planned_selected_backend = binding.selected_backend_for(workload);
    if selected_backend != planned_selected_backend {
        return Err(TensorExecutionContractError::PlanBackendMismatch {
            component,
            planned: planned_selected_backend,
            selected: selected_backend,
        });
    }

    Ok(PreparedTensorExecution {
        component,
        operation,
        workload,
        requested_backend: planned_backend,
        selected_backend,
        route: if threshold_cpu_route {
            PreparedRoute::CpuThreshold
        } else {
            PreparedRoute::Direct
        },
        fallback: binding.accelerator_fallback(),
        runtime_execution_plan_output_sha256: Some(binding.output_sha256()),
    })
}

impl PreparedTensorExecution {
    /// Validate the backend that produced the returned output.
    pub fn complete(
        self,
        executed_backend: TensorExecutionBackend,
        fallback: Option<TensorExecutionFallback>,
    ) -> Result<TensorExecutionCompletion, TensorExecutionContractError> {
        if self.workload.has_empty_output() {
            return Err(TensorExecutionContractError::InvalidCompletion(
                "empty output workloads must complete as no-op",
            ));
        }
        if executed_backend == TensorExecutionBackend::Auto {
            return Err(TensorExecutionContractError::InvalidCompletion(
                "completed operations require a concrete executed backend",
            ));
        }
        if !executed_backend.supports_component(self.component) {
            return Err(TensorExecutionContractError::InvalidCompletion(
                "executed backend is not implemented for the prepared component",
            ));
        }

        let (selected_backend, route_status) = match self.route {
            PreparedRoute::CpuThreshold => {
                if executed_backend != TensorExecutionBackend::Cpu || fallback.is_some() {
                    return Err(TensorExecutionContractError::InvalidCompletion(
                        "threshold-routed operation must complete directly on CPU",
                    ));
                }
                (
                    TensorExecutionBackend::Cpu,
                    TensorExecutionRouteStatus::CpuThreshold,
                )
            }
            PreparedRoute::Direct if self.selected_backend == TensorExecutionBackend::Auto => {
                if fallback.is_some() {
                    return Err(TensorExecutionContractError::InvalidCompletion(
                        "automatic resolution must not be reported as runtime fallback",
                    ));
                }
                (executed_backend, TensorExecutionRouteStatus::AutoResolved)
            }
            PreparedRoute::Direct if self.selected_backend == executed_backend => {
                if fallback.is_some() {
                    return Err(TensorExecutionContractError::InvalidCompletion(
                        "direct completion must not include fallback evidence",
                    ));
                }
                (self.selected_backend, TensorExecutionRouteStatus::Direct)
            }
            PreparedRoute::Direct => {
                if !self.fallback.allows_fallback() {
                    return Err(TensorExecutionContractError::InvalidCompletion(
                        "backend changed while the active contract forbids fallback",
                    ));
                }
                let evidence = fallback.ok_or(TensorExecutionContractError::InvalidCompletion(
                    "backend changed without typed fallback evidence",
                ))?;
                if evidence.from != self.selected_backend || evidence.to != executed_backend {
                    return Err(TensorExecutionContractError::InvalidCompletion(
                        "fallback evidence does not match selected and executed backends",
                    ));
                }
                (
                    self.selected_backend,
                    TensorExecutionRouteStatus::RuntimeFallback,
                )
            }
        };

        Ok(TensorExecutionCompletion {
            prepared: self,
            selected_backend,
            executed_backend: Some(executed_backend),
            route_status,
            fallback,
        })
    }

    /// Complete an empty logical operation without claiming kernel execution.
    pub fn complete_no_op(self) -> Result<TensorExecutionCompletion, TensorExecutionContractError> {
        if !self.workload.has_empty_output() {
            return Err(TensorExecutionContractError::InvalidCompletion(
                "no-op completion requires an empty output workload",
            ));
        }
        Ok(TensorExecutionCompletion {
            prepared: self,
            selected_backend: self.selected_backend,
            executed_backend: None,
            route_status: TensorExecutionRouteStatus::NoOp,
            fallback: None,
        })
    }
}

/// Validated completion used to build a receipt only when metadata is observed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorExecutionCompletion {
    prepared: PreparedTensorExecution,
    selected_backend: TensorExecutionBackend,
    executed_backend: Option<TensorExecutionBackend>,
    route_status: TensorExecutionRouteStatus,
    fallback: Option<TensorExecutionFallback>,
}

impl TensorExecutionCompletion {
    pub fn receipt(self) -> TensorExecutionReceipt {
        let receipt = TensorExecutionReceipt {
            kind: TENSOR_EXECUTION_RECEIPT_KIND.to_owned(),
            contract_version: TENSOR_EXECUTION_RECEIPT_CONTRACT_VERSION.to_owned(),
            semantic_owner: TENSOR_EXECUTION_RECEIPT_SEMANTIC_OWNER.to_owned(),
            component: self.prepared.component,
            operation: self.prepared.operation.to_owned(),
            workload: self.prepared.workload,
            requested_backend: self.prepared.requested_backend,
            selected_backend: self.selected_backend,
            executed_backend: self.executed_backend,
            kernel_backend: self
                .executed_backend
                .and_then(TensorExecutionKernelBackend::for_executed_backend),
            route_status: self.route_status,
            fallback: self.fallback,
            runtime_execution_plan_output_sha256: self
                .prepared
                .runtime_execution_plan_output_sha256
                .map(sha256_hex),
        };
        debug_assert!(receipt.validate().is_ok());
        receipt
    }
}

/// Emit canonical completion metadata plus a typed execution receipt.
///
/// `extra` is evaluated only when an observer exists. Canonical route fields
/// are written after it so operation-specific metadata cannot override them.
pub fn emit_tensor_execution_receipt<F>(completion: TensorExecutionCompletion, extra: F)
where
    F: FnOnce(&mut serde_json::Map<String, serde_json::Value>),
{
    crate::emit_tensor_op_meta(completion.prepared.operation, || {
        let receipt = completion.receipt();
        let mut data = serde_json::Map::new();
        extra(&mut data);
        data.insert("event_phase".to_owned(), serde_json::json!("completed"));
        data.insert(
            "semantic_owner".to_owned(),
            serde_json::json!(TENSOR_EXECUTION_RECEIPT_SEMANTIC_OWNER),
        );
        data.insert(
            "requested_backend".to_owned(),
            serde_json::json!(receipt.requested_backend.as_str()),
        );
        data.insert(
            "selected_backend".to_owned(),
            serde_json::json!(receipt.selected_backend.as_str()),
        );
        data.insert(
            "route_status".to_owned(),
            serde_json::json!(receipt.route_status.as_str()),
        );
        data.insert(
            "counts_as_execution".to_owned(),
            serde_json::json!(receipt.executed_backend.is_some()),
        );
        if let Some(executed) = receipt.executed_backend {
            data.insert("backend".to_owned(), serde_json::json!(executed.as_str()));
            data.insert(
                "executed_backend".to_owned(),
                serde_json::json!(executed.as_str()),
            );
        }
        if let Some(kernel) = receipt.kernel_backend {
            data.insert(
                "kernel_backend".to_owned(),
                serde_json::json!(kernel.as_str()),
            );
        }
        data.insert(
            "execution_plan_committed".to_owned(),
            serde_json::json!(receipt.runtime_execution_plan_output_sha256.is_some()),
        );
        if let Some(commitment) = receipt.runtime_execution_plan_output_sha256.as_deref() {
            data.insert(
                "runtime_execution_plan_output_sha256".to_owned(),
                serde_json::json!(commitment),
            );
        }
        data.insert(
            "execution_receipt".to_owned(),
            serde_json::to_value(receipt).expect("tensor execution receipt is serializable"),
        );
        serde_json::Value::Object(data)
    });
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorExecutionContractError {
    InvalidOperation(&'static str),
    PlanBackendMismatch {
        component: TensorExecutionComponent,
        planned: TensorExecutionBackend,
        selected: TensorExecutionBackend,
    },
    PlanWorkloadMismatch {
        component: TensorExecutionComponent,
        planned: TensorExecutionWorkload,
        actual: TensorExecutionWorkload,
    },
    PlanWorkloadNotAuthorized {
        component: TensorExecutionComponent,
        actual: TensorExecutionWorkload,
    },
    DuplicatePlanWorkload {
        component: TensorExecutionComponent,
        key: TensorExecutionWorkloadKey,
    },
    UnsupportedBackend {
        component: TensorExecutionComponent,
        backend: TensorExecutionBackend,
    },
    InvalidCompletion(&'static str),
    InvalidReceipt {
        field: &'static str,
        message: &'static str,
    },
    ReceiptPlanCommitmentMissing,
    ReceiptPlanCommitmentMismatch {
        planned: String,
        receipt: String,
    },
    ReceiptPlanWorkloadMismatch {
        component: TensorExecutionComponent,
        planned: TensorExecutionWorkload,
        receipt: TensorExecutionWorkload,
    },
    ReceiptPlanWorkloadNotAuthorized {
        component: TensorExecutionComponent,
        receipt: TensorExecutionWorkload,
    },
    ReceiptPlanBackendMismatch {
        field: &'static str,
        component: TensorExecutionComponent,
        planned: TensorExecutionBackend,
        receipt: TensorExecutionBackend,
    },
    ReceiptPlanRouteMismatch {
        component: TensorExecutionComponent,
        expected: &'static str,
        receipt: TensorExecutionRouteStatus,
    },
}

impl fmt::Display for TensorExecutionContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOperation(operation) => write!(
                formatter,
                "invalid tensor execution operation identifier '{operation}'"
            ),
            Self::PlanBackendMismatch {
                component,
                planned,
                selected,
            } => write!(
                formatter,
                "committed {} backend is {}, but the tensor operation selected {}",
                component.as_str(),
                planned.as_str(),
                selected.as_str()
            ),
            Self::PlanWorkloadMismatch {
                component,
                planned,
                actual,
            } => write!(
                formatter,
                "committed {} workload {planned:?} does not match tensor operation workload {actual:?}",
                component.as_str()
            ),
            Self::PlanWorkloadNotAuthorized { component, actual } => write!(
                formatter,
                "committed {} workload set does not authorize tensor operation workload {actual:?}",
                component.as_str()
            ),
            Self::DuplicatePlanWorkload { component, key } => write!(
                formatter,
                "committed execution plan contains duplicate {} operation workload '{}'",
                component.as_str(),
                key.as_str()
            ),
            Self::UnsupportedBackend { component, backend } => write!(
                formatter,
                "tensor execution backend {} is not implemented for {}",
                backend.as_str(),
                component.as_str()
            ),
            Self::InvalidCompletion(message) => {
                write!(formatter, "invalid tensor execution completion: {message}")
            }
            Self::InvalidReceipt { field, message } => {
                write!(
                    formatter,
                    "invalid tensor execution receipt field '{field}': {message}"
                )
            }
            Self::ReceiptPlanCommitmentMissing => write!(
                formatter,
                "tensor execution receipt is not bound to a runtime execution plan"
            ),
            Self::ReceiptPlanCommitmentMismatch { planned, receipt } => write!(
                formatter,
                "tensor execution receipt commitment {receipt} does not match runtime execution plan {planned}"
            ),
            Self::ReceiptPlanWorkloadMismatch {
                component,
                planned,
                receipt,
            } => write!(
                formatter,
                "committed {} workload {planned:?} does not match receipt workload {receipt:?}",
                component.as_str()
            ),
            Self::ReceiptPlanWorkloadNotAuthorized { component, receipt } => write!(
                formatter,
                "committed {} workload set does not authorize receipt workload {receipt:?}",
                component.as_str()
            ),
            Self::ReceiptPlanBackendMismatch {
                field,
                component,
                planned,
                receipt,
            } => write!(
                formatter,
                "committed {} {field} is {}, but the receipt claims {}",
                component.as_str(),
                planned.as_str(),
                receipt.as_str()
            ),
            Self::ReceiptPlanRouteMismatch {
                component,
                expected,
                receipt,
            } => write!(
                formatter,
                "committed {} route requires {expected}, but the receipt claims {}",
                component.as_str(),
                receipt.as_str()
            ),
        }
    }
}

impl std::error::Error for TensorExecutionContractError {}

fn invalid_route(message: &'static str) -> TensorExecutionContractError {
    TensorExecutionContractError::InvalidReceipt {
        field: "route_status",
        message,
    }
}

fn is_accelerator_backend(backend: TensorExecutionBackend) -> bool {
    matches!(
        backend,
        TensorExecutionBackend::Wgpu | TensorExecutionBackend::Hip
    )
}

fn is_host_backend(backend: TensorExecutionBackend) -> bool {
    matches!(
        backend,
        TensorExecutionBackend::Cpu
            | TensorExecutionBackend::CpuSimd
            | TensorExecutionBackend::Naive
            | TensorExecutionBackend::Faer
    )
}

fn valid_operation_name(operation: &str) -> bool {
    !operation.is_empty()
        && operation.len() <= 128
        && operation
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

fn operation_matches_workload(operation: &str, workload: TensorExecutionWorkload) -> bool {
    match workload {
        TensorExecutionWorkload::DenseMatmul { .. } => matches!(
            operation,
            "matmul" | "matmul_scaled" | "matmul_lhs_transpose_scaled"
        ),
        TensorExecutionWorkload::PrepackedMatmul { .. } => {
            matches!(operation, "matmul_prepacked" | "matmul_prepacked_bias")
        }
        TensorExecutionWorkload::LayerNorm { .. } => operation == "layer_norm",
        TensorExecutionWorkload::Attention { .. } => operation == "scaled_dot_attention",
        TensorExecutionWorkload::Softmax { .. } => operation == "row_softmax",
        TensorExecutionWorkload::TensorUtil {
            operation: crate::execution_capability::TensorUtilOperation::Scale,
            ..
        } => operation == "scale",
        TensorExecutionWorkload::TensorUtil {
            operation: crate::execution_capability::TensorUtilOperation::MaxAxis0,
            ..
        } => operation == "max_axis0",
        TensorExecutionWorkload::TensorUtil {
            operation: crate::execution_capability::TensorUtilOperation::MaxAxis0Backward,
            ..
        } => operation == "max_axis0_backward",
    }
}

fn valid_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn committed_binding(
        fallback: AcceleratorFallback,
        tensor_util_wgpu_min_values: usize,
    ) -> TensorExecutionPlanBinding {
        TensorExecutionPlanBinding::new(
            [0xab; 32],
            [
                TensorExecutionBackend::Faer,
                TensorExecutionBackend::Faer,
                TensorExecutionBackend::Cpu,
                TensorExecutionBackend::Cpu,
                TensorExecutionBackend::Cpu,
                TensorExecutionBackend::Wgpu,
            ],
            fallback,
            tensor_util_wgpu_min_values,
        )
    }

    fn workload_bound_binding<const N: usize>(
        workloads: [Option<TensorExecutionWorkload>; N],
    ) -> TensorExecutionPlanBinding {
        TensorExecutionPlanBinding::try_new_with_workloads(
            [0xbc; 32],
            [
                TensorExecutionBackend::Faer,
                TensorExecutionBackend::Faer,
                TensorExecutionBackend::Cpu,
                TensorExecutionBackend::Cpu,
                TensorExecutionBackend::Cpu,
                TensorExecutionBackend::Wgpu,
            ],
            workloads.into_iter().flatten(),
            AcceleratorFallback::Allow,
            1024,
        )
        .expect("test operation workloads are unique")
    }

    #[test]
    fn fallback_guard_restores_nested_contracts() {
        let outer = push_accelerator_fallback(AcceleratorFallback::Forbid);
        assert_eq!(current_accelerator_fallback(), AcceleratorFallback::Forbid);

        {
            let _inner = push_accelerator_fallback(AcceleratorFallback::Allow);
            assert_eq!(current_accelerator_fallback(), AcceleratorFallback::Allow);
        }

        assert_eq!(current_accelerator_fallback(), AcceleratorFallback::Forbid);
        drop(outer);
    }

    #[test]
    fn tensor_util_operation_name_must_match_the_typed_workload() {
        let workload = TensorExecutionWorkload::TensorUtil {
            operation: crate::execution_capability::TensorUtilOperation::MaxAxis0,
            rows: 3,
            cols: 2,
        };

        assert!(matches!(
            prepare_tensor_execution(workload, "scale", TensorExecutionBackend::Cpu),
            Err(TensorExecutionContractError::InvalidOperation("scale"))
        ));

        let mut receipt =
            prepare_tensor_execution(workload, "max_axis0", TensorExecutionBackend::Cpu)
                .unwrap()
                .complete(TensorExecutionBackend::Cpu, None)
                .unwrap()
                .receipt();
        receipt.validate().unwrap();
        receipt.operation = "max_axis0_backward".to_owned();
        assert!(matches!(
            receipt.validate(),
            Err(TensorExecutionContractError::InvalidReceipt {
                field: "operation",
                ..
            })
        ));
    }

    #[test]
    fn committed_plan_rejects_a_different_component_backend() {
        let _fallback = push_accelerator_fallback(AcceleratorFallback::Allow);
        let _plan =
            push_execution_plan_binding(committed_binding(AcceleratorFallback::Allow, 1024));

        let error = prepare_tensor_execution(
            TensorExecutionWorkload::DenseMatmul {
                rows: 2,
                inner: 2,
                cols: 2,
            },
            "matmul",
            TensorExecutionBackend::Naive,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            TensorExecutionContractError::PlanBackendMismatch {
                component: TensorExecutionComponent::DenseMatmul,
                planned: TensorExecutionBackend::Faer,
                selected: TensorExecutionBackend::Naive,
            }
        ));
    }

    #[test]
    fn committed_plan_enforces_declared_workloads_for_every_component_before_dispatch() {
        let planned = [
            TensorExecutionWorkload::DenseMatmul {
                rows: 2,
                inner: 3,
                cols: 4,
            },
            TensorExecutionWorkload::PrepackedMatmul {
                rows: 2,
                inner: 3,
                cols: 4,
                bias: true,
            },
            TensorExecutionWorkload::LayerNorm { rows: 2, cols: 4 },
            TensorExecutionWorkload::Attention {
                contexts: 1,
                sequence: 2,
                head_dim: 4,
                z_bias: true,
                attn_bias: true,
            },
            TensorExecutionWorkload::Softmax { rows: 2, cols: 4 },
            TensorExecutionWorkload::TensorUtil {
                operation: crate::execution_capability::TensorUtilOperation::Scale,
                rows: 2,
                cols: 4,
            },
        ];
        let actual = [
            TensorExecutionWorkload::DenseMatmul {
                rows: 2,
                inner: 3,
                cols: 5,
            },
            TensorExecutionWorkload::PrepackedMatmul {
                rows: 2,
                inner: 3,
                cols: 4,
                bias: false,
            },
            TensorExecutionWorkload::LayerNorm { rows: 3, cols: 4 },
            TensorExecutionWorkload::Attention {
                contexts: 1,
                sequence: 2,
                head_dim: 4,
                z_bias: false,
                attn_bias: true,
            },
            TensorExecutionWorkload::Softmax { rows: 2, cols: 5 },
            TensorExecutionWorkload::TensorUtil {
                operation: crate::execution_capability::TensorUtilOperation::Scale,
                rows: 2,
                cols: 5,
            },
        ];
        let operations = [
            "matmul",
            "matmul_prepacked_bias",
            "layer_norm",
            "scaled_dot_attention",
            "row_softmax",
            "scale",
        ];
        let backends = [
            TensorExecutionBackend::Faer,
            TensorExecutionBackend::Faer,
            TensorExecutionBackend::Cpu,
            TensorExecutionBackend::Cpu,
            TensorExecutionBackend::Cpu,
            TensorExecutionBackend::Cpu,
        ];
        let workloads = planned.map(Some);
        let _plan = push_execution_plan_binding(workload_bound_binding(workloads));

        for (((planned, actual), operation), backend) in planned
            .into_iter()
            .zip(actual)
            .zip(operations)
            .zip(backends)
        {
            prepare_tensor_execution(planned, operation, backend)
                .expect("the exact committed workload is accepted");
            let error = prepare_tensor_execution(actual, operation, backend).unwrap_err();
            assert_eq!(
                error,
                TensorExecutionContractError::PlanWorkloadMismatch {
                    component: planned.component(),
                    planned,
                    actual,
                }
            );
        }
    }

    #[test]
    fn undeclared_component_workloads_remain_unbound() {
        let _plan =
            push_execution_plan_binding(committed_binding(AcceleratorFallback::Allow, 1024));

        for workload in [
            TensorExecutionWorkload::Softmax { rows: 2, cols: 3 },
            TensorExecutionWorkload::Softmax {
                rows: 128,
                cols: 257,
            },
        ] {
            prepare_tensor_execution(workload, "row_softmax", TensorExecutionBackend::Cpu)
                .expect("undeclared deferred workloads remain operation-time checks");
        }
    }

    #[test]
    fn committed_plan_binding_rejects_duplicate_workload_operations() {
        let error = TensorExecutionPlanBinding::try_new_with_workloads(
            [0xbc; 32],
            [TensorExecutionBackend::Cpu; 6],
            [
                TensorExecutionWorkload::Softmax { rows: 2, cols: 3 },
                TensorExecutionWorkload::Softmax { rows: 4, cols: 5 },
            ],
            AcceleratorFallback::Allow,
            1024,
        )
        .expect_err("duplicate operation commitments must fail closed");

        assert_eq!(
            error,
            TensorExecutionContractError::DuplicatePlanWorkload {
                component: TensorExecutionComponent::Softmax,
                key: TensorExecutionWorkloadKey::Softmax,
            }
        );
    }

    #[test]
    fn committed_plan_authorizes_max_forward_and_backward_together() {
        let forward = TensorExecutionWorkload::TensorUtil {
            operation: crate::execution_capability::TensorUtilOperation::MaxAxis0,
            rows: 3,
            cols: 2,
        };
        let backward = TensorExecutionWorkload::TensorUtil {
            operation: crate::execution_capability::TensorUtilOperation::MaxAxis0Backward,
            rows: 3,
            cols: 2,
        };
        let binding = TensorExecutionPlanBinding::try_new_with_workloads(
            [0xcd; 32],
            [TensorExecutionBackend::Cpu; 6],
            [forward, backward],
            AcceleratorFallback::Allow,
            1024,
        )
        .expect("distinct tensor utility operations share one component plan");
        let receipts = {
            let _plan = push_execution_plan_binding(binding);
            [
                prepare_tensor_execution(forward, "max_axis0", TensorExecutionBackend::Cpu)
                    .unwrap()
                    .complete(TensorExecutionBackend::Cpu, None)
                    .unwrap()
                    .receipt(),
                prepare_tensor_execution(
                    backward,
                    "max_axis0_backward",
                    TensorExecutionBackend::Cpu,
                )
                .unwrap()
                .complete(TensorExecutionBackend::Cpu, None)
                .unwrap()
                .receipt(),
            ]
        };

        for receipt in receipts {
            binding
                .validate_receipt(&receipt)
                .expect("both operation receipts are authorized by the same plan");
        }
    }

    #[test]
    fn bound_component_rejects_an_undeclared_operation_kind() {
        let forward = TensorExecutionWorkload::TensorUtil {
            operation: crate::execution_capability::TensorUtilOperation::MaxAxis0,
            rows: 3,
            cols: 2,
        };
        let backward = TensorExecutionWorkload::TensorUtil {
            operation: crate::execution_capability::TensorUtilOperation::MaxAxis0Backward,
            rows: 3,
            cols: 2,
        };
        let binding = TensorExecutionPlanBinding::try_new_with_workloads(
            [0xde; 32],
            [TensorExecutionBackend::Cpu; 6],
            [forward],
            AcceleratorFallback::Allow,
            1024,
        )
        .unwrap();

        {
            let _plan = push_execution_plan_binding(binding);
            assert_eq!(
                prepare_tensor_execution(
                    backward,
                    "max_axis0_backward",
                    TensorExecutionBackend::Cpu,
                )
                .unwrap_err(),
                TensorExecutionContractError::PlanWorkloadNotAuthorized {
                    component: TensorExecutionComponent::TensorUtil,
                    actual: backward,
                }
            );
        }

        let mut receipt =
            prepare_tensor_execution(backward, "max_axis0_backward", TensorExecutionBackend::Cpu)
                .unwrap()
                .complete(TensorExecutionBackend::Cpu, None)
                .unwrap()
                .receipt();
        receipt.runtime_execution_plan_output_sha256 = Some(sha256_hex(binding.output_sha256()));
        assert_eq!(
            binding.validate_receipt(&receipt).unwrap_err(),
            TensorExecutionContractError::ReceiptPlanWorkloadNotAuthorized {
                component: TensorExecutionComponent::TensorUtil,
                receipt: backward,
            }
        );
    }

    #[test]
    fn receipt_validation_requires_the_authorizing_plan_commitment() {
        let workload = TensorExecutionWorkload::Softmax { rows: 2, cols: 3 };
        let binding = workload_bound_binding([None, None, None, None, Some(workload), None]);
        let receipt = {
            let _plan = push_execution_plan_binding(binding);
            prepare_tensor_execution(workload, "row_softmax", TensorExecutionBackend::Cpu)
                .unwrap()
                .complete(TensorExecutionBackend::Cpu, None)
                .unwrap()
                .receipt()
        };

        binding.validate_receipt(&receipt).unwrap();

        let mut uncommitted = receipt.clone();
        uncommitted.runtime_execution_plan_output_sha256 = None;
        assert_eq!(
            binding.validate_receipt(&uncommitted).unwrap_err(),
            TensorExecutionContractError::ReceiptPlanCommitmentMissing
        );

        let foreign_binding = committed_binding(AcceleratorFallback::Allow, 1024);
        assert!(matches!(
            foreign_binding.validate_receipt(&receipt).unwrap_err(),
            TensorExecutionContractError::ReceiptPlanCommitmentMismatch { .. }
        ));
    }

    #[test]
    fn receipt_validation_rejects_reconstructed_workload_and_backend_claims() {
        let workload = TensorExecutionWorkload::Softmax { rows: 2, cols: 3 };
        let binding = workload_bound_binding([None, None, None, None, Some(workload), None]);
        let receipt = {
            let _plan = push_execution_plan_binding(binding);
            prepare_tensor_execution(workload, "row_softmax", TensorExecutionBackend::Cpu)
                .unwrap()
                .complete(TensorExecutionBackend::Cpu, None)
                .unwrap()
                .receipt()
        };

        let mut reconstructed_workload = receipt.clone();
        reconstructed_workload.workload = TensorExecutionWorkload::Softmax { rows: 2, cols: 5 };
        reconstructed_workload.validate().unwrap();
        assert_eq!(
            binding
                .validate_receipt(&reconstructed_workload)
                .unwrap_err(),
            TensorExecutionContractError::ReceiptPlanWorkloadMismatch {
                component: TensorExecutionComponent::Softmax,
                planned: workload,
                receipt: reconstructed_workload.workload,
            }
        );

        let mut reconstructed_backend = receipt;
        reconstructed_backend.requested_backend = TensorExecutionBackend::Wgpu;
        reconstructed_backend.selected_backend = TensorExecutionBackend::Wgpu;
        reconstructed_backend.executed_backend = Some(TensorExecutionBackend::Wgpu);
        reconstructed_backend.kernel_backend = Some(TensorExecutionKernelBackend::WgpuDense);
        reconstructed_backend.validate().unwrap();
        assert_eq!(
            binding
                .validate_receipt(&reconstructed_backend)
                .unwrap_err(),
            TensorExecutionContractError::ReceiptPlanBackendMismatch {
                field: "requested_backend",
                component: TensorExecutionComponent::Softmax,
                planned: TensorExecutionBackend::Cpu,
                receipt: TensorExecutionBackend::Wgpu,
            }
        );
    }

    #[test]
    fn receipt_validation_reapplies_fallback_and_automatic_route_contracts() {
        let wgpu_backends = [TensorExecutionBackend::Wgpu; 6];
        let permissive = TensorExecutionPlanBinding::new(
            [0xdd; 32],
            wgpu_backends,
            AcceleratorFallback::Allow,
            1024,
        );
        let fallback_receipt = {
            let _plan = push_execution_plan_binding(permissive);
            prepare_tensor_execution(
                TensorExecutionWorkload::Softmax { rows: 2, cols: 3 },
                "row_softmax",
                TensorExecutionBackend::Wgpu,
            )
            .unwrap()
            .complete(
                TensorExecutionBackend::Cpu,
                Some(TensorExecutionFallback::runtime_unavailable(
                    TensorExecutionBackend::Wgpu,
                    TensorExecutionBackend::Cpu,
                )),
            )
            .unwrap()
            .receipt()
        };
        permissive.validate_receipt(&fallback_receipt).unwrap();

        let strict = TensorExecutionPlanBinding::new(
            [0xdd; 32],
            wgpu_backends,
            AcceleratorFallback::Forbid,
            1024,
        );
        assert_eq!(
            strict.validate_receipt(&fallback_receipt).unwrap_err(),
            TensorExecutionContractError::ReceiptPlanRouteMismatch {
                component: TensorExecutionComponent::Softmax,
                expected: "direct",
                receipt: TensorExecutionRouteStatus::RuntimeFallback,
            }
        );

        let automatic = TensorExecutionPlanBinding::new(
            [0xee; 32],
            [TensorExecutionBackend::Auto; 6],
            AcceleratorFallback::Allow,
            1024,
        );
        let automatic_receipt = {
            let _plan = push_execution_plan_binding(automatic);
            prepare_tensor_execution(
                TensorExecutionWorkload::Softmax { rows: 2, cols: 3 },
                "row_softmax",
                TensorExecutionBackend::Auto,
            )
            .unwrap()
            .complete(TensorExecutionBackend::Cpu, None)
            .unwrap()
            .receipt()
        };
        automatic.validate_receipt(&automatic_receipt).unwrap();
    }

    #[test]
    fn committed_tensor_util_threshold_produces_a_bound_cpu_receipt() {
        let _fallback = push_accelerator_fallback(AcceleratorFallback::Allow);
        let binding = committed_binding(AcceleratorFallback::Allow, 1024);
        let _plan = push_execution_plan_binding(binding);
        let prepared = prepare_tensor_execution(
            TensorExecutionWorkload::TensorUtil {
                operation: crate::execution_capability::TensorUtilOperation::Scale,
                rows: 2,
                cols: 4,
            },
            "scale",
            TensorExecutionBackend::Cpu,
        )
        .unwrap();
        let receipt = prepared
            .complete(TensorExecutionBackend::Cpu, None)
            .unwrap()
            .receipt();

        receipt.validate().unwrap();
        assert_eq!(receipt.requested_backend, TensorExecutionBackend::Wgpu);
        assert_eq!(receipt.selected_backend, TensorExecutionBackend::Cpu);
        assert_eq!(receipt.executed_backend, Some(TensorExecutionBackend::Cpu));
        assert_eq!(
            receipt.kernel_backend,
            Some(TensorExecutionKernelBackend::Cpu)
        );
        assert_eq!(
            receipt.route_status,
            TensorExecutionRouteStatus::CpuThreshold
        );
        assert_eq!(
            receipt.runtime_execution_plan_output_sha256.as_deref(),
            Some("abababababababababababababababababababababababababababababababab")
        );
        binding.validate_receipt(&receipt).unwrap();
    }

    #[test]
    fn committed_tensor_util_threshold_cannot_be_bypassed_by_direct_wgpu_selection() {
        let binding = committed_binding(AcceleratorFallback::Allow, 1024);
        let _plan = push_execution_plan_binding(binding);
        let error = prepare_tensor_execution(
            TensorExecutionWorkload::TensorUtil {
                operation: crate::execution_capability::TensorUtilOperation::Scale,
                rows: 2,
                cols: 4,
            },
            "scale",
            TensorExecutionBackend::Wgpu,
        )
        .expect_err("the committed CPU threshold route is mandatory");

        assert_eq!(
            error,
            TensorExecutionContractError::PlanBackendMismatch {
                component: TensorExecutionComponent::TensorUtil,
                planned: TensorExecutionBackend::Cpu,
                selected: TensorExecutionBackend::Wgpu,
            }
        );
    }

    #[test]
    fn runtime_fallback_requires_matching_typed_evidence() {
        let _fallback = push_accelerator_fallback(AcceleratorFallback::Allow);
        let prepared = prepare_tensor_execution(
            TensorExecutionWorkload::Softmax { rows: 2, cols: 3 },
            "row_softmax",
            TensorExecutionBackend::Wgpu,
        )
        .unwrap();

        assert!(prepared
            .complete(TensorExecutionBackend::Cpu, None)
            .is_err());
        let receipt = prepared
            .complete(
                TensorExecutionBackend::Cpu,
                Some(TensorExecutionFallback::runtime_unavailable(
                    TensorExecutionBackend::Wgpu,
                    TensorExecutionBackend::Cpu,
                )),
            )
            .unwrap()
            .receipt();
        receipt.validate().unwrap();
        assert_eq!(
            receipt.route_status,
            TensorExecutionRouteStatus::RuntimeFallback
        );
    }

    #[test]
    fn committed_plan_fallback_cannot_be_weakened_by_an_independent_guard() {
        let _legacy_fallback = push_accelerator_fallback(AcceleratorFallback::Allow);
        let _plan =
            push_execution_plan_binding(committed_binding(AcceleratorFallback::Forbid, 1024));
        assert_eq!(current_accelerator_fallback(), AcceleratorFallback::Forbid);
        let prepared = prepare_tensor_execution(
            TensorExecutionWorkload::TensorUtil {
                operation: crate::execution_capability::TensorUtilOperation::Scale,
                rows: 32,
                cols: 64,
            },
            "scale",
            TensorExecutionBackend::Wgpu,
        )
        .unwrap();

        assert!(prepared
            .complete(
                TensorExecutionBackend::Cpu,
                Some(TensorExecutionFallback::runtime_unavailable(
                    TensorExecutionBackend::Wgpu,
                    TensorExecutionBackend::Cpu,
                )),
            )
            .is_err());
    }

    #[test]
    fn no_op_receipt_rejects_an_uncommitted_backend_change() {
        let mut receipt = prepare_tensor_execution(
            TensorExecutionWorkload::Softmax { rows: 0, cols: 3 },
            "row_softmax",
            TensorExecutionBackend::Cpu,
        )
        .unwrap()
        .complete_no_op()
        .unwrap()
        .receipt();
        receipt.selected_backend = TensorExecutionBackend::Wgpu;

        assert!(receipt.validate().is_err());
    }

    #[test]
    fn no_op_completion_rejects_a_non_empty_workload() {
        let prepared = prepare_tensor_execution(
            TensorExecutionWorkload::Softmax { rows: 1, cols: 3 },
            "row_softmax",
            TensorExecutionBackend::Cpu,
        )
        .unwrap();

        assert!(prepared.complete_no_op().is_err());
    }

    #[test]
    fn receipt_validation_requires_route_to_match_output_emptiness() {
        let mut empty = prepare_tensor_execution(
            TensorExecutionWorkload::Softmax { rows: 0, cols: 3 },
            "row_softmax",
            TensorExecutionBackend::Cpu,
        )
        .unwrap()
        .complete_no_op()
        .unwrap()
        .receipt();
        empty.route_status = TensorExecutionRouteStatus::Direct;
        empty.executed_backend = Some(TensorExecutionBackend::Cpu);
        assert!(empty.validate().is_err());

        let mut non_empty = prepare_tensor_execution(
            TensorExecutionWorkload::Softmax { rows: 1, cols: 3 },
            "row_softmax",
            TensorExecutionBackend::Cpu,
        )
        .unwrap()
        .complete(TensorExecutionBackend::Cpu, None)
        .unwrap()
        .receipt();
        non_empty.route_status = TensorExecutionRouteStatus::NoOp;
        non_empty.executed_backend = None;
        assert!(non_empty.validate().is_err());
    }

    #[test]
    fn component_backend_pairs_are_validated_at_prepare_complete_and_decode() {
        let prepare_error = prepare_tensor_execution(
            TensorExecutionWorkload::Softmax { rows: 1, cols: 3 },
            "row_softmax",
            TensorExecutionBackend::Faer,
        )
        .unwrap_err();
        assert!(matches!(
            prepare_error,
            TensorExecutionContractError::UnsupportedBackend {
                component: TensorExecutionComponent::Softmax,
                backend: TensorExecutionBackend::Faer,
            }
        ));

        let prepared = prepare_tensor_execution(
            TensorExecutionWorkload::Softmax { rows: 1, cols: 3 },
            "row_softmax",
            TensorExecutionBackend::Auto,
        )
        .unwrap();
        assert!(prepared
            .complete(TensorExecutionBackend::Faer, None)
            .is_err());

        let mut receipt = prepared
            .complete(TensorExecutionBackend::Cpu, None)
            .unwrap()
            .receipt();
        receipt.selected_backend = TensorExecutionBackend::Faer;
        receipt.executed_backend = Some(TensorExecutionBackend::Faer);
        assert!(receipt.validate().is_err());
    }

    #[test]
    fn receipt_round_trip_is_strict_and_tamper_evident() {
        let _fallback = push_accelerator_fallback(AcceleratorFallback::Forbid);
        let prepared = prepare_tensor_execution(
            TensorExecutionWorkload::DenseMatmul {
                rows: 2,
                inner: 2,
                cols: 2,
            },
            "matmul",
            TensorExecutionBackend::Faer,
        )
        .unwrap();
        let receipt = prepared
            .complete(TensorExecutionBackend::Faer, None)
            .unwrap()
            .receipt();
        let encoded = serde_json::to_value(&receipt).unwrap();
        let decoded: TensorExecutionReceipt = serde_json::from_value(encoded.clone()).unwrap();
        decoded.validate().unwrap();
        assert_eq!(
            decoded.kernel_backend,
            Some(TensorExecutionKernelBackend::Faer)
        );

        let mut tampered = encoded;
        tampered["executed_backend"] = serde_json::json!("naive");
        let tampered: TensorExecutionReceipt = serde_json::from_value(tampered).unwrap();
        assert!(tampered.validate().is_err());

        let mut reconstructed_kernel = serde_json::to_value(&receipt).unwrap();
        reconstructed_kernel["kernel_backend"] = serde_json::json!("wgpu_dense");
        let reconstructed_kernel: TensorExecutionReceipt =
            serde_json::from_value(reconstructed_kernel).unwrap();
        assert!(reconstructed_kernel.validate().is_err());

        let mut mismatched_workload = serde_json::to_value(&receipt).unwrap();
        mismatched_workload["workload"] = serde_json::json!({
            "component": "softmax",
            "rows": 2,
            "cols": 2,
        });
        let mismatched_workload: TensorExecutionReceipt =
            serde_json::from_value(mismatched_workload).unwrap();
        assert!(mismatched_workload.validate().is_err());
    }
}
