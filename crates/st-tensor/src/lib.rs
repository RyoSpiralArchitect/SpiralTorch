#![cfg_attr(feature = "simd", feature(portable_simd))]
// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// crates/st-tensor/src/lib.rs
pub mod fractional;

pub mod autograd;
pub mod backend;
pub mod classification;
pub mod compaction;
pub mod dlpack;
pub mod execution;
pub mod execution_capability;
mod hardmax;
mod indexing;
mod memory;
mod normalization;
pub mod observability;

#[cfg(feature = "wgpu_frac")]
mod util;

pub use autograd::{
    AutogradBackwardReport, AutogradGraphSummary, AutogradSgd, AutogradTensor,
    AUTOGRAD_CONTRACT_VERSION, AUTOGRAD_SEMANTIC_OWNER,
};
pub use backend::faer_dense;
pub use classification::{class_indices_from_tensor, CrossEntropyConfig, LossReduction};
pub use execution::{
    emit_tensor_execution_receipt, prepare_tensor_execution, TensorExecutionCompletion,
    TensorExecutionContractError, TensorExecutionFallback, TensorExecutionFallbackReason,
    TensorExecutionKernelBackend, TensorExecutionPlanBinding, TensorExecutionReceipt,
    TensorExecutionRouteStatus, TENSOR_EXECUTION_RECEIPT_CONTRACT_VERSION,
    TENSOR_EXECUTION_RECEIPT_KIND, TENSOR_EXECUTION_RECEIPT_SEMANTIC_OWNER,
};
pub use execution_capability::{
    TensorExecutionBackend, TensorExecutionComponent, TensorExecutionWorkload,
    TensorExecutionWorkloadKey, TensorUtilOperation,
};

#[cfg(feature = "wgpu_frac")]
pub use backend::wgpu_frac;

#[cfg(feature = "wgpu_dense")]
pub use backend::wgpu_dense;

mod pure;

#[doc = "Re-exported for convenience."]
pub use pure::*;

pub use observability::{
    emit_tensor_op, emit_tensor_op_meta, set_tensor_op_meta_observer, set_tensor_op_observer,
    set_thread_meta_observer, tensor_op_meta_observer_installed, TensorOpEvent, TensorOpMetaEvent,
    TensorOpMetaObserver, TensorOpObserver,
};

#[doc = "Expose the hardmax backend selector."]
pub use hardmax::HardmaxBackend;

#[doc = "Re-exported for convenience."]
pub use pure::wasm_canvas;
