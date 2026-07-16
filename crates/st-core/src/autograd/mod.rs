// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2025 Ryo SpiralArchitect

//! Higher-order differentiation built on the `st-tensor` autograd contract.

pub mod hypergrad;

pub use hypergrad::{
    implicit, implicit_with_options, unrolled, FiniteDiffMode, HypergradError, HypergradResult,
    ImplicitDiagnostics, ImplicitOptions, ImplicitOut, Solver, UnrolledOut,
    HYPERGRAD_CONTRACT_VERSION, HYPERGRAD_SEMANTIC_OWNER,
};
pub use st_tensor::{
    AutogradBackwardReport, AutogradGraphSummary, AutogradTensor, AUTOGRAD_CONTRACT_VERSION,
    AUTOGRAD_SEMANTIC_OWNER,
};
