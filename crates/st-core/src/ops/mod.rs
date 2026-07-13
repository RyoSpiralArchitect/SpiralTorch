// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Lightweight public operations exposed from the SpiralTorch core crate.
//!
//! Historically this module mixed experimental helpers with runtime contracts.
//! The current surface keeps only stable adapters here: their mathematical
//! semantics live in focused crates such as `st-frac`, while `st-core` exposes
//! them to runtimes without reimplementing those semantics.

pub mod ablog;
pub mod frac;
pub mod frac_autograd;
pub mod operator_registry;
pub mod rank_entry;
pub mod realgrad;
pub mod zspace_round;

pub use operator_registry::{
    global_operator_registry, GradientFn, OperatorBuilder, OperatorFn, OperatorMetadata,
    OperatorRegistry, OperatorSignature, RegisteredOperator,
};
