// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// crates/st-amg/src/lib.rs
//! SpiralTorch algebraic multigrid utilities.
//!
//! This crate provides two primary integration points:
//! - Backend heuristics to select WGPU AMG tuning parameters.
//! - SoftRule learning helpers that persist collaborative roundtable outcomes.

pub mod backend;

pub mod sr_learn;

pub use backend::wgpu_heuristics_amg::{choose as choose_wgpu_amg, Choice as WgpuHeuristicChoice};
pub use sr_learn::{maybe_append_soft, on_abc_conversation, wilson_lower_bound};
