// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Lightweight public operations exposed from the SpiralTorch core crate.
//!
//! Historically this module re-exported a large collection of experimental
//! utilities (fractional autodiff, ndarray-based FFT helpers, prototype
//! hypergrad solvers, etc.). Those prototypes pulled in heavy dependencies and
//! routinely broke clean builds when the optional features were not enabled.
//!
//! The new high-level `st-nn` crate only needs the rank planner entry points
//! today, so we keep the surface tight and dependency free. The experimental
//! helpers live next to their callers and can grow behind dedicated features
//! without leaking into the default build.

pub mod ablog;
pub mod rank_entry;
pub mod zspace_round;
