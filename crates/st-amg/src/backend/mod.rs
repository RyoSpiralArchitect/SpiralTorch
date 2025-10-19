// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// crates/st-amg/src/backend/mod.rs
//! Backend specific AMG heuristics.

pub mod wgpu_heuristics_amg;

pub use wgpu_heuristics_amg::{choose as wgpu_choose, Choice as WgpuChoice};
