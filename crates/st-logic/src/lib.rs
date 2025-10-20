// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

pub use st_softlogic::{apply_softmode, beam_select, Ctx, SoftMode, SoftRule, SolveCfg};

#[cfg(feature = "learn_store")]
pub use st_softlogic::learn;

#[cfg(feature = "nerf")]
pub mod nerf_trainer;
pub mod quantum_reality;
pub mod temporal_dynamics;
