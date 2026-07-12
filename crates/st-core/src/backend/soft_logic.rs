// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#[cfg(feature = "logic")]
#[derive(Clone, Debug)]
pub struct SoftRule {
    pub name: &'static str,
    pub weight: f32,
    pub score: f32,
}

#[cfg(not(feature = "logic"))]
#[derive(Clone, Debug, Default)]
pub struct SoftRule;

#[cfg(feature = "logic-learn")]
pub use st_softlogic::learn;
