// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "redis")]

/// Controls conditional write behaviour for Redis `SET`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JsonSetCondition {
    /// Always perform the write, regardless of key existence.
    Always,
    /// Only write when the key does not yet exist (`NX`).
    Nx,
    /// Only write when the key already exists (`XX`).
    Xx,
}

impl Default for JsonSetCondition {
    fn default() -> Self {
        Self::Always
    }
}
