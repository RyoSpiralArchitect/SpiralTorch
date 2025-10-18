// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

/// Cooperative proposal emitted by each layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoopProposal {
    pub z_bias: f32,
    pub weight: f32,
}

impl CoopProposal {
    pub fn new(z_bias: f32, weight: f32) -> Self {
        Self { z_bias, weight }
    }

    pub fn neutral() -> Self {
        Self {
            z_bias: 0.0,
            weight: 0.0,
        }
    }
}

pub trait CoopAgent: Send {
    fn propose(&mut self) -> CoopProposal;

    fn observe(&mut self, team_reward: f32, credit: f32);
}
