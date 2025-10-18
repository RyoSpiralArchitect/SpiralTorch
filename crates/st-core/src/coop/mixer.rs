// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::ai::CoopProposal;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TeamTelemetry {
    pub flip_rate: f32,
    pub here_ratio: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TeamMix {
    pub fused_z_bias: f32,
    pub team_reward: f32,
    pub credits: Vec<f32>,
}

impl TeamMix {
    pub fn credit_for(&self, index: usize) -> Option<f32> {
        self.credits.get(index).copied()
    }
}

pub trait TeamMixer: Send {
    fn mix(&mut self, proposals: &[CoopProposal], telemetry: &TeamTelemetry) -> TeamMix;
}

pub fn fuse_z_bias(proposals: &[CoopProposal]) -> f32 {
    let mut weight_sum = 0.0f32;
    let mut weighted = 0.0f32;
    for proposal in proposals {
        if proposal.weight.abs() <= f32::EPSILON {
            continue;
        }
        weight_sum += proposal.weight;
        weighted += proposal.z_bias * proposal.weight;
    }

    if weight_sum.abs() <= f32::EPSILON {
        0.0
    } else {
        weighted / weight_sum
    }
}

pub struct DifferenceRewardMixer<R> {
    reward_fn: R,
}

impl<R> DifferenceRewardMixer<R> {
    pub fn new(reward_fn: R) -> Self {
        Self { reward_fn }
    }
}

impl<R> TeamMixer for DifferenceRewardMixer<R>
where
    R: Fn(f32, &TeamTelemetry) -> f32 + Send + Sync,
{
    fn mix(&mut self, proposals: &[CoopProposal], telemetry: &TeamTelemetry) -> TeamMix {
        let fused_z_bias = fuse_z_bias(proposals);
        let team_reward = (&self.reward_fn)(fused_z_bias, telemetry);

        let mut credits = Vec::with_capacity(proposals.len());
        let total_weight: f32 = proposals.iter().map(|p| p.weight).sum();
        let total_weighted: f32 = proposals.iter().map(|p| p.z_bias * p.weight).sum();

        for (idx, proposal) in proposals.iter().enumerate() {
            let remainder_weight = total_weight - proposal.weight;
            let remainder_weighted = total_weighted - proposal.z_bias * proposal.weight;
            let counterfactual_z = if remainder_weight.abs() <= f32::EPSILON {
                0.0
            } else {
                remainder_weighted / remainder_weight
            };
            let counter_reward = (&self.reward_fn)(counterfactual_z, telemetry);
            credits.push(team_reward - counter_reward);
        }

        TeamMix {
            fused_z_bias,
            team_reward,
            credits,
        }
    }
}

pub fn team_reward(z: f32, flip_rate: f32, here: f32) -> f32 {
    let s = (z.abs() - 0.2).max(0.0);
    s * 0.8 - flip_rate * 0.5 - here * 0.3
}

#[cfg(test)]
mod tests {
    use super::{fuse_z_bias, team_reward, DifferenceRewardMixer, TeamTelemetry};
    use crate::coop::ai::CoopProposal;

    #[test]
    fn fuse_z_bias_handles_empty() {
        let fused = fuse_z_bias(&[]);
        assert_eq!(fused, 0.0);
    }

    #[test]
    fn difference_reward_matches_definition() {
        let proposals = vec![
            CoopProposal::new(0.4, 1.0),
            CoopProposal::new(-0.1, 0.5),
            CoopProposal::new(0.7, 1.5),
        ];
        let telemetry = TeamTelemetry {
            flip_rate: 0.2,
            here_ratio: 0.1,
        };

        let mut mixer = DifferenceRewardMixer::new(|z, telemetry: &TeamTelemetry| {
            team_reward(z, telemetry.flip_rate, telemetry.here_ratio)
        });

        let mix = mixer.mix(&proposals, &telemetry);

        for (index, credit) in mix.credits.iter().enumerate() {
            let mut filtered = proposals.clone();
            filtered.remove(index);
            let counter_z = fuse_z_bias(&filtered);
            let counter_reward = team_reward(counter_z, telemetry.flip_rate, telemetry.here_ratio);
            let expected = mix.team_reward - counter_reward;
            assert!(
                (credit - expected).abs() < 1e-6,
                "credit mismatch at {index}"
            );
        }
    }
}
