// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::ai::{CoopAgent, CoopProposal};
use super::mixer::{TeamMix, TeamMixer, TeamTelemetry};

pub struct TeamStep {
    pub proposals: Vec<CoopProposal>,
    pub outcome: TeamMix,
}

pub struct CoopLoop<M: TeamMixer> {
    mixer: M,
    agents: Vec<Box<dyn CoopAgent>>,
}

impl<M: TeamMixer> CoopLoop<M> {
    pub fn new(mixer: M, agents: Vec<Box<dyn CoopAgent>>) -> Self {
        Self { mixer, agents }
    }

    pub fn step(&mut self, telemetry: &TeamTelemetry) -> TeamStep {
        let mut proposals = Vec::with_capacity(self.agents.len());
        for agent in self.agents.iter_mut() {
            proposals.push(agent.propose());
        }

        let outcome = self.mixer.mix(&proposals, telemetry);

        for (agent, credit) in self.agents.iter_mut().zip(outcome.credits.iter().copied()) {
            agent.observe(outcome.team_reward, credit);
        }

        TeamStep { proposals, outcome }
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::super::ai::{CoopAgent, CoopProposal};
    use super::super::mixer::{team_reward, DifferenceRewardMixer, TeamTelemetry};
    use super::{CoopLoop, TeamStep};

    struct FakeAgent {
        rng: StdRng,
        pub observed: Vec<(f32, f32)>,
    }

    impl FakeAgent {
        fn new(seed: u64) -> Self {
            Self {
                rng: StdRng::seed_from_u64(seed),
                observed: Vec::new(),
            }
        }
    }

    impl CoopAgent for FakeAgent {
        fn propose(&mut self) -> CoopProposal {
            let z_bias = self.rng.gen_range(-1.0..=1.0);
            let weight = self.rng.gen_range(0.1..=1.0);
            CoopProposal::new(z_bias, weight)
        }

        fn observe(&mut self, team_reward: f32, credit: f32) {
            self.observed.push((team_reward, credit));
        }
    }

    #[test]
    fn loop_produces_difference_reward_signals() {
        let agents: Vec<Box<dyn CoopAgent>> = (0..4)
            .map(|idx| Box::new(FakeAgent::new(idx as u64 + 1)) as Box<dyn CoopAgent>)
            .collect();

        let mut mixer = DifferenceRewardMixer::new(|z, telemetry: &TeamTelemetry| {
            team_reward(z, telemetry.flip_rate, telemetry.here_ratio)
        });

        let telemetry = TeamTelemetry {
            flip_rate: 0.25,
            here_ratio: 0.1,
        };

        let mut coop_loop = CoopLoop::new(mixer, agents);
        let TeamStep { proposals, outcome } = coop_loop.step(&telemetry);

        assert_eq!(proposals.len(), 4);
        assert_eq!(outcome.credits.len(), 4);

        for (index, proposal) in proposals.iter().enumerate() {
            let mut filtered = proposals.clone();
            filtered.remove(index);
            let alt_z = super::super::mixer::fuse_z_bias(&filtered);
            let counter_reward = team_reward(alt_z, telemetry.flip_rate, telemetry.here_ratio);
            let expected = outcome.team_reward - counter_reward;
            let credit = outcome.credits[index];
            assert!(
                (credit - expected).abs() < 1e-5,
                "credit mismatch at {index}"
            );

            if expected.abs() > f32::EPSILON {
                assert_eq!(credit.signum(), expected.signum());
            } else {
                assert!(credit.abs() <= f32::EPSILON);
            }

            assert!(proposal.weight > 0.0);
        }
    }
}
