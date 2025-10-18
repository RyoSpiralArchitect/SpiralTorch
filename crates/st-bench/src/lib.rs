// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)

use rand::{rngs::StdRng, Rng, SeedableRng};
use st_core::coop::ai::{CoopAgent, CoopProposal};
use st_core::coop::mixer::{team_reward, DifferenceRewardMixer, TeamTelemetry};
use st_core::coop::r#loop::CoopLoop;

struct BenchAgent {
    rng: StdRng,
}

impl BenchAgent {
    fn with_seed(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl CoopAgent for BenchAgent {
    fn propose(&mut self) -> CoopProposal {
        let z_bias = self.rng.gen_range(-1.0..=1.0);
        let weight = self.rng.gen_range(0.1..=1.0);
        CoopProposal::new(z_bias, weight)
    }

    fn observe(&mut self, _team_reward: f32, _credit: f32) {
        // Bench agents keep a simple policy; observations are ignored.
    }
}

/// Simulate a cooperative loop for a fixed number of steps and accumulate rewards.
pub fn simulate_linear_epoch(seed: u64, steps: usize) -> f32 {
    let agents: Vec<Box<dyn CoopAgent>> = (0..4)
        .map(|idx| Box::new(BenchAgent::with_seed(seed + idx as u64)) as Box<dyn CoopAgent>)
        .collect();

    let mut mixer = DifferenceRewardMixer::new(|z, telemetry: &TeamTelemetry| {
        team_reward(z, telemetry.flip_rate, telemetry.here_ratio)
    });

    let telemetry = TeamTelemetry {
        flip_rate: 0.25,
        here_ratio: 0.1,
    };

    let mut coop_loop = CoopLoop::new(mixer, agents);
    let mut total_reward = 0.0;

    for _ in 0..steps {
        let step = coop_loop.step(&telemetry);
        total_reward += step.outcome.team_reward;
    }

    total_reward
}
