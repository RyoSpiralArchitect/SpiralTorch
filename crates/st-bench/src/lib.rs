// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)

use rand::{rngs::StdRng, Rng, SeedableRng};
use st_core::coop::ai::{CoopAgent, CoopProposal};
use st_core::coop::mixer::{team_reward, DifferenceRewardMixer, TeamTelemetry};
use st_core::coop::r#loop::CoopLoop;

/// Synthetic backend identifier used by the benchmark harness.
#[derive(Clone, Debug, PartialEq)]
pub struct BackendProbe<'a> {
    pub name: &'a str,
    pub base_throughput: f32,
    pub latency_ms: f32,
}

/// Aggregated benchmark metrics for a backend run.
#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkSample {
    pub backend: String,
    pub throughput: f32,
    pub latency_ms: f32,
}

/// Report summarising all simulated backend runs.
#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkReport {
    pub samples: Vec<BenchmarkSample>,
}

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

    let mixer = DifferenceRewardMixer::new(|z, telemetry: &TeamTelemetry| {
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

/// Benchmarks backend probes using a stochastic throughput model.
pub fn benchmark_backends(probes: &[BackendProbe<'_>], steps: usize) -> BenchmarkReport {
    let mut rng = StdRng::seed_from_u64(steps as u64 + 0x5A5A);
    let samples = probes
        .iter()
        .map(|probe| {
            let jitter = rng.gen_range(0.9..=1.1);
            BenchmarkSample {
                backend: probe.name.to_string(),
                throughput: probe.base_throughput * jitter * steps as f32,
                latency_ms: (probe.latency_ms / jitter).max(0.1),
            }
        })
        .collect();
    BenchmarkReport { samples }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_benchmark_reports_all_probes() {
        let probes = [
            BackendProbe {
                name: "wgpu",
                base_throughput: 1.25,
                latency_ms: 9.0,
            },
            BackendProbe {
                name: "cuda",
                base_throughput: 2.75,
                latency_ms: 4.0,
            },
        ];
        let report = benchmark_backends(&probes, 8);
        assert_eq!(report.samples.len(), 2);
        assert!(report.samples.iter().any(|sample| sample.backend == "cuda"));
    }
}
