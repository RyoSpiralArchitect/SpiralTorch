// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)

use rand::{rngs::StdRng, Rng, SeedableRng};
use st_core::coop::ai::{CoopAgent, CoopProposal};
use st_core::coop::mixer::{team_reward, DifferenceRewardMixer, TeamTelemetry};
use st_core::coop::r#loop::CoopLoop;
use std::collections::HashMap;

pub mod backend_matrix;

pub use backend_matrix::{
    backend_summaries, capabilities_for_backend_with_state, capabilities_with_note_containing,
    capabilities_with_state, capability_by_name, capability_matrix, capability_matrix_json,
    capability_matrix_view, capability_summaries, matrix_summary, pending_capabilities_for_backend,
    readiness_leaderboard, summarize_backend, Backend, BackendNote, BackendSummary,
    CapabilityEntry, CapabilityMatrix, CapabilityRow, CapabilityState, CapabilitySummary,
    MatrixSummary, CAPABILITY_MATRIX,
};

mod model {
    use super::*;

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

    /// Statistical summary derived from one or more [`BenchmarkSample`] entries.
    #[derive(Clone, Debug, PartialEq)]
    pub struct BenchmarkStats {
        pub backend: String,
        pub throughput_mean: f32,
        pub throughput_std: f32,
        pub latency_mean: f32,
        pub latency_p95: f32,
    }

    impl BenchmarkStats {
        /// Relative throughput speedup compared to `baseline`.
        pub fn throughput_speedup(&self, baseline: &BenchmarkStats) -> f32 {
            let denom = baseline.throughput_mean.max(f32::EPSILON);
            self.throughput_mean / denom
        }

        /// Mean latency delta (positive values indicate slower execution than `baseline`).
        pub fn latency_delta(&self, baseline: &BenchmarkStats) -> f32 {
            self.latency_mean - baseline.latency_mean
        }
    }

    /// Report summarising all simulated backend runs.
    #[derive(Clone, Debug, PartialEq)]
    pub struct BenchmarkReport {
        pub samples: Vec<BenchmarkSample>,
    }

    impl BenchmarkReport {
        /// Collapses the samples into per-backend statistics capturing mean, standard deviation,
        /// and an approximate 95th percentile latency estimate.
        pub fn summaries(&self) -> Vec<BenchmarkStats> {
            if self.samples.is_empty() {
                return Vec::new();
            }

            let mut buckets: HashMap<&str, Vec<&BenchmarkSample>> = HashMap::new();
            for sample in &self.samples {
                buckets.entry(&sample.backend).or_default().push(sample);
            }

            let mut summaries: Vec<BenchmarkStats> = buckets
                .into_iter()
                .map(|(backend, bucket)| {
                    let count = bucket.len() as f32;
                    let throughput_mean = bucket.iter().map(|s| s.throughput).sum::<f32>() / count;
                    let latency_mean = bucket.iter().map(|s| s.latency_ms).sum::<f32>() / count;
                    let throughput_std = bucket
                        .iter()
                        .map(|s| {
                            let diff = s.throughput - throughput_mean;
                            diff * diff
                        })
                        .sum::<f32>()
                        .max(0.0)
                        .sqrt()
                        / count.max(1.0).sqrt();

                    let mut latencies: Vec<f32> = bucket.iter().map(|s| s.latency_ms).collect();
                    latencies.sort_by(|a, b| a.total_cmp(b));
                    let idx = ((latencies.len() as f32 * 0.95).ceil() as usize).saturating_sub(1);
                    let latency_p95 = latencies.get(idx).copied().unwrap_or(latency_mean);

                    BenchmarkStats {
                        backend: backend.to_string(),
                        throughput_mean,
                        throughput_std,
                        latency_mean,
                        latency_p95,
                    }
                })
                .collect();

            summaries.sort_by(|a, b| a.backend.cmp(&b.backend));
            summaries
        }

        /// Returns the backend with the highest average throughput, if any samples exist.
        pub fn best_backend(&self) -> Option<BenchmarkStats> {
            self.throughput_leaderboard().into_iter().next()
        }

        /// Returns the backend with the highest mean latency, if any samples exist.
        pub fn slowest_backend(&self) -> Option<BenchmarkStats> {
            self.latency_leaderboard().into_iter().rev().next()
        }

        /// Returns summaries sorted by descending throughput.
        pub fn throughput_leaderboard(&self) -> Vec<BenchmarkStats> {
            let mut summaries = self.summaries();
            summaries.sort_by(|a, b| b.throughput_mean.total_cmp(&a.throughput_mean));
            summaries
        }

        /// Returns summaries sorted by ascending mean latency.
        pub fn latency_leaderboard(&self) -> Vec<BenchmarkStats> {
            let mut summaries = self.summaries();
            summaries.sort_by(|a, b| a.latency_mean.total_cmp(&b.latency_mean));
            summaries
        }

        /// Computes relative throughput speedups compared to `baseline`.
        pub fn throughput_speedups(&self, baseline: &str) -> Vec<(String, f32)> {
            let summaries = self.summaries();
            let Some(reference) = summaries
                .iter()
                .find(|stats| stats.backend.eq_ignore_ascii_case(baseline))
            else {
                return Vec::new();
            };

            let baseline_throughput = reference.throughput_mean.max(f32::EPSILON);
            let mut comparisons: Vec<(String, f32)> = summaries
                .into_iter()
                .map(|stats| {
                    let name = stats.backend.clone();
                    let speedup = stats.throughput_mean / baseline_throughput;
                    (name, speedup)
                })
                .collect();

            comparisons.sort_by(|a, b| b.1.total_cmp(&a.1));
            comparisons
        }
    }
}

pub use model::{BackendProbe, BenchmarkReport, BenchmarkSample, BenchmarkStats};

/// Canonical backend probes derived from the documentation matrix.
///
/// The heuristics mirror the expectations outlined in `docs/backend_matrix.md`:
/// CPU acts as the baseline, CUDA leads throughput, and HIP starts slightly
/// behind CUDA while the outstanding watchlist items are addressed.
pub fn canonical_backend_probes() -> Vec<BackendProbe<'static>> {
    vec![
        BackendProbe {
            name: Backend::Cpu.as_str(),
            base_throughput: 1.0,
            latency_ms: 9.5,
        },
        BackendProbe {
            name: Backend::Wgpu.as_str(),
            base_throughput: 1.25,
            latency_ms: 8.0,
        },
        BackendProbe {
            name: Backend::Mps.as_str(),
            base_throughput: 1.35,
            latency_ms: 6.5,
        },
        BackendProbe {
            name: Backend::Cuda.as_str(),
            base_throughput: 2.1,
            latency_ms: 4.2,
        },
        BackendProbe {
            name: Backend::Hip.as_str(),
            base_throughput: 1.7,
            latency_ms: 5.3,
        },
    ]
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
    benchmark_backends_with_trials(probes, steps, 1)
}

/// Benchmarks backend probes repeatedly to capture variability across multiple trials.
pub fn benchmark_backends_with_trials(
    probes: &[BackendProbe<'_>],
    steps: usize,
    trials: usize,
) -> BenchmarkReport {
    let trials = trials.max(1);
    let mut rng = StdRng::seed_from_u64((steps as u64 + 0x5A5A) ^ trials as u64);
    let mut samples = Vec::with_capacity(probes.len() * trials);

    for _ in 0..trials {
        for probe in probes {
            let jitter = rng.gen_range(0.9..=1.1);
            samples.push(BenchmarkSample {
                backend: probe.name.to_string(),
                throughput: probe.base_throughput * jitter * steps as f32,
                latency_ms: (probe.latency_ms / jitter).max(0.1),
            });
        }
    }

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

    #[test]
    fn benchmark_summary_orders_backends() {
        let probes = [
            BackendProbe {
                name: "wgpu",
                base_throughput: 1.0,
                latency_ms: 8.0,
            },
            BackendProbe {
                name: "cuda",
                base_throughput: 2.0,
                latency_ms: 4.0,
            },
        ];
        let report = benchmark_backends_with_trials(&probes, 4, 5);
        let summaries = report.summaries();
        assert_eq!(summaries.len(), 2);
        assert!(summaries[0].backend <= summaries[1].backend);
        for summary in summaries {
            assert!(summary.throughput_mean > 0.0);
            assert!(summary.latency_p95 >= summary.latency_mean);
        }
    }

    #[test]
    fn canonical_probes_cover_all_backends() {
        let probes = canonical_backend_probes();
        assert_eq!(probes.len(), Backend::COUNT);
        for backend in Backend::ALL {
            assert!(probes.iter().any(|probe| probe.name == backend.as_str()));
        }
    }

    #[test]
    fn report_speedup_sorted_by_throughput() {
        let report = BenchmarkReport {
            samples: vec![
                BenchmarkSample {
                    backend: "CPU".to_string(),
                    throughput: 10.0,
                    latency_ms: 8.0,
                },
                BenchmarkSample {
                    backend: "GPU".to_string(),
                    throughput: 25.0,
                    latency_ms: 5.0,
                },
                BenchmarkSample {
                    backend: "MPS".to_string(),
                    throughput: 15.0,
                    latency_ms: 6.0,
                },
            ],
        };

        let speedups = report.throughput_speedups("cpu");
        assert_eq!(speedups.len(), 3);
        assert_eq!(speedups[0].0, "GPU");
        assert!(speedups[0].1 > speedups[1].1);
        let cpu_entry = speedups
            .iter()
            .find(|(backend, _)| backend == "CPU")
            .expect("cpu entry present");
        assert!((cpu_entry.1 - 1.0).abs() <= f32::EPSILON);
    }

    #[test]
    fn best_backend_prefers_highest_throughput() {
        let report = BenchmarkReport {
            samples: vec![
                BenchmarkSample {
                    backend: "CPU".into(),
                    throughput: 11.0,
                    latency_ms: 9.0,
                },
                BenchmarkSample {
                    backend: "CUDA".into(),
                    throughput: 29.0,
                    latency_ms: 4.5,
                },
            ],
        };

        let best = report.best_backend().expect("best backend present");
        assert_eq!(best.backend, "CUDA");
        let slowest = report.slowest_backend().expect("slowest backend present");
        assert_eq!(slowest.backend, "CPU");
    }

    #[test]
    fn latency_leaderboard_orders_fastest_first() {
        let report = BenchmarkReport {
            samples: vec![
                BenchmarkSample {
                    backend: "CPU".into(),
                    throughput: 10.0,
                    latency_ms: 9.5,
                },
                BenchmarkSample {
                    backend: "WGPU".into(),
                    throughput: 18.0,
                    latency_ms: 6.5,
                },
                BenchmarkSample {
                    backend: "CUDA".into(),
                    throughput: 26.0,
                    latency_ms: 4.0,
                },
            ],
        };

        let leaderboard = report.latency_leaderboard();
        assert_eq!(leaderboard.first().unwrap().backend, "CUDA");
        assert_eq!(leaderboard.last().unwrap().backend, "CPU");
        let slowest = report.slowest_backend().unwrap();
        assert_eq!(
            slowest.latency_mean,
            leaderboard.last().unwrap().latency_mean
        );
    }
}
