#![allow(dead_code)]

/// Deterministic, backend-pluggable NeRF trainer used by `st-logic`.
///
/// The trainer intentionally stays lightweight: callers can keep using
/// `new()/train()` while advanced integrations swap in a custom kernel.
pub struct NerfTrainer {
    cfg: NerfTrainerConfig,
    kernel: Box<dyn NerfTrainKernel>,
    stats: NerfTrainingStats,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NerfTrainerConfig {
    /// Number of optimization steps performed by [`NerfTrainer::train`].
    pub default_steps: usize,
    /// Initial synthetic reconstruction loss.
    pub initial_loss: f32,
    /// Multiplicative decay applied by the default kernel.
    pub decay: f32,
    /// Lower bound that keeps the simulated loss numerically stable.
    pub floor: f32,
}

impl Default for NerfTrainerConfig {
    fn default() -> Self {
        Self {
            default_steps: 32,
            initial_loss: 1.0,
            decay: 0.97,
            floor: 1.0e-4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NerfTrainingStats {
    pub total_steps: usize,
    pub last_batch_steps: usize,
    pub loss: f32,
    pub best_loss: f32,
}

impl NerfTrainingStats {
    fn fresh(initial_loss: f32) -> Self {
        Self {
            total_steps: 0,
            last_batch_steps: 0,
            loss: initial_loss,
            best_loss: initial_loss,
        }
    }
}

pub trait NerfTrainKernel: Send {
    fn train_step(&mut self, step: usize, loss: f32) -> f32;
}

#[derive(Clone, Copy, Debug)]
struct ExponentialDecayKernel {
    decay: f32,
    floor: f32,
}

impl ExponentialDecayKernel {
    fn from_config(cfg: NerfTrainerConfig) -> Self {
        Self {
            decay: cfg.decay.clamp(0.5, 0.9999),
            floor: cfg.floor.max(0.0),
        }
    }
}

impl NerfTrainKernel for ExponentialDecayKernel {
    fn train_step(&mut self, _step: usize, loss: f32) -> f32 {
        (loss * self.decay).max(self.floor)
    }
}

impl NerfTrainer {
    /// Creates a trainer with a deterministic decay kernel.
    pub fn new() -> Self {
        Self::with_config(NerfTrainerConfig::default())
    }

    pub fn with_config(cfg: NerfTrainerConfig) -> Self {
        let stats = NerfTrainingStats::fresh(cfg.initial_loss.max(cfg.floor.max(0.0)));
        let kernel = Box::new(ExponentialDecayKernel::from_config(cfg));
        Self { cfg, kernel, stats }
    }

    pub fn with_kernel(cfg: NerfTrainerConfig, kernel: Box<dyn NerfTrainKernel>) -> Self {
        let stats = NerfTrainingStats::fresh(cfg.initial_loss.max(cfg.floor.max(0.0)));
        Self { cfg, kernel, stats }
    }

    pub fn set_kernel(&mut self, kernel: Box<dyn NerfTrainKernel>) {
        self.kernel = kernel;
    }

    pub fn train_steps(&mut self, steps: usize) -> NerfTrainingStats {
        if steps == 0 {
            self.stats.last_batch_steps = 0;
            return self.stats;
        }

        for local_step in 0..steps {
            let global_step = self.stats.total_steps + local_step;
            let next_loss = self
                .kernel
                .train_step(global_step, self.stats.loss)
                .max(self.cfg.floor);
            self.stats.loss = next_loss;
            self.stats.best_loss = self.stats.best_loss.min(next_loss);
        }

        self.stats.total_steps += steps;
        self.stats.last_batch_steps = steps;
        self.stats
    }

    pub fn train(&mut self) {
        let _ = self.train_steps(self.cfg.default_steps);
    }

    pub fn stats(&self) -> NerfTrainingStats {
        self.stats
    }
}

impl Default for NerfTrainer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct LinearKernel {
        delta: f32,
    }

    impl NerfTrainKernel for LinearKernel {
        fn train_step(&mut self, _step: usize, loss: f32) -> f32 {
            loss - self.delta
        }
    }

    #[test]
    fn default_train_reduces_loss() {
        let mut trainer = NerfTrainer::new();
        let before = trainer.stats().loss;
        trainer.train();
        let after = trainer.stats().loss;
        assert!(after < before);
        assert_eq!(
            trainer.stats().last_batch_steps,
            NerfTrainerConfig::default().default_steps
        );
    }

    #[test]
    fn custom_kernel_is_pluggable() {
        let cfg = NerfTrainerConfig {
            default_steps: 3,
            initial_loss: 1.0,
            decay: 0.97,
            floor: 0.0,
        };
        let mut trainer = NerfTrainer::with_kernel(cfg, Box::new(LinearKernel { delta: 0.1 }));
        trainer.train();
        let stats = trainer.stats();
        assert_eq!(stats.total_steps, 3);
        assert!((stats.loss - 0.7).abs() < 1e-6);
    }
}
