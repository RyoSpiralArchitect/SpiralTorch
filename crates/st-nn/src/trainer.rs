use crate::loss::Loss;
use crate::module::Module;
use crate::plan::RankPlanner;
use crate::schedule::{BandEnergy, GradientBands, RoundtableConfig, RoundtableSchedule};
use crate::{PureResult, Tensor};
use st_core::backend::device_caps::DeviceCaps;
use st_core::backend::unison_heuristics::RankKind;
use st_core::runtime::autopilot::Autopilot;
use st_core::runtime::blackcat::{BlackCatRuntime, StepMetrics};
use st_tensor::pure::topos::OpenCartesianTopos;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// High-level orchestrator that keeps hypergrad, SpiralK, and module updates aligned.
pub struct ModuleTrainer {
    planner: RankPlanner,
    curvature: f32,
    hyper_learning_rate: f32,
    fallback_learning_rate: f32,
    blackcat: Option<BlackCatRuntime>,
    autopilot: Option<Autopilot>,
    band_weight_fn: Option<BandWeightFn>,
    injector_enabled: bool,
}

impl core::fmt::Debug for ModuleTrainer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "ModuleTrainer(curv={},lr_h={},lr_f={})",
            self.curvature, self.hyper_learning_rate, self.fallback_learning_rate
        )
    }
}

/// Function pointer used to convert band energy into Above/Here/Beneath weights.
pub type BandWeightFn = fn(BandEnergy) -> (f32, f32, f32);

impl ModuleTrainer {
    /// Creates a new trainer with the provided device capabilities and learning rates.
    pub fn new(
        caps: DeviceCaps,
        curvature: f32,
        hyper_learning_rate: f32,
        fallback_learning_rate: f32,
    ) -> Self {
        Self {
            planner: RankPlanner::new(caps),
            curvature,
            hyper_learning_rate,
            fallback_learning_rate,
            blackcat: None,
            autopilot: None,
            band_weight_fn: None,
            injector_enabled: false,
        }
    }

    /// Attaches the BlackCat runtime so contextual rewards update after each step.
    pub fn with_blackcat(mut self, runtime: BlackCatRuntime) -> Self {
        self.blackcat = Some(runtime);
        self
    }

    /// Attaches an Autopilot runtime for contextual kernel selection.
    pub fn with_autopilot(mut self, autopilot: Autopilot) -> Self {
        self.autopilot = Some(autopilot);
        self
    }

    /// Enables per-band reweighting of the loss/gradient.
    pub fn set_band_weights(&mut self, weight_fn: BandWeightFn) {
        self.band_weight_fn = Some(weight_fn);
    }

    /// Clears any registered band weighting rule.
    pub fn clear_band_weights(&mut self) {
        self.band_weight_fn = None;
    }

    /// Enables or disables the adaptive injector heuristics.
    pub fn enable_injector(&mut self, on: bool) {
        self.injector_enabled = on;
    }

    /// Returns the underlying planner.
    pub fn planner(&self) -> &RankPlanner {
        &self.planner
    }

    /// Produces a roundtable schedule for the provided output dimensions.
    pub fn roundtable(&self, rows: u32, cols: u32, config: RoundtableConfig) -> RoundtableSchedule {
        RoundtableSchedule::new(&self.planner, rows, cols, config)
    }

    /// Returns the fallback Euclidean learning rate.
    pub fn fallback_learning_rate(&self) -> f32 {
        self.fallback_learning_rate
    }

    /// Returns the curvature used for hypergrad preparation.
    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    /// Attaches hypergrad tapes to all parameters of the provided module.
    pub fn prepare<M: Module>(&self, module: &mut M) -> PureResult<()> {
        module.attach_hypergrad(self.curvature, self.hyper_learning_rate)
    }

    /// Attaches hypergrad tapes with an explicit topos shared across parameters.
    pub fn prepare_with_topos<M: Module>(
        &self,
        module: &mut M,
        topos: OpenCartesianTopos,
    ) -> PureResult<()> {
        module.attach_hypergrad_with_topos(self.curvature, self.hyper_learning_rate, topos)
    }

    /// Clears accumulated gradients or hypergrad buffers.
    pub fn zero<M: Module>(&self, module: &mut M) -> PureResult<()> {
        module.zero_accumulators()
    }

    /// Applies the parameter updates using either the hypergrad tape or the fallback rate.
    pub fn step<M: Module>(&self, module: &mut M) -> PureResult<()> {
        module.apply_step(self.fallback_learning_rate)
    }

    /// Runs a full epoch over the provided iterator of `(input, target)` pairs.
    pub fn train_epoch<M, L, I>(
        &mut self,
        module: &mut M,
        loss: &mut L,
        batches: I,
        schedule: &RoundtableSchedule,
    ) -> PureResult<EpochStats>
    where
        M: Module,
        L: Loss,
        I: IntoIterator<Item = (Tensor, Tensor)>,
    {
        self.zero(module)?;
        let mut total_loss = 0.0f32;
        let mut steps = 0usize;
        for (input, target) in batches.into_iter() {
            let step_start = Instant::now();
            if let Some(rt) = self.blackcat.as_mut() {
                rt.begin_step();
            }
            let device_load = self.estimate_device_load();
            if let Some(ap) = self.autopilot.as_mut() {
                let (rows, cols) = input.shape();
                let depth = schedule.above().k + schedule.here().k + schedule.beneath().k;
                let context = ap.build_context(rows as u32, cols as u32, depth, device_load, &[]);
                let _ = ap.suggest(context);
            }
            let prediction = module.forward(&input)?;
            let loss_value = loss.forward(&prediction, &target)?;
            let step_loss = loss_value.data().iter().copied().sum::<f32>();
            let grad_output = loss.backward(&prediction, &target)?;
            let mut band_energy = schedule.band_energy(&grad_output)?;
            if let Some(rt) = self.blackcat.as_ref() {
                band_energy.drift = rt.frac_penalty() as f32;
            }
            let mut bands: GradientBands = schedule.split(&grad_output)?;
            let weights = self
                .band_weight_fn
                .map(|f| f(band_energy))
                .unwrap_or((1.0, 1.0, 1.0));
            bands.scale_inplace(weights.0, weights.1, weights.2);
            let weighted_loss = if self.band_weight_fn.is_some() {
                let effective = weights.1 + 0.5 * (weights.0 + weights.2);
                step_loss * effective.max(0.0)
            } else {
                step_loss
            };
            total_loss += weighted_loss;
            let _ = module.backward_bands(&input, &bands)?;
            self.step(module)?;
            self.zero(module)?;
            steps += 1;

            let elapsed_ms = if let Some(rt) = self.blackcat.as_ref() {
                rt.elapsed_since_begin()
                    .unwrap_or_else(|| Duration::from_secs_f64(0.0))
                    .as_secs_f64()
                    * 1_000.0
            } else {
                step_start.elapsed().as_secs_f64() * 1_000.0
            };
            let mut extra = HashMap::new();
            extra.insert("band_above".to_string(), band_energy.above as f64);
            extra.insert("band_here".to_string(), band_energy.here as f64);
            extra.insert("band_beneath".to_string(), band_energy.beneath as f64);
            extra.insert("band_drift".to_string(), band_energy.drift as f64);
            extra.insert("step_loss".to_string(), step_loss as f64);
            extra.insert("loss_weighted".to_string(), weighted_loss as f64);
            let metrics = StepMetrics {
                step_time_ms: elapsed_ms,
                mem_peak_mb: 0.0,
                retry_rate: 0.0,
                extra,
            };
            if let Some(ap) = self.autopilot.as_mut() {
                ap.report(&metrics);
            }
            if let Some(rt) = self.blackcat.as_mut() {
                let reward = rt.post_step(&metrics);
                if reward > 0.0 {
                    let plan = schedule.above();
                    let script = plan
                        .choice
                        .to_unison_script(RankKind::TopK)
                        .replace('\n', "; ");
                    let _ = rt.try_adopt_soft(&script, 1, 1, 0.5);
                }
            }
        }
        Ok(EpochStats {
            batches: steps,
            total_loss,
            average_loss: if steps == 0 {
                0.0
            } else {
                total_loss / steps as f32
            },
        })
    }

    fn estimate_device_load(&self) -> f64 {
        let caps = self.planner.device_caps();
        caps.occupancy_score(caps.max_workgroup) as f64
    }
}

/// Metrics captured while running [`ModuleTrainer::train_epoch`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochStats {
    pub batches: usize,
    pub total_loss: f32,
    pub average_loss: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::linear::Linear;
    use crate::layers::sequential::Sequential;
    use crate::layers::wave_gate::WaveGate;
    use crate::loss::MeanSquaredError;
    use crate::schedule::RoundtableConfig;
    use st_tensor::pure::topos::OpenCartesianTopos;

    #[test]
    fn trainer_attaches_and_steps() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut layer = Linear::new("fc", 2, 1).unwrap();
        trainer.prepare(&mut layer).unwrap();
        let input = crate::Tensor::from_vec(1, 2, vec![1.0, -1.0]).unwrap();
        let target = crate::Tensor::from_vec(1, 1, vec![0.5]).unwrap();
        let out = layer.forward(&input).unwrap();
        let grad = out.sub(&target).unwrap();
        let _ = layer.backward(&input, &grad).unwrap();
        trainer.step(&mut layer).unwrap();
        assert!(trainer.planner().topk(64, 128, 32).k > 0);
    }

    #[test]
    fn trainer_prepares_with_topos_for_wave_gate() {
        let caps = DeviceCaps::wgpu(64, true, 512);
        let trainer = ModuleTrainer::new(caps, -0.9, 0.06, 0.02);
        let encoder_curvature = trainer.curvature();
        let topos = OpenCartesianTopos::new(encoder_curvature, 1e-6, 1e4, 512, 16384).unwrap();
        let mut gate = WaveGate::with_topos(
            "wg",
            8,
            st_tensor::pure::LanguageWaveEncoder::new(encoder_curvature, 0.7).unwrap(),
            topos.clone(),
        )
        .unwrap();
        trainer.prepare_with_topos(&mut gate, topos).unwrap();
        let input =
            Tensor::from_vec(1, 8, vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]).unwrap();
        let grad_out = gate.forward(&input).unwrap();
        let _ = gate.backward(&input, &grad_out).unwrap();
        trainer.step(&mut gate).unwrap();
        assert!(gate.gate().value().squared_l2_norm() > 0.0);
    }

    #[test]
    fn trainer_runs_epoch_with_roundtable_schedule() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.1, 0.05, 0.01);
        let mut model = Sequential::new();
        model.push(Linear::new("lin", 2, 1).unwrap());
        trainer.prepare(&mut model).unwrap();

        let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
        let dataset = vec![
            (
                Tensor::from_vec(1, 2, vec![0.0, 1.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            ),
            (
                Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
            ),
        ];

        let mut loss = MeanSquaredError::new();
        let stats = trainer
            .train_epoch(&mut model, &mut loss, dataset.clone(), &schedule)
            .unwrap();
        assert_eq!(stats.batches, dataset.len());
        assert!(stats.total_loss.is_finite());

        // Ensure the model parameters changed by running another batch and checking the outputs.
        let input = Tensor::from_vec(1, 2, vec![1.0, 1.0]).unwrap();
        let before = model.forward(&input).unwrap();
        trainer
            .train_epoch(&mut model, &mut loss, dataset, &schedule)
            .unwrap();
        let after = model.forward(&input).unwrap();
        assert_ne!(before.data(), after.data());
    }
}
