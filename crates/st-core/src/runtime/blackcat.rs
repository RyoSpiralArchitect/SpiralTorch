// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use bandit::{SoftBandit, SoftBanditMode};
use rewrite::HeurStore;
use st_frac::FracBackend;
use wilson::wilson_lower;
use zmeta::{ZMetaES, ZMetaParams};

/// Metrics reported by a training loop back into the runtime.
#[derive(Clone, Debug, Default)]
pub struct StepMetrics {
    pub step_time_ms: f64,
    pub mem_peak_mb: f64,
    pub retry_rate: f64,
    pub extra: HashMap<String, f64>,
}

/// Reward shaping configuration that blends timing, memory, and stability.
#[derive(Clone, Debug)]
pub struct RewardCfg {
    pub lam_speed: f64,
    pub lam_mem: f64,
    pub lam_stab: f64,
    pub scale_speed: f64,
    pub scale_mem: f64,
    pub scale_stab: f64,
}

impl Default for RewardCfg {
    fn default() -> Self {
        Self {
            lam_speed: 0.5,
            lam_mem: 0.3,
            lam_stab: 0.2,
            scale_speed: 10.0,
            scale_mem: 1024.0,
            scale_stab: 1.0,
        }
    }
}

impl RewardCfg {
    pub fn score(&self, metrics: &StepMetrics, frac_penalty: f64) -> f64 {
        let mut s = 0.0;
        s -= self.lam_speed * (metrics.step_time_ms / self.scale_speed.max(1e-9));
        s -= self.lam_mem * (metrics.mem_peak_mb / self.scale_mem.max(1e-9));
        s -= self.lam_stab * (metrics.retry_rate / self.scale_stab.max(1e-9));
        s -= frac_penalty;
        s
    }
}

/// Named groups of candidate choices (tile, merge strategy, etc.).
#[derive(Clone, Debug)]
pub struct ChoiceGroups {
    pub groups: HashMap<String, Vec<String>>,
}

/// Multi-armed contextual bandit that operates over named groups.
pub struct MultiBandit {
    arms: HashMap<String, SoftBandit>,
}

impl MultiBandit {
    pub fn new(groups: &ChoiceGroups, feat_dim: usize, mode: SoftBanditMode) -> Self {
        let mut arms = HashMap::new();
        for (name, opts) in &groups.groups {
            arms.insert(name.clone(), SoftBandit::new(opts.clone(), feat_dim, mode));
        }
        Self { arms }
    }

    pub fn select_all(&mut self, context: &[f64]) -> HashMap<String, String> {
        let mut picks = HashMap::new();
        for (name, bandit) in self.arms.iter_mut() {
            let choice = bandit.select(context);
            picks.insert(name.clone(), choice);
        }
        picks
    }

    pub fn update_all(&mut self, context: &[f64], reward: f64) {
        for bandit in self.arms.values_mut() {
            bandit.update_last(context, reward);
        }
    }
}

/// BlackCat orchestrator that joins ES search with contextual bandits.
pub struct BlackCatRuntime {
    pub z: ZMetaES,
    pub bandits: MultiBandit,
    pub heur: HeurStore,
    pub reward: RewardCfg,
    context_dim: usize,
    last_context: Vec<f64>,
    last_picks: HashMap<String, String>,
    last_step_start: Option<Instant>,
    stats_alpha: f64,
    stats_steps: u64,
    reward_mean: f64,
    reward_m2: f64,
    last_reward: f64,
    metrics_ema: MetricsEma,
    frac_penalty_ema: RollingEma,
    extra_ema: HashMap<String, RollingEma>,
}

impl BlackCatRuntime {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        z_params: ZMetaParams,
        groups: ChoiceGroups,
        feat_dim: usize,
        mode: SoftBanditMode,
        heur_path: Option<String>,
    ) -> Self {
        let bandits = MultiBandit::new(&groups, feat_dim, mode);
        let heur = HeurStore::new(heur_path);
        let stats_alpha = 0.2;
        Self {
            z: ZMetaES::new(z_params),
            bandits,
            heur,
            reward: RewardCfg::default(),
            context_dim: feat_dim.max(1),
            last_context: vec![0.0; feat_dim.max(1)],
            last_picks: HashMap::new(),
            last_step_start: None,
            stats_alpha,
            stats_steps: 0,
            reward_mean: 0.0,
            reward_m2: 0.0,
            last_reward: 0.0,
            metrics_ema: MetricsEma::new(stats_alpha),
            frac_penalty_ema: RollingEma::new(stats_alpha),
            extra_ema: HashMap::new(),
        }
    }

    /// Call at the beginning of a training step.
    pub fn begin_step(&mut self) {
        self.last_step_start = Some(Instant::now());
    }

    /// Build a contextual feature vector from runtime metrics.
    #[allow(
        clippy::too_many_arguments,
        reason = "Context vector requires these inputs for bandit feature parity"
    )]
    pub fn make_context(
        &self,
        batches: u32,
        tiles: u32,
        depth: u32,
        device_code: u32,
        load: f64,
        extras: &[(String, f64)],
        feat_dim: usize,
    ) -> Vec<f64> {
        let target_dim = if feat_dim == 0 {
            self.context_dim
        } else {
            feat_dim
        };
        let mut ctx = vec![
            1.0,
            batches as f64,
            tiles as f64,
            depth as f64,
            (device_code % 1024) as f64 / 1024.0,
            load,
        ];
        for (_, value) in extras.iter() {
            ctx.push(*value);
        }
        ctx.resize(target_dim, 0.0);
        ctx
    }

    /// Choose all groups at once, storing the picks and context internally.
    pub fn choose(&mut self, context: Vec<f64>) -> HashMap<String, String> {
        let picks = self.bandits.select_all(&context);
        self.context_dim = context.len().max(1);
        self.last_context = context;
        self.last_picks = picks.clone();
        picks
    }

    /// Update both the ES search and contextual bandits after a step.
    pub fn post_step(&mut self, metrics: &StepMetrics) -> f64 {
        let curr_penalty = self.z.frac_penalty();
        let reward_current = self.reward.score(metrics, curr_penalty);
        let proposed_penalty = self.z.frac_penalty_proposed();
        self.z.update(
            reward_current,
            reward_current - (proposed_penalty - curr_penalty),
            Some(&self.last_context),
        );
        let grad_norm = metrics.extra.get("grad_norm").copied().unwrap_or(0.0);
        let loss_var = metrics
            .extra
            .get("loss_var")
            .or_else(|| metrics.extra.get("loss_variance"))
            .copied()
            .unwrap_or(1.0);
        self.z
            .temp_schedule(metrics.retry_rate, grad_norm, loss_var);
        self.bandits.update_all(&self.last_context, reward_current);
        self.stats_steps = self.stats_steps.saturating_add(1);
        let delta = reward_current - self.reward_mean;
        if self.stats_steps == 1 {
            self.reward_mean = reward_current;
            self.reward_m2 = 0.0;
        } else {
            self.reward_mean += delta / self.stats_steps as f64;
            let delta2 = reward_current - self.reward_mean;
            self.reward_m2 += delta * delta2;
        }
        self.last_reward = reward_current;
        self.metrics_ema.update(metrics);
        self.frac_penalty_ema.update(curr_penalty);
        for (key, value) in metrics.extra.iter() {
            self.extra_ema
                .entry(key.clone())
                .or_insert_with(|| RollingEma::new(self.stats_alpha))
                .update(*value);
        }
        reward_current
    }

    /// Returns the dimensionality expected by the contextual bandits.
    pub fn context_dim(&self) -> usize {
        self.context_dim
    }

    /// Returns the current fractional regularisation penalty tracked by ZMeta.
    pub fn frac_penalty(&self) -> f64 {
        self.z.frac_penalty()
    }

    /// Overrides the fractional regulariser backend.
    pub fn set_frac_backend(&mut self, backend: FracBackend) {
        self.z.set_frac_backend(backend);
    }

    /// Returns the current exploration temperature tracked by the runtime.
    pub fn temperature(&self) -> f64 {
        self.z.temperature()
    }

    /// Try to adopt a new soft heuristic guarded by the Wilson lower bound.
    pub fn try_adopt_soft(
        &mut self,
        rule_text: &str,
        wins: u32,
        trials: u32,
        baseline_p: f64,
    ) -> bool {
        let lb = wilson_lower(wins as i32, trials as i32, 1.96);
        if lb > baseline_p {
            let mut info = HashMap::new();
            info.insert("wins".to_string(), wins as f64);
            info.insert("trials".to_string(), trials as f64);
            self.heur
                .append(&format!("{}  # blackcat", rule_text.trim()), &info);
            return true;
        }
        false
    }

    /// Returns the duration since the last [`begin_step`] call.
    pub fn elapsed_since_begin(&self) -> Option<Duration> {
        self.last_step_start.map(|start| start.elapsed())
    }

    /// Returns the last contextual feature vector used for bandit updates.
    pub fn last_context(&self) -> &[f64] {
        &self.last_context
    }

    /// Returns the picks that were selected during the last [`choose`] call.
    pub fn last_picks(&self) -> &HashMap<String, String> {
        &self.last_picks
    }

    /// Returns aggregated runtime statistics derived from recent updates.
    pub fn stats(&self) -> BlackcatRuntimeStats {
        let reward_std = if self.stats_steps > 1 {
            (self.reward_m2 / (self.stats_steps - 1) as f64)
                .abs()
                .sqrt()
        } else {
            0.0
        };
        let extras = self
            .extra_ema
            .iter()
            .filter_map(|(key, ema)| ema.value().map(|value| (key.clone(), value)))
            .collect();
        BlackcatRuntimeStats {
            steps: self.stats_steps,
            reward_mean: self.reward_mean,
            reward_std,
            last_reward: self.last_reward,
            step_time_ms_ema: self.metrics_ema.step_time(),
            mem_peak_mb_ema: self.metrics_ema.mem_peak(),
            retry_rate_ema: self.metrics_ema.retry_rate(),
            frac_penalty_ema: self
                .frac_penalty_ema
                .value()
                .unwrap_or_else(|| self.z.frac_penalty()),
            extras,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct BlackcatRuntimeStats {
    pub steps: u64,
    pub reward_mean: f64,
    pub reward_std: f64,
    pub last_reward: f64,
    pub step_time_ms_ema: f64,
    pub mem_peak_mb_ema: f64,
    pub retry_rate_ema: f64,
    pub frac_penalty_ema: f64,
    pub extras: HashMap<String, f64>,
}

#[derive(Clone, Debug)]
struct MetricsEma {
    step_time: RollingEma,
    mem_peak: RollingEma,
    retry_rate: RollingEma,
}

impl MetricsEma {
    fn new(alpha: f64) -> Self {
        Self {
            step_time: RollingEma::new(alpha),
            mem_peak: RollingEma::new(alpha),
            retry_rate: RollingEma::new(alpha),
        }
    }

    fn update(&mut self, metrics: &StepMetrics) {
        self.step_time.update(metrics.step_time_ms);
        self.mem_peak.update(metrics.mem_peak_mb);
        self.retry_rate.update(metrics.retry_rate);
    }

    fn step_time(&self) -> f64 {
        self.step_time.value().unwrap_or(0.0)
    }

    fn mem_peak(&self) -> f64 {
        self.mem_peak.value().unwrap_or(0.0)
    }

    fn retry_rate(&self) -> f64 {
        self.retry_rate.value().unwrap_or(0.0)
    }
}

#[derive(Clone, Debug)]
struct RollingEma {
    alpha: f64,
    value: f64,
    initialized: bool,
}

impl RollingEma {
    fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(1.0e-3, 0.999),
            value: 0.0,
            initialized: false,
        }
    }

    fn update(&mut self, sample: f64) {
        if !sample.is_finite() {
            return;
        }
        if !self.initialized {
            self.value = sample;
            self.initialized = true;
        } else {
            self.value = self.alpha * sample + (1.0 - self.alpha) * self.value;
        }
    }

    fn value(&self) -> Option<f64> {
        if self.initialized {
            Some(self.value)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn sample_runtime() -> BlackCatRuntime {
        let mut groups = HashMap::new();
        groups.insert("tile".to_string(), vec!["a".to_string(), "b".to_string()]);
        let groups = ChoiceGroups { groups };
        BlackCatRuntime::new(ZMetaParams::default(), groups, 4, SoftBanditMode::TS, None)
    }

    #[test]
    fn runtime_accumulates_stats() {
        let mut runtime = sample_runtime();
        runtime.begin_step();
        let mut metrics = StepMetrics::default();
        metrics.step_time_ms = 12.5;
        metrics.mem_peak_mb = 512.0;
        metrics.retry_rate = 0.1;
        metrics.extra.insert("grad_norm".into(), 0.5);
        let reward1 = runtime.post_step(&metrics);
        let stats1 = runtime.stats();
        assert_eq!(stats1.steps, 1);
        assert!((stats1.reward_mean - reward1).abs() < 1e-9);
        assert_eq!(stats1.reward_std, 0.0);
        assert!(stats1.step_time_ms_ema > 0.0);
        assert!(stats1.mem_peak_mb_ema > 0.0);
        assert_eq!(stats1.extras.get("grad_norm").cloned().unwrap(), 0.5);

        runtime.begin_step();
        let mut metrics2 = StepMetrics::default();
        metrics2.step_time_ms = 6.0;
        metrics2.mem_peak_mb = 256.0;
        metrics2.retry_rate = 0.05;
        metrics2.extra.insert("grad_norm".into(), 0.25);
        let _ = runtime.post_step(&metrics2);
        let stats2 = runtime.stats();
        assert_eq!(stats2.steps, 2);
        assert!(stats2.reward_std >= 0.0);
        assert!(stats2.step_time_ms_ema <= stats1.step_time_ms_ema);
        assert!(stats2.extras.get("grad_norm").cloned().unwrap() <= 0.5);
        assert!(stats2.frac_penalty_ema >= 0.0);
    }
}

// =================== zmeta.rs ===================
pub mod zmeta {
    use super::FracBackend;
    use randless::{Rng, StdRng};

    #[derive(Clone, Debug)]
    pub struct ZMetaParams {
        pub dim: usize,
        pub sigma: f64,
        pub lr: f64,
        pub alpha_frac: f64,
        pub lam_frac: f64,
        pub orientation_eta: f64,
        pub orientation_eps: f64,
        pub seed: u64,
    }

    impl Default for ZMetaParams {
        fn default() -> Self {
            Self {
                dim: 6,
                sigma: 0.15,
                lr: 0.1,
                alpha_frac: 0.35,
                lam_frac: 0.1,
                orientation_eta: 0.15,
                orientation_eps: 1e-3,
                seed: 42,
            }
        }
    }

    pub struct ZMetaES {
        z: Vec<f64>,
        dir: Vec<f64>,
        params: ZMetaParams,
        rng: StdRng,
        frac_backend: FracBackend,
        temp: f64,
        temp_min: f64,
        temp_max: f64,
        structural: Vec<f64>,
    }

    impl ZMetaES {
        pub fn new(params: ZMetaParams) -> Self {
            let rng = StdRng::seed_from_u64(params.seed);
            let mut dir = vec![0.0; params.dim];
            for value in dir.iter_mut() {
                *value = rng.gauss(0.0, 1.0);
            }
            normalize(&mut dir);
            let dim = params.dim;
            Self {
                z: vec![0.0; dim],
                dir,
                params,
                rng,
                frac_backend: FracBackend::CpuRadix2,
                temp: 1.0,
                temp_min: 0.1,
                temp_max: 4.0,
                structural: vec![0.0; dim],
            }
        }

        pub fn z(&self) -> &[f64] {
            &self.z
        }

        /// Returns the current exploration temperature.
        pub fn temperature(&self) -> f64 {
            self.temp
        }

        /// Overrides the exploration temperature bounds.
        pub fn set_temp_bounds(&mut self, t_min: f64, t_max: f64) {
            let mut lower = t_min.max(0.0);
            let mut upper = t_max.max(lower + f64::EPSILON);
            if lower > upper {
                std::mem::swap(&mut lower, &mut upper);
            }
            self.temp_min = lower;
            self.temp_max = upper;
            self.temp = self.temp.clamp(self.temp_min, self.temp_max);
        }

        /// Adjusts the exploration temperature using retry/gradient signals.
        pub fn temp_schedule(&mut self, retry: f64, grad_norm: f64, loss_var: f64) {
            let stagnation = (1.0 - (loss_var / (1.0 + loss_var))).clamp(0.0, 1.0);
            let grad_term = (grad_norm / (1.0 + grad_norm)).clamp(0.0, 1.0);
            let instability = retry + 0.5 * grad_term;
            let delta = 0.2 * stagnation - 0.3 * instability;
            self.temp = (self.temp + delta).clamp(self.temp_min, self.temp_max);
        }

        /// Sets the backend used for fractional regularisation.
        pub fn set_frac_backend(&mut self, backend: FracBackend) {
            self.frac_backend = backend;
        }

        pub fn frac_penalty(&self) -> f64 {
            frac_penalty_backend(
                &self.z,
                self.params.alpha_frac,
                self.params.lam_frac,
                &self.frac_backend,
            )
        }

        pub fn frac_penalty_proposed(&self) -> f64 {
            let proposed: Vec<f64> = self
                .z
                .iter()
                .zip(self.dir.iter())
                .map(|(z, d)| z + self.params.sigma * d)
                .collect();
            frac_penalty_backend(
                &proposed,
                self.params.alpha_frac,
                self.params.lam_frac,
                &self.frac_backend,
            )
        }

        pub fn update(
            &mut self,
            reward_current: f64,
            reward_proposed: f64,
            structural: Option<&[f64]>,
        ) {
            let delta_reward = reward_proposed - reward_current;
            let structural_delta = self.ingest_structural(structural);
            let improved = delta_reward > 0.0;

            if improved {
                for (z, d) in self.z.iter_mut().zip(self.dir.iter()) {
                    *z += self.params.lr * (self.params.sigma * d);
                }
                for dir in self.dir.iter_mut() {
                    *dir = 0.7 * (*dir) + 0.3 * self.rng.gauss(0.0, 1.0);
                }
            } else {
                let mut kick: Vec<f64> = (0..self.params.dim)
                    .map(|_| self.rng.gauss(0.0, 0.5))
                    .collect();
                let projection = dot(&kick, &self.dir);
                for (k, d) in kick.iter_mut().zip(self.dir.iter()) {
                    *k -= projection * d;
                }
                for (dir, kick) in self.dir.iter_mut().zip(kick.iter()) {
                    *dir = 0.9 * (*dir) + 0.1 * (*kick);
                }
            }

            normalize(&mut self.dir);

            if let Some(delta) = structural_delta {
                self.apply_structural_drive(delta, delta_reward);
            }
        }

        #[cfg(not(feature = "blackcat_v2"))]
        fn ingest_structural(&mut self, structural: Option<&[f64]>) -> Option<Vec<f64>> {
            let raw = structural?;
            if self.params.dim == 0 {
                return None;
            }

            let mut new_vec = vec![0.0f64; self.params.dim];
            let mut any = false;
            for (idx, slot) in new_vec.iter_mut().enumerate() {
                if let Some(value) = raw.get(idx).copied() {
                    if value.is_finite() {
                        *slot = value;
                        any |= value.abs() > 1e-12;
                    }
                }
            }
            if !any {
                return None;
            }

            let norm = (new_vec.iter().map(|v| v * v).sum::<f64>()).sqrt();
            if norm <= 1e-9 {
                return None;
            }
            for value in new_vec.iter_mut() {
                *value /= norm;
            }

            let prev = self.structural.clone();
            self.structural = new_vec;
            Some(
                self.structural
                    .iter()
                    .zip(prev.iter())
                    .map(|(new, old)| new - old)
                    .collect(),
            )
        }

        #[cfg(not(feature = "blackcat_v2"))]
        fn apply_structural_drive(&mut self, mut delta: Vec<f64>, delta_reward: f64) {
            if delta_reward.abs() <= 1e-9 {
                return;
            }
            let gain = delta_reward.tanh();
            if !gain.is_finite() || gain.abs() <= 1e-6 {
                return;
            }

            let delta_norm = (delta.iter().map(|v| v * v).sum::<f64>()).sqrt();
            if delta_norm <= 1e-9 {
                return;
            }

            for value in delta.iter_mut() {
                *value *= gain;
            }

            // legacy名を参照せず、現行の投影ステップを使う
            self.logistic_project_step(&delta);
        }

        #[cfg(not(feature = "blackcat_v2"))]
        fn logistic_project_step(&mut self, drive: &[f64]) {
            if self.dir.is_empty() {
                return;
            }

            let structural_norm_sq = self.structural.iter().map(|v| v * v).sum::<f64>();
            if structural_norm_sq <= 1e-12 {
                return;
            }

            let mut projected = Vec::with_capacity(self.dir.len());
            let dot_nd = dot(&self.dir, drive);
            for (n, d) in self.dir.iter().zip(drive.iter()) {
                projected.push(d - dot_nd * n);
            }

            let proj_norm_sq = projected.iter().map(|v| v * v).sum::<f64>();
            if proj_norm_sq <= 1e-12 {
                return;
            }

            // unused 警告を抑制（意味は変えない）
            let _proj_norm = proj_norm_sq.sqrt();

            let dot_nr = self
                .dir
                .iter()
                .zip(self.structural.iter())
                .map(|(n, r)| n * r)
                .sum::<f64>()
                .clamp(-1.0, 1.0);
            let p_t = 0.5 * (1.0 + dot_nr);
            let eps = self.params.orientation_eps.max(1e-6);
            let denom = 2.0 * p_t * (1.0 - p_t) + eps;
            let eta = self.params.orientation_eta.max(0.0);
            if eta <= 0.0 {
                return;
            }

            for (n, proj) in self.dir.iter_mut().zip(projected.iter()) {
                *n += eta * (*proj) / denom;
            }
            normalize(&mut self.dir);
        }

        #[cfg(feature = "blackcat_v2")]
        fn ingest_structural(&mut self, structural: Option<&[f64]>) -> Option<Vec<f64>> {
            let Some(raw) = structural else {
                return None;
            };
            if self.params.dim == 0 {
                return None;
            }

            let mut new_vec = vec![0.0f64; self.params.dim];
            let mut any = false;
            for (idx, slot) in new_vec.iter_mut().enumerate() {
                if let Some(value) = raw.get(idx).copied() {
                    if value.is_finite() {
                        *slot = value;
                        any |= value.abs() > 1e-12;
                    }
                }
            }
            if !any {
                return None;
            }

            let norm = (new_vec.iter().map(|v| v * v).sum::<f64>()).sqrt();
            if norm <= 1e-9 {
                return None;
            }
            for value in new_vec.iter_mut() {
                *value /= norm;
            }

            let prev = self.structural.clone();
            self.structural = new_vec;
            Some(
                self.structural
                    .iter()
                    .zip(prev.iter())
                    .map(|(new, old)| new - old)
                    .collect(),
            )
        }

        #[cfg(feature = "blackcat_v2")]
        fn apply_structural_drive(&mut self, mut delta: Vec<f64>, delta_reward: f64) {
            if delta_reward.abs() <= 1e-9 {
                return;
            }
            let gain = delta_reward.tanh();
            if !gain.is_finite() || gain.abs() <= 1e-6 {
                return;
            }

            let delta_norm = (delta.iter().map(|v| v * v).sum::<f64>()).sqrt();
            if delta_norm <= 1e-9 {
                return;
            }

            for value in delta.iter_mut() {
                *value *= gain;
            }

            self.logistic_project_step(&delta);
        }

        #[cfg(feature = "blackcat_v2")]
        fn logistic_project_step(&mut self, drive: &[f64]) {
            if self.dir.is_empty() {
                return;
            }

            let structural_norm_sq = self.structural.iter().map(|v| v * v).sum::<f64>();
            if structural_norm_sq <= 1.0e-12 {
                return;
            }

            let mut projected = Vec::with_capacity(self.dir.len());
            let dot_nd = dot(&self.dir, drive);
            for (n, d) in self.dir.iter().zip(drive.iter()) {
                projected.push(d - dot_nd * n);
            }

            let proj_norm = (projected.iter().map(|v| v * v).sum::<f64>()).sqrt();
            if proj_norm <= 1.0e-12 {
                return;
            }

            let scale = 1.0 / proj_norm.max(1.0);
            for value in projected.iter_mut() {
                *value *= scale;
            }

            let dot_nr = self
                .dir
                .iter()
                .zip(self.structural.iter())
                .map(|(n, r)| n * r)
                .sum::<f64>()
                .clamp(-1.0, 1.0);
            let p_t = 0.5 * (1.0 + dot_nr);
            let eps = self.params.orientation_eps.max(1e-6);
            let denom = 2.0 * p_t * (1.0 - p_t) + eps;
            let eta = self.params.orientation_eta.max(0.0);
            if eta <= 0.0 {
                return;
            }

            for (n, proj) in self.dir.iter_mut().zip(projected.iter()) {
                *n += eta * (*proj) / denom;
            }
            normalize(&mut self.dir);
        }
    }

    fn normalize(vec: &mut [f64]) {
        let norm = (vec.iter().map(|v| v * v).sum::<f64>()).sqrt().max(1.0e-12);
        for v in vec.iter_mut() {
            *v /= norm;
        }
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn frac_penalty(z: &[f64], alpha: f64, lam: f64) -> f64 {
        if z.len() < 3 {
            return 0.0;
        }
        let mut acc = 0.0;
        for idx in 1..(z.len() - 1) {
            let d2 = z[idx - 1] - 2.0 * z[idx] + z[idx + 1];
            acc += d2.abs().powf(1.0 + alpha);
        }
        lam * acc
    }

    fn frac_penalty_backend(z: &[f64], alpha: f64, lam: f64, backend: &FracBackend) -> f64 {
        let base = frac_penalty(z, alpha, lam);
        match backend {
            FracBackend::CpuRadix2 => base,
            FracBackend::Wgpu { radix } => {
                let radix_factor = (*radix as f64).max(2.0) / 2.0;
                base * (1.0 + 0.15 * (radix_factor - 1.0))
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn norm(vec: &[f64]) -> f64 {
            vec.iter().map(|v| v * v).sum::<f64>().sqrt()
        }

        #[test]
        fn logistic_projection_keeps_unit_norm() {
            let mut params = ZMetaParams::default();
            params.dim = 3;
            params.seed = 7;
            params.orientation_eta = 0.2;
            params.orientation_eps = 5e-3;
            let mut es = ZMetaES::new(params);
            assert!((norm(&es.dir) - 1.0).abs() < 1e-6);

            let context1 = [0.2, 0.5, -0.3];
            let delta1 = es.ingest_structural(Some(&context1)).unwrap();
            es.apply_structural_drive(delta1, 0.15);
            assert!((norm(&es.dir) - 1.0).abs() < 1e-6);

            let context2 = [0.8, -0.1, 0.3];
            let delta2 = es.ingest_structural(Some(&context2)).unwrap();
            let dir_before = es.dir.clone();
            es.apply_structural_drive(delta2, 0.25);
            assert!((norm(&es.dir) - 1.0).abs() < 1e-6);
            let delta_dir = es
                .dir
                .iter()
                .zip(dir_before.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            assert!(delta_dir > 1e-6);
        }
    }

    mod randless {
        use std::cell::Cell;

        pub trait Rng {
            fn gauss(&self, mu: f64, sigma: f64) -> f64;
        }

        pub struct StdRng {
            state: Cell<u64>,
        }

        impl StdRng {
            pub fn seed_from_u64(seed: u64) -> Self {
                Self {
                    state: Cell::new(seed | 1),
                }
            }

            fn next_u64(&self) -> u64 {
                let mut x = self.state.get();
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.state.set(x);
                x
            }

            fn uniform01(&self) -> f64 {
                (self.next_u64() as f64) / (u64::MAX as f64)
            }
        }

        impl Rng for StdRng {
            fn gauss(&self, mu: f64, sigma: f64) -> f64 {
                let u1 = self.uniform01().max(1e-12);
                let u2 = self.uniform01();
                let radius = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f64::consts::PI * u2;
                mu + sigma * radius * theta.cos()
            }
        }
    }
}

// =================== bandit.rs ===================
pub mod bandit {

    #[derive(Clone, Copy, Debug)]
    pub enum SoftBanditMode {
        TS,
        UCB,
    }

    #[derive(Clone, Debug)]
    pub struct LinTSArm {
        dim: usize,
        a: Vec<f64>,
        b: Vec<f64>,
    }

    impl LinTSArm {
        pub fn new(dim: usize, lambda: f64) -> Self {
            Self {
                dim,
                a: eye_flat(dim, lambda),
                b: vec![0.0; dim],
            }
        }

        pub fn sample_score(&self, x: &[f64]) -> f64 {
            let mean = solve_spd_diag(&self.a, &self.b, self.dim);
            dot(&mean, x)
        }

        pub fn ucb_score(&self, x: &[f64], c: f64) -> f64 {
            let ainv = inv_spd_diag(&self.a, self.dim);
            let mean = matvec_flat(&ainv, &self.b, self.dim);
            let variance = quad_form(&ainv, x);
            dot(&mean, x) + c * variance.max(0.0).sqrt()
        }

        pub fn update(&mut self, x: &[f64], reward: f64) {
            rank1_add(&mut self.a, x, self.dim);
            for (bi, xi) in self.b.iter_mut().zip(x) {
                *bi += reward * xi;
            }
        }
    }

    pub struct SoftBandit {
        choices: Vec<String>,
        arms: Vec<LinTSArm>,
        last_index: usize,
        mode: SoftBanditMode,
    }

    impl SoftBandit {
        pub fn new(choices: Vec<String>, feat_dim: usize, mode: SoftBanditMode) -> Self {
            let arms = (0..choices.len())
                .map(|_| LinTSArm::new(feat_dim, 1.0))
                .collect();
            Self {
                choices,
                arms,
                last_index: 0,
                mode,
            }
        }

        pub fn select(&mut self, x: &[f64]) -> String {
            let mut best = f64::MIN;
            let mut idx = 0usize;
            for (i, arm) in self.arms.iter().enumerate() {
                let score = match self.mode {
                    SoftBanditMode::TS => arm.sample_score(x),
                    SoftBanditMode::UCB => arm.ucb_score(x, 1.0),
                };
                if score > best {
                    best = score;
                    idx = i;
                }
            }
            self.last_index = idx;
            self.choices[idx].clone()
        }

        pub fn update_last(&mut self, x: &[f64], reward: f64) {
            if let Some(arm) = self.arms.get_mut(self.last_index) {
                arm.update(x, reward);
            }
        }
    }

    fn eye_flat(dim: usize, lambda: f64) -> Vec<f64> {
        let mut matrix = vec![0.0; dim * dim];
        for i in 0..dim {
            matrix[i * dim + i] = lambda;
        }
        matrix
    }

    fn matvec_flat(a: &[f64], x: &[f64], dim: usize) -> Vec<f64> {
        let mut y = vec![0.0; dim];
        for i in 0..dim {
            let mut sum = 0.0;
            for j in 0..dim {
                sum += a[i * dim + j] * x[j];
            }
            y[i] = sum;
        }
        y
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn rank1_add(a: &mut [f64], x: &[f64], dim: usize) {
        for i in 0..dim {
            for j in 0..dim {
                a[i * dim + j] += x[i] * x[j];
            }
        }
    }

    fn inv_spd_diag(a: &[f64], dim: usize) -> Vec<f64> {
        let mut inv = vec![0.0; dim * dim];
        for i in 0..dim {
            let value = a[i * dim + i];
            inv[i * dim + i] = if value.abs() > 1e-12 {
                1.0 / value
            } else {
                1.0
            };
        }
        inv
    }

    fn solve_spd_diag(a: &[f64], b: &[f64], dim: usize) -> Vec<f64> {
        let mut x = vec![0.0; dim];
        for i in 0..dim {
            let value = a[i * dim + i];
            x[i] = if value.abs() > 1e-12 {
                b[i] / value
            } else {
                b[i]
            };
        }
        x
    }

    fn quad_form(a: &[f64], x: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..x.len() {
            for j in 0..x.len() {
                sum += x[i] * a[i * x.len() + j] * x[j];
            }
        }
        sum
    }
}

// =================== wilson.rs ===================
pub mod wilson {
    pub fn wilson_lower(successes: i32, trials: i32, z: f64) -> f64 {
        if trials <= 0 {
            return 0.0;
        }
        let n = trials as f64;
        let s = successes as f64;
        let p = (s / n).clamp(0.0, 1.0);
        let denom = 1.0 + z * z / n;
        let center = (p + z * z / (2.0 * n)) / denom;
        let radius = z * ((p * (1.0 - p) + z * z / (4.0 * n)) / n).max(0.0).sqrt() / denom;
        (center - radius).max(0.0)
    }
}

// =================== rewrite.rs ===================
pub mod rewrite {
    use std::collections::HashMap;
    use std::fs::{self, OpenOptions};
    use std::io::Write;
    use std::path::PathBuf;

    pub struct HeurStore {
        path: PathBuf,
    }

    impl HeurStore {
        pub fn new(custom: Option<String>) -> Self {
            let path = custom.map(PathBuf::from).unwrap_or(default_path());
            if let Some(dir) = path.parent() {
                let _ = fs::create_dir_all(dir);
            }
            Self { path }
        }

        pub fn append(&self, rule_text: &str, info: &HashMap<String, f64>) {
            if let Ok(mut file) = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)
            {
                let meta = serde_like_json(info);
                let line = format!("{}  # {}\n", rule_text.trim(), meta);
                let _ = file.write_all(line.as_bytes());
            }
        }

        pub fn path(&self) -> &PathBuf {
            &self.path
        }
    }

    fn default_path() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| String::from("."));
        let mut path = PathBuf::from(home);
        path.push(".spiraltorch/heur/heur.kdsl");
        path
    }

    fn serde_like_json(info: &HashMap<String, f64>) -> String {
        let mut out = String::from("{");
        let mut first = true;
        for (key, value) in info.iter() {
            if !first {
                out.push_str(", ");
            }
            first = false;
            out.push_str(&format!("\"{}\":{:.6}", key, value));
        }
        out.push('}');
        out
    }
}

// =================== ab.rs ===================
pub mod ab {
    use super::{RewardCfg, StepMetrics};

    pub struct ABRunner {
        reward: RewardCfg,
    }

    impl ABRunner {
        pub fn new(reward: RewardCfg) -> Self {
            Self { reward }
        }

        pub fn run<F, G>(&self, mut a: F, mut b: G, trials: usize) -> (usize, usize)
        where
            F: FnMut() -> StepMetrics,
            G: FnMut() -> StepMetrics,
        {
            let mut wins_a = 0usize;
            let mut wins_b = 0usize;
            for _ in 0..trials {
                let ma = a();
                let mb = b();
                let ra = self.reward.score(&ma, 0.0);
                let rb = self.reward.score(&mb, 0.0);
                if ra > rb {
                    wins_a += 1;
                } else {
                    wins_b += 1;
                }
            }
            (wins_a, wins_b)
        }
    }
}
