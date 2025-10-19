// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Experimental Z-pulse conductor with scale tracking.
//!
//! This module preserves the refactor that previously lived in `zpulse.rs`
//! so the work can continue behind an opt-in Cargo feature without breaking
//! the default build.

#[cfg(feature = "experimental_zpulse")]
pub mod experimental {
    #![allow(dead_code)]
    #![allow(unused_imports)]
    // SPDX-License-Identifier: AGPL-3.0-or-later
    // © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
    // Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
    // Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

    //! Canonical Z-space pulse representations and a lightweight conductor used by
    //! microlocal, Maxwell, and RealGrad observers.  The previous revision of this
    //! file had diverged into an unreadable hybrid of half-finished refactors.  The
    //! implementation below intentionally keeps the surface area minimal while still
    //! exercising the behaviour required by the public unit tests and downstream
    //! callers.  Every routine focuses on deterministic, allocation-friendly data
    //! paths so the conductor can run inside the realtime control loop without
    //! pulling in async machinery.

    use rustc_hash::FxHashMap;
    use std::cmp::Ordering;
    use std::collections::VecDeque;

    /// Support triplet describing Above/Here/Beneath contributions backing a Z pulse.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct ZSupport {
        pub leading: f32,
        pub central: f32,
        pub trailing: f32,
    }

    impl ZSupport {
        /// Creates a new support triplet, clamping each component to a finite,
        /// non-negative value.
        pub fn new(leading: f32, central: f32, trailing: f32) -> Self {
            Self {
                leading: clamp_non_negative(leading),
                central: clamp_non_negative(central),
                trailing: clamp_non_negative(trailing),
            }
        }

        /// Builds a support triplet straight from an Above/Here/Beneath energy tuple.
        pub fn from_band_energy(bands: (f32, f32, f32)) -> Self {
            Self::new(bands.0, bands.1, bands.2)
        }

        /// Returns the total perimeter mass supporting the pulse.
        pub fn total(&self) -> f32 {
            self.leading + self.central + self.trailing
        }

        /// Returns `true` when all support components vanish.
        pub fn is_empty(&self) -> bool {
            self.leading <= f32::EPSILON
                && self.central <= f32::EPSILON
                && self.trailing <= f32::EPSILON
        }

        /// Maximum absolute component used by stability heuristics.
        pub fn max_component(&self) -> f32 {
            self.leading
                .abs()
                .max(self.central.abs())
                .max(self.trailing.abs())
        }
    }

    impl Default for ZSupport {
        fn default() -> Self {
            Self {
                leading: 0.0,
                central: 0.0,
                trailing: 0.0,
            }
        }
    }

    fn clamp_non_negative(value: f32) -> f32 {
        if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        }
    }

    /// Identifies the origin of a [`ZPulse`].
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum ZSource {
        Microlocal,
        Maxwell,
        Graph,
        Desire,
        GW,
        RealGrad,
        Other(&'static str),
    }

    impl Default for ZSource {
        fn default() -> Self {
            ZSource::Microlocal
        }
    }

    /// Snapshot of a single Z-space pulse observation.
    #[derive(Clone, Debug, PartialEq)]
    pub struct ZPulse {
        pub source: ZSource,
        pub ts: u64,
        pub tempo: f32,
        pub band_energy: (f32, f32, f32),
        pub drift: f32,
        pub z_bias: f32,
        pub support: ZSupport,
        pub scale: Option<ZScale>,
        pub quality: f32,
        pub stderr: f32,
        pub latency_ms: f32,
    }

    impl ZPulse {
        pub fn support_mass(&self) -> f32 {
            self.support.total()
        }

        pub fn total_energy(&self) -> f32 {
            let (a, h, b) = self.band_energy;
            a + h + b
        }

        pub fn normalised_drift(&self) -> f32 {
            let total = self.total_energy().max(1e-6);
            let (a, _, b) = self.band_energy;
            (a - b) / total
        }

        pub fn is_empty(&self) -> bool {
            self.support.is_empty() && self.total_energy() <= f32::EPSILON
        }

        fn support_strength(&self) -> f32 {
            self.support.total().max(0.0)
        }
    }

    impl Default for ZPulse {
        fn default() -> Self {
            Self {
                source: ZSource::Microlocal,
                ts: 0,
                tempo: 0.0,
                band_energy: (0.0, 0.0, 0.0),
                drift: 0.0,
                z_bias: 0.0,
                support: ZSupport::default(),
                scale: None,
                quality: 0.0,
                stderr: 0.0,
                latency_ms: 0.0,
            }
        }
    }

    impl From<(f32, f32, f32)> for ZSupport {
        fn from(value: (f32, f32, f32)) -> Self {
            Self::from_band_energy(value)
        }
    }

    /// Encodes the physical and logarithmic radius of a Z pulse.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct ZScale {
        pub physical_radius: f32,
        pub log_radius: f32,
    }

    impl ZScale {
        /// Creates a new scale from a physical radius, rejecting non-positive or
        /// non-finite values.
        pub fn new(physical_radius: f32) -> Option<Self> {
            if physical_radius.is_finite() && physical_radius > 0.0 {
                Some(Self {
                    physical_radius,
                    log_radius: physical_radius.ln(),
                })
            } else {
                None
            }
        }

        /// Reconstructs a scale from its logarithmic radius.
        pub fn from_log(log_radius: f32) -> Option<Self> {
            if log_radius.is_finite() {
                let physical = log_radius.exp();
                if physical.is_finite() && physical > 0.0 {
                    return Some(Self {
                        physical_radius: physical,
                        log_radius,
                    });
                }
            }
            None
        }

        /// Builds a scale directly from precomputed components, enforcing the
        /// invariants required by the other constructors.
        pub fn from_components(physical_radius: f32, log_radius: f32) -> Option<Self> {
            if physical_radius.is_finite() && physical_radius > 0.0 && log_radius.is_finite() {
                Some(Self {
                    physical_radius,
                    log_radius,
                })
            } else {
                None
            }
        }

        /// Linearly interpolates between two scales.
        pub fn lerp(a: ZScale, b: ZScale, t: f32) -> ZScale {
            let clamped = t.clamp(0.0, 1.0);
            let physical = a.physical_radius + (b.physical_radius - a.physical_radius) * clamped;
            let log = a.log_radius + (b.log_radius - a.log_radius) * clamped;
            Self::from_components(physical, log)
                .unwrap_or_else(|| if clamped < 0.5 { a } else { b })
        }

        /// Computes the weighted centroid of a collection of scales.
        pub fn weighted_average<I>(weights: I) -> Option<Self>
        where
            I: IntoIterator<Item = (ZScale, f32)>,
        {
            let mut total = 0.0f32;
            let mut physical = 0.0f32;
            let mut log = 0.0f32;
            for (scale, weight) in weights {
                if weight.is_finite() && weight > 0.0 {
                    total += weight;
                    physical += scale.physical_radius * weight;
                    log += scale.log_radius * weight;
                }
            }
            if total > 0.0 {
                Self::from_components(physical / total, log / total)
            } else {
                None
            }
        }
    }

    /// Identifies a source capable of emitting [`ZPulse`] records.
    pub trait ZEmitter: Send {
        /// Returns the canonical source identifier for pulses emitted by this implementation.
        fn name(&self) -> ZSource;

        /// Advances the emitter one step and returns the next available pulse, if any.
        fn tick(&mut self, now: u64) -> Option<ZPulse>;

        /// Optional quality hint used when the emitted pulse does not set one explicitly.
        fn quality_hint(&self) -> Option<f32> {
            None
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct ZFrequencyConfig {
        pub smoothing: f32,
        pub minimum_energy: f32,
    }

    impl Default for ZFrequencyConfig {
        fn default() -> Self {
            Self {
                smoothing: 0.0,
                minimum_energy: 0.0,
            }
        }
    }

    impl ZFrequencyConfig {
        pub fn new(smoothing: f32, minimum_energy: f32) -> Self {
            Self {
                smoothing,
                minimum_energy,
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct ZAdaptiveGainCfg {
        pub gain_floor: f32,
        pub gain_ceil: f32,
        pub responsiveness: f32,
    }

    impl Default for ZAdaptiveGainCfg {
        fn default() -> Self {
            Self {
                gain_floor: 0.0,
                gain_ceil: 1.0,
                responsiveness: 0.0,
            }
        }
    }

    impl ZAdaptiveGainCfg {
        pub fn new(gain_floor: f32, gain_ceil: f32, responsiveness: f32) -> Self {
            let gain_floor = gain_floor.max(0.0);
            Self {
                gain_floor,
                gain_ceil: gain_ceil.max(gain_floor),
                responsiveness,
            }
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    pub struct LatencyAlignerCfg {
        pub window: u64,
        pub hop: u64,
        pub max_lag_steps: u32,
        pub alpha: f32,
        pub coherence_min: f32,
        pub hold_steps: u32,
        pub fractional: bool,
    }

    impl LatencyAlignerCfg {
        pub fn balanced() -> Self {
            Self {
                window: 128,
                hop: 1,
                max_lag_steps: 48,
                alpha: 0.25,
                coherence_min: 0.2,
                hold_steps: 2,
                fractional: false,
            }
        }

        pub fn from_steps(
            max_lag_steps: u32,
            alpha: f32,
            coherence_min: f32,
            hold_steps: u32,
        ) -> Self {
            Self {
                max_lag_steps,
                alpha,
                coherence_min,
                hold_steps,
                ..Self::balanced()
            }
        }

        pub fn with_window(mut self, window: u64) -> Self {
            self.window = window.max(1);
            self
        }

        pub fn with_hop(mut self, hop: u64) -> Self {
            self.hop = hop.max(1);
            self
        }
    }

    impl Default for LatencyAlignerCfg {
        fn default() -> Self {
            Self::balanced()
        }
    }

    #[derive(Clone, Debug, Default)]
    struct SourceLast {
        ts: u64,
        strength: f32,
    }

    #[derive(Clone, Debug)]
    struct LagEstimate {
        lag: f32,
        frames_since_update: u32,
    }

    impl Default for LagEstimate {
        fn default() -> Self {
            Self {
                lag: 0.0,
                frames_since_update: u32::MAX,
            }
        }
    }

    #[derive(Clone, Debug)]
    pub(crate) struct LatencyAlignerState {
        cfg: LatencyAlignerCfg,
        last: FxHashMap<ZSource, SourceLast>,
        lags: FxHashMap<ZSource, LagEstimate>,
        pending_events: Vec<String>,
    }

    impl LatencyAlignerState {
        fn new(cfg: LatencyAlignerCfg) -> Self {
            Self {
                cfg,
                last: FxHashMap::default(),
                lags: FxHashMap::default(),
                pending_events: Vec::new(),
            }
        }

        fn record(&mut self, pulse: &ZPulse) {
            let strength = pulse.support_strength().max(1e-6);
            self.last.insert(
                pulse.source,
                SourceLast {
                    ts: pulse.ts,
                    strength,
                },
            );
            if pulse.latency_ms.is_finite() && pulse.latency_ms.abs() > f32::EPSILON {
                let entry = self
                    .lags
                    .entry(pulse.source)
                    .or_insert_with(LagEstimate::default);
                entry.lag = pulse.latency_ms;
                entry.frames_since_update = 0;
                self.pending_events.push(format!(
                    "latency.seeded:{:?}:{:.2}",
                    pulse.source, entry.lag
                ));
            }
        }

        fn increment_all(&mut self) {
            for lag in self.lags.values_mut() {
                if lag.frames_since_update != u32::MAX {
                    lag.frames_since_update = lag.frames_since_update.saturating_add(1);
                }
            }
        }

        fn prepare(&mut self, now: u64, events: &mut Vec<String>) {
            events.extend(self.pending_events.drain(..));
            let Some(anchor) = self.last.get(&ZSource::Microlocal).cloned() else {
                self.increment_all();
                return;
            };
            if self.cfg.coherence_min > 1.0 {
                for source in self.last.keys() {
                    if *source != ZSource::Microlocal {
                        events.push(format!("latency.low_coherence:{:?}", source));
                    }
                }
                for lag in self.lags.values_mut() {
                    if lag.frames_since_update != u32::MAX {
                        lag.frames_since_update = lag.frames_since_update.saturating_add(1);
                    }
                }
                return;
            }

            let hop = self.cfg.hop.max(1) as f32;
            let max_lag = self.cfg.max_lag_steps.max(1) as f32;
            for (source, state) in self.last.clone() {
                if source == ZSource::Microlocal {
                    continue;
                }
                let entry = self.lags.entry(source).or_insert_with(LagEstimate::default);
                if entry.frames_since_update != u32::MAX
                    && entry.frames_since_update < self.cfg.hold_steps
                {
                    events.push(format!("latency.held:{:?}", source));
                    entry.frames_since_update = entry.frames_since_update.saturating_add(1);
                    continue;
                }
                let raw_lag_bins = (state.ts as i64 - anchor.ts as i64) as f32;
                let clamped = raw_lag_bins.clamp(-max_lag, max_lag);
                let lag_units = clamped * hop;
                if entry.frames_since_update == u32::MAX {
                    entry.lag = lag_units;
                } else {
                    let alpha = self.cfg.alpha.clamp(0.0, 1.0);
                    entry.lag = lerp(entry.lag, lag_units, alpha);
                }
                entry.frames_since_update = 0;
                events.push(format!("latency.adjusted:{:?}:{:.2}", source, entry.lag));
            }

            self.increment_all();
            // Reset the anchor timestamp so future stale pulses do not inherit it.
            self.last.insert(
                ZSource::Microlocal,
                SourceLast {
                    ts: now,
                    strength: anchor.strength,
                },
            );
        }

        fn apply(&self, pulse: &mut ZPulse) {
            if let Some(lag) = self.lags.get(&pulse.source) {
                if lag.lag.is_finite() {
                    pulse.latency_ms = lag.lag;
                    pulse.ts = shift_timestamp(pulse.ts, lag.lag, self.cfg.hop);
                }
            }
        }

        fn lag_for(&self, source: &ZSource) -> Option<f32> {
            self.lags.get(source).map(|lag| lag.lag)
        }
    }

    #[derive(Clone, Debug)]
    pub struct ZConductorCfg {
        pub alpha_fast: f32,
        pub alpha_slow: f32,
        pub flip_hold: u64,
        pub slew_max: f32,
        pub z_budget: f32,
        pub robust_delta: f32,
        pub latency_align: bool,
        pub latency: Option<LatencyAlignerCfg>,
    }

    impl Default for ZConductorCfg {
        fn default() -> Self {
            Self {
                alpha_fast: 0.6,
                alpha_slow: 0.12,
                flip_hold: 5,
                slew_max: 0.08,
                z_budget: 0.9,
                robust_delta: 0.2,
                latency_align: true,
                latency: Some(LatencyAlignerCfg::balanced()),
            }
        }
    }

    #[derive(Clone, Debug, Default)]
    pub struct ZFused {
        pub ts: u64,
        pub z: f32,
        pub support: f32,
        pub drift: f32,
        pub quality: f32,
        pub events: Vec<String>,
        pub attributions: Vec<(ZSource, f32)>,
    }

    /// Stateful conductor that fuses heterogeneous Z pulses into a stabilised control signal.
    #[derive(Clone)]
    pub struct ZConductor {
        cfg: ZConductorCfg,
        pub(crate) freq: Option<ZFrequencyConfig>,
        pub(crate) adaptive: Option<ZAdaptiveGainCfg>,
        pub(crate) latency: Option<LatencyAlignerState>,
        pending: VecDeque<ZPulse>,
        last_step: Option<u64>,
        last_z: f32,
        hold_until: Option<u64>,
        pending_flip_sign: Option<f32>,
    }

    impl Default for ZConductor {
        fn default() -> Self {
            Self::new(ZConductorCfg::default())
        }
    }

    impl ZConductor {
        pub fn new(cfg: ZConductorCfg) -> Self {
            let latency_cfg = if cfg.latency_align {
                cfg.latency
                    .clone()
                    .or_else(|| Some(LatencyAlignerCfg::balanced()))
            } else {
                None
            };
            Self {
                cfg,
                freq: None,
                adaptive: None,
                latency: latency_cfg.map(LatencyAlignerState::new),
                pending: VecDeque::new(),
                last_step: None,
                last_z: 0.0,
                hold_until: None,
                pending_flip_sign: None,
            }
        }

        pub fn cfg(&self) -> &ZConductorCfg {
            &self.cfg
        }

        pub fn cfg_mut(&mut self) -> &mut ZConductorCfg {
            &mut self.cfg
        }

        pub fn set_frequency_config(&mut self, cfg: Option<ZFrequencyConfig>) {
            self.freq = cfg;
        }

        pub fn set_adaptive_gain_config(&mut self, cfg: Option<ZAdaptiveGainCfg>) {
            self.adaptive = cfg;
        }

        pub fn set_latency_aligner(&mut self, cfg: Option<LatencyAlignerCfg>) {
            self.cfg.latency = cfg.clone();
            self.cfg.latency_align = cfg.is_some();
            self.latency = cfg.map(LatencyAlignerState::new);
        }

        pub fn latency_for(&self, source: &ZSource) -> Option<f32> {
            self.latency
                .as_ref()
                .and_then(|state| state.lag_for(source))
        }

        pub fn ingest(&mut self, mut pulse: ZPulse) {
            if !pulse.quality.is_finite() || pulse.quality <= 0.0 {
                pulse.quality = derive_quality(&pulse);
            } else {
                pulse.quality = pulse.quality.clamp(0.0, 1.0);
            }
            if let Some(lat) = self.latency.as_mut() {
                lat.record(&pulse);
            }
            self.pending.push_back(pulse);
        }

        pub fn step(&mut self, now: u64) -> ZFused {
            let mut events = Vec::new();
            if let Some(latency) = self.latency.as_mut() {
                latency.prepare(now, &mut events);
            }

            let mut ready = Vec::new();
            let mut retained = VecDeque::with_capacity(self.pending.len());
            while let Some(mut pulse) = self.pending.pop_front() {
                if self.cfg.latency_align {
                    if let Some(lat) = self.latency.as_ref() {
                        lat.apply(&mut pulse);
                    }
                }
                if pulse.ts <= now {
                    ready.push(pulse);
                } else {
                    retained.push_back(pulse);
                }
            }
            self.pending = retained;

            let mut total_support = 0.0f32;
            let mut weighted_z = 0.0f32;
            let mut weighted_quality = 0.0f32;
            let mut quality_weight = 0.0f32;
            let mut drift_weight = 0.0f32;
            let mut drift_sum = 0.0f32;
            let mut attributions: Vec<(ZSource, f32)> = Vec::new();

            for pulse in &ready {
                if pulse.is_empty() {
                    continue;
                }
                let support = pulse.support_strength();
                total_support += support;
                weighted_z += pulse.z_bias * support;

                let weight = support.max(1e-6);
                weighted_quality += pulse.quality * weight;
                quality_weight += weight;
                drift_sum += pulse.normalised_drift() * pulse.quality.max(1e-6);
                drift_weight += pulse.quality.max(1e-6);
                attributions.push((pulse.source, support));
            }

            let mut raw_z = if total_support > 0.0 {
                weighted_z / total_support
            } else {
                0.0
            };

            let mut fused = ZFused {
                ts: now,
                support: total_support,
                drift: if drift_weight > 0.0 {
                    (drift_sum / drift_weight).clamp(-1.0, 1.0)
                } else {
                    0.0
                },
                quality: if quality_weight > 0.0 {
                    (weighted_quality / quality_weight).clamp(0.0, 1.0)
                } else {
                    0.0
                },
                events,
                attributions: Vec::new(),
                z: 0.0,
            };

            attributions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            fused.attributions = attributions;

            let incoming_sign = raw_z.signum();
            if let Some(hold_until) = self.hold_until {
                if now < hold_until {
                    fused.events.push("flip-held".to_string());
                    raw_z = self.last_z;
                } else {
                    self.hold_until = None;
                }
            }
            if self.hold_until.is_none() {
                if let Some(sign) = self.pending_flip_sign.take() {
                    fused.events.push("sign-flip".to_string());
                    let magnitude = raw_z.abs().max(self.cfg.robust_delta);
                    raw_z = magnitude.copysign(sign);
                } else if self.last_z != 0.0
                    && incoming_sign != 0.0
                    && incoming_sign != self.last_z.signum()
                {
                    self.hold_until = Some(now + self.cfg.flip_hold as u64);
                    self.pending_flip_sign = Some(incoming_sign);
                    fused.events.push("flip-held".to_string());
                    raw_z = self.last_z;
                }
            }

            let alpha = self.cfg.alpha_fast.clamp(0.0, 1.0);
            let mut target = lerp(self.last_z, raw_z, alpha);
            let delta = target - self.last_z;
            if delta.abs() > self.cfg.slew_max {
                target = self.last_z + self.cfg.slew_max.copysign(delta);
            }
            target = target.clamp(-self.cfg.z_budget, self.cfg.z_budget);
            self.last_z = target;
            fused.z = target;

            fused
        }

        pub fn step_from_registry(&mut self, registry: &mut ZRegistry, now: u64) -> ZFused {
            for mut pulse in registry.gather(now) {
                if pulse.quality <= 0.0 {
                    pulse.quality = derive_quality(&pulse);
                }
                self.ingest(pulse);
            }
            self.step(now)
        }
    }

    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t.clamp(0.0, 1.0)
    }

    fn shift_timestamp(ts: u64, lag: f32, hop: u64) -> u64 {
        let shift = if hop == 0 { 0.0 } else { lag / hop as f32 };
        let delta = shift.round() as i64;
        let base = ts as i64 + delta;
        if base < 0 {
            0
        } else {
            base as u64
        }
    }

    fn derive_quality(pulse: &ZPulse) -> f32 {
        let support = pulse.support.total().max(1e-6);
        let stderr = pulse.stderr.abs() + 1e-6;
        let ratio = support / (support + stderr);
        ratio.clamp(0.0, 1.0)
    }

    /// Registry used to poll multiple emitters and return their pending pulses.
    #[derive(Default)]
    pub struct ZRegistry {
        emitters: Vec<Box<dyn ZEmitter>>,
    }

    impl ZRegistry {
        pub fn with_capacity(capacity: usize) -> Self {
            Self {
                emitters: Vec::with_capacity(capacity),
            }
        }

        pub fn register<E>(&mut self, emitter: E)
        where
            E: ZEmitter + 'static,
        {
            self.emitters.push(Box::new(emitter));
        }

        pub fn gather(&mut self, now: u64) -> Vec<ZPulse> {
            let mut pulses = Vec::new();
            for emitter in &mut self.emitters {
                while let Some(mut pulse) = emitter.tick(now) {
                    if pulse.quality <= 0.0 {
                        if let Some(hint) = emitter.quality_hint() {
                            pulse.quality = hint;
                        }
                    }
                    pulses.push(pulse);
                }
            }
            pulses
        }
    }
}
