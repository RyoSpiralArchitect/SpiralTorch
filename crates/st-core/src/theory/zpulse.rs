// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Canonical representation of Z pulses together with a lightweight
//! conductor that fuses multiple sources into a single control signal.

use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::fs;
use std::io::{ErrorKind, Read};
use std::path::Path;
use std::sync::{Arc, Mutex};

use serde::Deserialize;

/// Identifies a source capable of emitting [`ZPulse`] records.
pub trait ZEmitter {
    /// Returns the canonical source identifier for pulses emitted by this
    /// implementation.
    fn name(&self) -> ZSource;

    /// Advances the emitter one step and returns the next available pulse, if
    /// any. Implementations may return more than one pulse per call by keeping
    /// an internal queue; [`ZRegistry::gather`] will keep polling the emitter
    /// until it reports `None`.
    fn tick(&mut self, now: u64) -> Option<ZPulse>;

    /// Optional quality hint describing the reliability of upcoming pulses.
    /// When absent the conductor infers quality from the pulses themselves.
    fn quality_hint(&self) -> Option<f32> {
        None
    }
}

/// Central registry that owns a fleet of [`ZEmitter`] implementations.
#[derive(Default)]
pub struct ZRegistry {
    emitters: Vec<Box<dyn ZEmitter + Send>>,
}

impl ZRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            emitters: Vec::new(),
        }
    }

    /// Creates a registry with pre-allocated storage for the provided number of
    /// emitters.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            emitters: Vec::with_capacity(capacity),
        }
    }

    /// Registers a new emitter so it participates in subsequent gather calls.
    pub fn register<E>(&mut self, emitter: E)
    where
        E: ZEmitter + Send + 'static,
    {
        self.emitters.push(Box::new(emitter));
    }

    /// Polls each registered emitter and aggregates all pulses emitted at the
    /// supplied timestamp.
    pub fn gather(&mut self, now: u64) -> Vec<ZPulse> {
        let mut pulses = Vec::new();
        for emitter in &mut self.emitters {
            while let Some(pulse) = emitter.tick(now) {
                pulses.push(pulse);
            }
        }
        pulses
    }

    /// Returns `true` when no emitters are registered.
    pub fn is_empty(&self) -> bool {
        self.emitters.is_empty()
    }
}

/// Identifies the origin of a [`ZPulse`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ZSource {
    Microlocal,
    Maxwell,
    RealGrad,
    Desire,
    External(&'static str),
    Other(&'static str),
}

/// Discrete Z pulse emitted by an upstream source.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZPulse {
    pub source: ZSource,
    pub ts: u64,
    pub band_energy: (f32, f32, f32),
    pub drift: f32,
    pub z_bias: f32,
    pub support: f32,
    pub quality: f32,
    pub stderr: f32,
    pub latency_ms: f32,
}

impl Default for ZPulse {
    fn default() -> Self {
        Self {
            source: ZSource::Microlocal,
            ts: 0,
            band_energy: (0.0, 0.0, 0.0),
            drift: 0.0,
            z_bias: 0.0,
            support: 0.0,
            quality: 0.0,
            stderr: 0.0,
            latency_ms: 0.0,
        }
    }
}

/// Result of fusing multiple [`ZPulse`] records.
#[derive(Clone, Debug, PartialEq)]
pub struct ZFused {
    pub ts: u64,
    pub support: f32,
    pub drift: f32,
    pub z: f32,
    pub quality: f32,
    pub events: Vec<String>,
    pub attributions: Vec<(ZSource, f32)>,
}

impl Default for ZFused {
    fn default() -> Self {
        Self {
            ts: 0,
            support: 0.0,
            drift: 0.0,
            z: 0.0,
            quality: 0.0,
            events: Vec::new(),
            attributions: Vec::new(),
        }
    }
}

/// Optional smoothing applied to the fused support/energy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZFrequencyConfig {
    pub smoothing: f32,
}

impl ZFrequencyConfig {
    pub fn new(smoothing: f32) -> Self {
        Self {
            smoothing: smoothing.clamp(0.0, 1.0),
        }
    }
}

/// Optional adaptive gain configuration for downstream consumers.
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
        let floor = gain_floor.max(0.0);
        let ceil = gain_ceil.max(floor);
        Self {
            gain_floor: floor,
            gain_ceil: ceil,
            responsiveness: responsiveness.clamp(0.0, 1.0),
        }
    }

    pub fn update(&mut self, gain_floor: f32, gain_ceil: f32, responsiveness: f32) {
        *self = Self::new(gain_floor, gain_ceil, responsiveness);
    }
}

/// Optional latency alignment configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZLatencyConfig {
    pub history: usize,
}

impl ZLatencyConfig {
    pub fn new(history: usize) -> Self {
        Self {
            history: history.max(1),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ZConductorCfg {
    pub alpha_fast: f32,
    pub alpha_slow: f32,
    pub flip_hold: u64,
    pub slew_max: f32,
    pub z_budget: f32,
    pub robust_delta: f32,
    pub latency_align: bool,
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
        }
    }
}

#[derive(Clone, Debug)]
pub struct ZConductor {
    cfg: ZConductorCfg,
    fused: ZFused,
    last_step: Option<u64>,
    hold_until: Option<u64>,
    pending_flip_sign: Option<f32>,
    frequency_cfg: Option<ZFrequencyConfig>,
    adaptive_cfg: Option<ZAdaptiveGainCfg>,
    latency_cfg: Option<ZLatencyConfig>,
    latency_events: VecDeque<String>,
    source_gains: HashMap<String, f32>,
    source_limits: HashMap<String, f32>,
    config_events: VecDeque<String>,
}

impl Default for ZConductor {
    fn default() -> Self {
        Self::new(ZConductorCfg::default())
    }
}

impl ZConductor {
    pub fn new(cfg: ZConductorCfg) -> Self {
        Self {
            cfg,
            fused: ZFused::default(),
            last_step: None,
            hold_until: None,
            pending_flip_sign: None,
            frequency_cfg: None,
            adaptive_cfg: None,
            latency_cfg: None,
            latency_events: VecDeque::new(),
            source_gains: HashMap::new(),
            source_limits: HashMap::new(),
            config_events: VecDeque::new(),
        }
    }

    pub fn cfg(&self) -> &ZConductorCfg {
        &self.cfg
    }

    pub fn cfg_mut(&mut self) -> &mut ZConductorCfg {
        &mut self.cfg
    }

    pub fn set_frequency_config(&mut self, cfg: Option<ZFrequencyConfig>) {
        self.frequency_cfg = cfg;
    }

    pub fn set_adaptive_gain_config(&mut self, cfg: Option<ZAdaptiveGainCfg>) {
        self.adaptive_cfg = cfg;
    }

    pub fn set_latency_config(&mut self, cfg: Option<ZLatencyConfig>) {
        self.latency_cfg = cfg;
        if let Some(cfg) = cfg {
            self.latency_events.truncate(cfg.history);
        }
    }
}

    fn record_config_event(&mut self, event: String) {
        const MAX_CONFIG_EVENTS: usize = 128;
        if self.config_events.len() >= MAX_CONFIG_EVENTS {
            self.config_events.pop_front();
        }
        self.config_events.push_back(event);
    }

    fn record_config_events<I>(&mut self, events: I)
    where
        I: IntoIterator<Item = String>,
    {
        for event in events {
            self.record_config_event(event);
        }
    }

    pub fn set_source_gains(&mut self, gains: HashMap<String, f32>) {
        self.source_gains = sanitize_tuning_map(gains);
    }

    pub fn set_source_limits(&mut self, limits: HashMap<String, f32>) {
        self.source_limits = sanitize_tuning_map(limits);
    }

    pub fn step<I>(&mut self, pulses: I, now: u64) -> ZFused
    where
        I: IntoIterator<Item = ZPulse>,
    {
        let mut fused = ZFused::default();
        fused.ts = now;

        let mut total_support = 0.0f32;
        let mut weighted_z = 0.0f32;
        let mut weighted_quality = 0.0f32;
        let mut total_quality_weight = 0.0f32;

        let mut had_pulse = false;

        for pulse in pulses {
            had_pulse = true;
            let support = pulse.support.max(0.0);
            total_support += support;
            fused.drift += pulse.drift;
            let key = source_lookup_key(&pulse.source);
            let gain = self
                .source_gains
                .get(key.as_ref())
                .copied()
                .unwrap_or(1.0)
                .max(0.0);
            let limit = self
                .source_limits
                .get(key.as_ref())
                .copied()
                .unwrap_or(f32::INFINITY)
                .max(0.0);
            let mut z_bias = pulse.z_bias * gain;
            if limit.is_finite() {
                if limit == 0.0 {
                    z_bias = 0.0;
                } else {
                    z_bias = z_bias.clamp(-limit, limit);
                }
            }
            weighted_z += z_bias * support;
            let weight = support.max(1e-6);
            weighted_quality += pulse.quality * weight;
            total_quality_weight += weight;
            fused.attributions.push((pulse.source, support));
        }

        if !had_pulse {
            self.fused.ts = now;
            self.last_step = Some(now);
            return self.fused.clone();
        }

        fused.support = total_support;
        if total_quality_weight > 0.0 {
            fused.quality = (weighted_quality / total_quality_weight).clamp(0.0, 1.0);
        }

        let mut raw_z = if total_support > 0.0 {
            weighted_z / total_support.max(1e-6)
        } else {
            weighted_z
        };

        let desired_raw = raw_z;
        let previous_z = self.fused.z;
        let mut events = Vec::new();
        let mut flip_armed = false;

        if let Some(hold) = self.hold_until {
            if now < hold {
                events.push("flip-held".to_string());
                raw_z = 0.0;
            } else {
                self.hold_until = None;
                flip_armed = true;
            }
        }

        let incoming_sign = if desired_raw == 0.0 {
            0.0
        } else {
            desired_raw.signum()
        };
        let previous_sign = if previous_z == 0.0 {
            0.0
        } else {
            previous_z.signum()
        };

        if self.hold_until.is_none() && !flip_armed {
            if previous_sign != 0.0 && incoming_sign != 0.0 && incoming_sign != previous_sign {
                self.hold_until = Some(now + self.cfg.flip_hold);
                self.pending_flip_sign = Some(incoming_sign);
                events.push("flip-held".to_string());
                raw_z = 0.0;
            }
        }

        if self.hold_until.is_none() {
            if flip_armed {
                let desired = self.pending_flip_sign.take().unwrap_or_else(|| {
                    if incoming_sign != 0.0 {
                        incoming_sign
                    } else {
                        -previous_sign
                    }
                });
                let sign = if desired == 0.0 {
                    if previous_sign >= 0.0 {
                        -1.0
                    } else {
                        1.0
                    }
                } else {
                    desired.signum()
                };
                events.push("sign-flip".to_string());
                let magnitude = desired_raw.abs().max(self.cfg.robust_delta.max(1e-6));
                raw_z = magnitude * sign;
            }
        }

        let alpha = if previous_sign == 0.0 || previous_sign == incoming_sign {
            self.cfg.alpha_fast
        } else {
            self.cfg.alpha_slow
        };
        let mut target = lerp(previous_z, raw_z, alpha);
        let delta = target - previous_z;
        if delta.abs() > self.cfg.slew_max {
            target = previous_z + self.cfg.slew_max.copysign(delta);
        }
        target = target.clamp(-self.cfg.z_budget, self.cfg.z_budget);

        fused.z = target;
        fused.events = events;

        if let Some(latency_cfg) = self.latency_cfg {
            if fused.z != previous_z {
                let label = format!(
                    "slew:{}->{}",
                    (previous_z * 100.0).round() / 100.0,
                    (fused.z * 100.0).round() / 100.0
                );
                self.latency_events.push_back(label);
                while self.latency_events.len() > latency_cfg.history {
                    self.latency_events.pop_front();
                }
            }
        }

        self.fused = fused.clone();
        self.last_step = Some(now);
        fused
    }

    /// Convenience helper that gathers pulses from the provided registry prior
    /// to fusing them.
    pub fn step_from_registry(&mut self, registry: &mut ZRegistry, now: u64) -> ZFused {
        let pulses = registry.gather(now);
        self.step(pulses, now)
    }

    pub fn latest(&self) -> ZFused {
        self.fused.clone()
    }

    pub fn drain_latency_events(&mut self) -> Vec<String> {
        self.latency_events.drain(..).collect()
    }

    pub fn drain_config_events(&mut self) -> Vec<String> {
        self.config_events.drain(..).collect()
    }
}

fn source_lookup_key(source: &ZSource) -> Cow<'_, str> {
    match source {
        ZSource::Microlocal => Cow::Borrowed("Microlocal"),
        ZSource::Maxwell => Cow::Borrowed("Maxwell"),
        ZSource::RealGrad => Cow::Borrowed("RealGrad"),
        ZSource::Desire => Cow::Borrowed("Desire"),
        ZSource::External(name) => Cow::Owned(format!("External.{name}")),
        ZSource::Other(name) => Cow::Owned(format!("Other.{name}")),
    }
}

fn sanitize_tuning_map(map: HashMap<String, f32>) -> HashMap<String, f32> {
    map.into_iter()
        .filter(|(_, value)| value.is_finite())
        .map(|(key, value)| (key, value.max(0.0)))
        .collect()
}

fn record_cfg_diff(events: &mut Vec<String>, label: &str, previous: f32, current: f32) {
    const FLOAT_EPSILON: f32 = 1e-6;
    if (previous - current).abs() > FLOAT_EPSILON {
        events.push(format!("{label}:{:.4}->{:.4}", previous, current));
    }
}

fn diff_tuning_map(
    label: &str,
    previous: &HashMap<String, f32>,
    current: &HashMap<String, f32>,
    events: &mut Vec<String>,
) {
    const FLOAT_EPSILON: f32 = 1e-6;
    let mut keys = BTreeSet::new();
    keys.extend(previous.keys().cloned());
    keys.extend(current.keys().cloned());

    for key in keys {
        match (previous.get(&key), current.get(&key)) {
            (Some(&old), Some(&new)) => {
                if (old - new).abs() > FLOAT_EPSILON {
                    events.push(format!("{label}[{key}]:{:.4}->{:.4}", old, new));
                }
            }
            (Some(&old), None) => {
                events.push(format!("{label}[{key}]:{:.4}->unset", old));
            }
            (None, Some(&new)) => {
                events.push(format!("{label}[{key}]:unset->{:.4}", new));
            }
            (None, None) => {}
        }
    }
}

fn lerp(current: f32, target: f32, alpha: f32) -> f32 {
    let alpha = alpha.clamp(0.0, 1.0);
    (1.0 - alpha) * current + alpha * target
}
#[derive(Clone, Default, Debug)]
pub struct DesireEmitter {
    queue: Arc<Mutex<VecDeque<ZPulse>>>,
}

impl DesireEmitter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enqueue(&self, mut pulse: ZPulse) {
        pulse.source = ZSource::Desire;
        let mut queue = self.queue.lock().expect("desire emitter queue poisoned");
        queue.push_back(pulse);
    }

    pub fn extend<I>(&self, pulses: I)
    where
        I: IntoIterator<Item = ZPulse>,
    {
        let mut queue = self.queue.lock().expect("desire emitter queue poisoned");
        queue.extend(pulses.into_iter().map(|mut pulse| {
            pulse.source = ZSource::Desire;
            pulse
        }));
    }
}

impl ZEmitter for DesireEmitter {
    fn name(&self) -> ZSource {
        ZSource::Desire
    }

    fn tick(&mut self, _now: u64) -> Option<ZPulse> {
        self.queue
            .lock()
            .expect("desire emitter queue poisoned")
            .pop_front()
    }
}

/// Parsed representation of a ZPulse configuration file.
#[derive(Debug, Clone, Deserialize)]
pub struct ZPulseConfigDocument {
    #[serde(default)]
    pub zpulse: ZPulseConfig,
}

impl Default for ZPulseConfigDocument {
    fn default() -> Self {
        Self {
            zpulse: ZPulseConfig::default(),
        }
    }
}

impl ZPulseConfigDocument {
    /// Parses a TOML configuration string.
    pub fn from_toml_str(input: &str) -> Result<Self, ZPulseConfigError> {
        toml::from_str(input).map_err(ZPulseConfigError::from)
    }

    /// Parses a JSON configuration string.
    pub fn from_json_str(input: &str) -> Result<Self, ZPulseConfigError> {
        serde_json::from_str(input).map_err(ZPulseConfigError::from)
    }

    /// Parses a TOML document from an arbitrary reader.
    pub fn from_toml_reader<R: Read>(mut reader: R) -> Result<Self, ZPulseConfigError> {
        let mut buf = String::new();
        reader.read_to_string(&mut buf)?;
        Self::from_toml_str(&buf)
    }

    /// Parses a JSON document from an arbitrary reader.
    pub fn from_json_reader<R: Read>(mut reader: R) -> Result<Self, ZPulseConfigError> {
        let mut buf = String::new();
        reader.read_to_string(&mut buf)?;
        Self::from_json_str(&buf)
    }

    /// Loads a configuration document from a filesystem path. The format is
    /// detected using the file extension (".json" selects JSON, everything else
    /// defaults to TOML).
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ZPulseConfigError> {
        let path = path.as_ref();
        let data = fs::read_to_string(path)?;
        match path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("json"))
        {
            Some(true) => Self::from_json_str(&data),
            _ => Self::from_toml_str(&data),
        }
    }

    /// Merges multiple configuration documents together, applying them in the
    /// order provided where later documents override earlier layers.
    pub fn merge_documents<I>(docs: I) -> Self
    where
        I: IntoIterator<Item = ZPulseConfigDocument>,
    {
        let mut merged = ZPulseConfigDocument::default();
        for doc in docs {
            merged.zpulse.merge_from(&doc.zpulse);
        }
        merged
    }

    /// Loads multiple configuration files and merges them in-order. Missing
    /// files are skipped while any other IO or parse error aborts the load.
    pub fn from_layered_paths<P, I>(paths: I) -> Result<Self, ZPulseConfigError>
    where
        P: AsRef<Path>,
        I: IntoIterator<Item = P>,
    {
        let mut merged = ZPulseConfigDocument::default();
        for path in paths.into_iter() {
            let path_ref = path.as_ref();
            match Self::from_path(path_ref) {
                Ok(doc) => {
                    merged.zpulse.merge_from(&doc.zpulse);
                }
                Err(ZPulseConfigError::Io(err))
                    if err.kind() == ErrorKind::NotFound =>
                {
                    continue;
                }
                Err(err) => return Err(err),
            }
        }
        Ok(merged)
    }
}

/// Runtime configuration for the ZPulse conductor.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ZPulseConfig {
    #[serde(default)]
    pub conductor: ZConductorCfgPartial,
    #[serde(default)]
    gain: ZSourceTable,
    #[serde(default)]
    limit: ZSourceTable,
}

impl ZPulseConfig {
    /// Applies the configuration to the supplied conductor. Existing values are
    /// overwritten when present in the configuration file while unspecified
    /// entries keep their previous defaults.
    pub fn apply(&self, conductor: &mut ZConductor) {
        let prev_cfg = conductor.cfg.clone();
        let prev_gains = conductor.source_gains.clone();
        let prev_limits = conductor.source_limits.clone();

        self.conductor.apply(conductor.cfg_mut());
        let gains = self.gain.resolve();
        if !gains.is_empty() || !conductor.source_gains.is_empty() {
            conductor.set_source_gains(gains);
        }
        let limits = self.limit.resolve();
        if !limits.is_empty() || !conductor.source_limits.is_empty() {
            conductor.set_source_limits(limits);
        }

        let mut events = Vec::new();

        record_cfg_diff(
            &mut events,
            "cfg.alpha_fast",
            prev_cfg.alpha_fast,
            conductor.cfg.alpha_fast,
        );
        record_cfg_diff(
            &mut events,
            "cfg.alpha_slow",
            prev_cfg.alpha_slow,
            conductor.cfg.alpha_slow,
        );
        record_cfg_diff(
            &mut events,
            "cfg.slew_max",
            prev_cfg.slew_max,
            conductor.cfg.slew_max,
        );
        record_cfg_diff(
            &mut events,
            "cfg.z_budget",
            prev_cfg.z_budget,
            conductor.cfg.z_budget,
        );
        record_cfg_diff(
            &mut events,
            "cfg.robust_delta",
            prev_cfg.robust_delta,
            conductor.cfg.robust_delta,
        );
        if prev_cfg.flip_hold != conductor.cfg.flip_hold {
            events.push(format!(
                "cfg.flip_hold:{}->{}",
                prev_cfg.flip_hold, conductor.cfg.flip_hold
            ));
        }
        if prev_cfg.latency_align != conductor.cfg.latency_align {
            events.push(format!(
                "cfg.latency_align:{}->{}",
                prev_cfg.latency_align, conductor.cfg.latency_align
            ));
        }

        diff_tuning_map("gain", &prev_gains, &conductor.source_gains, &mut events);
        diff_tuning_map("limit", &prev_limits, &conductor.source_limits, &mut events);

        if !events.is_empty() {
            conductor.record_config_events(events);
        }
    }

    /// Returns the parsed source gain table.
    pub fn source_gains(&self) -> HashMap<String, f32> {
        self.gain.resolve()
    }

    /// Returns the parsed source limit table.
    pub fn source_limits(&self) -> HashMap<String, f32> {
        self.limit.resolve()
    }

    fn merge_from(&mut self, other: &Self) {
        self.conductor.merge_from(&other.conductor);
        self.gain.merge_from(&other.gain);
        self.limit.merge_from(&other.limit);
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
struct ZSourceTable {
    #[serde(flatten)]
    entries: HashMap<String, ZSourceTableValue>,
}

impl ZSourceTable {
    fn resolve(&self) -> HashMap<String, f32> {
        let mut result = HashMap::new();
        for (key, value) in &self.entries {
            match value {
                ZSourceTableValue::Scalar(v) => {
                    result.insert(key.clone(), *v);
                }
                ZSourceTableValue::Nested(nested) => {
                    for (nested_key, nested_value) in nested {
                        result.insert(format!("{key}.{nested_key}"), *nested_value);
                    }
                }
            }
        }
        result
    }

    fn merge_from(&mut self, other: &Self) {
        for (key, value) in &other.entries {
            match (self.entries.get_mut(key), value) {
                (Some(ZSourceTableValue::Nested(existing)), ZSourceTableValue::Nested(nested)) => {
                    for (nested_key, nested_value) in nested {
                        existing.insert(nested_key.clone(), *nested_value);
                    }
                }
                _ => {
                    self.entries.insert(key.clone(), value.clone());
                }
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum ZSourceTableValue {
    Scalar(f32),
    Nested(HashMap<String, f32>),
}

/// Partial override for [`ZConductorCfg`] read from configuration files.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ZConductorCfgPartial {
    pub alpha_fast: Option<f32>,
    pub alpha_slow: Option<f32>,
    pub flip_hold: Option<u64>,
    pub slew_max: Option<f32>,
    pub z_budget: Option<f32>,
    pub robust_delta: Option<f32>,
    pub latency_align: Option<bool>,
}

impl ZConductorCfgPartial {
    fn apply(&self, cfg: &mut ZConductorCfg) {
        if let Some(alpha_fast) = self.alpha_fast {
            cfg.alpha_fast = alpha_fast.clamp(0.0, 1.0);
        }
        if let Some(alpha_slow) = self.alpha_slow {
            cfg.alpha_slow = alpha_slow.clamp(0.0, 1.0);
        }
        if let Some(flip_hold) = self.flip_hold {
            cfg.flip_hold = flip_hold;
        }
        if let Some(slew_max) = self.slew_max {
            cfg.slew_max = slew_max.max(0.0);
        }
        if let Some(z_budget) = self.z_budget {
            cfg.z_budget = z_budget.max(0.0);
        }
        if let Some(robust_delta) = self.robust_delta {
            cfg.robust_delta = robust_delta.max(0.0);
        }
        if let Some(latency_align) = self.latency_align {
            cfg.latency_align = latency_align;
        }
    }

    fn merge_from(&mut self, other: &Self) {
        if other.alpha_fast.is_some() {
            self.alpha_fast = other.alpha_fast;
        }
        if other.alpha_slow.is_some() {
            self.alpha_slow = other.alpha_slow;
        }
        if other.flip_hold.is_some() {
            self.flip_hold = other.flip_hold;
        }
        if other.slew_max.is_some() {
            self.slew_max = other.slew_max;
        }
        if other.z_budget.is_some() {
            self.z_budget = other.z_budget;
        }
        if other.robust_delta.is_some() {
            self.robust_delta = other.robust_delta;
        }
        if other.latency_align.is_some() {
            self.latency_align = other.latency_align;
        }
        self.pending = retained;

        let mut drift = 0.0;
        let mut attributions = Vec::new();
        let mut total_support = 0.0;

        if !ready.is_empty() {
            total_support = ready
                .iter()
                .filter(|p| !p.is_empty())
                .map(|pulse| pulse.support_mass().max(0.0))
                .sum();
            let mut contributions: Vec<(ZSource, f32, f32)> = ready
                .iter()
                .filter(|p| !p.is_empty())
                .map(|pulse| {
                    let gain = *self.cfg.gain.get(&pulse.source).unwrap_or(&1.0);
                    let base_w = (pulse.quality * gain).max(1e-6);
                    (pulse.source.clone(), base_w, pulse.normalised_drift())
                })
                .collect();

            if !contributions.is_empty() {
                let mut drifts: Vec<f32> = contributions.iter().map(|(_, _, d)| *d).collect();
                let median = median(&mut drifts);
                let mut weight_sum = 0.0f32;
                let mut numerator = 0.0f32;
                for (source, weight, drift_norm) in contributions.iter_mut() {
                    let robust = huber_weight(*drift_norm - median, self.cfg.robust_delta);
                    *weight *= robust;
                    weight_sum += *weight;
                    numerator += *weight * *drift_norm;
                    attributions.push((source.clone(), *weight));
                }
                if weight_sum > 0.0 {
                    drift = numerator / weight_sum;
                    let inv = 1.0 / weight_sum;
                    for attrib in &mut attributions {
                        attrib.1 *= inv;
                    }
                }
            }
        }

        if attributions.is_empty() {
            attributions.push((ZSource::Microlocal, 0.0));
        }

        let filtered = self.apply_temporal_filters(drift, &mut events);
        let mut z = filtered * self.mag_hat.abs().max(1e-6);
        if filtered.abs() <= f32::EPSILON {
            z = 0.0;
        }

        let z_before_limits = z;
        let limited = slew_limit(self.last_z, z, self.cfg.slew_max);
        if (limited - z).abs() > 1e-5 {
            events.push("slew-limited".to_string());
            z = limited;
        }

        let budget = self.cfg.z_budget.max(f32::EPSILON);
        if z.abs() > budget {
            let clamped = z.signum() * budget;
            if self.cfg.back_calculation > 0.0 {
                let correction = self.cfg.back_calculation * (clamped - z_before_limits);
                self.mag_hat = (self.mag_hat + correction).max(0.0);
            }
            z = clamped;
            events.push("saturated".to_string());
        }

        let attributions: Vec<ZAttribution> = attributions
            .into_iter()
            .map(|(source, weight)| ZAttribution { source, weight })
            .collect();

        let mut fused = ZFused {
            z,
            drift,
            support: total_support,
            z_bias: z_before_limits,
            attributions,
            events,
        };

        if let Some(state) = self.freq.as_mut() {
            state.fuse(&mut fused);
        }
        if let Some(state) = self.adaptive.as_mut() {
            state.adapt(&mut fused);
        }

        self.last_z = fused.z;
        self.last_step_ts = Some(now);

        fused
    }
}

/// Errors produced when loading ZPulse configuration files.
#[derive(Debug, thiserror::Error)]
pub enum ZPulseConfigError {
    #[error("failed to read configuration: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse TOML: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("failed to parse JSON: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn toml_config_parses_and_applies() {
        let toml = r#"
            [zpulse.conductor]
            alpha_fast = 0.7
            alpha_slow = 0.2
            flip_hold = 7
            slew_max = 0.5
            z_budget = 1.1
            robust_delta = 0.05
            latency_align = false

            [zpulse.gain]
            Microlocal = 0.8
            External."imu" = 1.3

            [zpulse.limit]
            Maxwell = 0.4
            External."imu" = 0.5
        "#;

        let doc = ZPulseConfigDocument::from_toml_str(toml).expect("valid toml config");
        assert!(!doc.zpulse.source_gains().is_empty());
        assert!(!doc.zpulse.source_limits().is_empty());

        let mut conductor = ZConductor::default();
        doc.zpulse.apply(&mut conductor);

        assert!((conductor.cfg.alpha_fast - 0.7).abs() < 1e-6);
        assert!((conductor.cfg.alpha_slow - 0.2).abs() < 1e-6);
        assert_eq!(conductor.cfg.flip_hold, 7);
        assert!((conductor.cfg.slew_max - 0.5).abs() < 1e-6);
        assert!((conductor.cfg.z_budget - 1.1).abs() < 1e-6);
        assert!((conductor.cfg.robust_delta - 0.05).abs() < 1e-6);
        assert!(!conductor.cfg.latency_align);

        assert!((conductor.source_gains["Microlocal"] - 0.8).abs() < 1e-6);
        assert!((conductor.source_gains["External.imu"] - 1.3).abs() < 1e-6);
        assert!((conductor.source_limits["Maxwell"] - 0.4).abs() < 1e-6);
        assert!((conductor.source_limits["External.imu"] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn json_config_round_trips() {
        let json = r#"
        {
            "zpulse": {
                "conductor": {"alpha_fast": 0.9},
                "gain": {"RealGrad": 0.75, "External": {"lidar": 1.1}},
                "limit": {"RealGrad": 0.6}
            }
        }
        "#;

        let doc = ZPulseConfigDocument::from_json_str(json).expect("valid json config");
        let mut conductor = ZConductor::default();
        doc.zpulse.apply(&mut conductor);

        assert!((conductor.cfg.alpha_fast - 0.9).abs() < 1e-6);
        assert!((conductor.source_gains["RealGrad"] - 0.75).abs() < 1e-6);
        assert!((conductor.source_gains["External.lidar"] - 1.1).abs() < 1e-6);
        assert!((conductor.source_limits["RealGrad"] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn layered_config_merges_in_order() {
        let base = ZPulseConfigDocument::from_toml_str(
            r#"
            [zpulse.conductor]
            alpha_fast = 0.5
            alpha_slow = 0.1

            [zpulse.gain]
            Microlocal = 0.9
            "#,
        )
        .expect("valid base layer");

        let site = ZPulseConfigDocument::from_toml_str(
            r#"
            [zpulse.conductor]
            alpha_slow = 0.3
            flip_hold = 9

            [zpulse.limit]
            Maxwell = 0.4
            "#,
        )
        .expect("valid site layer");

        let run = ZPulseConfigDocument::from_json_str(
            r#"{"zpulse":{"conductor":{"alpha_fast":0.95},"gain":{"External":{"imu":1.2}}}}"#,
        )
        .expect("valid run layer");

        let merged = ZPulseConfigDocument::merge_documents([base, site, run]);
        let mut conductor = ZConductor::default();
        merged.zpulse.apply(&mut conductor);

        assert!((conductor.cfg.alpha_fast - 0.95).abs() < 1e-6);
        assert!((conductor.cfg.alpha_slow - 0.3).abs() < 1e-6);
        assert_eq!(conductor.cfg.flip_hold, 9);
        assert!((conductor.source_gains["Microlocal"] - 0.9).abs() < 1e-6);
        assert!((conductor.source_gains["External.imu"] - 1.2).abs() < 1e-6);
        assert!((conductor.source_limits["Maxwell"] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn applying_config_records_diffs() {
        let mut conductor = ZConductor::default();
        conductor.set_source_gains(HashMap::from([(String::from("Microlocal"), 1.0)]));
        conductor.set_source_limits(HashMap::from([
            (String::from("Microlocal"), 0.5),
            (String::from("Maxwell"), 0.3),
        ]));

        let cfg = ZPulseConfigDocument::from_toml_str(
            r#"
            [zpulse.conductor]
            alpha_fast = 0.7
            latency_align = false

            [zpulse.gain]
            Microlocal = 0.8
            External."imu" = 1.1

            [zpulse.limit]
            Microlocal = 0.4
            "#,
        )
        .expect("valid config");

        cfg.zpulse.apply(&mut conductor);
        let mut events = conductor.drain_config_events();
        events.sort();

        assert!(events.contains(&"cfg.alpha_fast:0.6000->0.7000".to_string()));
        assert!(events.contains(&"cfg.latency_align:true->false".to_string()));
        assert!(events.contains(&"gain[External.imu]:unset->1.1000".to_string()));
        assert!(events.contains(&"gain[Microlocal]:1.0000->0.8000".to_string()));
        assert!(events.contains(&"limit[Maxwell]:0.3000->unset".to_string()));
        assert!(events.contains(&"limit[Microlocal]:0.5000->0.4000".to_string()));
    }

    #[test]
    fn layered_paths_skip_missing_files() {
        use std::fs;
        use std::time::{SystemTime, UNIX_EPOCH};

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time");
        let dir = std::env::temp_dir().join(format!("st_zpulse_cfg_{}", unique.as_nanos()));
        fs::create_dir_all(&dir).expect("create temp dir");

        let base_path = dir.join("base.toml");
        let run_path = dir.join("run.json");
        let missing_path = dir.join("site.toml");

        fs::write(
            &base_path,
            r#"
            [zpulse.conductor]
            z_budget = 0.8
            "#,
        )
        .expect("write base layer");
        fs::write(
            &run_path,
            r#"{"zpulse":{"limit":{"Microlocal":0.2}}}"#,
        )
        .expect("write run layer");

        let merged = ZPulseConfigDocument::from_layered_paths([
            base_path.as_path(),
            missing_path.as_path(),
            run_path.as_path(),
        ])
        .expect("merge layered paths");

        let mut conductor = ZConductor::default();
        merged.zpulse.apply(&mut conductor);

        assert!((conductor.cfg.z_budget - 0.8).abs() < 1e-6);
        assert!((conductor.source_limits["Microlocal"] - 0.2).abs() < 1e-6);

        fs::remove_file(&base_path).ok();
        fs::remove_file(&run_path).ok();
        fs::remove_dir(&dir).ok();
    }

    #[test]
    fn source_gain_and_limit_control_bias() {
        let mut cfg = ZConductorCfg::default();
        cfg.alpha_fast = 1.0;
        cfg.alpha_slow = 1.0;
        cfg.slew_max = 10.0;
        cfg.z_budget = 1.0;
        cfg.flip_hold = 0;

        let mut conductor = ZConductor::new(cfg);
        conductor.set_source_gains(HashMap::from([(String::from("Maxwell"), 2.0)]));
        conductor.set_source_limits(HashMap::from([(String::from("Maxwell"), 0.3)]));

        let pulse = ZPulse {
            source: ZSource::Maxwell,
            ts: 0,
            band_energy: (0.0, 0.0, 0.0),
            drift: 0.0,
            z_bias: 0.5,
            support: 1.0,
            quality: 1.0,
            stderr: 0.0,
            latency_ms: 0.0,
        };

        let fused = conductor.step(vec![pulse], 0);
        assert!((fused.z - 0.3).abs() < 1e-6);
    }
}
