use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;

use crate::backend::device_caps::{BackendKind, DeviceCaps};
use crate::runtime::blackcat::{BlackCatRuntime, StepMetrics};

/// Autopilot operating modes controlling how aggressively tuning overrides
/// user supplied hints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutoMode {
    /// Fully automatic (default). ENV/CLI are considered hints only.
    Auto,
    /// Respect ENV/CLI as priors; still allowed to adjust if harmful.
    Hint,
    /// Disabled. No dynamic tuning.
    Off,
}

/// Simple knob declaration (identifier plus finite domain).
#[derive(Clone, Debug)]
pub struct KnobSpec {
    pub id: &'static str,
    pub domain: Vec<String>,
    pub default_idx: usize,
}

/// Configuration consumed by [`Autopilot::new`].
#[derive(Clone, Debug)]
pub struct AutoConfig {
    pub mode: AutoMode,
    pub knobs: Vec<KnobSpec>,
    pub feat_dim: usize,
}

impl Default for AutoConfig {
    fn default() -> Self {
        Self {
            mode: AutoMode::Auto,
            knobs: Vec::new(),
            feat_dim: 8,
        }
    }
}

/// Minimal persisted profile per device key.
#[derive(Clone, Debug, Default)]
pub struct DeviceProfile {
    pub chosen: HashMap<String, String>,
    pub step_ms_p50: f32,
    pub mem_mb_p95: f32,
    pub retry_rate: f32,
}

/// Front-end around [`BlackCatRuntime`] that persists the best known
/// configuration per-device and per-context.
pub struct Autopilot {
    caps: DeviceCaps,
    runtime: BlackCatRuntime,
    profile_key: String,
    profile: DeviceProfile,
    picks: HashMap<String, String>,
    mode: AutoMode,
    feat_dim: usize,
}

impl Autopilot {
    pub fn new(caps: DeviceCaps, cfg: AutoConfig, runtime: BlackCatRuntime) -> Self {
        let key = make_key(&caps);
        let profile = load_profile(&key).unwrap_or_default();
        Self {
            caps,
            runtime,
            profile_key: key,
            profile,
            picks: HashMap::new(),
            mode: cfg.mode,
            feat_dim: cfg.feat_dim.max(1),
        }
    }

    /// Builds a contextual feature vector that joins workload statistics with
    /// device properties. The vector is padded or truncated to the configured
    /// feature dimension.
    pub fn build_context(
        &self,
        batch: u32,
        tiles: u32,
        depth: u32,
        device_load: f64,
        extras: &[(String, f64)],
    ) -> Vec<f64> {
        let mut ctx = vec![
            1.0,
            batch as f64,
            tiles as f64,
            depth as f64,
            (self.caps.lane_width as f64) / 1024.0,
            (self.caps.max_workgroup as f64) / 4096.0,
            device_load,
            if self.caps.subgroup { 1.0 } else { 0.0 },
        ];
        ctx.extend(extras.iter().map(|(_, value)| *value));
        ctx.truncate(self.feat_dim);
        if ctx.len() < self.feat_dim {
            ctx.resize(self.feat_dim, 0.0);
        }
        ctx
    }

    /// Suggests knob values for the provided context.
    pub fn suggest(&mut self, context: Vec<f64>) -> &HashMap<String, String> {
        if self.mode == AutoMode::Off {
            self.picks.clear();
            return &self.picks;
        }
        let picks = self.runtime.choose(context);
        self.picks = picks;
        if self.mode == AutoMode::Hint {
            for (id, choice) in self.profile.chosen.iter() {
                self.picks
                    .entry(id.clone())
                    .or_insert_with(|| choice.clone());
            }
        }
        &self.picks
    }

    /// Reports the observed metrics back to the runtime and persists an update
    /// to the on-disk profile.
    pub fn report(&mut self, metrics: &StepMetrics) {
        if self.mode == AutoMode::Off {
            return;
        }
        let _ = self.runtime.post_step(metrics);
        update_profile_stats(&mut self.profile, metrics);
        self.profile.chosen = self.picks.clone();
        let _ = save_profile(&self.profile_key, &self.profile);
    }

    /// Exposes the underlying runtime for advanced integrations.
    pub fn runtime_mut(&mut self) -> &mut BlackCatRuntime {
        &mut self.runtime
    }

    /// Returns the currently configured operating mode.
    pub fn mode(&self) -> AutoMode {
        self.mode
    }
}

fn make_key(caps: &DeviceCaps) -> String {
    let backend = match caps.backend {
        BackendKind::Wgpu => "wgpu",
        BackendKind::Cuda => "cuda",
        BackendKind::Hip => "hip",
        BackendKind::Cpu => "cpu",
    };
    format!(
        "backend={backend};lane={};wg={};subgroup={}",
        caps.lane_width, caps.max_workgroup, caps.subgroup as u8
    )
}

fn profile_dir() -> PathBuf {
    let mut dir = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    dir.push(".spiraltorch");
    dir.push("profile.d");
    let _ = fs::create_dir_all(&dir);
    dir
}

fn sanitize(component: &str) -> String {
    component
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

fn load_profile(key: &str) -> Option<DeviceProfile> {
    let mut path = profile_dir();
    path.push(sanitize(key));
    path.set_extension("prof");
    let mut file = File::open(path).ok()?;
    let mut buf = String::new();
    file.read_to_string(&mut buf).ok()?;
    parse_profile(&buf)
}

fn save_profile(key: &str, profile: &DeviceProfile) -> std::io::Result<()> {
    let mut path = profile_dir();
    path.push(sanitize(key));
    path.set_extension("prof");
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)?;
    let mut buf = String::new();
    buf.push_str(&format!("step_ms_p50={:.6}\n", profile.step_ms_p50));
    buf.push_str(&format!("mem_mb_p95={:.6}\n", profile.mem_mb_p95));
    buf.push_str(&format!("retry_rate={:.6}\n", profile.retry_rate));
    for (id, value) in profile.chosen.iter() {
        buf.push_str(&format!("chosen.{id}={value}\n"));
    }
    file.write_all(buf.as_bytes())
}

fn parse_profile(contents: &str) -> Option<DeviceProfile> {
    let mut profile = DeviceProfile::default();
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            if key == "step_ms_p50" {
                profile.step_ms_p50 = value.parse().unwrap_or(0.0);
            } else if key == "mem_mb_p95" {
                profile.mem_mb_p95 = value.parse().unwrap_or(0.0);
            } else if key == "retry_rate" {
                profile.retry_rate = value.parse().unwrap_or(0.0);
            } else if let Some(id) = key.strip_prefix("chosen.") {
                profile.chosen.insert(id.to_string(), value.to_string());
            }
        }
    }
    Some(profile)
}

fn update_profile_stats(profile: &mut DeviceProfile, metrics: &StepMetrics) {
    const ALPHA: f32 = 0.1;
    if profile.step_ms_p50 == 0.0 {
        profile.step_ms_p50 = metrics.step_time_ms as f32;
    } else {
        profile.step_ms_p50 =
            (1.0 - ALPHA) * profile.step_ms_p50 + ALPHA * metrics.step_time_ms as f32;
    }
    if profile.mem_mb_p95 == 0.0 {
        profile.mem_mb_p95 = metrics.mem_peak_mb as f32;
    } else {
        profile.mem_mb_p95 =
            (1.0 - ALPHA) * profile.mem_mb_p95 + ALPHA * metrics.mem_peak_mb as f32;
    }
    if profile.retry_rate == 0.0 {
        profile.retry_rate = metrics.retry_rate as f32;
    } else {
        profile.retry_rate = (1.0 - ALPHA) * profile.retry_rate + ALPHA * metrics.retry_rate as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::blackcat::{ChoiceGroups, RewardCfg};
    use std::collections::HashMap;

    fn demo_runtime() -> BlackCatRuntime {
        let groups = ChoiceGroups {
            groups: HashMap::from([
                ("wg".to_string(), vec!["128".to_string(), "256".to_string()]),
                (
                    "tile".to_string(),
                    vec!["512".to_string(), "1024".to_string(), "2048".to_string()],
                ),
            ]),
        };
        let mut runtime = BlackCatRuntime::new(
            crate::runtime::blackcat::zmeta::ZMetaParams::default(),
            groups,
            8,
            crate::runtime::blackcat::bandit::SoftBanditMode::TS,
            None,
        );
        runtime.reward = RewardCfg::default();
        runtime
    }

    #[test]
    fn autopilot_persists_profile() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut autopilot = Autopilot::new(caps, AutoConfig::default(), demo_runtime());
        let context = autopilot.build_context(8, 128, 64, 0.5, &[]);
        let picks = autopilot.suggest(context);
        assert!(!picks.is_empty());
        let metrics = StepMetrics {
            step_time_ms: 12.5,
            mem_peak_mb: 256.0,
            retry_rate: 0.1,
            extra: HashMap::new(),
        };
        autopilot.report(&metrics);
        assert!(autopilot.profile.step_ms_p50 > 0.0);
    }
}
