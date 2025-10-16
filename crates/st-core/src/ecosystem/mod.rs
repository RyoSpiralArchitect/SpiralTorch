use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::backend::unison_heuristics::RankKind;

/// Source that produced a heuristic decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeuristicSource {
    SoftLogic,
    HardDsl,
    KeyValue,
    Generated,
    Fallback,
}

impl HeuristicSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            HeuristicSource::SoftLogic => "soft_logic",
            HeuristicSource::HardDsl => "dsl",
            HeuristicSource::KeyValue => "kv",
            HeuristicSource::Generated => "generated",
            HeuristicSource::Fallback => "fallback",
        }
    }
}

/// Compact representation of a choice emitted by one of the heuristics pipelines.
#[derive(Clone, Debug, PartialEq)]
pub struct HeuristicChoiceSummary {
    pub use_two_stage: bool,
    pub workgroup: u32,
    pub lanes: u32,
    pub channel_stride: u32,
    pub algo_hint: Option<String>,
    pub compaction_tile: u32,
    pub fft_tile_cols: u32,
    pub fft_radix: u32,
    pub fft_segments: u32,
}

impl HeuristicChoiceSummary {
    pub fn new(
        use_two_stage: bool,
        workgroup: u32,
        lanes: u32,
        channel_stride: u32,
        algo_hint: Option<String>,
        compaction_tile: u32,
        fft_tile_cols: u32,
        fft_radix: u32,
        fft_segments: u32,
    ) -> Self {
        Self {
            use_two_stage,
            workgroup,
            lanes,
            channel_stride,
            algo_hint,
            compaction_tile,
            fft_tile_cols,
            fft_radix,
            fft_segments,
        }
    }
}

/// Record that captures a single heuristic decision.
#[derive(Clone, Debug, PartialEq)]
pub struct HeuristicDecision {
    pub subsystem: String,
    pub kind: String,
    pub rows: u32,
    pub cols: u32,
    pub k: u32,
    pub choice: HeuristicChoiceSummary,
    pub score_hint: Option<f32>,
    pub source: HeuristicSource,
    pub issued_at: SystemTime,
}

impl HeuristicDecision {
    pub fn issued_at_secs(&self) -> f64 {
        self.issued_at
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }
}

/// Summary of the configuration that produced a roundtable schedule.
#[derive(Clone, Debug, PartialEq)]
pub struct RoundtableConfigSummary {
    pub top_k: u32,
    pub mid_k: u32,
    pub bottom_k: u32,
    pub here_tolerance: f32,
    pub extras: HashMap<String, bool>,
}

impl RoundtableConfigSummary {
    pub fn new(top_k: u32, mid_k: u32, bottom_k: u32, here_tolerance: f32) -> Self {
        Self {
            top_k,
            mid_k,
            bottom_k,
            here_tolerance,
            extras: HashMap::new(),
        }
    }
}

/// Compact representation of a rank plan feeding the roundtable.
#[derive(Clone, Debug, PartialEq)]
pub struct RankPlanSummary {
    pub kind: RankKind,
    pub rows: u32,
    pub cols: u32,
    pub k: u32,
    pub workgroup: u32,
    pub lanes: u32,
    pub channel_stride: u32,
    pub tile: u32,
    pub compaction_tile: u32,
    pub subgroup: bool,
    pub fft_tile: u32,
    pub fft_radix: u32,
    pub fft_segments: u32,
}

impl RankPlanSummary {
    pub fn new(kind: RankKind, rows: u32, cols: u32, k: u32) -> Self {
        Self {
            kind,
            rows,
            cols,
            k,
            workgroup: 0,
            lanes: 0,
            channel_stride: 0,
            tile: 0,
            compaction_tile: 0,
            subgroup: false,
            fft_tile: 0,
            fft_radix: 0,
            fft_segments: 0,
        }
    }
}

/// Snapshot describing the live distribution wiring.
#[derive(Clone, Debug, PartialEq)]
pub struct DistributionSummary {
    pub node_id: String,
    pub mode: String,
    pub summary_window: usize,
    pub push_interval_ms: u64,
    pub meta_endpoints: Vec<String>,
}

/// Log entry that describes cross-cutting connectors (e.g. between RL/Rec/Nn).
#[derive(Clone, Debug, PartialEq)]
pub struct ConnectorEvent {
    pub name: String,
    pub stage: String,
    pub metadata: HashMap<String, String>,
    pub issued_at: SystemTime,
}

impl ConnectorEvent {
    pub fn issued_at_secs(&self) -> f64 {
        self.issued_at
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }
}

/// Summary for the current roundtable plan plus optional distribution hints.
#[derive(Clone, Debug, PartialEq)]
pub struct RoundtableSummary {
    pub rows: u32,
    pub cols: u32,
    pub config: RoundtableConfigSummary,
    pub plans: Vec<RankPlanSummary>,
    pub autopilot_enabled: bool,
    pub distribution: Option<DistributionSummary>,
    pub issued_at: SystemTime,
}

impl RoundtableSummary {
    pub fn issued_at_secs(&self) -> f64 {
        self.issued_at
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }
}

#[derive(Default)]
struct EcosystemState {
    heuristics: Vec<HeuristicDecision>,
    roundtables: Vec<RoundtableSummary>,
    connectors: Vec<ConnectorEvent>,
}

/// Global registry that collects heuristics and scheduling decisions across crates.
#[derive(Default)]
pub struct EcosystemRegistry {
    state: Mutex<EcosystemState>,
}

impl EcosystemRegistry {
    pub fn global() -> &'static EcosystemRegistry {
        static REGISTRY: OnceLock<EcosystemRegistry> = OnceLock::new();
        REGISTRY.get_or_init(EcosystemRegistry::default)
    }

    pub fn record_heuristic(&self, decision: HeuristicDecision) {
        let mut state = self.state.lock().unwrap();
        state.heuristics.push(decision);
    }

    pub fn record_roundtable(&self, summary: RoundtableSummary) {
        let mut state = self.state.lock().unwrap();
        state.roundtables.push(summary);
    }

    pub fn record_connector(&self, event: ConnectorEvent) {
        let mut state = self.state.lock().unwrap();
        state.connectors.push(event);
    }

    pub fn snapshot(&self) -> EcosystemReport {
        let state = self.state.lock().unwrap();
        EcosystemReport {
            heuristics: state.heuristics.clone(),
            roundtables: state.roundtables.clone(),
            connectors: state.connectors.clone(),
        }
    }

    pub fn drain(&self) -> EcosystemReport {
        let mut state = self.state.lock().unwrap();
        EcosystemReport {
            heuristics: std::mem::take(&mut state.heuristics),
            roundtables: std::mem::take(&mut state.roundtables),
            connectors: std::mem::take(&mut state.connectors),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EcosystemReport {
    pub heuristics: Vec<HeuristicDecision>,
    pub roundtables: Vec<RoundtableSummary>,
    pub connectors: Vec<ConnectorEvent>,
}

impl EcosystemReport {
    pub fn heuristics(&self) -> &[HeuristicDecision] {
        &self.heuristics
    }

    pub fn roundtables(&self) -> &[RoundtableSummary] {
        &self.roundtables
    }

    pub fn connectors(&self) -> &[ConnectorEvent] {
        &self.connectors
    }
}
