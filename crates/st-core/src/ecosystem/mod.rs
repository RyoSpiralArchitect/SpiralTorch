use std::collections::{BTreeMap, HashMap, VecDeque};
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
    #[allow(
        clippy::too_many_arguments,
        reason = "Telemetry summary retains explicit fields for downstream consumers"
    )]
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

/// Telemetry sample emitted by a component participating in the ecosystem.
#[derive(Clone, Debug, PartialEq)]
pub struct MetricSample {
    pub name: String,
    pub value: f64,
    pub tags: HashMap<String, String>,
    pub unit: Option<String>,
    pub issued_at: SystemTime,
}

impl MetricSample {
    pub fn new(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
            tags: HashMap::new(),
            unit: None,
            issued_at: SystemTime::now(),
        }
    }

    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }

    pub fn issued_at_secs(&self) -> f64 {
        self.issued_at
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }
}

/// Aggregate statistics derived from metric samples that share the same tags.
#[derive(Clone, Debug, PartialEq)]
pub struct MetricDigest {
    pub name: String,
    pub tags: Vec<(String, String)>,
    pub unit: Option<String>,
    pub count: usize,
    pub sum: f64,
    pub min: f64,
    pub max: f64,
    pub average: f64,
    pub first_at: SystemTime,
    pub last_value: f64,
    pub last_at: SystemTime,
}

impl MetricDigest {
    pub fn first_at_secs(&self) -> f64 {
        self.first_at
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }

    pub fn last_at_secs(&self) -> f64 {
        self.last_at
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EcosystemCapacity {
    pub heuristics: usize,
    pub roundtables: usize,
    pub connectors: usize,
    pub metrics: usize,
}

impl Default for EcosystemCapacity {
    fn default() -> Self {
        Self {
            heuristics: 2048,
            roundtables: 256,
            connectors: 512,
            metrics: 4096,
        }
    }
}

#[derive(Default)]
struct EcosystemState {
    heuristics: VecDeque<HeuristicDecision>,
    roundtables: VecDeque<RoundtableSummary>,
    connectors: VecDeque<ConnectorEvent>,
    metrics: VecDeque<MetricSample>,
    capacity: EcosystemCapacity,
}

impl EcosystemState {
    fn enforce_capacities(&mut self) {
        truncate_queue(&mut self.heuristics, self.capacity.heuristics);
        truncate_queue(&mut self.roundtables, self.capacity.roundtables);
        truncate_queue(&mut self.connectors, self.capacity.connectors);
        truncate_queue(&mut self.metrics, self.capacity.metrics);
    }
}

fn truncate_queue<T>(queue: &mut VecDeque<T>, cap: usize) {
    if cap == 0 {
        queue.clear();
        return;
    }
    while queue.len() > cap {
        queue.pop_front();
    }
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
        state.heuristics.push_back(decision);
        state.enforce_capacities();
    }

    pub fn record_roundtable(&self, summary: RoundtableSummary) {
        let mut state = self.state.lock().unwrap();
        state.roundtables.push_back(summary);
        state.enforce_capacities();
    }

    pub fn record_connector(&self, event: ConnectorEvent) {
        let mut state = self.state.lock().unwrap();
        state.connectors.push_back(event);
        state.enforce_capacities();
    }

    pub fn record_metric(&self, sample: MetricSample) {
        let mut state = self.state.lock().unwrap();
        state.metrics.push_back(sample);
        state.enforce_capacities();
    }

    pub fn configure<F>(&self, mutator: F)
    where
        F: FnOnce(&mut EcosystemCapacity),
    {
        let mut state = self.state.lock().unwrap();
        mutator(&mut state.capacity);
        state.enforce_capacities();
    }

    pub fn capacity(&self) -> EcosystemCapacity {
        let state = self.state.lock().unwrap();
        state.capacity
    }

    pub fn snapshot(&self) -> EcosystemReport {
        let state = self.state.lock().unwrap();
        EcosystemReport {
            heuristics: state.heuristics.iter().cloned().collect(),
            roundtables: state.roundtables.iter().cloned().collect(),
            connectors: state.connectors.iter().cloned().collect(),
            metrics: state.metrics.iter().cloned().collect(),
            metric_digests: summarise_metrics(state.metrics.iter()),
        }
    }

    pub fn drain(&self) -> EcosystemReport {
        let mut state = self.state.lock().unwrap();
        let metrics: Vec<MetricSample> = state.metrics.drain(..).collect();
        let metric_digests = summarise_metrics(metrics.iter());
        EcosystemReport {
            heuristics: state.heuristics.drain(..).collect(),
            roundtables: state.roundtables.drain(..).collect(),
            connectors: state.connectors.drain(..).collect(),
            metrics,
            metric_digests,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EcosystemReport {
    pub heuristics: Vec<HeuristicDecision>,
    pub roundtables: Vec<RoundtableSummary>,
    pub connectors: Vec<ConnectorEvent>,
    pub metrics: Vec<MetricSample>,
    pub metric_digests: Vec<MetricDigest>,
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

    pub fn metrics(&self) -> &[MetricSample] {
        &self.metrics
    }

    pub fn metric_digests(&self) -> &[MetricDigest] {
        &self.metric_digests
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct MetricKey {
    name: String,
    tags: Vec<(String, String)>,
}

#[derive(Clone, Debug)]
struct MetricAccumulator {
    count: usize,
    sum: f64,
    min: f64,
    max: f64,
    first_at: SystemTime,
    last_value: f64,
    last_at: SystemTime,
    unit: Option<String>,
}

impl MetricAccumulator {
    fn new(sample: &MetricSample) -> Self {
        Self {
            count: 1,
            sum: sample.value,
            min: sample.value,
            max: sample.value,
            first_at: sample.issued_at,
            last_value: sample.value,
            last_at: sample.issued_at,
            unit: sample.unit.clone(),
        }
    }

    fn push(&mut self, sample: &MetricSample) {
        self.count += 1;
        self.sum += sample.value;
        self.min = self.min.min(sample.value);
        self.max = self.max.max(sample.value);
        self.last_value = sample.value;
        self.last_at = sample.issued_at;
        if self.unit.is_none() {
            self.unit = sample.unit.clone();
        }
    }

    fn into_digest(self, key: MetricKey) -> MetricDigest {
        let average = self.sum / self.count as f64;
        MetricDigest {
            name: key.name,
            tags: key.tags,
            unit: self.unit,
            count: self.count,
            sum: self.sum,
            min: self.min,
            max: self.max,
            average,
            first_at: self.first_at,
            last_value: self.last_value,
            last_at: self.last_at,
        }
    }
}

fn summarise_metrics<'a>(samples: impl Iterator<Item = &'a MetricSample>) -> Vec<MetricDigest> {
    let mut map: BTreeMap<MetricKey, MetricAccumulator> = BTreeMap::new();
    for sample in samples {
        let mut tags: Vec<(String, String)> = sample
            .tags
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        tags.sort_unstable();
        let key = MetricKey {
            name: sample.name.clone(),
            tags,
        };
        map.entry(key)
            .and_modify(|acc| acc.push(sample))
            .or_insert_with(|| MetricAccumulator::new(sample));
    }

    map.into_iter()
        .map(|(key, acc)| acc.into_digest(key))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_digest_accumulates_stats() {
        let registry = EcosystemRegistry::default();
        registry.record_metric(MetricSample::new("loss", 1.0).with_tag("stage", "train"));
        registry.record_metric(
            MetricSample::new("loss", 0.5)
                .with_tag("stage", "train")
                .with_unit("nats"),
        );

        let report = registry.snapshot();
        let digest = report
            .metric_digests()
            .iter()
            .find(|d| d.name == "loss")
            .expect("missing digest");

        assert_eq!(digest.count, 2);
        assert!((digest.average - 0.75).abs() < f64::EPSILON);
        assert_eq!(digest.min, 0.5);
        assert_eq!(digest.max, 1.0);
        assert_eq!(digest.unit.as_deref(), Some("nats"));
    }

    #[test]
    fn capacities_trim_backlog() {
        let registry = EcosystemRegistry::default();
        registry.configure(|cap| {
            cap.heuristics = 1;
            cap.roundtables = 1;
            cap.connectors = 1;
            cap.metrics = 1;
        });

        registry.record_heuristic(HeuristicDecision {
            subsystem: "test".into(),
            kind: "k".into(),
            rows: 1,
            cols: 1,
            k: 1,
            choice: HeuristicChoiceSummary::new(false, 1, 1, 1, None, 1, 1, 1, 1),
            score_hint: None,
            source: HeuristicSource::Generated,
            issued_at: SystemTime::now(),
        });
        registry.record_heuristic(HeuristicDecision {
            subsystem: "test".into(),
            kind: "k".into(),
            rows: 2,
            cols: 2,
            k: 2,
            choice: HeuristicChoiceSummary::new(false, 1, 1, 1, None, 1, 1, 1, 1),
            score_hint: None,
            source: HeuristicSource::Generated,
            issued_at: SystemTime::now(),
        });

        registry.record_roundtable(RoundtableSummary {
            rows: 1,
            cols: 1,
            config: RoundtableConfigSummary::new(1, 1, 1, 0.1),
            plans: Vec::new(),
            autopilot_enabled: false,
            distribution: None,
            issued_at: SystemTime::now(),
        });
        registry.record_roundtable(RoundtableSummary {
            rows: 2,
            cols: 2,
            config: RoundtableConfigSummary::new(1, 1, 1, 0.1),
            plans: Vec::new(),
            autopilot_enabled: false,
            distribution: None,
            issued_at: SystemTime::now(),
        });

        registry.record_connector(ConnectorEvent {
            name: "a".into(),
            stage: "start".into(),
            metadata: HashMap::new(),
            issued_at: SystemTime::now(),
        });
        registry.record_connector(ConnectorEvent {
            name: "b".into(),
            stage: "end".into(),
            metadata: HashMap::new(),
            issued_at: SystemTime::now(),
        });

        registry.record_metric(MetricSample::new("x", 1.0));
        registry.record_metric(MetricSample::new("x", 2.0));

        let report = registry.snapshot();
        assert_eq!(report.heuristics.len(), 1);
        assert_eq!(report.roundtables.len(), 1);
        assert_eq!(report.connectors.len(), 1);
        assert_eq!(report.metrics.len(), 1);
    }
}
