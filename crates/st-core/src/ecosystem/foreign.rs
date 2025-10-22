use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::str::FromStr;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime};

/// Describes the implementation language of a foreign runtime.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ForeignLanguage {
    /// Go bindings (spiraltorch-go).
    Go,
    /// Julia bindings (SpiralTorch.jl).
    Julia,
    /// Any other runtime identified by name.
    Other(String),
}

impl ForeignLanguage {
    /// Returns a canonical lowercase representation of the language label.
    pub fn as_str(&self) -> &str {
        match self {
            ForeignLanguage::Go => "go",
            ForeignLanguage::Julia => "julia",
            ForeignLanguage::Other(value) => value.as_str(),
        }
    }

    /// Attempts to parse the provided label into a known language variant.
    pub fn parse(label: &str) -> Self {
        match label.trim().to_ascii_lowercase().as_str() {
            "go" | "golang" => ForeignLanguage::Go,
            "julia" => ForeignLanguage::Julia,
            other => ForeignLanguage::Other(other.to_string()),
        }
    }
}

impl fmt::Display for ForeignLanguage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ForeignLanguage {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(ForeignLanguage::parse(s))
    }
}

/// Public descriptor for a registered foreign runtime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForeignRuntimeDescriptor {
    pub id: String,
    pub language: ForeignLanguage,
    pub version: String,
    pub capabilities: Vec<String>,
}

impl ForeignRuntimeDescriptor {
    pub fn new(
        id: impl Into<String>,
        language: ForeignLanguage,
        version: impl Into<String>,
        capabilities: Vec<String>,
    ) -> Self {
        Self {
            id: id.into(),
            language,
            version: version.into(),
            capabilities,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ForeignKernelStatus {
    pub operation: String,
    pub average_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub invocations: u64,
    pub errors: u64,
    pub last_latency_ms: Option<f64>,
    pub window_len: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ForeignRuntimeStatus {
    pub descriptor: ForeignRuntimeDescriptor,
    pub registered_at: SystemTime,
    pub last_heartbeat: SystemTime,
    pub kernels: Vec<ForeignKernelStatus>,
}

struct KernelStats {
    history: VecDeque<f64>,
    invocations: u64,
    errors: u64,
    last_latency: Option<f64>,
    max_history: usize,
}

impl KernelStats {
    fn new(max_history: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_history.min(32)),
            invocations: 0,
            errors: 0,
            last_latency: None,
            max_history: max_history.max(1),
        }
    }

    fn record(&mut self, latency_ms: f64, ok: bool) {
        if self.history.len() == self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(latency_ms);
        self.invocations = self.invocations.saturating_add(1);
        if !ok {
            self.errors = self.errors.saturating_add(1);
        }
        self.last_latency = Some(latency_ms);
    }

    fn average(&self) -> f64 {
        if self.history.is_empty() {
            0.0
        } else {
            let sum: f64 = self.history.iter().copied().sum();
            sum / self.history.len() as f64
        }
    }

    fn p95(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f64> = self.history.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((sorted.len() as f64) * 0.95).ceil() as usize;
        let clamped = idx.saturating_sub(1).min(sorted.len().saturating_sub(1));
        sorted[clamped]
    }

    fn snapshot(&self, operation: &str) -> ForeignKernelStatus {
        ForeignKernelStatus {
            operation: operation.to_string(),
            average_latency_ms: self.average(),
            p95_latency_ms: self.p95(),
            invocations: self.invocations,
            errors: self.errors,
            last_latency_ms: self.last_latency,
            window_len: self.history.len(),
        }
    }
}

struct ForeignRuntimeRecord {
    descriptor: Option<ForeignRuntimeDescriptor>,
    registered_at: SystemTime,
    last_heartbeat: SystemTime,
    kernels: HashMap<String, KernelStats>,
}

impl ForeignRuntimeRecord {
    fn new(descriptor: ForeignRuntimeDescriptor) -> Self {
        let now = SystemTime::now();
        Self {
            descriptor: Some(descriptor),
            registered_at: now,
            last_heartbeat: now,
            kernels: HashMap::new(),
        }
    }

    fn update_descriptor(&mut self, descriptor: ForeignRuntimeDescriptor) {
        self.descriptor = Some(descriptor);
        self.last_heartbeat = SystemTime::now();
    }

    fn descriptor(&self) -> Option<&ForeignRuntimeDescriptor> {
        self.descriptor.as_ref()
    }

    fn record_latency(&mut self, operation: &str, latency_ms: f64, ok: bool, max_history: usize) {
        let stats = self
            .kernels
            .entry(operation.to_string())
            .or_insert_with(|| KernelStats::new(max_history));
        stats.record(latency_ms, ok);
        self.last_heartbeat = SystemTime::now();
    }

    fn snapshot(&self, max_history: usize) -> Option<ForeignRuntimeStatus> {
        let descriptor = self.descriptor()?.clone();
        let mut kernels: Vec<ForeignKernelStatus> = self
            .kernels
            .iter()
            .map(|(operation, stats)| {
                let mut clone = stats.clone();
                clone.max_history = max_history;
                clone.snapshot(operation)
            })
            .collect();
        kernels.sort_by(|a, b| a.operation.cmp(&b.operation));
        Some(ForeignRuntimeStatus {
            descriptor,
            registered_at: self.registered_at,
            last_heartbeat: self.last_heartbeat,
            kernels,
        })
    }
}

impl Clone for KernelStats {
    fn clone(&self) -> Self {
        Self {
            history: self.history.clone(),
            invocations: self.invocations,
            errors: self.errors,
            last_latency: self.last_latency,
            max_history: self.max_history,
        }
    }
}

struct ForeignRegistryInner {
    runtimes: HashMap<String, ForeignRuntimeRecord>,
    max_history: usize,
}

impl ForeignRegistryInner {
    fn new(max_history: usize) -> Self {
        Self {
            runtimes: HashMap::new(),
            max_history: max_history.max(1),
        }
    }
}

/// Registry that tracks foreign runtimes and their performance telemetry.
pub struct ForeignRegistry {
    inner: Mutex<ForeignRegistryInner>,
}

impl ForeignRegistry {
    /// Returns the process-wide registry instance.
    pub fn global() -> &'static Self {
        static REGISTRY: OnceLock<ForeignRegistry> = OnceLock::new();
        REGISTRY.get_or_init(|| ForeignRegistry::new(32))
    }

    /// Creates a new registry that retains up to `max_history` latency samples
    /// per operation.
    pub fn new(max_history: usize) -> Self {
        Self {
            inner: Mutex::new(ForeignRegistryInner::new(max_history)),
        }
    }

    /// Registers or updates a runtime with the provided descriptor.
    pub fn register_runtime(&self, descriptor: ForeignRuntimeDescriptor) -> bool {
        let mut guard = match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let entry = guard.runtimes.entry(descriptor.id.clone());
        use std::collections::hash_map::Entry;
        match entry {
            Entry::Occupied(mut occupied) => {
                occupied.get_mut().update_descriptor(descriptor);
            }
            Entry::Vacant(vacant) => {
                vacant.insert(ForeignRuntimeRecord::new(descriptor));
            }
        }
        true
    }

    /// Updates the heartbeat timestamp for the requested runtime.
    pub fn record_heartbeat(&self, runtime_id: &str) -> bool {
        let mut guard = match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(record) = guard.runtimes.get_mut(runtime_id) {
            record.last_heartbeat = SystemTime::now();
            true
        } else {
            false
        }
    }

    /// Records a latency sample for the given runtime and operation.
    pub fn record_latency(
        &self,
        runtime_id: &str,
        operation: &str,
        latency: Duration,
        ok: bool,
    ) -> bool {
        let mut guard = match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let max_history = guard.max_history;
        let record = match guard.runtimes.get_mut(runtime_id) {
            Some(record) => record,
            None => return false,
        };
        let latency_ms = latency.as_secs_f64() * 1_000.0;
        record.record_latency(operation, latency_ms, ok, max_history);
        true
    }

    /// Returns a snapshot of all registered runtimes.
    pub fn snapshot(&self) -> Vec<ForeignRuntimeStatus> {
        let guard = match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let max_history = guard.max_history;
        let mut statuses: Vec<ForeignRuntimeStatus> = guard
            .runtimes
            .values()
            .filter_map(|record| record.snapshot(max_history))
            .collect();
        statuses.sort_by(|a, b| a.descriptor.id.cmp(&b.descriptor.id));
        statuses
    }

    /// Returns the runtime that currently provides the lowest average latency
    /// for the requested capability (case-insensitive).
    pub fn best_runtime_for(&self, capability: &str) -> Option<ForeignRuntimeStatus> {
        let needle = capability.trim().to_ascii_lowercase();
        let guard = match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let max_history = guard.max_history;
        guard
            .runtimes
            .values()
            .filter_map(|record| {
                let descriptor = record.descriptor()?;
                if descriptor
                    .capabilities
                    .iter()
                    .any(|cap| cap.to_ascii_lowercase() == needle)
                {
                    record.snapshot(max_history)
                } else {
                    None
                }
            })
            .min_by(|a, b| {
                let avg_a = average_latency(a);
                let avg_b = average_latency(b);
                avg_a
                    .partial_cmp(&avg_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

fn average_latency(status: &ForeignRuntimeStatus) -> f64 {
    if status.kernels.is_empty() {
        f64::INFINITY
    } else {
        let sum: f64 = status
            .kernels
            .iter()
            .map(|kernel| kernel.average_latency_ms)
            .sum();
        sum / status.kernels.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_language_variants() {
        assert_eq!(ForeignLanguage::parse("Go"), ForeignLanguage::Go);
        assert_eq!(ForeignLanguage::parse("golang"), ForeignLanguage::Go);
        assert_eq!(ForeignLanguage::parse("JuLiA"), ForeignLanguage::Julia);
        assert_eq!(
            ForeignLanguage::parse("zig"),
            ForeignLanguage::Other("zig".into())
        );
    }

    #[test]
    fn register_and_snapshot() {
        let registry = ForeignRegistry::new(8);
        let descriptor = ForeignRuntimeDescriptor::new(
            "go-runtime",
            ForeignLanguage::Go,
            "1.22",
            vec!["tensor.add".into(), "tensor.matmul".into()],
        );
        assert!(registry.register_runtime(descriptor.clone()));
        assert!(registry.record_latency(
            "go-runtime",
            "tensor.add",
            Duration::from_millis(2),
            true
        ));
        assert!(registry.record_latency(
            "go-runtime",
            "tensor.add",
            Duration::from_millis(4),
            false
        ));
        assert!(registry.record_latency(
            "go-runtime",
            "tensor.matmul",
            Duration::from_millis(3),
            true
        ));
        let snapshot = registry.snapshot();
        assert_eq!(snapshot.len(), 1);
        let status = &snapshot[0];
        assert_eq!(status.descriptor.id, "go-runtime");
        assert_eq!(status.kernels.len(), 2);
        let add_stats = status
            .kernels
            .iter()
            .find(|kernel| kernel.operation == "tensor.add")
            .unwrap();
        assert_eq!(add_stats.invocations, 2);
        assert_eq!(add_stats.errors, 1);
        assert!(add_stats.average_latency_ms > 0.0);
    }

    #[test]
    fn best_runtime_selection() {
        let registry = ForeignRegistry::new(8);
        let go_desc = ForeignRuntimeDescriptor::new(
            "go-runtime",
            ForeignLanguage::Go,
            "1.22",
            vec!["tensor.add".into()],
        );
        let jl_desc = ForeignRuntimeDescriptor::new(
            "julia-runtime",
            ForeignLanguage::Julia,
            "1.11",
            vec!["tensor.add".into()],
        );
        assert!(registry.register_runtime(go_desc));
        assert!(registry.register_runtime(jl_desc));
        assert!(registry.record_latency(
            "go-runtime",
            "tensor.add",
            Duration::from_millis(5),
            true
        ));
        assert!(registry.record_latency(
            "julia-runtime",
            "tensor.add",
            Duration::from_millis(2),
            true
        ));
        let best = registry.best_runtime_for("tensor.add").unwrap();
        assert_eq!(best.descriptor.language, ForeignLanguage::Julia);
    }
}
