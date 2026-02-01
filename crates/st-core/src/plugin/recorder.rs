// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Plugin event recorder utilities.
//!
//! The recorder subscribes to the global [`PluginEventBus`] and stores a bounded,
//! JSON-friendly trace that can be written as JSONL or rendered into a simple
//! Mermaid flowchart for quick inspection.

use crate::{PureResult, TensorError};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::events::{EventListener, PluginEvent, PluginEventBus};

#[derive(Clone, Copy, Debug)]
pub struct PluginEventRecorderConfig {
    pub capacity: usize,
}

impl Default for PluginEventRecorderConfig {
    fn default() -> Self {
        Self { capacity: 2048 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data")]
pub enum PluginEventSnapshot {
    SystemInit,
    SystemShutdown,
    PluginLoaded {
        plugin_id: String,
    },
    PluginUnloaded {
        plugin_id: String,
    },
    TensorOp {
        op_name: String,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    },
    EpochStart {
        epoch: usize,
    },
    EpochEnd {
        epoch: usize,
        loss: f32,
    },
    BackendChanged {
        backend: String,
    },
    Telemetry {
        data: HashMap<String, f32>,
    },
    Custom {
        event_type: String,
        data: Option<Value>,
    },
}

impl PluginEventSnapshot {
    fn from_event(event: &PluginEvent) -> Self {
        match event {
            PluginEvent::SystemInit => PluginEventSnapshot::SystemInit,
            PluginEvent::SystemShutdown => PluginEventSnapshot::SystemShutdown,
            PluginEvent::PluginLoaded { plugin_id } => PluginEventSnapshot::PluginLoaded {
                plugin_id: plugin_id.clone(),
            },
            PluginEvent::PluginUnloaded { plugin_id } => PluginEventSnapshot::PluginUnloaded {
                plugin_id: plugin_id.clone(),
            },
            PluginEvent::TensorOp {
                op_name,
                input_shape,
                output_shape,
            } => PluginEventSnapshot::TensorOp {
                op_name: op_name.clone(),
                input_shape: input_shape.clone(),
                output_shape: output_shape.clone(),
            },
            PluginEvent::EpochStart { epoch } => PluginEventSnapshot::EpochStart { epoch: *epoch },
            PluginEvent::EpochEnd { epoch, loss } => PluginEventSnapshot::EpochEnd {
                epoch: *epoch,
                loss: *loss,
            },
            PluginEvent::BackendChanged { backend } => PluginEventSnapshot::BackendChanged {
                backend: backend.clone(),
            },
            PluginEvent::Telemetry { data } => {
                PluginEventSnapshot::Telemetry { data: data.clone() }
            }
            PluginEvent::Custom { event_type, .. } => {
                let data = event.downcast_data::<Value>().cloned();
                PluginEventSnapshot::Custom {
                    event_type: event_type.clone(),
                    data,
                }
            }
        }
    }

    pub fn label(&self) -> String {
        match self {
            PluginEventSnapshot::SystemInit => "SystemInit".to_string(),
            PluginEventSnapshot::SystemShutdown => "SystemShutdown".to_string(),
            PluginEventSnapshot::PluginLoaded { plugin_id } => format!("PluginLoaded: {plugin_id}"),
            PluginEventSnapshot::PluginUnloaded { plugin_id } => {
                format!("PluginUnloaded: {plugin_id}")
            }
            PluginEventSnapshot::TensorOp {
                op_name,
                input_shape,
                output_shape,
            } => format!("TensorOp: {op_name} {input_shape:?}→{output_shape:?}"),
            PluginEventSnapshot::EpochStart { epoch } => format!("EpochStart: {epoch}"),
            PluginEventSnapshot::EpochEnd { epoch, loss } => {
                format!("EpochEnd: {epoch} loss={loss:.6}")
            }
            PluginEventSnapshot::BackendChanged { backend } => format!("BackendChanged: {backend}"),
            PluginEventSnapshot::Telemetry { data } => format!("Telemetry: {} keys", data.len()),
            PluginEventSnapshot::Custom { event_type, .. } => format!("Custom: {event_type}"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PluginEventRecord {
    pub idx: u64,
    pub elapsed_ms: u64,
    pub event: PluginEventSnapshot,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PluginEventTrace {
    pub dropped_events: usize,
    pub events: Vec<PluginEventRecord>,
}

#[derive(Clone)]
pub struct PluginEventRecorder {
    bus: PluginEventBus,
    subscription_id: usize,
    started_at: Instant,
    inner: Arc<Mutex<RecorderState>>,
}

#[derive(Debug)]
struct RecorderState {
    next_idx: u64,
    dropped: usize,
    events: VecDeque<PluginEventRecord>,
    capacity: usize,
}

impl PluginEventRecorder {
    pub fn subscribe(bus: PluginEventBus, config: PluginEventRecorderConfig) -> Self {
        let started_at = Instant::now();
        let capacity = config.capacity.max(1);
        let inner = Arc::new(Mutex::new(RecorderState {
            next_idx: 0,
            dropped: 0,
            events: VecDeque::new(),
            capacity,
        }));

        let inner_clone = Arc::clone(&inner);
        let started_at_clone = started_at;
        let listener: EventListener = Arc::new(move |event: &PluginEvent| {
            let elapsed_ms = started_at_clone
                .elapsed()
                .as_millis()
                .min(u128::from(u64::MAX)) as u64;
            let snapshot = PluginEventSnapshot::from_event(event);
            let mut guard = inner_clone
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let idx = guard.next_idx;
            guard.next_idx = guard.next_idx.wrapping_add(1);
            guard.events.push_back(PluginEventRecord {
                idx,
                elapsed_ms,
                event: snapshot,
            });
            while guard.events.len() > guard.capacity {
                guard.events.pop_front();
                guard.dropped = guard.dropped.saturating_add(1);
            }
        });

        let subscription_id = bus.subscribe("*", listener);
        Self {
            bus,
            subscription_id,
            started_at,
            inner,
        }
    }

    pub fn snapshot(&self) -> PluginEventTrace {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        PluginEventTrace {
            dropped_events: guard.dropped,
            events: guard.events.iter().cloned().collect(),
        }
    }

    pub fn clear(&self) {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.events.clear();
        guard.dropped = 0;
        guard.next_idx = 0;
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.started_at
            .elapsed()
            .as_millis()
            .min(u128::from(u64::MAX)) as u64
    }

    pub fn write_jsonl(&self, path: impl AsRef<Path>) -> PureResult<()> {
        let trace = self.snapshot();
        let file =
            File::create(path.as_ref()).map_err(|err| TensorError::Generic(err.to_string()))?;
        let mut writer = BufWriter::new(file);
        for record in trace.events {
            let json = serde_json::to_string(&record)
                .map_err(|err| TensorError::Generic(err.to_string()))?;
            writer
                .write_all(json.as_bytes())
                .and_then(|_| writer.write_all(b"\n"))
                .map_err(|err| TensorError::Generic(err.to_string()))?;
        }
        Ok(())
    }

    /// Render a simple linear Mermaid flowchart for the recorded events.
    pub fn to_mermaid_flowchart(&self, max_nodes: usize) -> String {
        let trace = self.snapshot();
        let max_nodes = max_nodes.max(1);
        let mut lines = Vec::new();
        lines.push("flowchart TD".to_string());
        for (idx, record) in trace.events.iter().take(max_nodes).enumerate() {
            let node = format!("n{idx}");
            let mut label = record.event.label();
            label = label.replace('"', "\\\"");
            lines.push(format!("  {node}[\"{label}\"]"));
            if idx > 0 {
                lines.push(format!("  n{} --> {node}", idx - 1));
            }
        }
        if trace.events.len() > max_nodes {
            lines.push(format!("  n{} --> n_more", max_nodes - 1));
            lines.push("  n_more[\"… truncated …\"]".to_string());
        }
        lines.join("\n")
    }
}

impl Drop for PluginEventRecorder {
    fn drop(&mut self) {
        let _ = self.bus.unsubscribe("*", self.subscription_id);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PluginEventJsonlWriterConfig {
    /// When `false`, `TensorOp` events are skipped to keep traces compact.
    pub capture_tensor_ops: bool,
}

impl Default for PluginEventJsonlWriterConfig {
    fn default() -> Self {
        Self {
            capture_tensor_ops: false,
        }
    }
}

#[derive(Clone)]
pub struct PluginEventJsonlWriter {
    bus: PluginEventBus,
    subscription_id: usize,
    started_at: Instant,
    inner: Arc<Mutex<JsonlWriterState>>,
}

#[derive(Debug)]
struct JsonlWriterState {
    writer: BufWriter<File>,
    next_idx: u64,
    errored: bool,
}

impl PluginEventJsonlWriter {
    pub fn subscribe(
        bus: PluginEventBus,
        path: impl AsRef<Path>,
        config: PluginEventJsonlWriterConfig,
    ) -> PureResult<Self> {
        let started_at = Instant::now();
        let file =
            File::create(path.as_ref()).map_err(|err| TensorError::Generic(err.to_string()))?;
        let writer = BufWriter::new(file);
        let inner = Arc::new(Mutex::new(JsonlWriterState {
            writer,
            next_idx: 0,
            errored: false,
        }));

        let inner_clone = Arc::clone(&inner);
        let started_at_clone = started_at;
        let listener: EventListener = Arc::new(move |event: &PluginEvent| {
            if !config.capture_tensor_ops && matches!(event, PluginEvent::TensorOp { .. }) {
                return;
            }

            let elapsed_ms = started_at_clone
                .elapsed()
                .as_millis()
                .min(u128::from(u64::MAX)) as u64;
            let snapshot = PluginEventSnapshot::from_event(event);

            let mut guard = inner_clone
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let idx = guard.next_idx;
            guard.next_idx = guard.next_idx.wrapping_add(1);

            let record = PluginEventRecord {
                idx,
                elapsed_ms,
                event: snapshot,
            };

            let err = match serde_json::to_writer(&mut guard.writer, &record) {
                Ok(()) => guard.writer.write_all(b"\n").err().map(|err| err.to_string()),
                Err(err) => Some(err.to_string()),
            };
            if let Some(err) = err {
                if !guard.errored {
                    guard.errored = true;
                    eprintln!("[spiraltorch] plugin event JSONL writer error: {err}");
                }
            }
        });

        let subscription_id = bus.subscribe("*", listener);
        Ok(Self {
            bus,
            subscription_id,
            started_at,
            inner,
        })
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.started_at
            .elapsed()
            .as_millis()
            .min(u128::from(u64::MAX)) as u64
    }

    pub fn flush(&self) -> PureResult<()> {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard
            .writer
            .flush()
            .map_err(|err| TensorError::Generic(err.to_string()))?;
        Ok(())
    }
}

impl Drop for PluginEventJsonlWriter {
    fn drop(&mut self) {
        let _ = self.flush();
        let _ = self.bus.unsubscribe("*", self.subscription_id);
    }
}
