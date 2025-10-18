// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::automation::{DesireAutomatedStep, DesireRewriteTrigger};
use super::desire::DesireSolution;
use crate::PureResult;
use serde::{Deserialize, Serialize};
use st_tensor::TensorError;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Lines, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DesireLogRecord {
    pub ordinal: u64,
    pub timestamp_ms: u128,
    pub solution: DesireSolution,
    pub trigger: Option<DesireRewriteTrigger>,
}

pub struct DesireLogbook {
    path: PathBuf,
    writer: BufWriter<File>,
    flush_every: usize,
    pending: usize,
    ordinal: u64,
}

impl DesireLogbook {
    pub fn new<P: AsRef<Path>>(path: P) -> PureResult<Self> {
        Self::with_flush_every(path, 1)
    }

    pub fn with_flush_every<P: AsRef<Path>>(path: P, flush_every: usize) -> PureResult<Self> {
        let flush_every = flush_every.max(1);
        let path_ref = path.as_ref();
        if let Some(parent) = path_ref.parent() {
            fs::create_dir_all(parent).map_err(io_error)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(path_ref)
            .map_err(io_error)?;
        let ordinal = next_ordinal(path_ref)?;
        let writer = BufWriter::new(file);
        Ok(Self {
            path: path_ref.to_path_buf(),
            writer,
            flush_every,
            pending: 0,
            ordinal,
        })
    }

    pub fn record_now(&mut self, step: &DesireAutomatedStep) -> PureResult<()> {
        self.record(step, SystemTime::now())
    }

    pub fn record(&mut self, step: &DesireAutomatedStep, timestamp: SystemTime) -> PureResult<()> {
        let record = DesireLogRecord {
            ordinal: self.ordinal,
            timestamp_ms: timestamp_ms(timestamp),
            solution: step.solution.clone(),
            trigger: step.trigger.clone(),
        };
        serde_json::to_writer(&mut self.writer, &record).map_err(serde_error)?;
        self.writer.write_all(b"\n").map_err(io_error)?;
        self.pending += 1;
        self.ordinal = self.ordinal.saturating_add(1);
        if self.pending >= self.flush_every {
            self.writer.flush().map_err(io_error)?;
            self.pending = 0;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> PureResult<()> {
        self.writer.flush().map_err(io_error)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn ordinal(&self) -> u64 {
        self.ordinal
    }
}

pub struct DesireLogReplay {
    path: PathBuf,
    lines: Lines<BufReader<File>>,
}

impl DesireLogReplay {
    pub fn open<P: AsRef<Path>>(path: P) -> PureResult<Self> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref).map_err(io_error)?;
        let reader = BufReader::new(file);
        Ok(Self {
            path: path_ref.to_path_buf(),
            lines: reader.lines(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Iterator for DesireLogReplay {
    type Item = PureResult<DesireLogRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(line) = self.lines.next() {
            match line {
                Ok(text) => {
                    if text.trim().is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<DesireLogRecord>(&text) {
                        Ok(record) => return Some(Ok(record)),
                        Err(err) => return Some(Err(serde_error(err))),
                    }
                }
                Err(err) => return Some(Err(io_error(err))),
            }
        }
        None
    }
}

fn timestamp_ms(time: SystemTime) -> u128 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis()
}

fn next_ordinal(path: &Path) -> PureResult<u64> {
    if !path.exists() {
        return Ok(0);
    }
    let file = File::open(path).map_err(io_error)?;
    let reader = BufReader::new(file);
    let mut last = None;
    for line in reader.lines() {
        let line = line.map_err(io_error)?;
        if line.trim().is_empty() {
            continue;
        }
        let record: DesireLogRecord = serde_json::from_str(&line).map_err(serde_error)?;
        last = Some(record.ordinal);
    }
    Ok(last.map(|ordinal| ordinal.saturating_add(1)).unwrap_or(0))
}

fn io_error(err: std::io::Error) -> TensorError {
    TensorError::IoError {
        message: err.to_string(),
    }
}

fn serde_error(err: serde_json::Error) -> TensorError {
    TensorError::SerializationError {
        message: err.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::automation::DesireAutomation;
    use super::super::desire::{constant, warmup, DesireLagrangian, DesirePhase, DesireWeights};
    use super::super::geometry::{
        ConceptHint, RepressionField, SemanticBridge, SparseKernel, SymbolGeometry,
    };
    use super::super::temperature::TemperatureController;
    use super::*;
    use st_core::config::self_rewrite::SelfRewriteCfg;
    use std::collections::HashSet;
    use std::time::{Duration, Instant, SystemTime};
    use tempfile::tempdir;

    fn build_geometry() -> SymbolGeometry {
        let syn = SparseKernel::from_rows(
            vec![vec![(0, 0.6), (1, 0.4)], vec![(0, 0.5), (1, 0.5)]],
            1e-6,
        )
        .unwrap();
        let par = SparseKernel::from_rows(
            vec![vec![(0, 0.7), (1, 0.3)], vec![(0, 0.2), (1, 0.8)]],
            1e-6,
        )
        .unwrap();
        SymbolGeometry::new(syn, par).unwrap()
    }

    fn build_semantics() -> SemanticBridge {
        let log_pi = vec![
            vec![(0, (0.7f32).ln()), (1, (0.3f32).ln())],
            vec![(0, (0.4f32).ln()), (1, (0.6f32).ln())],
        ];
        let row = vec![1.0, 1.0];
        let col = vec![1.0, 1.0];
        let anchors = HashSet::new();
        let concept_kernel =
            SparseKernel::from_rows(vec![vec![(0, 1.0)], vec![(1, 1.0)]], 1e-6).unwrap();
        SemanticBridge::new(log_pi, row, col, anchors, 1e-6, concept_kernel).unwrap()
    }

    fn build_lagrangian() -> DesireLagrangian {
        let geometry = build_geometry();
        let repression = RepressionField::new(vec![0.05, 0.15]).unwrap();
        let semantics = build_semantics();
        let controller = TemperatureController::new(1.0, 0.8, 0.4, 0.4, 1.8);
        DesireLagrangian::new(geometry, repression, semantics, controller)
            .unwrap()
            .with_top_k(Some(2))
            .with_alpha_schedule(warmup(0.0, 0.1, 1))
            .with_beta_schedule(warmup(0.0, 0.05, 2))
            .with_gamma_schedule(constant(0.02))
            .with_lambda_schedule(constant(0.0))
            .with_observation_horizon(Some(1))
            .with_integration_horizon(Some(3))
    }

    #[test]
    fn logbook_writes_records() {
        let lagrangian = build_lagrangian();
        let cfg = SelfRewriteCfg {
            score_thresh: 0.0,
            min_samples: 2,
            cooldown_sec: 0,
        };
        let mut automation = DesireAutomation::new(lagrangian, cfg);
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("desire.ndjson");
        let mut logbook = DesireLogbook::with_flush_every(&path, 1).expect("logbook");
        let logits = vec![2.0, 0.5];
        let concept = ConceptHint::Distribution(vec![0.6, 0.4]);
        let base = Instant::now();
        for step in 0..3 {
            let now = base + Duration::from_secs(step as u64 + 1);
            let event = automation
                .step(&logits, step % 2, &concept, now)
                .expect("automation step");
            logbook.record(&event, SystemTime::now()).expect("record");
        }
        logbook.flush().expect("flush");
        let contents = std::fs::read_to_string(&path).expect("read log");
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 3);
        let record: DesireLogRecord = serde_json::from_str(lines[0]).expect("decode");
        assert_eq!(record.ordinal, 0);
        assert_eq!(record.solution.phase, DesirePhase::Observation);
    }

    #[test]
    fn logbook_tracks_ordinal() {
        let lagrangian = build_lagrangian();
        let cfg = SelfRewriteCfg {
            score_thresh: 0.0,
            min_samples: 2,
            cooldown_sec: 0,
        };
        let mut automation = DesireAutomation::new(lagrangian, cfg);
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("desire.ndjson");
        let mut logbook = DesireLogbook::with_flush_every(&path, 10).expect("logbook");
        let logits = vec![2.0, 0.5];
        let concept = ConceptHint::Distribution(vec![0.6, 0.4]);
        let base = Instant::now();
        for step in 0..5 {
            let now = base + Duration::from_secs(step as u64 + 1);
            let event = automation
                .step(&logits, step % 2, &concept, now)
                .expect("automation step");
            logbook.record(&event, SystemTime::now()).expect("record");
        }
        assert_eq!(logbook.ordinal(), 5);
        logbook.flush().expect("flush");
        let contents = std::fs::read_to_string(&path).expect("read log");
        assert_eq!(contents.lines().count(), 5);
    }

    #[test]
    fn logbook_resumes_existing_file() {
        let lagrangian = build_lagrangian();
        let cfg = SelfRewriteCfg {
            score_thresh: 0.0,
            min_samples: 2,
            cooldown_sec: 0,
        };
        let mut automation = DesireAutomation::new(lagrangian, cfg);
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("desire.ndjson");
        let logits = vec![2.0, 0.5];
        let concept = ConceptHint::Distribution(vec![0.6, 0.4]);
        let base = Instant::now();
        {
            let mut logbook = DesireLogbook::with_flush_every(&path, 1).expect("logbook");
            for step in 0..3 {
                let now = base + Duration::from_secs(step as u64 + 1);
                let event = automation
                    .step(&logits, step % 2, &concept, now)
                    .expect("automation step");
                let timestamp = SystemTime::UNIX_EPOCH + Duration::from_secs(step as u64);
                logbook.record(&event, timestamp).expect("record");
            }
            logbook.flush().expect("flush");
        }
        let mut resumed = DesireLogbook::with_flush_every(&path, 4).expect("resume");
        assert_eq!(resumed.ordinal(), 3);
        let event = automation
            .step(&logits, 1, &concept, base + Duration::from_secs(8))
            .expect("automation step");
        resumed
            .record(&event, SystemTime::UNIX_EPOCH + Duration::from_secs(99))
            .expect("record");
        resumed.flush().expect("flush");
        let records: Vec<DesireLogRecord> = DesireLogReplay::open(&path)
            .expect("open")
            .map(|entry| entry.expect("record"))
            .collect();
        assert_eq!(records.len(), 4);
        assert_eq!(records[3].ordinal, 3);
    }

    #[test]
    fn log_replay_streams_records() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("desire.ndjson");
        let mut logbook = DesireLogbook::with_flush_every(&path, 1).expect("logbook");
        let mut solution = DesireSolution {
            indices: vec![0, 1],
            probabilities: vec![0.6, 0.4],
            logit_offsets: vec![0.0, 0.0],
            temperature: 1.0,
            entropy: 1.0,
            weights: DesireWeights::new(0.1, 0.0, 0.0, 0.0),
            phase: DesirePhase::Observation,
            avoidance: None,
            hypergrad_penalty: 0.1,
            narrative: None,
        };
        for ordinal in 0..3 {
            solution.phase = match ordinal {
                0 => DesirePhase::Observation,
                1 => DesirePhase::Injection,
                _ => DesirePhase::Integration,
            };
            let step = DesireAutomatedStep {
                solution: solution.clone(),
                trigger: None,
            };
            logbook
                .record(&step, SystemTime::UNIX_EPOCH + Duration::from_secs(ordinal))
                .expect("record");
        }
        logbook.flush().expect("flush");
        let mut replay = DesireLogReplay::open(&path).expect("open");
        let mut phases = Vec::new();
        while let Some(record) = replay.next() {
            let record = record.expect("record");
            phases.push(record.solution.phase);
        }
        assert_eq!(
            phases,
            vec![
                DesirePhase::Observation,
                DesirePhase::Injection,
                DesirePhase::Integration
            ]
        );
    }
}
