// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Optional device-choice ranker loaded from `soft_ranker.json`.
//!
//! This store is intentionally separate from the soft-rule bandit store. A
//! legacy `soft_weights.json` is accepted only when its schema is recognisably
//! a ranker document.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;

use crate::backend::device_caps::BackendKind;
use crate::backend::unison_heuristics::Choice;

#[derive(Debug, Clone)]
pub struct SoftRuleLearner {
    weights: HashMap<String, f32>,
    bias: f32,
    temperature: f32,
    beam: usize,
}

#[derive(Debug, Clone)]
pub struct SoftContext {
    rows: u32,
    cols: u32,
    k: u32,
    backend: BackendKind,
    subgroup: bool,
}

#[derive(Debug, Deserialize)]
struct StoredWeights {
    #[serde(default)]
    weights: HashMap<String, f32>,
    #[serde(default)]
    bias: f32,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_beam")]
    beam: usize,
}

/// Overrides the device-choice ranker store path.
pub const SOFT_RANKER_PATH_ENV: &str = "SPIRALTORCH_SOFT_RANKER_PATH";
const MAX_SOFT_RANKER_BYTES: u64 = 16 * 1024 * 1024;
const MAX_SOFT_RANKER_WEIGHTS: usize = 100_000;
const MAX_SOFT_RANKER_KEY_BYTES: usize = 4 * 1024;
const MAX_SOFT_RANKER_BEAM: usize = 65_536;

const fn default_temperature() -> f32 {
    1.0
}

const fn default_beam() -> usize {
    1
}

impl SoftRuleLearner {
    /// Loads the configured ranker, logging invalid persisted state and failing closed.
    pub fn maybe_load() -> Option<Self> {
        match Self::try_load() {
            Ok(learner) => learner,
            Err(error) => {
                eprintln!("[heur] {error} (ignoring learned ranker)");
                None
            }
        }
    }

    /// Strictly loads the configured ranker.
    ///
    /// If no new ranker file exists, a legacy `soft_weights.json` file is read
    /// only when it contains the old ranker schema. Bandit stores are ignored.
    pub fn try_load() -> Result<Option<Self>, SoftRankerStoreError> {
        if let Some(path) = std::env::var_os(SOFT_RANKER_PATH_ENV).filter(|value| !value.is_empty())
        {
            return Self::try_load_from(path);
        }

        let primary = default_store_path();
        if let Some(learner) = Self::try_load_from(&primary)? {
            return Ok(Some(learner));
        }
        match load_stored_weights(&legacy_store_path()) {
            Ok(Some(stored)) => Ok(Some(Self::from_stored(stored))),
            Ok(None) | Err(SoftRankerStoreError::UnrecognizedSchema { .. }) => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Strictly loads a ranker from an explicit path.
    pub fn try_load_from(path: impl AsRef<Path>) -> Result<Option<Self>, SoftRankerStoreError> {
        load_stored_weights(path.as_ref()).map(|stored| stored.map(Self::from_stored))
    }

    fn from_stored(stored: StoredWeights) -> Self {
        Self {
            weights: stored.weights,
            bias: stored.bias,
            temperature: stored.temperature,
            beam: stored.beam,
        }
    }

    pub fn learned_bonus(&mut self, ctx: &SoftContext, choice: &Choice) -> f32 {
        let mut dot = f64::from(self.bias);
        for (name, value) in ctx.features(choice) {
            if let Some(weight) = self.weights.get(name) {
                dot += f64::from(*weight) * f64::from(value);
            }
        }
        (dot / f64::from(self.temperature)).tanh() as f32
    }

    pub fn rank(
        &mut self,
        ctx: &SoftContext,
        candidates: &[(&'static str, Choice, f32)],
    ) -> (usize, Vec<f32>) {
        let mut scored = Vec::with_capacity(candidates.len());
        for (_, choice, base) in candidates {
            let bonus = self.learned_bonus(ctx, choice);
            let score = *base + bonus;
            scored.push(if score.is_finite() {
                score
            } else {
                f32::NEG_INFINITY
            });
        }
        if scored.is_empty() {
            return (0, scored);
        }

        let mut indices: Vec<usize> = (0..scored.len()).collect();
        indices.sort_by(|&a, &b| scored[b].total_cmp(&scored[a]));
        if self.beam < indices.len() {
            indices.truncate(self.beam);
        }
        let best = *indices
            .iter()
            .max_by(|&&lhs, &&rhs| scored[lhs].total_cmp(&scored[rhs]))
            .unwrap_or(&0);
        (best, scored)
    }
}

impl SoftContext {
    pub fn new(rows: u32, cols: u32, k: u32, backend: BackendKind, subgroup: bool) -> Self {
        Self {
            rows,
            cols,
            k,
            backend,
            subgroup,
        }
    }

    fn features(&self, choice: &Choice) -> Vec<(&'static str, f32)> {
        let rows = (self.rows.max(1) as f32).ln_1p();
        let cols = (self.cols.max(1) as f32).ln_1p();
        let k = (self.k.max(1) as f32).ln_1p();
        let wg = choice.wg as f32 / 512.0;
        let kl = choice.kl as f32 / 32.0;
        let tile = choice.tile as f32 / 4_096.0;
        let ctile = choice.ctile as f32 / 4_096.0;
        let mut feats = vec![
            ("bias", 1.0),
            ("rows_ln", rows),
            ("cols_ln", cols),
            ("k_ln", k),
            ("wg_ratio", wg),
            ("kl_ratio", kl),
            ("tile_ratio", tile),
            ("ctile_ratio", ctile),
            ("mk_kind", choice.mk as f32 / 2.0),
            ("mkd_kind", choice.mkd as f32 / 5.0),
            ("two_stage", if choice.use_2ce { 1.0 } else { 0.0 }),
            ("ch_norm", choice.ch as f32 / 8_192.0),
        ];
        feats.push(match self.backend {
            BackendKind::Wgpu => ("backend_wgpu", 1.0),
            BackendKind::Mps => ("backend_mps", 1.0),
            BackendKind::Cuda => ("backend_cuda", 1.0),
            BackendKind::Hip => ("backend_hip", 1.0),
            BackendKind::Cpu => ("backend_cpu", 1.0),
        });
        feats.push(("subgroup", if self.subgroup { 1.0 } else { 0.0 }));
        feats
    }
}

/// Returns the device-choice ranker's namespaced default path.
pub fn default_store_path() -> PathBuf {
    if let Some(home) = dirs::home_dir() {
        home.join(".spiraltorch").join("soft_ranker.json")
    } else {
        PathBuf::from("soft_ranker.json")
    }
}

fn legacy_store_path() -> PathBuf {
    if let Some(home) = dirs::home_dir() {
        home.join(".spiraltorch").join("soft_weights.json")
    } else {
        PathBuf::from("soft_weights.json")
    }
}

fn load_stored_weights(path: &Path) -> Result<Option<StoredWeights>, SoftRankerStoreError> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(source) if source.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(SoftRankerStoreError::Io {
                operation: "open",
                path: path.to_path_buf(),
                source,
            });
        }
    };
    let metadata = file.metadata().map_err(|source| SoftRankerStoreError::Io {
        operation: "inspect",
        path: path.to_path_buf(),
        source,
    })?;
    if metadata.len() > MAX_SOFT_RANKER_BYTES {
        return Err(SoftRankerStoreError::TooLarge {
            path: path.to_path_buf(),
            bytes: metadata.len(),
            max: MAX_SOFT_RANKER_BYTES,
        });
    }

    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_SOFT_RANKER_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|source| SoftRankerStoreError::Io {
            operation: "read",
            path: path.to_path_buf(),
            source,
        })?;
    if bytes.len() as u64 > MAX_SOFT_RANKER_BYTES {
        return Err(SoftRankerStoreError::TooLarge {
            path: path.to_path_buf(),
            bytes: bytes.len() as u64,
            max: MAX_SOFT_RANKER_BYTES,
        });
    }

    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|source| SoftRankerStoreError::Json {
            path: path.to_path_buf(),
            source,
        })?;
    let recognized = value.as_object().is_some_and(|object| {
        ["weights", "bias", "temperature", "beam"]
            .iter()
            .any(|field| object.contains_key(*field))
    });
    if !recognized {
        return Err(SoftRankerStoreError::UnrecognizedSchema {
            path: path.to_path_buf(),
        });
    }

    let stored: StoredWeights =
        serde_json::from_value(value).map_err(|source| SoftRankerStoreError::Json {
            path: path.to_path_buf(),
            source,
        })?;
    validate_stored_weights(path, &stored)?;
    Ok(Some(stored))
}

fn validate_stored_weights(
    path: &Path,
    stored: &StoredWeights,
) -> Result<(), SoftRankerStoreError> {
    if stored.weights.len() > MAX_SOFT_RANKER_WEIGHTS {
        return Err(SoftRankerStoreError::TooManyWeights {
            path: path.to_path_buf(),
            count: stored.weights.len(),
            max: MAX_SOFT_RANKER_WEIGHTS,
        });
    }
    for (name, value) in &stored.weights {
        if name.trim().is_empty() || name.len() > MAX_SOFT_RANKER_KEY_BYTES {
            return Err(SoftRankerStoreError::InvalidWeightName {
                path: path.to_path_buf(),
                name: name.clone(),
            });
        }
        if !value.is_finite() {
            return Err(SoftRankerStoreError::NonFiniteValue {
                path: path.to_path_buf(),
                name: name.clone(),
                value: *value,
            });
        }
    }
    for (name, value) in [("bias", stored.bias), ("temperature", stored.temperature)] {
        if !value.is_finite() {
            return Err(SoftRankerStoreError::NonFiniteValue {
                path: path.to_path_buf(),
                name: name.to_string(),
                value,
            });
        }
    }
    if stored.temperature <= 0.0 {
        return Err(SoftRankerStoreError::InvalidTemperature {
            path: path.to_path_buf(),
            value: stored.temperature,
        });
    }
    if stored.beam == 0 || stored.beam > MAX_SOFT_RANKER_BEAM {
        return Err(SoftRankerStoreError::InvalidBeam {
            path: path.to_path_buf(),
            value: stored.beam,
            max: MAX_SOFT_RANKER_BEAM,
        });
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum SoftRankerStoreError {
    #[error("failed to {operation} soft-ranker store `{path}`")]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to decode soft-ranker JSON `{path}`")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("soft-ranker store `{path}` is {bytes} bytes; maximum is {max}")]
    TooLarge { path: PathBuf, bytes: u64, max: u64 },
    #[error("file `{path}` does not contain the soft-ranker schema")]
    UnrecognizedSchema { path: PathBuf },
    #[error("soft-ranker store `{path}` has {count} weights; maximum is {max}")]
    TooManyWeights {
        path: PathBuf,
        count: usize,
        max: usize,
    },
    #[error("soft-ranker store `{path}` has invalid weight name `{name}`")]
    InvalidWeightName { path: PathBuf, name: String },
    #[error("soft-ranker value `{name}` in `{path}` is non-finite: {value}")]
    NonFiniteValue {
        path: PathBuf,
        name: String,
        value: f32,
    },
    #[error("soft-ranker temperature in `{path}` must be finite and positive, got {value}")]
    InvalidTemperature { path: PathBuf, value: f32 },
    #[error("soft-ranker beam in `{path}` must be in 1..={max}, got {value}")]
    InvalidBeam {
        path: PathBuf,
        value: usize,
        max: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learned_bonus_changes_with_weights() {
        let mut learner = SoftRuleLearner {
            weights: HashMap::from([
                ("wg_ratio".to_string(), 0.5),
                ("backend_wgpu".to_string(), 0.2),
            ]),
            bias: 0.1,
            temperature: 1.0,
            beam: 2,
        };
        let ctx = SoftContext::new(1_024, 65_536, 256, BackendKind::Wgpu, true);
        let choice = Choice {
            use_2ce: true,
            wg: 256,
            kl: 32,
            ch: 8_192,
            mk: 2,
            mkd: 4,
            tile: 2_048,
            ctile: 1_024,
            subgroup: true,
            fft_tile: 2_048,
            fft_radix: 4,
            fft_segments: 1,
            latency_window: None,
        };
        let bonus = learner.learned_bonus(&ctx, &choice);
        assert!(bonus > 0.0);
    }

    #[test]
    fn rank_selects_highest_score() {
        let mut learner = SoftRuleLearner {
            weights: HashMap::new(),
            bias: 0.0,
            temperature: 1.0,
            beam: 2,
        };
        let ctx = SoftContext::new(512, 16_384, 64, BackendKind::Cuda, true);
        let candidates = vec![
            (
                "a",
                Choice {
                    use_2ce: false,
                    wg: 128,
                    kl: 8,
                    ch: 0,
                    mk: 0,
                    mkd: 3,
                    tile: 256,
                    ctile: 256,
                    subgroup: true,
                    fft_tile: 256,
                    fft_radix: 2,
                    fft_segments: 1,
                    latency_window: None,
                },
                0.1,
            ),
            (
                "b",
                Choice {
                    use_2ce: true,
                    wg: 256,
                    kl: 16,
                    ch: 0,
                    mk: 1,
                    mkd: 2,
                    tile: 512,
                    ctile: 256,
                    subgroup: true,
                    fft_tile: 512,
                    fft_radix: 2,
                    fft_segments: 1,
                    latency_window: None,
                },
                0.8,
            ),
        ];
        let (idx, scores) = learner.rank(&ctx, &candidates);
        assert_eq!(idx, 1);
        assert!(scores[1] > scores[0]);
    }

    #[test]
    fn rank_demotes_non_finite_base_scores() {
        let mut learner = SoftRuleLearner {
            weights: HashMap::new(),
            bias: 0.0,
            temperature: 1.0,
            beam: 2,
        };
        let ctx = SoftContext::new(512, 16_384, 64, BackendKind::Cuda, true);
        let choice = Choice {
            use_2ce: false,
            wg: 128,
            kl: 8,
            ch: 0,
            mk: 0,
            mkd: 3,
            tile: 256,
            ctile: 256,
            subgroup: true,
            fft_tile: 256,
            fft_radix: 2,
            fft_segments: 1,
            latency_window: None,
        };
        let candidates = vec![("invalid", choice, f32::NAN), ("finite", choice, 0.25)];

        let (index, scores) = learner.rank(&ctx, &candidates);

        assert_eq!(index, 1);
        assert_eq!(scores[0], f32::NEG_INFINITY);
        assert!(scores[1].is_finite());
    }

    #[test]
    fn explicit_load_accepts_the_legacy_ranker_schema() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("ranker.json");
        std::fs::write(
            &path,
            br#"{
                "weights": {"wg_ratio": 0.5},
                "bias": 0.1,
                "temperature": 0.75,
                "beam": 3
            }"#,
        )
        .expect("write ranker store");

        let learner = SoftRuleLearner::try_load_from(path)
            .expect("valid ranker store")
            .expect("ranker exists");

        assert_eq!(learner.weights.get("wg_ratio"), Some(&0.5));
        assert_eq!(learner.bias, 0.1);
        assert_eq!(learner.temperature, 0.75);
        assert_eq!(learner.beam, 3);
    }

    #[test]
    fn explicit_load_rejects_the_bandit_schema() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("ranker.json");
        std::fs::write(&path, br#"{"rule_beta": {}, "base_coef": {}}"#)
            .expect("write bandit store");

        let error =
            SoftRuleLearner::try_load_from(path).expect_err("bandit data is not ranker data");

        assert!(matches!(
            error,
            SoftRankerStoreError::UnrecognizedSchema { .. }
        ));
    }

    #[test]
    fn explicit_load_rejects_invalid_ranker_controls() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("ranker.json");
        std::fs::write(&path, br#"{"weights": {}, "temperature": 0.0, "beam": 1}"#)
            .expect("write invalid ranker store");

        let error = SoftRuleLearner::try_load_from(path).expect_err("zero temperature must fail");

        assert!(matches!(
            error,
            SoftRankerStoreError::InvalidTemperature { .. }
        ));
    }

    #[test]
    fn default_path_is_namespaced_away_from_bandit_weights() {
        assert_eq!(
            default_store_path()
                .file_name()
                .and_then(|name| name.to_str()),
            Some("soft_ranker.json")
        );
        assert_ne!(default_store_path(), legacy_store_path());
    }
}
