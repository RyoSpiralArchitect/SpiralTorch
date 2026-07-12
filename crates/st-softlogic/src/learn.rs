// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "learn_store")]

//! Durable shared state for soft-rule bandits and online coefficients.
//!
//! [`SoftWeightStore`] provides strict, bounded reads and atomic locked writes.
//! The top-level [`load`] and [`save`] functions retain the historical facade,
//! while [`try_load`], [`try_save`], and [`update_store`] expose full failures.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Overrides the default soft-weight store path when set to a non-empty value.
pub const SOFT_WEIGHTS_PATH_ENV: &str = "SPIRALTORCH_SOFT_WEIGHTS_PATH";
/// Maximum accepted encoded store size.
pub const MAX_LEARN_STORE_BYTES: u64 = 16 * 1024 * 1024;
/// Maximum number of rule and coefficient entries in one store.
pub const MAX_LEARN_STORE_ENTRIES: usize = 100_000;
/// Maximum UTF-8 byte length of a rule or coefficient name.
pub const MAX_LEARN_STORE_KEY_BYTES: usize = 4 * 1024;

const BANDIT_COUNT_RESCALE_THRESHOLD: f32 = 1_000_000.0;
const MIN_BANDIT_COUNT: f32 = 1.0e-6;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct BetaStat {
    pub alpha: f32,
    pub beta: f32,
}

impl BetaStat {
    fn is_valid(&self) -> bool {
        self.alpha.is_finite()
            && self.alpha > 0.0
            && self.beta.is_finite()
            && self.beta > 0.0
            && (f64::from(self.alpha) + f64::from(self.beta)).is_finite()
    }

    fn prepare_for_observation(&mut self) {
        if !self.is_valid() {
            *self = Self::default();
            return;
        }

        let largest = self.alpha.max(self.beta);
        if largest >= BANDIT_COUNT_RESCALE_THRESHOLD {
            let scale = BANDIT_COUNT_RESCALE_THRESHOLD / largest;
            self.alpha = (self.alpha * scale).max(MIN_BANDIT_COUNT);
            self.beta = (self.beta * scale).max(MIN_BANDIT_COUNT);
        }
    }
}

impl Default for BetaStat {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct SoftWeights {
    pub rule_beta: HashMap<String, BetaStat>,
    pub base_coef: HashMap<String, f32>,
}

impl SoftWeights {
    /// Verifies that the store can be used and persisted without poisoning scores.
    pub fn validate(&self) -> Result<(), SoftWeightValidationError> {
        let entry_count = self.rule_beta.len().saturating_add(self.base_coef.len());
        if entry_count > MAX_LEARN_STORE_ENTRIES {
            return Err(SoftWeightValidationError::TooManyEntries {
                count: entry_count,
                max: MAX_LEARN_STORE_ENTRIES,
            });
        }

        for (rule, stat) in &self.rule_beta {
            validate_key("rule", rule)?;
            if !stat.is_valid() {
                return Err(SoftWeightValidationError::InvalidBandit {
                    rule: rule.clone(),
                    alpha: stat.alpha,
                    beta: stat.beta,
                });
            }
        }
        for (name, value) in &self.base_coef {
            validate_key("coefficient", name)?;
            if !value.is_finite() {
                return Err(SoftWeightValidationError::InvalidCoefficient {
                    name: name.clone(),
                    value: *value,
                });
            }
        }
        Ok(())
    }
}

fn validate_key(kind: &'static str, key: &str) -> Result<(), SoftWeightValidationError> {
    if key.trim().is_empty() {
        return Err(SoftWeightValidationError::EmptyKey { kind });
    }
    if key.len() > MAX_LEARN_STORE_KEY_BYTES {
        return Err(SoftWeightValidationError::KeyTooLong {
            kind,
            key: key.to_string(),
            bytes: key.len(),
            max: MAX_LEARN_STORE_KEY_BYTES,
        });
    }
    Ok(())
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum SoftWeightValidationError {
    #[error("soft-weight store has {count} entries; maximum is {max}")]
    TooManyEntries { count: usize, max: usize },
    #[error("soft-weight {kind} name cannot be empty")]
    EmptyKey { kind: &'static str },
    #[error("soft-weight {kind} name `{key}` has {bytes} bytes; maximum is {max}")]
    KeyTooLong {
        kind: &'static str,
        key: String,
        bytes: usize,
        max: usize,
    },
    #[error("bandit `{rule}` has invalid beta counts alpha={alpha}, beta={beta}")]
    InvalidBandit { rule: String, alpha: f32, beta: f32 },
    #[error("coefficient `{name}` has non-finite value {value}")]
    InvalidCoefficient { name: String, value: f32 },
}

#[derive(Debug, Error)]
pub enum LearnStoreError {
    #[error("soft-weight store path `{0}` does not name a file")]
    InvalidPath(PathBuf),
    #[error("failed to {operation} soft-weight store `{path}`")]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to {operation} soft-weight JSON `{path}`")]
    Json {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("soft-weight store `{path}` is {bytes} bytes; maximum is {max}")]
    TooLarge { path: PathBuf, bytes: u64, max: u64 },
    #[error(transparent)]
    InvalidWeights(#[from] SoftWeightValidationError),
}

impl LearnStoreError {
    fn io(operation: &'static str, path: &Path, source: io::Error) -> Self {
        Self::Io {
            operation,
            path: path.to_path_buf(),
            source,
        }
    }

    fn json(operation: &'static str, path: &Path, source: serde_json::Error) -> Self {
        Self::Json {
            operation,
            path: path.to_path_buf(),
            source,
        }
    }
}

/// A file-backed soft-weight store with atomic replacement and locked updates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SoftWeightStore {
    path: PathBuf,
}

impl SoftWeightStore {
    /// Creates a store at an explicit path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Returns the JSON store path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Loads and validates the store. A missing file is a new empty store.
    pub fn load(&self) -> Result<SoftWeights, LearnStoreError> {
        self.validate_path()?;
        self.load_unlocked()
    }

    /// Atomically replaces the store after validating it.
    pub fn save(&self, weights: &SoftWeights) -> Result<(), LearnStoreError> {
        let _lock = self.acquire_exclusive_lock()?;
        self.save_unlocked(weights)
    }

    /// Applies one locked read-modify-write transaction.
    pub fn update<R>(
        &self,
        update: impl FnOnce(&mut SoftWeights) -> R,
    ) -> Result<R, LearnStoreError> {
        let _lock = self.acquire_exclusive_lock()?;
        let mut weights = self.load_unlocked()?;
        let result = update(&mut weights);
        self.save_unlocked(&weights)?;
        Ok(result)
    }

    fn validate_path(&self) -> Result<(), LearnStoreError> {
        if self.path.file_name().is_none() {
            return Err(LearnStoreError::InvalidPath(self.path.clone()));
        }
        Ok(())
    }

    fn load_unlocked(&self) -> Result<SoftWeights, LearnStoreError> {
        let file = match File::open(&self.path) {
            Ok(file) => file,
            Err(source) if source.kind() == io::ErrorKind::NotFound => {
                return Ok(SoftWeights::default());
            }
            Err(source) => return Err(LearnStoreError::io("open", &self.path, source)),
        };

        let metadata = file
            .metadata()
            .map_err(|source| LearnStoreError::io("inspect", &self.path, source))?;
        if metadata.len() > MAX_LEARN_STORE_BYTES {
            return Err(LearnStoreError::TooLarge {
                path: self.path.clone(),
                bytes: metadata.len(),
                max: MAX_LEARN_STORE_BYTES,
            });
        }

        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        file.take(MAX_LEARN_STORE_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|source| LearnStoreError::io("read", &self.path, source))?;
        if bytes.len() as u64 > MAX_LEARN_STORE_BYTES {
            return Err(LearnStoreError::TooLarge {
                path: self.path.clone(),
                bytes: bytes.len() as u64,
                max: MAX_LEARN_STORE_BYTES,
            });
        }

        let weights: SoftWeights = serde_json::from_slice(&bytes)
            .map_err(|source| LearnStoreError::json("decode", &self.path, source))?;
        weights.validate()?;
        Ok(weights)
    }

    fn save_unlocked(&self, weights: &SoftWeights) -> Result<(), LearnStoreError> {
        weights.validate()?;
        let encoded = serde_json::to_vec_pretty(weights)
            .map_err(|source| LearnStoreError::json("encode", &self.path, source))?;
        if encoded.len() as u64 > MAX_LEARN_STORE_BYTES {
            return Err(LearnStoreError::TooLarge {
                path: self.path.clone(),
                bytes: encoded.len() as u64,
                max: MAX_LEARN_STORE_BYTES,
            });
        }

        let parent = parent_dir(&self.path);
        fs::create_dir_all(parent)
            .map_err(|source| LearnStoreError::io("create parent for", &self.path, source))?;
        let mut temporary = tempfile::Builder::new()
            .prefix(".soft-weights-")
            .suffix(".tmp")
            .tempfile_in(parent)
            .map_err(|source| {
                LearnStoreError::io("create temporary file for", &self.path, source)
            })?;
        temporary.write_all(&encoded).map_err(|source| {
            LearnStoreError::io("write temporary file for", &self.path, source)
        })?;
        temporary
            .as_file_mut()
            .sync_all()
            .map_err(|source| LearnStoreError::io("sync temporary file for", &self.path, source))?;
        temporary
            .persist(&self.path)
            .map_err(|error| LearnStoreError::io("replace", &self.path, error.error))?;
        Ok(())
    }

    fn acquire_exclusive_lock(&self) -> Result<File, LearnStoreError> {
        self.validate_path()?;
        let parent = parent_dir(&self.path);
        fs::create_dir_all(parent)
            .map_err(|source| LearnStoreError::io("create parent for", &self.path, source))?;
        let lock_path = lock_path(&self.path);
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|source| LearnStoreError::io("open lock for", &self.path, source))?;
        lock.lock()
            .map_err(|source| LearnStoreError::io("lock", &self.path, source))?;
        Ok(lock)
    }
}

impl Default for SoftWeightStore {
    fn default() -> Self {
        Self::new(default_store_path())
    }
}

fn parent_dir(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn lock_path(path: &Path) -> PathBuf {
    let mut lock_path: OsString = path.as_os_str().to_owned();
    lock_path.push(".lock");
    PathBuf::from(lock_path)
}

/// Resolves the compatible default store location.
pub fn default_store_path() -> PathBuf {
    if let Some(path) = std::env::var_os(SOFT_WEIGHTS_PATH_ENV).filter(|value| !value.is_empty()) {
        return PathBuf::from(path);
    }

    std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .or_else(|| std::env::var_os("USERPROFILE").filter(|value| !value.is_empty()))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".spiraltorch")
        .join("soft_weights.json")
}

/// Strictly loads the default store and reports corruption or I/O failures.
pub fn try_load() -> Result<SoftWeights, LearnStoreError> {
    SoftWeightStore::default().load()
}

/// Strictly loads a store from an explicit path.
pub fn try_load_from(path: impl Into<PathBuf>) -> Result<SoftWeights, LearnStoreError> {
    SoftWeightStore::new(path).load()
}

/// Loads the default store, preserving the historical fail-open behavior.
///
/// Use [`try_load`] when a caller can surface diagnostics.
pub fn load() -> SoftWeights {
    try_load().unwrap_or_default()
}

/// Strictly and atomically saves the default store.
pub fn try_save(weights: &SoftWeights) -> Result<(), LearnStoreError> {
    SoftWeightStore::default().save(weights)
}

/// Strictly and atomically saves a store at an explicit path.
pub fn try_save_to(path: impl Into<PathBuf>, weights: &SoftWeights) -> Result<(), LearnStoreError> {
    SoftWeightStore::new(path).save(weights)
}

/// Atomically saves the default store through the compatible I/O error facade.
pub fn save(weights: &SoftWeights) -> io::Result<()> {
    try_save(weights).map_err(io::Error::other)
}

/// Applies one locked transaction to the default store.
pub fn update_store<R>(update: impl FnOnce(&mut SoftWeights) -> R) -> Result<R, LearnStoreError> {
    SoftWeightStore::default().update(update)
}

pub fn update_bandit(weights: &mut SoftWeights, winner_rules: &[&str], loser_rules: &[&str]) {
    for rule in winner_rules {
        observe_rule(weights, rule, true);
    }
    for rule in loser_rules {
        observe_rule(weights, rule, false);
    }
}

fn observe_rule(weights: &mut SoftWeights, rule: &str, won: bool) {
    let rule = rule.trim();
    if rule.is_empty() {
        return;
    }
    let stat = weights.rule_beta.entry(rule.to_string()).or_default();
    stat.prepare_for_observation();
    if won {
        stat.alpha += 1.0;
    } else {
        stat.beta += 1.0;
    }
}

pub fn weight_from_bandit(weights: &SoftWeights, rule: &str) -> f32 {
    let Some(stat) = weights.rule_beta.get(rule.trim()) else {
        return 0.5;
    };
    if !stat.is_valid() {
        return 0.5;
    }
    let alpha = f64::from(stat.alpha);
    let probability = alpha / (alpha + f64::from(stat.beta));
    probability.clamp(0.0, 1.0) as f32
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum LearnUpdateError {
    #[error("learning parameter `{name}` has invalid value {value}")]
    InvalidParameter { name: &'static str, value: f32 },
    #[error("feature `{name}` has non-finite value {value}")]
    InvalidFeature { name: String, value: f32 },
    #[error("coefficient `{name}` has non-finite value {value}")]
    InvalidCoefficient { name: String, value: f32 },
    #[error("coefficient update for `{name}` overflowed")]
    ArithmeticOverflow { name: String },
}

fn stable_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Applies a finite, transactional online logistic update.
pub fn try_online_logistic_update(
    coefs: &mut HashMap<String, f32>,
    feats: &HashMap<String, f32>,
    y: f32,
    eta: f32,
    l2: f32,
) -> Result<(), LearnUpdateError> {
    if !y.is_finite() || !(0.0..=1.0).contains(&y) {
        return Err(LearnUpdateError::InvalidParameter {
            name: "y",
            value: y,
        });
    }
    if !eta.is_finite() || eta < 0.0 {
        return Err(LearnUpdateError::InvalidParameter {
            name: "eta",
            value: eta,
        });
    }
    if !l2.is_finite() || l2 < 0.0 {
        return Err(LearnUpdateError::InvalidParameter {
            name: "l2",
            value: l2,
        });
    }

    let mut feature_names = feats.keys().collect::<Vec<_>>();
    feature_names.sort();
    let mut dot = 0.0f64;
    for name in &feature_names {
        let value = feats[*name];
        if !value.is_finite() {
            return Err(LearnUpdateError::InvalidFeature {
                name: (*name).clone(),
                value,
            });
        }
        let coefficient = *coefs.get(*name).unwrap_or(&0.0);
        if !coefficient.is_finite() {
            return Err(LearnUpdateError::InvalidCoefficient {
                name: (*name).clone(),
                value: coefficient,
            });
        }
        dot += f64::from(coefficient) * f64::from(value);
    }
    if !dot.is_finite() {
        return Err(LearnUpdateError::ArithmeticOverflow {
            name: "dot_product".to_string(),
        });
    }

    let prediction = stable_sigmoid(dot);
    let mut updates = Vec::with_capacity(feature_names.len());
    for name in feature_names {
        let value = feats[name];
        let coefficient = *coefs.get(name).unwrap_or(&0.0);
        let next = f64::from(coefficient)
            + f64::from(eta)
                * ((f64::from(y) - prediction) * f64::from(value)
                    - f64::from(l2) * f64::from(coefficient));
        if !next.is_finite() || next.abs() > f64::from(f32::MAX) {
            return Err(LearnUpdateError::ArithmeticOverflow { name: name.clone() });
        }
        updates.push((name.clone(), next as f32));
    }
    for (name, value) in updates {
        coefs.insert(name, value);
    }
    Ok(())
}

/// Applies the compatible fail-closed online update.
///
/// Invalid input leaves `coefs` unchanged. Use [`try_online_logistic_update`]
/// when diagnostics are required.
pub fn online_logistic_update(
    coefs: &mut HashMap<String, f32>,
    feats: &HashMap<String, f32>,
    y: f32,
    eta: f32,
    l2: f32,
) {
    let _ = try_online_logistic_update(coefs, feats, y, eta, l2);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn beta_stat_default_is_a_neutral_prior() {
        assert_eq!(
            BetaStat::default(),
            BetaStat {
                alpha: 1.0,
                beta: 1.0
            }
        );
    }

    #[test]
    fn update_bandit_tracks_wins_and_losses() {
        let mut weights = SoftWeights::default();
        update_bandit(&mut weights, &["rule_a", "rule_b"], &["rule_c"]);
        update_bandit(&mut weights, &["rule_a"], &["rule_b", "rule_c"]);

        let rule_a = weights.rule_beta.get("rule_a").expect("rule_a exists");
        assert!(rule_a.alpha > rule_a.beta);
        let rule_c = weights.rule_beta.get("rule_c").expect("rule_c exists");
        assert!(rule_c.beta > rule_c.alpha);
    }

    #[test]
    fn bandit_counts_rescale_before_they_lose_f32_resolution() {
        let mut weights = SoftWeights::default();
        weights.rule_beta.insert(
            "rule".to_string(),
            BetaStat {
                alpha: 4_000_000.0,
                beta: 2_000_000.0,
            },
        );

        update_bandit(&mut weights, &["rule"], &[]);

        let stat = weights.rule_beta.get("rule").expect("rule exists");
        assert!(stat.alpha <= BANDIT_COUNT_RESCALE_THRESHOLD + 1.0);
        assert!((stat.beta - 500_000.0).abs() < 1.0);
    }

    #[test]
    fn invalid_bandit_stat_uses_a_neutral_weight() {
        let mut weights = SoftWeights::default();
        weights.rule_beta.insert(
            "rule".to_string(),
            BetaStat {
                alpha: f32::NAN,
                beta: 1.0,
            },
        );
        assert_eq!(weight_from_bandit(&weights, "rule"), 0.5);
    }

    #[test]
    fn bandit_lookup_uses_the_same_trimmed_identity_as_updates() {
        let mut weights = SoftWeights::default();
        update_bandit(&mut weights, &["  rule  "], &[]);

        assert!(weight_from_bandit(&weights, "rule") > 0.5);
        assert_eq!(
            weight_from_bandit(&weights, "  rule  "),
            weight_from_bandit(&weights, "rule")
        );
    }

    #[test]
    fn logistic_update_adjusts_weights() {
        let mut coefs = HashMap::new();
        let feats = HashMap::from([("bias".to_string(), 1.0), ("signal".to_string(), 2.0)]);

        try_online_logistic_update(&mut coefs, &feats, 1.0, 0.5, 0.1)
            .expect("valid positive update");
        assert!(coefs.get("signal").expect("signal exists") > &0.0);
        assert!(coefs.get("bias").expect("bias exists") > &0.0);

        try_online_logistic_update(&mut coefs, &feats, 0.0, 0.5, 0.1)
            .expect("valid negative update");
        assert!(coefs.get("signal").expect("signal exists") < &1.0);
        assert!(coefs.get("bias").expect("bias exists") < &1.0);
    }

    #[test]
    fn invalid_logistic_update_is_transactional() {
        let original = HashMap::from([("signal".to_string(), 0.25)]);
        let mut coefs = original.clone();
        let feats = HashMap::from([("signal".to_string(), f32::NAN)]);

        let error = try_online_logistic_update(&mut coefs, &feats, 1.0, 0.5, 0.1)
            .expect_err("NaN feature must fail");

        assert!(matches!(error, LearnUpdateError::InvalidFeature { .. }));
        assert_eq!(coefs, original);
    }

    #[test]
    fn store_round_trips_and_atomically_replaces_existing_data() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("weights.json");
        let store = SoftWeightStore::new(&path);
        let mut first = SoftWeights::default();
        update_bandit(&mut first, &["first"], &[]);
        store.save(&first).expect("save first store");
        assert_eq!(store.load().expect("load first store"), first);

        let mut second = SoftWeights::default();
        update_bandit(&mut second, &[], &["second"]);
        store.save(&second).expect("replace store");
        assert_eq!(store.load().expect("load replacement"), second);

        let temporary_files = fs::read_dir(directory.path())
            .expect("list store directory")
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().contains(".tmp"))
            .count();
        assert_eq!(temporary_files, 0);
    }

    #[test]
    fn strict_load_surfaces_corrupt_json() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("weights.json");
        fs::write(&path, b"not-json").expect("write corrupt store");

        let error = SoftWeightStore::new(path)
            .load()
            .expect_err("corrupt JSON must fail");
        assert!(matches!(error, LearnStoreError::Json { .. }));
    }

    #[test]
    fn strict_load_rejects_oversized_files_before_decoding() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("weights.json");
        let file = File::create(&path).expect("create oversized store");
        file.set_len(MAX_LEARN_STORE_BYTES + 1)
            .expect("extend oversized store");

        let error = SoftWeightStore::new(path)
            .load()
            .expect_err("oversized store must fail");
        assert!(matches!(error, LearnStoreError::TooLarge { .. }));
    }

    #[test]
    fn save_rejects_non_finite_weights_without_touching_existing_data() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("weights.json");
        let store = SoftWeightStore::new(path);
        let mut valid = SoftWeights::default();
        update_bandit(&mut valid, &["stable"], &[]);
        store.save(&valid).expect("save valid store");

        let mut invalid = valid.clone();
        invalid.base_coef.insert("bad".to_string(), f32::NAN);
        let error = store
            .save(&invalid)
            .expect_err("non-finite coefficient must fail");

        assert!(matches!(error, LearnStoreError::InvalidWeights(_)));
        assert_eq!(store.load().expect("load preserved store"), valid);
    }

    #[test]
    fn concurrent_updates_do_not_lose_observations() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let store = Arc::new(SoftWeightStore::new(directory.path().join("weights.json")));
        let mut workers = Vec::new();
        for _ in 0..4 {
            let store = Arc::clone(&store);
            workers.push(std::thread::spawn(move || {
                for _ in 0..20 {
                    store
                        .update(|weights| update_bandit(weights, &["shared"], &[]))
                        .expect("locked update");
                }
            }));
        }
        for worker in workers {
            worker.join().expect("worker completes");
        }

        let weights = store.load().expect("load final store");
        let stat = weights.rule_beta.get("shared").expect("shared rule exists");
        assert_eq!(stat.alpha, 81.0);
        assert_eq!(stat.beta, 1.0);
    }

    #[test]
    fn invalid_transaction_preserves_existing_data() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let store = SoftWeightStore::new(directory.path().join("weights.json"));
        let mut valid = SoftWeights::default();
        update_bandit(&mut valid, &["stable"], &[]);
        store.save(&valid).expect("save valid store");

        let error = store
            .update(|weights| {
                weights.base_coef.insert("bad".to_string(), f32::NAN);
            })
            .expect_err("invalid transaction must fail");

        assert!(matches!(error, LearnStoreError::InvalidWeights(_)));
        assert_eq!(store.load().expect("load preserved store"), valid);
    }

    #[test]
    fn updater_panic_releases_lock_without_persisting_mutation() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let store = SoftWeightStore::new(directory.path().join("weights.json"));
        let mut initial = SoftWeights::default();
        update_bandit(&mut initial, &["stable"], &[]);
        store.save(&initial).expect("save initial store");

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = store.update(|weights| {
                update_bandit(weights, &["discarded"], &[]);
                panic!("updater failed");
            });
        }));
        assert!(panic.is_err());
        assert_eq!(store.load().expect("load unchanged store"), initial);

        store
            .update(|weights| update_bandit(weights, &["recovered"], &[]))
            .expect("subsequent update acquires lock");
        let recovered = store.load().expect("load recovered store");
        assert!(recovered.rule_beta.contains_key("recovered"));
        assert!(!recovered.rule_beta.contains_key("discarded"));
    }
}
