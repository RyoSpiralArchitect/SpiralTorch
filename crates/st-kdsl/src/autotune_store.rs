// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTuneEntry {
    pub updated_unix: u64,
    pub score: f64,
    pub params: Value,
    pub context: Value,
    pub features: Vec<Feature>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AutoTuneBucket {
    entries: Vec<AutoTuneEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Feature {
    key: String,
    value: f64,
}

#[derive(Debug, Clone)]
pub struct AutoTuneMatch {
    pub entry: AutoTuneEntry,
    pub distance: f64,
}

pub type AutoTuneStore = BTreeMap<String, AutoTuneBucket>;

const BUCKET_HISTORY_LIMIT: usize = 32;

fn now_unix() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawStore {
    Bucketed(AutoTuneStore),
    Flat(BTreeMap<String, LegacyEntry>),
}

#[derive(Debug, Deserialize)]
struct LegacyEntry {
    updated_unix: u64,
    score: f64,
    params: Value,
}

pub fn load_store(path: &Path) -> AutoTuneStore {
    if let Ok(mut f) = File::open(path) {
        let mut buf = String::new();
        if f.read_to_string(&mut buf).is_ok() {
            if let Ok(store) = serde_json::from_str::<RawStore>(&buf) {
                return match store {
                    RawStore::Bucketed(map) => map,
                    RawStore::Flat(map) => map
                        .into_iter()
                        .map(|(key, legacy)| {
                            (
                                key,
                                AutoTuneBucket {
                                    entries: vec![AutoTuneEntry {
                                        updated_unix: legacy.updated_unix,
                                        score: legacy.score,
                                        params: legacy.params,
                                        context: Value::Null,
                                        features: Vec::new(),
                                    }],
                                },
                            )
                        })
                        .collect(),
                };
            }
        }
    }
    AutoTuneStore::new()
}

fn save_store_atomic(path: &Path, store: &AutoTuneStore) -> io::Result<()> {
    let tmp = path.with_extension("json.tmp");
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).ok();
    }
    let mut f = File::create(&tmp)?;
    f.write_all(serde_json::to_string_pretty(store)?.as_bytes())?;
    f.flush()?;
    fs::rename(tmp, path)?;
    Ok(())
}

fn extract_features(value: &Value) -> Vec<Feature> {
    fn walk(path: &mut String, value: &Value, out: &mut Vec<Feature>) {
        match value {
            Value::Number(num) => {
                if let Some(v) = num.as_f64() {
                    out.push(Feature {
                        key: path.clone(),
                        value: v,
                    });
                }
            }
            Value::Bool(v) => {
                out.push(Feature {
                    key: path.clone(),
                    value: if *v { 1.0 } else { 0.0 },
                });
            }
            Value::Array(values) => {
                for (index, value) in values.iter().enumerate() {
                    let prev_len = path.len();
                    if !path.is_empty() {
                        path.push('.');
                    }
                    path.push_str(&index.to_string());
                    walk(path, value, out);
                    path.truncate(prev_len);
                }
            }
            Value::Object(map) => {
                for (key, value) in map.iter() {
                    let prev_len = path.len();
                    if !path.is_empty() {
                        path.push('.');
                    }
                    path.push_str(key);
                    walk(path, value, out);
                    path.truncate(prev_len);
                }
            }
            Value::Null | Value::String(_) => {}
        }
    }

    let mut features = Vec::new();
    let mut path = String::new();
    walk(&mut path, value, &mut features);
    features.sort_by(|a, b| a.key.cmp(&b.key));
    features
}

fn distance_between(a: &[Feature], b: &[Feature]) -> f64 {
    let mut i = 0;
    let mut j = 0;
    let mut distance = 0.0;
    while i < a.len() || j < b.len() {
        match (a.get(i), b.get(j)) {
            (Some(left), Some(right)) if left.key == right.key => {
                let delta = (left.value - right.value).abs();
                let scale = left.value.abs().max(right.value.abs()).max(1.0);
                distance += delta / scale;
                i += 1;
                j += 1;
            }
            (Some(left), Some(right)) => {
                if left.key < right.key {
                    distance += left.value.abs();
                    i += 1;
                } else {
                    distance += right.value.abs();
                    j += 1;
                }
            }
            (Some(left), None) => {
                distance += left.value.abs();
                i += 1;
            }
            (None, Some(right)) => {
                distance += right.value.abs();
                j += 1;
            }
            (None, None) => break,
        }
    }
    distance
}

impl AutoTuneBucket {
    fn insert_entry(&mut self, entry: AutoTuneEntry) -> bool {
        if let Some(existing) = self
            .entries
            .iter_mut()
            .find(|candidate| candidate.context == entry.context)
        {
            if entry.score < existing.score {
                *existing = entry;
                self.entries
                    .sort_by(|a, b| b.updated_unix.cmp(&a.updated_unix));
                return true;
            } else {
                existing.updated_unix = entry.updated_unix;
                self.entries
                    .sort_by(|a, b| b.updated_unix.cmp(&a.updated_unix));
                return false;
            }
        }

        self.entries.push(entry);
        self.entries
            .sort_by(|a, b| b.updated_unix.cmp(&a.updated_unix));
        if self.entries.len() > BUCKET_HISTORY_LIMIT {
            self.entries.truncate(BUCKET_HISTORY_LIMIT);
        }
        true
    }

    fn matches_for(&self, context: &Value) -> Vec<AutoTuneMatch> {
        let features = extract_features(context);
        self.entries
            .iter()
            .cloned()
            .map(|entry| {
                let distance = distance_between(&features, &entry.features);
                AutoTuneMatch { entry, distance }
            })
            .collect()
    }
}

pub fn record_best<P: Serialize, C: Serialize>(
    path: &Path,
    key: &str,
    context: &C,
    score: f64,
    params: &P,
) -> io::Result<bool> {
    let mut store = load_store(path);
    let params = serde_json::to_value(params).unwrap_or(Value::Null);
    let context_value = serde_json::to_value(context).unwrap_or(Value::Null);
    let entry = AutoTuneEntry {
        updated_unix: now_unix(),
        score,
        params,
        context: context_value.clone(),
        features: extract_features(&context_value),
    };

    let bucket = store
        .entry(key.to_string())
        .or_insert_with(AutoTuneBucket::default);
    let inserted = bucket.insert_entry(entry);
    save_store_atomic(path, &store)?;
    Ok(inserted)
}

pub fn lookup_similar<C: Serialize>(
    path: &Path,
    key: &str,
    context: &C,
    limit: usize,
) -> Vec<AutoTuneMatch> {
    let store = load_store(path);
    let context_value = serde_json::to_value(context).unwrap_or(Value::Null);
    store
        .get(key)
        .map(|bucket| {
            let mut matches = bucket.matches_for(&context_value);
            matches.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| {
                        a.entry
                            .score
                            .partial_cmp(&b.entry.score)
                            .unwrap_or(Ordering::Equal)
                    })
            });
            matches.truncate(limit);
            matches
        })
        .unwrap_or_default()
}

pub fn lookup_best<C: Serialize>(path: &Path, key: &str, context: &C) -> Option<AutoTuneEntry> {
    lookup_similar(path, key, context, 1)
        .into_iter()
        .map(|m| m.entry)
        .next()
}

/// Returns the best typed parameters stored for the given key or the provided
/// default when the store does not contain a matching entry or deserialization
/// fails.
pub fn load_best_typed<T: for<'de> Deserialize<'de> + Clone>(
    path: &Path,
    key: &str,
    default: T,
) -> T {
    load_store(path)
        .get(key)
        .and_then(|bucket| {
            bucket
                .entries
                .iter()
                .min_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal))
                .and_then(|entry| serde_json::from_value::<T>(entry.params.clone()).ok())
        })
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::fs;
    use tempfile::NamedTempFile;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct Params {
        x: u32,
    }

    #[derive(Serialize)]
    struct Ctx {
        dim: u32,
    }

    #[test]
    fn record_and_lookup_roundtrip() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();
        assert!(record_best(path, "key", &Ctx { dim: 16 }, 3.0, &Params { x: 1 }).unwrap());
        assert!(!record_best(path, "key", &Ctx { dim: 16 }, 4.0, &Params { x: 2 }).unwrap());
        assert!(record_best(path, "key", &Ctx { dim: 16 }, 2.5, &Params { x: 3 }).unwrap());
        let entry = lookup_best(path, "key", &Ctx { dim: 16 }).expect("entry");
        assert_eq!(entry.score, 2.5);
        assert_eq!(entry.params["x"].as_u64(), Some(3));
        assert_eq!(entry.features.len(), 1);
    }

    #[test]
    fn load_best_typed_returns_best_params() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();
        let ctx = Ctx { dim: 16 };
        assert!(record_best(path, "key", &ctx, 3.0, &Params { x: 1 }).unwrap());
        assert!(record_best(path, "key", &ctx, 2.0, &Params { x: 42 }).unwrap());

        let loaded = load_best_typed(path, "key", None::<Params>);
        assert_eq!(loaded, Some(Params { x: 42 }));
    }

    #[test]
    fn load_store_handles_bad_json() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();
        fs::write(path, "not json").unwrap();
        let store = load_store(path);
        assert!(store.is_empty());
    }

    #[test]
    fn lookup_prefers_nearest_context() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();
        record_best(path, "key", &Ctx { dim: 16 }, 5.0, &Params { x: 1 }).unwrap();
        record_best(path, "key", &Ctx { dim: 64 }, 2.0, &Params { x: 2 }).unwrap();
        record_best(path, "key", &Ctx { dim: 32 }, 4.0, &Params { x: 3 }).unwrap();

        let matches = lookup_similar(path, "key", &Ctx { dim: 34 }, 2);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].entry.params["x"].as_u64(), Some(3));
        assert!(matches[0].distance < matches[1].distance);
    }

    #[test]
    fn respects_history_limit() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();
        for i in 0..40u32 {
            let ctx = Ctx { dim: i };
            let params = Params { x: i };
            record_best(path, "key", &ctx, i as f64, &params).unwrap();
        }

        let store = load_store(path);
        let bucket = store.get("key").expect("bucket");
        assert!(bucket.entries.len() <= BUCKET_HISTORY_LIMIT);
    }
}
