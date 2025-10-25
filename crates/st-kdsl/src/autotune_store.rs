// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTuneEntry {
    pub updated_unix: u64,
    pub score: f64,
    pub params: Value,
}

pub type AutoTuneStore = BTreeMap<String, AutoTuneEntry>;

fn now_unix() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub fn load_store(path: &Path) -> AutoTuneStore {
    if let Ok(mut f) = File::open(path) {
        let mut buf = String::new();
        if f.read_to_string(&mut buf).is_ok() {
            if let Ok(map) = serde_json::from_str::<AutoTuneStore>(&buf) {
                return map;
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

pub fn record_best<P: Serialize>(
    path: &Path,
    key: &str,
    score: f64,
    params: &P,
) -> io::Result<bool> {
    let mut store = load_store(path);
    let replace = match store.get(key) {
        None => true,
        Some(prev) => score < prev.score,
    };
    if replace {
        store.insert(
            key.to_string(),
            AutoTuneEntry {
                updated_unix: now_unix(),
                score,
                params: serde_json::to_value(params).unwrap_or(Value::Null),
            },
        );
        save_store_atomic(path, &store)?;
        return Ok(true);
    }
    Ok(false)
}

pub fn lookup_best(path: &Path, key: &str) -> Option<AutoTuneEntry> {
    load_store(path).remove(key)
}

/// Deserialize the best parameters into the requested type, falling back to `default`.
pub fn load_best_typed<T: for<'de> Deserialize<'de> + Clone>(
    path: &Path,
    key: &str,
    default: T,
) -> T {
    if let Some(entry) = lookup_best(path, key) {
        if let Ok(value) = serde_json::from_value::<T>(entry.params) {
            return value;
        }
    }
    default
}

/// Returns `true` when persisted autotune data should be applied.
pub fn autotune_enabled() -> bool {
    std::env::var("SPIRALTORCH_AUTOTUNE")
        .ok()
        .map_or(true, |v| v != "0")
}

/// Load the persisted parameters unless autotuning has been disabled.
pub fn load_best_typed_if_enabled<T: for<'de> Deserialize<'de> + Clone>(
    path: &Path,
    key: &str,
    default: T,
) -> T {
    if autotune_enabled() {
        load_best_typed(path, key, default)
    } else {
        default
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache_key::stable_cache_key;
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;
    use std::fs;
    use tempfile::NamedTempFile;

    #[derive(Serialize)]
    struct Params {
        x: u32,
    }

    #[derive(Clone, Deserialize, Serialize, Debug, PartialEq)]
    struct TypedCfg {
        factor: u32,
    }

    #[test]
    fn record_and_lookup_roundtrip() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();
        assert!(record_best(path, "key", 3.0, &Params { x: 1 }).unwrap());
        assert!(!record_best(path, "key", 4.0, &Params { x: 2 }).unwrap());
        assert!(record_best(path, "key", 2.5, &Params { x: 3 }).unwrap());
        let entry = lookup_best(path, "key").expect("entry");
        assert_eq!(entry.score, 2.5);
        assert_eq!(entry.params["x"].as_u64(), Some(3));
    }

    #[test]
    fn load_best_typed_roundtrip() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();
        let mut defs = HashMap::new();
        defs.insert("FOO".into(), "BAR".into());
        let key = stable_cache_key(
            "rustc",
            "wasm32",
            "wgpu",
            "f32",
            vec!["-O3".into()],
            &defs,
            &42u32,
        );
        let default = TypedCfg { factor: 1 };
        assert_eq!(
            load_best_typed(path, &key, default.clone()),
            default,
            "missing entries fall back to default",
        );
        record_best(path, &key, 1.5, &default).unwrap();
        let cfg = load_best_typed(path, &key, TypedCfg { factor: 7 });
        assert_eq!(cfg, default);
    }

    #[test]
    fn load_best_respects_env_toggle() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();
        let key = "env-toggle";
        record_best(path, key, 0.5, &TypedCfg { factor: 11 }).unwrap();
        std::env::set_var("SPIRALTORCH_AUTOTUNE", "0");
        let cfg = load_best_typed_if_enabled(path, key, TypedCfg { factor: 3 });
        assert_eq!(cfg.factor, 3);
        std::env::remove_var("SPIRALTORCH_AUTOTUNE");
        let cfg = load_best_typed_if_enabled(path, key, TypedCfg { factor: 5 });
        assert_eq!(cfg.factor, 11);
    }

    #[test]
    fn load_store_handles_bad_json() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();
        fs::write(path, "not json").unwrap();
        let store = load_store(path);
        assert!(store.is_empty());
    }
}
