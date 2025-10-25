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

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;
    use std::fs;
    use tempfile::NamedTempFile;

    #[derive(Serialize)]
    struct Params {
        x: u32,
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
    fn load_store_handles_bad_json() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();
        fs::write(path, "not json").unwrap();
        let store = load_store(path);
        assert!(store.is_empty());
    }
}
