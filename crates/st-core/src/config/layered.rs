// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::telemetry::hub::{self, ConfigDiffEvent, ConfigLayer};
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

/// Collection of configuration layers that are merged in order.
#[derive(Clone, Debug, Default)]
pub struct ConfigLayering {
    pub base: Option<PathBuf>,
    pub site: Option<PathBuf>,
    pub run: Option<PathBuf>,
}

impl ConfigLayering {
    /// Discovers configuration files using the environment and standard
    /// SpiralTorch paths. Files that do not exist are ignored.
    pub fn discover() -> Self {
        let root = std::env::var("SPIRAL_CONFIG_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| default_root());

        let base = std::env::var("SPIRAL_CONFIG_BASE")
            .map(PathBuf::from)
            .ok()
            .or_else(|| Some(root.join("base.toml")))
            .and_then(existing_path);

        let site = std::env::var("SPIRAL_CONFIG_SITE")
            .map(PathBuf::from)
            .ok()
            .or_else(|| Some(root.join("site.toml")))
            .and_then(existing_path);

        let run = std::env::var("SPIRAL_CONFIG_RUN")
            .map(PathBuf::from)
            .ok()
            .or_else(|| Some(root.join("run.json")))
            .and_then(existing_path);

        ConfigLayering { base, site, run }
    }

    /// Overrides the base layer path.
    pub fn with_base<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.base = Some(path.into());
        self
    }

    /// Overrides the site layer path.
    pub fn with_site<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.site = Some(path.into());
        self
    }

    /// Overrides the run layer path.
    pub fn with_run<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.run = Some(path.into());
        self
    }
}

fn existing_path(path: PathBuf) -> Option<PathBuf> {
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

fn default_root() -> PathBuf {
    if let Some(home) = dirs::home_dir() {
        let config_dir = home.join(".spiraltorch").join("config");
        if config_dir.exists() {
            config_dir
        } else {
            home.join(".spiraltorch")
        }
    } else {
        PathBuf::from(".")
    }
}

/// Result of merging layered configuration files.
#[derive(Clone, Debug)]
pub struct LayeredConfig {
    layering: ConfigLayering,
    value: Value,
    events: Vec<ConfigDiffEvent>,
}

impl LayeredConfig {
    /// Loads the configured layers, merging base → site → run.
    pub fn load(layering: ConfigLayering) -> Result<Self, LayeredConfigError> {
        let mut value = Value::Object(Default::default());
        let mut events = Vec::new();

        if let Some(base_path) = layering.base.as_ref() {
            if let Some(layer) = load_toml(base_path)? {
                apply_layer(&mut value, &layer, ConfigLayer::Base, &mut events);
            }
        }
        if let Some(site_path) = layering.site.as_ref() {
            if let Some(layer) = load_toml(site_path)? {
                apply_layer(&mut value, &layer, ConfigLayer::Site, &mut events);
            }
        }
        if let Some(run_path) = layering.run.as_ref() {
            if let Some(layer) = load_json(run_path)? {
                apply_layer(&mut value, &layer, ConfigLayer::Run, &mut events);
            }
        }

        hub::record_config_events(&events);

        Ok(LayeredConfig {
            layering,
            value,
            events,
        })
    }

    /// Returns the merged configuration as a `serde_json::Value`.
    pub fn value(&self) -> &Value {
        &self.value
    }

    /// Returns the layering metadata used for this configuration.
    pub fn layering(&self) -> &ConfigLayering {
        &self.layering
    }

    /// Returns the diff events emitted while applying the layers.
    pub fn events(&self) -> &[ConfigDiffEvent] {
        &self.events
    }

    /// Extracts a typed view of a nested configuration section. The path is
    /// expressed as a slice of keys that will be traversed in order.
    pub fn section<T>(&self, path: &[&str]) -> Result<Option<T>, serde_json::Error>
    where
        T: DeserializeOwned,
    {
        let mut node = &self.value;
        for key in path {
            match node {
                Value::Object(map) => match map.get(*key) {
                    Some(value) => node = value,
                    None => return Ok(None),
                },
                _ => return Ok(None),
            }
        }
        serde_json::from_value(node.clone()).map(Some)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LayeredConfigError {
    #[error("failed to read {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse TOML {path:?}: {source}")]
    Toml {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error("failed to parse JSON {path:?}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

fn load_toml(path: &Path) -> Result<Option<Value>, LayeredConfigError> {
    if !path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(path).map_err(|source| LayeredConfigError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let value: toml::Value = toml::from_str(&text).map_err(|source| LayeredConfigError::Toml {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::to_value(value)
        .map(Some)
        .map_err(|source| LayeredConfigError::Json {
            path: path.to_path_buf(),
            source,
        })
}

fn load_json(path: &Path) -> Result<Option<Value>, LayeredConfigError> {
    if !path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(path).map_err(|source| LayeredConfigError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&text)
        .map(Some)
        .map_err(|source| LayeredConfigError::Json {
            path: path.to_path_buf(),
            source,
        })
}

fn apply_layer(
    dest: &mut Value,
    layer: &Value,
    kind: ConfigLayer,
    events: &mut Vec<ConfigDiffEvent>,
) {
    let before = dest.clone();
    merge(dest, layer);
    diff(&before, dest, &mut Vec::new(), kind, events);
}

fn merge(dest: &mut Value, src: &Value) {
    match (dest, src) {
        (Value::Object(dest_map), Value::Object(src_map)) => {
            for (key, value) in src_map {
                match dest_map.get_mut(key) {
                    Some(existing) => merge(existing, value),
                    None => {
                        dest_map.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        (dest_slot, src_value) => {
            *dest_slot = src_value.clone();
        }
    }
}

fn diff(
    before: &Value,
    after: &Value,
    path: &mut Vec<String>,
    layer: ConfigLayer,
    out: &mut Vec<ConfigDiffEvent>,
) {
    if before == after {
        return;
    }

    match (before, after) {
        (Value::Object(before_map), Value::Object(after_map)) => {
            let mut keys: Vec<&String> = before_map.keys().collect();
            for key in after_map.keys() {
                if !keys.contains(&key) {
                    keys.push(key);
                }
            }
            keys.sort();
            keys.dedup();
            for key in keys {
                path.push(key.clone());
                let before_child = before_map.get(key).unwrap_or(&Value::Null);
                let after_child = after_map.get(key).unwrap_or(&Value::Null);
                diff(before_child, after_child, path, layer, out);
                path.pop();
            }
        }
        _ => {
            let field = path.join(".");
            let previous = if before.is_null() {
                None
            } else {
                Some(before.clone())
            };
            let current = if after.is_null() {
                None
            } else {
                Some(after.clone())
            };
            out.push(ConfigDiffEvent {
                layer,
                path: field,
                previous,
                current,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir() -> PathBuf {
        let mut dir = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        dir.push(format!("st_layered_{nanos}"));
        dir
    }

    #[test]
    fn merges_layers_and_reports_diffs() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();

        let base = root.join("base.toml");
        fs::write(
            &base,
            r#"
            [desire.self_rewrite]
            score_thresh = 0.02
            min_samples = 16
            "#,
        )
        .unwrap();

        let site = root.join("site.toml");
        fs::write(
            &site,
            r#"
            [desire.self_rewrite]
            min_samples = 24
            cooldown_sec = 120
            "#,
        )
        .unwrap();

        let run = root.join("run.json");
        fs::write(&run, r#"{"desire":{"self_rewrite":{"score_thresh":0.05}}}"#).unwrap();

        std::env::set_var("SPIRAL_CONFIG_ROOT", &root);
        hub::record_config_events(&[]);

        let layering = ConfigLayering::discover();
        let stacked = LayeredConfig::load(layering.clone()).unwrap();

        let desire: Value = stacked
            .section::<Value>(&["desire", "self_rewrite"])
            .unwrap()
            .unwrap();
        assert_eq!(desire["score_thresh"], Value::from(0.05));
        assert_eq!(desire["min_samples"], Value::from(24));
        assert_eq!(desire["cooldown_sec"], Value::from(120));

        let events = stacked.events();
        assert!(events.iter().any(|event| {
            event.layer == ConfigLayer::Run
                && event.path == "desire.self_rewrite.score_thresh"
                && event.current == Some(Value::from(0.05))
        }));
        assert!(events.iter().any(|event| {
            event.layer == ConfigLayer::Site
                && event.path == "desire.self_rewrite.cooldown_sec"
                && event.current == Some(Value::from(120))
        }));

        std::env::remove_var("SPIRAL_CONFIG_ROOT");
        let _ = fs::remove_file(base);
        let _ = fs::remove_file(site);
        let _ = fs::remove_file(run);
        let _ = fs::remove_dir(root);
    }
}
