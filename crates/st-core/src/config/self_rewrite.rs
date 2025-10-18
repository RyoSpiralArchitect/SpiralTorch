// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::config::layered::{ConfigLayering, LayeredConfig};
use serde::Deserialize;
use std::env;

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct SelfRewriteCfg {
    pub score_thresh: f32,
    pub min_samples: usize,
    pub cooldown_sec: u64,
}

impl Default for SelfRewriteCfg {
    fn default() -> Self {
        Self {
            score_thresh: 0.02,
            min_samples: 30,
            cooldown_sec: 300,
        }
    }
}

pub fn read_cfg() -> SelfRewriteCfg {
    let mut cfg = load_layered().unwrap_or_default();
    if let Some(value) = env::var("SPIRAL_SELF_REWRITE_THRESH")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        cfg.score_thresh = value;
    }
    if let Some(value) = env::var("SPIRAL_SELF_REWRITE_MIN_SAMPLES")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        cfg.min_samples = value;
    }
    if let Some(value) = env::var("SPIRAL_SELF_REWRITE_COOLDOWN_SEC")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        cfg.cooldown_sec = value;
    }
    cfg
}

fn load_layered() -> Option<SelfRewriteCfg> {
    let layering = ConfigLayering::discover();
    match LayeredConfig::load(layering) {
        Ok(config) => config
            .section::<SelfRewriteCfg>(&["desire", "self_rewrite"])
            .ok()
            .flatten(),
        Err(err) => {
            eprintln!("[config] failed to load layered Desire config: {err}");
            None
        }
    }
}
