// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#[derive(Clone, Copy, Debug)]
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
    let d = SelfRewriteCfg::default();
    let t = std::env::var("SPIRAL_SELF_REWRITE_THRESH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d.score_thresh);
    let m = std::env::var("SPIRAL_SELF_REWRITE_MIN_SAMPLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d.min_samples);
    let c = std::env::var("SPIRAL_SELF_REWRITE_COOLDOWN_SEC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d.cooldown_sec);
    SelfRewriteCfg {
        score_thresh: t,
        min_samples: m,
        cooldown_sec: c,
    }
}
