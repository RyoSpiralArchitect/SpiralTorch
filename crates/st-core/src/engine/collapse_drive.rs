// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// ============================================================================
//  SpiralReality Proprietary
// Copyright (c) 2025 SpiralReality. All Rights Reserved.
//
//  NOTICE: This file contains confidential and proprietary information of
//  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
//  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
//  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
//
//  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
//  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
//  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
//  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
//  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
//  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
//  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ============================================================================

use crate::telemetry::psi::{PsiAutomationHint, PsiReading};

#[derive(Clone, Debug)]
pub struct CollapseConfig {
    pub enabled: bool,
    pub hi: f32,
    pub lo: f32,
    pub hysteresis: f32,
    pub warmup_steps: u64,
    pub cooldown_steps: u32,
    pub collapse_scale: f32,
    pub collapse_max_norm: Option<f32>,
    pub collapse_lr_decay: Option<f32>,
    pub bloom_lr_mul: f32,
    pub trend_alpha: f32,
    pub collapse_trend_threshold: f32,
    pub bloom_trend_threshold: f32,
    pub baseline_alpha: f32,
    pub collapse_deviation_threshold: f32,
    pub bloom_deviation_threshold: f32,
}

impl CollapseConfig {
    pub fn automated(hint: PsiAutomationHint) -> Self {
        let depth = hint.depth().max(1) as f32;
        let hi = (1.4_f32 + depth.ln_1p() * 0.6_f32).clamp(1.8_f32, 4.5_f32);
        let lo = (0.18_f32 + 0.4_f32 / depth.sqrt()).clamp(0.2_f32, 0.6_f32);
        let hysteresis = 0.08_f32 + 0.02_f32 * depth.ln_1p();
        let warmup_steps = (hint.depth() as u64 * 2).clamp(20, 240);
        let cooldown_steps = (depth / 8.0_f32).round().clamp(2.0_f32, 12.0_f32) as u32;
        let collapse_scale = (0.18_f32 + 0.9_f32 / depth.sqrt()).clamp(0.1_f32, 0.4_f32);
        let bloom_lr_mul = (1.05_f32 + hint.band_focus / depth).clamp(1.05_f32, 1.6_f32);
        let trend_alpha = (0.18_f32 + 1.2_f32 / depth.sqrt()).clamp(0.2_f32, 0.65_f32);
        let collapse_trend_threshold =
            (0.08_f32 + hint.band_focus / (depth + 4.0_f32)).clamp(0.06_f32, 0.25_f32);
        let bloom_trend_threshold = -(0.06_f32 + hint.drift_weight.clamp(0.05_f32, 0.6_f32));
        let baseline_alpha = (0.04_f32 + 1.5_f32 / (depth + 6.0_f32)).clamp(0.05_f32, 0.3_f32);
        let collapse_deviation_threshold =
            (0.12_f32 + 0.35_f32 / depth.sqrt() + hint.band_focus * 0.02_f32)
                .clamp(0.12_f32, 0.35_f32);
        let bloom_deviation_threshold =
            (0.1_f32 + hint.drift_weight * 0.3_f32).clamp(0.1_f32, 0.35_f32);
        Self {
            enabled: true,
            hi,
            lo,
            hysteresis,
            warmup_steps,
            cooldown_steps,
            collapse_scale,
            collapse_max_norm: Some(1.0),
            collapse_lr_decay: Some(0.12),
            bloom_lr_mul,
            trend_alpha,
            collapse_trend_threshold,
            bloom_trend_threshold,
            baseline_alpha,
            collapse_deviation_threshold,
            bloom_deviation_threshold,
        }
    }
}

impl Default for CollapseConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            hi: 2.0,
            lo: 0.3,
            hysteresis: 0.1,
            warmup_steps: 50,
            cooldown_steps: 10,
            collapse_scale: 0.2,
            collapse_max_norm: Some(1.0),
            collapse_lr_decay: Some(0.1),
            bloom_lr_mul: 1.2,
            trend_alpha: 0.0,
            collapse_trend_threshold: f32::INFINITY,
            bloom_trend_threshold: f32::NEG_INFINITY,
            baseline_alpha: 0.0,
            collapse_deviation_threshold: f32::INFINITY,
            bloom_deviation_threshold: f32::INFINITY,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriveState {
    Warmup,
    Stable,
    Collapse,
    Bloom,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DriveCmd {
    None,
    Collapse {
        grad_scale: f32,
        max_norm: Option<f32>,
        lr_decay: Option<f32>,
    },
    Bloom {
        lr_mul: f32,
    },
}

#[derive(Clone, Debug)]
pub struct CollapseDrive {
    cfg: CollapseConfig,
    state: DriveState,
    cooldown: u32,
    last_step: u64,
    prev_total: Option<f32>,
    trend: f32,
    baseline: Option<f32>,
}

impl CollapseDrive {
    pub fn new(cfg: CollapseConfig) -> Self {
        Self {
            cfg,
            state: DriveState::Warmup,
            cooldown: 0,
            last_step: 0,
            prev_total: None,
            trend: 0.0,
            baseline: None,
        }
    }

    pub fn enabled(&self) -> bool {
        self.cfg.enabled
    }

    pub fn update(&mut self, reading: &PsiReading) -> DriveCmd {
        self.last_step = reading.step;
        if !self.cfg.enabled {
            return DriveCmd::None;
        }
        if self.last_step < self.cfg.warmup_steps {
            self.state = DriveState::Warmup;
            return DriveCmd::None;
        }
        if self.state == DriveState::Warmup {
            self.state = DriveState::Stable;
        }
        if self.cooldown > 0 {
            self.cooldown -= 1;
            return DriveCmd::None;
        }

        let psi = reading.total;
        let mut collapse_due_to_trend = false;
        let mut bloom_due_to_trend = false;

        if let Some(prev) = self.prev_total {
            let delta = psi - prev;
            let trend_alpha = self.cfg.trend_alpha.clamp(0.0, 1.0);
            if trend_alpha > 0.0 {
                self.trend = trend_alpha * delta + (1.0 - trend_alpha) * self.trend;
            } else {
                self.trend = delta;
            }

            let baseline_alpha = self.cfg.baseline_alpha.clamp(0.0, 1.0);
            if baseline_alpha > 0.0 {
                let base = self.baseline.unwrap_or(prev);
                let updated = (1.0 - baseline_alpha) * base + baseline_alpha * psi;
                let deviation = psi - updated;
                self.baseline = Some(updated);
                if self.cfg.collapse_deviation_threshold.is_finite()
                    && deviation >= self.cfg.collapse_deviation_threshold
                {
                    collapse_due_to_trend = true;
                }
                if self.cfg.bloom_deviation_threshold.is_finite()
                    && deviation <= -self.cfg.bloom_deviation_threshold
                {
                    bloom_due_to_trend = true;
                }
            } else {
                self.baseline = Some(psi);
            }

            if self.cfg.collapse_trend_threshold.is_finite()
                && self.trend >= self.cfg.collapse_trend_threshold
            {
                collapse_due_to_trend = true;
            }
            if self.cfg.bloom_trend_threshold.is_finite()
                && self.trend <= self.cfg.bloom_trend_threshold
            {
                bloom_due_to_trend = true;
            }
        } else {
            self.baseline = Some(psi);
            self.trend = 0.0;
        }
        self.prev_total = Some(psi);

        match self.state {
            DriveState::Warmup | DriveState::Stable => {
                if psi >= self.cfg.hi || collapse_due_to_trend {
                    self.state = DriveState::Collapse;
                    self.cooldown = self.cfg.cooldown_steps;
                    DriveCmd::Collapse {
                        grad_scale: self.cfg.collapse_scale,
                        max_norm: self.cfg.collapse_max_norm,
                        lr_decay: self.cfg.collapse_lr_decay,
                    }
                } else if psi <= self.cfg.lo || bloom_due_to_trend {
                    self.state = DriveState::Bloom;
                    self.cooldown = self.cfg.cooldown_steps;
                    DriveCmd::Bloom {
                        lr_mul: self.cfg.bloom_lr_mul,
                    }
                } else {
                    DriveCmd::None
                }
            }
            DriveState::Collapse => {
                if psi <= self.cfg.hi - self.cfg.hysteresis {
                    self.state = DriveState::Stable;
                }
                DriveCmd::None
            }
            DriveState::Bloom => {
                if psi >= self.cfg.lo + self.cfg.hysteresis {
                    self.state = DriveState::Stable;
                }
                DriveCmd::None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn reading(step: u64, total: f32) -> PsiReading {
        PsiReading {
            total,
            breakdown: HashMap::new(),
            step,
        }
    }

    #[test]
    fn transitions_and_cooldown() {
        let mut cfg = CollapseConfig::default();
        cfg.enabled = true;
        cfg.hi = 2.0;
        cfg.lo = 0.5;
        cfg.warmup_steps = 0;
        cfg.cooldown_steps = 2;
        let mut drive = CollapseDrive::new(cfg);
        assert!(matches!(drive.update(&reading(1, 0.6)), DriveCmd::None));
        assert!(matches!(
            drive.update(&reading(2, 2.2)),
            DriveCmd::Collapse { .. }
        ));
        assert!(matches!(drive.update(&reading(3, 1.8)), DriveCmd::None));
        assert!(matches!(drive.update(&reading(4, 1.8)), DriveCmd::None));
        assert!(matches!(drive.update(&reading(5, 1.8)), DriveCmd::None));
        assert!(matches!(
            drive.update(&reading(6, 0.3)),
            DriveCmd::Bloom { .. }
        ));
    }

    #[test]
    fn collapse_triggers_on_trend_and_deviation() {
        let mut cfg = CollapseConfig::default();
        cfg.enabled = true;
        cfg.hi = 9.0;
        cfg.lo = -2.0;
        cfg.warmup_steps = 0;
        cfg.cooldown_steps = 0;
        cfg.trend_alpha = 1.0;
        cfg.collapse_trend_threshold = 0.5;
        cfg.baseline_alpha = 0.5;
        cfg.collapse_deviation_threshold = 0.4;
        cfg.bloom_trend_threshold = f32::NEG_INFINITY;
        cfg.bloom_deviation_threshold = f32::INFINITY;

        let mut drive = CollapseDrive::new(cfg);
        assert!(matches!(drive.update(&reading(1, 0.0)), DriveCmd::None));
        assert!(matches!(drive.update(&reading(2, 0.3)), DriveCmd::None));
        assert!(matches!(
            drive.update(&reading(3, 1.1)),
            DriveCmd::Collapse { .. }
        ));
    }

    #[test]
    fn bloom_triggers_on_trend_and_deviation() {
        let mut cfg = CollapseConfig::default();
        cfg.enabled = true;
        cfg.hi = 9.0;
        cfg.lo = -2.0;
        cfg.warmup_steps = 0;
        cfg.cooldown_steps = 0;
        cfg.trend_alpha = 1.0;
        cfg.collapse_trend_threshold = f32::INFINITY;
        cfg.baseline_alpha = 0.5;
        cfg.bloom_trend_threshold = -0.4;
        cfg.bloom_deviation_threshold = 0.3;
        cfg.collapse_deviation_threshold = f32::INFINITY;

        let mut drive = CollapseDrive::new(cfg);
        assert!(matches!(drive.update(&reading(1, 1.0)), DriveCmd::None));
        assert!(matches!(drive.update(&reading(2, 0.9)), DriveCmd::None));
        assert!(matches!(
            drive.update(&reading(3, 0.2)),
            DriveCmd::Bloom { .. }
        ));
    }
}
