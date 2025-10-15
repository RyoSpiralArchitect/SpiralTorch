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
}

impl CollapseDrive {
    pub fn new(cfg: CollapseConfig) -> Self {
        Self {
            cfg,
            state: DriveState::Warmup,
            cooldown: 0,
            last_step: 0,
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
        if self.cooldown > 0 {
            self.cooldown -= 1;
            return DriveCmd::None;
        }

        let psi = reading.total;
        match self.state {
            DriveState::Warmup | DriveState::Stable => {
                if psi >= self.cfg.hi {
                    self.state = DriveState::Collapse;
                    self.cooldown = self.cfg.cooldown_steps;
                    DriveCmd::Collapse {
                        grad_scale: self.cfg.collapse_scale,
                        max_norm: self.cfg.collapse_max_norm,
                        lr_decay: self.cfg.collapse_lr_decay,
                    }
                } else if psi <= self.cfg.lo {
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
}
