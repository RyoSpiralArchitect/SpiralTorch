// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// ============================================================================
//  SpiralReality Proprietary
//  Copyright (c) 2025 SpiralReality. All Rights Reserved.
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

#[cfg(any(feature = "psi", feature = "psychoid"))]
use once_cell::sync::Lazy;
use std::sync::{OnceLock, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "psi")]
use super::psi::{PsiEvent, PsiReading};
#[cfg(feature = "psychoid")]
use super::psychoid::PsychoidReading;

#[cfg(feature = "psi")]
static LAST_PSI: Lazy<RwLock<Option<PsiReading>>> = Lazy::new(|| RwLock::new(None));

#[cfg(feature = "psi")]
static LAST_PSI_EVENTS: Lazy<RwLock<Vec<PsiEvent>>> = Lazy::new(|| RwLock::new(Vec::new()));

#[cfg(feature = "psi")]
pub fn set_last_psi(reading: &PsiReading) {
    if let Ok(mut guard) = LAST_PSI.write() {
        *guard = Some(reading.clone());
    }
}

#[cfg(feature = "psi")]
pub fn get_last_psi() -> Option<PsiReading> {
    LAST_PSI
        .read()
        .ok()
        .and_then(|guard| guard.as_ref().cloned())
}

#[cfg(feature = "psi")]
pub fn set_last_psi_events(events: &[PsiEvent]) {
    if let Ok(mut guard) = LAST_PSI_EVENTS.write() {
        guard.clear();
        guard.extend(events.iter().cloned());
    }
}

#[cfg(feature = "psi")]
pub fn get_last_psi_events() -> Vec<PsiEvent> {
    LAST_PSI_EVENTS
        .read()
        .map(|guard| guard.clone())
        .unwrap_or_default()
}

#[cfg(feature = "psychoid")]
static LAST_PSYCHOID: Lazy<RwLock<Option<PsychoidReading>>> = Lazy::new(|| RwLock::new(None));

#[cfg(feature = "psychoid")]
pub fn set_last_psychoid(reading: &PsychoidReading) {
    if let Ok(mut guard) = LAST_PSYCHOID.write() {
        *guard = Some(reading.clone());
    }
}

#[cfg(feature = "psychoid")]
pub fn get_last_psychoid() -> Option<PsychoidReading> {
    LAST_PSYCHOID
        .read()
        .ok()
        .and_then(|guard| guard.as_ref().cloned())
}

/// Latest SoftLogic-derived telemetry that has been fed back into the "Z" control space.
#[derive(Debug, Clone, Copy, Default)]
pub struct SoftlogicZFeedback {
    /// Aggregate PSI total used when the sample was recorded.
    pub psi_total: f32,
    /// Weighted loss that triggered the feedback pulse.
    pub weighted_loss: f32,
    /// Above/Here/Beneath energy tuple at the moment of sampling.
    pub band_energy: (f32, f32, f32),
    /// Drift term captured from the gradient bands.
    pub drift: f32,
    /// Normalized control signal in the Z space. Positive values bias Above, negative bias Beneath.
    pub z_signal: f32,
}

static LAST_SOFTLOGIC_Z: OnceLock<RwLock<Option<SoftlogicZFeedback>>> = OnceLock::new();

fn softlogic_z_cell() -> &'static RwLock<Option<SoftlogicZFeedback>> {
    LAST_SOFTLOGIC_Z.get_or_init(|| RwLock::new(None))
}

/// Stores the most recent SoftLogic Z feedback sample.
pub fn set_softlogic_z(feedback: SoftlogicZFeedback) {
    if let Ok(mut guard) = softlogic_z_cell().write() {
        *guard = Some(feedback);
    }
}

/// Returns the latest SoftLogic Z feedback sample if one has been recorded.
pub fn get_softlogic_z() -> Option<SoftlogicZFeedback> {
    softlogic_z_cell()
        .read()
        .ok()
        .and_then(|guard| guard.as_ref().copied())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DesirePhaseTelemetry {
    Observation,
    Injection,
    Integration,
}

#[derive(Debug, Clone)]
pub struct DesireStepTelemetry {
    pub timestamp: SystemTime,
    pub phase: DesirePhaseTelemetry,
    pub temperature: f32,
    pub entropy: f32,
    pub hypergrad_penalty: f32,
    pub avoidance_energy: f32,
    pub logit_energy: f32,
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub lambda: f32,
    pub trigger_emitted: bool,
}

impl Default for DesireStepTelemetry {
    fn default() -> Self {
        Self {
            timestamp: UNIX_EPOCH,
            phase: DesirePhaseTelemetry::Observation,
            temperature: 0.0,
            entropy: 0.0,
            hypergrad_penalty: 0.0,
            avoidance_energy: 0.0,
            logit_energy: 0.0,
            alpha: 0.0,
            beta: 0.0,
            gamma: 0.0,
            lambda: 0.0,
            trigger_emitted: false,
        }
    }
}

static LAST_DESIRE_STEP: OnceLock<RwLock<Option<DesireStepTelemetry>>> = OnceLock::new();

fn desire_step_cell() -> &'static RwLock<Option<DesireStepTelemetry>> {
    LAST_DESIRE_STEP.get_or_init(|| RwLock::new(None))
}

pub fn set_last_desire_step(step: DesireStepTelemetry) {
    if let Ok(mut guard) = desire_step_cell().write() {
        *guard = Some(step);
    }
}

pub fn get_last_desire_step() -> Option<DesireStepTelemetry> {
    desire_step_cell()
        .read()
        .ok()
        .and_then(|guard| guard.as_ref().cloned())
}
