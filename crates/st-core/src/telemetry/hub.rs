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

use super::chrono::ChronoLoopSignal;
#[cfg(feature = "psi")]
use super::psi::PsiReading;
#[cfg(feature = "psychoid")]
use super::psychoid::PsychoidReading;
#[cfg(feature = "collapse")]
use crate::engine::collapse_drive::DriveCmd;

#[cfg(feature = "psi")]
static LAST_PSI: Lazy<RwLock<Option<PsiReading>>> = Lazy::new(|| RwLock::new(None));

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

static LAST_CHRONO_LOOP: OnceLock<RwLock<Option<ChronoLoopSignal>>> = OnceLock::new();

fn chrono_loop_cell() -> &'static RwLock<Option<ChronoLoopSignal>> {
    LAST_CHRONO_LOOP.get_or_init(|| RwLock::new(None))
}

/// Stores the most recent chrono loop signal so other nodes can consume it.
pub fn set_chrono_loop(signal: ChronoLoopSignal) {
    if let Ok(mut guard) = chrono_loop_cell().write() {
        *guard = Some(signal);
    }
}

/// Returns the latest chrono loop signal, if any has been recorded.
pub fn get_chrono_loop() -> Option<ChronoLoopSignal> {
    chrono_loop_cell()
        .read()
        .ok()
        .and_then(|guard| guard.as_ref().cloned())
}

#[cfg(feature = "collapse")]
#[derive(Clone, Debug)]
pub struct CollapsePulse {
    /// Step of the PSI reading that triggered the command.
    pub step: u64,
    /// Aggregate PSI total associated with the command.
    pub total: f32,
    /// Command emitted by the collapse drive.
    pub command: DriveCmd,
    /// Latest chrono loop signal observed when the command was issued.
    pub loop_signal: Option<ChronoLoopSignal>,
}

#[cfg(feature = "collapse")]
static LAST_COLLAPSE: OnceLock<RwLock<Option<CollapsePulse>>> = OnceLock::new();

#[cfg(feature = "collapse")]
fn collapse_cell() -> &'static RwLock<Option<CollapsePulse>> {
    LAST_COLLAPSE.get_or_init(|| RwLock::new(None))
}

#[cfg(feature = "collapse")]
/// Stores the most recent collapse command pulse.
pub fn set_collapse_pulse(pulse: CollapsePulse) {
    if let Ok(mut guard) = collapse_cell().write() {
        *guard = Some(pulse);
    }
}

#[cfg(feature = "collapse")]
/// Returns the most recent collapse pulse, if any.
pub fn get_collapse_pulse() -> Option<CollapsePulse> {
    collapse_cell()
        .read()
        .ok()
        .and_then(|guard| guard.as_ref().cloned())
}
