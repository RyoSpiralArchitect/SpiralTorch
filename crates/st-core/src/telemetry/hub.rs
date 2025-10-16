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
use std::collections::VecDeque;

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

/// Envelope combining chrono loop telemetry with collapse/Z feedback so other nodes can replay it.
#[derive(Clone, Debug)]
pub struct LoopbackEnvelope {
    /// Timestamp associated with the captured loop signal.
    pub timestamp: f32,
    /// Optional identifier describing which node produced the envelope.
    pub source: Option<String>,
    /// Participation/support weight contributed by the source node.
    pub support: f32,
    /// Optional aggregate collapse total associated with the envelope.
    pub collapse_total: Option<f32>,
    /// Optional Z-space control bias produced by the softlogic observer.
    pub z_signal: Option<f32>,
    /// Optional SpiralK script hint that accompanied the telemetry.
    pub script_hint: Option<String>,
    /// Chrono loop signal captured at the timestamp.
    pub loop_signal: ChronoLoopSignal,
}

impl LoopbackEnvelope {
    /// Creates a new envelope from the supplied chrono loop signal.
    pub fn new(loop_signal: ChronoLoopSignal) -> Self {
        let timestamp = loop_signal.summary.latest_timestamp;
        Self {
            timestamp,
            source: None,
            support: 1.0,
            collapse_total: None,
            z_signal: None,
            script_hint: None,
            loop_signal,
        }
    }

    /// Annotates the envelope with a source identifier.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Updates the support weight carried by the envelope.
    pub fn with_support(mut self, support: f32) -> Self {
        self.support = if support.is_finite() { support } else { 1.0 };
        self
    }

    /// Records an optional collapse total associated with the envelope.
    pub fn with_collapse_total(mut self, total: Option<f32>) -> Self {
        self.collapse_total = total.filter(|value| value.is_finite());
        self
    }

    /// Records an optional Z-space control signal.
    pub fn with_z_signal(mut self, z: Option<f32>) -> Self {
        self.z_signal = z.filter(|value| value.is_finite());
        self
    }

    /// Annotates the envelope with a SpiralK script hint.
    pub fn with_script_hint(mut self, script: Option<String>) -> Self {
        self.script_hint = script;
        self
    }
}

static LOOPBACK_BUFFER: OnceLock<RwLock<VecDeque<LoopbackEnvelope>>> = OnceLock::new();

fn loopback_cell() -> &'static RwLock<VecDeque<LoopbackEnvelope>> {
    LOOPBACK_BUFFER.get_or_init(|| RwLock::new(VecDeque::with_capacity(32)))
}

/// Pushes a new loopback envelope into the global queue, keeping the buffer bounded.
pub fn push_loopback_envelope(envelope: LoopbackEnvelope) {
    if let Ok(mut guard) = loopback_cell().write() {
        guard.push_back(envelope);
        while guard.len() > 64 {
            guard.pop_front();
        }
    }
}

/// Drains up to `limit` loopback envelopes from the queue in FIFO order.
pub fn drain_loopback_envelopes(limit: usize) -> Vec<LoopbackEnvelope> {
    if limit == 0 {
        return Vec::new();
    }
    if let Ok(mut guard) = loopback_cell().write() {
        let mut drained = Vec::new();
        for _ in 0..limit {
            if let Some(envelope) = guard.pop_front() {
                drained.push(envelope);
            } else {
                break;
            }
        }
        drained
    } else {
        Vec::new()
    }
}

/// Returns up to `limit` envelopes without mutating the queue.
pub fn peek_loopback_envelopes(limit: usize) -> Vec<LoopbackEnvelope> {
    if limit == 0 {
        return Vec::new();
    }
    if let Ok(guard) = loopback_cell().read() {
        guard.iter().take(limit).cloned().collect()
    } else {
        Vec::new()
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::chrono::{ChronoHarmonics, ChronoPeak, ChronoSummary};

    fn sample_summary(timestamp: f32) -> ChronoSummary {
        ChronoSummary {
            frames: 4,
            duration: 1.0,
            latest_timestamp: timestamp,
            mean_drift: 0.2,
            mean_abs_drift: 0.4,
            drift_std: 0.1,
            mean_energy: 2.0,
            energy_std: 0.3,
            mean_decay: -0.1,
            min_energy: 1.5,
            max_energy: 2.5,
        }
    }

    #[test]
    fn loopback_queue_drains_in_order() {
        // Ensure the buffer starts empty for the test.
        let _ = drain_loopback_envelopes(usize::MAX);
        let signal_a = ChronoLoopSignal::new(sample_summary(1.0), None);
        let mut harmonics = ChronoHarmonics {
            frames: 4,
            duration: 1.0,
            sample_rate: 4.0,
            nyquist: 2.0,
            drift_power: vec![0.1; 4],
            energy_power: vec![0.2; 4],
            dominant_drift: Some(ChronoPeak {
                frequency: 0.5,
                magnitude: 0.3,
                phase: 0.0,
            }),
            dominant_energy: None,
        };
        let signal_b = ChronoLoopSignal::new(sample_summary(2.0), Some(harmonics.clone()));
        harmonics.dominant_energy = Some(ChronoPeak {
            frequency: 0.8,
            magnitude: 0.6,
            phase: 0.1,
        });
        let signal_c = ChronoLoopSignal::new(sample_summary(3.0), Some(harmonics));

        push_loopback_envelope(LoopbackEnvelope::new(signal_a).with_support(1.0));
        push_loopback_envelope(LoopbackEnvelope::new(signal_b).with_support(2.0));
        push_loopback_envelope(
            LoopbackEnvelope::new(signal_c)
                .with_support(3.0)
                .with_collapse_total(Some(1.2))
                .with_z_signal(Some(0.4)),
        );

        let drained = drain_loopback_envelopes(2);
        assert_eq!(drained.len(), 2);
        assert!(drained[0].timestamp <= drained[1].timestamp);
        let remaining = drain_loopback_envelopes(2);
        assert_eq!(remaining.len(), 1);
        assert!(drain_loopback_envelopes(1).is_empty());
    }
}
