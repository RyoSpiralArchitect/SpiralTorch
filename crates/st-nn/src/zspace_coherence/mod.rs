// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

pub mod coherence_engine;
pub mod sequencer;

pub use coherence_engine::{
    CoherenceBackend, CoherenceEngine, DomainConcept, DomainSemanticProfile,
};
pub use sequencer::ZSpaceCoherenceSequencer;
