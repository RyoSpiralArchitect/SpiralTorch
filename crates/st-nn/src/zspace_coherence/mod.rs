// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

pub mod coherence_engine;
pub mod psi_synchro;
pub mod sequencer;

pub use coherence_engine::{
    BackendCapabilities, CoherenceBackend, CoherenceEngine, DomainConcept, DomainLinguisticProfile,
    LinguisticChannelReport, LinguisticContour,
};
pub use psi_synchro::{
    heatmaps_to_zpulses, run_multibranch_demo, CircleLockMapConfig, HeatmapResult, MetaMembConfig,
    MetaMembSampler, PsiBranchState, PsiSynchroConfig, PsiSynchroPulse, SyncState, SynchroBus,
    SynchroEvent,
};
pub use sequencer::{
    CoherenceDiagnostics, PreDiscardPolicy, PreDiscardRegulator, PreDiscardSnapshot,
    PreDiscardTelemetry, ZSpaceCoherenceSequencer, ZSpaceSequencerPlugin, ZSpaceSequencerStage,
};
