// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

pub mod coherence_engine;
pub mod plugin_bridge;
pub mod psi_synchro;
pub mod sequencer;
pub mod text_vae;
pub mod trace;
pub mod vae;

pub use coherence_engine::{
    BackendCapabilities, CoherenceBackend, CoherenceEngine, DomainConcept, DomainLinguisticProfile,
    LinguisticChannelReport, LinguisticContour,
};
pub use plugin_bridge::{ZSpacePluginAdapter, ZSpacePluginRegistry};
#[cfg(feature = "psi")]
pub use psi_synchro::BranchPsiReading;
#[cfg(feature = "golden")]
pub use psi_synchro::{heatmaps_to_golden_telemetry, PsiGoldenTelemetry};
pub use psi_synchro::{
    heatmaps_to_zpulses, run_multibranch_demo, run_zspace_learning_pass, ArnoldTongueSummary,
    BranchAtlasFragment, CircleLockMapConfig, HeatmapAnalytics, HeatmapResult, MetaMembConfig,
    MetaMembSampler, PsiBranchState, PsiSynchroConfig, PsiSynchroPulse, PsiSynchroResult,
    PsiTelemetryConfig, SyncState, SynchroBus, SynchroEvent,
};
pub use sequencer::{
    is_swap_invariant, CoherenceDiagnostics, CoherenceLabel, CoherenceObservation,
    CoherenceSignature, PreDiscardPolicy, PreDiscardRegulator, PreDiscardSnapshot,
    PreDiscardTelemetry, ZSpaceCoherenceSequencer, ZSpaceSequencerPlugin, ZSpaceSequencerStage,
};
pub use trace::{
    coherence_relation_tensor, ZSpaceTrace, ZSpaceTraceConfig, ZSpaceTraceEvent,
    ZSpaceTraceRecorder,
};
pub use text_vae::ZSpaceTextVae;
pub use vae::{MellinBasis, ZSpaceVae, ZSpaceVaeState, ZSpaceVaeStats};
