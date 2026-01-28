// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! High-level neural module API built on top of SpiralTorch primitives.
//!
//! This crate offers a lightweight `nn.Module` style surface that keeps the
//! stack entirely in Rust while remaining fully compatible with the hypergrad
//! tape and SpiralK planners.

pub mod cloud;
pub mod dataset;
pub mod discovery;
pub mod gnn;
#[cfg(feature = "golden")]
pub mod golden;
pub mod highlevel;
pub mod injector;
pub mod io;
pub mod language;
pub mod layers;
pub mod lightning;
pub mod loss;
pub mod mixed_precision;
pub mod module;
pub mod optim;
pub mod plan;
pub mod roundtable;
pub mod schedule;
pub mod trainer;
pub mod z_rba;
pub mod zspace_coherence;

pub use dataset::{from_vec as dataset_from_vec, BatchIter, DataLoader, Dataset};
pub use discovery::{
    ModuleCategory, ModuleDiscoveryRegistry, ModuleMetadata, ModulePipelineBuilder,
};
pub use gnn::{
    embed_into_biome, flows_to_canvas_tensor, flows_to_canvas_tensor_with_shape,
    fold_into_roundtable, fold_with_band_energy, AggregationReducer, GraphActivation,
    GraphConsensusBridge, GraphConsensusDigest, GraphContext, GraphContextBuilder, GraphLayerSpec,
    GraphMonadExport, GraphNormalization, NeighborhoodAggregation, QuadBandEnergy,
    RoundtableBandInfluence, RoundtableBandSignal, ZSpaceGraphConvolution, ZSpaceGraphNetwork,
    ZSpaceGraphNetworkBuilder,
};
#[cfg(feature = "golden")]
pub use golden::{
    CouncilDigest, CouncilEvidence, GoldenBlackcatPulse, GoldenCooperativeDirective,
    GoldenCouncilSnapshot, GoldenEpochReport, GoldenRetriever, GoldenRetrieverConfig,
    GoldenSelfRewriteConfig,
};
pub use highlevel::{BarycenterConfig, DifferentialTrace, SpiralSession, SpiralSessionBuilder};
pub use injector::Injector;
pub use io::{
    load_bincode, load_json, load_state_dict_bincode, load_state_dict_json, save_bincode,
    save_json, save_state_dict_bincode, save_state_dict_json,
};
pub use language::entropy as desire_entropy;
pub use language::{
    constant, warmup, ConceptHint, DesireAutomatedStep, DesireAutomation, DesireAvoidanceReport,
    DesireChannelSink, DesireGraphBridge, DesireGraphEvent, DesireGraphSummary, DesireLagrangian,
    DesireLogRecord, DesireLogReplay, DesireLogbook, DesirePhase, DesirePipeline,
    DesirePipelineBuilder, DesirePipelineEvent, DesirePipelineSink, DesireRewriteTrigger,
    DesireRoundtableBridge, DesireRoundtableEvent, DesireRoundtableImpulse,
    DesireRoundtableSummary, DesireSchedule, DesireSolution, DesireTelemetryBundle,
    DesireTelemetrySink, DesireTrainerBridge, DesireTrainerEvent, DesireTrainerSummary,
    DesireTriggerBuffer, DesireTriggerEvent, DesireWeights, DistanceMatrix, EntropicGwSolver,
    GeometryBiasConfig, GeometryBiasContext, GeometryBiasMetrics, GeometryBiasSnapshot,
    GeometryBiasUpdate, GeometryCoherenceSample, LanguagePipeline, LanguagePipelineBuilder,
    MaxwellDesireBridge, NarrativeHint, NarrativeSummary, PipelineError, PipelineResult,
    RepressionField, SemanticBridge, SparseKernel, SymbolGeometry, TemperatureController,
};
#[cfg(feature = "psi")]
pub use language::{DesirePsiBridge, DesirePsiEvent, DesirePsiSummary};
pub use layers::conv::{AvgPool2d, Conv1d, Conv2d, Conv3d, Conv4d, Conv6da, MaxPool2d};
pub use layers::linear::Linear;
pub use layers::sequential::Sequential;
pub use layers::wave_gate::WaveGate;
pub use layers::wave_rnn::WaveRnn;
pub use layers::zspace_projector::ZSpaceProjector;
pub use layers::{
    BatchNorm1d, Dropout, Embedding, Gelu, HamiltonJacobiFlow, KleinGordonPropagation, LayerNorm,
    Lstm, Relu, Scaler, StochasticSchrodingerLayer, ToposResonator, ZRelativityModule,
    ZSpaceBatchNorm1d, ZSpaceBatchNormTelemetry, ZSpaceCoherenceScan, ZSpaceCoherenceWaveBlock,
    ZSpaceLayerNorm, ZSpaceLayerNormTelemetry, ZSpaceMixer,
};
pub use lightning::{
    LightningBuilder, LightningConfig, LightningConfigBuilder, LightningEpoch, LightningReport,
    LightningStage, LightningStageReport, SpiralLightning,
};
pub use loss::{CategoricalCrossEntropy, HyperbolicCrossEntropy, Loss, MeanSquaredError};
pub use module::{Module, Parameter};
pub use optim::{
    LocalLearningRateAdapter, LrScheduler, OptimizerMode, SpectralLrAdapter, WarmupCosineScheduler,
    ZSpaceOptimizer,
};
pub use plan::RankPlanner;
pub use roundtable::{
    simulate_proposal_locally, BlackcatModerator, BlackcatScore, DistConfig, DistMode,
    GlobalProposal, HeurOp, HeurOpKind, HeurOpLog, MetaConductor, MetaSummary, ModeratorMinutes,
    OutcomeBand, RoundtableGnnBridge, RoundtableNode,
};
pub use schedule::{BandEnergy, GradientBands, RoundtableConfig, RoundtableSchedule};
pub use st_core::runtime::blackcat::{
    BlackCatRuntime, BlackcatRuntimeStats, ChoiceGroups, StepMetrics,
};
#[cfg(feature = "selfsup")]
pub use trainer::selfsup::{
    InfoNCEConfig, InfoNCEEpochMetrics, SelfSupBatch, SelfSupEpoch, SelfSupEpochReport,
    SelfSupEpochTelemetry, SelfSupObjective, SelfSupPlanReport, SelfSupStage, SelfSupStageReport,
};
pub use trainer::{
    EpochStats, ModuleTrainer, SpectralAdjustmentMetrics, SpectralLearningRatePolicy,
};
pub use z_rba::{
    AttentionTelemetry as ZAttentionTelemetry, BetaGate, BetaGateConfig, BetaGateSample, CovHead,
    CovHeadTelemetry, ReliabilityBin as ZReliabilityBin, SimpleZFrame, ZCov, ZIndex,
    ZMetricWeights, ZRBAConfig, ZRBAMetrics, ZRBATelemetry, ZTelemetryBundle, ZTensor, ZRBA,
};
#[cfg(feature = "psi")]
pub use zspace_coherence::BranchPsiReading;
#[cfg(feature = "golden")]
pub use zspace_coherence::{heatmaps_to_golden_telemetry, PsiGoldenTelemetry};
pub use zspace_coherence::{
    coherence_relation_tensor, heatmaps_to_zpulses, is_swap_invariant, run_multibranch_demo,
    run_zspace_learning_pass, ArnoldTongueSummary, BackendCapabilities, BranchAtlasFragment,
    CircleLockMapConfig, CoherenceBackend, CoherenceEngine, CoherenceLabel, CoherenceObservation,
    CoherenceSignature, DomainConcept, DomainLinguisticProfile, HeatmapAnalytics, HeatmapResult,
    LinguisticChannelReport, LinguisticContour, MellinBasis, MetaMembConfig, MetaMembSampler,
    PreDiscardPolicy, PreDiscardRegulator, PsiBranchState, PsiSynchroConfig, PsiSynchroPulse,
    PsiSynchroResult, PsiTelemetryConfig, SyncState, SynchroBus, SynchroEvent,
    ZSpaceCoherenceSequencer, ZSpaceTrace, ZSpaceTraceConfig, ZSpaceTraceEvent,
    ZSpaceTraceRecorder, ZSpaceVae, ZSpaceVaeState, ZSpaceVaeStats,
};

pub use mixed_precision::{autocast_enabled, AutocastGuard, GradScaler};
pub use st_core::telemetry::chrono::{ChronoFrame, ChronoSummary, ChronoTimeline};
pub use st_tensor::topos::OpenCartesianTopos;
pub use st_tensor::{
    AmegaHypergrad, ComplexTensor, LanguageWaveEncoder, PureResult, Tensor, TensorError,
};
pub use st_text::{ResonanceNarrative, TextResonator};
