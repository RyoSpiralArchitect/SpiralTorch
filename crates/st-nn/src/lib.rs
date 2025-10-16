// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! High-level neural module API built on top of SpiralTorch primitives.
//!
//! This crate offers a lightweight `nn.Module` style surface that keeps the
//! stack entirely in Rust while remaining fully compatible with the hypergrad
//! tape and SpiralK planners.

pub mod dataset;
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
pub mod module;
pub mod plan;
pub mod roundtable;
pub mod schedule;
pub mod trainer;

pub use dataset::{from_vec as dataset_from_vec, BatchIter, DataLoader, Dataset};
pub use gnn::{
    embed_into_biome, flows_to_canvas_tensor, flows_to_canvas_tensor_with_shape,
    fold_into_roundtable, fold_with_band_energy, GraphConsensusBridge, GraphConsensusDigest,
    GraphContext, GraphContextBuilder, GraphMonadExport, GraphNormalization, QuadBandEnergy,
    ZSpaceGraphConvolution,
};
#[cfg(feature = "golden")]
pub use golden::{
    CouncilDigest, CouncilEvidence, GoldenBlackcatPulse, GoldenCooperativeDirective,
    GoldenCouncilSnapshot, GoldenEpochReport, GoldenRetriever, GoldenRetrieverConfig,
    GoldenSelfRewriteConfig,
};
pub use highlevel::{BarycenterConfig, DifferentialTrace, SpiralSession, SpiralSessionBuilder};
pub use injector::Injector;
pub use io::{load_bincode, load_json, save_bincode, save_json};
pub use language::entropy as desire_entropy;
pub use language::pipeline::{
    LanguagePipeline, LanguagePipelineBuilder, PipelineError, PipelineResult,
};
pub use language::{
    constant, warmup, ConceptHint, DesireAutomatedStep, DesireAutomation, DesireAvoidanceReport,
    DesireChannelSink, DesireGraphBridge, DesireGraphEvent, DesireGraphSummary, DesireLagrangian,
    DesireLogRecord, DesireLogReplay, DesireLogbook, DesirePhase, DesirePipeline,
    DesirePipelineBuilder, DesirePipelineEvent, DesirePipelineSink, DesireRewriteTrigger,
    DesireRoundtableBridge, DesireRoundtableEvent, DesireRoundtableImpulse,
    DesireRoundtableSummary, DesireSchedule, DesireSolution, DesireTelemetrySink,
    DesireTrainerBridge, DesireTrainerEvent, DesireTrainerSummary, DesireTriggerBuffer,
    DesireTriggerEvent, DesireWeights, DistanceMatrix, EntropicGwSolver, MaxwellDesireBridge,
    RepressionField, SemanticBridge, SparseKernel, SymbolGeometry, TemperatureController,
};
#[cfg(feature = "psi")]
pub use language::{DesirePsiBridge, DesirePsiEvent, DesirePsiSummary};
pub use layers::conv::{AvgPool2d, Conv1d, Conv2d, MaxPool2d};
pub use layers::linear::Linear;
pub use layers::sequential::Sequential;
pub use layers::wave_gate::WaveGate;
pub use layers::wave_rnn::WaveRnn;
pub use layers::zspace_projector::ZSpaceProjector;
pub use layers::{Relu, ToposResonator, ZSpaceMixer};
pub use lightning::{
    LightningBuilder, LightningConfig, LightningConfigBuilder, LightningEpoch, LightningReport,
    LightningStage, LightningStageReport, SpiralLightning,
};
pub use loss::{HyperbolicCrossEntropy, Loss, MeanSquaredError};
pub use module::{Module, Parameter};
pub use plan::RankPlanner;
pub use roundtable::{
    simulate_proposal_locally, BlackcatModerator, BlackcatScore, DistConfig, DistMode,
    GlobalProposal, HeurOp, HeurOpKind, HeurOpLog, MetaConductor, MetaSummary, ModeratorMinutes,
    OutcomeBand, RoundtableNode,
};
pub use schedule::{BandEnergy, GradientBands, RoundtableConfig, RoundtableSchedule};
pub use st_core::runtime::blackcat::{
    BlackCatRuntime, BlackcatRuntimeStats, ChoiceGroups, StepMetrics,
};
pub use trainer::{EpochStats, ModuleTrainer};

pub use st_core::telemetry::chrono::{ChronoFrame, ChronoSummary, ChronoTimeline};
pub use st_tensor::topos::OpenCartesianTopos;
pub use st_tensor::{
    AmegaHypergrad, ComplexTensor, LanguageWaveEncoder, PureResult, Tensor, TensorError,
};
pub use st_text::{ResonanceNarrative, TextResonator};
