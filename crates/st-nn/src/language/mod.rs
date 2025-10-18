// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

mod automation;
mod desire;
mod geometry;
mod gw;
mod logbook;
mod maxwell;
mod pipeline;
mod schrodinger;
mod temperature;

use self::pipeline as pipeline_exports;

pub use automation::{DesireAutomatedStep, DesireAutomation, DesireRewriteTrigger};
pub use desire::{
    constant, warmup, DesireAvoidanceReport, DesireLagrangian, DesirePhase, DesireSchedule,
    DesireSolution, DesireWeights,
};
pub use geometry::{ConceptHint, RepressionField, SemanticBridge, SparseKernel, SymbolGeometry};
pub use gw::{DistanceMatrix, EntropicGwSolver};
pub use logbook::{DesireLogRecord, DesireLogReplay, DesireLogbook};
pub use maxwell::MaxwellDesireBridge;
pub use pipeline::{
    DesireChannelSink, DesireGraphBridge, DesireGraphEvent, DesireGraphSummary, DesirePipeline,
    DesirePipelineBuilder, DesirePipelineEvent, DesirePipelineSink, DesireRoundtableBridge,
    DesireRoundtableEvent, DesireRoundtableImpulse, DesireRoundtableSummary, DesireTelemetrySink,
    DesireTrainerBridge, DesireTrainerEvent, DesireTrainerSummary, DesireTriggerBuffer,
    DesireTriggerEvent,
};
#[cfg(feature = "psi")]
pub use pipeline_exports::{DesirePsiBridge, DesirePsiEvent, DesirePsiSummary};
pub use pipeline_exports::{
    LanguagePipeline, LanguagePipelineBuilder, PipelineError, PipelineResult,
};
pub use temperature::{entropy, TemperatureController};
