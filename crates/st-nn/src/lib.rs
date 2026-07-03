//! High-level neural module API built on top of SpiralTorch primitives.
//!
//! This crate offers a lightweight `nn.Module` style surface that keeps the
//! stack entirely in Rust while remaining fully compatible with the hypergrad
//! tape and SpiralK planners.

pub mod dataset;
#[cfg(feature = "golden")]
pub mod golden;
pub mod highlevel;
pub mod injector;
pub mod io;
pub mod layers;
pub mod loss;
pub mod module;
pub mod plan;
pub mod roundtable;
pub mod schedule;
pub mod trainer;

pub use dataset::{
    byte_lm_corpus_windows, byte_lm_sample_stats, byte_lm_windows, from_vec as dataset_from_vec,
    interleave_replay_samples, padded_byte_lm_samples, validate_byte_lm_samples, BatchIter,
    ByteLmSampleStats, DataLoader, Dataset, BYTE_LM_VOCAB,
};
#[cfg(feature = "golden")]
pub use golden::{GoldenEpochReport, GoldenRetriever, GoldenRetrieverConfig};
pub use highlevel::{BarycenterConfig, DifferentialTrace, SpiralSession, SpiralSessionBuilder};
pub use injector::Injector;
pub use io::{
    load_bincode, load_bincode_checked, load_json, load_json_checked, save_bincode, save_json,
};
pub use layers::conv::{AvgPool2d, Conv1d, Conv2d, MaxPool2d};
pub use layers::linear::Linear;
pub use layers::lora_linear::LoraLinear;
pub use layers::sequential::Sequential;
pub use layers::wave_gate::WaveGate;
pub use layers::wave_rnn::WaveRnn;
pub use layers::zspace_projector::ZSpaceProjector;
pub use layers::{Relu, ToposResonator, ZSpaceMixer};
pub use loss::{
    HyperbolicCrossEntropy, Loss, MeanSquaredError, SoftmaxCrossEntropy, SparseClassificationDelta,
    SparseClassificationMetrics,
};
pub use module::{
    adapt_state_dict_keys, fingerprint_state_dict, remap_state_dict_keys, state_key_rules_from_map,
    Module, Parameter, ParameterMovement, ParameterMovementReport, ParameterTrainingFingerprint,
    StateCompatibilityEntry, StateCompatibilityReport, StateCompatibilityStatus, StateFingerprint,
    StateKeyMapRule, StateLoadReport, StateTensorTransform,
};
pub use plan::RankPlanner;
pub use roundtable::{
    simulate_proposal_locally, BlackcatModerator, DistConfig, DistMode, GlobalProposal, HeurOp,
    HeurOpKind, HeurOpLog, MetaConductor, MetaSummary, ModeratorMinutes, OutcomeBand,
    RoundtableNode,
};
pub use schedule::{BandEnergy, GradientBands, RoundtableConfig, RoundtableSchedule};
pub use trainer::{
    summarize_epoch_history, EarlyStoppingConfig, EpochBestState, EpochHistory,
    EpochRetentionBestState, EpochSparseRetentionBestState, EpochStats, EpochValidationBestState,
    LrPlateauConfig, ModuleTrainer, RetentionGuardConfig, SparseFineTuneRegressionLimits,
    SparseFineTuneReport, SparseFineTuneReportSummary, SparseFineTuneSummaryComparison,
    SparseRetentionGuardConfig, TrainerStateFingerprint, TrainerStateSnapshot, TrainingResumeAudit,
    TrainingResumeFingerprint, ValidationTrainingControls,
};

pub use st_tensor::pure::topos::OpenCartesianTopos;
pub use st_tensor::pure::{
    AmegaHypergrad, ComplexTensor, LanguageWaveEncoder, PureResult, Tensor, TensorError,
};
