//! High-level neural module API built on top of SpiralTorch primitives.
//!
//! This crate offers a lightweight `nn.Module` style surface that keeps the
//! stack entirely in Rust while remaining fully compatible with the hypergrad
//! tape and SpiralK planners.

pub mod dataset;
pub mod injector;
pub mod io;
pub mod layers;
pub mod loss;
pub mod module;
pub mod plan;
pub mod schedule;
pub mod trainer;

pub use dataset::{BatchIter, Dataset};
pub use injector::Injector;
pub use io::{load_bincode, load_json, save_bincode, save_json};
pub use layers::conv::{AvgPool2d, Conv1d, Conv2d, MaxPool2d};
pub use layers::linear::Linear;
pub use layers::sequential::Sequential;
pub use layers::wave_gate::WaveGate;
pub use layers::wave_rnn::WaveRnn;
pub use layers::zspace_projector::ZSpaceProjector;
pub use layers::{Relu, ToposResonator, ZSpaceMixer};
pub use loss::{HyperbolicCrossEntropy, Loss, MeanSquaredError};
pub use module::{Module, Parameter};
pub use plan::RankPlanner;
pub use schedule::{BandEnergy, GradientBands, RoundtableConfig, RoundtableSchedule};
pub use trainer::{EpochStats, ModuleTrainer};

pub use st_tensor::pure::{
    fractal::LanguageWaveEncoder, topos::OpenCartesianTopos, AmegaHypergrad, ComplexTensor,
    PureResult, Tensor, TensorError,
};
