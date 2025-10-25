//! Quantization-aware training (QAT) and structured pruning primitives for SpiralTorch.
//!
//! The utilities implemented here are intentionally lightweight.  They provide
//! deterministic, allocation-friendly routines that can be wired into higher
//! level training loops from Python or Rust without forcing a dependency on a
//! full deep learning framework.  Instead we operate on plain slices of `f32`
//! values and emit rich reports describing the transformation.

mod ops;
pub mod pruning;
pub mod quantization;
pub mod report;

pub use pruning::{StructuredPruner, StructuredPruningConfig, StructuredPruningWorkspace};
pub use quantization::{QatConfig, QatObserver, QuantizationLeveling};
pub use report::{
    CompressionReport, OptimisationError, QuantizationReport, StructuredPruningReport,
};
