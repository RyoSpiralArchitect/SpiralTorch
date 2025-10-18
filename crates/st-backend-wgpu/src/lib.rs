//! Lightweight wrappers around WGPU compute pipelines used by SpiralTorch.
//! The module exposes helper routines to load WGSL shaders from disk and
//! construct the compute pipelines that power higher level tensor operators.

pub mod compaction;
pub mod compaction2ce;
pub mod compaction_2ce;
pub mod nd_indexer;
pub mod topk_keepk;

mod util;

pub use util::ShaderLoadError;
