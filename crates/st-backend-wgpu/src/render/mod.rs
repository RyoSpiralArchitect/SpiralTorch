//! Rendering helpers that project Z-space volumes into time-aware
//! representations. The actual GPU pipelines are mirrored with CPU
//! fallbacks so that the logic can be tested without a live device.

mod temporal;

pub use temporal::{
    TemporalRenderOutput, TemporalRenderSlice, TemporalRenderer, TemporalRendererConfig,
    TemporalVolumeLike,
};
