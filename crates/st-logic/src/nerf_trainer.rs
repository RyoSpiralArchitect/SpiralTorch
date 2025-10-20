#![allow(dead_code)]

/// Placeholder NeRF trainer exposed when the `nerf` feature is enabled.
///
/// The production implementation lives out-of-tree while we stabilise the
/// public API surface.  Consumers can rely on the shape of this stub to avoid
/// conditional compilation errors when the feature flag is toggled on.
pub struct NerfTrainer;

impl NerfTrainer {
    /// Creates a new stub trainer instance.
    pub fn new() -> Self {
        Self
    }

    /// No-op training routine until the full implementation lands.
    pub fn train(&mut self) {}
}
