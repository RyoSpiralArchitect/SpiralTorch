// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

pub mod context;
pub mod handoff;
pub mod layer;

pub use context::{GraphContext, GraphContextBuilder, GraphNormalization};
pub use handoff::{
    embed_into_biome, flows_to_canvas_tensor, flows_to_canvas_tensor_with_shape,
    fold_into_roundtable, GraphMonadExport, QuadBandEnergy,
};
pub use layer::ZSpaceGraphConvolution;
