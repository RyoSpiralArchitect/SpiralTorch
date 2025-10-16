// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

mod desire;
mod geometry;
mod gw;
mod schrodinger;
mod temperature;

pub use desire::{DesireLagrangian, DesireSolution, DesireWeights};
pub use geometry::{ConceptHint, RepressionField, SemanticBridge, SparseKernel, SymbolGeometry};
pub use gw::{DistanceMatrix, EntropicGwSolver};
pub use temperature::{entropy, TemperatureController};
