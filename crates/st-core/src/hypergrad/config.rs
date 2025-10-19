// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Read hypergradient solver from SpiralK (`sv` field) or env fallback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Solver { Neumann, Cg }
pub fn choose_solver() -> Solver {
    if let Ok(s) = std::env::var("SPIRAL_HEUR_K") {
        // very light parse: look for "sv:1" to pick CG
        if s.contains("sv:1") { return Solver::Cg; }
    }
    if let Ok(s) = std::env::var("SPIRAL_HYPER_SOLVE") {
        if s.to_lowercase().contains("cg") { return Solver::Cg; }
    }
    Solver::Neumann
}
