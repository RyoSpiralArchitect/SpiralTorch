// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Unison: backend‑agnostic heuristic chooser.
//! Reads SpiralK / SoftLogic / Redis the same way for WGPU, HIP, CUDA.
//! Falls back to generated table & safe defaults.

use super::wgpu_heuristics; // Choice shape; you can refactor into a common type crate

pub fn choose_unified(rows:usize, cols:usize, k:usize, has_subgroup:bool)->wgpu_heuristics::Choice{
    // Reuse WGPU path (which already stacks: SoftLogic -> SpiralK hard -> KV -> Table -> Fallback)
    // For HIP/CUDA callers, pass subgroup=false (or device capability) and consume 'mk' for merge_kind.
    wgpu_heuristics::choose(rows as u32, cols as u32, k as u32, has_subgroup).unwrap_or_else(||
        // Fallback (same as inside)
        super::wgpu_heuristics::Choice{
            use_2ce: cols>32_768 || k>128,
            wg: if has_subgroup {256} else {128},
            kl: if k>=64 {32} else if k>=16 {16} else {8},
            ch: if cols>16_384 {8192} else {0},
            mk: if k<=128 {2} else if k<=2048 {1} else {0},
        }
    )
}
