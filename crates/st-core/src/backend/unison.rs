// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Stable facade for the backend-aware unified rank chooser.
//!
//! The implementation lives in `unison_heuristics`; this module deliberately
//! contains no fallback table or client-specific normalization. Rust callers,
//! bindings, and executors therefore enter the same chooser.

use super::device_caps::DeviceCaps;

pub use super::unison_heuristics::{Choice, RankKind};

/// Selects one rich rank-k choice through the canonical backend-aware chooser.
pub fn choose_unified_rank(
    rows: u32,
    cols: u32,
    k: u32,
    caps: DeviceCaps,
    kind: RankKind,
) -> Choice {
    super::unison_heuristics::choose_unified_rank(rows, cols, k, caps, kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::device_caps::BackendKind;

    #[test]
    fn facade_preserves_the_rich_backend_choice() {
        let caps = BackendKind::Wgpu.default_caps();
        let choice = choose_unified_rank(64, 4_096, 48, caps, RankKind::MidK);

        assert!(choice.wg > 0);
        assert!(choice.kl > 0);
        assert!(choice.ctile > 0);
        assert!(choice.latency_window.is_some());
        assert_eq!(choice.subgroup, caps.subgroup);
    }
}
