// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Backend-aware unified chooser (TopK/MidK/BottomK).
//! The goal is to stitch together the generated WGPU tables, environment DSL,
//! Redis/KV hints, and the pure Rust fallback so that every backend has a sane
//! starting point even when optional features are disabled.

use super::device_caps::{BackendKind, DeviceCaps};
use super::kdsl_bridge;
use super::wgpu_heuristics;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RankKind {
    TopK,
    MidK,
    BottomK,
}

impl RankKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            RankKind::TopK => "topk",
            RankKind::MidK => "midk",
            RankKind::BottomK => "bottomk",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Choice {
    pub use_2ce: bool,
    pub wg: u32,
    pub kl: u32,
    pub ch: u32,
    pub mk: u32,           // 0=bitonic,1=shared,2=warp
    pub mkd: u32,          // 0=auto,1=heap,2=kway,3=bitonic,4=warp_heap,5=warp_bitonic
    pub tile: u32,         // TopK sweep tile
    pub ctile: u32,        // MidK/BottomK compaction tile
    pub subgroup: bool,    // Whether the planner assumed subgroup execution
    pub fft_tile: u32,     // Column tile for FFT/fractional kernels
    pub fft_radix: u32,    // Preferred radix for the FFT planner
    pub fft_segments: u32, // Number of ND segments folded by the kernel
}

impl Choice {
    fn merge_kind_name(&self) -> &'static str {
        match self.mk {
            1 => "shared",
            2 => "warp",
            _ => "bitonic",
        }
    }

    fn merge_detail_name(&self) -> &'static str {
        match self.mkd {
            1 => "heap",
            2 => "kway",
            3 => "bitonic",
            4 => "warp_heap",
            5 => "warp_bitonic",
            _ => "auto",
        }
    }

    /// Formats the choice into a SpiralK unison snippet that mirrors the runtime decision.
    pub fn to_unison_script(&self, kind: RankKind) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let _ = writeln!(&mut out, "unison {} {{", kind.as_str());
        let _ = writeln!(&mut out, "  workgroup {}", self.wg);
        let _ = writeln!(&mut out, "  lanes {}", self.kl);
        if self.ch != 0 {
            let _ = writeln!(&mut out, "  channel_stride {}", self.ch);
        }
        let _ = writeln!(
            &mut out,
            "  merge {} {}",
            self.merge_kind_name(),
            self.merge_detail_name()
        );
        if self.tile != 0 {
            let _ = writeln!(&mut out, "  tile {}", self.tile);
        }
        if self.ctile != 0 {
            let _ = writeln!(&mut out, "  compaction_tile {}", self.ctile);
        }
        let _ = writeln!(&mut out, "  two_stage {}", self.use_2ce);
        if self.fft_tile != 0 {
            let _ = writeln!(&mut out, "  fft_tile {}", self.fft_tile);
        }
        if self.fft_radix != 0 {
            let _ = writeln!(&mut out, "  fft_radix {}", self.fft_radix);
        }
        if self.fft_segments != 0 {
            let _ = writeln!(&mut out, "  fft_segments {}", self.fft_segments);
        }
        let _ = writeln!(&mut out, "}}");
        out
    }
}

fn latency_sensitive(rows: u32, cols: u32, k: u32, caps: &DeviceCaps) -> bool {
    if rows == 0 {
        return false;
    }

    let lanes = caps.lane_width.max(1);
    let small_rows = rows <= lanes.saturating_mul(4).max(64);
    let modest_cols = cols <= 131_072;
    let modest_k = k <= lanes.saturating_mul(8);

    small_rows && modest_cols && modest_k
}

fn fallback(rows: u32, cols: u32, k: u32, caps: &DeviceCaps, kind: RankKind) -> Choice {
    let wg = caps.recommended_workgroup(rows);
    let kl = caps.recommended_kl(k);
    let mut ch = caps.recommended_channel_stride(cols);
    let (mut tile, mut ctile) = caps.recommended_tiles(cols);

    let low_latency = latency_sensitive(rows, cols, k, caps);
    if low_latency {
        ch = 0;
        let lanes = caps.lane_width.max(1);
        let latency_cap = if cols <= 16_384 { 512 } else { 1024 };
        let aligned_cap = ((latency_cap + lanes - 1) / lanes) * lanes;
        let min_cap = ((128 + lanes - 1) / lanes) * lanes;
        tile = tile.min(aligned_cap.max(lanes)).max(min_cap.max(lanes));
    }

    if matches!(kind, RankKind::BottomK) {
        ctile = ctile.min(tile / 2).max(128);
    }

    let mut mk = caps.preferred_merge_kind(k);
    let mut mkd = caps.preferred_substrategy(mk, k);
    if low_latency && caps.subgroup && matches!(kind, RankKind::TopK) {
        if k <= caps.lane_width.saturating_mul(8) {
            mk = 2;
            mkd = if k <= 128 { 4 } else { 5 };
        }
    }
    let mut use_2ce = caps.prefers_two_stage_with_rows(rows, cols, k);
    if low_latency {
        use_2ce = false;
    }

    let fft_tile = ((cols.max(1) + 1023) / 1024) as u32 * 1024;
    let fft_radix = if k.is_power_of_two() { 4 } else { 2 };
    let fft_segments = if cols > 131_072 {
        4
    } else if cols > 32_768 {
        2
    } else {
        1
    };

    Choice {
        use_2ce,
        wg,
        kl,
        ch,
        mk,
        mkd,
        tile,
        ctile,
        subgroup: caps.subgroup,
        fft_tile,
        fft_radix,
        fft_segments,
    }
}

fn uses_shared_memory(choice: &Choice) -> bool {
    choice.mk == 1 || matches!(choice.mkd, 1 | 2 | 4 | 5)
}

fn enforce_shared_memory(choice: &mut Choice, caps: &DeviceCaps, k: u32) {
    if let Some(limit) = caps.shared_mem_per_workgroup {
        if limit < 32 * 1024 {
            choice.use_2ce = false;
        }
        if uses_shared_memory(choice) && limit < 48 * 1024 {
            choice.mk = caps.preferred_merge_kind(k);
            choice.mkd = caps.preferred_substrategy(choice.mk, k);
        }
    }
}

fn refine_choice(
    mut choice: Choice,
    baseline: Choice,
    caps: &DeviceCaps,
    rows: u32,
    cols: u32,
    k: u32,
    kind: RankKind,
) -> Choice {
    if choice.wg == 0 {
        choice.wg = baseline.wg;
    }
    choice.wg = caps.align_workgroup(choice.wg);

    if choice.kl == 0 {
        choice.kl = baseline.kl;
    }
    if choice.ch == 0 {
        choice.ch = baseline.ch;
    }
    if choice.mk == 0 {
        choice.mk = baseline.mk;
    }
    if choice.mkd == 0 {
        choice.mkd = caps.preferred_substrategy(choice.mk, k);
    }
    if !choice.subgroup {
        choice.subgroup = baseline.subgroup;
    }
    if choice.tile == 0 {
        choice.tile = baseline.tile;
    }
    choice.tile = caps.preferred_tile(cols, choice.tile);

    let low_latency = latency_sensitive(rows, cols, k, caps);
    if low_latency {
        choice.use_2ce = false;
        if baseline.ch == 0 {
            choice.ch = 0;
        }
        if choice.tile > baseline.tile {
            choice.tile = baseline.tile;
        }
    }

    if matches!(kind, RankKind::MidK | RankKind::BottomK) {
        if choice.ctile == 0 {
            choice.ctile = baseline.ctile;
        }
        choice.ctile = caps.preferred_compaction_tile(cols, choice.ctile);
    } else {
        choice.ctile = 0;
    }

    if choice.fft_tile == 0 {
        choice.fft_tile = baseline.fft_tile;
    }
    choice.fft_tile = caps.preferred_tile(cols, choice.fft_tile);

    if choice.fft_radix == 0 {
        choice.fft_radix = baseline.fft_radix;
    }
    choice.fft_radix = choice.fft_radix.clamp(2, 4);

    if choice.fft_segments == 0 {
        choice.fft_segments = baseline.fft_segments;
    }
    choice.fft_segments = choice.fft_segments.clamp(1, 4);

    let expected_two_stage = caps.prefers_two_stage_with_rows(rows, cols, k);
    if expected_two_stage && !choice.use_2ce {
        choice.use_2ce = baseline.use_2ce;
    }
    if !expected_two_stage {
        choice.use_2ce = false;
    }

    if low_latency {
        choice.use_2ce = false;
        if baseline.ch == 0 {
            choice.ch = 0;
        }
    }

    enforce_shared_memory(&mut choice, caps, k);

    choice
}

fn closeness(actual: u32, target: u32) -> f32 {
    if actual == 0 || target == 0 {
        return 0.0;
    }
    let diff = actual.abs_diff(target) as f32;
    let denom = target.max(1) as f32;
    (1.0 - (diff / denom).min(1.0)).max(0.0)
}

fn score_choice(
    choice: &Choice,
    caps: &DeviceCaps,
    rows: u32,
    cols: u32,
    k: u32,
    baseline: &Choice,
    kind: RankKind,
) -> f32 {
    let mut score = 0.0;

    let low_latency = latency_sensitive(rows, cols, k, caps);

    let expected_two_stage = caps.prefers_two_stage_with_rows(rows, cols, k);
    if choice.use_2ce == expected_two_stage {
        score += 0.25;
    } else {
        score -= 0.1;
    }

    score += closeness(choice.wg, baseline.wg) * 0.2;
    score += caps.occupancy_score(choice.wg) * 0.2;

    score += closeness(choice.kl, baseline.kl) * 0.15;
    score += closeness(choice.tile, baseline.tile) * 0.1;

    if matches!(kind, RankKind::MidK | RankKind::BottomK) {
        score += closeness(choice.ctile, baseline.ctile) * 0.1;
    }

    score += closeness(choice.fft_tile, baseline.fft_tile) * 0.05;
    if choice.fft_radix == baseline.fft_radix {
        score += 0.025;
    }
    if choice.fft_segments == baseline.fft_segments {
        score += 0.025;
    }

    if choice.mk == caps.preferred_merge_kind(k) {
        score += 0.1;
    }
    if choice.mkd == caps.preferred_substrategy(choice.mk, k) {
        score += 0.1;
    }

    if low_latency {
        if !choice.use_2ce {
            score += 0.05;
        } else {
            score -= 0.05;
        }
        if choice.ch == 0 {
            score += 0.025;
        } else {
            score -= 0.025;
        }
        let lanes = caps.lane_width.max(1);
        let latency_cap = if cols <= 16_384 { 512 } else { 1024 };
        let aligned_cap = ((latency_cap + lanes - 1) / lanes) * lanes;
        score += closeness(choice.tile, aligned_cap) * 0.05;
        if choice.mk == 2 {
            score += 0.025;
        }
    }

    score
}

fn convert_wgpu_choice(choice: wgpu_heuristics::Choice, subgroup: bool) -> Choice {
    Choice {
        use_2ce: choice.use_2ce,
        wg: choice.wg,
        kl: choice.kl,
        ch: choice.ch,
        mk: 0,
        mkd: 0,
        tile: choice.tile_cols,
        ctile: choice.ctile,
        subgroup,
        fft_tile: choice.tile_cols,
        fft_radix: choice.radix,
        fft_segments: choice.segments,
    }
}

pub fn choose_unified_rank(
    rows: u32,
    cols: u32,
    k: u32,
    caps: DeviceCaps,
    kind: RankKind,
) -> Choice {
    let baseline = fallback(rows, cols, k, &caps, kind);
    let mut best = baseline;
    let mut best_score = score_choice(&baseline, &caps, rows, cols, k, &baseline, kind);

    // Hard DSL overrides from the environment.
    let (dsl_hard, _, _) = kdsl_bridge::parse_env_dsl_plus_kind(
        rows,
        cols,
        k,
        caps.subgroup,
        match kind {
            RankKind::TopK => "topk",
            RankKind::MidK => "midk",
            RankKind::BottomK => "bottomk",
        },
    );
    if let Some(hard) = dsl_hard {
        let refined = refine_choice(
            convert_wgpu_choice(hard, caps.subgroup),
            baseline,
            &caps,
            rows,
            cols,
            k,
            kind,
        );
        let score = score_choice(&refined, &caps, rows, cols, k, &baseline, kind);
        if score > best_score {
            best = refined;
            best_score = score;
        }
    }

    if let Some(kv) = kdsl_bridge::choose_from_kv(rows, cols, k, caps.subgroup) {
        let refined = refine_choice(
            convert_wgpu_choice(kv, caps.subgroup),
            baseline,
            &caps,
            rows,
            cols,
            k,
            kind,
        );
        let score = score_choice(&refined, &caps, rows, cols, k, &baseline, kind);
        if score > best_score {
            best = refined;
            best_score = score;
        }
    }

    if caps.backend == BackendKind::Wgpu {
        if let Some(choice) =
            wgpu_heuristics::choose(rows as usize, cols as usize, k as usize, caps.subgroup)
        {
            let refined = refine_choice(
                convert_wgpu_choice(choice, caps.subgroup),
                baseline,
                &caps,
                rows,
                cols,
                k,
                kind,
            );
            let score = score_choice(&refined, &caps, rows, cols, k, &baseline, kind);
            if score > best_score {
                best = refined;
                best_score = score;
            }
        }
    }

    let _ = best_score;
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_tracks_backend_expectations() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let choice = fallback(1024, 65_536, 256, &caps, RankKind::TopK);
        assert!(choice.use_2ce);
        assert!(choice.tile >= 1024);
        assert_eq!(choice.mk, 2);
        assert!(choice.fft_tile >= 1024);
        assert_eq!(choice.fft_radix, 4);
        assert!(choice.fft_segments >= 1);
        assert!(choice.subgroup);
    }

    #[test]
    fn unified_rank_prefers_generated_when_available() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let out = choose_unified_rank(512, 16_384, 256, caps, RankKind::TopK);
        assert!(out.wg >= 128);
        assert!(out.tile >= 512);
        assert!(out.fft_tile >= 512);
    }

    #[test]
    fn choice_to_unison_script_includes_core_fields() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let choice = choose_unified_rank(256, 32_768, 64, caps, RankKind::MidK);
        let script = choice.to_unison_script(RankKind::MidK);
        assert!(script.contains("unison midk"));
        assert!(script.contains("workgroup"));
        assert!(script.contains("merge"));
        assert!(script.contains("fft_tile"));
    }

    #[test]
    fn latency_sensitive_topk_avoids_two_stage_and_channel_stride() {
        let caps = DeviceCaps::cuda(32, 1024, Some(64 * 1024));
        let choice = fallback(96, 8_192, 64, &caps, RankKind::TopK);
        assert!(!choice.use_2ce);
        assert_eq!(choice.ch, 0);
        assert!(choice.tile <= 1024);
        assert_eq!(choice.mk, 2);
    }
}
