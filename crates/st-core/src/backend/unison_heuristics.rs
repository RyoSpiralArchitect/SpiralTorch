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
use crate::backend::temporal_fusion::TemporalSpectralFusion;

mod foreign;
use crate::ops::realgrad::GradientSummary;
use crate::telemetry::hub;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JuliaSpan {
    pub start: u32,
    pub step: u32,
    pub end: u32,
}

#[derive(Debug, Clone)]
pub struct JuliaSpanBuf {
    bytes: [u8; JuliaSpan::MAX_RENDERED_LEN],
    len: usize,
}

impl Default for JuliaSpanBuf {
    fn default() -> Self {
        Self {
            bytes: [0; JuliaSpan::MAX_RENDERED_LEN],
            len: 0,
        }
    }
}

impl JuliaSpanBuf {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    fn clear(&mut self) {
        self.len = 0;
    }

    #[inline]
    fn ensure_capacity(&self, additional: usize) {
        debug_assert!(self.len + additional <= self.bytes.len());
    }

    #[inline]
    fn push_colon(&mut self) {
        self.ensure_capacity(1);
        self.bytes[self.len] = b':';
        self.len += 1;
    }

    #[inline]
    fn push_number(&mut self, value: u32) {
        let digits = JuliaSpan::decimal_len(value);
        self.ensure_capacity(digits);
        let end = self.len + digits;
        let start = encode_decimal(value, &mut self.bytes[..], end);
        debug_assert_eq!(start, self.len);
        self.len = end;
    }

    #[inline]
    pub fn render_span<'a>(&'a mut self, span: &JuliaSpan) -> &'a str {
        self.clear();
        self.push_number(span.start);
        self.push_colon();
        self.push_number(span.step);
        self.push_colon();
        self.push_number(span.end);
        self.as_str()
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        // SAFETY: we only ever populate ASCII digits and ':' characters.
        unsafe { std::str::from_utf8_unchecked(&self.bytes[..self.len]) }
    }

    #[inline]
    pub fn to_owned(&self) -> String {
        self.as_str().to_owned()
    }
}

#[inline(always)]
fn encode_decimal(mut value: u32, buf: &mut [u8], end: usize) -> usize {
    let mut idx = end;
    loop {
        idx -= 1;
        buf[idx] = (value % 10) as u8 + b'0';
        value /= 10;
        if value == 0 {
            break;
        }
    }
    idx
}

impl JuliaSpan {
    const MAX_DIGITS: usize = 10; // u32::MAX has 10 decimal digits.
    const MAX_RENDERED_LEN: usize = Self::MAX_DIGITS * 3 + 2;

    #[inline(always)]
    const fn decimal_len(value: u32) -> usize {
        if value >= 1_000_000_000 {
            10
        } else if value >= 100_000_000 {
            9
        } else if value >= 10_000_000 {
            8
        } else if value >= 1_000_000 {
            7
        } else if value >= 100_000 {
            6
        } else if value >= 10_000 {
            5
        } else if value >= 1_000 {
            4
        } else if value >= 100 {
            3
        } else if value >= 10 {
            2
        } else {
            1
        }
    }

    #[inline]
    pub fn new(start: u32, step: u32, end: u32) -> Self {
        Self { start, step, end }
    }

    #[inline]
    pub fn as_tuple(&self) -> (u32, u32, u32) {
        (self.start, self.step, self.end)
    }

    #[inline]
    pub fn render_into<'a>(&self, buf: &'a mut JuliaSpanBuf) -> &'a str {
        buf.render_span(self)
    }

    #[inline]
    pub fn write_into<W: std::fmt::Write>(&self, out: &mut W) -> std::fmt::Result {
        let mut buf = [0u8; Self::MAX_RENDERED_LEN];
        let mut cursor = 0usize;

        let start_digits = Self::decimal_len(self.start);
        let start_end = cursor + start_digits;
        let start_idx = encode_decimal(self.start, &mut buf[..], start_end);
        debug_assert_eq!(start_idx, cursor);
        cursor = start_end;
        buf[cursor] = b':';
        cursor += 1;

        let step_digits = Self::decimal_len(self.step);
        let step_end = cursor + step_digits;
        let step_idx = encode_decimal(self.step, &mut buf[..], step_end);
        debug_assert_eq!(step_idx, cursor);
        cursor = step_end;
        buf[cursor] = b':';
        cursor += 1;

        let end_digits = Self::decimal_len(self.end);
        let end_end = cursor + end_digits;
        let end_idx = encode_decimal(self.end, &mut buf[..], end_end);
        debug_assert_eq!(end_idx, cursor);
        cursor = end_end;

        // SAFETY: the buffer only contains ASCII digits and ':' characters.
        unsafe { out.write_str(std::str::from_utf8_unchecked(&buf[..cursor])) }
    }

    #[inline]
    pub fn to_string_fast(&self) -> String {
        let mut out = String::with_capacity(Self::MAX_RENDERED_LEN);
        self.write_into(&mut out)
            .expect("writing into String should not fail");
        out
    }
}

impl std::fmt::Display for JuliaSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_into(f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneWindow {
    pub target: u32,
    pub lower: u32,
    pub upper: u32,
    pub min_lane: u32,
    pub max_lane: u32,
    pub slack: u32,
    pub stride: u32,
}

impl LaneWindow {
    pub fn julia_span(&self) -> JuliaSpan {
        JuliaSpan::new(self.lower, self.stride, self.upper)
    }

    pub fn julia_span_into<'a>(&self, buf: &'a mut JuliaSpanBuf) -> &'a str {
        self.julia_span().render_into(buf)
    }

    pub fn clamp(&self, value: u32) -> u32 {
        value.max(self.lower).min(self.upper)
    }

    pub fn snapped(&self, value: u32) -> u32 {
        closest_lane_multiple(value, self.stride, self.min_lane, self.max_lane)
    }

    pub fn snapshot(&self) -> LaneWindowSnapshot {
        LaneWindowSnapshot {
            target: self.target,
            lower: self.lower,
            upper: self.upper,
            min_lane: self.min_lane,
            max_lane: self.max_lane,
            slack: self.slack,
            stride: self.stride,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneWindowSnapshot {
    pub target: u32,
    pub lower: u32,
    pub upper: u32,
    pub min_lane: u32,
    pub max_lane: u32,
    pub slack: u32,
    pub stride: u32,
}

impl From<LaneWindow> for LaneWindowSnapshot {
    fn from(window: LaneWindow) -> Self {
        window.snapshot()
    }
}

impl LaneWindowSnapshot {
    #[inline]
    pub fn into_window(self) -> LaneWindow {
        LaneWindow {
            target: self.target,
            lower: self.lower,
            upper: self.upper,
            min_lane: self.min_lane,
            max_lane: self.max_lane,
            slack: self.slack,
            stride: self.stride.max(1),
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
    pub latency_window: Option<LaneWindow>,
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
            if let Some(window) = self.latency_window {
                let span = window.julia_span();
                let _ = writeln!(
                    &mut out,
                    "  #= latency window {} (target {}) =#",
                    span, window.target
                );
            }
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

    /// Returns the latency window expressed as a Julia span without allocating.
    pub fn ctile_julia_span_components(&self) -> Option<JuliaSpan> {
        self.latency_window.map(|window| window.julia_span())
    }

    /// Returns the latency window expressed in Julia-style span syntax.
    /// Prefer [`Self::ctile_julia_span_components`] when you only need the numeric values.
    pub fn ctile_julia_span(&self) -> Option<String> {
        self.ctile_julia_span_components()
            .map(|span| span.to_string_fast())
    }

    /// Writes the latency window span into a reusable scratch buffer.
    #[inline]
    pub fn ctile_julia_span_into<'a>(&self, buf: &'a mut JuliaSpanBuf) -> Option<&'a str> {
        self.latency_window
            .map(|window| window.julia_span().render_into(buf))
    }

    /// Writes the latency window span into the provided formatter.
    /// Returns `Ok(())` even when the latency window is absent.
    pub fn write_ctile_julia_span<W: std::fmt::Write>(&self, mut out: W) -> std::fmt::Result {
        if let Some(span) = self.ctile_julia_span_components() {
            span.write_into(&mut out)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct LatencyCache {
    min_ctile: u32,
    max_ctile: u32,
    slack: u32,
}

#[derive(Clone, Copy)]
struct RankScenario<'a> {
    caps: &'a DeviceCaps,
    rows: u32,
    cols: u32,
    k: u32,
    kind: RankKind,
    lanes: u32,
    low_latency: bool,
    expected_two_stage: bool,
    latency_tile_caps: Option<(u32, u32)>,
    latency: Option<LatencyCache>,
    cols_log2: u32,
}

impl<'a> RankScenario<'a> {
    fn new(rows: u32, cols: u32, k: u32, caps: &'a DeviceCaps, kind: RankKind) -> Self {
        let lanes = caps.lane_width.max(1);
        let low_latency = latency_sensitive(rows, cols, k, caps);
        let expected_two_stage = caps.prefers_two_stage_with_rows(rows, cols, k);
        let latency_tile_caps = if low_latency {
            let latency_cap = if cols <= 16_384 { 512 } else { 1024 };
            let aligned_cap = align_to_lanes(latency_cap, lanes);
            let min_cap = align_to_lanes(128, lanes);
            Some((aligned_cap.max(lanes), min_cap.max(lanes)))
        } else {
            None
        };
        let latency = if matches!(kind, RankKind::MidK | RankKind::BottomK) {
            let (min_ctile, max_ctile) = latency_ctile_bounds(cols, lanes);
            let slack = latency_ctile_slack(rows, cols, k, lanes);
            Some(LatencyCache {
                min_ctile,
                max_ctile,
                slack,
            })
        } else {
            None
        };
        let cols_log2 = if cols <= 1 {
            0
        } else {
            (u32::BITS - 1).saturating_sub(cols.leading_zeros())
        };
        Self {
            caps,
            rows,
            cols,
            k,
            kind,
            lanes,
            low_latency,
            expected_two_stage,
            latency_tile_caps,
            latency,
            cols_log2,
        }
    }

    #[inline]
    fn caps(&self) -> &'a DeviceCaps {
        self.caps
    }

    #[inline]
    fn rows(&self) -> u32 {
        self.rows
    }

    #[inline]
    fn cols(&self) -> u32 {
        self.cols
    }

    #[inline]
    fn k(&self) -> u32 {
        self.k
    }

    #[inline]
    fn kind(&self) -> RankKind {
        self.kind
    }

    #[inline]
    fn lanes(&self) -> u32 {
        self.lanes
    }

    #[inline]
    fn low_latency(&self) -> bool {
        self.low_latency
    }

    #[inline]
    fn expected_two_stage(&self) -> bool {
        self.expected_two_stage
    }

    #[inline]
    fn requires_compaction(&self) -> bool {
        matches!(self.kind, RankKind::MidK | RankKind::BottomK)
    }

    #[inline]
    fn is_bottom(&self) -> bool {
        matches!(self.kind, RankKind::BottomK)
    }

    #[inline]
    fn latency_bounds(&self) -> Option<(u32, u32)> {
        self.latency.map(|cache| (cache.min_ctile, cache.max_ctile))
    }

    #[inline]
    fn latency_slack(&self) -> Option<u32> {
        self.latency.map(|cache| cache.slack)
    }

    #[inline]
    fn latency_tile_caps(&self) -> (u32, u32) {
        self.latency_tile_caps
            .expect("latency tile caps only available in low-latency scenarios")
    }

    #[inline]
    fn bottom_lane_cap(&self, tile: u32, min_ctile: u32) -> u32 {
        let half_tile = tile / 2;
        if half_tile == 0 {
            return min_ctile;
        }
        align_down_to_lanes(half_tile, self.lanes).max(min_ctile)
    }

    #[inline]
    fn snap_latency_ctile(
        &self,
        current: u32,
        min_ctile: u32,
        max_ctile: u32,
    ) -> (u32, LaneWindow) {
        let window = self.tuned_latency_window(min_ctile, max_ctile);
        let mut candidate = window.snapped(current);
        if candidate < window.lower
            || candidate > window.upper
            || candidate.abs_diff(window.target) >= window.slack
        {
            candidate = window.snapped(window.target);
        }
        if candidate < window.lower {
            candidate = window.lower;
        } else if candidate > window.upper {
            candidate = window.upper;
        }
        (candidate, window)
    }

    #[inline]
    fn snap_latency_ctile_snapshot(
        &self,
        current: u32,
        min_ctile: u32,
        max_ctile: u32,
    ) -> (u32, LaneWindowSnapshot) {
        let (candidate, window) = self.snap_latency_ctile(current, min_ctile, max_ctile);
        (candidate, window.snapshot())
    }

    #[inline]
    fn latency_window(&self, min_ctile: u32, max_ctile: u32) -> LaneWindow {
        self.tuned_latency_window(min_ctile, max_ctile)
    }

    #[inline]
    fn latency_window_snapshot(&self, min_ctile: u32, max_ctile: u32) -> LaneWindowSnapshot {
        self.tuned_latency_window(min_ctile, max_ctile).snapshot()
    }

    #[inline]
    fn legacy_latency_target(&self, min_ctile: u32, max_ctile: u32) -> u32 {
        latency_ctile_target_legacy(self.rows, self.k, self.lanes, min_ctile, max_ctile)
    }

    fn tuned_latency_window(&self, min_ctile: u32, max_ctile: u32) -> LaneWindow {
        let slack = self
            .latency_slack()
            .unwrap_or_else(|| latency_ctile_slack(self.rows, self.cols, self.k, self.lanes));
        let mut window = latency_ctile_window_with_slack(
            self.rows, self.cols, self.k, self.lanes, min_ctile, max_ctile, slack,
        );
        let baseline_window = window;
        if let (Some(fusion), Some(pulse)) = (
            TemporalSpectralFusion::analyse(&window, self.rows, self.cols, self.k, self.lanes),
            hub::get_last_realgrad(),
        ) {
            let summary = pulse.gradient_summary();
            if summary.norm > 0.0 {
                let mut tuner = AdaptiveWindowTuner::new(self.lanes);
                window = tuner.tune(window, min_ctile, max_ctile, &fusion, Some(summary));
                if window.slack < baseline_window.slack {
                    let target = window.target;
                    let lower = target
                        .saturating_sub(baseline_window.slack)
                        .max(window.min_lane);
                    let upper = target
                        .saturating_add(baseline_window.slack)
                        .min(window.max_lane);
                    window.lower = lower;
                    window.upper = upper;
                    window.slack = target.abs_diff(lower).max(target.abs_diff(upper));
                }
            }
        }
        window
    }

    #[inline]
    #[cfg_attr(not(test), allow(dead_code))]
    fn cols_log2(&self) -> u32 {
        self.cols_log2
    }

    #[inline]
    fn fft_segments_hint(&self) -> u32 {
        let cols_log2 = self.cols_log2();
        if cols_log2 > 17 || (cols_log2 == 17 && self.cols > 131_072) {
            4
        } else if cols_log2 > 15 || (cols_log2 == 15 && self.cols > 32_768) {
            2
        } else {
            1
        }
    }
}

#[derive(Clone, Debug)]
struct TempoSmoother {
    avg: f32,
    jitter: f32,
}

impl TempoSmoother {
    fn new() -> Self {
        Self {
            avg: 0.0,
            jitter: 0.0,
        }
    }

    fn observe(&mut self, tempo: f32) {
        let tempo = tempo.clamp(0.0, 1.0);
        let alpha = 0.35;
        let prev = self.avg;
        self.avg = (1.0 - alpha) * self.avg + alpha * tempo;
        self.jitter = 0.5 * self.jitter + 0.5 * (tempo - prev).abs();
    }

    fn tempo(&self) -> f32 {
        self.avg
    }

    fn jitter(&self) -> f32 {
        self.jitter
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.avg = 0.0;
        self.jitter = 0.0;
    }
}

#[derive(Clone, Debug)]
struct AdaptiveWindowTuner {
    lanes: u32,
    tempo: TempoSmoother,
    energy_state: f32,
}

impl AdaptiveWindowTuner {
    fn new(lanes: u32) -> Self {
        Self {
            lanes: lanes.max(1),
            tempo: TempoSmoother::new(),
            energy_state: 0.0,
        }
    }

    fn tune(
        &mut self,
        mut window: LaneWindow,
        min_ctile: u32,
        max_ctile: u32,
        fusion: &TemporalSpectralFusion,
        gradient: Option<GradientSummary>,
    ) -> LaneWindow {
        self.tempo.observe(fusion.tempo_hint());
        if let Some(summary) = gradient {
            let norm_pressure = (summary.norm / (summary.norm + 1.0)).clamp(0.0, 1.0);
            self.tempo.observe(norm_pressure);
        }
        let tempo = self.tempo.tempo();
        let jitter = self.tempo.jitter();
        let temporal_density = fusion.temporal().iter().copied().sum::<f32>()
            / (fusion.temporal().len() as f32).max(1.0);
        self.energy_state = 0.6 * self.energy_state
            + 0.4 * (fusion.spectral_energy() / (fusion.temporal().len() as f32).max(1.0));

        let grad_norm = gradient.map(|g| g.norm).unwrap_or(0.0);
        let grad_sparsity = gradient.map(|g| g.sparsity).unwrap_or(0.5);
        let grad_pressure = (grad_norm / (grad_norm + 1.0)).clamp(0.0, 1.0);

        let stride_scale =
            1.0 + tempo * 0.6 + self.energy_state * 0.15 + jitter * 0.25 + grad_pressure * 0.2;
        let base_stride = window.stride.max(1);
        let mut stride = ((base_stride as f32) / stride_scale).round().max(1.0) as u32;
        stride = stride.max(1);
        stride = stride.min(self.lanes.max(1));
        window.stride = stride.max(1);

        let mut slack = ((window.slack.max(1) as f32)
            * (1.0 + tempo * 0.4 + temporal_density * 0.2 - (grad_sparsity - 0.5) * 0.6))
            .round() as u32;
        slack = slack.max(self.lanes);
        slack = align_to_lanes(slack, self.lanes);
        let (min_lane, max_lane) = lane_range(min_ctile, max_ctile, self.lanes);
        let max_span = max_lane.saturating_sub(min_lane).max(self.lanes);
        if slack > max_span {
            slack = align_down_to_lanes(max_span, self.lanes);
        }
        if slack == 0 {
            slack = self.lanes.max(1);
        }

        let gradient_bias = ((max_span as f32) * grad_pressure * 0.3) as u32;
        let mut target = closest_lane_multiple(window.target, self.lanes, min_lane, max_lane);
        if gradient_bias > 0 {
            let desired = target.saturating_add(gradient_bias).min(max_lane);
            target = closest_lane_multiple(desired, self.lanes, min_lane, max_lane);
        }

        let mut lower = target.saturating_sub(slack).max(min_lane);
        let mut upper = target.saturating_add(slack).min(max_lane);
        if lower > upper {
            lower = min_lane;
            upper = min_lane;
        }
        window.target = target;
        window.lower = lower;
        window.upper = upper;
        window.min_lane = min_lane;
        window.max_lane = max_lane;
        window.slack = target.abs_diff(lower).max(target.abs_diff(upper));

        window
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

fn align_to_lanes(value: u32, lanes: u32) -> u32 {
    if lanes <= 1 {
        return value.max(1);
    }
    value.div_ceil(lanes) * lanes
}

fn align_down_to_lanes(value: u32, lanes: u32) -> u32 {
    if lanes <= 1 {
        return value.max(1);
    }
    let aligned = (value / lanes) * lanes;
    if aligned == 0 {
        lanes
    } else {
        aligned
    }
}

fn lane_range(min: u32, max: u32, lanes: u32) -> (u32, u32) {
    if lanes <= 1 {
        let floor = min.max(1);
        return (floor, max.max(floor));
    }
    let lanes = lanes.max(1);
    let min_lane = align_to_lanes(min.max(lanes), lanes);
    let max_lane = align_down_to_lanes(max.max(lanes), lanes);
    if max_lane < min_lane {
        (min_lane, min_lane)
    } else {
        (min_lane, max_lane)
    }
}

fn closest_lane_multiple(value: u32, lanes: u32, min: u32, max: u32) -> u32 {
    if lanes <= 1 {
        return value.clamp(min.max(1), max.max(min.max(1)));
    }

    let (min_lane, max_lane) = lane_range(min, max, lanes);
    let mut base = align_down_to_lanes(value.max(min_lane), lanes);
    if base < min_lane {
        base = min_lane;
    }
    if base > max_lane {
        base = max_lane;
    }

    let mut best = base;
    let mut best_diff = best.abs_diff(value);
    for step in 1i32..=4 {
        let offsets = [step, -step];
        for offset in offsets {
            let candidate_i64 = base as i64 + offset as i64 * lanes as i64;
            if candidate_i64 < min_lane as i64 || candidate_i64 > max_lane as i64 {
                continue;
            }
            let candidate = candidate_i64 as u32;
            let diff = candidate.abs_diff(value);
            if diff < best_diff || (diff == best_diff && candidate < best) {
                best = candidate;
                best_diff = diff;
            }
        }
    }

    best
}

fn latency_ctile_bounds(cols: u32, lanes: u32) -> (u32, u32) {
    let latency_cap = if cols <= 16_384 {
        lanes.saturating_mul(8)
    } else {
        lanes.saturating_mul(16)
    };
    let aligned_cap = align_to_lanes(latency_cap, lanes);
    let min_latency = align_to_lanes(64, lanes);
    (min_latency.max(lanes), aligned_cap.max(lanes))
}

fn latency_ctile_core_target(rows: u32, k: u32, lanes: u32) -> u32 {
    let lanes = lanes.max(1);

    let row_bucket = if rows <= lanes.saturating_mul(2) {
        lanes.saturating_mul(4)
    } else if rows <= lanes.saturating_mul(4) {
        lanes.saturating_mul(6)
    } else if rows <= lanes.saturating_mul(8) {
        lanes.saturating_mul(8)
    } else {
        lanes.saturating_mul(10)
    };

    let k_bucket = if k <= lanes.saturating_mul(2) {
        lanes.saturating_mul(4)
    } else if k <= lanes.saturating_mul(4) {
        lanes.saturating_mul(6)
    } else if k <= lanes.saturating_mul(8) {
        lanes.saturating_mul(8)
    } else {
        lanes.saturating_mul(10)
    };

    row_bucket.max(k_bucket)
}

fn latency_ctile_target(
    rows: u32,
    cols: u32,
    k: u32,
    lanes: u32,
    min_ctile: u32,
    max_ctile: u32,
) -> u32 {
    let lanes = lanes.max(1);

    let base = latency_ctile_core_target(rows, k, lanes);

    let column_bias = if cols <= 4_096 {
        lanes.saturating_mul(3)
    } else if cols <= 16_384 {
        lanes.saturating_mul(4)
    } else if cols <= 65_536 {
        lanes.saturating_mul(6)
    } else {
        lanes.saturating_mul(8)
    };

    let (lower, upper) = if min_ctile <= max_ctile {
        (min_ctile, max_ctile)
    } else {
        (min_ctile, min_ctile)
    };
    let desired = base.max(column_bias).clamp(lower, upper);
    let (min_lane, max_lane) = lane_range(lower, upper, lanes);
    closest_lane_multiple(desired, lanes, min_lane, max_lane)
}

fn latency_ctile_target_legacy(
    rows: u32,
    k: u32,
    lanes: u32,
    min_ctile: u32,
    max_ctile: u32,
) -> u32 {
    let lanes = lanes.max(1);
    let (lower, upper) = if min_ctile <= max_ctile {
        (min_ctile, max_ctile)
    } else {
        (min_ctile, min_ctile)
    };
    let base = latency_ctile_core_target(rows, k, lanes).clamp(lower, upper);
    let (min_lane, max_lane) = lane_range(lower, upper, lanes);
    closest_lane_multiple(base, lanes, min_lane, max_lane)
}

fn latency_ctile_column_slack_range(cols: u32, lanes: u32) -> (u32, u32) {
    let lanes = lanes.max(1);
    if cols <= 4_096 {
        (lanes, lanes.saturating_mul(2))
    } else if cols <= 16_384 {
        (
            lanes.saturating_mul(1),
            lanes.saturating_mul(3).max(lanes.saturating_mul(1)),
        )
    } else if cols <= 65_536 {
        (
            lanes.saturating_mul(2),
            lanes.saturating_mul(4).max(lanes.saturating_mul(2)),
        )
    } else {
        (
            lanes.saturating_mul(3),
            lanes.saturating_mul(5).max(lanes.saturating_mul(3)),
        )
    }
}

fn latency_ctile_slack(rows: u32, cols: u32, k: u32, lanes: u32) -> u32 {
    let lanes = lanes.max(1);
    let tight_rows = rows <= lanes.saturating_mul(2);
    let tight_k = k <= lanes.saturating_mul(2);
    if tight_rows && tight_k {
        let (floor, _) = latency_ctile_column_slack_range(cols, lanes);
        return floor;
    }

    let medium_rows = rows <= lanes.saturating_mul(6);
    let medium_k = k <= lanes.saturating_mul(6);
    let mut slack = if medium_rows && medium_k {
        lanes.saturating_mul(2)
    } else {
        lanes.saturating_mul(3).max(lanes)
    };

    let (floor, ceil) = latency_ctile_column_slack_range(cols, lanes);
    if slack < floor {
        slack = floor;
    }
    if slack > ceil {
        slack = ceil;
    }

    slack
}

#[cfg_attr(not(test), allow(dead_code))]
fn latency_ctile_window(
    rows: u32,
    cols: u32,
    k: u32,
    lanes: u32,
    min_ctile: u32,
    max_ctile: u32,
) -> LaneWindow {
    let slack = latency_ctile_slack(rows, cols, k, lanes);
    latency_ctile_window_with_slack(rows, cols, k, lanes, min_ctile, max_ctile, slack)
}

fn latency_ctile_window_with_slack(
    rows: u32,
    cols: u32,
    k: u32,
    lanes: u32,
    min_ctile: u32,
    max_ctile: u32,
    slack: u32,
) -> LaneWindow {
    let (min_lane, max_lane) = lane_range(min_ctile, max_ctile, lanes);
    let target = latency_ctile_target(rows, cols, k, lanes, min_lane, max_lane);
    let mut lower = align_down_to_lanes(target.saturating_sub(slack), lanes);
    if lower < min_lane {
        lower = min_lane;
    }
    let mut upper = align_to_lanes(target.saturating_add(slack), lanes);
    if upper > max_lane {
        upper = max_lane;
    }
    if lower > upper {
        lower = min_lane;
        upper = min_lane;
    }
    let mut window = LaneWindow {
        target,
        lower,
        upper,
        min_lane,
        max_lane,
        slack,
        stride: lanes.max(1),
    };
    foreign::apply_latency_refinements(
        rows,
        cols,
        k,
        lanes,
        min_ctile,
        max_ctile,
        slack,
        &mut window,
    );
    window
}

#[cfg_attr(not(test), allow(dead_code))]
fn snap_latency_ctile(
    current: u32,
    rows: u32,
    cols: u32,
    k: u32,
    lanes: u32,
    min_ctile: u32,
    max_ctile: u32,
) -> (u32, LaneWindow) {
    let slack = latency_ctile_slack(rows, cols, k, lanes);
    snap_latency_ctile_with_slack(current, rows, cols, k, lanes, min_ctile, max_ctile, slack)
}

#[allow(
    clippy::too_many_arguments,
    reason = "Heuristics signature is stable; refactor is out-of-scope for clippy sweep"
)]
fn snap_latency_ctile_with_slack(
    current: u32,
    rows: u32,
    cols: u32,
    k: u32,
    lanes: u32,
    min_ctile: u32,
    max_ctile: u32,
    slack: u32,
) -> (u32, LaneWindow) {
    let window = latency_ctile_window_with_slack(rows, cols, k, lanes, min_ctile, max_ctile, slack);
    let mut candidate = window.snapped(current);
    if candidate < window.lower
        || candidate > window.upper
        || candidate.abs_diff(window.target) >= window.slack
    {
        candidate = window.snapped(window.target);
    }
    if candidate < window.lower {
        candidate = window.lower;
    }
    if candidate > window.upper {
        candidate = window.upper;
    }
    (candidate, window)
}

#[cfg_attr(not(test), allow(dead_code))]
fn latency_ctile_window_snapshot(
    rows: u32,
    cols: u32,
    k: u32,
    lanes: u32,
    min_ctile: u32,
    max_ctile: u32,
) -> LaneWindowSnapshot {
    let slack = latency_ctile_slack(rows, cols, k, lanes);
    latency_ctile_window_snapshot_with_slack(rows, cols, k, lanes, min_ctile, max_ctile, slack)
}

fn latency_ctile_window_snapshot_with_slack(
    rows: u32,
    cols: u32,
    k: u32,
    lanes: u32,
    min_ctile: u32,
    max_ctile: u32,
    slack: u32,
) -> LaneWindowSnapshot {
    latency_ctile_window_with_slack(rows, cols, k, lanes, min_ctile, max_ctile, slack).snapshot()
}

#[cfg_attr(not(test), allow(dead_code))]
fn snap_latency_ctile_snapshot(
    current: u32,
    rows: u32,
    cols: u32,
    k: u32,
    lanes: u32,
    min_ctile: u32,
    max_ctile: u32,
) -> (u32, LaneWindowSnapshot) {
    let slack = latency_ctile_slack(rows, cols, k, lanes);
    snap_latency_ctile_snapshot_with_slack(
        current, rows, cols, k, lanes, min_ctile, max_ctile, slack,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "Heuristics snapshot helper mirrors runtime signature for consistency"
)]
fn snap_latency_ctile_snapshot_with_slack(
    current: u32,
    rows: u32,
    cols: u32,
    k: u32,
    lanes: u32,
    min_ctile: u32,
    max_ctile: u32,
    slack: u32,
) -> (u32, LaneWindowSnapshot) {
    let (candidate, window) =
        snap_latency_ctile_with_slack(current, rows, cols, k, lanes, min_ctile, max_ctile, slack);
    (candidate, window.snapshot())
}

#[cfg_attr(not(test), allow(dead_code))]
fn fallback(rows: u32, cols: u32, k: u32, caps: &DeviceCaps, kind: RankKind) -> Choice {
    fallback_with_scenario(RankScenario::new(rows, cols, k, caps, kind))
}

fn fallback_with_scenario(scenario: RankScenario<'_>) -> Choice {
    let caps = scenario.caps();
    let wg = caps.recommended_workgroup(scenario.rows());
    let kl = caps.recommended_kl(scenario.k());
    let mut ch = caps.recommended_channel_stride(scenario.cols());
    let (mut tile, mut ctile) = caps.recommended_tiles(scenario.cols());
    let mut latency_window: Option<LaneWindow> = None;

    if scenario.low_latency() {
        ch = 0;
        let (tile_cap, tile_floor) = scenario.latency_tile_caps();
        tile = tile.min(tile_cap).max(tile_floor);
        if scenario.requires_compaction() {
            let (min_ctile, max_ctile) = scenario
                .latency_bounds()
                .expect("latency bounds should exist for compaction");
            let (snapped, window) = scenario.snap_latency_ctile(ctile, min_ctile, max_ctile);
            debug_assert_eq!(
                window.snapshot(),
                scenario.latency_window_snapshot(min_ctile, max_ctile)
            );
            let (legacy_snapped, legacy_window) =
                scenario.snap_latency_ctile_snapshot(ctile, min_ctile, max_ctile);
            debug_assert_eq!(legacy_snapped, snapped);
            debug_assert_eq!(legacy_window.target, window.target);
            ctile = snapped;
            latency_window = Some(window);
        }
    }

    if scenario.is_bottom() {
        let half_tile = tile / 2;
        ctile = ctile.min(half_tile.max(128)).max(128);
        if half_tile > 0 {
            let lane_cap = align_down_to_lanes(half_tile, scenario.lanes()).max(128);
            ctile = ctile.min(lane_cap);
        }
        if scenario.low_latency() {
            if let Some((min_ctile, _)) = scenario.latency_bounds() {
                let lane_cap = scenario.bottom_lane_cap(tile, min_ctile);
                let (snapped, window) = scenario.snap_latency_ctile(
                    ctile.min(lane_cap).max(min_ctile),
                    min_ctile,
                    lane_cap.max(min_ctile),
                );
                debug_assert_eq!(
                    window.snapshot(),
                    scenario.latency_window_snapshot(min_ctile, lane_cap.max(min_ctile))
                );
                let (legacy_snapped, legacy_window) = scenario.snap_latency_ctile_snapshot(
                    ctile.min(lane_cap).max(min_ctile),
                    min_ctile,
                    lane_cap.max(min_ctile),
                );
                debug_assert_eq!(legacy_snapped, snapped);
                debug_assert_eq!(legacy_window.target, window.target);
                ctile = snapped;
                latency_window = Some(window);
            }
        }
    }

    let mut mk = caps.preferred_merge_kind(scenario.k());
    let mut mkd = caps.preferred_substrategy(mk, scenario.k());
    if scenario.low_latency()
        && caps.subgroup
        && matches!(scenario.kind(), RankKind::TopK)
        && scenario.k() <= scenario.lanes().saturating_mul(8)
    {
        mk = 2;
        mkd = if scenario.k() <= 128 { 4 } else { 5 };
    }
    let mut use_2ce = scenario.expected_two_stage();
    if scenario.low_latency() {
        use_2ce = false;
    }

    let fft_tile = align_to_lanes(scenario.cols().max(1), 1024);
    let fft_radix = if scenario.k().is_power_of_two() { 4 } else { 2 };
    let fft_segments = scenario.fft_segments_hint();

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
        latency_window,
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

#[derive(Debug, Clone, Copy)]
pub struct TempoFeedback {
    pub choice: Choice,
    pub latency_window: Option<LaneWindow>,
    pub weight: f32,
}

impl TempoFeedback {
    pub fn new(choice: Choice) -> Self {
        Self {
            latency_window: choice.latency_window,
            choice,
            weight: 1.0,
        }
    }

    pub fn with_latency(mut self, latency_window: Option<LaneWindow>) -> Self {
        self.latency_window = latency_window;
        self
    }

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }
}

#[derive(Debug, Clone)]
pub struct TempoLearner {
    baseline: Choice,
    aggregate: WeightedChoice,
    best_latency: Option<(f32, LaneWindow)>,
    total_weight: f32,
}

impl TempoLearner {
    pub fn new(baseline: Choice) -> Self {
        let best_latency = baseline.latency_window.map(|window| (0.0f32, window));
        Self {
            baseline,
            aggregate: WeightedChoice::default(),
            best_latency,
            total_weight: 0.0,
        }
    }

    pub fn observe(&mut self, feedback: TempoFeedback) {
        let weight = feedback.weight.max(0.0);
        if weight == 0.0 {
            return;
        }

        self.aggregate.accumulate(&feedback.choice, weight);
        self.total_weight += weight;

        if let Some(window) = feedback.latency_window.or(feedback.choice.latency_window) {
            match &mut self.best_latency {
                Some((best_weight, best_window)) => {
                    if weight > *best_weight {
                        *best_weight = weight;
                        *best_window = window;
                    } else if (weight - *best_weight).abs() <= f32::EPSILON {
                        *best_window = intersect_latency(*best_window, window);
                    }
                }
                None => self.best_latency = Some((weight, window)),
            }
        }
    }

    pub fn merge(&mut self, other: &TempoLearner) {
        if other.total_weight == 0.0 {
            return;
        }

        self.aggregate.combine(&other.aggregate);
        self.total_weight += other.total_weight;

        if let Some((weight, window)) = other.best_latency {
            match &mut self.best_latency {
                Some((best_weight, best_window)) => {
                    if weight > *best_weight {
                        *best_weight = weight;
                        *best_window = window;
                    }
                }
                None => self.best_latency = Some((weight, window)),
            }
        }
    }

    pub fn into_choice(self) -> Choice {
        let mut choice = if self.total_weight > 0.0 {
            self.aggregate.finalize(self.baseline, self.total_weight)
        } else {
            self.baseline
        };

        if let Some((_, window)) = self.best_latency {
            choice.ctile = window.clamp(choice.ctile);
            choice.latency_window = Some(window);
        }

        choice
    }
}

#[derive(Debug, Default, Clone)]
struct WeightedChoice {
    use_2ce: f32,
    subgroup: f32,
    wg: f32,
    kl: f32,
    ch: f32,
    mk: f32,
    mkd: f32,
    tile: f32,
    ctile: f32,
    fft_tile: f32,
    fft_radix: f32,
    fft_segments: f32,
}

impl WeightedChoice {
    fn accumulate(&mut self, choice: &Choice, weight: f32) {
        if choice.use_2ce {
            self.use_2ce += weight;
        }
        if choice.subgroup {
            self.subgroup += weight;
        }
        self.wg += choice.wg as f32 * weight;
        self.kl += choice.kl as f32 * weight;
        self.ch += choice.ch as f32 * weight;
        self.mk += choice.mk as f32 * weight;
        self.mkd += choice.mkd as f32 * weight;
        self.tile += choice.tile as f32 * weight;
        self.ctile += choice.ctile as f32 * weight;
        self.fft_tile += choice.fft_tile as f32 * weight;
        self.fft_radix += choice.fft_radix as f32 * weight;
        self.fft_segments += choice.fft_segments as f32 * weight;
    }

    fn combine(&mut self, other: &WeightedChoice) {
        self.use_2ce += other.use_2ce;
        self.subgroup += other.subgroup;
        self.wg += other.wg;
        self.kl += other.kl;
        self.ch += other.ch;
        self.mk += other.mk;
        self.mkd += other.mkd;
        self.tile += other.tile;
        self.ctile += other.ctile;
        self.fft_tile += other.fft_tile;
        self.fft_radix += other.fft_radix;
        self.fft_segments += other.fft_segments;
    }

    fn finalize(self, baseline: Choice, total_weight: f32) -> Choice {
        if total_weight <= 0.0 {
            return baseline;
        }

        let inv = 1.0 / total_weight;
        let mut choice = baseline;

        choice.use_2ce = self.use_2ce * inv >= 0.5;
        choice.subgroup = self.subgroup * inv >= 0.5;
        choice.wg = round_u32(self.wg * inv, baseline.wg);
        choice.kl = round_u32(self.kl * inv, baseline.kl);
        choice.ch = round_u32(self.ch * inv, baseline.ch);
        choice.mk = round_u32(self.mk * inv, baseline.mk);
        choice.mkd = round_u32(self.mkd * inv, baseline.mkd);
        choice.tile = round_u32(self.tile * inv, baseline.tile);
        choice.ctile = round_u32(self.ctile * inv, baseline.ctile);
        choice.fft_tile = round_u32(self.fft_tile * inv, baseline.fft_tile);
        choice.fft_radix = round_u32(self.fft_radix * inv, baseline.fft_radix);
        choice.fft_segments = round_u32(self.fft_segments * inv, baseline.fft_segments);

        choice
    }
}

fn round_u32(value: f32, fallback: u32) -> u32 {
    if !value.is_finite() {
        return fallback;
    }
    let rounded = value.round();
    if rounded <= 0.0 {
        0
    } else if rounded >= u32::MAX as f32 {
        u32::MAX
    } else {
        rounded as u32
    }
}

fn intersect_latency(a: LaneWindow, b: LaneWindow) -> LaneWindow {
    let lower = a.lower.max(b.lower);
    let upper = a.upper.min(b.upper);
    let target = a.target.clamp(lower, upper);
    LaneWindow {
        target,
        lower,
        upper,
        min_lane: a.min_lane.max(b.min_lane),
        max_lane: a.max_lane.min(b.max_lane),
        slack: a.slack.min(b.slack),
        stride: a.stride.max(b.stride),
    }
}

fn refine_choice(mut choice: Choice, baseline: Choice, scenario: RankScenario<'_>) -> Choice {
    let caps = scenario.caps();
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
        choice.mkd = caps.preferred_substrategy(choice.mk, scenario.k());
    }
    if !choice.subgroup {
        choice.subgroup = baseline.subgroup;
    }
    if choice.tile == 0 {
        choice.tile = baseline.tile;
    }
    choice.tile = caps.preferred_tile(scenario.cols(), choice.tile);

    if scenario.low_latency() {
        choice.use_2ce = false;
        if baseline.ch == 0 {
            choice.ch = 0;
        }
        if choice.tile > baseline.tile {
            choice.tile = baseline.tile;
        }
    }

    if scenario.requires_compaction() {
        if choice.ctile == 0 {
            choice.ctile = baseline.ctile;
        }
        choice.ctile = caps.preferred_compaction_tile(scenario.cols(), choice.ctile);
        if scenario.is_bottom() {
            let half_tile = choice.tile / 2;
            if half_tile > 0 {
                let lane_cap = align_down_to_lanes(half_tile, scenario.lanes()).max(128);
                choice.ctile = choice.ctile.min(lane_cap);
            }
            choice.ctile = choice.ctile.min((choice.tile / 2).max(128)).max(128);
        }
        if scenario.low_latency() {
            let (min_ctile, max_ctile) = scenario
                .latency_bounds()
                .expect("latency bounds should exist for compaction");
            if choice.ctile > baseline.ctile {
                choice.ctile = baseline.ctile;
            }
            let mut upper = max_ctile;
            if scenario.is_bottom() {
                let half_tile = choice.tile / 2;
                if half_tile > 0 {
                    upper =
                        upper.min(align_down_to_lanes(half_tile, scenario.lanes()).max(min_ctile));
                }
            }
            let (snapped, window) =
                scenario.snap_latency_ctile(choice.ctile, min_ctile, upper.max(min_ctile));
            choice.ctile = snapped;
            if choice.ctile != window.target {
                choice.ctile = window.target;
            }
            if choice.ctile < window.lower {
                choice.ctile = window.lower;
            }
            if choice.ctile > window.upper {
                choice.ctile = window.upper;
            }
            choice.latency_window = Some(window);
        } else {
            choice.latency_window = None;
        }
    } else {
        choice.ctile = 0;
        choice.latency_window = None;
    }

    if choice.fft_tile == 0 {
        choice.fft_tile = baseline.fft_tile;
    }
    choice.fft_tile = caps.preferred_tile(scenario.cols(), choice.fft_tile);

    if choice.fft_radix == 0 {
        choice.fft_radix = baseline.fft_radix;
    }
    choice.fft_radix = choice.fft_radix.clamp(2, 4);

    if choice.fft_segments == 0 {
        choice.fft_segments = baseline.fft_segments;
    }
    choice.fft_segments = choice.fft_segments.clamp(1, 4);

    let expected_two_stage = scenario.expected_two_stage();
    if expected_two_stage && !choice.use_2ce {
        choice.use_2ce = baseline.use_2ce;
    }
    if !expected_two_stage {
        choice.use_2ce = false;
    }

    if scenario.low_latency() {
        choice.use_2ce = false;
        if baseline.ch == 0 {
            choice.ch = 0;
        }
    }

    enforce_shared_memory(&mut choice, caps, scenario.k());

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

fn score_choice(choice: &Choice, baseline: &Choice, scenario: RankScenario<'_>) -> f32 {
    let caps = scenario.caps();
    let mut score = 0.0;

    if choice.use_2ce == scenario.expected_two_stage() {
        score += 0.25;
    } else {
        score -= 0.1;
    }

    score += closeness(choice.wg, baseline.wg) * 0.2;
    score += caps.occupancy_score(choice.wg) * 0.2;

    score += closeness(choice.kl, baseline.kl) * 0.15;
    score += closeness(choice.tile, baseline.tile) * 0.1;

    if scenario.requires_compaction() {
        score += closeness(choice.ctile, baseline.ctile) * 0.1;
    }

    score += closeness(choice.fft_tile, baseline.fft_tile) * 0.05;
    if choice.fft_radix == baseline.fft_radix {
        score += 0.025;
    }
    if choice.fft_segments == baseline.fft_segments {
        score += 0.025;
    }

    if choice.mk == caps.preferred_merge_kind(scenario.k()) {
        score += 0.1;
    }
    if choice.mkd == caps.preferred_substrategy(choice.mk, scenario.k()) {
        score += 0.1;
    }

    if scenario.low_latency() {
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
        let (tile_cap, _) = scenario.latency_tile_caps();
        score += closeness(choice.tile, tile_cap) * 0.05;
        if choice.mk == 2 {
            score += 0.025;
        }
        if scenario.requires_compaction() {
            let (min_ctile, max_ctile) = scenario
                .latency_bounds()
                .expect("latency bounds should exist for compaction");
            let window = choice
                .latency_window
                .unwrap_or_else(|| scenario.latency_window(min_ctile, max_ctile));
            score += closeness(choice.ctile, window.target) * 0.05;
            let snapped = window.snapped(choice.ctile);
            if snapped == choice.ctile {
                score += 0.02;
            } else {
                score -= 0.02;
            }
            if choice.ctile >= window.lower && choice.ctile <= window.upper {
                score += 0.015;
            } else {
                score -= 0.03;
            }
            if choice.ctile.abs_diff(window.target) <= window.slack {
                score += 0.01;
            } else {
                score -= 0.015;
            }
            let legacy_target = scenario.legacy_latency_target(min_ctile, max_ctile);
            score += closeness(choice.ctile, legacy_target) * 0.02;
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
        latency_window: None,
    }
}

pub fn choose_unified_rank(
    rows: u32,
    cols: u32,
    k: u32,
    caps: DeviceCaps,
    kind: RankKind,
) -> Choice {
    let scenario = RankScenario::new(rows, cols, k, &caps, kind);
    let baseline = fallback_with_scenario(scenario);
    let mut best = baseline;
    let mut best_score = score_choice(&baseline, &baseline, scenario);

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
        let refined = refine_choice(convert_wgpu_choice(hard, caps.subgroup), baseline, scenario);
        let score = score_choice(&refined, &baseline, scenario);
        if score > best_score {
            best = refined;
            best_score = score;
        }
    }

    if let Some(kv) = kdsl_bridge::choose_from_kv(rows, cols, k, caps.subgroup) {
        let refined = refine_choice(convert_wgpu_choice(kv, caps.subgroup), baseline, scenario);
        let score = score_choice(&refined, &baseline, scenario);
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
                scenario,
            );
            let score = score_choice(&refined, &baseline, scenario);
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
    use crate::telemetry::hub;

    #[test]
    fn tuned_latency_window_responds_to_gradient_feedback() {
        hub::clear_last_realgrad_for_test();
        let pulse = hub::RealGradPulse {
            gradient_norm: 64.0,
            gradient_sparsity: 0.1,
            ..Default::default()
        };
        hub::set_last_realgrad(&pulse);
        let caps = DeviceCaps::wgpu(32, true, 256);
        let scenario = RankScenario::new(64, 4_096, 48, &caps, RankKind::MidK);
        let (min_ctile, max_ctile) = scenario.latency_bounds().expect("latency bounds available");
        let window = scenario.latency_window(min_ctile, max_ctile);
        let baseline_slack = scenario.latency_slack().expect("latency slack available");
        assert!(window.slack >= baseline_slack);
        hub::clear_last_realgrad_for_test();
    }

    #[derive(Default)]
    struct CountingWriter {
        writes: usize,
        chars: usize,
        buf: String,
    }

    impl std::fmt::Write for CountingWriter {
        fn write_str(&mut self, s: &str) -> std::fmt::Result {
            self.writes += 1;
            self.buf.push_str(s);
            Ok(())
        }

        fn write_char(&mut self, c: char) -> std::fmt::Result {
            self.chars += 1;
            self.buf.push(c);
            Ok(())
        }
    }

    fn legacy_score_choice(choice: &Choice, baseline: &Choice, scenario: RankScenario<'_>) -> f32 {
        let caps = scenario.caps();
        let mut score = 0.0;

        let expected_two_stage = scenario.expected_two_stage();
        if choice.use_2ce == expected_two_stage {
            score += 0.25;
        } else {
            score -= 0.1;
        }

        score += closeness(choice.wg, baseline.wg) * 0.2;
        score += caps.occupancy_score(choice.wg) * 0.2;

        score += closeness(choice.kl, baseline.kl) * 0.15;
        score += closeness(choice.tile, baseline.tile) * 0.1;

        if scenario.requires_compaction() {
            score += closeness(choice.ctile, baseline.ctile) * 0.1;
        }

        score += closeness(choice.fft_tile, baseline.fft_tile) * 0.05;
        if choice.fft_radix == baseline.fft_radix {
            score += 0.025;
        }
        if choice.fft_segments == baseline.fft_segments {
            score += 0.025;
        }

        if choice.mk == caps.preferred_merge_kind(scenario.k()) {
            score += 0.1;
        }
        if choice.mkd == caps.preferred_substrategy(choice.mk, scenario.k()) {
            score += 0.1;
        }

        if scenario.low_latency() {
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
            let lanes = scenario.lanes();
            let latency_cap = if scenario.cols() <= 16_384 { 512 } else { 1024 };
            let aligned_cap = align_to_lanes(latency_cap, lanes);
            score += closeness(choice.tile, aligned_cap) * 0.05;
            if choice.mk == 2 {
                score += 0.025;
            }
            if scenario.requires_compaction() {
                let (min_ctile, max_ctile) = scenario
                    .latency_bounds()
                    .expect("latency bounds should exist for compaction");
                let window = choice
                    .latency_window
                    .unwrap_or_else(|| scenario.latency_window(min_ctile, max_ctile));
                score += closeness(choice.ctile, window.target) * 0.05;
                let snapped = window.snapped(choice.ctile);
                if snapped == choice.ctile {
                    score += 0.02;
                } else {
                    score -= 0.02;
                }
                if choice.ctile >= window.lower && choice.ctile <= window.upper {
                    score += 0.015;
                } else {
                    score -= 0.03;
                }
                if choice.ctile.abs_diff(window.target) <= window.slack {
                    score += 0.01;
                } else {
                    score -= 0.015;
                }
            }
        }

        score
    }

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
    fn latency_span_uses_julia_notation() {
        let caps = DeviceCaps::cuda(32, 1024, Some(64 * 1024));
        let choice = fallback(96, 8_192, 64, &caps, RankKind::MidK);
        let window = choice
            .latency_window
            .expect("latency window should be captured in fallback");
        let span_components = choice
            .ctile_julia_span_components()
            .expect("latency span should be available");
        let mut scratch = JuliaSpanBuf::new();
        let rendered = span_components.render_into(&mut scratch);
        let span = rendered.to_owned();
        assert!(span.matches(':').count() >= 2);
        assert!(span.split(':').all(|part| !part.is_empty()));
        assert_eq!(span_components.as_tuple().0, window.lower);
        let script = choice.to_unison_script(RankKind::MidK);
        assert!(script.contains("#= latency window"));
        assert!(script.contains(&span));
        let mut buf = String::new();
        choice
            .write_ctile_julia_span(&mut buf)
            .expect("writing julia span should succeed");
        assert_eq!(buf, span);
        assert!(!buf.is_empty());
        let mut window_scratch = JuliaSpanBuf::new();
        let reused = window.julia_span_into(&mut window_scratch).to_owned();
        assert_eq!(reused, span);
        let mut choice_scratch = JuliaSpanBuf::new();
        let from_choice = choice
            .ctile_julia_span_into(&mut choice_scratch)
            .expect("span should be renderable into scratch");
        assert_eq!(from_choice, span);
    }

    #[test]
    fn julia_span_buffer_matches_display() {
        let span = JuliaSpan::new(128, 32, 640);
        let mut buf = JuliaSpanBuf::new();
        let rendered = span.render_into(&mut buf).to_owned();
        assert_eq!(rendered, "128:32:640");
        assert_eq!(rendered, span.to_string());
        let mut second = JuliaSpanBuf::new();
        let reused = span.render_into(&mut second).to_owned();
        assert_eq!(rendered, reused);
        assert_eq!(buf.as_str(), rendered);
        let mut direct = String::new();
        span.write_into(&mut direct)
            .expect("writing into a String should succeed");
        assert_eq!(direct, rendered);
    }

    #[test]
    fn julia_span_write_into_is_single_chunk() {
        let span = JuliaSpan::new(1_024, 16, 8_192);
        let mut writer = CountingWriter::default();
        span.write_into(&mut writer)
            .expect("writing into counting writer should succeed");
        assert_eq!(writer.writes, 1);
        assert_eq!(writer.chars, 0);
        assert_eq!(writer.buf, "1024:16:8192");

        // Ensure reusing the stack buffer does not allocate by rendering twice.
        span.write_into(&mut writer)
            .expect("rewriting into counting writer should succeed");
        assert_eq!(writer.writes, 2);
        assert_eq!(writer.chars, 0);
        assert_eq!(writer.buf, "1024:16:81921024:16:8192");
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

    #[test]
    fn latency_sensitive_midk_limits_compaction_tile() {
        let caps = DeviceCaps::cuda(32, 1024, Some(64 * 1024));
        let choice = fallback(96, 8_192, 64, &caps, RankKind::MidK);
        assert!(!choice.use_2ce);
        assert_eq!(choice.ch, 0);
        let lanes = caps.lane_width.max(1);
        let (min_ctile, max_ctile) = latency_ctile_bounds(8_192, lanes);
        let expected = latency_ctile_target(96, 8_192, 64, lanes, min_ctile, max_ctile);
        assert_eq!(choice.ctile, expected);
        assert!(choice.latency_window.is_some());
    }

    #[test]
    fn latency_sensitive_bottomk_respects_lane_alignment() {
        let caps = DeviceCaps::cuda(32, 1024, Some(64 * 1024));
        let choice = fallback(96, 8_192, 64, &caps, RankKind::BottomK);
        assert!(!choice.use_2ce);
        assert_eq!(choice.ch, 0);
        let lanes = caps.lane_width.max(1);
        let half_tile = choice.tile / 2;
        let lane_cap = if half_tile == 0 {
            lanes
        } else {
            align_down_to_lanes(half_tile, lanes)
        };
        assert!(choice.ctile <= lane_cap.max(128));
        let (min_ctile, _max_ctile) = latency_ctile_bounds(8_192, lanes);
        let window = latency_ctile_window(96, 8_192, 64, lanes, min_ctile, lane_cap.max(min_ctile));
        assert!(choice.ctile >= window.lower);
        assert!(choice.ctile <= window.upper);
        let snapped = window.snapped(choice.ctile);
        assert_eq!(choice.ctile, snapped);
        assert_eq!(choice.ctile, window.target);
    }

    #[test]
    fn latency_sensitive_midk_refine_nudges_back_to_target() {
        let caps = DeviceCaps::cuda(32, 1024, Some(64 * 1024));
        let baseline = fallback(64, 4_096, 48, &caps, RankKind::MidK);
        let mut candidate = baseline;
        candidate.ctile = align_to_lanes(baseline.ctile.saturating_mul(2), caps.lane_width);
        let scenario = RankScenario::new(64, 4_096, 48, &caps, RankKind::MidK);
        let refined = refine_choice(candidate, baseline, scenario);
        assert_eq!(refined.ctile, baseline.ctile);
    }

    #[test]
    fn snap_latency_ctile_respects_window() {
        let caps = DeviceCaps::cuda(16, 1024, Some(64 * 1024));
        let lanes = caps.lane_width.max(1);
        let (min_ctile, max_ctile) = latency_ctile_bounds(4_096, lanes);
        let (snapped, window) =
            snap_latency_ctile(2048, 24, 4_096, 32, lanes, min_ctile, max_ctile);
        let window_check = latency_ctile_window(24, 4_096, 32, lanes, min_ctile, max_ctile);
        assert!(snapped >= window_check.lower && snapped <= window_check.upper);
        let lane_snapped = window.snapped(snapped);
        assert_eq!(snapped, lane_snapped);
        assert_eq!(snapped, window.target);
        assert!(snapped.abs_diff(window.target) <= window.slack);
    }

    #[test]
    fn bottomk_refine_retains_lane_cap() {
        let caps = DeviceCaps::cuda(32, 1024, Some(64 * 1024));
        let baseline = fallback(80, 16_384, 96, &caps, RankKind::BottomK);
        let mut candidate = baseline;
        candidate.ctile = baseline.ctile.saturating_add(256);
        let scenario = RankScenario::new(80, 16_384, 96, &caps, RankKind::BottomK);
        let refined = refine_choice(candidate, baseline, scenario);
        let lanes = caps.lane_width.max(1);
        let half_tile = refined.tile / 2;
        let lane_cap = if half_tile == 0 {
            lanes
        } else {
            align_down_to_lanes(half_tile, lanes)
        };
        assert!(refined.ctile <= lane_cap.max(128));
    }

    #[test]
    fn latency_ctile_slack_scales_with_rows_and_k() {
        let lanes = 32;
        let tight = latency_ctile_slack(32, 8_192, 32, lanes);
        let medium = latency_ctile_slack(160, 8_192, 160, lanes);
        let wide = latency_ctile_slack(512, 8_192, 768, lanes);
        assert_eq!(tight, latency_ctile_column_slack_range(8_192, lanes).0);
        assert!(medium >= lanes.saturating_mul(2));
        assert!(wide >= lanes.saturating_mul(3));
        assert!(tight < medium && medium <= wide);
    }

    #[test]
    fn latency_ctile_slack_respects_column_tiers() {
        let lanes = 32;
        let small = latency_ctile_slack(96, 4_096, 64, lanes);
        let medium = latency_ctile_slack(96, 16_384, 64, lanes);
        let huge = latency_ctile_slack(96, 131_072, 64, lanes);
        assert_eq!(small, lanes.saturating_mul(2));
        assert_eq!(medium, lanes.saturating_mul(2));
        assert_eq!(huge, lanes.saturating_mul(3));
        assert!(huge > medium);
    }

    #[test]
    fn latency_refine_clamps_to_window_bounds() {
        let caps = DeviceCaps::cuda(32, 1024, Some(64 * 1024));
        let baseline = fallback(48, 8_192, 64, &caps, RankKind::MidK);
        let mut candidate = baseline;
        candidate.ctile = baseline.ctile.saturating_add(1024);
        let scenario = RankScenario::new(48, 8_192, 64, &caps, RankKind::MidK);
        let refined = refine_choice(candidate, baseline, scenario);
        let lanes = caps.lane_width.max(1);
        let (min_ctile, max_ctile) = latency_ctile_bounds(8_192, lanes);
        let window = latency_ctile_window(48, 8_192, 64, lanes, min_ctile, max_ctile);
        assert!(refined.ctile >= window.lower && refined.ctile <= window.upper);
    }

    #[test]
    fn latency_score_rewards_window_alignment() {
        let caps = DeviceCaps::cuda(32, 1024, Some(64 * 1024));
        let baseline = fallback(64, 8_192, 48, &caps, RankKind::BottomK);
        let mut aligned = baseline;
        let mut off = baseline;
        aligned.ctile = baseline.ctile;
        off.ctile = baseline.ctile.saturating_add(caps.lane_width * 4);
        let scenario = RankScenario::new(64, 8_192, 48, &caps, RankKind::BottomK);
        let aligned_score = score_choice(&aligned, &baseline, scenario);
        let off_score = score_choice(&off, &baseline, scenario);
        assert!(aligned_score > off_score);
    }

    #[test]
    fn legacy_target_matches_precolumn_formula() {
        let lanes = 32;
        let (min_ctile, max_ctile) = latency_ctile_bounds(8_192, lanes);
        let legacy = latency_ctile_target_legacy(96, 64, lanes, min_ctile, max_ctile);

        let row_bucket = if 96 <= lanes.saturating_mul(2) {
            lanes.saturating_mul(4)
        } else if 96 <= lanes.saturating_mul(4) {
            lanes.saturating_mul(6)
        } else if 96 <= lanes.saturating_mul(8) {
            lanes.saturating_mul(8)
        } else {
            lanes.saturating_mul(10)
        };

        let k_bucket = if 64 <= lanes.saturating_mul(2) {
            lanes.saturating_mul(4)
        } else if 64 <= lanes.saturating_mul(4) {
            lanes.saturating_mul(6)
        } else if 64 <= lanes.saturating_mul(8) {
            lanes.saturating_mul(8)
        } else {
            lanes.saturating_mul(10)
        };

        let desired = row_bucket.max(k_bucket).clamp(min_ctile, max_ctile);
        let (min_lane, max_lane) = lane_range(min_ctile, max_ctile, lanes);
        let expected = closest_lane_multiple(desired, lanes, min_lane, max_lane);
        assert_eq!(legacy, expected);
    }

    #[test]
    fn lane_window_snapshot_round_trips() {
        let caps = DeviceCaps::cuda(32, 1024, Some(64 * 1024));
        let lanes = caps.lane_width.max(1);
        let (min_ctile, max_ctile) = latency_ctile_bounds(8_192, lanes);
        let window = latency_ctile_window(96, 8_192, 64, lanes, min_ctile, max_ctile);
        let snapshot = latency_ctile_window_snapshot(96, 8_192, 64, lanes, min_ctile, max_ctile);
        assert_eq!(snapshot.target, window.target);
        assert_eq!(snapshot.lower, window.lower);
        assert_eq!(snapshot.upper, window.upper);
        assert_eq!(snapshot.min_lane, window.min_lane);
        assert_eq!(snapshot.max_lane, window.max_lane);
        assert_eq!(snapshot.slack, window.slack);
        assert_eq!(snapshot.stride, window.stride);

        let restored = snapshot.into_window();
        assert_eq!(restored, window);

        let (snapped, snap_snapshot) =
            snap_latency_ctile_snapshot(window.target, 96, 8_192, 64, lanes, min_ctile, max_ctile);
        assert_eq!(snapped, window.target);
        assert_eq!(snap_snapshot.target, window.target);
        assert_eq!(snap_snapshot.lower, window.lower);
        assert_eq!(snap_snapshot.upper, window.upper);
        assert_eq!(snap_snapshot.stride, window.stride);
        assert_eq!(snap_snapshot.into_window(), window);
    }

    #[test]
    fn lane_window_snapshot_identity_extremes() {
        let windows = [
            LaneWindow {
                target: 0,
                lower: 0,
                upper: 0,
                min_lane: 0,
                max_lane: 0,
                slack: 0,
                stride: 1,
            },
            LaneWindow {
                target: 256,
                lower: 256,
                upper: 256,
                min_lane: 256,
                max_lane: 256,
                slack: 0,
                stride: 32,
            },
            LaneWindow {
                target: 2_048,
                lower: 1_024,
                upper: 3_072,
                min_lane: 1_024,
                max_lane: 3_072,
                slack: 1_024,
                stride: 64,
            },
        ];

        for window in windows {
            let snapshot = window.snapshot();
            assert_eq!(snapshot.into_window(), window);
            assert_eq!(snapshot.stride, window.stride);
        }
    }

    #[test]
    fn lane_window_snapshot_zero_stride_promotes_to_one() {
        let snapshot = LaneWindowSnapshot {
            target: 512,
            lower: 512,
            upper: 512,
            min_lane: 512,
            max_lane: 512,
            slack: 0,
            stride: 0,
        };
        let restored = snapshot.into_window();
        assert_eq!(restored.stride, 1);
        assert_eq!(restored.target, 512);
        assert_eq!(restored.lower, 512);
        assert_eq!(restored.upper, 512);
    }

    #[test]
    fn scenario_score_tracks_legacy_within_expected_band() {
        let caps = DeviceCaps::cuda(32, 1024, Some(64 * 1024));
        let scenarios = [
            (RankKind::TopK, 64, 4_096, 64),
            (RankKind::MidK, 80, 16_384, 96),
            (RankKind::BottomK, 48, 8_192, 32),
        ];
        const EPS: f32 = 1e-3;

        for (kind, rows, cols, k) in scenarios {
            let scenario = RankScenario::new(rows, cols, k, &caps, kind);
            let baseline = fallback_with_scenario(scenario);
            let lanes = scenario.lanes();
            assert!(scenario.cols_log2() <= 31);

            let mut variations = [baseline; 6];
            variations[1].wg = baseline.wg.saturating_add(32);
            variations[2].tile = baseline.tile.saturating_add(lanes);
            variations[3].use_2ce = !baseline.use_2ce;
            variations[4].fft_tile = baseline.fft_tile.saturating_add(1024);
            if scenario.requires_compaction() {
                variations[5].ctile = baseline.ctile.saturating_add(lanes);
            }

            for choice in variations {
                let new_score = score_choice(&choice, &baseline, scenario);
                let legacy_score = legacy_score_choice(&choice, &baseline, scenario);
                assert!(new_score.is_finite() && legacy_score.is_finite());
                assert!(new_score + EPS >= legacy_score);
                assert!((new_score - legacy_score).abs() <= 0.020001 + EPS);
            }
        }
    }

    fn sample_window(target: u32, lower: u32, upper: u32, stride: u32) -> LaneWindow {
        LaneWindow {
            target,
            lower,
            upper,
            min_lane: lower,
            max_lane: upper,
            slack: upper.saturating_sub(lower),
            stride,
        }
    }

    fn sample_choice() -> Choice {
        Choice {
            use_2ce: false,
            wg: 128,
            kl: 32,
            ch: 0,
            mk: 2,
            mkd: 4,
            tile: 1024,
            ctile: 256,
            subgroup: true,
            fft_tile: 1024,
            fft_radix: 4,
            fft_segments: 1,
            latency_window: Some(sample_window(256, 128, 512, 32)),
        }
    }

    #[test]
    fn tempo_learner_averages_weighted_feedback() {
        let baseline = sample_choice();
        let mut learner = TempoLearner::new(baseline);

        let mut rich = baseline;
        rich.use_2ce = true;
        rich.wg = 256;
        rich.tile = 2048;
        rich.ctile = 384;
        rich.fft_tile = 2048;
        rich.fft_radix = 2;
        rich.fft_segments = 2;
        rich.latency_window = Some(sample_window(384, 256, 640, 64));

        learner.observe(
            TempoFeedback::new(rich)
                .with_latency(rich.latency_window)
                .with_weight(2.0),
        );

        let mut lean = baseline;
        lean.wg = 96;
        lean.tile = 768;
        lean.ctile = 192;
        lean.fft_tile = 768;
        lean.fft_segments = 1;

        learner.observe(TempoFeedback::new(lean).with_weight(1.0));

        let choice = learner.into_choice();

        assert!(choice.use_2ce);
        assert!(choice.subgroup);
        assert_eq!(choice.wg, 203);
        assert_eq!(choice.tile, 1621);
        assert_eq!(choice.ctile, 320);
        assert_eq!(choice.fft_tile, 1621);
        assert_eq!(choice.fft_radix, 3);
        assert_eq!(choice.fft_segments, 2);

        let window = choice.latency_window.expect("latency window retained");
        assert!(window.lower >= 256);
        assert!(window.upper <= 640);
        assert!(choice.ctile >= window.lower && choice.ctile <= window.upper);
    }

    #[test]
    fn tempo_learner_merge_combines_observations() {
        let baseline = sample_choice();
        let mut left = TempoLearner::new(baseline);
        let mut right = TempoLearner::new(baseline);

        let mut lhs_choice = baseline;
        lhs_choice.wg = 256;
        lhs_choice.tile = 1536;
        lhs_choice.ctile = 320;
        lhs_choice.latency_window = Some(sample_window(320, 192, 448, 32));

        left.observe(TempoFeedback::new(lhs_choice).with_weight(1.0));

        let mut rhs_choice = baseline;
        rhs_choice.use_2ce = true;
        rhs_choice.wg = 64;
        rhs_choice.tile = 896;
        rhs_choice.ctile = 224;
        rhs_choice.latency_window = Some(sample_window(224, 160, 320, 32));

        right.observe(TempoFeedback::new(rhs_choice).with_weight(3.0));

        left.merge(&right);

        let merged = left.into_choice();
        assert!(merged.use_2ce);
        assert!(merged.wg < baseline.wg);
        assert!(merged.tile < lhs_choice.tile);
        let window = merged.latency_window.expect("latency window merged");
        assert_eq!(merged.ctile, 248);
        assert!(merged.ctile >= window.lower && merged.ctile <= window.upper);
    }
}
