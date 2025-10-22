// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Foreign language latency advisors (Julia/Go) that complement the Rust heuristics.
//!
//! The advisors are optional and feature-gated. When enabled they produce hints that
//! adjust the latency window computed by the baseline planner. The integration keeps
//! the heuristics hot path allocation-free and gracefully degrades when the external
//! runtimes are unavailable.

use super::{align_down_to_lanes, align_to_lanes, lane_range, LaneWindow};

/// Applies foreign-language latency refinements to the provided window.
#[cfg_attr(not(feature = "go-bridge"), allow(unused_variables))]
pub(super) fn apply_latency_refinements(
    rows: u32,
    cols: u32,
    k: u32,
    lanes: u32,
    min_ctile: u32,
    max_ctile: u32,
    slack: u32,
    window: &mut LaneWindow,
) {
    #[allow(unused_mut)]
    let mut changed = false;

    #[cfg(feature = "julia-ffi")]
    {
        if adjust_with_julia(window) {
            changed = true;
        }
    }

    #[cfg(feature = "go-bridge")]
    {
        if adjust_with_go(rows, cols, k, lanes, window) {
            changed = true;
        }
    }

    if changed {
        recompute_bounds(lanes, min_ctile, max_ctile, window, slack);
    }
}

fn recompute_bounds(
    lanes: u32,
    min_ctile: u32,
    max_ctile: u32,
    window: &mut LaneWindow,
    slack_hint: u32,
) {
    // Ensure lane limits remain consistent after external adjustments.
    let (min_lane, max_lane) = lane_range(min_ctile, max_ctile, lanes);
    window.min_lane = min_lane;
    window.max_lane = max_lane;

    let stride = window.stride.max(1);
    let slack = window.slack.max(slack_hint.max(1));
    window.slack = slack;

    let lower = window.target.saturating_sub(slack).max(window.min_lane);
    let upper = window.target.saturating_add(slack).min(window.max_lane);

    window.lower = align_down_to_lanes(lower, stride).clamp(min_lane, max_lane);
    window.upper = align_to_lanes(upper, stride).clamp(window.lower, max_lane);

    if window.lower > window.upper {
        window.lower = min_lane;
        window.upper = min_lane;
    }

    window.target = window
        .snapped(window.target)
        .clamp(window.lower, window.upper);
}

#[cfg(feature = "julia-ffi")]
fn adjust_with_julia(window: &mut LaneWindow) -> bool {
    use julia_ffi_poc::tempo_latency_score;

    let tile = window.target.max(window.lower).max(1);
    let slack = window.slack.max(1);
    match tempo_latency_score(tile, slack) {
        Ok(score) if score.is_finite() => {
            let mut changed = false;
            let composite = (tile as f64 + slack as f64).max(1.0);
            let ratio = (score / composite).clamp(0.6, 1.4);
            let new_target = (window.target as f64 * ratio).round() as u32;
            let snapped = window
                .snapped(new_target.max(window.min_lane).min(window.max_lane))
                .clamp(window.min_lane, window.max_lane);
            if snapped != window.target {
                window.target = snapped;
                changed = true;
            }

            let slack_ratio = (score / (slack as f64).max(1.0)).clamp(0.7, 1.3);
            let mut new_slack = (window.slack as f64 * slack_ratio).round() as u32;
            if new_slack == 0 {
                new_slack = 1;
            }
            if new_slack != window.slack {
                window.slack = new_slack;
                changed = true;
            }

            changed
        }
        _ => false,
    }
}

#[cfg(feature = "go-bridge")]
fn adjust_with_go(rows: u32, cols: u32, k: u32, lanes: u32, window: &mut LaneWindow) -> bool {
    if let Some(hints) = go_bridge::request_hint(rows, cols, k, lanes, window) {
        let mut changed = false;

        if let Some(scale) = hints.slack_scale {
            let mut new_slack = ((window.slack as f64) * scale).round() as u32;
            if new_slack == 0 {
                new_slack = 1;
            }
            if new_slack != window.slack {
                window.slack = new_slack;
                changed = true;
            }
        }

        if let Some(stride_scale) = hints.stride_scale {
            let mut new_stride = ((window.stride as f64) * stride_scale).round() as u32;
            if new_stride == 0 {
                new_stride = 1;
            }
            if new_stride > lanes.max(1) {
                new_stride = lanes.max(1);
            }
            if new_stride != window.stride {
                window.stride = new_stride;
                changed = true;
            }
        }

        if let Some(target_shift) = hints.target_shift {
            let mut new_target = window.target as i64 + target_shift as i64;
            if new_target < window.min_lane as i64 {
                new_target = window.min_lane as i64;
            }
            if new_target > window.max_lane as i64 {
                new_target = window.max_lane as i64;
            }
            let new_target = new_target as u32;
            let snapped = window
                .snapped(new_target)
                .clamp(window.min_lane, window.max_lane);
            if snapped != window.target {
                window.target = snapped;
                changed = true;
            }
        }

        changed
    } else {
        false
    }
}

#[cfg(feature = "go-bridge")]
mod go_bridge {
    use super::LaneWindow;
    use serde::Deserialize;
    use std::sync::OnceLock;
    use std::time::Duration;

    #[derive(Debug, Deserialize)]
    struct PredictResponse {
        average: f64,
        sum: f64,
        count: u32,
    }

    #[derive(Debug)]
    pub(super) struct GoHints {
        pub slack_scale: Option<f64>,
        pub stride_scale: Option<f64>,
        pub target_shift: Option<f64>,
    }

    static ENDPOINT: OnceLock<Option<String>> = OnceLock::new();

    pub(super) fn request_hint(
        rows: u32,
        cols: u32,
        k: u32,
        lanes: u32,
        window: &LaneWindow,
    ) -> Option<GoHints> {
        let endpoint = ENDPOINT
            .get_or_init(|| std::env::var("SPIRALTORCH_GO_LATENCY_ENDPOINT").ok())
            .as_ref()?;

        let payload = serde_json::json!({
            "input": [
                rows as f64,
                cols as f64,
                k as f64,
                lanes as f64,
                window.target as f64,
                window.slack as f64,
                window.stride as f64,
            ]
        });

        let response = ureq::post(endpoint)
            .timeout_connect(Duration::from_millis(50))
            .timeout_read(Duration::from_millis(50))
            .set("content-type", "application/json")
            .send_json(payload)
            .ok()?;

        let PredictResponse {
            average,
            sum,
            count,
        } = response.into_json().ok()?;
        if !average.is_finite() || count == 0 {
            return None;
        }

        let base = (window.slack as f64).max(1.0);
        let slack_scale = Some((average / base).clamp(0.5, 1.5));
        let stride_scale =
            Some(((sum / count as f64) / (window.stride as f64).max(1.0)).clamp(0.5, 2.0));
        let target_shift = Some((average - base).clamp(-(base / 2.0), base / 2.0));

        Some(GoHints {
            slack_scale,
            stride_scale,
            target_shift,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_when_no_features_enabled() {
        let mut window = LaneWindow {
            target: 256,
            lower: 128,
            upper: 384,
            min_lane: 128,
            max_lane: 512,
            slack: 96,
            stride: 32,
        };

        let original = window;
        apply_latency_refinements(512, 4096, 64, 32, 128, 512, 96, &mut window);
        assert_eq!(window, original);
    }
}
