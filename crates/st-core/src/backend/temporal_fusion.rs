// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Temporal × spectral fusion helpers that couple lane windows with FFT
//! analytics. The routines keep allocations deterministic so they can be used
//! inside hot ranking paths.

use st_frac::fft::{self, Complex32};
use st_tensor::{emit_tensor_op, emit_tensor_op_meta};

use super::unison_heuristics::LaneWindow;

const MAX_HARMONICS: usize = 8;

/// Combined temporal/spectral snapshot for a latency window.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalSpectralFusion {
    temporal: Vec<f32>,
    spectral: Vec<f32>,
    fusion: Vec<f32>,
    fusion_cols: usize,
    dominant_frequency: f32,
    tempo_hint: f32,
    spectral_energy: f32,
}

impl TemporalSpectralFusion {
    /// Analyses the provided latency window against the matrix dimensions,
    /// returning the fused temporal/spectral profile if at least two samples are
    /// available.
    pub fn analyse(window: &LaneWindow, rows: u32, cols: u32, k: u32, lanes: u32) -> Option<Self> {
        let span = span_len(window)?;
        let temporal = build_temporal_series(window, rows, cols, k, lanes, span);
        let (spectral, dominant_frequency, tempo_hint, spectral_energy) =
            compute_spectrum(&temporal)?;
        let fusion_cols = temporal.len();
        let fusion = fuse_series(&temporal, &spectral);
        let fusion = Self {
            temporal,
            spectral,
            fusion,
            fusion_cols,
            dominant_frequency,
            tempo_hint,
            spectral_energy,
        };
        emit_temporal_spectral_fusion_meta(&fusion, window, rows, cols, k, lanes);
        Some(fusion)
    }

    /// Returns the temporal series that was analysed.
    pub fn temporal(&self) -> &[f32] {
        &self.temporal
    }

    /// Returns the truncated spectral magnitudes (up to [`MAX_HARMONICS`]).
    pub fn spectral(&self) -> &[f32] {
        &self.spectral
    }

    /// Returns the fused 2D grid where rows correspond to harmonics and columns
    /// correspond to temporal samples. The grid reuses a single contiguous
    /// allocation so callers can index rows without triggering per-row Vec
    /// allocations.
    pub fn fusion(&self) -> FusionGrid<'_> {
        FusionGrid {
            data: &self.fusion,
            rows: self.spectral.len(),
            cols: self.fusion_cols,
        }
    }

    /// Index of the dominant frequency discovered by the FFT (normalised to the
    /// FFT length).
    pub fn dominant_frequency(&self) -> f32 {
        self.dominant_frequency
    }

    /// Relative weight of the dominant frequency within the spectrum.
    pub fn tempo_hint(&self) -> f32 {
        self.tempo_hint
    }

    /// Total spectral energy carried by the analysed signal.
    pub fn spectral_energy(&self) -> f32 {
        self.spectral_energy
    }
}

fn finite_meta_f32(value: f32) -> serde_json::Value {
    serde_json::Value::from(if value.is_finite() { value as f64 } else { 0.0 })
}

fn emit_temporal_spectral_fusion_meta(
    fusion: &TemporalSpectralFusion,
    window: &LaneWindow,
    rows: u32,
    cols: u32,
    k: u32,
    lanes: u32,
) {
    emit_tensor_op(
        "temporal_spectral_fusion",
        &[fusion.temporal.len().max(1), fusion.spectral.len().max(1)],
        &[fusion.fusion().rows().max(1), fusion.fusion().cols().max(1)],
    );
    emit_tensor_op_meta("temporal_spectral_fusion", || {
        serde_json::json!({
            "backend": "cpu",
            "requested_backend": "auto",
            "kind": "st_core_temporal_spectral_fusion",
            "rows": rows,
            "cols": cols,
            "k": k,
            "lanes": lanes,
            "window_target": window.target,
            "window_lower": window.lower,
            "window_upper": window.upper,
            "window_stride": window.stride,
            "window_slack": window.slack,
            "temporal_len": fusion.temporal.len(),
            "spectral_len": fusion.spectral.len(),
            "fusion_rows": fusion.fusion().rows(),
            "fusion_cols": fusion.fusion().cols(),
            "dominant_frequency": finite_meta_f32(fusion.dominant_frequency),
            "tempo_hint": finite_meta_f32(fusion.tempo_hint),
            "spectral_energy": finite_meta_f32(fusion.spectral_energy),
            "temporal_first": finite_meta_f32(fusion.temporal.first().copied().unwrap_or(0.0)),
            "spectral_first": finite_meta_f32(fusion.spectral.first().copied().unwrap_or(0.0)),
        })
    });
}

fn span_len(window: &LaneWindow) -> Option<usize> {
    if window.upper <= window.lower {
        return Some(1);
    }
    let step = window.stride.max(1);
    let len = ((window.upper - window.lower) / step).saturating_add(1) as usize;
    if len == 0 {
        None
    } else {
        Some(len)
    }
}

fn build_temporal_series(
    window: &LaneWindow,
    rows: u32,
    cols: u32,
    k: u32,
    lanes: u32,
    span: usize,
) -> Vec<f32> {
    let mut series = Vec::with_capacity(span);
    let lanes_f = lanes.max(1) as f32;
    let rows_norm = (rows.max(1) as f32 / lanes_f).ln_1p();
    let cols_norm = (cols.max(1) as f32 / lanes_f).ln_1p();
    for idx in 0..span {
        let lane = window.lower + window.stride.saturating_mul(idx as u32);
        let lane_f = lane.max(1) as f32;
        let occupancy = (k.max(1) as f32 / lane_f).ln_1p();
        let combined = occupancy + 0.5 * rows_norm + 0.25 * cols_norm;
        series.push(combined);
    }
    series
}

fn compute_spectrum(series: &[f32]) -> Option<(Vec<f32>, f32, f32, f32)> {
    if series.is_empty() {
        return None;
    }
    let fft_len = series.len().next_power_of_two().max(2);
    let mut signal = vec![Complex32::new(0.0, 0.0); fft_len];
    for (idx, &value) in series.iter().enumerate() {
        signal[idx].re = value;
    }
    fft::fft_inplace(&mut signal, false).ok()?;
    let half = fft_len / 2;
    let mut spectral: Vec<f32> = signal
        .iter()
        .take(half)
        .map(|value| finite_or_zero((value.re * value.re + value.im * value.im).sqrt()))
        .collect();
    if spectral.is_empty() {
        spectral.push(0.0);
    }
    let spectral_energy = spectral.iter().sum::<f32>().max(f32::EPSILON);
    let (dominant_idx, dominant_mag) = spectral
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap();
    let tempo_hint = (dominant_mag / spectral_energy).clamp(0.0, 1.0);
    spectral.truncate(MAX_HARMONICS.min(spectral.len()));
    Some((
        spectral,
        dominant_idx as f32 / fft_len as f32,
        tempo_hint,
        spectral_energy,
    ))
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn fuse_series(temporal: &[f32], spectral: &[f32]) -> Vec<f32> {
    let rows = spectral.len();
    let cols = temporal.len();
    let mut fused = Vec::with_capacity(rows * cols);
    for &harmonic in spectral {
        for &sample in temporal {
            fused.push(sample * harmonic);
        }
    }
    fused
}

/// Borrowed view over the fused temporal × spectral grid.
#[derive(Clone, Copy, Debug)]
pub struct FusionGrid<'a> {
    data: &'a [f32],
    rows: usize,
    cols: usize,
}

impl<'a> FusionGrid<'a> {
    /// Number of harmonic rows in the fusion grid.
    pub fn rows(self) -> usize {
        self.rows
    }

    /// Number of temporal samples represented in each row.
    pub fn cols(self) -> usize {
        self.cols
    }

    /// Returns a row slice for the requested harmonic.
    pub fn row(self, index: usize) -> &'a [f32] {
        debug_assert!(index < self.rows);
        let start = index * self.cols;
        let end = start + self.cols;
        &self.data[start..end]
    }

    /// Returns the backing contiguous data for the fusion grid.
    pub fn data(self) -> &'a [f32] {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn demo_window() -> LaneWindow {
        LaneWindow {
            target: 192,
            lower: 128,
            upper: 256,
            min_lane: 128,
            max_lane: 256,
            slack: 64,
            stride: 32,
        }
    }

    #[test]
    fn fusion_produces_temporal_and_spectral_components() {
        let window = demo_window();
        let fusion =
            TemporalSpectralFusion::analyse(&window, 64, 4096, 48, 32).expect("fusion available");
        assert!(!fusion.temporal().is_empty());
        assert!(!fusion.spectral().is_empty());
        let grid = fusion.fusion();
        assert!(grid.rows() > 0);
        assert_eq!(grid.cols(), fusion.temporal().len());
        assert_eq!(grid.row(0).len(), fusion.temporal().len());
        assert_eq!(grid.data().len(), grid.rows() * grid.cols());
        assert!(fusion.dominant_frequency() >= 0.0);
        assert!(fusion.tempo_hint() >= 0.0);
        assert!(fusion.spectral_energy() > 0.0);
    }

    #[test]
    fn fusion_emits_backend_meta() {
        let _lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let window = demo_window();
        let fusion =
            TemporalSpectralFusion::analyse(&window, 64, 4096, 48, 32).expect("fusion available");
        st_tensor::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        let meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "temporal_spectral_fusion"
                    && data["kind"] == "st_core_temporal_spectral_fusion"
                    && data["rows"] == 64
                    && data["cols"] == 4096
                    && data["k"] == 48
                    && data["window_target"] == 192
                    && data["window_lower"] == 128
                    && data["window_upper"] == 256
            })
            .expect("temporal spectral fusion metadata event");
        assert_eq!(meta.1["backend"], "cpu");
        assert_eq!(meta.1["window_target"], 192);
        assert_eq!(meta.1["temporal_len"], fusion.temporal().len());
        assert_eq!(meta.1["spectral_len"], fusion.spectral().len());
        assert_eq!(meta.1["fusion_cols"], fusion.fusion().cols());
        assert!(meta.1["spectral_energy"].as_f64().unwrap() > 0.0);
        assert!(meta.1["tempo_hint"].as_f64().unwrap() >= 0.0);
    }

    #[test]
    fn spectrum_handles_non_finite_samples_without_panicking() {
        let (spectral, dominant_frequency, tempo_hint, spectral_energy) =
            compute_spectrum(&[1.0, f32::NAN, 2.0, f32::INFINITY])
                .expect("spectrum should tolerate non-finite samples");
        assert!(!spectral.is_empty());
        assert!(spectral.iter().all(|value| value.is_finite()));
        assert!(dominant_frequency.is_finite());
        assert!(tempo_hint.is_finite());
        assert!(spectral_energy.is_finite());
    }
}
