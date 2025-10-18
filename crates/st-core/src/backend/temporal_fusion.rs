// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Temporal × spectral fusion helpers that couple lane windows with FFT
//! analytics. The routines keep allocations deterministic so they can be used
//! inside hot ranking paths.

use st_frac::fft::{self, Complex32};

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
        Some(Self {
            temporal,
            spectral,
            fusion,
            fusion_cols,
            dominant_frequency,
            tempo_hint,
            spectral_energy,
        })
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
        .map(|value| (value.re * value.re + value.im * value.im).sqrt())
        .collect();
    if spectral.is_empty() {
        spectral.push(0.0);
    }
    let spectral_energy = spectral.iter().sum::<f32>().max(f32::EPSILON);
    let (dominant_idx, dominant_mag) = spectral
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
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
}
