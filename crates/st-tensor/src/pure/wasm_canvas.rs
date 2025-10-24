// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Minimal WASM canvas interop helpers for the pure fractal stack.
//!
//! The goal of this module is to keep the browser pathway fully Rust-native:
//! we stream `UringFractalScheduler` relations into a persistent tensor buffer,
//! then map that buffer onto an RGBA pixel grid that JavaScript can blit into a
//! `<canvas>` element without allocating intermediate copies. Everything here
//! stays in safe Rust so it can compile to WASM as-is, while HTML/JS glue code
//! can simply forward the produced byte slice into `ImageData`.

use super::{
    fractal::{FractalPatch, UringFractalScheduler},
    AmegaHypergrad, AmegaRealgrad, DesireGradientControl, DesireGradientInterpretation,
    GradientSummary, PureResult, Tensor, TensorError,
};
use core::f32::consts::PI;
use st_frac::fft::{self, Complex32};

/// Streaming Z-space normaliser that keeps canvas updates stable even when the
/// underlying tensor swings across vastly different value ranges.
///
/// The normaliser keeps a smoothed min/max window so that every refresh can be
/// mapped into the `[0, 1]` interval without triggering harsh brightness jumps
/// (common when exploring high-curvature Z-space during large model training).
#[derive(Clone, Debug)]
pub struct CanvasNormalizer {
    alpha: f32,
    epsilon: f32,
    state: Option<(f32, f32)>,
}

impl CanvasNormalizer {
    /// Construct a normaliser with a smoothing factor and minimum range
    /// epsilon. `alpha` controls how aggressively we adapt to new values while
    /// `epsilon` guards against zero-width ranges that would otherwise yield
    /// NaNs and panic the renderer.
    pub fn new(alpha: f32, epsilon: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
            epsilon: epsilon.max(1e-6),
            state: None,
        }
    }

    /// Reset any accumulated state so the next update starts from scratch.
    pub fn reset(&mut self) {
        self.state = None;
    }

    /// Last observed min/max range.
    pub fn state(&self) -> Option<(f32, f32)> {
        self.state
    }

    /// Fold a new tensor slice into the normaliser. Returns the smoothed
    /// `(min, max)` pair that will be used to map values into `[0, 1]`.
    pub fn update(&mut self, data: &[f32]) -> (f32, f32) {
        let mut local_min = f32::INFINITY;
        let mut local_max = f32::NEG_INFINITY;
        for &value in data {
            if value.is_nan() {
                continue;
            }
            local_min = local_min.min(value);
            local_max = local_max.max(value);
        }

        if !local_min.is_finite() || !local_max.is_finite() {
            // Fallback if everything was NaN/inf; reuse previous state or clamp
            // to zero so we still emit a valid texture.
            let (min, max) = self.state.unwrap_or((0.0, 1.0));
            return (min, max.max(min + self.epsilon));
        }

        let (mut smoothed_min, mut smoothed_max) = (local_min, local_max);
        if let Some((prev_min, prev_max)) = self.state {
            let alpha = self.alpha;
            let one_minus = 1.0 - alpha;
            smoothed_min = prev_min * one_minus + local_min * alpha;
            smoothed_max = prev_max * one_minus + local_max * alpha;
        }

        if smoothed_max - smoothed_min < self.epsilon {
            smoothed_max = smoothed_min + self.epsilon;
        }

        self.state = Some((smoothed_min, smoothed_max));
        (smoothed_min, smoothed_max)
    }

    /// Map a raw tensor value into `[0, 1]` using the last computed range.
    pub fn normalize(&self, value: f32) -> f32 {
        match self.state {
            Some((min, max)) => ((value - min) / (max - min)).clamp(0.0, 1.0),
            None => 0.5,
        }
    }
}

impl Default for CanvasNormalizer {
    fn default() -> Self {
        Self::new(0.2, 1e-3)
    }
}

/// Palette used to colourise the normalised tensor energy.
#[derive(Clone, Copy, Debug)]
pub enum CanvasPalette {
    /// Blue → Magenta gradient tuned for Z-space phase portraits.
    BlueMagenta,
    /// "Turbo" inspired gradient with warm highlights.
    Turbo,
    /// Plain monochrome luminance for debugging / Python interop.
    Grayscale,
}

impl CanvasPalette {
    fn map(self, t: f32) -> [u8; 4] {
        match self {
            CanvasPalette::BlueMagenta => {
                let intensity = (t * 255.0) as u8;
                let accent = ((1.0 - t) * 255.0) as u8;
                [accent, intensity / 2, intensity, 255]
            }
            CanvasPalette::Turbo => {
                // Lightweight polynomial fit for the Turbo colour map.
                let r = (34.61
                    + t * (1172.33
                        + t * (-10793.56 + t * (33300.12 + t * (-38394.49 + t * 14825.05)))))
                    / 255.0;
                let g = (23.31
                    + t * (557.33 + t * (-1224.52 + t * (3934.66 + t * (-4372.56 + t * 1641.93)))))
                    / 255.0;
                let b = (27.2
                    + t * (321.15 + t * (-1449.66 + t * (2144.05 + t * (-1177.27 + t * 234.125)))))
                    / 255.0;
                [
                    (r.clamp(0.0, 1.0) * 255.0) as u8,
                    (g.clamp(0.0, 1.0) * 255.0) as u8,
                    (b.clamp(0.0, 1.0) * 255.0) as u8,
                    255,
                ]
            }
            CanvasPalette::Grayscale => {
                let luminance = (t * 255.0) as u8;
                [luminance, luminance, luminance, 255]
            }
        }
    }

    fn map_vectorised(self, t: f32) -> ([u8; 4], [f32; 3]) {
        let rgba = self.map(t);
        let chroma = [
            rgba[0] as f32 / 255.0 * 2.0 - 1.0,
            rgba[1] as f32 / 255.0 * 2.0 - 1.0,
            rgba[2] as f32 / 255.0 * 2.0 - 1.0,
        ];
        (rgba, chroma)
    }
}

impl Default for CanvasPalette {
    fn default() -> Self {
        CanvasPalette::BlueMagenta
    }
}

/// Window function used when projecting the vector field into frequency space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CanvasWindow {
    /// Plain rectangular window (no tapering).
    Rectangular,
    /// Hann window for moderate sidelobe suppression.
    Hann,
    /// Hamming window balancing main-lobe width and sidelobe height.
    Hamming,
    /// Blackman window for aggressive sidelobe attenuation.
    Blackman,
}

impl CanvasWindow {
    fn coefficients(self, len: usize) -> Vec<f32> {
        if len <= 1 {
            return vec![1.0; len];
        }
        let mut coeffs = Vec::with_capacity(len);
        let denom = (len - 1) as f32;
        for n in 0..len {
            let ratio = n as f32 / denom;
            let value = match self {
                CanvasWindow::Rectangular => 1.0,
                CanvasWindow::Hann => 0.5 - 0.5 * (2.0 * PI * ratio).cos(),
                CanvasWindow::Hamming => 0.54 - 0.46 * (2.0 * PI * ratio).cos(),
                CanvasWindow::Blackman => {
                    0.42 - 0.5 * (2.0 * PI * ratio).cos() + 0.08 * (4.0 * PI * ratio).cos()
                }
            };
            coeffs.push(value);
        }
        coeffs
    }
}

/// Vector field that captures both the normalised tensor energy and the
/// palette-projected chroma in Z-space friendly coordinates.
#[derive(Clone, Debug)]
pub struct ColorVectorField {
    width: usize,
    height: usize,
    vectors: Vec<[f32; 4]>,
}

impl ColorVectorField {
    const FFT_CHANNELS: usize = 4;
    const FFT_COMPLEX_STRIDE: usize = 2;
    const FFT_INTERLEAVED_STRIDE: usize = Self::FFT_CHANNELS * Self::FFT_COMPLEX_STRIDE;
    const POWER_DB_EPSILON: f32 = 1e-12;
    const POWER_DB_FLOOR: f32 = -160.0;

    pub fn new(width: usize, height: usize) -> Self {
        let mut field = Self {
            width,
            height,
            vectors: Vec::with_capacity(width * height),
        };
        field.ensure_shape(width, height);
        field
    }

    fn ensure_fft_dimensions(&self) -> PureResult<()> {
        if self.width == 0 || self.height == 0 {
            Err(TensorError::EmptyInput("canvas_fft"))
        } else {
            Ok(())
        }
    }

    fn ensure_shape(&mut self, width: usize, height: usize) {
        if self.width != width || self.height != height {
            self.width = width;
            self.height = height;
        }
        let expected = width * height;
        if self.vectors.len() != expected {
            self.vectors.resize(expected, [0.0; 4]);
        }
    }

    fn set(&mut self, idx: usize, energy: f32, chroma: [f32; 3]) {
        self.vectors[idx] = [energy, chroma[0], chroma[1], chroma[2]];
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn vectors(&self) -> &[[f32; 4]] {
        &self.vectors
    }

    pub fn iter(&self) -> impl Iterator<Item = [f32; 4]> + '_ {
        self.vectors.iter().copied()
    }

    pub fn as_tensor(&self) -> PureResult<Tensor> {
        let mut flat = Vec::with_capacity(self.vectors.len() * 4);
        for vector in &self.vectors {
            flat.extend_from_slice(vector);
        }
        Tensor::from_vec(self.height, self.width * 4, flat)
    }

    pub fn energy_tensor(&self) -> PureResult<Tensor> {
        let mut energy = Vec::with_capacity(self.vectors.len());
        for vector in &self.vectors {
            energy.push(vector[0]);
        }
        Tensor::from_vec(self.height, self.width, energy)
    }

    pub fn to_zspace_patch(
        &self,
        coherence: f32,
        tension: f32,
        depth: u32,
    ) -> PureResult<FractalPatch> {
        let relation = self.energy_tensor()?;
        FractalPatch::new(relation, coherence, tension, depth)
    }

    /// Compute a row-wise FFT over the energy + chroma channels and expose the
    /// result as interleaved `[re, im]` floats for each component. This keeps
    /// the layout transformer-friendly while letting WASM consumers reuse the
    /// CPU fallback when no tuned plan is available yet.
    pub fn fft_rows_interleaved(&self, inverse: bool) -> PureResult<Vec<f32>> {
        let width = self.width;
        let height = self.height;
        self.ensure_fft_dimensions()?;
        let mut energy = vec![Complex32::default(); width];
        let mut chroma_r = vec![Complex32::default(); width];
        let mut chroma_g = vec![Complex32::default(); width];
        let mut chroma_b = vec![Complex32::default(); width];
        let mut out = Vec::with_capacity(self.height * width * Self::FFT_INTERLEAVED_STRIDE);

        for row in 0..height {
            for col in 0..width {
                let vector = self.vectors[row * width + col];
                energy[col] = Complex32::new(vector[0], 0.0);
                chroma_r[col] = Complex32::new(vector[1], 0.0);
                chroma_g[col] = Complex32::new(vector[2], 0.0);
                chroma_b[col] = Complex32::new(vector[3], 0.0);
            }

            compute_fft(&mut energy, inverse)?;
            compute_fft(&mut chroma_r, inverse)?;
            compute_fft(&mut chroma_g, inverse)?;
            compute_fft(&mut chroma_b, inverse)?;

            for inner in 0..width {
                out.push(energy[inner].re);
                out.push(energy[inner].im);
                out.push(chroma_r[inner].re);
                out.push(chroma_r[inner].im);
                out.push(chroma_g[inner].re);
                out.push(chroma_g[inner].im);
                out.push(chroma_b[inner].re);
                out.push(chroma_b[inner].im);
            }
        }

        Ok(out)
    }

    /// Row-wise FFT with an explicit window applied before the transform.
    pub fn fft_rows_interleaved_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Vec<f32>> {
        let width = self.width;
        let height = self.height;
        self.ensure_fft_dimensions()?;
        let mut energy = vec![Complex32::default(); width];
        let mut chroma_r = vec![Complex32::default(); width];
        let mut chroma_g = vec![Complex32::default(); width];
        let mut chroma_b = vec![Complex32::default(); width];
        let mut out = Vec::with_capacity(self.height * width * Self::FFT_INTERLEAVED_STRIDE);
        let coeffs = window.coefficients(width);

        for row in 0..height {
            for col in 0..width {
                let vector = self.vectors[row * width + col];
                let w = coeffs[col];
                energy[col] = Complex32::new(vector[0] * w, 0.0);
                chroma_r[col] = Complex32::new(vector[1] * w, 0.0);
                chroma_g[col] = Complex32::new(vector[2] * w, 0.0);
                chroma_b[col] = Complex32::new(vector[3] * w, 0.0);
            }

            compute_fft(&mut energy, inverse)?;
            compute_fft(&mut chroma_r, inverse)?;
            compute_fft(&mut chroma_g, inverse)?;
            compute_fft(&mut chroma_b, inverse)?;

            for inner in 0..width {
                out.push(energy[inner].re);
                out.push(energy[inner].im);
                out.push(chroma_r[inner].re);
                out.push(chroma_r[inner].im);
                out.push(chroma_g[inner].re);
                out.push(chroma_g[inner].im);
                out.push(chroma_b[inner].re);
                out.push(chroma_b[inner].im);
            }
        }

        Ok(out)
    }

    /// Convenience wrapper around [`fft_rows_interleaved`] that returns the
    /// spectrum as a tensor with shape `(height, width * 8)`.
    pub fn fft_rows_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        let data = self.fft_rows_interleaved(inverse)?;
        Tensor::from_vec(self.height, self.width * Self::FFT_INTERLEAVED_STRIDE, data)
    }

    /// Row-wise FFT tensor helper that applies `window` prior to transformation.
    pub fn fft_rows_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        let data = self.fft_rows_interleaved_with_window(window, inverse)?;
        Tensor::from_vec(self.height, self.width * Self::FFT_INTERLEAVED_STRIDE, data)
    }

    /// Convenience wrapper that converts the row-wise FFT into magnitude space.
    /// The returned tensor has shape `(height, width * 4)` where each pixel packs
    /// the magnitude of the energy and chroma channels in order.
    pub fn fft_rows_magnitude_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        let spectrum = self.fft_rows_interleaved(inverse)?;
        Self::magnitude_tensor_from_interleaved(self.height, self.width, &spectrum)
    }

    /// Row-wise FFT magnitude helper with a pre-transform window.
    pub fn fft_rows_magnitude_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        let spectrum = self.fft_rows_interleaved_with_window(window, inverse)?;
        Self::magnitude_tensor_from_interleaved(self.height, self.width, &spectrum)
    }

    /// Row-wise FFT power helper mirroring [`fft_rows_interleaved`]. The
    /// returned tensor has shape `(height, width * 4)` storing the squared
    /// magnitude per channel so WASM integrations can directly sample spectral
    /// energy without recomputing it on the JavaScript side.
    pub fn fft_rows_power_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        let spectrum = self.fft_rows_interleaved(inverse)?;
        Self::power_tensor_from_interleaved(self.height, self.width, &spectrum)
    }

    /// Row-wise FFT power helper with an explicit window.
    pub fn fft_rows_power_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        let spectrum = self.fft_rows_interleaved_with_window(window, inverse)?;
        Self::power_tensor_from_interleaved(self.height, self.width, &spectrum)
    }

    /// Row-wise FFT log-power helper mirroring [`fft_rows_power_tensor`]. The
    /// returned tensor has shape `(height, width * 4)` storing the decibel-scaled
    /// magnitude with a floor at [`Self::POWER_DB_FLOOR`] to keep zeros finite.
    pub fn fft_rows_power_db_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        let spectrum = self.fft_rows_interleaved(inverse)?;
        Self::power_db_tensor_from_interleaved(self.height, self.width, &spectrum)
    }

    /// Row-wise FFT log-power helper with a pre-transform window.
    pub fn fft_rows_power_db_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        let spectrum = self.fft_rows_interleaved_with_window(window, inverse)?;
        Self::power_db_tensor_from_interleaved(self.height, self.width, &spectrum)
    }

    /// Row-wise FFT phase helper mirroring [`fft_rows_magnitude_tensor`]. The
    /// returned tensor has shape `(height, width * 4)` and stores phases in
    /// radians using `atan2(im, re)` for each channel.
    pub fn fft_rows_phase_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        let spectrum = self.fft_rows_interleaved(inverse)?;
        Self::phase_tensor_from_interleaved(self.height, self.width, &spectrum)
    }

    /// Row-wise FFT phase helper with a window applied before transformation.
    pub fn fft_rows_phase_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        let spectrum = self.fft_rows_interleaved_with_window(window, inverse)?;
        Self::phase_tensor_from_interleaved(self.height, self.width, &spectrum)
    }

    /// Compute both the magnitude and phase spectra for the row-wise FFT in a
    /// single pass. Returns `(magnitude, phase)` tensors, each with shape
    /// `(height, width * 4)`.
    pub fn fft_rows_polar_tensors(&self, inverse: bool) -> PureResult<(Tensor, Tensor)> {
        let spectrum = self.fft_rows_interleaved(inverse)?;
        Self::polar_tensors_from_interleaved(self.height, self.width, &spectrum)
    }

    /// Row-wise FFT polar helper with windowing.
    pub fn fft_rows_polar_tensors_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<(Tensor, Tensor)> {
        let spectrum = self.fft_rows_interleaved_with_window(window, inverse)?;
        Self::polar_tensors_from_interleaved(self.height, self.width, &spectrum)
    }

    /// Compute a column-wise FFT over the energy + chroma channels and expose
    /// the result as interleaved `[re, im]` floats for each component. This is
    /// the vertical counterpart to [`fft_rows_interleaved`] so WASM callers can
    /// analyse spectra along either axis without bouncing back to native Rust
    /// helpers.
    pub fn fft_cols_interleaved(&self, inverse: bool) -> PureResult<Vec<f32>> {
        let height = self.height;
        let width = self.width;
        self.ensure_fft_dimensions()?;

        let mut energy = vec![Complex32::default(); height];
        let mut chroma_r = vec![Complex32::default(); height];
        let mut chroma_g = vec![Complex32::default(); height];
        let mut chroma_b = vec![Complex32::default(); height];
        let mut out = Vec::with_capacity(self.height * self.width * Self::FFT_INTERLEAVED_STRIDE);

        for col in 0..width {
            for row in 0..height {
                let vector = self.vectors[row * width + col];
                energy[row] = Complex32::new(vector[0], 0.0);
                chroma_r[row] = Complex32::new(vector[1], 0.0);
                chroma_g[row] = Complex32::new(vector[2], 0.0);
                chroma_b[row] = Complex32::new(vector[3], 0.0);
            }

            compute_fft(&mut energy, inverse)?;
            compute_fft(&mut chroma_r, inverse)?;
            compute_fft(&mut chroma_g, inverse)?;
            compute_fft(&mut chroma_b, inverse)?;

            for row in 0..height {
                out.push(energy[row].re);
                out.push(energy[row].im);
                out.push(chroma_r[row].re);
                out.push(chroma_r[row].im);
                out.push(chroma_g[row].re);
                out.push(chroma_g[row].im);
                out.push(chroma_b[row].re);
                out.push(chroma_b[row].im);
            }
        }

        Ok(out)
    }

    /// Column-wise FFT with an explicit window applied per column sample.
    pub fn fft_cols_interleaved_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Vec<f32>> {
        let height = self.height;
        let width = self.width;
        self.ensure_fft_dimensions()?;

        let mut energy = vec![Complex32::default(); height];
        let mut chroma_r = vec![Complex32::default(); height];
        let mut chroma_g = vec![Complex32::default(); height];
        let mut chroma_b = vec![Complex32::default(); height];
        let mut out = Vec::with_capacity(self.height * self.width * Self::FFT_INTERLEAVED_STRIDE);
        let coeffs = window.coefficients(height);

        for col in 0..width {
            for row in 0..height {
                let vector = self.vectors[row * width + col];
                let w = coeffs[row];
                energy[row] = Complex32::new(vector[0] * w, 0.0);
                chroma_r[row] = Complex32::new(vector[1] * w, 0.0);
                chroma_g[row] = Complex32::new(vector[2] * w, 0.0);
                chroma_b[row] = Complex32::new(vector[3] * w, 0.0);
            }

            compute_fft(&mut energy, inverse)?;
            compute_fft(&mut chroma_r, inverse)?;
            compute_fft(&mut chroma_g, inverse)?;
            compute_fft(&mut chroma_b, inverse)?;

            for row in 0..height {
                out.push(energy[row].re);
                out.push(energy[row].im);
                out.push(chroma_r[row].re);
                out.push(chroma_r[row].im);
                out.push(chroma_g[row].re);
                out.push(chroma_g[row].im);
                out.push(chroma_b[row].re);
                out.push(chroma_b[row].im);
            }
        }

        Ok(out)
    }

    /// Convenience wrapper around [`fft_cols_interleaved`] that returns the
    /// spectrum as a tensor with shape `(width, height * 8)` laid out in column
    /// order (one row per column with interleaved complex components).
    pub fn fft_cols_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        let data = self.fft_cols_interleaved(inverse)?;
        Tensor::from_vec(self.width, self.height * Self::FFT_INTERLEAVED_STRIDE, data)
    }

    /// Column-wise FFT tensor helper that applies `window` prior to transformation.
    pub fn fft_cols_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        let data = self.fft_cols_interleaved_with_window(window, inverse)?;
        Tensor::from_vec(self.width, self.height * Self::FFT_INTERLEAVED_STRIDE, data)
    }

    /// Column-wise FFT magnitude helper mirroring
    /// [`fft_rows_magnitude_tensor`]. The returned tensor has shape
    /// `(width, height * 4)` where each row corresponds to a column in the
    /// original canvas.
    pub fn fft_cols_magnitude_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        let spectrum = self.fft_cols_interleaved(inverse)?;
        Self::magnitude_tensor_from_interleaved(self.width, self.height, &spectrum)
    }

    /// Column-wise FFT magnitude helper with a pre-transform window.
    pub fn fft_cols_magnitude_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        let spectrum = self.fft_cols_interleaved_with_window(window, inverse)?;
        Self::magnitude_tensor_from_interleaved(self.width, self.height, &spectrum)
    }

    /// Column-wise FFT power helper mirroring [`fft_cols_interleaved`]. The
    /// returned tensor has shape `(width, height * 4)` storing squared
    /// magnitudes per channel for direct spectral energy sampling.
    pub fn fft_cols_power_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        let spectrum = self.fft_cols_interleaved(inverse)?;
        Self::power_tensor_from_interleaved(self.width, self.height, &spectrum)
    }

    /// Column-wise FFT power helper that applies `window` before transformation.
    pub fn fft_cols_power_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        let spectrum = self.fft_cols_interleaved_with_window(window, inverse)?;
        Self::power_tensor_from_interleaved(self.width, self.height, &spectrum)
    }

    /// Column-wise FFT log-power helper that mirrors [`fft_cols_power_tensor`].
    /// The returned tensor has shape `(width, height * 4)` storing decibel-scaled
    /// spectral energy for each channel.
    pub fn fft_cols_power_db_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        let spectrum = self.fft_cols_interleaved(inverse)?;
        Self::power_db_tensor_from_interleaved(self.width, self.height, &spectrum)
    }

    /// Column-wise FFT log-power helper with windowing.
    pub fn fft_cols_power_db_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        let spectrum = self.fft_cols_interleaved_with_window(window, inverse)?;
        Self::power_db_tensor_from_interleaved(self.width, self.height, &spectrum)
    }

    /// Column-wise FFT phase helper mirroring [`fft_rows_phase_tensor`].
    pub fn fft_cols_phase_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        let spectrum = self.fft_cols_interleaved(inverse)?;
        Self::phase_tensor_from_interleaved(self.width, self.height, &spectrum)
    }

    /// Column-wise FFT phase helper with an explicit window.
    pub fn fft_cols_phase_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        let spectrum = self.fft_cols_interleaved_with_window(window, inverse)?;
        Self::phase_tensor_from_interleaved(self.width, self.height, &spectrum)
    }

    /// Compute both the magnitude and phase spectra for the column-wise FFT in
    /// a single pass. Returns `(magnitude, phase)` tensors, each with shape
    /// `(width, height * 4)`.
    pub fn fft_cols_polar_tensors(&self, inverse: bool) -> PureResult<(Tensor, Tensor)> {
        let spectrum = self.fft_cols_interleaved(inverse)?;
        Self::polar_tensors_from_interleaved(self.width, self.height, &spectrum)
    }

    /// Column-wise FFT polar helper with windowing.
    pub fn fft_cols_polar_tensors_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<(Tensor, Tensor)> {
        let spectrum = self.fft_cols_interleaved_with_window(window, inverse)?;
        Self::polar_tensors_from_interleaved(self.width, self.height, &spectrum)
    }

    /// Compute a 2D FFT over the energy + chroma channels. The returned buffer
    /// is laid out in row-major order with interleaved `[re, im]` floats for
    /// each channel, matching the row-wise layout so WASM callers can reuse the
    /// same upload paths.
    pub fn fft_2d_interleaved(&self, inverse: bool) -> PureResult<Vec<f32>> {
        let width = self.width;
        let height = self.height;
        self.ensure_fft_dimensions()?;

        let size = width * height;
        let mut energy = vec![Complex32::default(); size];
        let mut chroma_r = vec![Complex32::default(); size];
        let mut chroma_g = vec![Complex32::default(); size];
        let mut chroma_b = vec![Complex32::default(); size];

        for row in 0..height {
            for col in 0..width {
                let idx = row * width + col;
                let vector = self.vectors[idx];
                energy[idx] = Complex32::new(vector[0], 0.0);
                chroma_r[idx] = Complex32::new(vector[1], 0.0);
                chroma_g[idx] = Complex32::new(vector[2], 0.0);
                chroma_b[idx] = Complex32::new(vector[3], 0.0);
            }

            let start = row * width;
            let end = start + width;
            compute_fft(&mut energy[start..end], inverse)?;
            compute_fft(&mut chroma_r[start..end], inverse)?;
            compute_fft(&mut chroma_g[start..end], inverse)?;
            compute_fft(&mut chroma_b[start..end], inverse)?;
        }

        let mut column_energy = vec![Complex32::default(); height];
        let mut column_chroma_r = vec![Complex32::default(); height];
        let mut column_chroma_g = vec![Complex32::default(); height];
        let mut column_chroma_b = vec![Complex32::default(); height];

        for col in 0..width {
            for row in 0..height {
                let idx = row * width + col;
                column_energy[row] = energy[idx];
                column_chroma_r[row] = chroma_r[idx];
                column_chroma_g[row] = chroma_g[idx];
                column_chroma_b[row] = chroma_b[idx];
            }

            compute_fft(&mut column_energy, inverse)?;
            compute_fft(&mut column_chroma_r, inverse)?;
            compute_fft(&mut column_chroma_g, inverse)?;
            compute_fft(&mut column_chroma_b, inverse)?;

            for row in 0..height {
                let idx = row * width + col;
                energy[idx] = column_energy[row];
                chroma_r[idx] = column_chroma_r[row];
                chroma_g[idx] = column_chroma_g[row];
                chroma_b[idx] = column_chroma_b[row];
            }
        }

        let mut out = Vec::with_capacity(size * Self::FFT_INTERLEAVED_STRIDE);
        for idx in 0..size {
            out.push(energy[idx].re);
            out.push(energy[idx].im);
            out.push(chroma_r[idx].re);
            out.push(chroma_r[idx].im);
            out.push(chroma_g[idx].re);
            out.push(chroma_g[idx].im);
            out.push(chroma_b[idx].re);
            out.push(chroma_b[idx].im);
        }

        Ok(out)
    }

    /// 2D FFT with an explicit window applied along both axes.
    pub fn fft_2d_interleaved_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Vec<f32>> {
        let width = self.width;
        let height = self.height;
        self.ensure_fft_dimensions()?;

        let size = width * height;
        let mut energy = vec![Complex32::default(); size];
        let mut chroma_r = vec![Complex32::default(); size];
        let mut chroma_g = vec![Complex32::default(); size];
        let mut chroma_b = vec![Complex32::default(); size];
        let row_coeffs = window.coefficients(width);
        let col_coeffs = window.coefficients(height);

        for row in 0..height {
            let row_weight = col_coeffs[row];
            for col in 0..width {
                let idx = row * width + col;
                let vector = self.vectors[idx];
                let weight = row_weight * row_coeffs[col];
                energy[idx] = Complex32::new(vector[0] * weight, 0.0);
                chroma_r[idx] = Complex32::new(vector[1] * weight, 0.0);
                chroma_g[idx] = Complex32::new(vector[2] * weight, 0.0);
                chroma_b[idx] = Complex32::new(vector[3] * weight, 0.0);
            }

            let start = row * width;
            let end = start + width;
            compute_fft(&mut energy[start..end], inverse)?;
            compute_fft(&mut chroma_r[start..end], inverse)?;
            compute_fft(&mut chroma_g[start..end], inverse)?;
            compute_fft(&mut chroma_b[start..end], inverse)?;
        }

        let mut column_energy = vec![Complex32::default(); height];
        let mut column_chroma_r = vec![Complex32::default(); height];
        let mut column_chroma_g = vec![Complex32::default(); height];
        let mut column_chroma_b = vec![Complex32::default(); height];

        for col in 0..width {
            for row in 0..height {
                let idx = row * width + col;
                column_energy[row] = energy[idx];
                column_chroma_r[row] = chroma_r[idx];
                column_chroma_g[row] = chroma_g[idx];
                column_chroma_b[row] = chroma_b[idx];
            }

            compute_fft(&mut column_energy, inverse)?;
            compute_fft(&mut column_chroma_r, inverse)?;
            compute_fft(&mut column_chroma_g, inverse)?;
            compute_fft(&mut column_chroma_b, inverse)?;

            for row in 0..height {
                let idx = row * width + col;
                energy[idx] = column_energy[row];
                chroma_r[idx] = column_chroma_r[row];
                chroma_g[idx] = column_chroma_g[row];
                chroma_b[idx] = column_chroma_b[row];
            }
        }

        let mut out = Vec::with_capacity(size * Self::FFT_INTERLEAVED_STRIDE);
        for idx in 0..size {
            out.push(energy[idx].re);
            out.push(energy[idx].im);
            out.push(chroma_r[idx].re);
            out.push(chroma_r[idx].im);
            out.push(chroma_g[idx].re);
            out.push(chroma_g[idx].im);
            out.push(chroma_b[idx].re);
            out.push(chroma_b[idx].im);
        }

        Ok(out)
    }

    /// Convenience wrapper around [`fft_2d_interleaved`] that returns the
    /// spectrum as a tensor with shape `(height, width * 8)`.
    pub fn fft_2d_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        let data = self.fft_2d_interleaved(inverse)?;
        Tensor::from_vec(self.height, self.width * Self::FFT_INTERLEAVED_STRIDE, data)
    }

    /// 2D FFT tensor helper that applies `window` prior to transformation.
    pub fn fft_2d_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        let data = self.fft_2d_interleaved_with_window(window, inverse)?;
        Tensor::from_vec(self.height, self.width * Self::FFT_INTERLEAVED_STRIDE, data)
    }

    /// 2D FFT magnitude helper mirroring [`fft_2d_interleaved`]. The returned
    /// tensor has shape `(height, width * 4)` with magnitudes for each channel.
    pub fn fft_2d_magnitude_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        let spectrum = self.fft_2d_interleaved(inverse)?;
        Self::magnitude_tensor_from_interleaved(self.height, self.width, &spectrum)
    }

    /// 2D FFT magnitude helper with windowing.
    pub fn fft_2d_magnitude_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        let spectrum = self.fft_2d_interleaved_with_window(window, inverse)?;
        Self::magnitude_tensor_from_interleaved(self.height, self.width, &spectrum)
    }

    /// 2D FFT power helper mirroring [`fft_2d_interleaved`]. The returned tensor
    /// has shape `(height, width * 4)` storing squared magnitudes per channel so
    /// integrators can probe energy across both axes without recomputing.
    pub fn fft_2d_power_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        let spectrum = self.fft_2d_interleaved(inverse)?;
        Self::power_tensor_from_interleaved(self.height, self.width, &spectrum)
    }

    /// 2D FFT power helper with an explicit window across both axes.
    pub fn fft_2d_power_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        let spectrum = self.fft_2d_interleaved_with_window(window, inverse)?;
        Self::power_tensor_from_interleaved(self.height, self.width, &spectrum)
    }

    /// 2D FFT log-power helper mirroring [`fft_2d_power_tensor`]. Returns a
    /// tensor with shape `(height, width * 4)` packed with decibel-scaled
    /// spectral energy for each channel so WASM integrations can visualise
    /// logarithmic energy without extra processing.
    pub fn fft_2d_power_db_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        let spectrum = self.fft_2d_interleaved(inverse)?;
        Self::power_db_tensor_from_interleaved(self.height, self.width, &spectrum)
    }

    /// 2D FFT log-power helper with windowing.
    pub fn fft_2d_power_db_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        let spectrum = self.fft_2d_interleaved_with_window(window, inverse)?;
        Self::power_db_tensor_from_interleaved(self.height, self.width, &spectrum)
    }

    /// 2D FFT phase helper mirroring [`fft_2d_interleaved`]. The returned tensor
    /// has shape `(height, width * 4)` storing per-channel phase angles.
    pub fn fft_2d_phase_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        let spectrum = self.fft_2d_interleaved(inverse)?;
        Self::phase_tensor_from_interleaved(self.height, self.width, &spectrum)
    }

    /// 2D FFT phase helper with windowing.
    pub fn fft_2d_phase_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        let spectrum = self.fft_2d_interleaved_with_window(window, inverse)?;
        Self::phase_tensor_from_interleaved(self.height, self.width, &spectrum)
    }

    /// Compute both the magnitude and phase spectra for the 2D FFT in a single
    /// pass. Returns `(magnitude, phase)` tensors, each with shape
    /// `(height, width * 4)`.
    pub fn fft_2d_polar_tensors(&self, inverse: bool) -> PureResult<(Tensor, Tensor)> {
        let spectrum = self.fft_2d_interleaved(inverse)?;
        Self::polar_tensors_from_interleaved(self.height, self.width, &spectrum)
    }

    /// 2D FFT polar helper with a pre-transform window.
    pub fn fft_2d_polar_tensors_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<(Tensor, Tensor)> {
        let spectrum = self.fft_2d_interleaved_with_window(window, inverse)?;
        Self::polar_tensors_from_interleaved(self.height, self.width, &spectrum)
    }

    fn map_tensor_from_interleaved<F>(
        rows: usize,
        cols: usize,
        spectrum: &[f32],
        mut map: F,
    ) -> PureResult<Tensor>
    where
        F: FnMut(&[f32]) -> [f32; Self::FFT_CHANNELS],
    {
        let expected_pairs = rows
            .checked_mul(cols)
            .ok_or(TensorError::TensorVolumeExceeded {
                label: "canvas_fft_map",
                volume: rows.saturating_mul(cols),
                max_volume: usize::MAX,
            })?;
        let expected_len = expected_pairs
            .checked_mul(Self::FFT_INTERLEAVED_STRIDE)
            .ok_or(TensorError::TensorVolumeExceeded {
                label: "canvas_fft_map",
                volume: rows
                    .saturating_mul(cols)
                    .saturating_mul(Self::FFT_INTERLEAVED_STRIDE),
                max_volume: usize::MAX,
            })?;

        if spectrum.len() != expected_len {
            return Err(TensorError::DataLength {
                expected: expected_len,
                got: spectrum.len(),
            });
        }

        let mut out = Vec::with_capacity(expected_pairs * Self::FFT_CHANNELS);
        for chunk in spectrum.chunks_exact(Self::FFT_INTERLEAVED_STRIDE) {
            let mapped = map(chunk);
            out.extend_from_slice(&mapped);
        }

        Tensor::from_vec(rows, cols * Self::FFT_CHANNELS, out)
    }

    fn magnitude_tensor_from_interleaved(
        rows: usize,
        cols: usize,
        spectrum: &[f32],
    ) -> PureResult<Tensor> {
        Self::map_tensor_from_interleaved(rows, cols, spectrum, |chunk| {
            [
                chunk[0].hypot(chunk[1]),
                chunk[2].hypot(chunk[3]),
                chunk[4].hypot(chunk[5]),
                chunk[6].hypot(chunk[7]),
            ]
        })
    }

    fn phase_tensor_from_interleaved(
        rows: usize,
        cols: usize,
        spectrum: &[f32],
    ) -> PureResult<Tensor> {
        Self::map_tensor_from_interleaved(rows, cols, spectrum, |chunk| {
            [
                chunk[1].atan2(chunk[0]),
                chunk[3].atan2(chunk[2]),
                chunk[5].atan2(chunk[4]),
                chunk[7].atan2(chunk[6]),
            ]
        })
    }

    fn polar_tensors_from_interleaved(
        rows: usize,
        cols: usize,
        spectrum: &[f32],
    ) -> PureResult<(Tensor, Tensor)> {
        let expected_pairs = rows
            .checked_mul(cols)
            .ok_or(TensorError::TensorVolumeExceeded {
                label: "canvas_fft_polar",
                volume: rows.saturating_mul(cols),
                max_volume: usize::MAX,
            })?;
        let expected_len = expected_pairs
            .checked_mul(Self::FFT_INTERLEAVED_STRIDE)
            .ok_or(TensorError::TensorVolumeExceeded {
                label: "canvas_fft_polar",
                volume: rows
                    .saturating_mul(cols)
                    .saturating_mul(Self::FFT_INTERLEAVED_STRIDE),
                max_volume: usize::MAX,
            })?;

        if spectrum.len() != expected_len {
            return Err(TensorError::DataLength {
                expected: expected_len,
                got: spectrum.len(),
            });
        }

        let mut magnitudes = Vec::with_capacity(expected_pairs * Self::FFT_CHANNELS);
        let mut phases = Vec::with_capacity(expected_pairs * Self::FFT_CHANNELS);
        for chunk in spectrum.chunks_exact(Self::FFT_INTERLEAVED_STRIDE) {
            let (re_energy, im_energy) = (chunk[0], chunk[1]);
            let (re_chroma_r, im_chroma_r) = (chunk[2], chunk[3]);
            let (re_chroma_g, im_chroma_g) = (chunk[4], chunk[5]);
            let (re_chroma_b, im_chroma_b) = (chunk[6], chunk[7]);

            magnitudes.push(re_energy.hypot(im_energy));
            phases.push(im_energy.atan2(re_energy));

            magnitudes.push(re_chroma_r.hypot(im_chroma_r));
            phases.push(im_chroma_r.atan2(re_chroma_r));

            magnitudes.push(re_chroma_g.hypot(im_chroma_g));
            phases.push(im_chroma_g.atan2(re_chroma_g));

            magnitudes.push(re_chroma_b.hypot(im_chroma_b));
            phases.push(im_chroma_b.atan2(re_chroma_b));
        }

        let magnitude = Tensor::from_vec(rows, cols * Self::FFT_CHANNELS, magnitudes)?;
        let phase = Tensor::from_vec(rows, cols * Self::FFT_CHANNELS, phases)?;
        Ok((magnitude, phase))
    }

    fn map_power_tensor_from_interleaved<F>(
        rows: usize,
        cols: usize,
        spectrum: &[f32],
        mut map: F,
    ) -> PureResult<Tensor>
    where
        F: FnMut(f32) -> f32,
    {
        let expected_pairs = rows
            .checked_mul(cols)
            .ok_or(TensorError::TensorVolumeExceeded {
                label: "canvas_fft_power",
                volume: rows.saturating_mul(cols),
                max_volume: usize::MAX,
            })?;
        let expected_len = expected_pairs
            .checked_mul(Self::FFT_INTERLEAVED_STRIDE)
            .ok_or(TensorError::TensorVolumeExceeded {
                label: "canvas_fft_power",
                volume: rows
                    .saturating_mul(cols)
                    .saturating_mul(Self::FFT_INTERLEAVED_STRIDE),
                max_volume: usize::MAX,
            })?;

        if spectrum.len() != expected_len {
            return Err(TensorError::DataLength {
                expected: expected_len,
                got: spectrum.len(),
            });
        }

        let mut power = Vec::with_capacity(expected_pairs * Self::FFT_CHANNELS);
        for chunk in spectrum.chunks_exact(Self::FFT_INTERLEAVED_STRIDE) {
            let (re_energy, im_energy) = (chunk[0], chunk[1]);
            let (re_chroma_r, im_chroma_r) = (chunk[2], chunk[3]);
            let (re_chroma_g, im_chroma_g) = (chunk[4], chunk[5]);
            let (re_chroma_b, im_chroma_b) = (chunk[6], chunk[7]);

            let energy = re_energy.mul_add(re_energy, im_energy * im_energy);
            let chroma_r = re_chroma_r.mul_add(re_chroma_r, im_chroma_r * im_chroma_r);
            let chroma_g = re_chroma_g.mul_add(re_chroma_g, im_chroma_g * im_chroma_g);
            let chroma_b = re_chroma_b.mul_add(re_chroma_b, im_chroma_b * im_chroma_b);

            power.push(map(energy));
            power.push(map(chroma_r));
            power.push(map(chroma_g));
            power.push(map(chroma_b));
        }

        Tensor::from_vec(rows, cols * Self::FFT_CHANNELS, power)
    }

    fn power_tensor_from_interleaved(
        rows: usize,
        cols: usize,
        spectrum: &[f32],
    ) -> PureResult<Tensor> {
        Self::map_power_tensor_from_interleaved(rows, cols, spectrum, |power| power)
    }

    fn power_db_tensor_from_interleaved(
        rows: usize,
        cols: usize,
        spectrum: &[f32],
    ) -> PureResult<Tensor> {
        Self::map_power_tensor_from_interleaved(rows, cols, spectrum, |power| {
            let clamped = power.max(Self::POWER_DB_EPSILON);
            let db = 10.0 * clamped.log10();
            db.max(Self::POWER_DB_FLOOR)
        })
    }
}
/// Byte layout metadata for the WGSL canvas FFT pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CanvasFftLayout {
    field_bytes: usize,
    field_stride: usize,
    spectrum_bytes: usize,
    spectrum_stride: usize,
    uniform_bytes: usize,
}

impl CanvasFftLayout {
    /// Total byte length required for the vector field storage buffer.
    pub fn field_bytes(&self) -> usize {
        self.field_bytes
    }

    /// Size in bytes of each vector field sample.
    pub fn field_stride(&self) -> usize {
        self.field_stride
    }

    /// Total byte length required for the FFT spectrum storage buffer.
    pub fn spectrum_bytes(&self) -> usize {
        self.spectrum_bytes
    }

    /// Size in bytes of each FFT spectrum sample.
    pub fn spectrum_stride(&self) -> usize {
        self.spectrum_stride
    }

    /// Byte length of the uniform buffer used by the WGSL kernel.
    pub fn uniform_bytes(&self) -> usize {
        self.uniform_bytes
    }

    /// Number of pixels captured by the layout.
    pub fn pixel_count(&self) -> usize {
        if self.field_stride == 0 {
            0
        } else {
            self.field_bytes / self.field_stride
        }
    }
}

/// RGBA pixel surface that mirrors a HTML canvas.
#[derive(Clone, Debug)]
pub struct CanvasSurface {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

impl CanvasSurface {
    /// Create a canvas surface. Width/height must both be non-zero so the WASM
    /// side does not attempt to upload empty textures.
    pub fn new(width: usize, height: usize) -> PureResult<Self> {
        if width == 0 || height == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: height,
                cols: width,
            });
        }
        Ok(Self {
            width,
            height,
            pixels: vec![0; width * height * 4],
        })
    }

    /// Returns the raw RGBA pixel buffer ready for `ImageData`.
    pub fn as_rgba(&self) -> &[u8] {
        &self.pixels
    }

    /// Mutable access to the underlying buffer.
    pub fn as_rgba_mut(&mut self) -> &mut [u8] {
        &mut self.pixels
    }

    /// Clear the canvas to transparent black.
    pub fn clear(&mut self) {
        self.pixels.fill(0);
    }

    /// Paint a tensor onto the pixel grid using a simple blue-magenta gradient
    /// that emphasises relative relations instead of absolute brightness.
    pub fn paint_tensor(&mut self, tensor: &Tensor) -> PureResult<()> {
        let mut normalizer = CanvasNormalizer::default();
        self.paint_tensor_with_palette(tensor, &mut normalizer, CanvasPalette::default())
    }

    /// Paint the tensor using a caller-provided normaliser and palette. This
    /// keeps huge models stable while making it trivial to swap visual styles
    /// from Python or JavaScript glue code.
    pub fn paint_tensor_with_palette(
        &mut self,
        tensor: &Tensor,
        normalizer: &mut CanvasNormalizer,
        palette: CanvasPalette,
    ) -> PureResult<()> {
        let expected = (self.height, self.width);
        if tensor.shape() != expected {
            return Err(TensorError::ShapeMismatch {
                left: tensor.shape(),
                right: expected,
            });
        }

        let data = tensor.data();
        normalizer.update(data);
        for (idx, &value) in data.iter().enumerate() {
            let normalized = normalizer.normalize(value);
            let [r, g, b, a] = palette.map(normalized);
            let offset = idx * 4;
            self.pixels[offset] = r;
            self.pixels[offset + 1] = g;
            self.pixels[offset + 2] = b;
            self.pixels[offset + 3] = a;
        }
        Ok(())
    }

    /// Paint the tensor while simultaneously vectorising the chroma so that the
    /// caller can feed the result straight back into Z-space monads.
    pub fn paint_tensor_with_palette_into_vectors(
        &mut self,
        tensor: &Tensor,
        normalizer: &mut CanvasNormalizer,
        palette: CanvasPalette,
        vectors: &mut ColorVectorField,
    ) -> PureResult<()> {
        let expected = (self.height, self.width);
        if tensor.shape() != expected {
            return Err(TensorError::ShapeMismatch {
                left: tensor.shape(),
                right: expected,
            });
        }

        vectors.ensure_shape(self.width, self.height);
        let data = tensor.data();
        normalizer.update(data);
        for (idx, &value) in data.iter().enumerate() {
            let normalized = normalizer.normalize(value);
            let (rgba, chroma) = palette.map_vectorised(normalized);
            let offset = idx * 4;
            self.pixels[offset] = rgba[0];
            self.pixels[offset + 1] = rgba[1];
            self.pixels[offset + 2] = rgba[2];
            self.pixels[offset + 3] = rgba[3];
            vectors.set(idx, normalized, chroma);
        }
        Ok(())
    }

    /// Consume the surface returning the owned pixel vector.
    pub fn into_pixels(self) -> Vec<u8> {
        self.pixels
    }

    /// Canvas width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Canvas height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }
}

/// Drives the fractal scheduler into an on-canvas tensor without extra copies.
#[derive(Clone, Debug)]
pub struct CanvasProjector {
    scheduler: UringFractalScheduler,
    surface: CanvasSurface,
    workspace: Tensor,
    normalizer: CanvasNormalizer,
    palette: CanvasPalette,
    vectors: ColorVectorField,
}

/// Convenience wrapper that mirrors the WASM `FractalCanvas` bindings while
/// remaining entirely in-process for native smoke tests.
#[derive(Clone, Debug)]
pub struct FractalCanvas {
    projector: CanvasProjector,
    width: usize,
    height: usize,
}

impl FractalCanvas {
    /// Construct a projector-backed canvas with the requested queue capacity.
    pub fn new(capacity: usize, width: usize, height: usize) -> PureResult<Self> {
        let scheduler = UringFractalScheduler::new(capacity)?;
        let projector = CanvasProjector::new(scheduler, width, height)?;
        Ok(Self {
            projector,
            width,
            height,
        })
    }

    /// Canvas width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Canvas height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Immutable access to the projector.
    pub fn projector(&self) -> &CanvasProjector {
        &self.projector
    }

    /// Mutable access to the projector.
    pub fn projector_mut(&mut self) -> &mut CanvasProjector {
        &mut self.projector
    }

    /// Immutable access to the underlying scheduler.
    pub fn scheduler(&self) -> &UringFractalScheduler {
        self.projector.scheduler()
    }

    /// Mutable access to the underlying scheduler.
    pub fn scheduler_mut(&mut self) -> &mut UringFractalScheduler {
        self.projector.scheduler_mut()
    }

    /// Refresh the canvas returning the latest RGBA buffer.
    pub fn refresh(&mut self) -> PureResult<&[u8]> {
        self.projector.refresh()
    }

    /// Refresh the canvas returning both the RGBA buffer and the vector field.
    pub fn refresh_with_vectors(&mut self) -> PureResult<(&[u8], &ColorVectorField)> {
        self.projector.refresh_with_vectors()
    }

    /// Refresh the canvas returning the latest tensor relation.
    pub fn refresh_tensor(&mut self) -> PureResult<&Tensor> {
        self.projector.refresh_tensor()
    }

    /// Push a new fractal patch into the scheduler.
    pub fn push_patch(
        &self,
        relation: Tensor,
        coherence: f32,
        tension: f32,
        depth: u32,
    ) -> PureResult<()> {
        let patch = FractalPatch::new(relation, coherence, tension, depth)?;
        self.projector.scheduler().push(patch)
    }
}

impl CanvasProjector {
    /// Construct a projector with a shared scheduler. The workspace tensor is
    /// reused for every refresh call to avoid hitting the WASM allocator.
    pub fn new(scheduler: UringFractalScheduler, width: usize, height: usize) -> PureResult<Self> {
        Self::with_config(scheduler, width, height, CanvasProjectorConfig::default())
    }

    /// Construct a projector with an explicit config.
    pub fn with_config(
        scheduler: UringFractalScheduler,
        width: usize,
        height: usize,
        config: CanvasProjectorConfig,
    ) -> PureResult<Self> {
        let surface = CanvasSurface::new(width, height)?;
        let workspace = Tensor::zeros(height, width)?;
        let vectors = ColorVectorField::new(width, height);
        Ok(Self {
            scheduler,
            surface,
            workspace,
            normalizer: config.normalizer,
            palette: config.palette,
            vectors,
        })
    }

    /// Expose the scheduler so producers can push new relation patches.
    pub fn scheduler(&self) -> &UringFractalScheduler {
        &self.scheduler
    }

    /// Mutable access to the scheduler for advanced consumers.
    pub fn scheduler_mut(&mut self) -> &mut UringFractalScheduler {
        &mut self.scheduler
    }

    /// Immutable view of the canvas surface.
    pub fn surface(&self) -> &CanvasSurface {
        &self.surface
    }

    /// Mutable access to the canvas surface (useful for custom palettes).
    pub fn surface_mut(&mut self) -> &mut CanvasSurface {
        &mut self.surface
    }

    /// Mutable access to the normaliser used for colouring.
    pub fn normalizer_mut(&mut self) -> &mut CanvasNormalizer {
        &mut self.normalizer
    }

    /// Current palette.
    pub fn palette(&self) -> CanvasPalette {
        self.palette
    }

    /// Swap the palette used for the next refresh.
    pub fn set_palette(&mut self, palette: CanvasPalette) {
        self.palette = palette;
    }

    /// Refresh the canvas by folding the queued patches straight into the
    /// workspace tensor and painting the result. Callers can ship the returned
    /// slice to JavaScript without cloning.
    pub fn refresh(&mut self) -> PureResult<&[u8]> {
        self.render()?;
        Ok(self.surface.as_rgba())
    }

    /// Refresh the canvas and expose the latest relation tensor.
    pub fn refresh_tensor(&mut self) -> PureResult<&Tensor> {
        self.render()?;
        Ok(&self.workspace)
    }

    /// Returns the last relation tensor without forcing a refresh.
    pub fn tensor(&self) -> &Tensor {
        &self.workspace
    }

    fn render(&mut self) -> PureResult<()> {
        let relation = self.scheduler.fold_coherence()?;
        if relation.shape() != self.workspace.shape() {
            self.workspace = relation;
        } else {
            self.workspace.data_mut().copy_from_slice(relation.data());
        }
        self.surface.paint_tensor_with_palette_into_vectors(
            &self.workspace,
            &mut self.normalizer,
            self.palette,
            &mut self.vectors,
        )
    }

    /// Refresh the canvas returning both the RGBA buffer and the vector field
    /// used to bind colours back into Z-space dynamics.
    pub fn refresh_with_vectors(&mut self) -> PureResult<(&[u8], &ColorVectorField)> {
        self.render()?;
        Ok((self.surface.as_rgba(), &self.vectors))
    }

    /// Ensure the projector is up to date and expose the vector field.
    pub fn refresh_vector_field(&mut self) -> PureResult<&ColorVectorField> {
        self.render()?;
        Ok(&self.vectors)
    }

    /// Last computed vector field without forcing a refresh.
    pub fn vector_field(&self) -> &ColorVectorField {
        &self.vectors
    }

    /// Refresh the canvas and expose the interleaved FFT spectrum for each row
    /// (energy + chroma channels). When `inverse` is `true`, the spectrum is
    /// inverted before returning.
    pub fn refresh_vector_fft(&mut self, inverse: bool) -> PureResult<Vec<f32>> {
        self.render()?;
        self.vectors.fft_rows_interleaved(inverse)
    }

    /// Refresh the canvas and expose the row-wise FFT with a custom window.
    pub fn refresh_vector_fft_with_window(
        &mut self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Vec<f32>> {
        self.render()?;
        self.vectors
            .fft_rows_interleaved_with_window(window, inverse)
    }

    /// Refresh the canvas and expose the row-wise FFT magnitudes as a tensor
    /// with shape `(height, width * 4)`.
    pub fn refresh_vector_fft_magnitude_tensor(&mut self, inverse: bool) -> PureResult<Tensor> {
        self.render()?;
        self.vectors.fft_rows_magnitude_tensor(inverse)
    }

    /// Refresh and expose the row-wise FFT magnitudes with windowing.
    pub fn refresh_vector_fft_magnitude_tensor_with_window(
        &mut self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.render()?;
        self.vectors
            .fft_rows_magnitude_tensor_with_window(window, inverse)
    }

    /// Refresh the canvas and expose the row-wise FFT power as a tensor with
    /// shape `(height, width * 4)`.
    pub fn refresh_vector_fft_power_tensor(&mut self, inverse: bool) -> PureResult<Tensor> {
        self.render()?;
        self.vectors.fft_rows_power_tensor(inverse)
    }

    /// Refresh and expose the row-wise FFT power with windowing applied.
    pub fn refresh_vector_fft_power_tensor_with_window(
        &mut self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.render()?;
        self.vectors
            .fft_rows_power_tensor_with_window(window, inverse)
    }

    /// Refresh the canvas and expose the row-wise FFT log-power (decibel) tensor
    /// with shape `(height, width * 4)`.
    pub fn refresh_vector_fft_power_db_tensor(&mut self, inverse: bool) -> PureResult<Tensor> {
        self.render()?;
        self.vectors.fft_rows_power_db_tensor(inverse)
    }

    /// Refresh and expose the row-wise FFT log-power with windowing.
    pub fn refresh_vector_fft_power_db_tensor_with_window(
        &mut self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.render()?;
        self.vectors
            .fft_rows_power_db_tensor_with_window(window, inverse)
    }

    /// Refresh the canvas and expose the row-wise FFT phases as a tensor with
    /// shape `(height, width * 4)`.
    pub fn refresh_vector_fft_phase_tensor(&mut self, inverse: bool) -> PureResult<Tensor> {
        self.render()?;
        self.vectors.fft_rows_phase_tensor(inverse)
    }

    /// Refresh and expose the row-wise FFT phases with windowing applied.
    pub fn refresh_vector_fft_phase_tensor_with_window(
        &mut self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.render()?;
        self.vectors
            .fft_rows_phase_tensor_with_window(window, inverse)
    }

    /// Refresh the canvas and expose both the row-wise FFT magnitudes and
    /// phases as tensors with shape `(height, width * 4)`.
    pub fn refresh_vector_fft_polar_tensors(
        &mut self,
        inverse: bool,
    ) -> PureResult<(Tensor, Tensor)> {
        self.render()?;
        self.vectors.fft_rows_polar_tensors(inverse)
    }

    /// Refresh and expose the row-wise FFT magnitude/phase pair with windowing.
    pub fn refresh_vector_fft_polar_tensors_with_window(
        &mut self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<(Tensor, Tensor)> {
        self.render()?;
        self.vectors
            .fft_rows_polar_tensors_with_window(window, inverse)
    }

    /// Refresh the canvas and expose the interleaved FFT spectrum for each
    /// column (energy + chroma channels). This mirrors
    /// [`refresh_vector_fft`] but operates along the vertical axis so WASM
    /// consumers can probe anisotropic structures without reshaping data on the
    /// JavaScript side.
    pub fn refresh_vector_fft_columns(&mut self, inverse: bool) -> PureResult<Vec<f32>> {
        self.render()?;
        self.vectors.fft_cols_interleaved(inverse)
    }

    /// Refresh and expose the column-wise FFT with windowing applied vertically.
    pub fn refresh_vector_fft_columns_with_window(
        &mut self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Vec<f32>> {
        self.render()?;
        self.vectors
            .fft_cols_interleaved_with_window(window, inverse)
    }

    /// Refresh the canvas and expose column-wise FFT magnitudes as a tensor
    /// with shape `(width, height * 4)`.
    pub fn refresh_vector_fft_columns_magnitude_tensor(
        &mut self,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.render()?;
        self.vectors.fft_cols_magnitude_tensor(inverse)
    }

    /// Refresh and expose the column-wise FFT magnitudes with windowing.
    pub fn refresh_vector_fft_columns_magnitude_tensor_with_window(
        &mut self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.render()?;
        self.vectors
            .fft_cols_magnitude_tensor_with_window(window, inverse)
    }

    /// Refresh the canvas and expose column-wise FFT power as a tensor with
    /// shape `(width, height * 4)`.
    pub fn refresh_vector_fft_columns_power_tensor(&mut self, inverse: bool) -> PureResult<Tensor> {
        self.render()?;
        self.vectors.fft_cols_power_tensor(inverse)
    }

    /// Refresh and expose the column-wise FFT power with windowing.
    pub fn refresh_vector_fft_columns_power_tensor_with_window(
        &mut self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.render()?;
        self.vectors
            .fft_cols_power_tensor_with_window(window, inverse)
    }

    /// Refresh the canvas and expose the column-wise FFT log-power (decibel)
    /// tensor with shape `(width, height * 4)`.
    pub fn refresh_vector_fft_columns_power_db_tensor(
        &mut self,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.render()?;
        self.vectors.fft_cols_power_db_tensor(inverse)
    }

    /// Refresh and expose the column-wise FFT log-power with windowing.
    pub fn refresh_vector_fft_columns_power_db_tensor_with_window(
        &mut self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.render()?;
        self.vectors
            .fft_cols_power_db_tensor_with_window(window, inverse)
    }

    /// Refresh the canvas and expose column-wise FFT phases as a tensor with
    /// shape `(width, height * 4)`.
    pub fn refresh_vector_fft_columns_phase_tensor(&mut self, inverse: bool) -> PureResult<Tensor> {
        self.render()?;
        self.vectors.fft_cols_phase_tensor(inverse)
    }

    /// Refresh and expose the column-wise FFT phases with windowing.
    pub fn refresh_vector_fft_columns_phase_tensor_with_window(
        &mut self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.render()?;
        self.vectors
            .fft_cols_phase_tensor_with_window(window, inverse)
    }

    /// Refresh the canvas and expose both the column-wise FFT magnitudes and
    /// phases as tensors with shape `(width, height * 4)`.
    pub fn refresh_vector_fft_columns_polar_tensors(
        &mut self,
        inverse: bool,
    ) -> PureResult<(Tensor, Tensor)> {
        self.render()?;
        self.vectors.fft_cols_polar_tensors(inverse)
    }

    /// Refresh and expose the column-wise FFT magnitude/phase with windowing.
    pub fn refresh_vector_fft_columns_polar_tensors_with_window(
        &mut self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<(Tensor, Tensor)> {
        self.render()?;
        self.vectors
            .fft_cols_polar_tensors_with_window(window, inverse)
    }

    /// Refresh the canvas and expose the full 2D FFT spectrum (energy + chroma
    /// channels). This applies the row and column transforms sequentially so
    /// integrators can probe anisotropic features without piecing together two
    /// separate passes on the JavaScript side.
    pub fn refresh_vector_fft_2d(&mut self, inverse: bool) -> PureResult<Vec<f32>> {
        self.render()?;
        self.vectors.fft_2d_interleaved(inverse)
    }

    /// Refresh and expose the 2D FFT with windowing along both axes.
    pub fn refresh_vector_fft_2d_with_window(
        &mut self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Vec<f32>> {
        self.render()?;
        self.vectors.fft_2d_interleaved_with_window(window, inverse)
    }

    /// Refresh the canvas and expose the 2D FFT magnitudes as a tensor with
    /// shape `(height, width * 4)`.
    pub fn refresh_vector_fft_2d_magnitude_tensor(&mut self, inverse: bool) -> PureResult<Tensor> {
        self.render()?;
        self.vectors.fft_2d_magnitude_tensor(inverse)
    }

    /// Refresh and expose the 2D FFT magnitudes with windowing.
    pub fn refresh_vector_fft_2d_magnitude_tensor_with_window(
        &mut self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.render()?;
        self.vectors
            .fft_2d_magnitude_tensor_with_window(window, inverse)
    }

    /// Refresh the canvas and expose the 2D FFT power as a tensor with shape
    /// `(height, width * 4)`.
    pub fn refresh_vector_fft_2d_power_tensor(&mut self, inverse: bool) -> PureResult<Tensor> {
        self.render()?;
        self.vectors.fft_2d_power_tensor(inverse)
    }

    /// Refresh and expose the 2D FFT power with windowing.
    pub fn refresh_vector_fft_2d_power_tensor_with_window(
        &mut self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.render()?;
        self.vectors
            .fft_2d_power_tensor_with_window(window, inverse)
    }

    /// Refresh the canvas and expose the 2D FFT log-power (decibel) tensor with
    /// shape `(height, width * 4)`.
    pub fn refresh_vector_fft_2d_power_db_tensor(&mut self, inverse: bool) -> PureResult<Tensor> {
        self.render()?;
        self.vectors.fft_2d_power_db_tensor(inverse)
    }

    /// Refresh and expose the 2D FFT log-power with windowing.
    pub fn refresh_vector_fft_2d_power_db_tensor_with_window(
        &mut self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.render()?;
        self.vectors
            .fft_2d_power_db_tensor_with_window(window, inverse)
    }

    /// Refresh the canvas and expose the 2D FFT phases as a tensor with shape
    /// `(height, width * 4)`.
    pub fn refresh_vector_fft_2d_phase_tensor(&mut self, inverse: bool) -> PureResult<Tensor> {
        self.render()?;
        self.vectors.fft_2d_phase_tensor(inverse)
    }

    /// Refresh and expose the 2D FFT phases with windowing.
    pub fn refresh_vector_fft_2d_phase_tensor_with_window(
        &mut self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.render()?;
        self.vectors
            .fft_2d_phase_tensor_with_window(window, inverse)
    }

    /// Refresh the canvas and expose both the 2D FFT magnitudes and phases as
    /// tensors with shape `(height, width * 4)`.
    pub fn refresh_vector_fft_2d_polar_tensors(
        &mut self,
        inverse: bool,
    ) -> PureResult<(Tensor, Tensor)> {
        self.render()?;
        self.vectors.fft_2d_polar_tensors(inverse)
    }

    /// Refresh and expose the 2D FFT magnitude/phase pair with windowing.
    pub fn refresh_vector_fft_2d_polar_tensors_with_window(
        &mut self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<(Tensor, Tensor)> {
        self.render()?;
        self.vectors
            .fft_2d_polar_tensors_with_window(window, inverse)
    }

    /// Accumulate the refreshed tensor into the provided hypergradient tape.
    pub fn accumulate_hypergrad(&mut self, tape: &mut AmegaHypergrad) -> PureResult<()> {
        let tensor = self.refresh_tensor()?;
        tape.accumulate_wave(tensor)
    }

    /// Accumulate the refreshed tensor into the provided Euclidean gradient tape.
    pub fn accumulate_realgrad(&mut self, tape: &mut AmegaRealgrad) -> PureResult<()> {
        let tensor = self.refresh_tensor()?;
        tape.accumulate_wave(tensor)
    }

    /// Refresh the canvas and return gradient summary statistics for both the
    /// hypergradient and Euclidean tapes. The returned tuple packs
    /// `(hypergrad_summary, realgrad_summary)`.
    pub fn gradient_summary(
        &mut self,
        curvature: f32,
    ) -> PureResult<(GradientSummary, GradientSummary)> {
        let tensor = self.refresh_tensor()?;
        let (rows, cols) = tensor.shape();
        let mut hypergrad = AmegaHypergrad::new(curvature, 1.0, rows, cols)?;
        hypergrad.accumulate_wave(tensor)?;
        let mut realgrad = AmegaRealgrad::new(1.0, rows, cols)?;
        realgrad.accumulate_wave(tensor)?;
        Ok((hypergrad.summary(), realgrad.summary()))
    }

    /// Interpret the gradient summaries into Desire-aligned feedback signals.
    pub fn gradient_interpretation(
        &mut self,
        curvature: f32,
    ) -> PureResult<DesireGradientInterpretation> {
        let (hyper, real) = self.gradient_summary(curvature)?;
        Ok(DesireGradientInterpretation::from_summaries(hyper, real))
    }

    /// Derive Desire control signals directly from the refreshed canvas tensor.
    pub fn gradient_control(&mut self, curvature: f32) -> PureResult<DesireGradientControl> {
        let interpretation = self.gradient_interpretation(curvature)?;
        Ok(interpretation.control())
    }

    /// Access the last computed FFT spectrum without forcing a refresh.
    pub fn vector_fft(&self, inverse: bool) -> PureResult<Vec<f32>> {
        self.vectors.fft_rows_interleaved(inverse)
    }

    /// Last computed row-wise FFT spectrum with a custom window.
    pub fn vector_fft_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Vec<f32>> {
        self.vectors
            .fft_rows_interleaved_with_window(window, inverse)
    }

    /// Last computed row-wise FFT magnitudes without forcing a refresh.
    pub fn vector_fft_magnitude_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        self.vectors.fft_rows_magnitude_tensor(inverse)
    }

    /// Last computed row-wise FFT magnitudes with windowing.
    pub fn vector_fft_magnitude_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.vectors
            .fft_rows_magnitude_tensor_with_window(window, inverse)
    }

    /// Last computed row-wise FFT power without forcing a refresh.
    pub fn vector_fft_power_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        self.vectors.fft_rows_power_tensor(inverse)
    }

    /// Last computed row-wise FFT power with windowing.
    pub fn vector_fft_power_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.vectors
            .fft_rows_power_tensor_with_window(window, inverse)
    }

    /// Last computed row-wise FFT log-power tensor without forcing a refresh.
    pub fn vector_fft_power_db_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        self.vectors.fft_rows_power_db_tensor(inverse)
    }

    /// Last computed row-wise FFT log-power with windowing.
    pub fn vector_fft_power_db_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.vectors
            .fft_rows_power_db_tensor_with_window(window, inverse)
    }

    /// Last computed row-wise FFT phases without forcing a refresh.
    pub fn vector_fft_phase_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        self.vectors.fft_rows_phase_tensor(inverse)
    }

    /// Last computed row-wise FFT phases with windowing.
    pub fn vector_fft_phase_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.vectors
            .fft_rows_phase_tensor_with_window(window, inverse)
    }

    /// Access both the row-wise FFT magnitudes and phases without forcing a
    /// refresh. Returns `(magnitude, phase)` tensors with shape
    /// `(height, width * 4)`.
    pub fn vector_fft_polar_tensors(&self, inverse: bool) -> PureResult<(Tensor, Tensor)> {
        self.vectors.fft_rows_polar_tensors(inverse)
    }

    /// Last computed row-wise FFT magnitude/phase pair with windowing.
    pub fn vector_fft_polar_tensors_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<(Tensor, Tensor)> {
        self.vectors
            .fft_rows_polar_tensors_with_window(window, inverse)
    }

    /// Access the last computed column-wise FFT spectrum without forcing a
    /// refresh. The returned buffer mirrors [`refresh_vector_fft_columns`]
    /// layout (columns laid out sequentially with interleaved `[re, im]`
    /// components).
    pub fn vector_fft_columns(&self, inverse: bool) -> PureResult<Vec<f32>> {
        self.vectors.fft_cols_interleaved(inverse)
    }

    /// Last computed column-wise FFT spectrum with windowing.
    pub fn vector_fft_columns_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Vec<f32>> {
        self.vectors
            .fft_cols_interleaved_with_window(window, inverse)
    }

    /// Access the last computed column-wise FFT magnitudes without forcing a
    /// refresh.
    pub fn vector_fft_columns_magnitude_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        self.vectors.fft_cols_magnitude_tensor(inverse)
    }

    /// Last computed column-wise FFT magnitudes with windowing.
    pub fn vector_fft_columns_magnitude_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.vectors
            .fft_cols_magnitude_tensor_with_window(window, inverse)
    }

    /// Access the last computed column-wise FFT power without forcing a refresh.
    pub fn vector_fft_columns_power_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        self.vectors.fft_cols_power_tensor(inverse)
    }

    /// Last computed column-wise FFT power with windowing.
    pub fn vector_fft_columns_power_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.vectors
            .fft_cols_power_tensor_with_window(window, inverse)
    }

    /// Access the last computed column-wise FFT log-power tensor without
    /// forcing a refresh.
    pub fn vector_fft_columns_power_db_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        self.vectors.fft_cols_power_db_tensor(inverse)
    }

    /// Last computed column-wise FFT log-power with windowing.
    pub fn vector_fft_columns_power_db_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.vectors
            .fft_cols_power_db_tensor_with_window(window, inverse)
    }

    /// Access the last computed column-wise FFT phases without forcing a
    /// refresh.
    pub fn vector_fft_columns_phase_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        self.vectors.fft_cols_phase_tensor(inverse)
    }

    /// Last computed column-wise FFT phases with windowing.
    pub fn vector_fft_columns_phase_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.vectors
            .fft_cols_phase_tensor_with_window(window, inverse)
    }

    /// Access both the column-wise FFT magnitudes and phases without forcing a
    /// refresh. Returns `(magnitude, phase)` tensors with shape
    /// `(width, height * 4)`.
    pub fn vector_fft_columns_polar_tensors(&self, inverse: bool) -> PureResult<(Tensor, Tensor)> {
        self.vectors.fft_cols_polar_tensors(inverse)
    }

    /// Last computed column-wise FFT magnitude/phase pair with windowing.
    pub fn vector_fft_columns_polar_tensors_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<(Tensor, Tensor)> {
        self.vectors
            .fft_cols_polar_tensors_with_window(window, inverse)
    }

    /// Access the last computed 2D FFT spectrum without forcing a refresh. The
    /// returned buffer matches [`refresh_vector_fft_2d`] and can be fed
    /// directly into GPU upload pipelines.
    pub fn vector_fft_2d(&self, inverse: bool) -> PureResult<Vec<f32>> {
        self.vectors.fft_2d_interleaved(inverse)
    }

    /// Last computed 2D FFT spectrum with windowing.
    pub fn vector_fft_2d_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Vec<f32>> {
        self.vectors.fft_2d_interleaved_with_window(window, inverse)
    }

    /// Access the last computed 2D FFT magnitudes without forcing a refresh.
    pub fn vector_fft_2d_magnitude_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        self.vectors.fft_2d_magnitude_tensor(inverse)
    }

    /// Last computed 2D FFT magnitudes with windowing.
    pub fn vector_fft_2d_magnitude_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.vectors
            .fft_2d_magnitude_tensor_with_window(window, inverse)
    }

    /// Access the last computed 2D FFT power without forcing a refresh.
    pub fn vector_fft_2d_power_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        self.vectors.fft_2d_power_tensor(inverse)
    }

    /// Last computed 2D FFT power with windowing.
    pub fn vector_fft_2d_power_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.vectors
            .fft_2d_power_tensor_with_window(window, inverse)
    }

    /// Access the last computed 2D FFT log-power tensor without forcing a refresh.
    pub fn vector_fft_2d_power_db_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        self.vectors.fft_2d_power_db_tensor(inverse)
    }

    /// Last computed 2D FFT log-power with windowing.
    pub fn vector_fft_2d_power_db_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.vectors
            .fft_2d_power_db_tensor_with_window(window, inverse)
    }

    /// Access the last computed 2D FFT phases without forcing a refresh.
    pub fn vector_fft_2d_phase_tensor(&self, inverse: bool) -> PureResult<Tensor> {
        self.vectors.fft_2d_phase_tensor(inverse)
    }

    /// Last computed 2D FFT phases with windowing.
    pub fn vector_fft_2d_phase_tensor_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<Tensor> {
        self.vectors
            .fft_2d_phase_tensor_with_window(window, inverse)
    }

    /// Access both the 2D FFT magnitudes and phases without forcing a refresh.
    /// Returns `(magnitude, phase)` tensors with shape `(height, width * 4)`.
    pub fn vector_fft_2d_polar_tensors(&self, inverse: bool) -> PureResult<(Tensor, Tensor)> {
        self.vectors.fft_2d_polar_tensors(inverse)
    }

    /// Last computed 2D FFT magnitude/phase pair with windowing.
    pub fn vector_fft_2d_polar_tensors_with_window(
        &self,
        window: CanvasWindow,
        inverse: bool,
    ) -> PureResult<(Tensor, Tensor)> {
        self.vectors
            .fft_2d_polar_tensors_with_window(window, inverse)
    }

    /// Uniform parameters expected by [`vector_fft_wgsl`]. The layout mirrors
    /// the WGSL `CanvasFftParams` struct and includes padding so the buffer
    /// occupies 16 bytes.
    pub fn vector_fft_uniform(&self, inverse: bool) -> [u32; 4] {
        [
            self.surface.width() as u32,
            self.surface.height() as u32,
            inverse as u32,
            0,
        ]
    }

    /// Byte layout metadata that mirrors the WGSL buffers emitted by
    /// [`vector_fft_wgsl`]. Callers can size storage/uniform buffers directly
    /// from the returned values without hard-coding struct sizes.
    pub fn vector_fft_layout(&self) -> CanvasFftLayout {
        const FIELD_STRIDE: usize = 4 * core::mem::size_of::<f32>();
        const SPECTRUM_STRIDE: usize = 8 * core::mem::size_of::<f32>();
        const UNIFORM_BYTES: usize = 4 * core::mem::size_of::<u32>();

        let width = self.surface.width();
        let height = self.surface.height();
        CanvasFftLayout {
            field_bytes: width * height * FIELD_STRIDE,
            field_stride: FIELD_STRIDE,
            spectrum_bytes: width * height * SPECTRUM_STRIDE,
            spectrum_stride: SPECTRUM_STRIDE,
            uniform_bytes: UNIFORM_BYTES,
        }
    }

    /// Suggested dispatch dimensions for [`vector_fft_wgsl`]. The kernel
    /// operates over the full canvas grid, so we pack the height into the
    /// `y`-dimension while the `x`-dimension is chunked by the workgroup size.
    /// Consumers can feed the returned triplet directly into
    /// `queue.write_buffer` / `compute_pass.dispatch_workgroups` without
    /// recomputing the ceil division in JavaScript.
    pub fn vector_fft_dispatch(&self, subgroup: bool) -> [u32; 3] {
        let width = self.surface.width() as u32;
        let height = self.surface.height() as u32;
        let workgroup = if subgroup { 32 } else { 64 };
        let groups_x = if width == 0 {
            0
        } else {
            (width + workgroup - 1) / workgroup
        };
        [groups_x, height, 1]
    }

    /// Uniform parameters for the hypergradient WGSL operator. The layout packs
    /// the canvas dimensions alongside the blend/gain controls as four floats
    /// (16 bytes) so WebGPU callers can upload them without manual padding.
    pub fn hypergrad_operator_uniform(&self, mix: f32, gain: f32) -> [f32; 4] {
        [
            self.surface.width() as f32,
            self.surface.height() as f32,
            mix.clamp(0.0, 1.0),
            gain,
        ]
    }

    /// Convenience wrapper that feeds Desire's control signals straight into the
    /// hypergradient WGSL uniform layout.
    pub fn hypergrad_operator_uniform_from_control(
        &self,
        control: &DesireGradientControl,
    ) -> [f32; 4] {
        self.hypergrad_operator_uniform(control.operator_mix(), control.operator_gain())
    }

    /// Pack Desire's control feedback into a 16-float uniform suitable for WGSL
    /// consumption. The layout keeps every block aligned to 16 bytes so WebGPU
    /// callers can upload it without manual padding or serde churn.
    pub fn desire_control_uniform(&self, control: &DesireGradientControl) -> [u32; 16] {
        [
            control.target_entropy().to_bits(),
            control.learning_rate_eta().to_bits(),
            control.learning_rate_min().to_bits(),
            control.learning_rate_max().to_bits(),
            control.learning_rate_slew().to_bits(),
            control.clip_norm().to_bits(),
            control.clip_floor().to_bits(),
            control.clip_ceiling().to_bits(),
            control.clip_ema().to_bits(),
            control.temperature_kappa().to_bits(),
            control.temperature_slew().to_bits(),
            control.hyper_rate_scale().to_bits(),
            control.real_rate_scale().to_bits(),
            control.tuning_gain().to_bits(),
            control.quality_gain().to_bits(),
            control.events().bits(),
        ]
    }

    /// Compute the workgroup triplet for the hypergradient WGSL operator.
    pub fn hypergrad_operator_dispatch(&self, subgroup: bool) -> [u32; 3] {
        let width = self.surface.width() as u32;
        let height = self.surface.height() as u32;
        let workgroup = if subgroup { 32 } else { 64 };
        let groups_x = if width == 0 {
            0
        } else {
            (width + workgroup - 1) / workgroup
        };
        [groups_x, height, 1]
    }

    /// Emit a WGSL kernel that accumulates the canvas relation directly into a
    /// hypergradient buffer entirely on the GPU.
    pub fn hypergrad_operator_wgsl(&self, subgroup: bool) -> String {
        emit_canvas_hypergrad_wgsl(
            self.surface.width() as u32,
            self.surface.height() as u32,
            subgroup,
        )
    }

    /// Emit a WGSL kernel that mirrors [`refresh_vector_fft`] so GPU/WebGPU
    /// callers can reproduce the spectrum without leaving the browser. The
    /// shader expects the following bindings:
    ///
    /// - `@group(0) @binding(0)`: storage buffer containing one
    ///   `FieldSample {{ energy: f32, chroma: vec3<f32> }}` per pixel laid out
    ///   in row-major order.
    /// - `@group(0) @binding(1)`: storage buffer containing one
    ///   `SpectrumSample` per pixel (output – 8 floats for the complex energy
    ///   and chroma channels).
    /// - `@group(0) @binding(2)`: uniform `CanvasFftParams` with the canvas
    ///   `width`, `height`, and an `inverse` flag (1 = inverse, 0 = forward)
    ///   plus one padding lane so the struct spans 16 bytes.
    pub fn vector_fft_wgsl(&self, subgroup: bool) -> String {
        emit_canvas_fft_wgsl(
            self.surface.width() as u32,
            self.surface.height() as u32,
            subgroup,
        )
    }

    /// Refresh the canvas and return the FFT spectrum as a tensor with shape
    /// `(height, width * 8)`.
    pub fn refresh_vector_fft_tensor(&mut self, inverse: bool) -> PureResult<Tensor> {
        self.render()?;
        self.vectors.fft_rows_tensor(inverse)
    }

    /// Refresh the canvas and return the column-wise FFT spectrum as a tensor.
    /// The resulting tensor has shape `(width, height * 8)` and matches the
    /// interleaved layout returned by [`refresh_vector_fft_columns`].
    pub fn refresh_vector_fft_columns_tensor(&mut self, inverse: bool) -> PureResult<Tensor> {
        self.render()?;
        self.vectors.fft_cols_tensor(inverse)
    }

    /// Refresh the canvas and return the full 2D FFT spectrum as a tensor with
    /// shape `(height, width * 8)`.
    pub fn refresh_vector_fft_2d_tensor(&mut self, inverse: bool) -> PureResult<Tensor> {
        self.render()?;
        self.vectors.fft_2d_tensor(inverse)
    }

    /// Emit a Z-space fractal patch built from the colour energy field so
    /// higher-level schedulers can feed the canvas feedback loop.
    pub fn emit_zspace_patch(
        &mut self,
        coherence: f32,
        tension: f32,
        depth: u32,
    ) -> PureResult<FractalPatch> {
        self.render()?;
        self.vectors.to_zspace_patch(coherence, tension, depth)
    }
}

/// Projector construction parameters so integrators (including Python glue)
/// can pick palettes and stability behaviour explicitly.
#[derive(Clone, Debug)]
pub struct CanvasProjectorConfig {
    pub normalizer: CanvasNormalizer,
    pub palette: CanvasPalette,
}

impl Default for CanvasProjectorConfig {
    fn default() -> Self {
        Self {
            normalizer: CanvasNormalizer::default(),
            palette: CanvasPalette::default(),
        }
    }
}

fn compute_fft(line: &mut [Complex32], inverse: bool) -> PureResult<()> {
    if line.is_empty() {
        return Err(TensorError::EmptyInput("canvas_fft"));
    }

    if line.len().is_power_of_two() {
        fft::fft_inplace(line, inverse).map_err(|err| match err {
            fft::FftError::Empty => TensorError::EmptyInput("canvas_fft"),
            fft::FftError::NonPowerOfTwo => TensorError::InvalidDimensions {
                rows: 1,
                cols: line.len(),
            },
        })?;
        return Ok(());
    }

    let len = line.len();
    let mut output = vec![Complex32::default(); len];
    let sign = if inverse { 1.0 } else { -1.0 };
    for k in 0..len {
        let mut acc = Complex32::default();
        for (n, value) in line.iter().enumerate() {
            let angle = 2.0 * PI * k as f32 * n as f32 / len as f32 * sign;
            let twiddle = Complex32::new(angle.cos(), angle.sin());
            acc = acc.add(value.mul(twiddle));
        }
        if inverse {
            acc = acc.scale(1.0 / len as f32);
        }
        output[k] = acc;
    }
    line.copy_from_slice(&output);
    Ok(())
}

fn emit_canvas_fft_wgsl(width: u32, height: u32, subgroup: bool) -> String {
    let workgroup = if subgroup { 32 } else { 64 };
    format!(
        "// Canvas vector FFT WGSL (width {width}, height {height})\n\
         const WORKGROUP_SIZE: u32 = {workgroup}u;\n\
         struct FieldSample {{\n\
             energy: f32,\n\
             chroma: vec3<f32>,\n\
         }};\n\
         struct SpectrumSample {{\n\
             energy: vec2<f32>,\n\
             chroma_r: vec2<f32>,\n\
             chroma_g: vec2<f32>,\n\
             chroma_b: vec2<f32>,\n\
         }};\n\
         struct CanvasFftParams {{\n\
             width: u32,\n\
             height: u32,\n\
             inverse: u32,\n\
             _pad: u32,\n\
         }};\n\
         @group(0) @binding(0) var<storage, read> field: array<FieldSample>;\n\
         @group(0) @binding(1) var<storage, read_write> spectrum: array<SpectrumSample>;\n\
         @group(0) @binding(2) var<uniform> params: CanvasFftParams;\n\
         fn twiddle(angle: f32, inverse: bool) -> vec2<f32> {{\n\
             let cos_a = cos(angle);\n\
             let sin_a = sin(angle);\n\
             let sign = select(-1.0, 1.0, inverse);\n\
             return vec2<f32>(cos_a, sign * sin_a);\n\
         }}\n\
         fn accumulate(acc: vec2<f32>, sample: f32, tw: vec2<f32>) -> vec2<f32> {{\n\
             return vec2<f32>(acc.x + sample * tw.x, acc.y + sample * tw.y);\n\
         }}\n\
         @compute @workgroup_size(WORKGROUP_SIZE)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
             if (gid.x >= params.width || gid.y >= params.height) {{\n\
                 return;\n\
             }}\n\
             let row_offset = gid.y * params.width;\n\
             let inverse = params.inverse == 1u;\n\
             let frequency = f32(gid.x);\n\
             let norm = select(1.0, 1.0 / f32(params.width), inverse);\n\
             var energy = vec2<f32>(0.0, 0.0);\n\
             var chroma_r = vec2<f32>(0.0, 0.0);\n\
             var chroma_g = vec2<f32>(0.0, 0.0);\n\
             var chroma_b = vec2<f32>(0.0, 0.0);\n\
             for (var n = 0u; n < params.width; n = n + 1u) {{\n\
                 let sample = field[row_offset + n];\n\
                 let angle = 6.2831855 * frequency * f32(n) / f32(params.width);\n\
                 let tw = twiddle(angle, inverse);\n\
                 energy = accumulate(energy, sample.energy, tw);\n\
                 chroma_r = accumulate(chroma_r, sample.chroma.x, tw);\n\
                 chroma_g = accumulate(chroma_g, sample.chroma.y, tw);\n\
                 chroma_b = accumulate(chroma_b, sample.chroma.z, tw);\n\
             }}\n\
             spectrum[row_offset + gid.x] = SpectrumSample(\n\
                 energy * norm,\n\
                 chroma_r * norm,\n\
                 chroma_g * norm,\n\
                 chroma_b * norm,\n\
             );\n\
         }}\n",
        width = width,
        height = height,
        workgroup = workgroup,
    )
}

fn emit_canvas_hypergrad_wgsl(width: u32, height: u32, subgroup: bool) -> String {
    let workgroup = if subgroup { 32 } else { 64 };
    format!(
        "// Canvas hypergrad operator WGSL (width {width}, height {height})\n\
         const WORKGROUP_SIZE: u32 = {workgroup}u;\n\
         struct CanvasHypergradParams {{\n\
             width: f32;\n\
             height: f32;\n\
             mix: f32;\n\
             gain: f32;\n\
         }};\n\
         @group(0) @binding(0) var<storage, read> relation: array<f32>;\n\
         @group(0) @binding(1) var<storage, read_write> hypergrad: array<f32>;\n\
         @group(0) @binding(2) var<uniform> params: CanvasHypergradParams;\n\
         @compute @workgroup_size(WORKGROUP_SIZE)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
             let width = u32(params.width + 0.5);\n\
             let height = u32(params.height + 0.5);\n\
             if (gid.x >= width || gid.y >= height) {{\n\
                 return;\n\
             }}\n\
             let index = gid.y * width + gid.x;\n\
             let blend = clamp(params.mix, 0.0, 1.0);\n\
             let gain = params.gain;\n\
             let current = hypergrad[index];\n\
             let update = relation[index] * gain;\n\
             hypergrad[index] = current * (1.0 - blend) + update;\n\
         }}\n",
        width = width,
        height = height,
        workgroup = workgroup,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pure::fractal::FractalPatch;

    fn tensor(values: &[f32]) -> Tensor {
        Tensor::from_vec(1, values.len(), values.to_vec()).unwrap()
    }

    fn tensor_with_shape(rows: usize, cols: usize, values: &[f32]) -> Tensor {
        assert_eq!(rows * cols, values.len());
        Tensor::from_vec(rows, cols, values.to_vec()).unwrap()
    }

    #[test]
    fn canvas_window_rectangular_returns_unity() {
        let coeffs = CanvasWindow::Rectangular.coefficients(6);
        assert_eq!(coeffs.len(), 6);
        for value in coeffs {
            assert!((value - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn canvas_window_hann_tapers_ends() {
        let coeffs = CanvasWindow::Hann.coefficients(8);
        assert_eq!(coeffs.len(), 8);
        assert!(coeffs.first().unwrap().abs() < 1e-6);
        assert!(coeffs.last().unwrap().abs() < 1e-6);
        let middle = coeffs[coeffs.len() / 2];
        assert!(middle > 0.9);
    }

    fn seeded_color_field(width: usize, height: usize) -> ColorVectorField {
        let mut field = ColorVectorField::new(width, height);
        for row in 0..height {
            for col in 0..width {
                let idx = row * width + col;
                let energy = idx as f32 / 10.0;
                let chroma = [0.1 * (idx as f32 + 1.0), -0.05 * (idx as f32 + 1.0), 0.2];
                field.set(idx, energy, chroma);
            }
        }
        field
    }

    #[test]
    fn rectangular_window_matches_row_fft_baseline() {
        let field = seeded_color_field(4, 3);
        let baseline = field.fft_rows_interleaved(false).unwrap();
        let windowed = field
            .fft_rows_interleaved_with_window(CanvasWindow::Rectangular, false)
            .unwrap();
        for (lhs, rhs) in baseline.iter().zip(windowed.iter()) {
            assert!((lhs - rhs).abs() < 1e-5);
        }
    }

    #[test]
    fn rectangular_window_matches_column_fft_baseline() {
        let field = seeded_color_field(5, 2);
        let baseline = field.fft_cols_interleaved(false).unwrap();
        let windowed = field
            .fft_cols_interleaved_with_window(CanvasWindow::Rectangular, false)
            .unwrap();
        for (lhs, rhs) in baseline.iter().zip(windowed.iter()) {
            assert!((lhs - rhs).abs() < 1e-5);
        }
    }

    #[test]
    fn rectangular_window_matches_fft_2d_baseline() {
        let field = seeded_color_field(4, 3);
        let baseline = field.fft_2d_interleaved(false).unwrap();
        let windowed = field
            .fft_2d_interleaved_with_window(CanvasWindow::Rectangular, false)
            .unwrap();
        for (lhs, rhs) in baseline.iter().zip(windowed.iter()) {
            assert!((lhs - rhs).abs() < 1e-5);
        }
    }

    #[test]
    fn normaliser_handles_constant_input() {
        let mut normaliser = CanvasNormalizer::new(0.5, 1e-3);
        let zeros = [0.0; 8];
        let (min, max) = normaliser.update(&zeros);
        assert!((min - 0.0).abs() < 1e-6);
        assert!((max - 1e-3).abs() < 1e-6);
        assert_eq!(normaliser.normalize(0.0), 0.0);
        assert_eq!(normaliser.normalize(1.0), 1.0);
    }

    #[test]
    fn palette_switch_applies_without_allocation() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(2, 2).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 2, 2).unwrap();
        projector.set_palette(CanvasPalette::Grayscale);
        let bytes = projector.refresh().unwrap();
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn refresh_with_vectors_surfaces_color_field() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(2, 2).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 2, 2).unwrap();
        let (_bytes, field) = projector.refresh_with_vectors().unwrap();
        assert_eq!(field.vectors().len(), 4);
        for vector in field.iter() {
            assert!(vector[0] >= 0.0 && vector[0] <= 1.0);
        }
    }

    #[test]
    fn vector_fft_wgsl_covers_expected_bindings() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(2, 2).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let projector = CanvasProjector::new(scheduler, 4, 2).unwrap();
        let wgsl = projector.vector_fft_wgsl(false);
        assert!(wgsl.contains("@binding(2)"));
        assert!(wgsl.contains("SpectrumSample"));
        assert!(wgsl.contains("@compute"));
    }

    #[test]
    fn vector_fft_uniform_matches_canvas_dimensions() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(2, 2).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let projector = CanvasProjector::new(scheduler, 3, 5).unwrap();
        let params = projector.vector_fft_uniform(true);
        assert_eq!(params, [3, 5, 1, 0]);
    }

    #[test]
    fn vector_fft_dispatch_respects_workgroup_chunks() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(2, 2).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let projector = CanvasProjector::new(scheduler, 130, 4).unwrap();
        assert_eq!(projector.vector_fft_dispatch(false), [3, 4, 1]);
        assert_eq!(projector.vector_fft_dispatch(true), [5, 4, 1]);
    }

    #[test]
    fn vector_fft_layout_matches_wgsl_structs() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        let projector = CanvasProjector::new(scheduler, 64, 3).unwrap();
        let layout = projector.vector_fft_layout();
        assert_eq!(layout.field_stride(), 16);
        assert_eq!(layout.spectrum_stride(), 32);
        assert_eq!(layout.uniform_bytes(), 16);
        assert_eq!(layout.field_bytes(), 64 * 3 * 16);
        assert_eq!(layout.spectrum_bytes(), 64 * 3 * 32);
        assert_eq!(layout.pixel_count(), 64 * 3);
    }

    #[test]
    fn vector_field_fft_emits_interleaved_channels() {
        let mut field = ColorVectorField::new(4, 1);
        field.set(0, 1.0, [0.0, 0.0, 0.0]);
        let spectrum = field.fft_rows_interleaved(false).unwrap();
        assert_eq!(spectrum.len(), 32);
        for chunk in spectrum.chunks_exact(8) {
            assert!((chunk[0] - 1.0).abs() < 1e-6);
            assert!(chunk[1].abs() < 1e-6);
            for value in &chunk[2..] {
                assert!(value.abs() < 1e-6);
            }
        }
    }

    #[test]
    fn vector_field_fft_columns_emits_interleaved_channels() {
        let mut field = ColorVectorField::new(1, 4);
        field.set(0, 1.0, [0.0, 0.0, 0.0]);
        let spectrum = field.fft_cols_interleaved(false).unwrap();
        assert_eq!(spectrum.len(), 32);
        for chunk in spectrum.chunks_exact(8) {
            assert!((chunk[0] - 1.0).abs() < 1e-6);
            assert!(chunk[1].abs() < 1e-6);
            for value in &chunk[2..] {
                assert!(value.abs() < 1e-6);
            }
        }
    }

    #[test]
    fn vector_field_fft_2d_emits_interleaved_channels() {
        let mut field = ColorVectorField::new(2, 2);
        field.set(0, 1.0, [0.0, 0.0, 0.0]);
        let spectrum = field.fft_2d_interleaved(false).unwrap();
        assert_eq!(spectrum.len(), 2 * 2 * 8);
        for chunk in spectrum.chunks_exact(8) {
            assert!((chunk[0] - 1.0).abs() < 1e-6);
            assert!(chunk[1].abs() < 1e-6);
            for value in &chunk[2..] {
                assert!(value.abs() < 1e-6);
            }
        }
    }

    #[test]
    fn vector_field_fft_2d_handles_rectangular_canvas() {
        let mut field = ColorVectorField::new(3, 2);
        field.set(0, 1.0, [0.0, 0.0, 0.0]);
        let spectrum = field.fft_2d_interleaved(false).unwrap();
        assert_eq!(spectrum.len(), 3 * 2 * 8);
        assert!(spectrum.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn vector_field_fft_columns_handles_non_power_of_two_height() {
        let mut field = ColorVectorField::new(2, 3);
        field.set(0, 1.0, [0.0, 0.0, 0.0]);
        let spectrum = field.fft_cols_interleaved(false).unwrap();
        assert_eq!(spectrum.len(), 48);
        for value in spectrum {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn vector_field_fft_handles_non_power_of_two_width() {
        let mut field = ColorVectorField::new(3, 1);
        field.set(0, 1.0, [0.0, 0.0, 0.0]);
        let spectrum = field.fft_rows_interleaved(false).unwrap();
        assert_eq!(spectrum.len(), 24);
        for value in spectrum {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn vector_field_fft_rows_rejects_empty_dimensions() {
        let field = ColorVectorField::new(0, 1);
        assert!(matches!(
            field.fft_rows_interleaved(false),
            Err(TensorError::EmptyInput("canvas_fft"))
        ));
    }

    #[test]
    fn vector_field_fft_columns_rejects_empty_dimensions() {
        let field = ColorVectorField::new(1, 0);
        assert!(matches!(
            field.fft_cols_interleaved(false),
            Err(TensorError::EmptyInput("canvas_fft"))
        ));
    }

    #[test]
    fn vector_field_fft_2d_rejects_empty_dimensions() {
        let field = ColorVectorField::new(0, 0);
        assert!(matches!(
            field.fft_2d_interleaved(false),
            Err(TensorError::EmptyInput("canvas_fft"))
        ));
    }

    #[test]
    fn vector_field_fft_rows_tensor_rejects_empty_dimensions() {
        let field = ColorVectorField::new(0, 1);
        assert!(matches!(
            field.fft_rows_tensor(false),
            Err(TensorError::EmptyInput("canvas_fft"))
        ));
    }

    #[test]
    fn vector_field_fft_rows_magnitude_tensor_matches_shape() {
        let mut field = ColorVectorField::new(2, 1);
        field.set(0, 1.0, [0.0, 0.0, 0.0]);
        let tensor = field.fft_rows_magnitude_tensor(false).unwrap();
        assert_eq!(tensor.shape(), (1, 8));
        assert!(tensor
            .data()
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
    }

    #[test]
    fn vector_field_fft_rows_power_tensor_matches_squared_magnitude() {
        let mut field = ColorVectorField::new(3, 2);
        for idx in 0..6 {
            let energy = 0.2 * idx as f32;
            let chroma = [
                0.15 * (idx as f32 + 1.0),
                -0.1 * idx as f32,
                0.05 * (idx as f32 - 2.0),
            ];
            field.set(idx, energy, chroma);
        }

        let power = field.fft_rows_power_tensor(false).unwrap();
        let magnitude = field.fft_rows_magnitude_tensor(false).unwrap();

        assert_eq!(power.shape(), magnitude.shape());
        for (p, m) in power.data().iter().zip(magnitude.data()) {
            assert!((p - m * m).abs() < 1e-6);
        }
    }

    #[test]
    fn vector_field_fft_rows_power_db_tensor_matches_logarithmic_projection() {
        let mut field = ColorVectorField::new(3, 2);
        for idx in 0..6 {
            let energy = 0.25 * idx as f32;
            let chroma = [
                (-0.05 * idx as f32).sin(),
                0.1 * (idx as f32 + 0.5),
                0.075 * (idx as f32 - 1.0),
            ];
            field.set(idx, energy, chroma);
        }

        let power = field.fft_rows_power_tensor(false).unwrap();
        let power_db = field.fft_rows_power_db_tensor(false).unwrap();

        assert_eq!(power.shape(), power_db.shape());
        for (&linear, &db) in power.data().iter().zip(power_db.data()) {
            let expected = (10.0 * linear.max(ColorVectorField::POWER_DB_EPSILON).log10())
                .max(ColorVectorField::POWER_DB_FLOOR);
            assert!((expected - db).abs() < 1e-6);
        }
    }

    #[test]
    fn vector_field_fft_rows_phase_tensor_matches_shape() {
        let mut field = ColorVectorField::new(2, 1);
        field.set(0, 1.0, [0.0, 0.0, 0.0]);
        let tensor = field.fft_rows_phase_tensor(false).unwrap();
        assert_eq!(tensor.shape(), (1, 8));
        assert!(tensor.data().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn vector_field_fft_rows_polar_tensors_align_with_components() {
        let mut field = ColorVectorField::new(3, 2);
        for idx in 0..6 {
            let energy = 0.25 * idx as f32;
            let chroma = [
                0.1 * (idx as f32 + 1.0),
                -0.05 * idx as f32,
                0.075 * (idx as f32 - 1.0),
            ];
            field.set(idx, energy, chroma);
        }

        let (magnitude, phase) = field.fft_rows_polar_tensors(false).unwrap();
        let magnitude_only = field.fft_rows_magnitude_tensor(false).unwrap();
        let phase_only = field.fft_rows_phase_tensor(false).unwrap();

        assert_eq!(magnitude.shape(), magnitude_only.shape());
        assert_eq!(phase.shape(), phase_only.shape());

        for (lhs, rhs) in magnitude.data().iter().zip(magnitude_only.data()) {
            assert!((lhs - rhs).abs() < 1e-6);
        }
        for (lhs, rhs) in phase.data().iter().zip(phase_only.data()) {
            assert!((lhs - rhs).abs() < 1e-6);
        }
    }

    #[test]
    fn polar_tensor_helper_rejects_mismatched_spectrum_length() {
        let spectrum = vec![0.0; 7];
        let err = ColorVectorField::polar_tensors_from_interleaved(1, 1, &spectrum)
            .expect_err("expected data length mismatch");
        match err {
            TensorError::DataLength { expected, got } => {
                assert_eq!(expected, 8);
                assert_eq!(got, 7);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn vector_field_fft_columns_tensor_rejects_empty_dimensions() {
        let field = ColorVectorField::new(1, 0);
        assert!(matches!(
            field.fft_cols_tensor(false),
            Err(TensorError::EmptyInput("canvas_fft"))
        ));
    }

    #[test]
    fn vector_field_fft_columns_magnitude_tensor_matches_shape() {
        let mut field = ColorVectorField::new(1, 2);
        field.set(0, 1.0, [0.0, 0.0, 0.0]);
        let tensor = field.fft_cols_magnitude_tensor(false).unwrap();
        assert_eq!(tensor.shape(), (1, 8));
        assert!(tensor
            .data()
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
    }

    #[test]
    fn vector_field_fft_columns_power_tensor_matches_squared_magnitude() {
        let mut field = ColorVectorField::new(2, 3);
        for idx in 0..6 {
            let energy = 0.3 * idx as f32;
            let chroma = [
                0.07 * (idx as f32 + 0.5),
                0.02 * (idx as f32 - 1.0),
                -0.03 * idx as f32,
            ];
            field.set(idx, energy, chroma);
        }

        let power = field.fft_cols_power_tensor(false).unwrap();
        let magnitude = field.fft_cols_magnitude_tensor(false).unwrap();

        assert_eq!(power.shape(), magnitude.shape());
        for (p, m) in power.data().iter().zip(magnitude.data()) {
            assert!((p - m * m).abs() < 1e-6);
        }
    }

    #[test]
    fn vector_field_fft_columns_power_db_tensor_matches_logarithmic_projection() {
        let mut field = ColorVectorField::new(2, 3);
        for idx in 0..6 {
            let energy = 0.18 * idx as f32;
            let chroma = [
                0.12 * (idx as f32 + 0.25),
                -0.07 * (idx as f32 - 0.5),
                0.05 * (idx as f32 - 1.75),
            ];
            field.set(idx, energy, chroma);
        }

        let power = field.fft_cols_power_tensor(false).unwrap();
        let power_db = field.fft_cols_power_db_tensor(false).unwrap();

        assert_eq!(power.shape(), power_db.shape());
        for (&linear, &db) in power.data().iter().zip(power_db.data()) {
            let expected = (10.0 * linear.max(ColorVectorField::POWER_DB_EPSILON).log10())
                .max(ColorVectorField::POWER_DB_FLOOR);
            assert!((expected - db).abs() < 1e-6);
        }
    }

    #[test]
    fn vector_field_fft_columns_phase_tensor_matches_shape() {
        let mut field = ColorVectorField::new(1, 2);
        field.set(0, 1.0, [0.0, 0.0, 0.0]);
        let tensor = field.fft_cols_phase_tensor(false).unwrap();
        assert_eq!(tensor.shape(), (1, 8));
        assert!(tensor.data().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn vector_field_fft_columns_polar_tensors_align_with_components() {
        let mut field = ColorVectorField::new(2, 3);
        for idx in 0..6 {
            let energy = 0.3 * idx as f32;
            let chroma = [
                0.2 * (idx as f32 + 0.5),
                0.1 * (idx as f32 - 0.5),
                -0.15 * idx as f32,
            ];
            field.set(idx, energy, chroma);
        }

        let (magnitude, phase) = field.fft_cols_polar_tensors(false).unwrap();
        let magnitude_only = field.fft_cols_magnitude_tensor(false).unwrap();
        let phase_only = field.fft_cols_phase_tensor(false).unwrap();

        assert_eq!(magnitude.shape(), magnitude_only.shape());
        assert_eq!(phase.shape(), phase_only.shape());

        for (lhs, rhs) in magnitude.data().iter().zip(magnitude_only.data()) {
            assert!((lhs - rhs).abs() < 1e-6);
        }
        for (lhs, rhs) in phase.data().iter().zip(phase_only.data()) {
            assert!((lhs - rhs).abs() < 1e-6);
        }
    }

    #[test]
    fn vector_field_fft_2d_tensor_rejects_empty_dimensions() {
        let field = ColorVectorField::new(0, 0);
        assert!(matches!(
            field.fft_2d_tensor(false),
            Err(TensorError::EmptyInput("canvas_fft"))
        ));
    }

    #[test]
    fn vector_field_fft_2d_magnitude_tensor_matches_shape() {
        let mut field = ColorVectorField::new(2, 2);
        field.set(0, 1.0, [0.0, 0.0, 0.0]);
        let tensor = field.fft_2d_magnitude_tensor(false).unwrap();
        assert_eq!(tensor.shape(), (2, 8));
        assert!(tensor
            .data()
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
    }

    #[test]
    fn vector_field_fft_2d_power_tensor_matches_squared_magnitude() {
        let mut field = ColorVectorField::new(3, 2);
        for idx in 0..6 {
            let energy = -0.1 * idx as f32;
            let chroma = [
                0.05 * (idx as f32 + 1.5),
                0.025 * (idx as f32 - 0.5),
                -0.04 * idx as f32,
            ];
            field.set(idx, energy, chroma);
        }

        let power = field.fft_2d_power_tensor(false).unwrap();
        let magnitude = field.fft_2d_magnitude_tensor(false).unwrap();

        assert_eq!(power.shape(), magnitude.shape());
        for (p, m) in power.data().iter().zip(magnitude.data()) {
            assert!((p - m * m).abs() < 1e-6);
        }
    }

    #[test]
    fn vector_field_fft_2d_power_db_tensor_matches_logarithmic_projection() {
        let mut field = ColorVectorField::new(3, 2);
        for idx in 0..6 {
            let energy = 0.35 * idx as f32;
            let chroma = [
                0.21 * (idx as f32 + 0.25),
                -0.14 * (idx as f32 - 0.75),
                0.09 * (idx as f32 - 1.25),
            ];
            field.set(idx, energy, chroma);
        }

        let power = field.fft_2d_power_tensor(false).unwrap();
        let power_db = field.fft_2d_power_db_tensor(false).unwrap();

        assert_eq!(power.shape(), power_db.shape());
        for (&linear, &db) in power.data().iter().zip(power_db.data()) {
            let expected = (10.0 * linear.max(ColorVectorField::POWER_DB_EPSILON).log10())
                .max(ColorVectorField::POWER_DB_FLOOR);
            assert!((expected - db).abs() < 1e-6);
        }
    }

    #[test]
    fn vector_field_fft_2d_phase_tensor_matches_shape() {
        let mut field = ColorVectorField::new(2, 2);
        field.set(0, 1.0, [0.0, 0.0, 0.0]);
        let tensor = field.fft_2d_phase_tensor(false).unwrap();
        assert_eq!(tensor.shape(), (2, 8));
        assert!(tensor.data().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn vector_field_fft_2d_polar_tensors_align_with_components() {
        let mut field = ColorVectorField::new(3, 2);
        for idx in 0..6 {
            let energy = 0.15 * idx as f32;
            let chroma = [
                -0.05 * (idx as f32 + 1.0),
                0.08 * (idx as f32 + 0.25),
                0.06 * (idx as f32 - 0.75),
            ];
            field.set(idx, energy, chroma);
        }

        let (magnitude, phase) = field.fft_2d_polar_tensors(false).unwrap();
        let magnitude_only = field.fft_2d_magnitude_tensor(false).unwrap();
        let phase_only = field.fft_2d_phase_tensor(false).unwrap();

        assert_eq!(magnitude.shape(), magnitude_only.shape());
        assert_eq!(phase.shape(), phase_only.shape());

        for (lhs, rhs) in magnitude.data().iter().zip(magnitude_only.data()) {
            assert!((lhs - rhs).abs() < 1e-6);
        }
        for (lhs, rhs) in phase.data().iter().zip(phase_only.data()) {
            assert!((lhs - rhs).abs() < 1e-6);
        }
    }

    #[test]
    fn projector_refresh_tensor_exposes_workspace() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(2, 2).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 2, 2).unwrap();
        let tensor = projector.refresh_tensor().unwrap();
        assert_eq!(tensor.shape(), (2, 2));
        assert!(tensor.data().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn projector_refresh_vector_fft_columns_tensor_matches_shape() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(3, 5).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 5, 3).unwrap();
        let tensor = projector.refresh_vector_fft_columns_tensor(false).unwrap();
        assert_eq!(tensor.shape(), (5, 3 * 8));
    }

    #[test]
    fn projector_refresh_vector_fft_magnitude_tensor_matches_shape() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(2, 4).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 4, 2).unwrap();
        let tensor = projector
            .refresh_vector_fft_magnitude_tensor(false)
            .unwrap();
        assert_eq!(tensor.shape(), (2, 4 * 4));
        assert!(tensor
            .data()
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
    }

    #[test]
    fn projector_refresh_vector_fft_power_tensor_matches_squared_magnitude() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(2, 4).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 4, 2).unwrap();
        let power = projector.refresh_vector_fft_power_tensor(false).unwrap();
        let magnitude = projector
            .refresh_vector_fft_magnitude_tensor(false)
            .unwrap();

        assert_eq!(power.shape(), magnitude.shape());
        for (p, m) in power.data().iter().zip(magnitude.data()) {
            assert!((p - m * m).abs() < 1e-6);
        }
    }

    #[test]
    fn projector_refresh_vector_fft_power_db_tensor_matches_logarithmic_projection() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(2, 4).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 4, 2).unwrap();
        let power = projector.refresh_vector_fft_power_tensor(false).unwrap();
        let power_db = projector.refresh_vector_fft_power_db_tensor(false).unwrap();

        assert_eq!(power.shape(), power_db.shape());
        for (&linear, &db) in power.data().iter().zip(power_db.data()) {
            let expected = (10.0 * linear.max(ColorVectorField::POWER_DB_EPSILON).log10())
                .max(ColorVectorField::POWER_DB_FLOOR);
            assert!((expected - db).abs() < 1e-6);
        }
    }

    #[test]
    fn projector_refresh_vector_fft_phase_tensor_matches_shape() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(2, 4).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 4, 2).unwrap();
        let tensor = projector.refresh_vector_fft_phase_tensor(false).unwrap();
        assert_eq!(tensor.shape(), (2, 4 * 4));
        assert!(tensor.data().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn projector_refresh_vector_fft_polar_tensors_align_with_components() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(2, 4).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 4, 2).unwrap();
        let (magnitude, phase) = projector.refresh_vector_fft_polar_tensors(false).unwrap();
        let magnitude_only = projector
            .refresh_vector_fft_magnitude_tensor(false)
            .unwrap();
        let phase_only = projector.refresh_vector_fft_phase_tensor(false).unwrap();

        assert_eq!(magnitude.shape(), (2, 4 * 4));
        assert_eq!(phase.shape(), (2, 4 * 4));

        for (lhs, rhs) in magnitude.data().iter().zip(magnitude_only.data()) {
            assert!((lhs - rhs).abs() < 1e-6);
        }
        for (lhs, rhs) in phase.data().iter().zip(phase_only.data()) {
            assert!((lhs - rhs).abs() < 1e-6);
        }
    }

    #[test]
    fn projector_refresh_vector_fft_columns_magnitude_tensor_matches_shape() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(3, 5).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 5, 3).unwrap();
        let tensor = projector
            .refresh_vector_fft_columns_magnitude_tensor(false)
            .unwrap();
        assert_eq!(tensor.shape(), (5, 3 * 4));
        assert!(tensor
            .data()
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
    }

    #[test]
    fn projector_refresh_vector_fft_columns_power_tensor_matches_squared_magnitude() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(3, 5).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 5, 3).unwrap();
        let power = projector
            .refresh_vector_fft_columns_power_tensor(false)
            .unwrap();
        let magnitude = projector
            .refresh_vector_fft_columns_magnitude_tensor(false)
            .unwrap();

        assert_eq!(power.shape(), magnitude.shape());
        for (p, m) in power.data().iter().zip(magnitude.data()) {
            assert!((p - m * m).abs() < 1e-6);
        }
    }

    #[test]
    fn projector_refresh_vector_fft_columns_power_db_tensor_matches_logarithmic_projection() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(3, 5).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 5, 3).unwrap();
        let power = projector
            .refresh_vector_fft_columns_power_tensor(false)
            .unwrap();
        let power_db = projector
            .refresh_vector_fft_columns_power_db_tensor(false)
            .unwrap();

        assert_eq!(power.shape(), power_db.shape());
        for (&linear, &db) in power.data().iter().zip(power_db.data()) {
            let expected = (10.0 * linear.max(ColorVectorField::POWER_DB_EPSILON).log10())
                .max(ColorVectorField::POWER_DB_FLOOR);
            assert!((expected - db).abs() < 1e-6);
        }
    }

    #[test]
    fn projector_refresh_vector_fft_columns_phase_tensor_matches_shape() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(3, 5).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 5, 3).unwrap();
        let tensor = projector
            .refresh_vector_fft_columns_phase_tensor(false)
            .unwrap();
        assert_eq!(tensor.shape(), (5, 3 * 4));
        assert!(tensor.data().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn projector_refresh_vector_fft_columns_polar_tensors_align_with_components() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(3, 5).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 5, 3).unwrap();
        let (magnitude, phase) = projector
            .refresh_vector_fft_columns_polar_tensors(false)
            .unwrap();
        let magnitude_only = projector
            .refresh_vector_fft_columns_magnitude_tensor(false)
            .unwrap();
        let phase_only = projector
            .refresh_vector_fft_columns_phase_tensor(false)
            .unwrap();

        assert_eq!(magnitude.shape(), (5, 3 * 4));
        assert_eq!(phase.shape(), (5, 3 * 4));

        for (lhs, rhs) in magnitude.data().iter().zip(magnitude_only.data()) {
            assert!((lhs - rhs).abs() < 1e-6);
        }
        for (lhs, rhs) in phase.data().iter().zip(phase_only.data()) {
            assert!((lhs - rhs).abs() < 1e-6);
        }
    }

    #[test]
    fn projector_refresh_vector_fft_2d_magnitude_tensor_matches_shape() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(3, 5).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 5, 3).unwrap();
        let tensor = projector
            .refresh_vector_fft_2d_magnitude_tensor(false)
            .unwrap();
        assert_eq!(tensor.shape(), (3, 5 * 4));
        assert!(tensor
            .data()
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
    }

    #[test]
    fn projector_refresh_vector_fft_2d_power_tensor_matches_squared_magnitude() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(3, 5).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 5, 3).unwrap();
        let power = projector.refresh_vector_fft_2d_power_tensor(false).unwrap();
        let magnitude = projector
            .refresh_vector_fft_2d_magnitude_tensor(false)
            .unwrap();

        assert_eq!(power.shape(), magnitude.shape());
        for (p, m) in power.data().iter().zip(magnitude.data()) {
            assert!((p - m * m).abs() < 1e-6);
        }
    }

    #[test]
    fn projector_refresh_vector_fft_2d_power_db_tensor_matches_logarithmic_projection() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(3, 5).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 5, 3).unwrap();
        let power = projector.refresh_vector_fft_2d_power_tensor(false).unwrap();
        let power_db = projector
            .refresh_vector_fft_2d_power_db_tensor(false)
            .unwrap();

        assert_eq!(power.shape(), power_db.shape());
        for (&linear, &db) in power.data().iter().zip(power_db.data()) {
            let expected = (10.0 * linear.max(ColorVectorField::POWER_DB_EPSILON).log10())
                .max(ColorVectorField::POWER_DB_FLOOR);
            assert!((expected - db).abs() < 1e-6);
        }
    }

    #[test]
    fn projector_refresh_vector_fft_2d_phase_tensor_matches_shape() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(3, 5).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 5, 3).unwrap();
        let tensor = projector.refresh_vector_fft_2d_phase_tensor(false).unwrap();
        assert_eq!(tensor.shape(), (3, 5 * 4));
        assert!(tensor.data().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn projector_refresh_vector_fft_2d_polar_tensors_align_with_components() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(3, 5).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 5, 3).unwrap();
        let (magnitude, phase) = projector
            .refresh_vector_fft_2d_polar_tensors(false)
            .unwrap();
        let magnitude_only = projector
            .refresh_vector_fft_2d_magnitude_tensor(false)
            .unwrap();
        let phase_only = projector.refresh_vector_fft_2d_phase_tensor(false).unwrap();

        assert_eq!(magnitude.shape(), (3, 5 * 4));
        assert_eq!(phase.shape(), (3, 5 * 4));

        for (lhs, rhs) in magnitude.data().iter().zip(magnitude_only.data()) {
            assert!((lhs - rhs).abs() < 1e-6);
        }
        for (lhs, rhs) in phase.data().iter().zip(phase_only.data()) {
            assert!((lhs - rhs).abs() < 1e-6);
        }
    }

    #[test]
    fn projector_refresh_vector_fft_2d_tensor_matches_shape() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(3, 5).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 5, 3).unwrap();
        let tensor = projector.refresh_vector_fft_2d_tensor(false).unwrap();
        assert_eq!(tensor.shape(), (3, 5 * 8));
    }

    #[test]
    fn projector_accumulates_into_gradients() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(2, 2).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 2, 2).unwrap();
        let mut hypergrad = AmegaHypergrad::new(-1.0, 0.05, 2, 2).unwrap();
        let mut realgrad = AmegaRealgrad::new(0.05, 2, 2).unwrap();
        projector.accumulate_hypergrad(&mut hypergrad).unwrap();
        projector.accumulate_realgrad(&mut realgrad).unwrap();
        assert!(hypergrad.gradient().iter().all(|value| value.is_finite()));
        assert_eq!(realgrad.gradient().len(), 4);
    }

    #[test]
    fn projector_surfaces_gradient_summary() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(
                FractalPatch::new(tensor_with_shape(2, 2, &[1.0, -2.0, 0.5, 0.0]), 1.0, 1.0, 0)
                    .unwrap(),
            )
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 2, 2).unwrap();
        let (hyper, real) = projector.gradient_summary(-1.0).unwrap();
        assert_eq!(hyper.count(), 4);
        assert_eq!(real.count(), 4);
        assert!(hyper.l2() > 0.0);
        assert!(real.l1() > 0.0);
        let mean_identity = real.mean_abs() * real.count() as f32;
        assert!((mean_identity - real.l1()).abs() < 1e-6);
        let rms_identity = real.rms() * (real.count() as f32).sqrt();
        assert!((rms_identity - real.l2()).abs() < 1e-6);
    }

    #[test]
    fn projector_surfaces_desire_interpretation() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(
                FractalPatch::new(tensor_with_shape(2, 2, &[0.2, -0.1, 0.4, 0.3]), 1.0, 1.0, 0)
                    .unwrap(),
            )
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 2, 2).unwrap();
        let interpretation = projector.gradient_interpretation(-1.0).unwrap();
        assert!(interpretation.penalty_gain() >= 1.0);
        assert!(interpretation.bias_mix() > 0.0);
    }

    #[test]
    fn projector_surfaces_desire_control() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(
                FractalPatch::new(
                    tensor_with_shape(2, 2, &[0.25, 0.1, -0.35, 0.6]),
                    1.0,
                    1.0,
                    0,
                )
                .unwrap(),
            )
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 2, 2).unwrap();
        let control = projector.gradient_control(-1.0).unwrap();
        assert!(control.penalty_gain() >= 1.0);
        assert!(control.hyper_rate_scale().is_finite());
        let uniform = projector.hypergrad_operator_uniform_from_control(&control);
        assert_eq!(uniform[0], 2.0);
        assert_eq!(uniform[1], 2.0);
        assert!((uniform[2] - control.operator_mix()).abs() < 1e-6);
        assert!((uniform[3] - control.operator_gain()).abs() < 1e-6);
        let packed = projector.desire_control_uniform(&control);
        assert_eq!(packed.len(), 16);
        assert!((f32::from_bits(packed[0]) - control.target_entropy()).abs() < 1e-6);
        assert!((f32::from_bits(packed[1]) - control.learning_rate_eta()).abs() < 1e-6);
        assert!((f32::from_bits(packed[5]) - control.clip_norm()).abs() < 1e-6);
        assert!((f32::from_bits(packed[9]) - control.temperature_kappa()).abs() < 1e-6);
    }

    #[test]
    fn hypergrad_operator_surfaces_wgsl_and_metadata() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(
                FractalPatch::new(tensor_with_shape(2, 2, &[0.1, 0.2, 0.3, 0.4]), 1.0, 1.0, 0)
                    .unwrap(),
            )
            .unwrap();
        let projector = CanvasProjector::new(scheduler, 2, 2).unwrap();
        let shader = projector.hypergrad_operator_wgsl(false);
        assert!(shader.contains("CanvasHypergradParams"));
        assert!(shader.contains("width 2"));
        let uniform = projector.hypergrad_operator_uniform(0.5, 2.0);
        assert_eq!(uniform[0], 2.0);
        assert_eq!(uniform[1], 2.0);
        assert!((uniform[2] - 0.5).abs() < 1e-6);
        assert!((uniform[3] - 2.0).abs() < 1e-6);
        let dispatch = projector.hypergrad_operator_dispatch(false);
        assert_eq!(dispatch, [1, 2, 1]);
    }

    #[test]
    fn projector_refresh_vector_fft_matches_canvas_shape() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(2, 2).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 2, 2).unwrap();
        let spectrum = projector.refresh_vector_fft(false).unwrap();
        assert_eq!(spectrum.len(), 2 * 2 * 8);
    }

    #[test]
    fn projector_refresh_vector_fft_2d_matches_canvas_shape() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(FractalPatch::new(Tensor::zeros(2, 2).unwrap(), 1.0, 1.0, 0).unwrap())
            .unwrap();
        let mut projector = CanvasProjector::new(scheduler, 2, 2).unwrap();
        let spectrum = projector.refresh_vector_fft_2d(false).unwrap();
        assert_eq!(spectrum.len(), 2 * 2 * 8);
    }

    #[test]
    fn canvas_rejects_invalid_dimensions() {
        assert!(matches!(
            CanvasSurface::new(0, 1),
            Err(TensorError::InvalidDimensions { .. })
        ));
    }

    #[test]
    fn projector_refreshes_without_extra_allocations() {
        let scheduler = UringFractalScheduler::new(8).unwrap();
        scheduler
            .push(FractalPatch::new(tensor(&[1.0, 0.0, 0.5, 1.0]), 1.5, 1.0, 0).unwrap())
            .unwrap();
        scheduler
            .push(FractalPatch::new(tensor(&[0.25, 1.0, 0.75, 0.5]), 0.75, 0.5, 1).unwrap())
            .unwrap();

        let mut projector = CanvasProjector::new(scheduler.clone(), 4, 1).unwrap();
        let first = projector.refresh().unwrap().as_ptr();
        let second = projector.refresh().unwrap().as_ptr();
        assert_eq!(first, second);
        assert_eq!(projector.surface().as_rgba().len(), 4 * 4);
    }

    #[test]
    fn emit_zspace_patch_respects_canvas_shape() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        scheduler
            .push(
                FractalPatch::new(
                    tensor_with_shape(2, 2, &[1.0, 0.5, 0.25, 0.75]),
                    1.0,
                    1.0,
                    0,
                )
                .unwrap(),
            )
            .unwrap();

        let mut projector = CanvasProjector::new(scheduler, 2, 2).unwrap();
        let patch = projector.emit_zspace_patch(1.0, 1.0, 2).unwrap();
        assert_eq!(patch.relation().shape(), (2, 2));
        let data = patch.relation().data();
        assert_eq!(data.len(), 4);
    }
}
