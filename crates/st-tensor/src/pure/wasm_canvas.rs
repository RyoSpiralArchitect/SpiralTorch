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
    PureResult, Tensor, TensorError,
};

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

/// Vector field that captures both the normalised tensor energy and the
/// palette-projected chroma in Z-space friendly coordinates.
#[derive(Clone, Debug)]
pub struct ColorVectorField {
    width: usize,
    height: usize,
    vectors: Vec<[f32; 4]>,
}

impl ColorVectorField {
    pub fn new(width: usize, height: usize) -> Self {
        let mut field = Self {
            width,
            height,
            vectors: Vec::with_capacity(width * height),
        };
        field.ensure_shape(width, height);
        field
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

    fn render(&mut self) -> PureResult<()> {
        self.scheduler.fold_coherence_into(&mut self.workspace)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pure::fractal::FractalPatch;

    fn tensor(values: &[f32]) -> Tensor {
        Tensor::from_vec(1, values.len(), values.to_vec()).unwrap()
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
        let mut projector = CanvasProjector::new(scheduler, 2, 2).unwrap();
        projector.set_palette(CanvasPalette::Grayscale);
        let bytes = projector.refresh().unwrap();
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn refresh_with_vectors_surfaces_color_field() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        let mut projector = CanvasProjector::new(scheduler, 2, 2).unwrap();
        let (_bytes, field) = projector.refresh_with_vectors().unwrap();
        assert_eq!(field.vectors().len(), 4);
        for vector in field.iter() {
            assert!(vector[0] >= 0.0 && vector[0] <= 1.0);
        }
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
                FractalPatch::new(tensor(&[1.0, 0.5, 0.25, 0.75]), 1.0, 1.0, 0)
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
