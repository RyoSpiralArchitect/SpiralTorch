//! Minimal WASM canvas interop helpers for the pure fractal stack.
//!
//! The goal of this module is to keep the browser pathway fully Rust-native:
//! we stream `UringFractalScheduler` relations into a persistent tensor buffer,
//! then map that buffer onto an RGBA pixel grid that JavaScript can blit into a
//! `<canvas>` element without allocating intermediate copies. Everything here
//! stays in safe Rust so it can compile to WASM as-is, while HTML/JS glue code
//! can simply forward the produced byte slice into `ImageData`.

use super::{fractal::UringFractalScheduler, PureResult, Tensor, TensorError};

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
        let expected = (self.height, self.width);
        if tensor.shape() != expected {
            return Err(TensorError::ShapeMismatch {
                left: tensor.shape(),
                right: expected,
            });
        }

        let data = tensor.data();
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for value in data.iter().copied() {
            min = min.min(value);
            max = max.max(value);
        }
        let range = if max > min { max - min } else { 1.0 };
        for (idx, value) in data.iter().enumerate() {
            let normalized = if range == 0.0 {
                0.5
            } else {
                ((*value - min) / range).clamp(0.0, 1.0)
            };
            let intensity = (normalized * 255.0) as u8;
            let accent = ((1.0 - normalized) * 255.0) as u8;
            let offset = idx * 4;
            self.pixels[offset] = accent; // R
            self.pixels[offset + 1] = intensity / 2; // G
            self.pixels[offset + 2] = intensity; // B
            self.pixels[offset + 3] = 255; // A
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
}

impl CanvasProjector {
    /// Construct a projector with a shared scheduler. The workspace tensor is
    /// reused for every refresh call to avoid hitting the WASM allocator.
    pub fn new(scheduler: UringFractalScheduler, width: usize, height: usize) -> PureResult<Self> {
        let surface = CanvasSurface::new(width, height)?;
        let workspace = Tensor::zeros(height, width)?;
        Ok(Self {
            scheduler,
            surface,
            workspace,
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

    /// Refresh the canvas by folding the queued patches straight into the
    /// workspace tensor and painting the result. Callers can ship the returned
    /// slice to JavaScript without cloning.
    pub fn refresh(&mut self) -> PureResult<&[u8]> {
        self.scheduler.fold_coherence_into(&mut self.workspace)?;
        self.surface.paint_tensor(&self.workspace)?;
        Ok(self.surface.as_rgba())
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
}
