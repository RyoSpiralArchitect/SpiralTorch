use js_sys::{Array, Float32Array, Uint32Array, Uint8Array};
use st_tensor::fractal::{FractalPatch, UringFractalScheduler};
use st_tensor::wasm_canvas::{CanvasPalette, CanvasProjector};
use st_tensor::{Tensor, TensorError};
use wasm_bindgen::prelude::*;
use wasm_bindgen::{Clamped, JsCast};
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, ImageData};

use crate::utils::js_error;

#[wasm_bindgen]
pub struct FractalCanvas {
    projector: CanvasProjector,
    width: usize,
    height: usize,
}

#[wasm_bindgen]
impl FractalCanvas {
    /// Construct a projector-backed canvas with the requested queue capacity.
    #[wasm_bindgen(constructor)]
    pub fn new(capacity: usize, width: usize, height: usize) -> Result<FractalCanvas, JsValue> {
        let scheduler = UringFractalScheduler::new(capacity).map_err(js_error)?;
        let projector = CanvasProjector::new(scheduler, width, height).map_err(js_error)?;
        Ok(Self {
            projector,
            width,
            height,
        })
    }

    /// Canvas width in pixels.
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width as u32
    }

    /// Canvas height in pixels.
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height as u32
    }

    /// Push a new fractal relation patch into the scheduler.
    ///
    /// The input buffer must contain `width * height` values laid out in row-major
    /// order.
    pub fn push_patch(
        &self,
        relation: &Float32Array,
        coherence: f32,
        tension: f32,
        depth: u32,
    ) -> Result<(), JsValue> {
        let len = relation.length() as usize;
        let expected = self.width * self.height;
        if len != expected {
            return Err(js_error(TensorError::ShapeMismatch {
                left: (1, len),
                right: (self.height, self.width),
            }));
        }
        let mut values = vec![0.0f32; len];
        relation.copy_to(&mut values);
        let tensor = Tensor::from_vec(self.height, self.width, values).map_err(js_error)?;
        let patch = FractalPatch::new(tensor, coherence, tension, depth).map_err(js_error)?;
        self.projector.scheduler().push(patch).map_err(js_error)
    }

    /// Render the current scheduler state onto the provided HTML canvas.
    pub fn render_to_canvas(&mut self, canvas: HtmlCanvasElement) -> Result<(), JsValue> {
        let width = self.width as u32;
        let height = self.height as u32;
        if canvas.width() != width {
            canvas.set_width(width);
        }
        if canvas.height() != height {
            canvas.set_height(height);
        }
        let context = canvas
            .get_context("2d")?
            .ok_or_else(|| js_error("missing 2d context"))?
            .dyn_into::<CanvasRenderingContext2d>()?;
        let pixels = self.projector.refresh().map_err(js_error)?;
        let image = ImageData::new_with_u8_clamped_array_and_sh(Clamped(pixels), width, height)?;
        context.put_image_data(&image, 0.0, 0.0)?;
        Ok(())
    }

    /// Refresh the projector and return a view of the RGBA buffer as a typed array.
    pub fn pixels(&mut self) -> Result<Uint8Array, JsValue> {
        let pixels = self.projector.refresh().map_err(js_error)?;
        Ok(Uint8Array::from(pixels))
    }

    /// Refresh the projector and expose the colour vector field as a `Float32Array`.
    pub fn vector_field(&mut self) -> Result<Float32Array, JsValue> {
        let field = self.projector.refresh_vector_field().map_err(js_error)?;
        let mut data = Vec::with_capacity(field.vectors().len() * 4);
        for vector in field.iter() {
            data.extend_from_slice(&vector);
        }
        Ok(Float32Array::from(data.as_slice()))
    }

    /// Refresh the projector and return the interleaved FFT spectrum for each
    /// canvas row. Each frequency sample contributes eight floats (real/imag
    /// pairs for energy + RGB chroma).
    #[wasm_bindgen(js_name = vectorFieldFft)]
    pub fn vector_field_fft(&mut self, inverse: bool) -> Result<Float32Array, JsValue> {
        let spectrum = self
            .projector
            .refresh_vector_fft(inverse)
            .map_err(js_error)?;
        Ok(Float32Array::from(spectrum.as_slice()))
    }

    /// Emit the WGSL kernel that mirrors [`vector_field_fft`] so WebGPU
    /// consumers can reproduce the spectral pass directly on the GPU.
    #[wasm_bindgen(js_name = vectorFieldFftKernel)]
    pub fn vector_field_fft_kernel(&self, subgroup: bool) -> String {
        self.projector.vector_fft_wgsl(subgroup)
    }

    /// Generate the uniform parameters expected by [`vector_field_fft_kernel`].
    ///
    /// The returned array packs the canvas `width`, `height`, the `inverse`
    /// flag (1 = inverse, 0 = forward) and a padding slot so the buffer aligns
    /// to 16 bytes as required by WGSL uniform layout rules.
    #[wasm_bindgen(js_name = vectorFieldFftUniform)]
    pub fn vector_field_fft_uniform(&self, inverse: bool) -> Uint32Array {
        let params = self.projector.vector_fft_uniform(inverse);
        Uint32Array::from(params.as_slice())
    }

    /// Reset the internal normaliser so the next frame recomputes brightness ranges.
    pub fn reset_normalizer(&mut self) {
        self.projector.normalizer_mut().reset();
    }

    /// Switch the active colour palette.
    pub fn set_palette(&mut self, name: &str) -> Result<(), JsValue> {
        let palette = parse_palette(name)?;
        self.projector.set_palette(palette);
        Ok(())
    }

    /// Current palette name.
    pub fn palette(&self) -> String {
        palette_to_name(self.projector.palette()).to_string()
    }
}

#[wasm_bindgen]
pub fn available_palettes() -> Array {
    let out = Array::new();
    out.push(&JsValue::from_str("blue-magenta"));
    out.push(&JsValue::from_str("turbo"));
    out.push(&JsValue::from_str("grayscale"));
    out
}

fn parse_palette(name: &str) -> Result<CanvasPalette, JsValue> {
    match name.to_ascii_lowercase().as_str() {
        "blue-magenta" | "blue_magenta" | "blue" => Ok(CanvasPalette::BlueMagenta),
        "turbo" => Ok(CanvasPalette::Turbo),
        "grayscale" | "grey" | "gray" => Ok(CanvasPalette::Grayscale),
        other => Err(js_error(format!("unknown palette '{other}'"))),
    }
}

fn palette_to_name(palette: CanvasPalette) -> &'static str {
    match palette {
        CanvasPalette::BlueMagenta => "blue-magenta",
        CanvasPalette::Turbo => "turbo",
        CanvasPalette::Grayscale => "grayscale",
    }
}
