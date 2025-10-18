use js_sys::{Array, Float32Array, Uint32Array, Uint8Array};
use st_tensor::fractal::{FractalPatch, UringFractalScheduler};
use st_tensor::wasm_canvas::{
    CanvasFftLayout as ProjectorCanvasFftLayout, CanvasPalette, CanvasProjector,
};
use st_tensor::{
    AmegaHypergrad, AmegaRealgrad, DesireControlEvents, DesireGradientControl,
    DesireGradientInterpretation, GradientSummary, Tensor, TensorError,
};
use wasm_bindgen::prelude::*;
use wasm_bindgen::{Clamped, JsCast};
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, ImageData};

use crate::utils::js_error;

#[wasm_bindgen]
pub struct CanvasFftLayout {
    field_bytes: u32,
    field_stride: u32,
    spectrum_bytes: u32,
    spectrum_stride: u32,
    uniform_bytes: u32,
}

impl From<ProjectorCanvasFftLayout> for CanvasFftLayout {
    fn from(layout: ProjectorCanvasFftLayout) -> Self {
        let clamp = |value: usize| -> u32 { value.min(u32::MAX as usize) as u32 };
        Self {
            field_bytes: clamp(layout.field_bytes()),
            field_stride: clamp(layout.field_stride()),
            spectrum_bytes: clamp(layout.spectrum_bytes()),
            spectrum_stride: clamp(layout.spectrum_stride()),
            uniform_bytes: clamp(layout.uniform_bytes()),
        }
    }
}

#[wasm_bindgen]
impl CanvasFftLayout {
    /// Total byte length required for the `FieldSample` storage buffer.
    #[wasm_bindgen(getter, js_name = fieldBytes)]
    pub fn field_bytes(&self) -> u32 {
        self.field_bytes
    }

    /// Size in bytes of each vector field sample.
    #[wasm_bindgen(getter, js_name = fieldStride)]
    pub fn field_stride(&self) -> u32 {
        self.field_stride
    }

    /// Total byte length required for the FFT spectrum storage buffer.
    #[wasm_bindgen(getter, js_name = spectrumBytes)]
    pub fn spectrum_bytes(&self) -> u32 {
        self.spectrum_bytes
    }

    /// Size in bytes of each FFT spectrum sample.
    #[wasm_bindgen(getter, js_name = spectrumStride)]
    pub fn spectrum_stride(&self) -> u32 {
        self.spectrum_stride
    }

    /// Byte length of the `CanvasFftParams` uniform buffer.
    #[wasm_bindgen(getter, js_name = uniformBytes)]
    pub fn uniform_bytes(&self) -> u32 {
        self.uniform_bytes
    }
}

#[wasm_bindgen]
pub struct CanvasGradientSummary {
    hypergrad_l1: f32,
    hypergrad_l2: f32,
    hypergrad_linf: f32,
    hypergrad_mean: f32,
    hypergrad_rms: f32,
    hypergrad_count: u32,
    realgrad_l1: f32,
    realgrad_l2: f32,
    realgrad_linf: f32,
    realgrad_mean: f32,
    realgrad_rms: f32,
    realgrad_count: u32,
}

impl CanvasGradientSummary {
    fn from_summaries(hyper: GradientSummary, real: GradientSummary) -> Self {
        Self {
            hypergrad_l1: hyper.l1(),
            hypergrad_l2: hyper.l2(),
            hypergrad_linf: hyper.linf(),
            hypergrad_mean: hyper.mean_abs(),
            hypergrad_rms: hyper.rms(),
            hypergrad_count: hyper.count().min(u32::MAX as usize) as u32,
            realgrad_l1: real.l1(),
            realgrad_l2: real.l2(),
            realgrad_linf: real.linf(),
            realgrad_mean: real.mean_abs(),
            realgrad_rms: real.rms(),
            realgrad_count: real.count().min(u32::MAX as usize) as u32,
        }
    }
}

#[wasm_bindgen]
impl CanvasGradientSummary {
    #[wasm_bindgen(getter, js_name = hypergradL1)]
    pub fn hypergrad_l1(&self) -> f32 {
        self.hypergrad_l1
    }

    #[wasm_bindgen(getter, js_name = hypergradL2)]
    pub fn hypergrad_l2(&self) -> f32 {
        self.hypergrad_l2
    }

    #[wasm_bindgen(getter, js_name = hypergradLInf)]
    pub fn hypergrad_linf(&self) -> f32 {
        self.hypergrad_linf
    }

    #[wasm_bindgen(getter, js_name = hypergradMeanAbs)]
    pub fn hypergrad_mean_abs(&self) -> f32 {
        self.hypergrad_mean
    }

    #[wasm_bindgen(getter, js_name = hypergradRms)]
    pub fn hypergrad_rms(&self) -> f32 {
        self.hypergrad_rms
    }

    #[wasm_bindgen(getter, js_name = hypergradCount)]
    pub fn hypergrad_count(&self) -> u32 {
        self.hypergrad_count
    }

    #[wasm_bindgen(getter, js_name = realgradL1)]
    pub fn realgrad_l1(&self) -> f32 {
        self.realgrad_l1
    }

    #[wasm_bindgen(getter, js_name = realgradL2)]
    pub fn realgrad_l2(&self) -> f32 {
        self.realgrad_l2
    }

    #[wasm_bindgen(getter, js_name = realgradLInf)]
    pub fn realgrad_linf(&self) -> f32 {
        self.realgrad_linf
    }

    #[wasm_bindgen(getter, js_name = realgradMeanAbs)]
    pub fn realgrad_mean_abs(&self) -> f32 {
        self.realgrad_mean
    }

    #[wasm_bindgen(getter, js_name = realgradRms)]
    pub fn realgrad_rms(&self) -> f32 {
        self.realgrad_rms
    }

    #[wasm_bindgen(getter, js_name = realgradCount)]
    pub fn realgrad_count(&self) -> u32 {
        self.realgrad_count
    }
}

#[wasm_bindgen]
pub struct CanvasDesireInterpretation {
    hyper_pressure: f32,
    real_pressure: f32,
    balance: f32,
    stability: f32,
    saturation: f32,
    penalty_gain: f32,
    bias_mix: f32,
    observation_gain: f32,
}

impl From<DesireGradientInterpretation> for CanvasDesireInterpretation {
    fn from(value: DesireGradientInterpretation) -> Self {
        Self {
            hyper_pressure: value.hyper_pressure(),
            real_pressure: value.real_pressure(),
            balance: value.balance(),
            stability: value.stability(),
            saturation: value.saturation(),
            penalty_gain: value.penalty_gain(),
            bias_mix: value.bias_mix(),
            observation_gain: value.observation_gain(),
        }
    }
}

#[wasm_bindgen]
impl CanvasDesireInterpretation {
    #[wasm_bindgen(getter, js_name = hyperPressure)]
    pub fn hyper_pressure(&self) -> f32 {
        self.hyper_pressure
    }

    #[wasm_bindgen(getter, js_name = realPressure)]
    pub fn real_pressure(&self) -> f32 {
        self.real_pressure
    }

    #[wasm_bindgen(getter)]
    pub fn balance(&self) -> f32 {
        self.balance
    }

    #[wasm_bindgen(getter)]
    pub fn stability(&self) -> f32 {
        self.stability
    }

    #[wasm_bindgen(getter)]
    pub fn saturation(&self) -> f32 {
        self.saturation
    }

    #[wasm_bindgen(getter, js_name = penaltyGain)]
    pub fn penalty_gain(&self) -> f32 {
        self.penalty_gain
    }

    #[wasm_bindgen(getter, js_name = biasMix)]
    pub fn bias_mix(&self) -> f32 {
        self.bias_mix
    }

    #[wasm_bindgen(getter, js_name = observationGain)]
    pub fn observation_gain(&self) -> f32 {
        self.observation_gain
    }
}

#[wasm_bindgen]
pub struct CanvasDesireControl {
    penalty_gain: f32,
    bias_mix: f32,
    observation_gain: f32,
    damping: f32,
    hyper_rate_scale: f32,
    real_rate_scale: f32,
    operator_mix: f32,
    operator_gain: f32,
    tuning_gain: f32,
    target_entropy: f32,
    learning_rate_eta: f32,
    learning_rate_min: f32,
    learning_rate_max: f32,
    learning_rate_slew: f32,
    clip_norm: f32,
    clip_floor: f32,
    clip_ceiling: f32,
    clip_ema: f32,
    temperature_kappa: f32,
    temperature_slew: f32,
    quality_gain: f32,
    quality_bias: f32,
    events: u32,
}

impl From<DesireGradientControl> for CanvasDesireControl {
    fn from(value: DesireGradientControl) -> Self {
        Self {
            penalty_gain: value.penalty_gain(),
            bias_mix: value.bias_mix(),
            observation_gain: value.observation_gain(),
            damping: value.damping(),
            hyper_rate_scale: value.hyper_rate_scale(),
            real_rate_scale: value.real_rate_scale(),
            operator_mix: value.operator_mix(),
            operator_gain: value.operator_gain(),
            tuning_gain: value.tuning_gain(),
            target_entropy: value.target_entropy(),
            learning_rate_eta: value.learning_rate_eta(),
            learning_rate_min: value.learning_rate_min(),
            learning_rate_max: value.learning_rate_max(),
            learning_rate_slew: value.learning_rate_slew(),
            clip_norm: value.clip_norm(),
            clip_floor: value.clip_floor(),
            clip_ceiling: value.clip_ceiling(),
            clip_ema: value.clip_ema(),
            temperature_kappa: value.temperature_kappa(),
            temperature_slew: value.temperature_slew(),
            quality_gain: value.quality_gain(),
            quality_bias: value.quality_bias(),
            events: value.events().bits(),
        }
    }
}

#[wasm_bindgen]
impl CanvasDesireControl {
    #[wasm_bindgen(getter, js_name = penaltyGain)]
    pub fn penalty_gain(&self) -> f32 {
        self.penalty_gain
    }

    #[wasm_bindgen(getter, js_name = biasMix)]
    pub fn bias_mix(&self) -> f32 {
        self.bias_mix
    }

    #[wasm_bindgen(getter, js_name = observationGain)]
    pub fn observation_gain(&self) -> f32 {
        self.observation_gain
    }

    #[wasm_bindgen(getter)]
    pub fn damping(&self) -> f32 {
        self.damping
    }

    #[wasm_bindgen(getter, js_name = hyperLearningRateScale)]
    pub fn hyper_learning_rate_scale(&self) -> f32 {
        self.hyper_rate_scale
    }

    #[wasm_bindgen(getter, js_name = realLearningRateScale)]
    pub fn real_learning_rate_scale(&self) -> f32 {
        self.real_rate_scale
    }

    #[wasm_bindgen(getter, js_name = operatorMix)]
    pub fn operator_mix(&self) -> f32 {
        self.operator_mix
    }

    #[wasm_bindgen(getter, js_name = operatorGain)]
    pub fn operator_gain(&self) -> f32 {
        self.operator_gain
    }

    #[wasm_bindgen(getter, js_name = tuningGain)]
    pub fn tuning_gain(&self) -> f32 {
        self.tuning_gain
    }

    #[wasm_bindgen(getter, js_name = targetEntropy)]
    pub fn target_entropy(&self) -> f32 {
        self.target_entropy
    }

    #[wasm_bindgen(getter, js_name = learningRateEta)]
    pub fn learning_rate_eta(&self) -> f32 {
        self.learning_rate_eta
    }

    #[wasm_bindgen(getter, js_name = learningRateMin)]
    pub fn learning_rate_min(&self) -> f32 {
        self.learning_rate_min
    }

    #[wasm_bindgen(getter, js_name = learningRateMax)]
    pub fn learning_rate_max(&self) -> f32 {
        self.learning_rate_max
    }

    #[wasm_bindgen(getter, js_name = learningRateSlew)]
    pub fn learning_rate_slew(&self) -> f32 {
        self.learning_rate_slew
    }

    #[wasm_bindgen(getter, js_name = clipNorm)]
    pub fn clip_norm(&self) -> f32 {
        self.clip_norm
    }

    #[wasm_bindgen(getter, js_name = clipFloor)]
    pub fn clip_floor(&self) -> f32 {
        self.clip_floor
    }

    #[wasm_bindgen(getter, js_name = clipCeiling)]
    pub fn clip_ceiling(&self) -> f32 {
        self.clip_ceiling
    }

    #[wasm_bindgen(getter, js_name = clipEma)]
    pub fn clip_ema(&self) -> f32 {
        self.clip_ema
    }

    #[wasm_bindgen(getter, js_name = temperatureKappa)]
    pub fn temperature_kappa(&self) -> f32 {
        self.temperature_kappa
    }

    #[wasm_bindgen(getter, js_name = temperatureSlew)]
    pub fn temperature_slew(&self) -> f32 {
        self.temperature_slew
    }

    #[wasm_bindgen(getter, js_name = qualityGain)]
    pub fn quality_gain(&self) -> f32 {
        self.quality_gain
    }

    #[wasm_bindgen(getter, js_name = qualityBias)]
    pub fn quality_bias(&self) -> f32 {
        self.quality_bias
    }

    #[wasm_bindgen(js_name = eventsMask)]
    pub fn events_mask(&self) -> u32 {
        self.events
    }

    #[wasm_bindgen(js_name = eventLabels)]
    pub fn event_labels(&self) -> Array {
        let array = Array::new();
        if self.events & DesireControlEvents::LR_INCREASE.bits() != 0 {
            array.push(&JsValue::from_str("lr_increase"));
        }
        if self.events & DesireControlEvents::LR_DECREASE.bits() != 0 {
            array.push(&JsValue::from_str("lr_decrease"));
        }
        if self.events & DesireControlEvents::CLIPPED.bits() != 0 {
            array.push(&JsValue::from_str("clip_adjust"));
        }
        if self.events & DesireControlEvents::TEMPERATURE_ADJUST.bits() != 0 {
            array.push(&JsValue::from_str("temperature_adjust"));
        }
        if self.events & DesireControlEvents::QUALITY_BOOST.bits() != 0 {
            array.push(&JsValue::from_str("quality_weight"));
        }
        array
    }
}

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

    /// Refresh the projector and expose the raw relation tensor feeding the canvas.
    pub fn relation(&mut self) -> Result<Float32Array, JsValue> {
        let tensor = self.projector.refresh_tensor().map_err(js_error)?;
        Ok(Float32Array::from(tensor.data()))
    }

    /// Refresh the projector and emit the hypergradient-aligned update for the
    /// current canvas relation.
    #[wasm_bindgen(js_name = hypergradWave)]
    pub fn hypergrad_wave(&mut self, curvature: f32) -> Result<Float32Array, JsValue> {
        let tensor = self.projector.refresh_tensor().map_err(js_error)?;
        let (rows, cols) = tensor.shape();
        let mut tape = AmegaHypergrad::new(curvature, 1.0, rows, cols).map_err(js_error)?;
        tape.accumulate_wave(tensor).map_err(js_error)?;
        Ok(Float32Array::from(tape.gradient()))
    }

    /// Refresh the projector and emit the Euclidean gradient update for the
    /// current canvas relation.
    #[wasm_bindgen(js_name = realgradWave)]
    pub fn realgrad_wave(&mut self) -> Result<Float32Array, JsValue> {
        let tensor = self.projector.refresh_tensor().map_err(js_error)?;
        let (rows, cols) = tensor.shape();
        let mut tape = AmegaRealgrad::new(1.0, rows, cols).map_err(js_error)?;
        tape.accumulate_wave(tensor).map_err(js_error)?;
        Ok(Float32Array::from(tape.gradient()))
    }

    /// Summarise the current canvas relation across both the hypergradient and
    /// Euclidean tapes. The returned object exposes the common norms so callers
    /// can monitor gradient stability without materialising the full buffers.
    #[wasm_bindgen(js_name = gradientSummary)]
    pub fn gradient_summary(&mut self, curvature: f32) -> Result<CanvasGradientSummary, JsValue> {
        let (hyper, real) = self
            .projector
            .gradient_summary(curvature)
            .map_err(js_error)?;
        Ok(CanvasGradientSummary::from_summaries(hyper, real))
    }

    /// Refresh the projector and interpret the gradient health into Desire's
    /// feedback coordinates.
    #[wasm_bindgen(js_name = desireInterpretation)]
    pub fn desire_interpretation(
        &mut self,
        curvature: f32,
    ) -> Result<CanvasDesireInterpretation, JsValue> {
        let interpretation = self
            .projector
            .gradient_interpretation(curvature)
            .map_err(js_error)?;
        Ok(CanvasDesireInterpretation::from(interpretation))
    }

    /// Refresh the projector and collapse the gradient summaries into Desire's
    /// control packet, exposing precomputed gains and learning-rate scales.
    #[wasm_bindgen(js_name = desireControl)]
    pub fn desire_control(&mut self, curvature: f32) -> Result<CanvasDesireControl, JsValue> {
        let control = self
            .projector
            .gradient_control(curvature)
            .map_err(js_error)?;
        Ok(CanvasDesireControl::from(control))
    }

    /// Refresh the projector, derive Desire's control packet, and pack it into
    /// a WGSL-friendly uniform layout.
    #[wasm_bindgen(js_name = desireControlUniform)]
    pub fn desire_control_uniform(&mut self, curvature: f32) -> Result<Float32Array, JsValue> {
        let control = self
            .projector
            .gradient_control(curvature)
            .map_err(js_error)?;
        let packed = self.projector.desire_control_uniform(&control);
        Ok(Float32Array::from(packed.as_slice()))
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

    /// Compute the workgroup dispatch dimensions that pair with
    /// [`vector_field_fft_kernel`]. The returned `[x, y, z]` triplet already
    /// accounts for the workgroup size when toggling subgroup execution.
    #[wasm_bindgen(js_name = vectorFieldFftDispatch)]
    pub fn vector_field_fft_dispatch(&self, subgroup: bool) -> Uint32Array {
        let dispatch = self.projector.vector_fft_dispatch(subgroup);
        Uint32Array::from(dispatch.as_slice())
    }

    /// Emit the WGSL kernel that accumulates the relation tensor directly into
    /// a hypergradient buffer without leaving the GPU.
    #[wasm_bindgen(js_name = hypergradOperatorKernel)]
    pub fn hypergrad_operator_kernel(&self, subgroup: bool) -> String {
        self.projector.hypergrad_operator_wgsl(subgroup)
    }

    /// Uniform parameters (width, height, blend, gain) consumed by the
    /// hypergradient WGSL operator.
    #[wasm_bindgen(js_name = hypergradOperatorUniform)]
    pub fn hypergrad_operator_uniform(&self, mix: f32, gain: f32) -> Float32Array {
        let params = self.projector.hypergrad_operator_uniform(mix, gain);
        Float32Array::from(params.as_ref())
    }

    /// Compute the hypergradient operator uniform directly from the current
    /// Desire control packet. Useful when the control data is computed once on
    /// the Rust side and then cached in JavaScript.
    #[wasm_bindgen(js_name = hypergradOperatorUniformFromControl)]
    pub fn hypergrad_operator_uniform_from_control(
        &self,
        control: &CanvasDesireControl,
    ) -> Float32Array {
        let params = self
            .projector
            .hypergrad_operator_uniform(control.operator_mix, control.operator_gain);
        Float32Array::from(params.as_ref())
    }

    /// Refresh the canvas, derive the Desire control packet, and emit the
    /// matching hypergradient operator uniform in a single step.
    #[wasm_bindgen(js_name = hypergradOperatorUniformAuto)]
    pub fn hypergrad_operator_uniform_auto(
        &mut self,
        curvature: f32,
    ) -> Result<Float32Array, JsValue> {
        let control = self
            .projector
            .gradient_control(curvature)
            .map_err(js_error)?;
        let params = self
            .projector
            .hypergrad_operator_uniform_from_control(&control);
        Ok(Float32Array::from(params.as_ref()))
    }

    /// Workgroup dispatch dimensions matching `hypergradOperatorKernel`.
    #[wasm_bindgen(js_name = hypergradOperatorDispatch)]
    pub fn hypergrad_operator_dispatch(&self, subgroup: bool) -> Uint32Array {
        let dispatch = self.projector.hypergrad_operator_dispatch(subgroup);
        Uint32Array::from(dispatch.as_slice())
    }

    /// Byte layout metadata mirroring the WGSL `FieldSample`, `SpectrumSample`
    /// and uniform structs so WebGPU callers can size buffers without manual
    /// calculations.
    #[wasm_bindgen(js_name = vectorFieldFftLayout)]
    pub fn vector_field_fft_layout(&self) -> CanvasFftLayout {
        self.projector.vector_fft_layout().into()
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
