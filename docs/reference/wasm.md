# WASM demos and browser execution

[Documentation index](../README.md) | [Project entry](../../README.md)

This is the detailed source-tree reference, moved from the README.
Examples retain their individual feature, device, and optional-dependency requirements.
A source-tree API is not a claim that an older PyPI wheel exposes it.
Run repository commands from the repository root.

## Contents

- [🌐 WebAssembly (WASM) demos (not a stub)](#-webassembly-wasm-demos-not-a-stub)
- [Fractal uring scheduler + WASM canvas loop](#fractal-uring-scheduler--wasm-canvas-loop)

### 🌐 WebAssembly (WASM) demos (not a stub)

SpiralTorch’s WASM bindings are now **fully runnable** in the browser: geometry evaluation, small training loops, and WebGPU visualisation work end-to-end.

- **Mellin log grid (WASM):** evaluate meshes + run a tiny “match reference” training loop
  `bash scripts/wasm_demo.sh mellin-log-grid dev`
- **Canvas hypertrain (WASM):** `FractalCanvas` learning loop + WebGPU trail + FFT probe + **target-MSE supervised mode**
  `bash scripts/wasm_demo.sh canvas-hypertrain dev`
- **WASM → API LLM context:** pass exported browser reports into the hosted-model
  route matrix with `--wasm-report report.json`, or run the keyless bridge with
  `PYTHONPATH=bindings/st-py python3 bindings/st-py/examples/api_llm_wasm_context_runtime.py --wasm-report report.json`
  For a live OpenAI smoke, run
  `PYTHONPATH=bindings/st-py python3 bindings/st-py/examples/openai_api_llm_wasm_context_runtime.py --wasm-report report.json --trace-jsonl /tmp/spiraltorch-openai-wasm-trace.jsonl`
  Use `--wasm-report-dir runs --wasm-report-recursive --wasm-max-reports 3` to
  select the best browser-side learning runs before they become LLM context;
  Python can audit promotion readiness with `st.audit_wasm_report(...)` /
  `st.audit_wasm_report_context(...)`, then persist the selected handoff with
  `st.write_wasm_report_context_artifact(...)`.

Browser runtimes can also fuse telemetry and Z-space partials through the same
Rust-owned contracts used by Python. WASM adds only client metadata; metric
aliases, weighting, suppression, reduction, flattening, and audit semantics stay
in `st-core::telemetry::zspace_fusion`:

```ts
const projected = zspaceMetricGradientProjectionObject({
  metrics: { speed: 0.4, memory: 0.2, stability: 0.8, frac: 0.3, drs: -0.1 },
  dimension: 2,
});
const fused = zspacePartialFusionObject({
  partials: [
    {
      metrics: { velocity: 0.4, gradient: projected.gradient },
      gradient_basis: projected.basis,
      origin: "canvas",
    },
    { metrics: { speed: 0.8 }, weight: 2.0, origin: "webgpu" },
  ],
  strategy: "mean",
  gradient_alignment: "strict",
  telemetry: [{ browser: { webgpu_ready: true } }],
});
```

Partial fusion contract v3 rejects gradients whose declared bases differ even
when their vector lengths match. Tagged and untagged gradients cannot be mixed,
so positional coincidence is not treated as semantic compatibility. It also
rejects ragged active gradients by default instead of silently inventing
coordinates. Set `gradient_alignment: "pad_zero"` only for an explicit
legacy-compatible replay within one basis; the result reports
`gradient_padding_applied`, `gradient_padded_source_count`, and per-source
`gradient_padded` audit fields. `zspaceMetricGradientProjectionObject` and
`st.zspace_metric_gradient_projection(...)` expose the same Rust-owned periodic
projection from `[speed, memory, stability, frac, drs]`; API response and
distortion adapters use that basis rather than constructing unrelated feature
vectors in Python. `ApiLLMZSpaceRuntime` additionally requests
`metric_gradient_dimension` during partial fusion, so Rust first fuses the
named scalar metrics and then projects one canonical runtime-width gradient;
heterogeneous positional gradients from context clients are replaced and
reported instead of being silently relabelled. Python preserves that native
receipt as `ZSpaceInference.fusion`, so API traces expose the exact projection
and replacement counts. The `mean`, `last`, `max`,
`min`, `median`, and `sum` reducers apply identically to scalar metrics and
compatible gradient coordinates in Rust.

The Rust Topos projection follows the same rule: contract v2 includes the
six-axis `spiraltorch.topos.control_signal.axes.v1` basis, its ordered channels,
and the exact truncation/zero-padding formula. Python and WASM transport that
identity rather than naming or reconstructing the control vector themselves.

Posterior decoding follows that same Rust-first boundary. Python's
`ZSpacePosterior` and the browser's `zspacePosteriorDecodeObject` /
`zspacePosteriorProjectObject` are clients of
`st-core::inference::zspace_posterior`; neither layer reconstructs spectral
energy, gradients, barycentric weights, residual confidence, or telemetry
adjustments. For an auditable Python payload, call
`st.zspace_posterior_decode(...)` or `st.zspace_posterior_project(...)` and
check the returned `contract_version` and `semantic_owner`. Posterior v2 uses a
one-sided Parseval-normalized spectrum, reports spectral energy/error/centroid,
and keeps its latent finite-difference gradient in the versioned
`st.ZSPACE_POSTERIOR_LATENT_GRADIENT_BASIS`. An external gradient is never
resized or substituted for that latent gradient: attach an explicit
`gradient_basis`, then read the preserved values from `control_gradient`.
`ZSpaceTrainer.step_partial(...)` consumes the Rust latent gradient and does not
silently apply that external control; a control-to-latent projection must be an
explicit, basis-aware Rust contract.
Residual RMS is computed only over observed canonical metrics, while telemetry
can reduce confidence through an audited reliability factor but cannot erase
geometric residual or increase confidence.

Coherence diagnostics follow the same rule. `ZSpaceCoherenceSequencer` exposes
the complete Rust diagnostics and linguistic contour, while
`st.zspace_coherence_project(...)` delegates gain validation, base metric
projection, dimension-normalized entropy/concentration summaries, structural
classification, and contour metrics to `st-core::inference::zspace_coherence`:

```python
import spiraltorch as st

topos = st.OpenCartesianTopos(-0.9, 1e-5, 10.0, 32, 1024)
sequencer = st.ZSpaceCoherenceSequencer(16, 2, -0.9, topos=topos)
x = st.Tensor((1, 16), data=[0.1] * 8 + [0.8] * 8)
out, coherence, diagnostics = sequencer.forward_with_diagnostics(x)
contour = sequencer.emit_linguistic_contour(x)
control = diagnostics.control
contract = st.zspace_coherence_project(
    diagnostics,
    coherence=coherence,
    contour=contour,
)
assert contract["semantic_owner"] == "st-core::inference::zspace_coherence"
assert contract["derived"]["distribution_source"] == "normalized_weights"
assert contract["classification"]["label"] == diagnostics.observation.label
assert control["contract_version"] == "spiraltorch.zspace_coherence_control.v1"
assert contract["control"]["spectral_radius"] == control["spectral_radius"]
```

Python remains the orchestrator; it does not reconstruct the Rust diagnostics
or carry a second projection formula. Rust preserves raw `mean_coherence` and
raw Shannon entropy for audit, but derives trainer-facing spectral radius,
entropy, and pressure from normalized HHI concentration and `H / ln(N)` so
channel count alone cannot change the control signal. Pass diagnostics directly
with `trainer.push_coherence_diagnostics(diagnostics)`; replayed trace events are
accepted only when their Rust contract provenance, complete probability-simplex
witness, summaries, classification, and control values still agree. Trace schema
v2 carries that witness, and the Rust bridge decodes the stable plugin record
before rebuilding the projection; legacy scalar-only traces remain readable for
inspection but cannot command trainer controls.
The same versioned Rust policy emits `background`, `symmetric_pulse`,
`cascade_imbalance`, or `diffuse_drift` with an explicit reason and thresholds;
trace, Python, and WASM only transport that decision. Call
`diagnostics.classify(...)` or pass `background_energy_ratio_max` and
`cascade_energy_ratio_min` to `st.zspace_coherence_project(...)` to select a
different Rust policy.

Projection contract v2 also validates that all supplied evidence describes one
observation: entropy, support counts, and the dominant channel must agree with
`normalized_weights`, while `mean_coherence` must agree with the raw
`coherence` response when present. Python and WASM surface the resulting Rust
error instead of repairing or reinterpreting contradictory evidence.

Portable clients can explicitly build and validate the same evidence boundary:

```python
import spiraltorch as st

witness = st.zspace_coherence_distribution_witness([0.5, 0.3, 0.2])
summary = st.validate_zspace_coherence_distribution_witness(witness)
assert witness["semantic_backend"] == "rust"
assert summary["channels"] == 3
```

Runtime plan scoring follows the same ownership rule. Variational free energy
is evaluated only by `st-core::heur::free_energy`; Python and WASM transport the
same request and return the versioned Rust report. Missing or numerically tiny
band evidence returns to the configured prior, band potentials are centred on
that prior, and `acceptance_probability` is the explicit two-state Gibbs
probability against a neutral `F=0` candidate rather than a calibrated
confidence claim:

```python
import spiraltorch as st

report = st.zspace_free_energy(
    reference_loss=0.8,
    candidate_loss=0.5,
    step_time_ms=12.0,
    memory_mb=256.0,
    band={"above": 0.6, "here": 0.3, "beneath": 0.1},
)
print(report["free_energy"], report["acceptance_probability"])
```

```ts
const report = zspaceFreeEnergyObject({
  observation: {
    reference_loss: 0.8,
    candidate_loss: 0.5,
    band: { above: 0.6, here: 0.3, beneath: 0.1 },
  },
});
```


### Fractal uring scheduler + WASM canvas loop

Feed those spectra directly into an async-friendly fractal loop without ever
allocating more than a small ring buffer. The `UringFractalScheduler` keeps the
latest relation patches in a Tokio-uring style queue, blends them by coherence,
and hands the result straight to your browser front-end.

```rust
use st_tensor::{Tensor, PureResult};
use st_tensor::fractal::{FractalPatch, UringFractalScheduler};

async fn stream_waveforms(samples: Vec<Tensor>) -> PureResult<Tensor> {
    let scheduler = UringFractalScheduler::new(32)?;
    for (depth, relation) in samples.into_iter().enumerate() {
        let patch = FractalPatch::new(relation, 0.9, 0.7, depth as u32)?;
        // Works on any executor; tokio-uring, tokio, or synchronous loops.
        scheduler.push_async(patch).await?;
    }
    scheduler.fold_coherence()
}
```

For browser builds, wire the folded relation into a WebAssembly export that
paints onto `<canvas>` without tokenising text or duplicating buffers:

```rust
use st_tensor::fractal::UringFractalScheduler;
use wasm_bindgen::prelude::*;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

#[wasm_bindgen]
pub struct FractalCanvas {
    scheduler: UringFractalScheduler,
}

#[wasm_bindgen]
impl FractalCanvas {
    #[wasm_bindgen(constructor)]
    pub fn new(capacity: usize) -> Result<FractalCanvas, JsValue> {
        let scheduler = UringFractalScheduler::new(capacity)
            .map_err(|err| JsValue::from_str(&err.to_string()))?;
        Ok(Self { scheduler })
    }

    pub fn render(&self, canvas: HtmlCanvasElement) -> Result<(), JsValue> {
        let ctx: CanvasRenderingContext2d = canvas
            .get_context("2d")?
            .ok_or("missing 2d context")?
            .dyn_into()?;
        let frame = self
            .scheduler
            .fold_coherence()
            .map_err(|err| JsValue::from_str(&err.to_string()))?;
        let spectrum = frame.data();
        for (x, value) in spectrum.iter().enumerate() {
            let intensity = (value.clamp(0.0, 1.0) * 255.0) as u8;
            ctx.set_fill_style(&format!("rgb({0},{0},{0})", intensity).into());
            ctx.fill_rect(x as f64, 0.0, 1.0, canvas.height() as f64);
        }
        Ok(())
    }
}
```

And keep the JavaScript glue feather-light:

```html
<canvas id="zspace" width="512" height="32"></canvas>
<script type="module">
import init, { FractalCanvas } from "./pkg/spiraltorch_wasm.js";
const wasm = await init();
const canvas = document.getElementById("zspace");
const fractal = new FractalCanvas(64);
await fractal.render(canvas);
const spectrum = fractal.vectorFieldFft(false);
console.log(`fft bins=${spectrum.length / 8}`);
const kernel = fractal.vectorFieldFftKernel(true);
console.log(kernel.split("\n")[0]);
const uniform = fractal.vectorFieldFftUniform(false);
console.log(`fft uniform=${uniform.join(',')}`);
const layout = fractal.vectorFieldFftLayout();
console.log(`fft field bytes=${layout.fieldBytes} stride=${layout.fieldStride}`);
const dispatch = fractal.vectorFieldFftDispatch(true);
console.log(`fft dispatch=${dispatch.join('x')}`);
</script>
```

Pixels become Z-space relations, the scheduler keeps memory bounded, and the
entire loop stays panic-free even under aggressive streaming.

The returned spectrum stores `[energy_re, energy_im, chroma_r_re, chroma_r_im,
chroma_g_re, chroma_g_im, chroma_b_re, chroma_b_im]` per bin, so Canvas
Transformers can slice the energy or chroma lanes directly or feed the full
tensor back through `fft_inverse_in_place` for quick spatial reconstruction.

When dispatching the WGSL kernel, bind the colour field as a tightly-packed
array of `FieldSample { energy, chroma }`, store the complex spectrum in a
matching `SpectrumSample` buffer, and provide the canvas dimensions plus an
inverse flag through a `CanvasFftParams` uniform struct. The
`vectorFieldFftUniform` helper yields the `[width, height, inverse, padding]`
`Uint32Array` so you can upload the uniform buffer directly without worrying
about alignment, `vectorFieldFftLayout` reports the byte lengths and strides for
the field/spectrum storage buffers, and `vectorFieldFftDispatch` returns the
`[x, y, z]` workgroup counts that correspond to the generated WGSL (respecting
subgroup or full wave execution).

Need FFT heuristics alongside the canvas?  WebAssembly exports now ship auto
planning helpers and CPU fallbacks:

```javascript
import init, { auto_plan_fft, fft_forward } from "./pkg/spiraltorch_wasm.js";

await init();
const plan = auto_plan_fft(512, 4096, 128, true);
if (plan) {
  console.log(`radix=${plan.radix} tile=${plan.tileCols}`);
  const wgsl = plan.wgsl();
  const spiralk = plan.spiralkHint();
}

// Run a radix-2/4 FFT on interleaved re/im data
const freqDomain = fft_forward(timeDomainBuffer);
```

If you maintain a `WasmTuner`, call `planFft` to reuse your override table and
capture WGSL/SpiralK artifacts without leaving the browser.  The bindings now
understand plain JavaScript objects in addition to JSON strings, so you can
hydrate a tuner from baked data and persist live edits without extra parsing:

```javascript
const records = [{ rows: 256, cols_min: 0, cols_max: 4095, k_max: 128, sg: true, wg: 128 }];
const tuner = WasmTuner.fromObject(records);
tuner.mergeObject([
  { rows: 512, cols_min: 4096, cols_max: 16383, k_max: 256, sg: true, tile_cols: 1024 },
]);
const overrides = tuner.toObject();
// Extract overrides and resolved plans as JSON or plain JS objects without
// constructing intermediate WasmFftPlan instances by hand.
const fallbackPlan = tuner.planFftWithFallback(512, 4096, 128, true);
const fallbackJson = tuner.planFftWithFallbackJson(512, 4096, 128, true);
const fallbackObject = tuner.planFftWithFallbackObject(512, 4096, 128, true);
const resolution = tuner.planFftResolution(512, 4096, 128, true);
const resolutionJson = tuner.planFftResolutionJson(512, 4096, 128, true);
const resolutionObject = tuner.planFftResolutionObject(512, 4096, 128, true);
if (resolution.source === WasmFftPlanSource.Override) {
  console.log(`override tile=${resolution.plan.tileCols}`);
}
const snapshot = resolution.toJson();
const hydrated = ResolvedWasmFftPlan.fromJson(snapshot);
const report = tuner.planFftReport(512, 4096, 128, true);
const overrideJson = tuner.planFftJson(512, 4096, 128, true);
const overrideObject = tuner.planFftObject(512, 4096, 128, true);
```

---
