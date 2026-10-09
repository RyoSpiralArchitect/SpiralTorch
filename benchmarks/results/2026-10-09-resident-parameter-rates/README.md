# Per-parameter SGD and frozen geometry

This is a synthetic correctness record, not a speed or language-quality result.
The Rust parameter owner accepts one SGD rate per tensor, preserving one atomic
accept/reject decision. A zero rate freezes parameter bits without detaching the
geometry-to-embedding VJP. Native WGPU and actual browser WebGPU run the same Rust
controls. Python supplies only the independent CPU-f32 reference and comparison.

## Fixed checks and observed results

- Two existing causal-geometry fixtures: 23 and 37 parameter tensors, respectively
  one and two residual blocks. The first isolates metric scores with zero Q/K;
  the second includes external score biases and Topos.
- Sixteen queued CE/VJP/SGD updates per model, no intermediate learner readback.
  Geometry's five/six tensor rates are zero; other tensor rates are `0.125`.
- All losses, parameter values and embedding cotangents pass the existing
  `3e-6 + 5e-5 * abs(reference)` bound. Every geometry derivative also has a
  nonzero reference norm above `1e-8` and relative L2 error at most `0.002`.
- Native maximum weight error: `3.7997961044311523e-7`; browser:
  `6.034970283508301e-7`. Maximum geometry-gradient relative L2 error:
  `6.20160047443641e-5` native, `0.00011958028992853661` browser.
- Geometry parameters retain initial float32 bits at every step, while both
  embedding tables change. Existing off/detached controls still distinguish a
  missing geometry pullback. Freezing weights does not freeze coordinates.
- Five native parameter-owner tests pass, including nonuniform rates, strided
  values/gradients, signed zero, invalid late-rate preflight, foreign/stale
  gradients, later-candidate overflow, a guarded invalid gradient on a frozen
  tensor, all-or-none rejection and retained snapshots/receipts. The same shared
  contract passes in the browser, including bitwise scalar-vs-uniform-vector SGD.
- Both native model tests pass. The browser also reruns the existing four-model
  global-rate/causality/VJP/checkpoint regression. Existing default PyTorch fixture
  regeneration is byte-for-byte unchanged. Ten stdlib comparator tests pass;
  their fabricated traces test rejection behavior, not the actual learner.

Independent read-only review found one test-coverage weakness: the frozen
invalid-gradient case initially had both an invalid guard and nonfinite payload.
That could not isolate a guard-dropping regression. The final shared test adds a
finite zero payload with an inherited invalid guard, requires stage 1 plus
`INVALID_TENSOR_FLAG`, and checks both parameter tensors remain bitwise unchanged.
The parent rebuilt and reran the native and actual-browser backend controls;
both pass. Focused independent source re-review has no remaining findings. The
reviewer did not execute GPU tests. Earlier backend artifacts are retained, but
the `*-reviewed` artifacts are the selected guard-isolation evidence. Model
numerical sources/comparisons are unchanged and were not rerun for this test-only
repair.

The original frozen-geometry reference was generated and hashed before either
GPU run. It is not regenerated to fit a device result. Reports, weights,
reference arrays, binaries and logs stay local. Only scalar comparison results,
hashes, source and reproduction instructions are public.

## Reproduce

Use a fresh local directory in `$RAW`, PyTorch 2.12.1 CPU, Rust 1.97.0, and
`wasm-bindgen` 0.2.129. Measurements here used one Apple M4 host. Start from this
commit; `validation.json` identifies changed executable sources and artifacts.

```sh
python3 -I -B tools/generate_resident_byte_geometry_torch_fixture.py "$RAW/frozen-geometry-torch.json" --freeze-geometry
shasum -a 256 "$RAW/frozen-geometry-torch.json"
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 CARGO_INCREMENTAL=0 cargo test --locked --release -p st-nn --no-default-features --features wgpu --test resident_byte_decoder -- --nocapture --test-threads=1 > "$RAW/native-tests.log" 2>&1
awk '/spiraltorch.resident_byte_parameter_rates.validation.v1/ { print substr($0,index($0,"{")) }' "$RAW/native-tests.log" > "$RAW/native.json"
python3 -I -S -B tools/verify_byte_parameter_rates.py "$RAW/frozen-geometry-torch.json" "$RAW/native.json" "$RAW/native-comparison.json"
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 CARGO_INCREMENTAL=0 cargo test --locked --release -p st-backend-wgpu resident_training::parameters::tests -- --nocapture --test-threads=1
python3 -I -S -B tools/test_verify_byte_parameter_rates.py -v
```

For the browser, build `resident_byte_decoder_browser` from `st-nn` and
`resident_parameters_browser` from `st-backend-wgpu` for
`wasm32-unknown-unknown`, release/locked, with host linker environment removed.
For `st-nn`, use `--no-default-features --features wgpu`. Run `wasm-bindgen
--target web` into `target/resident-byte-decoder-web` and
`target/resident-parameter-rates-web`, respectively. Serve the repository on
loopback and open these pages **sequentially**, never concurrent GPU tests:

- `crates/st-nn/tests/byte_parameter_rates_browser.html`
- `crates/st-backend-wgpu/tests/parameter_rates_browser.html`
- `crates/st-nn/tests/byte_decoder_browser.html` (existing four-model regression)

Download the first page's Rust JSON unchanged to `$RAW/browser.json` and run
the same comparator with that report. The pages execute Rust/WASM, not a
JavaScript training surrogate. The backend report identifies `BrowserWebGpu`;
browser adapter metadata does not establish a cross-hardware qualification.

## Boundaries

Rates are caller-owned inputs to each update. A model checkpoint does not retain
the rate vector, freeze policy or schedule. The corpus-study v1 protocol remains
ordinary-vs-learned geometry with its scalar rate; it has not yet added frozen
geometry or capacity-matched arms. Those are the next quality-study controls,
not conclusions established by these two tiny deterministic fixtures.
