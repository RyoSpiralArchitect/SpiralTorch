# NeRF submission scheduling: keep the default

Measured source: `89281c5874c72e1659254f3d3cf4cddb963cd22d`.
This artifact compares separate direct sampling/NN/compositing submissions
against the explicit single-submission API, not against stable-workspace copies.
Both routes use the same current NN implementation and shaders.

## Numerical Result

All twelve shapes/fields, six runtime routes and three rounds passed the fixed
independent f64-integration/f32-NN oracle gate:
`abs(error) <= 4e-7 + 4e-6 * abs(reference)`.
Maximum absolute error was `1.2516975402832031e-6` for WGPU/MPS and
`1.1920928955078125e-7` for eager CPU. The maximum scaled error was below 0.494.
Every measured terminal result was also checked against its route reference.

All 175 backend tests passed with real-GPU tests explicitly enabled, including
producer/consumer aborts that leave a GPU marker unchanged, stale binding-key
regressions, output/guard retention, shape/device/width admission and both public
rendering paths. Ten admission/archive tests passed; native and wasm32 strict
Clippy and the legacy native/browser benchmark modes passed.

## Performance Result

3,888 measured intervals: twelve shapes/fields, bursts 1/4, nine paired blocks,
three rotated serial rounds and six routes. Geometric means below aggregate the
24 per-condition medians of paired separate/single ratios; greater than one
favors single submission.

| Runtime | Geometric Mean | Range | Strictly Above 1 |
| --- | ---: | ---: | ---: |
| Native Metal | 1.0447 | 0.9337-1.2059 | 19/24 |
| Browser WebGPU | 0.9551 | 0.7500-1.0000 | 0/24 |

These are descriptive observations, not confidence bounds. Browser timings are
coarsely quantized (approximately 0.1 ms); several ratios differ from one only
by floating-point subtraction noise. Do not interpret strict counts as
statistically significant wins or regressions.

**Decision: keep `render_graph` as the three-submission default.** The single
route is explicitly selectable for target-specific measurements and all-or-none
host encoding, not an automatic optimization. Fewer queue submissions do not
remove position packing, allocations or arithmetic, and may change host/GPU
overlap; the cause of the observed timing differences was not profiled.

The eager Torch controls are not uniformly slower: at 1x1/affine/burst1, CPU
took 0.0849 ms versus native separate/single 0.4964/0.4748 ms; at
256x256/affine/burst1, MPS took 1.0957 ms versus 1.2783/1.2343 ms.
The full CPU/MPS timings and all unfavorable WGPU conditions are in
`results.json`; no per-condition winners were selected for aggregation.

## Provenance And Replay

Apple M4 native Metal; Chrome 153.0.8010.48; PyTorch 2.12.1 CPU/MPS, eager f32,
4 intra-op/1 inter-op CPU threads, no compile/autograd. Rust 1.98.0,
wasm-bindgen 0.2.104; formatting nightly-2026-04-15.
The browser runtime reports Other/BrowserWebGpu without a device name.
A separate non-fallback apple/metal-3 probe is not exact runtime attestation.
This was a shared desktop, not dedicated hardware. No CUDA, Vulkan, training,
scene-quality or universal PyTorch superiority conclusion follows.

`results.json` contains every accepted interval and reported error;
`exploration.json` retains the earlier full screening result and all recorded
preflight receipts with deduplicated source identities. That exploratory run
(native 1.0310, browser 0.9644) is not pooled with the clean-source result.
Raw arrays, native/WASM binaries and logs (about 51 MiB at publication) remain at
`/Users/ryospiralarchitect/Library/Logs/SpiralTorch/nerf-single-submit-20260921`.
`local-raw-manifest.json` records their hashes and sizes.

Use the [protocol and replay commands](../../nerf-single-submit/README.md),
with the frozen source above. The 21 accepted source-stable stages are recorded
in `validation.json`. Raw fixity, complete-summary recomputation and measured
source checks all passed before publication. Verify published bytes with:

```sh
python3 -I -B benchmarks/nerf-single-submit/archive.py verify benchmarks/results/2026-09-21-nerf-single-submit
```

Add `--raw-root RAW` to verify retained bytes and recompute summaries, or
`--source-root FROZEN_CHECKOUT` to check source hashes. These checks do not
reexecute the numerical oracle or GPU. Later source changes require a new
artifact rather than rewriting this archive.
