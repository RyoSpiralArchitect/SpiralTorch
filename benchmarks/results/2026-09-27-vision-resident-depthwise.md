# Vision resident depthwise: bounded browser result

This slice starts from `origin/main` at `fce3502331f232ab73c98720b0d48b8be010b076`
(PR #2137). `ResidentTensor::depthwise_conv2d` keeps NCHW input, `[C, KH, KW]`
weights, `[C]` bias, and output on the same WGPU queue. Python `WgpuTensor`
and browser `WgpuTensor` expose the same Rust operation. Non-contiguous inputs
are packed on GPU in the same submission; inherited validity flags and
non-finite outputs remain guarded until an explicit snapshot. The existing
`st-vision` batch geometry output now feeds this primitive and a `relu`
without a geometry-to-depthwise readback. This is **forward-only** and does
not make the `ConvNeXtBackbone` graph or its backward pass GPU-resident.

## Browser comparison

Three independent Chrome 153.0.8010.54 processes ran on one Apple M4 host.
The Rust runtime reported `BrowserWebGpu` / `Other`; a separate browser probe
reported Apple Metal 3 and no fallback adapter. The probe is not attestation
of identical device identity. Each arm used the same float32 CHW 3x128x128
input, resize to 80x96, seeded horizontal flip at 0.5, center crop to 64x64,
then depthwise 3x3 with a center weight of 1.1 per channel, zero bias, and
`relu`. Weights and bias were uploaded once before timing. The round-trip arm
read the transformed image to the host and re-uploaded it as NCHW before the
**same resident depthwise operation**; the resident arm reshaped the GPU
transform result and continued directly. Both mapped only their final output
after depthwise. Timings are browser wall time from initial image upload
through final output readback, not kernel-only times. Each process used five
warmups and 20 paired samples per arm, alternating execution order.

| Chrome run | Round-trip median | Resident median | Max output error |
| --- | ---: | ---: | ---: |
| 1 | 1.0 ms | 0.7 ms | 0 |
| 2 | 1.0 ms | 0.7 ms | 0 |
| 3 | 1.0 ms | 0.7 ms | 0 |

The same browser fixture compared a five-image, two-channel 9x11 batch after
seeded resize/flip/crop to 6x6 against the CPU reference. With two independent
3x3 center weights (2 and -1), bias, and `relu`, the terminal output's
maximum absolute error was `1.4305114748314196e-7` in all three runs. An
invalid bias shape was rejected. An overflow in an input position unused by
the output was also rejected by the inherited guard. All 12 geometry parity frames and prior
resident NN/training checks passed in each process; page errors were zero.
These timings do not establish cross-device or full-model speedup, nor a
PyTorch comparison. Chrome's short wall intervals are quantized near 0.1 ms.

## Validation and artifacts

- `SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test -p st-backend-wgpu --lib resident_depthwise_matches_reference_and_preserves_guards_on_real_gpu` passed on a non-CPU adapter, including non-contiguous input/weights/bias, asymmetric stride/padding/dilation, inherited invalid flags, and output overflow.
- `cargo test -p st-backend-wgpu --lib` passed 230 tests; `cargo test -p st-vision --features wgpu` passed 68 library and 23 additional tests, including the geometry-to-depthwise handoff.
- `cargo test -p st-vision --lib` passed 61 CPU-default tests. The macOS WGPU CI job now explicitly runs the handoff test with an adapter-required flag.
- `cargo check --manifest-path bindings/st-py/Cargo.toml --features wgpu` and `cargo check --target wasm32-unknown-unknown --manifest-path bindings/st-wasm/Cargo.toml --features webgpu` passed.
- `cargo +1.98.0 clippy --locked -p st-backend-wgpu --target wasm32-unknown-unknown --all-targets -- -D warnings`, the pinned-nightly workspace formatting check, and `git diff --check` passed. The local 1.97 strict-clippy attempt stopped on existing `clippy::chunks_exact_to_as_chunks` allowances unknown to that toolchain, not on this change.
- A fresh 0.4.27 wheel in an isolated venv passed all five `test_vision_wgpu_pipeline.py` tests; the new test exercised the public Python resident handoff.
- A fresh WASM release package passed the real-Chrome `vision-transforms` fixture in three independent processes. The checked-in TypeScript declaration passed `tsc --noEmit --strict --lib esnext,dom`.

Full browser reports, including every paired sample, page errors, adapters,
and asset hashes, plus the exact WASM package and wheel, are retained locally
under `~/Library/Logs/SpiralTorch/vision-resident-depthwise-v1/`. The public
claim basis is the table and checks above. Earlier pre-guard browser runs
remain in the same archive as exploratory records and are not used for the
table. Hashes bind this summary to the unabridged local originals:

| Local artifact | SHA-256 |
| --- | --- |
| `spiraltorch-vision-resident-depthwise-guard-run-1.json` | `ad1c75ad2ce8cd66e19088b4e27412388439b5501e68665d4499f60e8691d17f` |
| `spiraltorch-vision-resident-depthwise-guard-run-2.json` | `ee070bbc495b53fa504dca116b23150ef76c3fdd37fe88fff5a02364ea887930` |
| `spiraltorch-vision-resident-depthwise-guard-run-3.json` | `0d067e9f526dc93698fcc6815ea697c316fa2951ff4105c644e4196d7000fbdd` |
| `spiraltorch_wasm_bg.wasm` | `f8b12997de6875c66bdcf445539cf35620bb6bc64cb57bc5515c0fd732cf6105` |
| `spiraltorch-0.4.27-cp38-abi3-macosx_11_0_arm64.whl` | `091b7890697a07ef5d2d6521805ff37c74758430df58281f076df04cea6a0d1b` |

The browser page hash in all three reports is
`50fae0daccfc45e369c1c093db26defab180df99c58c25eba009ac2fbc17d7ee`.
From the repository root, rebuild and replay into fresh output paths:

```sh
wasm-pack build bindings/st-wasm --target web --out-dir /tmp/vision-resident-depthwise-module --release -- --features webgpu
NODE_PATH=/path/to/playwright/node_modules node tools/test_resident_browser.cjs \
  /tmp/vision-resident-depthwise-module /path/to/Chrome \
  /tmp/vision-resident-depthwise-run.json '' '' '' '' vision-transforms
maturin build --release --manifest-path bindings/st-py/Cargo.toml --out /tmp/vision-resident-depthwise-wheels
```

The remaining gates are a resident ConvNeXt-style multi-layer graph with
parameter lifecycle, backward/training ownership, and matched real-dataset
quality/throughput. The native host-Tensor WGPU path still reads back on every
depthwise forward and keeps its CPU backward reference.
