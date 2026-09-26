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
| 2 | 1.1 ms | 0.7 ms | 0 |
| 3 | 1.0 ms | 0.7 ms | 0 |

The same browser fixture compared a five-image, two-channel 9x11 batch after
seeded resize/flip/crop to 6x6 against the CPU reference. With two independent
3x3 center weights (2 and -1), bias, and `relu`, the terminal output's
maximum absolute error was `1.4305114748314196e-7` in all three runs. An
invalid bias shape was rejected. An overflow in an input position unused by
the output was also rejected by the inherited guard. Aliased read-only
input/weight/bias views were accepted. All 12 geometry parity frames and prior
resident NN/training checks passed in each process; page errors were zero.
These timings do not establish cross-device or full-model speedup, nor a
PyTorch comparison. Chrome's short wall intervals are quantized near 0.1 ms.

## Validation and artifacts

- `SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test -p st-backend-wgpu --lib resident_depthwise_matches_reference_and_preserves_guards_on_real_gpu` passed on a non-CPU adapter, including non-contiguous input/weights/bias, asymmetric stride/padding/dilation, inherited invalid flags, and output overflow.
- `cargo test -p st-backend-wgpu --lib` passed 230 tests; `cargo test -p st-vision --features wgpu` passed 68 library and 23 additional tests, including the geometry-to-depthwise handoff.
- `cargo test -p st-vision --lib` passed 61 CPU-default tests. The macOS WGPU CI job now explicitly runs the handoff test with an adapter-required flag.
- `cargo check --manifest-path bindings/st-py/Cargo.toml --features wgpu` and `cargo check --target wasm32-unknown-unknown --manifest-path bindings/st-wasm/Cargo.toml --features webgpu` passed.
- `cargo +1.98.0 clippy --locked -p st-backend-wgpu --target wasm32-unknown-unknown --all-targets -- -D warnings`, the pinned-nightly workspace formatting check, and `git diff --check` passed. The local 1.97 strict-clippy attempt stopped on existing `clippy::chunks_exact_to_as_chunks` allowances unknown to that toolchain, not on this change.
- A fresh 0.4.27 wheel in an isolated Python 3.12 venv passed all five `test_vision_wgpu_pipeline.py` tests; the new test exercised the public Python resident handoff. An additional Python 3.9 import attempt failed on existing `zspace_inference.py` use of `@dataclass(slots=True)`, despite `requires-python = ">=3.8"`. That packaging/support mismatch is not fixed by this vision patch.
- A fresh WASM release package passed the real-Chrome `vision-transforms` fixture in three independent processes. The checked-in TypeScript declaration passed `tsc --noEmit --strict --lib esnext,dom`.

Full browser reports, including every paired sample, page errors, adapters,
and asset hashes, plus the exact WASM package and wheel, are retained locally
under `~/Library/Logs/SpiralTorch/vision-resident-depthwise-v1/`. The public
claim basis is the table and checks above. Earlier pre-guard and pre-read-only
browser runs remain in the same archive as exploratory records and are not used for the
table. Hashes bind this summary to the unabridged local originals:

| Local artifact | SHA-256 |
| --- | --- |
| `spiraltorch-vision-resident-depthwise-final-run-1.json` | `3ff5b61fa2ad077c0b6b78a65697c5c3f514307e09ce5dd129b46bf0fccd301f` |
| `spiraltorch-vision-resident-depthwise-final-run-2.json` | `43abfe90dfd281dbefeadda5b6e21caae785c9ed20bb69816f9be17da15a0896` |
| `spiraltorch-vision-resident-depthwise-final-run-3.json` | `e1a7a5b7aba8c265a1853f30a5338e9e4fa237ae32bd89a00c4b1659592aaa9f` |
| `spiraltorch_wasm_bg.wasm` (`spiraltorch-vision-resident-depthwise-wasm-v2/`) | `88f4d852c566baa9a8f9e3ff45d1fb35c4820122098fc08fc8bbb54e533366e1` |
| `spiraltorch-0.4.27-cp38-abi3-macosx_11_0_arm64.whl` (`spiraltorch-vision-resident-depthwise-wheels-v2/`) | `d2630716460abfa519e9dddcb621aef9598ea62cf9cc74425c7077540628f61f` |

The browser page hash in all three reports is
`1aac1e03d234cbc36bff346fecb8210274d19366e5c35d8580e855ab5a94265b`.
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
