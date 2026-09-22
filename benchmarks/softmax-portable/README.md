# Portable softmax: finite domain and synchronization

The canonical 32-byte-parameter WGPU softmax accepts finite f32 logits,
row strides, Chimera layout, softmax, all-peak hardmax, and an optional all-peak
mask. Maxima initialize to the lowest **finite f32**, not `-1e30`: a one-element
row containing `-f32::MAX` must produce 1, not 0. This is not a new policy for
NaN/infinite inputs. Equal maxima all receive 1 in the mask; this low-level
mask is not a first-index/one-hot argmax.

`st_backend_wgpu::softmax::Pipelines::from_embedded(device, supports_subgroup)`
constructs the same bundled shaders and binding layout without a filesystem,
including on WASM. Subgroups require an actually enabled device feature;
the portable workgroup pipeline always exists. The caller retains the existing
responsibility for valid finite data, correctly sized buffers, parameter
layout, and explicit observation. No new Python/WASM semantic implementation,
host-Tensor residency change, ResidentGraph stage, training, or autograd path
is introduced. The existing WGPU-backed Tensor softmax embeds the repaired
canonical shader; the strict Tensor regression verifies that connection.

The two 16-byte row-softmax subgroup aliases retain their separate ABI. The
old `row_softmax_subgroup.wgsl` now selects a row by workgroup rather than
global invocation and handles zero columns uniformly. Real native subgroup
tests check both aliases, padded rows, and widths beyond a workgroup.

## What Is Measured

Both timed routes use the **corrected** finite-domain shader, identical explicit
binding-layout policy, prepared inputs/outputs, and one dispatch per operation.
The control reinstates just two redundant entry barriers. The candidate relies
on the already-present first barrier inside each reduction, preserving all
necessary synchronization. A wrong numerical result is never a speed baseline.

Six row/column pairs: (1,31), (1,256), (17,257), (65,1025), (128,1025),
(17,4096). Each runs softmax alone and softmax plus an all-peak mask.
Three warmups, nine paired blocks, bursts 1/4, three serial rounds with rotating
runtime order: native/browser/Torch, browser/Torch/native, Torch/native/browser.
Every run first checks 48 finite-domain/layout/mode cases; subgroup correctness
is exercised when available but subgroup performance is **not** timed.

Intervals include encoding, binding creation, submission, terminal owning
CPU copy/map/completion; inputs, shader compilation, numerical validation and
serialization are outside. Burst4 observes its last output. Eager f32 Torch
CPU/MPS also uses reusable output/scratch buffers (`softmax(out=...)`), with
`amax/eq` for all tied peaks. Its final CPU tensor copy is timed, conversion to
a Python list is not. Rust constructs an owning f32 Vec within its interval.
These are application-level paths, not matched kernels or `torch.compile`.

The frozen gate is `abs(error) <= 2e-7 + 2e-6 * abs(reference)`.
Every output is compared with an independent scalar-f64 oracle, and Torch also
cross-checks its CPU-f64 softmax. All conditions and slower/noisy results are
retained. A shared M4 desktop and a coarse browser clock do not support a
universal speedup, training, model-quality, or PyTorch superiority claim.
The browser nonfallback probe is separate from the Rust device, not an
attestation of it; native and browser runtime identities are also recorded.

## Replay

Use the measured source commit, Rust 1.98.0, and matching wasm-bindgen 0.2.104.
Keep each output path new. Limit local CPU work to four threads.

```sh
CARGO_BUILD_JOBS=4 cargo +1.98.0 build --locked --release -p st-backend-wgpu --example softmax_portable_bench
target/release/examples/softmax_portable_bench > /NEW/native.json
CARGO_BUILD_JOBS=4 cargo +1.98.0 build --locked --release -p st-backend-wgpu --target wasm32-unknown-unknown --example softmax_portable_bench_browser
wasm-bindgen target/wasm32-unknown-unknown/release/examples/softmax_portable_bench_browser.wasm --target web --out-dir /NEW/wasm --out-name spiraltorch_wasm
NODE_PATH=/PATH/TO/node_modules node tools/test_resident_browser.cjs /NEW/wasm /PATH/TO/CHROME /NEW/browser.json '' '' '' '' softmax-portable-bench
python3 -I -B benchmarks/softmax-portable/softmax_torch.py /NEW/native.json > /NEW/torch.json
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.98.0 test --locked --release -p st-backend-wgpu -- --test-threads=1
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 SPIRALTORCH_STRICT_GPU=1 cargo +1.98.0 test --locked --release -p st-tensor --no-default-features --features cpu,wgpu_dense --test wgpu_softmax_extremes
python3 -I -B benchmarks/softmax-portable/test_softmax_protocol.py
python3 -I -B benchmarks/softmax-portable/test_softmax_evidence.py
```

`softmax_protocol.py` requires three each of `--native`, `--browser`, `--torch`,
in round order. The complete 3,888 intervals are summarized without dropping
unfavorable cells. `softmax_evidence.py` reuses the existing append-only evidence
engine with a distinct softmax contract, not NeRF input/math assumptions.

Raw arrays, binaries, and all failed attempts remain local. The public archive
keeps intervals, numerical errors, commands, source and raw hashes, runtime
metadata, screening results, and clean-source validation. `publish RESULTS
--raw-root RAW` requires every listed stage at one clean implementation commit;
`verify RESULTS [--raw-root RAW] [--source-root CHECKOUT]` checks fixity and
recomputes summaries, **not** GPU/numerical reexecution. Screening and final
rounds are published separately and never pooled.
