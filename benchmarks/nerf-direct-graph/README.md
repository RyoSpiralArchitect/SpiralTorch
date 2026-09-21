# Resident NeRF: direct NN connection

`ResidentNerf::render_graph` connects the existing sampling, direct resident NN
forward and compositing APIs. It is position-only and forward-only; it does not
migrate `NerfField`, provide a NeRF VJP, or claim scene quality. The old staged
graph connection remains supported as the within-runtime control.

## Protocol

- Frozen ray/sample shapes: 1x1, 1x64, 65x64, 256x64, 1024x64, 256x256.
- Affine 3->4 and 3->32->4 ReLU fields; emitted f32 rays/weights, seed 17 counter
  jitter; register-2x2 sequential matmul for both WGPU routes.
- Three warmup blocks and nine measured paired blocks, per condition and burst.
  Rotate route order each block. Measure bursts of 1 and 4 complete renders,
  observing only the last owning RGBA/guard result. Submitted intermediate work
  completes on the same queue; every temporary result is **not** read back.
- Repeat the full grid three times, serial runtime orders native/browser/Torch,
  browser/Torch/native, Torch/native/browser. No concurrent benchmark jobs.
- Setup, ray/parameter uploads, pipeline construction, reference comparisons and
  serialization are outside timing. All per-render allocation, RNG, sampling,
  NN, compositing, terminal owning copy/map and completion are inside.
- Check every timed result against its route reference. Compare WGPU references
  and final outputs against independent eager CPU f64 geometry/integration with
  f32 NN. Fixed tolerance: `4e-7 + 4e-6 * abs(reference)`.
- Timed PyTorch CPU and MPS use eager f32 tensor operations, CPU 4 intra-op/1
  inter-op threads, no autograd, no `torch.compile`, no MPS CPU fallback.
  The counter hash is recomputed on each device. Finite reductions and an owning
  CPU output/validity copy are included. Stabilize thin opacity below 0.01 using
  a fourth-order expansion; the independent f64 oracle retains `expm1`.

These are complete application-path timings, **not isolated GPU execution**.
Torch uses vectorized prefix sums and eager guard reductions, WGPU uses its
checked shaders and compensated sums. Equal inputs/readback boundaries do not
make implementation overhead or arithmetic identical. M4 desktop measurements
do not establish other hardware performance, training or universal superiority.

## Reproduce

Use a clean checkout. Retain stdout/stderr, exact commands, source hashes,
toolchain/adapter identity and all unsuccessful attempts in a new output root.
Keep raw arrays/binaries local; publish all conditions, intervals and hashes.

```sh
cargo test --locked --release -p st-backend-wgpu
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked --release -p st-backend-wgpu --lib nerf -- --test-threads=1
cargo build --locked --release -p st-backend-wgpu --example resident_nerf_bench
cargo build --locked --release -p st-backend-wgpu --target wasm32-unknown-unknown --example resident_nerf_bench_browser
wasm-bindgen target/wasm32-unknown-unknown/release/examples/resident_nerf_bench_browser.wasm --target web --out-dir /NEW/wasm --out-name spiraltorch_wasm
target/release/examples/resident_nerf_bench > /NEW/native.json
NODE_PATH=/PATH/TO/node_modules node tools/test_resident_browser.cjs /NEW/wasm /PATH/TO/CHROME /NEW/browser.json '' '' '' '' nerf-bench
python3 -I -B benchmarks/nerf-direct-graph/torch_control.py /NEW/native.json > /NEW/torch.json
python3 -I -B benchmarks/nerf-direct-graph/test_contract.py
```

The browser driver uses a temporary isolated profile and rejects fallback
adapters. Browser metadata includes a separate adapter probe, not an attestation
of the Rust runtime device. The Rust report also rejects a CPU adapter.

`analyze.py` takes three `--native`, three `--browser` and three `--torch` paths,
in round order; it rejects missing/duplicate conditions, input-byte drift,
non-finite values, guard failures, unknown timing cells and numerical mismatches.
Its paired ratios cover every shape/field/burst, including regressions and ties.

`archive.py verify RESULTS` checks compact archive fixity and timing summaries.
Optional `--raw-root RAW` checks the listed local bytes and recomputes the compact
result; `--source-root CHECKOUT` checks measured source hashes. Neither option
reexecutes the GPU or independent Torch oracle. Reexecution requires the commands
above. The publication helper expects the source-stable stage layout recorded in
`validation.json`; its command names are in `archive.py::STAGES`.
Accepted stage logs live under `RAW/accepted/`; earlier attempts remain at the
raw root, including the rejected MPS opacity calculation and metadata-gate run.
