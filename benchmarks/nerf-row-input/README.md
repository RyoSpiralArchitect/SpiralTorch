# Resident graph: read N-D rows without packing

`ResidentGraph::forward_tensor` reads eligible first-linear inputs directly:
`offset + row * row_stride + col`. `NdLayout::row_major_rows()` describes the
logical last-axis rows; leading axes must flatten affinely and columns must
have unit stride (singleton strides are unobservable). Row broadcasts are
read-only and valid. This is not arbitrary strided matmul or mutable aliasing.
Irregular prefixes, strided columns and pointwise-first graphs still pack on
the GPU. No host readback or CPU fallback is introduced.

NeRF positions are a `[R,N,3]` view of `[R,N,4]` xyz/t storage. The first Linear
stage can now read its xyz rows without a temporary packed-value allocation
and packing dispatch. Native Rust, Python and WASM graph forwards reach the
same backend. Host `pure::Tensor`, graph training and automatic autograd are
unchanged. The inference-only 40-byte uniform leaves the canonical 32-byte
host/training ABI and validation mask untouched. Held tensors, old bindings
and inherited guards remain immutable across view changes, reuse and drop.

## Controlled Comparison

`forward_tensor_packed` is the explicit Rust reference path, not a second
Python/JavaScript control. It forces a pack only when needed and keeps it
inside the graph submission. The measured packed and rows routes both sample,
evaluate the same NN and composite using **three render submissions** plus the
same owning terminal RGBA/guard observation. This is neither the earlier
submission-count comparison nor a before/after-binary measurement.

Reuse the [direct-graph grid and numerical oracle](../nerf-direct-graph/README.md):
12 ray/sample/field combinations, bursts 1/4, 3 warmups, 9 paired blocks,
3 serial rounds. Rotate native/browser/Torch, browser/Torch/native,
Torch/native/browser. All input bytes and parameters must agree. The 1x1
cases are already contiguous and intentionally save no packing; retain them
as controls rather than attributing their timing noise to this optimization.

Timing includes per-render allocations, encoding, submissions, and the final
owning copy/map/completion; compilation, setup, correctness checks and JSON
serialization are excluded. Burst4 observes only its last output. Eager
PyTorch CPU/MPS remains an application-path comparator and independent
numerical control, not matched WGSL or `torch.compile`. The oracle uses f64
geometry/integration with f32 NN. Tolerance stays `4e-7 + 4e-6 * abs(reference)`.
Preserve all conditions and regressions. Shared-desktop and coarse browser
timings do not establish universal speed, training, or scene quality.

```sh
cargo build --locked --release -p st-backend-wgpu --example resident_nerf_bench
target/release/examples/resident_nerf_bench --compare-input-layouts > /NEW/native.json
cargo build --locked --release -p st-backend-wgpu --target wasm32-unknown-unknown --example resident_nerf_bench_browser
wasm-bindgen target/wasm32-unknown-unknown/release/examples/resident_nerf_bench_browser.wasm --target web --out-dir /NEW/wasm --out-name spiraltorch_wasm
NODE_PATH=/PATH/TO/node_modules node tools/test_resident_browser.cjs /NEW/wasm /PATH/TO/CHROME /NEW/browser.json '' '' '' '' nerf-row-input-bench
python3 -I -B benchmarks/nerf-row-input/compare.py torch /NEW/native.json > /NEW/torch.json
python3 -I -B benchmarks/nerf-row-input/test_compare.py
python3 -I -B benchmarks/nerf-row-input/test_evidence.py
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked --release -p st-backend-wgpu --lib -- --test-threads=1
```

Repeat into new round paths, reusing the first native report as the fixed
Torch input fixture. `compare.py analyze` requires three of each `--native`,
`--browser` and `--torch` in round order. It shares strict admission/statistics
with the existing benchmarks, adapting route labels only temporarily. Raw
and published reports keep distinct packed/rows identities.

## Publication And Replay

Keep raw arrays, native/WASM binaries, source-stable receipts and unsuccessful
attempts locally. Publish all intervals, errors, exact commands, input and
source hashes, adapter metadata and rerun instructions. The browser's separate
nonfallback probe does not attest the Rust runtime's device.

`evidence.py publish RESULTS --raw-root RAW` uses the shared append-only archive
implementation with this protocol's stage list and validators. It requires
committed clean-source stages under `RAW/accepted`, with exploratory attempts
and three screening rounds at the raw root. `evidence.py verify RESULTS`
checks manifest fixity, grid, receipts and recomputed aggregation; add
`--raw-root RAW` for raw hashes and summaries and `--source-root CHECKOUT` for
measured source hashes. This does not rerun the oracle or GPU. Use the frozen
implementation commit for replay, never silently replace historical evidence.
