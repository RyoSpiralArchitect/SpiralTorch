# Resident NeRF: explicit single submission

`render_graph` keeps the existing three-submission direct NN connection.
`render_graph_single_submission` records sampling, position packing, NN and
compositing in one submission. Both retain owning outputs and numerical guards.
Only the latter discards the entire recorded render on a typed host failure;
this is not rollback after device loss or deferred numerical rejection.

Fewer submissions are **not** an automatic performance policy. On the shared
M4 desktop, an exploratory three-round comparison favored the single route
slightly on native Metal but not browser WebGPU. Keep both results, regressions
and ties; do not select a default using submission counts alone.

## Matched Comparison

Reuse the [direct-graph grid, numerical oracle and timing
boundary](../nerf-direct-graph/README.md). The controls here are **separate direct
submissions versus one composed submission**, not stable-workspace copies versus
direct execution. Both use the same current graph implementation and shaders.
This is not a pre-change/post-change binary comparison.

Twelve ray/sample/field combinations, bursts 1/4, three warmups and nine measured
paired blocks, repeated for three rounds. Rotate runtime order
native/browser/Torch, browser/Torch/native, Torch/native/browser. Reuse the
first native report only as the fixed Torch input fixture; every route/round
must match its emitted input bytes. Keep GPU jobs serial.

The complete application path includes allocations, encoding, submissions,
terminal owning RGBA/guard copy, map and completion; setup and validation are
outside timing. Burst 4 observes only its final output. Eager Torch CPU/MPS is
the shared independent control (f32 timed, f64 integration/f32 NN oracle);
the fixed tolerance is `4e-7 + 4e-6 * abs(reference)`. This is not GPU timestamp
profiling, training, scene quality or a general claim of superiority to PyTorch.

The old benchmark mode is unchanged. To select this comparison:

```sh
cargo build --locked --release -p st-backend-wgpu --example resident_nerf_bench
target/release/examples/resident_nerf_bench --compare-submissions > /NEW/native.json
cargo build --locked --release -p st-backend-wgpu --target wasm32-unknown-unknown --example resident_nerf_bench_browser
wasm-bindgen target/wasm32-unknown-unknown/release/examples/resident_nerf_bench_browser.wasm --target web --out-dir /NEW/wasm --out-name spiraltorch_wasm
NODE_PATH=/PATH/TO/node_modules node tools/test_resident_browser.cjs /NEW/wasm /PATH/TO/CHROME /NEW/browser.json '' '' '' '' nerf-submit-bench
python3 -I -B benchmarks/nerf-single-submit/protocol.py torch /NEW/native.json > /NEW/torch.json
python3 -I -B benchmarks/nerf-single-submit/test_protocol.py
python3 -I -B benchmarks/nerf-single-submit/test_archive.py
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked --release -p st-backend-wgpu --lib -- --test-threads=1
```

Repeat into new round-specific paths. `protocol.py analyze` requires three of
each `--native`, `--browser` and `--torch` path in round order. It reuses the
strict admission/statistics engine with temporary route-label normalization;
raw and public reports always retain separate/single identities.

## Evidence Boundary

Keep raw arrays, binaries, source-stable receipts and unsuccessful attempts
locally. Publish complete intervals, errors, exact commands, source hashes,
adapter metadata and replay instructions. The browser's separate adapter probe
does not attest the Rust runtime's device identity.

`archive.py publish RESULTS --raw-root RAW` expects clean-source accepted
stages under `RAW/accepted` (names in `STAGES`), plus earlier attempts at the
raw root. `archive.py verify RESULTS` checks fixity, grid and aggregation.
Adding `--raw-root RAW` checks raw bytes and recomputes the summaries;
`--source-root CHECKOUT` checks the measured source hashes. Neither reexecutes
the numerical oracle or GPU. Use the frozen source commit for replay, not a
later checkout. Never overwrite earlier evidence or silently drop a bad run.
