# Resident graph: pointwise-first N-D views

`ResidentGraph::forward_tensor` reads any admitted input view directly when
the graph starts with a pointwise stage. The existing shader already supports
offsets, permutations, strided columns, broadcasts and residual input slots.
Retargeting uploads immutable addressing metadata, shares the original pipeline
and binding layout, and caches one exact input layout. A binding retains its
metadata plan even if a later aborted composition warms another cached layout.

No new shader family or CPU fallback is added. First-linear row addressing and
its packing fallback remain unchanged. Rust, Python and WASM direct graph
forwards use this backend; host `pure::Tensor`, training and autograd do not
automatically migrate. Held outputs and inherited guard snapshots remain owned.

## Matched Comparison

Both routes sample rays, run **ReLU(input) -> Linear [-> ReLU -> Linear]**,
and composite with three submissions and the same terminal owning RGBA/guard
observation. Rust's `forward_tensor_packed` is the packing control within the
same current binary and submission schedule, not a second Python/WASM switch.
The added input ReLU is explicit in every report/case and also executed by the
independent Torch oracle and timed CPU/MPS controls. Old no-prelude reports
cannot be mixed in. This is not a before/after-binary comparison.

Reuse the [existing grid](../nerf-direct-graph/README.md): 12 ray/sample/field
conditions, bursts 1/4, 3 warmups, 9 paired blocks, three serial runtime rounds.
Rotate native/browser/Torch, browser/Torch/native, Torch/native/browser. Retain
all conditions, including the already-contiguous 1x1 inputs which save no pack
and serve as timing-noise controls. Browser/native setup also checks six view
layouts, broadcast parameters/residual inputs, stable dispatch transitions,
retained outputs and masked inherited errors outside every timed interval.

Timing includes per-render allocation, encoding, submissions and terminal
copy/map/completion; excludes setup, checks and serialization. Burst4 observes
only its last output. Torch is eager CPU/MPS f32 with stable thin alpha; the
oracle uses CPU f64 geometry/integration and f32 NN, with the same input ReLU.
Tolerance remains `4e-7 + 4e-6 * abs(reference)`. These are application-path
timings on a shared M4 desktop, with a coarse browser clock: not kernel-only,
`torch.compile`, training, scene quality, or universal PyTorch superiority.

```sh
cargo build --locked --release -p st-backend-wgpu --example resident_nerf_bench
target/release/examples/resident_nerf_bench --compare-pointwise-inputs > /NEW/native.json
cargo build --locked --release -p st-backend-wgpu --target wasm32-unknown-unknown --example resident_nerf_bench_browser
wasm-bindgen target/wasm32-unknown-unknown/release/examples/resident_nerf_bench_browser.wasm --target web --out-dir /NEW/wasm --out-name spiraltorch_wasm
NODE_PATH=/PATH/TO/node_modules node tools/test_resident_browser.cjs /NEW/wasm /PATH/TO/CHROME /NEW/browser.json '' '' '' '' nerf-pointwise-input-bench
python3 -I -B benchmarks/nerf-pointwise-input/compare_pointwise.py torch /NEW/native.json > /NEW/torch.json
python3 -I -B benchmarks/nerf-pointwise-input/test_compare_pointwise.py
python3 -I -B benchmarks/nerf-pointwise-input/test_evidence_pointwise.py
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked --release -p st-backend-wgpu --lib -- --test-threads=1
```

Use new paths for each round and one fixed native source for Torch.
`compare_pointwise.py analyze` requires three each of `--native`, `--browser`,
`--torch`, in round order. Shared admission/statistics adapt labels only
temporarily; raw reports and public summaries keep the packed/views identity.

## Evidence And Replay

Keep all raw arrays, binaries and failed attempts locally. Publish all intervals,
errors, source/input hashes, exact commands and adapter metadata. The browser's
separate nonfallback probe is not attestation of the Rust runtime's device.
`evidence_pointwise.py publish RESULTS --raw-root RAW` shares the append-only
archiver with a distinct protocol and stage list. It requires clean committed
source receipts under `RAW/accepted` and three screening rounds at the raw root.

`evidence_pointwise.py verify RESULTS` checks fixity, receipts, complete grids
and recomputed aggregation; add `--raw-root RAW` for local hashes/summaries,
`--source-root CHECKOUT` for measured source hashes. It does not rerun GPU or
numerical calculations. Replay from the frozen source commit, not current HEAD.
