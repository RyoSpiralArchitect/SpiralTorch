# GPU-Resident Topos Sensitivity Capture

Source: `0bbc5a65f0dfc23d78add8d3f61f3d3a391d7125`.
Before: `6c58ef2435c7ae1c476e1517f4254be805d63875` plus the same profiling
probe, built and retained before backend edits. Later probe formatting was
non-semantic. Independent read-only review of all ten source-patch files found
no actionable P1/P2 issue; it did not independently execute these measurements.

Topos graph forward now stores its already-computed drive sensitivity; VJP
reads it rather than rerunning the finite recurrence. This is not a new model,
closed-form approximation or changed optimizer. Inference and public low-level
recomputing VJPs are unchanged. Additional storage is `4 * output_len` bytes
per Topos graph stage. Dispatch count and shared-gate reduction are unchanged.

## Matched Measurement

One Apple M4 / Metal host, scalar matmul plan, exact graph gradients, coupling
0.2, saturation 1, porosity 0.3, deterministic inputs and zero-rate SGD.
Saved release binaries were run **before, after, after, before**. Case order
was forward, reverse, forward, reverse. Every case had three warmups and nine
samples per run: 18 observations per variant/case, 324 total.

All four final states for every case are **bit-identical**, including loss,
prediction, input/raw gate/effective gate gradients and gate values. The probe
also asserts the gates did not change. Full original state-bit arrays remain
local; `profile.json.gz` retains every timing sample and per-case state digest.

Median GPU span in microseconds (before/after > 1 means shorter after):

| Shape | Iterations | Before | After | Before/After |
| --- | ---: | ---: | ---: | ---: |
| 2 x 3 | 1 | 206.020 | 183.021 | 1.126 |
| 2 x 3 | 5 | 197.021 | 204.480 | 0.964 |
| 2 x 3 | 64 | 237.729 | 201.042 | 1.182 |
| 32 x 256 | 1 | 198.520 | 206.333 | 0.962 |
| 32 x 256 | 5 | 214.980 | 202.625 | 1.061 |
| 32 x 256 | 64 | 402.292 | 209.812 | 1.917 |
| 128 x 1025 | 1 | 304.084 | 265.792 | 1.144 |
| 128 x 1025 | 5 | 498.209 | 349.854 | 1.424 |
| 128 x 1025 | 64 | 4349.063 | 2215.521 | 1.963 |

Small cases include regressions, not universal wins. GPU spans include gaps
between passes but exclude uploads, host encoding and readback. Individual
pass samples and phase totals are retained, not only the favorable VJP phase.
Instrumentation, host load, cold effects and a single-device/two-run-per-variant
sample limit inference. These are not end-to-end Python/browser timings,
confidence intervals, PyTorch speed comparisons or LLM quality evidence.

## Correctness And Boundaries

- Six focused backend Topos tests, including 1/5/64/4096 iterations, repeated
  seeds, stale tokens, owning outputs, inherited errors and valid retries.
- 48 backend training tests; 53 resident NN tests, including 300 Topos updates.
- Isolated fresh native Python: 119 tests plus 50 subtests. 100 independent
  CPU Torch updates use unchanged rtol=5e-4/atol=3e-5; maximum absolute error
  is 1.1920928955078125e-7 (gate gradient).
- Actual Chrome WebGPU: 100 updates and 4,119 checks against scalar Rust/WASM,
  plus four existing graph/client fixtures. Complete Topos receipt is in
  `browser.json.gz`; this reference shares Rust semantics, unlike CPU Torch.
- Native/WASM builds, Node-hosted WASM contract and touched-file formatting pass.
- Strict local Clippy 0.1.97 fails on an unknown newer lint annotation in seven
  byte-unchanged files. A diagnostic rerun allowing only `unknown-lints` passes;
  this is not a clean strict-Clippy claim. No suppression was added to source.

`verification.json` records source/runtime/log byte hashes and scope. Initial
interactive build metadata accidentally hashed path strings; the hashes here
are corrected file-byte hashes, cross-checked with native import and browser
receipts. Digests detect mismatches, not independent provenance or execution.
No old frozen artifacts, weights or corpora were changed or deleted.

## Reproduce

Use separate output directories, retaining the same probe for both revisions:

```sh
cargo build --locked --release -p st-backend-wgpu --example topos_graph_profile
target/release/examples/topos_graph_profile
target/release/examples/topos_graph_profile --reverse
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked -p st-backend-wgpu topos -- --test-threads=1
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked --release -p st-nn --features wgpu --lib resident:: -- --test-threads=1
```

After installing a freshly built Python extension, run the resident tests with
`SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1`. The test-file inventory and hashes are in
`verification.json`. For an isolated fresh WebGPU package:

```sh
cargo build --locked --release -p spiraltorch-wasm --target wasm32-unknown-unknown --features webgpu
wasm-bindgen target/wasm32-unknown-unknown/release/spiraltorch_wasm.wasm --target web --out-dir /tmp/topos-capture-new-web
node tools/test_resident_browser.cjs /tmp/topos-capture-new-web "$CHROME_EXECUTABLE" /tmp/topos-capture-new.json "" "" "" "" topos-resident-graph
python3 -I -S -B tools/test_topos_resident_capture_results.py
```

The final command checks saved-record consistency, not fresh GPU execution.
