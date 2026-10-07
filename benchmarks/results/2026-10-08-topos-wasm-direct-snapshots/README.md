# Direct JS-owned Topos snapshots

## Change and ownership

WASM's captured-output and two pullback getters now construct a JS-owned
`Float32Array` directly from the Rust slice. The previous path cloned an
intermediate Rust `Vec` before generated glue copied it into JavaScript.
This removes that intermediate allocation/copy, **not** the final snapshot copy.
It is not a zero-copy WASM-memory view. Rust mathematics and Python are unchanged.

Baseline source: `370eb3fb59483f59c940a737d94437a798d82e15`.
Candidate source: `6b3bcd5cc68bf342a1294b8ee489dad5d0c6442f`.
Both generated packages were preserved. The candidate was built from the tracked
patch before that source commit, then checked against the committed source bytes;
`candidate-build.json` records this distinction rather than claiming a clean-HEAD
build. The only changed production source is the three WASM getters.

The three public Topos TypeScript class declarations are identical. The generated
low-level WASM return ABI changes, so each generated JS wrapper must remain paired
with its matching WASM binary. Neither generated package is published here.

Both arms passed the same checks in Node and isolated real Chrome:

- 27 shared/expanded transport cases, 469 checks and 240 learning updates.
- Six ownership cases, 121 checks and 12 explicit WASM memory growth operations.
  Snapshots survive mutation of other copies, growth before/after owner destruction,
  failed VJP/retry, and empty/shared/elementwise shapes.
- The candidate additionally passed the legacy elementwise suite: 24 cases,
  54 guards and 240 updates with the captured API explicitly required.

## Complete browser comparison

Apple M4, macOS 26.4.1, Rust 1.97.0 release, wasm-bindgen 0.2.104,
Chrome 154.0.8037.98. The build enables `webgpu`, but **these operations execute
scalar WASM**, not GPU kernels. No native-NN or PyTorch speed ratio is claimed.
The prior native-NN/Torch studies have different timing boundaries.

The complete nine-condition plan was frozen before measurement. Four fresh
browser processes run ABBA with forward/reverse/forward/reverse case ordering.
There are two warmup blocks and 20 measured blocks per route, with alternating
route order. Each snapshot block contains 32 repetitions for volumes below 8192,
otherwise eight. Each learning block performs four updates. Times below are
milliseconds per operation, aggregated as the median of two process medians.
No condition was removed and no speed threshold was applied.

The snapshot scope reads prepared output, input gradient and shared-gate gradient.
The learning scope includes capture, all three getters, JS mean-MSE/upstream,
Rust VJP, JS f32-rounded shared-gate SGD, and Rust object disposal. Allocation,
clock-block overhead and JS execution are included. Setup, checks and hashes are
outside timing. The synthetic target and learning rate are fixed by the retained
harness; this is not pretrained-model fine-tuning or quality evidence.

| Rows x features | K | Snapshot before | Snapshot after | Learning before | Learning after | Learning change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 x 3 | 1 | unresolved | unresolved | unresolved | unresolved | n/a |
| 8 x 3 | 5 | unresolved | unresolved | unresolved | unresolved | n/a |
| 8 x 3 | 16 | unresolved | unresolved | unresolved | unresolved | n/a |
| 64 x 128 | 1 | unresolved | unresolved | 0.150000 | 0.125000 | -16.7% |
| 64 x 128 | 5 | unresolved | unresolved | 0.300000 | 0.300000 | 0.0% |
| 64 x 128 | 16 | unresolved | unresolved | 0.731250 | 0.725000 | -0.9% |
| 256 x 768 | 1 | 0.093750 | 0.056250 | 3.137500 | 3.062500 | -2.4% |
| 256 x 768 | 5 | 0.096875 | 0.059375 | 6.987500 | 6.956250 | -0.4% |
| 256 x 768 | 16 | 0.093750 | 0.056250 | 17.331250 | 17.300000 | -0.2% |

At 256x768, snapshot medians are about 39-40% shorter; complete-step effects are
much smaller. These are observations on this host, not significance estimates
or a broad training-throughput guarantee. Of 1,440 samples, 547 are zero at the
browser clock's resolution. A ratio is omitted whenever either arm median is
zero; this is **not a 100% speedup**. All raw timing samples and process medians
remain in `measurements.json.gz`, including the unresolved conditions.

All 3,168 update losses agree exactly between arms. Every retained block-end
output/input-gradient/gate-gradient/gate hash, initial input/target/snapshot hash,
and final gate/output hash also agrees. Hashes are recorded at four-update block
boundaries, not for every intermediate VJP. Every condition reduces its target
loss. This demonstrates preserved execution, not independent mathematical proof.

## Validation and reproduction

Read-only independent source review found no actionable P1/P2 issue, including
inspection of the locked js-sys and wasm-bindgen snapshot-constructor semantics.
That review did not execute the clients or benchmark. Python/Torch were not
rebuilt or retimed for this binding-only change. Earlier frozen artifacts remain
unchanged; full binaries, generated packages and logs remain local.

The bundle includes the complete reports, plan, harness, public class declarations
and a hash/byte-length inventory of 49 retained original files. The first derived
summary is retained locally; publication uses the zero-median-aware derivation.
Reconstruct the frozen evidence without rerunning measurements:

```sh
python -I -S -B tools/test_topos_wasm_snapshot_results.py
```

For fresh runs, build each pinned source with:

```sh
cargo build --locked --offline --release -p spiraltorch-wasm --target wasm32-unknown-unknown --features webgpu
wasm-bindgen target/wasm32-unknown-unknown/release/spiraltorch_wasm.wasm --target web --out-dir /tmp/topos-snapshot-new-web
```

Use a separate package directory per arm and fresh output names. With Playwright
available to Node, replay all four phases of the saved plan:

```sh
node tools/test_resident_browser.cjs /tmp/topos-snapshot-new-web "$CHROME_EXECUTABLE" /tmp/topos-snapshot-new.json "" "" "" "" topos-snapshot-bench "" forward
```

Use `reverse` for the reverse-order phases. The harness commit is recorded in the
plan and its exact files are included. Run the `topos-shared-transport` browser
fixture and `tools/probe_topos_shared_transport.mjs` against matching web/nodejs
packages to repeat the ownership and learning checks separately from timing.
