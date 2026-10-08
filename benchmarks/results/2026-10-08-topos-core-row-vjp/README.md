# Row-wise captured Topos pullbacks

## Change and limits

The Rust semantic core selects the gate layout once and traverses shared gates
by complete rows instead of computing a remainder for every element. The f32
multiply order, per-contribution finite guards, row-ordered f64 sums, elementwise
RMS, and final shared-gradient rounding are unchanged. There is no expanded
gate-gradient buffer or new Python/WASM-specific mathematical implementation.

Baseline: `f9fc455dbf793af371e6682e4b63140288e6e1b3`.
Candidate: `17a368b0711b01b1ae1da243322502f13e419d96`.
The only changed production source is the captured VJP in
`crates/st-core/src/dynamics/topos_resonator.rs`. Tests also retain the previous
loop as an independent implementation for bit/error-order comparisons.

**Bounded result:** the captured-VJP diagnostic is about 9-11% shorter at
256x768, but the full native NN measurements are mixed. This is not evidence of
a general end-to-end training speedup, WebGPU speedup, or improved model quality.
No new pretrained-model training or corpus acquisition was performed.

## Complete measurements

Apple M4, macOS 26.4.1, Rust 1.97.0 release, Python 3.12.6, Torch 2.12.1.
The unchanged native NN/Torch timing harness uses one CPU thread, all nine
conditions, ABBA process order, forward/reverse/forward/reverse case order,
two warmups per route, and 20 measured rounds. Reported times are the median of
two process medians per arm. The two small transport pilots are excluded.
The plan and both binary identities were frozen before this run. There was no
speed acceptance threshold and no case removal.

All times below are milliseconds for **forward plus backward**. The change
column is candidate/baseline minus one, not a statistical significance estimate.

| Rows x features | K | Native before | Native after | Change | Eager Torch after-arm |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 x 3 | 1 | 0.000750 | 0.001282 | +70.9% | 0.062885 |
| 8 x 3 | 5 | 0.000896 | 0.000917 | +2.3% | 0.238010 |
| 8 x 3 | 16 | 0.001240 | 0.001219 | -1.7% | 0.725875 |
| 64 x 128 | 1 | 0.064469 | 0.063104 | -2.1% | 0.094844 |
| 64 x 128 | 5 | 0.109729 | 0.108813 | -0.8% | 0.396198 |
| 64 x 128 | 16 | 0.237604 | 0.234823 | -1.2% | 1.231104 |
| 256 x 768 | 1 | 1.572177 | 1.472511 | -6.3% | 1.665344 |
| 256 x 768 | 5 | 2.585948 | 2.739115 | +5.9% | 8.452250 |
| 256 x 768 | 16 | 5.389990 | 5.284979 | -1.9% | 27.182407 |

The tiny K=1 regression and large K=5 regression remain visible. Forward-only
timings also vary even though the forward algorithm was not changed. Do not
attribute every full-NN timing difference to the VJP loop. All 2,880 raw timing
samples, both forward routes, and all process medians are retained in
`measurements.json.gz`. Eager Torch performs the same finite recurrence,
broadcast, both VJPs, and gate-gradient accumulation. Only native Rust includes
semantic audits. Setup, transport, gradient reset, and result comparison are
outside the timing scope; neither optimizer updates nor compiled Torch are timed.

After observing the mixed full-NN results, a separate diagnostic plan was frozen
for the unchanged phase probe: the same nine conditions and ABBA order, 24
samples per scope, two warmups, all four scopes retained. At 256x768:

| K | Captured VJP before | Captured VJP after | Change |
| ---: | ---: | ---: | ---: |
| 1 | 0.427719 | 0.384510 | -10.1% |
| 5 | 0.425354 | 0.379833 | -10.7% |
| 16 | 0.424521 | 0.386771 | -8.9% |

`diagnostics.json.gz` retains all 36 process reports and 3,456 diagnostic samples.
These overlapping scopes are not additive phases and must not be subtracted
from one another. Core VJP diagnostics exclude forward and gradient accumulation;
they are not another Torch comparison. Tiny samples can approach clock resolution.

## Correctness and reproduction

- All six native vector hashes and both semantic audits match across the ABBA
  arms. All independent Torch comparisons pass the unchanged tolerances.
- Rust core: 24 tests and strict core Clippy passed. Native NN: 27 CPU and 31
  WGPU-feature tests passed, including actual WGPU execution.
- Freshly rebuilt and identity-checked Python extension: 136 tests passed with
  real-WGPU opt-in, none skipped. This includes config-only tiny language-model
  controls, not pretrained-model fine-tuning.
- Fresh Node and real isolated Chrome WASM clients: each passed 27 cases,
  469 checks, and 240 learning updates. Legacy elementwise Node: 24 cases,
  54 guard checks, and 240 updates. These are scalar WASM correctness checks,
  not browser GPU timing or independent mathematics.
- Native shared-gate learning: 200 updates, byte-identical to the earlier
  retained trajectory; independent Torch output/gradient/gate/loss checks pass.
- Read-only independent source review found no actionable P1/P2 issue. That
  review did not independently execute the runtime tests or measurements.

The public bundle contains complete result JSON, harness source, validation
records, and hashes/lengths of retained local originals. Input/output vector
blobs, native binaries, generated packages, and full logs remain local, as agreed.
Initial tool-path failures and the first Python run without GPU opt-in are
documented in `verification.json`; they were not successful executions.

Reconstruct the frozen evidence (this does not rerun a benchmark):

```sh
python -I -S -B tools/test_topos_core_row_vjp_results.py
```

For fresh measurements, build both pinned revisions into separate probe files
and replay `plan` from `measurements.json.gz`, retaining every condition and
using new output paths:

```sh
cargo build --locked --offline --release -p st-nn --example topos_shared_module_probe --example topos_shared_phase_probe
topos_shared_module_probe 256 768 5 0.25 20 /tmp/topos-row-vjp-new.json
python tools/benchmark_topos_shared_module_reference.py /tmp/topos-row-vjp-new.json /tmp/topos-row-vjp-torch-new.json
topos_shared_phase_probe 256 768 5 /tmp/topos-row-vjp-phase-new.json
```

The native 200-update check uses `topos_shared_gate_probe` and
`tools/check_topos_shared_gate_learning.py`. Rebuild Python with
`cargo build --locked --offline --release -p spiraltorch-py`; verify the loaded
extension before running the test files listed in `verification.json`.
Rebuild WASM with `--target wasm32-unknown-unknown -p spiraltorch-wasm --features
webgpu`, generate `nodejs`/`web` packages using wasm-bindgen 0.2.104, and run
`tools/probe_topos_shared_transport.mjs` and the `topos-shared-transport` fixture
of `tools/test_resident_browser.cjs`. Keep these client checks separate from
the native CPU speed measurements.
