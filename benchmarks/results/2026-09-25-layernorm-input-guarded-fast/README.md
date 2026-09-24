# Guarded fast resident LayerNorm input VJP

This is a bounded WGPU input-VJP optimization, not a change to CPU Tensor
semantics or default backend routing. The shared WGSL kernel now uses f32 for
the final input-VJP numerator only when row scale, epsilon, and per-element
cancellation guards pass. All other elements use the previous Wide path,
including the tiny-epsilon and near-null cases. The row reductions, affine
gradient kernel, validation flags, and public API are unchanged.

Baseline source: `9e9e10a62dc7a0bd2cb410b4234dad79e8c60cf2`.
Candidate source: `c988b80db080412b18be3707fde2dd902cb479fa`.
The follow-up preflight correction in this PR changes only the declared
minimum workgroup-storage limit from 8288 to 8292 bytes; it does not change
the measured shader or dispatch schedule.
Measurements used one Apple M4 / Metal host on macOS 26.4.1, rustc 1.98.0
for WASM and the numerical control, PyTorch 2.12.1 for numerical validation,
and Chrome 153.0.8010.53 on a non-fallback Apple WebGPU adapter. All runs
were local; the Furnace GPU was not used.

## Observed result

| 128x1025, 32 training steps | Baseline | Candidate | Repeated-pair result |
| --- | ---: | ---: | --- |
| Native prepared graph, median of five run medians | 527.068 ms | 437.908 ms | 5/5 shorter; paired gains 7.06%-19.41% |
| Native diagnostic input pass, median of five | 7.625 ms | 6.252 ms | 5/5 shorter |
| Chrome matched graph, median of three run medians | 497.800 ms | 454.400 ms | 3/3 shorter; paired gains 8.72%-10.82% |

The native graph benchmark alternated baseline/candidate order across five
pairs. Each report contains three shapes, two warmups, five measured
intervals per route, and 32 steps per interval. The prepared graph excludes
initial upload/preparation but includes a guarded terminal readback. Its
standalone route is also published, but is not identical work to the
prepared-graph route. Host load made absolute native times drift noticeably,
so the table uses paired comparisons rather than a single best run.
[`native-graph.json`](native-graph.json) publishes every shape/route median,
terminal loss, and numerical check for all ten reports.

[`dispatch-profile.json`](dispatch-profile.json) contains all 30 diagnostic
shape/run results. The split profile changes one compute-pass boundary and
is **not** an unsplit throughput measurement; it only confirms that the
input dispatch, rather than affine, is where time fell. Every profile was
accepted, timing-complete, and bitwise state-equal between its original and
split diagnostic steps.

The browser fixture alternated routes within each run, used the same 32-step
graph and input/target on both modules, and compared prediction, input
gradient, parameters, parameter gradients, and loss. Three reports passed
with no page/console errors. The 32x256 browser result was mixed, so no
improvement is claimed there. [`browser.json`](browser.json) publishes both
shapes and each run's module/page hashes, adapter probe, timing, and maximum
scaled error. A separate seven-case browser fixture passed all eight VJP
masks per case, adaptive workgroup and cancellation cases, guards, and 400
learning steps; its raw report remains local.

## Numerical gates

The Rust backend's 221 real-WGPU tests passed, including new wide-row f64
input-VJP tests for ordinary, scale-direction, tiny-epsilon, large-offset,
and near-cancellation seeds. The `st-tensor` LayerNorm autograd and numerics
integration suites passed 12 and 4 tests. The existing six-shape, all-VJP
native/PyTorch protocol returned `validated`; the largest WGPU native
scaled error was 0.000091 of its allowed bound and the largest CPU/MPS
cross-check was 0.000272 of that bound.
[`pytorch-validation.json`](pytorch-validation.json) contains every shape's
validation record and hashes of its raw inputs. That protocol validates
final loss, affine gradients, and parameters; the new f64 tests and the
browser matched fixture independently cover input gradients. PyTorch
timings are **not** used to claim a framework speed advantage.

All 33 original reports, binaries, and browser modules listed in
[`raw-SHA256SUMS`](raw-SHA256SUMS) remain under
`~/Library/Logs/SpiralTorch/layernorm-input-guarded-fast-v1/`, not in the
repository. Verify them locally from that directory with
`shasum -a 256 -c /path/to/raw-SHA256SUMS`. Verify the published artifacts
from this directory with `shasum -a 256 -c SHA256SUMS`.

## Replay

Use separate clean checkouts at the two source commits. Build each native
release example, keep the executables separate, then run five sequential
baseline/candidate pairs, reversing order on even pairs:

```sh
cargo build --release -p st-backend-wgpu \
  --example resident_graph_layer_norm_bench \
  --example resident_graph_layer_norm_profile
target/release/examples/resident_graph_layer_norm_bench
target/release/examples/resident_graph_layer_norm_profile --paired
```

For the browser comparison, build `spiraltorch-wasm` in each checkout with
`--target wasm32-unknown-unknown --features webgpu`, bind each WASM file
with `wasm-bindgen 0.2.104 --target web --out-name spiraltorch_wasm`, and
run the existing matched fixture with installed Playwright and Chrome:

```sh
NODE_PATH=/path/to/playwright/node_modules node tools/test_resident_browser.cjs \
  /path/to/candidate-module /path/to/Chrome /path/to/report.json \
  '' '' '' '' nn-graph-layer-norm-matched /path/to/baseline-module
```

Run the six-shape numerical control using
`benchmarks/layer-norm-training/README.md` and its `compare.py`.
The normal and browser measurements are separate observations on one
device family, not a universal GPU speed claim. Exploratory 1/16 and 1/64
cancellation margins were weaker or noisier on this host and are not part
of the admitted result; their original probes remain local.
