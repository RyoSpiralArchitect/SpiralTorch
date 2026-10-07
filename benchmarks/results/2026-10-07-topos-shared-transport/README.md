# Shared Topos Gate Transport

Production source: `f52629934a408ee353aceeb3c0b8b461e93cb3b5`.
Before runtime: `0bbc5a65f0dfc23d78add8d3f61f3d3a391d7125`.
Follow-up `cfe793e6c833871b1b4c62a566bfdd3f308c6910` changes only a Rust
test lifetime scope and the browser fixture favicon, not production math.

The Rust core, Python/Torch bridge and scalar WASM now carry a feature gate of
length F instead of expanding it to N input elements. The same finite-unroll
tape returns N input gradients and F summed gate gradients. Contributions
remain f32; the row sum uses f64 and one final checked f32 conversion, matching
CPU Tensor reduction. It is not an average or a bitwise legacy-Torch-sum claim.

For N=196608 and F=768, gate input/output traffic is 256 times smaller and the
tape's f32 payload changes from 3 MiB to 2.253 MiB (24.90% less). These are
array-size calculations, not measured process peak memory. Input/output and
cotangent transfers still exist; this is not GPU-resident HF training.

## All Matched CPU Conditions

One Apple M4 host, CPU f32, two Torch threads, seed 239, coupling 0.25,
porosity 0.2, saturation 1. Preserved before/after runtimes ran in ABBA order,
reversing case order on alternate runs. Five equivalent finite-unroll routes
were position-balanced across 15 rounds after three warmups per route.
There are 30 observations per variant/route/condition, 2,700 timed calls total.
Two earlier pilot reports are retained separately and excluded from summaries.

Median forward + both VJPs + host transport, milliseconds:

| Shape | Iterations | Before public | After public | Before/after | After expanded/shared | After Torch/shared |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2x4x3 | 1 | 0.037354 | 0.036001 | 1.038 | 0.968 | 1.789 |
| 2x4x3 | 5 | 0.037437 | 0.035084 | 1.067 | 0.994 | 6.650 |
| 2x4x3 | 16 | 0.039792 | 0.040959 | 0.972 | 0.947 | 18.777 |
| 2x32x128 | 1 | 0.116792 | 0.116417 | 1.003 | 1.043 | 0.969 |
| 2x32x128 | 5 | 0.171042 | 0.168542 | 1.015 | 1.024 | 2.465 |
| 2x32x128 | 16 | 0.313520 | 0.320646 | 0.978 | 1.035 | 3.925 |
| 2x128x768 | 1 | 3.044625 | 2.628417 | 1.158 | 1.149 | 0.476 |
| 2x128x768 | 5 | 6.129188 | 5.490125 | 1.116 | 1.078 | 1.057 |
| 2x128x768 | 16 | 13.793042 | 12.835937 | 1.075 | 1.032 | 1.526 |

Ratios above one favor the shared route. Small cases include regressions;
Torch is still faster for the large one-iteration case. The Torch control is
eager independent arithmetic for this same finite recurrence, not torch.compile,
a custom fused Torch kernel, a model throughput result or general superiority.
Two runs per variant on one host do not establish confidence intervals. Small
differences can be host-load/order noise. No optimizer step is timed.

## Correctness And Limits

All legacy routes retain their output/gradient digests across builds. Public
output/input VJP digests also match. Shared gate gradients are checked bitwise
against an independent row-order f64 sum of legacy elementwise contributions;
the largest difference from the old Torch f32 sum in this matrix is
9.5367431640625e-6. Historical optimizer trajectories need not be bit-identical.
The independent Torch reference uses unchanged rtol=5e-4/atol=3e-5.

21 Rust core tests, 22 native NN regression tests, 119 isolated Python tests,
14 benchmark tests and strict core Clippy pass. Python includes finite
differences, empty/noncontiguous/broadcast/version guards, optional NumPy,
Adam checkpoint continuation and tiny config-only GPT-2/Llama connectivity.
No pretrained weights, corpora or FT runs were acquired or started.

Node and actual Chrome each pass 27 conditions, 469 checks and 240 synthetic
learning updates with exact continuation and an expanded/wide-sum control.
This WASM reference shares the Rust mathematical core, not independent math.
The old Node API also passes 24 cases and 240 updates. The first Chrome receipt
contains a console 404; adding an explicit empty favicon produced the retained
clean rerun with no page or console errors. The first receipt is not rewritten.
Initial strict Clippy found a test-only drop_non_drop warning; lexical lifetime
scoping fixed it without a lint suppression, followed by a clean strict rerun.

Independent read-only review of all 15 implementation files found no actionable
P1/P2; it did not rerun measurements. Native st-nn's CPU shared layer still
expands its gate internally, as documented; the resident GPU path is separate.
All conditions, samples, pilots and browser receipts are in measurements.json.gz.
Verification records source/native/WASM/log byte hashes; digests detect drift,
not independent provenance. No historical frozen records were edited.

## Reproduce

Use distinct runtime/output paths for each source, and the same benchmark script
for both. The docs describe isolated builds and the public shared APIs:
[Topos learning](../../../docs/topos_learning.md#unexpanded-shared-row-transport).

```sh
python tools/benchmark_topos_learning.py --shape 2 128 768 --iterations 5 --threads 2 --warmup 3 --rounds 15 --include-expanded-capture --native-profile release --output /tmp/topos-shared-new.json
node tools/probe_topos_shared_transport.mjs /tmp/shared-node /tmp/shared-node-new.json
node tools/test_resident_browser.cjs /tmp/shared-web "$CHROME_EXECUTABLE" /tmp/shared-browser-new.json "" "" "" "" topos-shared-transport
python3 -I -S -B tools/test_topos_shared_transport_results.py
```

The last command checks saved-record consistency, not fresh execution.
