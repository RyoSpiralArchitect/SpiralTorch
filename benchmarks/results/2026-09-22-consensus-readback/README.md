# Ordered Softmax/Consensus Observation, 2026-09-22

Measured implementation: `e1727e1a571c27de0a25a7226aac27fddd7c57f1`.
Base: `826887e6e5eb93e0b7483087a0023d8cb1bd03fe`.

**The repeatable improvement here is batching terminal readback, not a faster
consensus reduction in isolation.** The usual Tensor pair path now collects its
two results together. The separate dense GPU consensus helper collects four.
The ordinary Tensor spiral API retains its CPU consensus/telemetry blend.

## Results

Three runtime-order-rotated serial rounds, six shapes, pair/raw-consensus output contracts,
bursts 1/4, nine measured blocks after three warm-ups: **6,480 intervals**.
All final cells use balanced-cycle-v1: each route occupies every execution
position two or three times in nine blocks.
Native Metal and browser WebGPU each cross separate/paired reductions with
separate/batched readback. Torch 2.12.1 eager CPU/MPS is a matched-math control.

| Terminal output contract | Native combined ratio | Browser combined ratio |
| --- | ---: | ---: |
| Probabilities + all-peak mask | 1.510x | 1.585x |
| Pair + spiral weights + row metrics | 2.113x | 2.434x |

Ratios are geometric means of the twelve cell-level median paired
separate/separate-over-paired/batched ratios, not ratios of global medians.
All twelve combined cells per contract/runtime exceed 1 here. The two
readback-only factors tell the same story. In contrast, paired reductions with
already-batched observation yield 0.999x native and 0.997x browser for four
outputs. This does **not** establish a reduction-only speedup; slower/noisy
cells remain in [results.json](results.json).

The prepared interval includes encoding/binding, per-operation submission and
terminal owning CPU completion. The separate control uses the same new Rust
lease/decoder one buffer at a time; it is not a byte-identical benchmark of the
old native read_buffer helper. Pair mode never executes consensus and acts as
a reduction placebo. No whole-Tensor-API speedup multiplier is claimed.

For perspective, at 128x1025, pair/burst=1, median intervals are 0.586 ms for
native paired/batched, 0.900 ms for browser paired/batched, 0.204 ms for Torch
CPU and 1.063 ms for Torch MPS. CPU is still preferable in some cells. These
are shared-desktop application intervals, not GPU timestamps or a universal
SpiralTorch-over-PyTorch result.

## Correctness And Validation

- All four WGPU routes have bitwise-identical final outputs. Twelve domain
  cases additionally cover widths 1/3/31/256/257/1025, finite extremes,
  all-tied peaks, padding and Chimera addressing.
- Independent scalar f64 and Torch CPU f64 oracles pass the frozen
  abs=2e-6, rel=5e-6 gate. Maximum absolute error is 2.86102294921875e-6;
  maximum scaled error is 0.07478335356, below the limit of 1.
- Twenty-three clean-source stages pass: format, admission/archive tests,
  old-archive compatibility, backend/Tensor native/WASM strict lint,
  real-GPU backend tests (189 library + 1 ownership + 1 consensus fixture),
  two strict Tensor regressions, builds and all nine runtime rounds.
- Ownership checks cover ordered prefixes, raw bits, empty prefixes, held
  snapshots across writes/source drop, unread drop, validation errors, and
  pending-map cancellation in the actual browser. Native GPU tests exercise
  multiple staging chunks under a deliberately small simulated limit.

The required GPU flag is enabled in the recorded native tests. The Tensor test
also checks that the raw helper truly reports GPU consensus, and that the
ordinary API's additional CPU telemetry blend stays unchanged.

## Preserved Evidence

[exploration.json](exploration.json) contains a separate, complete 6,480-interval
legacy-order screening and every recorded pre-publication top-level attempt. Its complete screening
uses the explicit screen2 prefix; no prior files were overwritten.
[early-attempts.json](early-attempts.json) retains the incomplete initial native
run's 864 intervals and three browser crashes. Progress checkpoints in the
third attempt reached the final condition after ownership/domain checks passed.
Bounded, per-case JSONL export then allowed all screening and final browser
runs to complete. The renderer crash mechanism itself is not established.

The first committed 6,480-interval run at 9b065669 is also retained in
[legacy-order-results.json](legacy-order-results.json), with its separate
[source](legacy-order-source.json) and [validation](legacy-order-validation.json).
Self-review found that alternating reversal coupled to rotation left only
routes 0/2 first. The final implementation reverses only after complete
four-block cycles and has a regression test for every position. Earlier
readback comparisons remain diagnostic; no legacy-order intervals are pooled
with the final balanced study. The validator rejects mixed ordering protocols.

The public archive contains all conditions, timing intervals, validations,
source identity and hashes. Raw arrays, executables, generated WASM/JS and
streams remain local: the snapshot manifest lists **1,133 files / 4,347,540,352
bytes**. All three final browser streams have twelve complete cases and hashes.
Native runtime identifies Apple M4/Metal; browser runtime identifies
BrowserWebGpu, with a separate non-fallback adapter probe. The probe does not
attest the Rust runtime's device identity. Chrome: 153.0.8010.53.

Replay boundaries and commands are in
[the protocol guide](../../consensus-readback/README.md). Verify public fixity:

```sh
python3 -I -B benchmarks/consensus-readback/evidence.py verify benchmarks/results/2026-09-22-consensus-readback
```

Use --raw-root and --source-root to additionally re-hash local raw artifacts,
recompute summaries and compare measured source files. This is not numerical
GPU reexecution. Chunk packing respects per-buffer limits, not a global memory
budget; batched snapshots may retain more staging memory than sequential reads.
No new resident graph softmax, autograd/training migration, model-quality claim,
CUDA result or universal performance guarantee is included.
