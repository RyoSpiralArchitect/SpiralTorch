# Rust-owned autograd packed RHS and WASM CPU routing

This slice exposes `AutogradPackedRhs` through Rust, Python and WASM. Packing
retains the original immutable node and its backward graph, including non-leaf
sources. It is not a live optimizer cache: pack the replacement parameter after
an SGD update. Frozen RHS values can be reused across training steps.

## The benchmark found a missing dispatch

The initial API candidate was numerically correct but slower than ordinary
matmul at 128x128. Tensor's prepacked Auto path tried WGPU and Faer, then skipped
the existing CPU microkernel and fell through to naive multiplication. Both
prepacked variants now try the CPU kernel before naive, without changing the
existing WGPU/Faer priority. Python/WASM implement no duplicate math or routing.

The four raw reports retain both the negative candidate and the routing fix.
At 128x128, prepacked WASM forward calls decreased from about 0.442 to 0.358 ms
(about 19% less time). Ordinary matmul remains about 0.367 ms. This is evidence
for fixing the missed route, not a general prepacking or training speedup.
PyTorch CPU remains substantially faster, and the 32x32 measurements are highly
JIT/order-sensitive. Reversed operation order also shifts the PyTorch control;
do not interpret the unpaired runtime ratios as a controlled library ranking.

## Measurement boundary

- Furnace, RTX 5090 host, Linux x86_64; these timings run on CPU, not CUDA or WebGPU.
- Actual compiled WASM `AutogradTensor` in Node 22.16.0 versus PyTorch CPU with
  intra-op and inter-op thread counts set to one. No other task was stopped.
- Sizes 32/64/128, seeds 17/29/43, 5 warmups and 20 samples per case. Two fresh
  processes per version reverse the four-operation order.
- The Node WASM phase precedes the PyTorch phase, so runtime timings are not
  interleaved. Forward output allocation is included; backward, JS input/output
  conversion and one-time RHS packing are excluded. `prepack_ms` retains packing
  cost separately, including the first-call/JIT penalty.
- 288 numerical gates against independent float64 references passed across
  four runs, including duplicate-index gather/scatter controls.
- `summary.json` pools six per-seed medians for each size/operation/version.
  Raw reports retain all samples, first-call times, packing costs, input hashes,
  request hashes, harness hashes and WASM binary hashes. No browser-GPU or model
  quality claim is made.

Recompute and validate the retained evidence:

```sh
node benchmarks/results/2026-09-04-autograd-prepacked/summarize.cjs
```

The source baseline is `17c380039639fe79efa1ce825abb8cbe0b04b0c1` (PR #2065).
Both measured versions include the new autograd API. `cpu-route.patch` is the
only Rust source difference between the measured versions; source hashes
record the final candidate. Build outputs were isolated from installed packages.
Private Python wheels remain 0.4.27 and were not published.

## Correctness and integration

- Rust CPU: 429 library tests plus one explicit prepacked CPU receipt test.
- Rust strict CPU Clippy and repository rustfmt checks.
- Real Node WASM: nine scripts, including existing classification/indexing/
  layer-norm learning loops and the new 40-step frozen projection loop.
- The native WGPU integration test requires a real adapter and strict GPU mode.
  It checks numerical outputs and completed receipts for the prepacked forward
  and both backward matmuls, rather than treating a successful build as GPU use.

Python/WASM README examples document source lifetime, graph retention and
explicit refresh after optimizer replacement. The prepacked Auto policy can
select a different backend from ordinary matmul; it is not a residency promise.
