# Explicit WASM SIMD128 build profile

This experiment compares byte-identical Rust source built with the portable
WASM target and with `-Ctarget-feature=+simd128`. It led to an explicit opt-in
profile in `scripts/build_wasm_web.sh`; portable output remains the default and
older clients can keep using it as a fallback.

## Result

Across two fresh processes with reversed operation order, 128x128 WASM matmul
decreased from 0.374 to 0.170 ms (2.20x), and prepacked matmul from 0.366 to
0.163 ms (2.25x). The corresponding PyTorch controls changed from 0.0380 to
0.0389 ms and 0.0289 to 0.0290 ms. PyTorch remains faster; this closes part of
the browser CPU-kernel gap rather than claiming parity.

At 64x64, ordinary and prepacked matmul improved by 1.81x and 2.11x. At 32x32,
the call boundary and JIT effects dominate, so no small-matrix speedup claim is
made. Gather and scatter are retained as non-matmul controls in `summary.json`.

## Measurement boundary

- Furnace Linux x86_64 host and Node 22.16.0; CPU WASM, not CUDA or WebGPU.
- PyTorch CPU uses one intra-op and one inter-op thread.
- Sizes 32/64/128, seeds 17/29/43, 5 warmups and 20 samples per case.
- Four strictly serial fresh processes use candidate/portable/portable/candidate
  order, with the operation list reversed in the second process per build.
- 288 numerical gates pass against float64 references. Reusable RHS packing is
  outside repeated-call timing and retained separately as `prepack_ms`.
- Timings are not interleaved between WASM and PyTorch. Ratios are directional
  diagnostics, not confidence intervals or a controlled library ranking.
- The candidate module declares `+simd128` in its `target_features` section;
  the portable module does not. Binary and request hashes are retained in every
  report. All ten Node validation scripts, including learning loops and the
  generated WebGPU declaration check, pass on the SIMD module.

Recompute and integrity-check all retained results:

```sh
node benchmarks/results/2026-09-04-wasm-simd128/summarize.cjs
```

The code baseline is PR #2066 head `95d62669a0f66ff93b813131bde662c312937560`.
No wheel, package version, release tag, or default browser requirement changed.

