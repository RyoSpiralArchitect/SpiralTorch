# Portable CPU Accumulator Layout, 2026-09-04

The stable, non-`simd` 8x12 and 4x16 microkernels now store accumulators as
`[column][row]`, matching the contiguous packed-A row lanes in the inner loop.
Every output retains the same sequential K reduction, multiplication/addition
semantics, packed-RHS format and destination update. There is no fast-math,
new dependency, global safe/slow fallback, or Python/JavaScript math rewrite.

The CPU autotune revision advances to 2 so an old measured kernel winner is not
blindly reused after the implementation changes. This matters in the native
measurements below: the old tuner selected `m8n12`, the new tuner selected
`m4n16`. The improvement includes that selection; it is not an isolated
same-microkernel result. Both tuning records are preserved here.

## Results

Same Furnace Linux/x86_64 host, Rust 1.98.1 / LLVM 22.1.8, release builds without
the optional Rust `simd` feature or host-specific target flags. WASM builds use
`--features webgpu --target wasm32-unknown-unknown`, but the measured WASM work
is CPU Tensor execution in Node 22.16.0, **not WebGPU or CUDA**. No `simd128`
target-feature flag was added. JS glue is generated with wasm-bindgen 0.2.104.

Seeds 17/29/43, float32 inputs, float64 reference, unchanged `rtol=1e-4` and
`atol=1e-5`, 5 warmups and 20 samples. Each runtime is measured twice in separate
processes: baseline then candidate, followed by candidate then baseline.
Medians below combine six per-seed medians, in milliseconds per call.

| Runtime / operation | Shape | Before | Candidate | PyTorch CPU, candidate run |
| --- | --- | ---: | ---: | ---: |
| Native packed `out=` | 128 cubed | 0.33259 | 0.13072 | 0.02826 |
| Native packed `out=` | 256 cubed | 4.18290 | 1.63995 | 0.21664 |
| Native Faer `out=` control | 256 cubed | 0.35971 | 0.35169 | 0.21664 |
| WASM matmul | 32 cubed | 0.01640 | 0.01626 | 0.00229 |
| WASM matmul | 64 cubed | 0.05272 | 0.05242 | 0.00705 |
| WASM matmul | 128 cubed | 0.38815 | 0.38751 | 0.04768 |

Native packed throughput improves about 2.5x on these two shapes in both
process orderings, but still loses to Torch by about 4.6x / 7.6x. WASM is
essentially unchanged, not a demonstrated WASM speedup. Gather/scatter controls
are retained in the raw results. Small WASM cases show JIT warmup effects across
seeds; do not interpret their small differences as established improvements.
No confidence interval, universal performance claim or training-quality claim.

All **288 numerical gates** pass across the eight retained runs: 72 native
checks and 216 WASM/Torch checks. Inputs are hash-identical within each runtime's
before/after comparison. Native/WASM timings have different allocation/API
boundaries and are not interchangeable: native uses prepacked RHS and `out=`,
while WASM includes the Rust autograd wrapper and allocates the result tensor.
Neither includes JS/Python input conversion in the timed region.

No Cargo build ran during the retained timings. Existing unrelated host jobs
were not stopped; CPU affinity/frequency was not pinned. The earlier WASM
attempt without a Node executable, an exploratory older WASM build, and a
baseline attempt whose completion overlapped build dispatch are excluded from
the accepted timings and retained only in the local audit directory.

## Validation And Provenance

- Rust `st-tensor --no-default-features --lib`: **426 passed**. The new test
  checks both kernels, packed/unpacked paths, nonzero destinations, tails and
  K=1025 against a sequential reference bit-for-bit.
- Strict CPU all-target Clippy and pinned rustfmt pass.
- New actual WASM 17x37x29 forward/backward test passes against independent
  dyadic references, and is wired into the existing WASM CI job. All eight Node
  regression scripts pass, including the existing learning loops.
- Full CUDA/WGPU/Golden-enabled private wheel built; all six Python output
  tests pass, including strict non-CPU WGPU reuse. No public wheel was uploaded.

Baseline: PR2064 source (`f1f6cac9ac246e5f04e0d28541180cca3f67a76d`), whose
Tensor math is unchanged from PR2063. Candidate: only the CPU layout/revision
change plus tests. The candidate CPU source hash was verified on both worktrees
as `c3cea8368fdf7ab87665e4198b938f7cd2ceea30e4a7992c5a368bab9b97669d`.
Both WASM artifacts were rebuilt on Furnace with the same compiler/features.
Native extension hashes and WASM hashes are retained in each report.

Recompute and validate all raw timing samples and numerical gates:

```bash
node benchmarks/results/2026-09-04-portable-cpu-layout/summarize.cjs
```

Native harness: `../2026-09-04-tensor-out-borrow-guard/bench_out.py`.
WASM harness: `tools/bench_wasm_vs_torch.py` with an explicit `--node` and
`--module`. Audit artifacts remain under
`~/.local/state/spiraltorch/benchmarks/portable-cpu-accumulator-layout-v1/`
on the Mac and Furnace. Node was installed only in that remote directory after
checking its archive against the official SHA-256 manifest; no system changes.
