# Exact Support And Tiled History Input VJP

Rust skips exact-zero coefficient endpoints while retaining nonzero order
derivatives, and computes history input adjoints in contiguous feature tiles.
Full-K normalization, recurrence, validation, budgets, accumulation order and
first-order operator semantics are unchanged. See the
[implementation and reproduction protocol](../../../docs/fractional_active_support.md).

## Native Measurement

Same [2,128,768] CPU f32 input, K=32, amplitude exp(log(5)), seed 239. Each row
uses four fresh processes in before/after/after/before order; each process has
two warmups and 12 alternating Rust/Torch rounds. Values below are pooled
24-sample medians per build. Rust's GL host kernel is serial; Torch intra-op
threads are two. Forward, AD and Python/native buffer transport are included.

| Order / window | Requested VJP | Old Rust ms | Final Rust ms | Old/final | Torch ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.09 / [1,32) | scalars + input | 7.658 | 3.988 | 1.92x | 3.623 |
| 0.09 / [1,32) | scalars only | 2.826 | 2.806 | 1.01x | 2.760 |
| 0.55 / [1,3) | scalars + input | 7.692 | 2.396 | 3.21x | 12.474 |
| 0.55 / [1,3) | scalars only | 2.697 | 1.849 | 1.46x | 10.781 |
| 2 / [1,3) | scalars + input | 7.768 | 2.510 | 3.09x | 12.731 |
| 2 / [1,3) | scalars only | 2.814 | 1.810 | 1.55x | 10.877 |
| 2 / [3,32) | scalars + input | 7.160 | 3.376 | 2.12x | 3.653 |
| 2 / [3,32) | scalars only | 2.829 | 2.684 | 1.05x | 2.669 |

Torch normalizes the complete polynomial before selecting taps, then uses a
support-length conv1d with the correct lag offset. It is not forced to convolve
zero tails. This is one concrete Torch implementation, not the fastest possible
Torch formulation. Full-history input-gradient work remains slower than this
Torch reference. Do not generalize short-window ratios to library superiority.

Every before/after output and requested-gradient hash is identical within each
matched case in all 96 native processes across all three variants. Repeats are bitwise
checked; Rust/Torch comparison uses the unchanged rtol=atol=3e-5 gate. Equal
hashes here are receipts from the recorded executions, not new model training.

The first indexed-support version regressed full-history joint work from
7.334 to 7.959 ms; slice iteration still measured 7.788 to 8.129 ms. Both
unsuccessful variants and every measurement are retained. Hoisting the lag
origin generated a byte-identical native binary to the sliced version, so it
is not claimed as a further optimization or given another native timing set.
The final improvement comes from the tiled input adjoint, not erased failures.

## WASM And Correctness

Final release WASM in Node passes 60 before/after bitwise cases spanning axes,
tile edges, LM shape, empty windows and integer-order zero tails with nonzero
order derivatives. Forward, joint/selective parameter/input VJPs and JVP are
compared separately against the corresponding old API. Old joint/selective
scalar reductions already differ in zero sign for some zero maps; that old
cross-API difference is preserved rather than canonicalized away.

WASM timing covers forward plus parameter VJP, output copying and snapshot
free, not input VJP. The final short-window median is 5.983 to 3.699 ms at
order 0.55. Full history is essentially flat, 6.050 to 5.916 ms. This is Node
WASM host execution, not browser/WebGPU performance. All four WASM measurement
sets are included, including the source-only hoisted variant.

Both 100-update synthetic WASM window-learning loops still reduce MSE and
update angle/gain. The 151 default-feature st-frac tests, Clippy with denied warnings, 848 geometry
Python regressions and 84 benchmark regressions pass. The publication's
stdlib-only tests recompute pooled medians, preserve initial regressions and
bind runtime hashes; they do not independently execute private native modules.

## Evidence And Limits

- `native-*.json.gz` stores all 96 original numeric reports, including native/bridge/script hashes, requested gradients, input hashes, route order, correctness errors and individual timings.
- `summary.json` contains derived pooled medians; the publication tests recompute them without model imports.
- `build-identities.json` pins all source/native identities; `runtime-sha256.json` pins the final 71-file package. Only the native file differs from the preserved original package.
- `wasm-*.json` separates compiled-WASM comparison receipts from synthetic learning. `validation.json` records checks, exits and private log hashes. `SHA256SUMS` covers every public file.

No pretrained run was restarted, endpoint rescored, weight changed, duration
selected, or old study overwritten. Frozen-base adapter-only FT does not request
input gradients; the input-adjoint gains matter when composing trainable earlier
layers/adapters. These operator measurements therefore do not establish an
end-to-end training speedup or any new quality effect. Thermal/load drift and
the reused local machine limit inference; no GPU, CUDA or WGPU speed is claimed.

For reproduction, use the exact source/native identities and commands in the
protocol, with a NEW output file and the same shapes, windows, gradient requests,
threads and build profiles. Keep original results and unsuccessful attempts.
