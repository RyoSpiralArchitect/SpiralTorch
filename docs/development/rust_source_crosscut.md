# Rust Source Crosscut: First Pass

## Scope

The inventory at `7f64e24a8bcccc1464faea85c1eae878e51041ca` covers 35 workspace
packages and 693 tracked Rust/shader files under their `src` directories. Age is
the last reachable Git change, not filesystem modification time after restoration.
Age and textual allocation hints are triage signals, not correctness findings.
The inventory is exhaustive within that file boundary; the manual code review is
not an exhaustive review of all 693 files.

## Implemented

- `spiral-selfsup::contrastive`: the Tensor-returning entry point used an independent
  scalar matrix product and interpreted column-major storage as row-major. All
  three entry points now share dense dispatch, normalization, and objective
  evaluation. Tensor inputs are borrowed/shared in compatible layouts; the
  vector-result adapter no longer builds one Vec per example. Empty dimensions
  return an error before chunking. This preserves the vector API's optional WGPU
  attempt and CPU fallback policy, not a new strict GPU execution contract.
- `st-frac::fractal_field`: reuse the generated branch buffer when weaving a grid,
  removing a full complex sample copy/allocation without changing addition order.
  Exact float32 output bits and base immutability are checked. The existing Python
  `frac.FractalFieldGenerator.weave_with_grid` directly uses this implementation.
- `st-logic::temporal_dynamics` and `st-vision::ZSpaceVolume`: absent harmonic
  channels were read from adjacent voxels, and flat resizing changed row identity.
  Zero-pad missing channels explicitly and remap rows in place (backwards on
  expansion, forwards on contraction). Preserve exact requested channel counts;
  do not silently change integration policy to retain additional channels.

Five failures were reproduced before the fixes: column-major InfoNCE, empty
Tensor adapter panic, harmonic resize, interpolation, and integration. The real
ZSpaceVolume implementation, not only a mock, is exercised. Independent float64
objective checks and 108 small channel-resize combinations supplement wrapper
agreement tests. Fixed-width test slices were modernized to satisfy scoped strict
Clippy; no lint allowances were added.

## Boundaries

The standalone InfoNCE APIs do not expose autograd or a resident GPU objective.
This does not merge their implementation with `st-nn`'s differentiable trainer
loss. Python's existing self-supervision binding calls the vector API; this patch
does not change Python feature forwarding or add a Tensor binding.

WASM compatibility is checked separately. Fractal weaving and the temporal
functions are not newly exposed to JavaScript in this patch. A successful WASM
build or the existing numerical smoke suite is not a browser performance claim.

An initial strict Clippy invocation including dependencies encountered 22 existing
`st-nn` lints. The changed packages pass the repository's `--no-deps` strict
inspection boundary. The failed attempt is retained, rather than relabeled a
successful all-workspace lint run.

## Next Candidates

The [CPU workspace follow-up](../../benchmarks/results/2026-09-21-cpu-dense-workspace/README.md)
implements fixed-width direct products, serial panel-buffer reuse, checked sizes,
and direct Tensor label construction. It retains rejected experiments and shows
that allocation savings are not equivalent to large-matrix speedups.

The [panel-reuse follow-up](../../benchmarks/results/2026-09-21-cpu-panel-reuse/README.md)
now reuses packed A rows across bounded column groups and the full prepacked RHS.
On the fixed wider CPU grid, prepacked median-time ratios geometrically average
1.618x serial and 2.141x with four threads, relative to the workspace follow-up.
Unpacked results remain mixed, can require more RHS scratch, and retain repeated
work when the inner dimension permits only one panel per group. All negative
results and an excluded shared-build-cache attempt are preserved. The comparison
now checks source/build identity before trusting timing or numerical success.
PyTorch is still faster across this grid; Auto routing is not changed.

The [row-major follow-up](../../benchmarks/results/2026-09-21-cpu-row-major/README.md)
removes unpacked A/B packing entirely, uses fixed-width tail accumulators, and
avoids parallel launch for a lone partial row group. The extended native grid
improves 2.173x/3.458x serial/four-thread relative to the panel-reuse implementation;
the real WASM/Node ordinary-layout grid improves 1.166x. Packed controls and tiny
cases remain mixed. Layout-specific tuning now measures and caches the actual
execution path; the public prepacked format and high-level routing stay intact.

1. CPU dense throughput and NN dispatch: profile actual Linear/MLP workloads,
   prepacked versus ordinary weights, and SIMD/microkernel throughput. The native
   ordinary-layout PyTorch gap is still about 18.41x/11.72x on the extended grid;
   neither zero packing allocations nor Node timing proves browser or FT gains.
2. `st-core/src/util/rope_lru.rs`: epsilon-based equality and bitwise hashing do
   not define the same key identity; equality is also non-transitive. Review
   exact angle identity, finite values, eviction, and real consumers before
   optimizing trigonometry or broadening this currently lightly connected cache.
3. `st-vision/src/nerf/encoding.rs`: benchmark paired sine/cosine evaluation and
   validate output-size arithmetic; do not assume an intrinsic is faster on
   native and WASM targets without measurements.
4. Contrastive backend policy: profile small-batch WGPU transfer cost separately
   from CPU kernels, and design explicit execution ownership before introducing
   a resident loss or sharing the trainer's gradient implementation.
5. Temporal volume contracts: validate buffer lengths, dimension arithmetic,
   finite controls, and failure atomicity before exposing the full propagation
   API to Python/WASM clients. This patch fixes row identity, not every malformed
   third-party implementation of the public volume trait.

`docs/backend_matrix.md` also needs a source-backed capability refresh: some
planned WGPU/fusion labels predate the explicit resident graph path. Update the
generator and its contracts together rather than editing a generated table alone.
