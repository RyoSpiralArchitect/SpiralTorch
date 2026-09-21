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

The [NN layout follow-up](../../benchmarks/results/2026-09-21-cpu-nn-layout/README.md)
measures real `Sequential`/`Linear`/`Gelu` calls separately from layout preparation.
Native prepacked Auto uses Faer for these shapes, rather than the ordinary-layout
CPU kernel measured above. One blocked transpose now serves packing, layout
conversion and CPU transpose; small matrices retain a simple loop after a WASM
regression was measured. Column-major transpose packing no longer creates a
temporary Tensor. Faer entry points now reject malformed lengths/overflow before
mutation and construct safe slice views instead of unchecked pointer views.

The [checked GELU follow-up](../../benchmarks/results/2026-09-21-checked-gelu/README.md)
reproduces and fixes host forward/VJP layout errors, then moves checked Module
forward into `Tensor::try_gelu`. Direct aligned output construction removes one
allocation and full-buffer copy in both CPU directions. The scalar arithmetic,
finite/error policy and runtime routing remain unchanged. Strict WGPU handoff now
receives logical input/seed pairs, including column-major and Chimera storage.
The fixed GELU grid improves 1.062x forward and 1.010x VJP, but warmed MLP and Torch
comparisons remain mixed. Local Python/WASM/WGPU checks and all negative results
are published separately from the timing claims.

The [bounded GELU follow-up](../../benchmarks/results/2026-09-21-bounded-gelu/README.md)
uses the input-validation pass to prove intermediate finiteness for whole bounded
batches, retaining the original checked fallback for outliers. Exact old/new bits,
wide exponent coverage and error precedence pass on native and a test-only WASM
host adapter. Non-tiny forward improves 1.216x native and 1.090x Node/WASM, while
tiny WASM and full MLP remain mixed. Non-tiny Torch CPU remains faster overall.
Public WASM autograd/resident paths are unchanged; no browser/FT speedup is claimed.

The [owned host-forward follow-up](../../benchmarks/results/2026-09-21-owned-host-forward/README.md)
adds default-compatible `Module::forward_owned` and transfers intermediate
ownership through `Sequential`. Checked GELU reuses only unique, tracked,
row-major owned storage; shared/exported/snapshot/foreign buffers stay protected.
Depth-16 GELU chains reduce allocation requests from 48 to 3, and the real MLP
grid saves three requests in every paired condition. Non-tiny depth-16 chains
improve 1.079x native and 1.098x through a test-only Node/WASM host adapter.
Whole-MLP latency remains mixed and non-tiny Torch remains faster. Python uses
the same Rust path; backward and resident/autograd kernels remain unchanged.

The [positional-geometry follow-up](../../benchmarks/results/2026-09-21-positional-geometry/README.md)
addresses the next two old-source candidates:
`RopeKey` now uses one exact float-bit identity for equality and hashing, so
nearby-angle requests do not depend on cache history. This utility still has no
production attention caller; its phase recurrence and non-finite-key behavior
are not redesigned. NeRF encoding and field assembly/backward now honor logical
row/column/Chimera layouts, reject invalid phases and overflowing dimensions, and
write encoding results directly into aligned Tensor storage with paired sin/cos.

The actual `st-vision/nerf` WASM dependency graph now excludes native-only Faer;
a test-only Node adapter exercises the real crate rather than a path-imported
substitute. This does not add a public browser NeRF API or GPU NeRF kernel.
Parameter VJPs and one optimizer step are checked across input/seed layouts;
the documented zero gradient for raw ray inputs is preserved.

Re-enabling the dormant NeRF training regression also exposed unbounded legacy
ramp initialization. `st_nn::Linear::new_xavier(name, in, out, seed)` offers
seed-local, fan-scaled uniform weights and zero biases; `Linear::new` keeps
its old values for compatibility. `NerfField::new_with_seed(config, seed)`
uses independent Xavier layers, except its density head begins at constant
0.1 in inverse ray-parameter units to avoid an entirely inactive ReLU field.
`NerfField::new` uses seed 13. New NeRF models therefore initialize differently;
parameter names and shapes are unchanged. The four-seed, 20-step synthetic
regression measures fixed-evaluation progress, not convergence or scene quality.
The fixed multi-row/nonzero-frequency grid improves 1.031x native and 1.146x
Node/WASM, while new finite checks regress zero-band cases and the all-condition
aggregate. Torch remains faster overall at 1,024 rows; tiny Python/Rust entry
cost differences are not a general PyTorch speed claim. All conditions and
failed preflight attempts are preserved.

The [ray-integral follow-up](../../benchmarks/results/2026-09-21-nerf-ray-integral/README.md)
fixes the trainer's half-bin/minimum-width quadrature, shares stable compositing
and its VJP, and validates/snapshots logical ray datasets. Physical ray motion
no longer disappears when field direction conditioning is disabled. Direct
sample-input assembly and one-ray VJP scratch reduce intermediates, but do not
establish a measured peak-memory reduction. Native/Node-WASM and an independent
PyTorch/autograd control agree across all 324 measured records. The largest
condition still favors PyTorch by about 2x versus native and 4.6x versus WASM;
small-input entry-cost wins are not a universal backend-speed claim.

1. CPU NN throughput: separate warmed forward, cache refresh and complete optimizer
   steps. The fixed harness now covers the first two, not training throughput.
   Profile the remaining checked GELU/transcendental cost and explicit
   ordinary-versus-prepacked Faer execution before changing Auto policy or the
   packed representation. The earlier 18.41x/11.72x
   PyTorch gap applies to a different, explicit ordinary-kernel grid, not to all
   high-level NN calls; neither Node timing nor packing wins prove browser/FT gains.
2. RoPE: design finite-angle/size admission and actual attention ownership before
   introducing public bindings or claiming attention performance improvements.
3. NeRF: the Rust trainer's sampling/compositing contract is now tested; its
   separate WGSL sampler still contains the old half-bin expression and needs
   independent execution/parity tests. Profile the larger-render CPU path.
   Larger scenes, camera-input derivatives, resident training and public browser
   ownership remain separate work. Training checks are small synthetic regressions.
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
