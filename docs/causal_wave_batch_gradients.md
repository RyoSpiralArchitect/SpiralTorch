# Causal Wave Batch Gradients

The causal wave shared by the resident geometric byte decoder and the pure Rust
contract now retains decay/phase batch partials until the final parameter sum.
No forward recurrence, coordinate chart, geometry identity, public shape,
checkpoint schema or learning rate changes.

Previously each batch partial was narrowed to f32 before an f32 batch sum. For
opposing contributions this could round away a real small learning signal, or
reject an intermediate overflow even though all final gradients were finite.
The analytic one-step decay example `[2^24, 1, -2^24]` produced `0.25` instead
of `0.2475` in both pure Rust and WGPU before the fix.

## Implementation

- The pure contract accumulates f64 batch partials using Neumaier compensation,
  then narrows and validates the final decay/phase gradients once.
- WGPU keeps batch partials in the existing private `Wide` scratch binding,
  reduces them with extended arithmetic, and validates the final f32 stores.
  The same shader is used from native WGPU and browser WASM/WebGPU. No optional
  shader-f64 feature, CPU fallback, extra binding or extra dispatch is needed.
- Drive and initial-state gradients are still individually narrowed and
  validated. Existing input/cotangent flags and nonrepresentable final outputs
  still invalidate the entire VJP family. Only private parameter partials may
  exceed f32 before cancellation.

`Wide` is a multi-component significand with an extended exponent, not IEEE
f64 or exact arithmetic. Neither implementation promises exact cancellation
for every exponent span, batch ordering, sequence length or ill-conditioned
input. Tests establish specific analytic cases, not such a universal property.

## Resource Cost

For `B` sequences, `T` steps and `C` real coordinate channels, the extended
scratch payload grows from `16*B*T*C` to `16*(B*T*C+B*C)` bytes. The packed f32
gradient payload shrinks by `4*B*C` bytes, since partials moved out of it.
The net additional payload is **`12*B*C` bytes**, excluding allocation overhead.
Short large batches can hit a storage-binding limit earlier; preflight checks
the enlarged scratch before allocation. Tests cover that boundary explicitly.
This is a numerical-correctness change, not a measured speed optimization.

## Learning Boundary

The [recorded study](../benchmarks/results/2026-10-13-wave-batch-adjoint-v1/README.md)
contains 36 analytic cancellation cases across both decay/phase, three magnitude
scales and all six batch permutations; two recoverable-overflow cases; and
matching nonrecoverable controls. Existing wave reference/streaming/learning
checks and the fixed QKV decoder cases are rerun on native and browser.

The original projective qualification remains 4/6. In particular, the two-block
step-12 phase-gradient trajectory failure is **not resolved** by this fix.
Its changed numerical value is retained, including a slight native increase.
Previous passing cases remain passing, and paired router controls remain
separate from the old mixed-runtime gate. No corpus quality, general FT benefit,
large-combination qualification or CUDA result is claimed.
