# Resident Attention Projection Kernels

This study changes the QKV/output projection implementation, not the attention
function or the Z-RBF geometry. The existing Register2x2 backend is now reachable
through `AttentionInferencePlan::compile_wgpu_with_options` on native WGPU and
browser WASM. The ordinary compile method remains Scalar: these measurements
are not a portable device/shape autotuning policy.

**Register2x2/16x16 improved the measured larger shapes, not enough to beat
PyTorch MPS.** Across the 12 larger-shape conditions, final pooled-median speed
ratios versus Scalar are 1.18-1.48 resident and 1.04-1.34 host-to-host. Short
results vary between runs, so no blanket default change was made.

## Controlled Change

The same release executable runs three explicit projection presets:

| Preset | Matmul kernel | Output tile / inner tile |
| --- | --- | --- |
| `scalar` | Scalar | 8x8 / 16 |
| `register8` | Register2x2 | 8x8 / 16 |
| `register16` | Register2x2 | 16x16 / 16 |

All use the existing default accumulation. Both projection graphs use the
selected preset. Parameters, biases, geometry, structural masks, attention
key-tiling policy, owning outputs and runtime non-finite guards are unchanged.
No attention math was reimplemented in Python or JavaScript. Invalid register
tiles are rejected rather than silently falling back to Scalar.

The complete chain is frozen QKV projection -> head views/packing -> attention
-> GPU head merge -> output projection. Inputs are `[1,32,64]`, `[2,128,128]`
and `[1,256,256]`, with 4/4/8 heads. Each shape has unmasked/causal versions of
plain, zero-bias and nonzero Z-RBF attention, giving 18 conditions per engine.
Torch receives the same bias: biased ST is not compared with plain Torch math.

Five engines (the three ST presets, Torch CPU and Torch MPS) rotate through five
order positions. Each engine/case/boundary has 50 warmup blocks and nine retained
blocks per round, giving 45 retained samples. Resident blocks contain four
forwards plus completion; host-to-host blocks upload fresh input/bias and return
an owning host output, with weights resident. Numerical checks run after the
timer and check every output. These are full-chain host-observed latencies, not
GPU kernel timestamps or projection-only microbenchmarks.

## Results

Illustrative **causal + Z-RBF** pooled medians from the final run, milliseconds
per forward. The full plain/zero/nonzero matrix and every sample are retained.

| Shape | ST Scalar resident | ST Register16 resident | Torch CPU resident | Torch MPS resident | ST Scalar H2H | ST Register16 H2H |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `[1,32,64]` | 0.532 | 0.534 | 0.026 | 0.167 | 0.851 | 0.703 |
| `[2,128,128]` | 0.905 | 0.719 | 0.273 | 0.194 | 1.298 | 1.193 |
| `[1,256,256]` | 3.089 | 2.089 | 0.863 | 0.365 | 3.881 | 2.944 |

All 120 larger-shape within-round paired medians favor Register16 (12 cases,
five rounds, two boundaries). These correlated observations are descriptive,
not 120 independent trials or a significance test.

The initial wiring probe had the same direction for the larger cases: its
causal Z-RBF resident times were 0.901 -> 0.702 ms and 3.079 -> 2.085 ms.
Its short causal Z-RBF result was instead 0.451 -> 0.491 ms, with a host-to-host
speed ratio of 0.745 (slower). The final run's short result differs; neither
observation is discarded or treated as proof of a clock/driver cause. The short
case does not establish a stable speed advantage or a no-regression guarantee.

Register8 is not a substitute for the larger tile: final causal Z-RBF resident
times were 0.443, 0.931 and 3.143 ms for the three shapes. Exposing an existing
"fast" primitive is not sufficient without choosing and measuring its geometry.
The measured larger-input benefit is available explicitly; portable automatic
selection remains outside this patch. Head packing and multiple submissions
remain further costs to investigate, not measured kernel-level attribution.

## Validation And Limits

- Native attention tests: 15 passed, including the original Z-RBF geometry and
  frozen-parameter tests, output retention and failed-guard reuse across all
  three projection variants, and rejection of invalid register tiles.
- Native chain integration: four test functions passed, including 36 output
  comparisons with the independent PyTorch fixture. Tail dimensions and unequal
  input/output widths are included, not only tile-aligned benchmark shapes.
- Browser WASM: three presets, each on both fixture suites, passed 90 output and
  15 geometry comparisons. Maximum output error was 1.7881393432617188e-7.
  This is numerical validation, not browser throughput evidence. The browser
  adapter report does not identify a physical GPU.
- Four Python benchmark-control tests passed. Explicit modes must be confirmed
  by native reports; absent mode flags retain historical executable compatibility.
- Rust formatting passed. Focused Clippy passed with 24 pre-existing NN-library
  warnings; this is not a claim that the whole workspace is warning-free.

Outputs must be finite and satisfy `abs(actual - expected) <= 3e-6 + 3e-5 *
abs(expected)` against the frozen PyTorch 2.12.1 CPU math reference. Timed Torch
uses eager default SDPA, one CPU thread, no MPS fallback/fast-math and no global
Spiralton patches. ST includes runtime guard kernels; Torch is not given an
equivalent guard implementation. Setup, compilation and fixed geometry/mask
preparation are excluded from both timing boundaries.

This is one Apple M4 with synthetic deterministic inputs, no clock isolation,
and correlated rounds. It does not establish a PyTorch performance win, broader
hardware/shape generalization, full-decoder performance or a learning benefit.
The path remains mean-only frozen inference: no backward, uncertainty output,
KV-cache manager or complete decoder is added by this change.

## Reproduction And Provenance

The frozen implementation and shared benchmark presets are at
`34cd278d46d2ac014df2730a2b7b0526ec52e802`. Build the release example with Rust
1.98.0, then follow the [projection-only controls](../../../docs/resident_zspace_attention.md#projection-only-controls)
with five rounds, nine samples, 50 warmups and burst four. Register the **same**
executable under all three native labels. Earlier attention-kernel studies and
their hashes are unchanged.

- `initial-comparison.json` preserves the five-round exploratory wiring probe
  before preset selection was moved to a shared Rust helper. Its executable
  hash is recorded, but its pre-commit source/binary was not frozen for replay;
  use the final source-bound comparison for reproduction.
- `comparison.json` contains the repeated full matrix on the frozen source.
- `browser.json` contains the actual six rendered WASM validation reports.
- `provenance.json` binds the final executable, fixture, source files and WASM
  artifact to their recorded hashes. The final executable, generated fixtures
  and validation logs remain local; no model weights or large binaries are
  added to Git. Rebuilt binaries may differ across build environments.
- Run `shasum -a 256 -c SHA256SUMS` here to verify the published result set.
