# Direct merged-head attention output

This study removes the separate resident head-merge copy between attention and
the output Linear, without changing Z-space geometry, masks or the mean function.
It compares the complete frozen QKV/attention/output-projection chain, not just
an attention kernel. There is no decoder, backward or learning-quality claim.

## Results And Adoption Boundary

The final three-way comparison completed all 64 engine-runs and 18 conditions.
Short resident before/specialized median ratios were 1.23-1.41 for Scalar and
1.10-1.45 for Register16. Larger inputs remain mixed: all six wide Scalar
resident medians were slower (ratios 0.951-0.999), while all six wide Register16
resident medians improved modestly (1.007-1.063). Five of six batched Register16
resident medians were still slower.

Selected causal Z-RBF resident medians, milliseconds per forward:

| Input [B,T,I] | Scalar before / specialized | Register16 before / specialized | Torch CPU | Torch MPS |
| --- | --- | --- | --- | --- |
| [1,32,64] | 0.384 / 0.272 | 0.338 / 0.306 | 0.031 | 0.174 |
| [2,128,128] | 0.913 / 0.908 | 0.661 / 0.669 | 0.307 | 0.209 |
| [1,256,256] | 3.264 / 3.342 | 2.329 / 2.280 | 0.958 | 0.393 |

Across both projection modes and boundaries, 26 of 72 specialized pooled
medians were slower. The largest observed reversal was wide/unmasked/plain
Scalar host-to-host: 4.650 -> 5.241 ms. Of 576 within-round paired medians,
371 improved, but the pairs are correlated and do not establish significance.
PyTorch remains faster in the displayed comparisons.

The initial six-round endpoint had larger-input reversals too. The three-way
follow-up changes round count and engine set, retaining the original endpoint
and including its candidate rather than selecting only a favorable rerun.
Specialization did not remove every reversal. No compiler/register-pressure,
thermal or driver cause is established by these host timings.

**Adoption:** keep ordinary NN `forward` on its established head-merge path.
Expose direct output through `forward_merged_heads` for explicit, measured
selection. Do not silently enable a shape threshold or promote Register16 to
the default. The WebGPU constant-forwarding correctness repair is retained
independently of any speed claim.

## Implementation And Rejected Intermediate State

The ordinary Rust API still returns contiguous `[B,H,Q,D]`. The new
`scaled_dot_attention_merged_heads` returns contiguous `[B,Q,H*D]` directly,
preserving immutable ownership and inherited finite-value guards. NN inference
can opt in with `forward_merged_heads`, without the former permute/pack/reshape.
The ordinary `forward` keeps its established head-major/packing route because
larger-input latency still has regressions. There is no automatic shape policy.
For the three measured
shapes, one temporary f32 values buffer of 8/128/256 KiB is eliminated. These are
computed buffer capacities, not measured peak memory. A single-head/single-query
case need not have required a head-merge copy in the old implementation.

The initial implementation selected output order with a uniform flag and
computed its address at the end of the key loop. Its six-round comparison
(`initial-comparison.json`) showed short resident gains but frequent larger-input
regressions. This is retained, not overwritten by the follow-up.

The next candidate specializes output order when constructing the pipeline and
resolves its row index before the key loop. An initial browser execution of that
candidate failed: the pinned WGPU WebGPU bridge silently omitted compute
compilation constants. `browser-failure.json` preserves that failure. A ten-line
bridge repair forwards the constant dictionary, including numeric override IDs.
Three independent compute probes and both attention output orders now exercise
the repaired browser path. Render-stage options are outside this repair.

The earlier browser evidence remains valid for the numeric checks it actually
performed. It did not prove that key-tile overrides were honored: two different
specializations can agree numerically. No older frozen record is rewritten.

## Measurement Boundaries

All engines receive the same four projections, deterministic input and fixed
geometry. Each of three shapes has plain, zero-bias and Z-RBF-bias conditions,
both unmasked and causal: 18 cases per engine. Zero bias is an identity control;
Torch receives the same nonzero geometry rather than a different plain function.

The initial comparison uses six rotated rounds with before/after Scalar,
before/after Register16, Torch CPU and Torch MPS. The follow-up compares all
three implementations in both projection modes, plus Torch CPU/MPS, over eight
rotated rounds. Each case/boundary uses 50 warmup blocks, nine timed samples per
round and a resident burst of four forwards. This yields 54 and 72 samples per
engine/case/boundary respectively. These samples and rounds are correlated, not
independent training seeds or statistical proof of universal gains.

Resident timings include forward submissions and completion, excluding output
readback/checks. Host-to-host timings include new input/bias uploads and owning
output readback; weights stay resident. Setup, compilation, geometry construction
and mask preparation are excluded. ST retains its guard kernels; Torch does not
run an equivalent guard implementation. CPU uses one thread; MPS fallback,
fast-math and optional global Spiralton patches are disabled. Torch uses default
eager SDPA, not forced math dispatch.

All sampled outputs must be finite and within `3e-6 + 3e-5 * abs(expected)`
of the frozen Torch 2.12.1 CPU math oracle. Browser results establish correctness,
not browser throughput or a physical adapter identity.

## Verification

- Six attention-contract tests passed, including merged-width overflow on an
  empty batch.
- 72 native resident-tensor tests passed, including 15 attention tests. Both
  output orders cover multiple heads/batches, widths through 256, rectangular
  sequences, strided/broadcast operands, causal offsets and both key-tile paths.
  Empty outputs, hidden upstream failures and ownership across reuse are tested.
- 15 NN attention tests and four chain integration test functions passed; the
  final API checks 72 outputs across Scalar, Register8 and Register16, with
  both ordinary and opt-in forwarding.
- Final browser WebGPU checks passed: 80 standalone outputs, 180 complete-chain
  outputs, 15 geometry checks and three compute-constant probes. Maximum output
  errors were 8.940696716308594e-8 and 1.7881393432617188e-7 respectively.
- Pinned formatting and strict backend library/test Clippy passed. Existing
  vendored WGPU warnings remain; this is not a warning-free-workspace claim.

Backend regression tests/strict Clippy were run at `9469bd2a`; `06a02fa1`
changes only the browser bridge/probe. The measured specialized executable and
fixed browser bundles were built from `06a02fa18dd9d93f1ae7f45cf6bf3b9a750726a4`.
The final opt-in NN API at `2ea9341ea0289fc6759eb9332074c1d45c97f9ce` was then
retested natively and in the browser, on both paths. NN library/tests/examples
Clippy completed with existing warnings, and the explicit benchmark example
passed `cargo check`. These are validation, not a new timing endpoint.

## Reproduction

Build and preserve separate release executables with Rust 1.98.0:

- Before: `e70f7cc80d39e9faec7648782ebba9d0dffb9613`.
- Dynamic output order: `9438d2e86249dd1b4accdc92ad60561e01c7f073`.
- Specialized, browser-fixed: `06a02fa18dd9d93f1ae7f45cf6bf3b9a750726a4`.

The benchmark and fixture generator are unchanged across these revisions.
These timing revisions temporarily routed NN `forward` through the candidate.
The final API keeps it opt-in instead. The current native benchmark explicitly
calls `forward_merged_heads` and reports `output_path=direct_merged_heads`;
the frozen comparisons remain bound to their recorded revisions, not silently
relabelled as measurements of every later wrapper or the default NN route.
Generate the benchmark fixture and use the startup environment settings in
the [benchmark instructions](../../../docs/resident_zspace_attention.md#bounded-performance-comparison).
For the eight-round follow-up:

```bash
python3 -I tools/bench_attention_chain_vs_torch.py --fixture "$FIXTURE" \
  --native "before_scalar=$BEFORE" --native-projection before_scalar=scalar \
  --native "before_register16=$BEFORE" --native-projection before_register16=register16 \
  --native "dynamic_scalar=$DYNAMIC" --native-projection dynamic_scalar=scalar \
  --native "dynamic_register16=$DYNAMIC" --native-projection dynamic_register16=register16 \
  --native "specialized_scalar=$SPECIALIZED" --native-projection specialized_scalar=scalar \
  --native "specialized_register16=$SPECIALIZED" --native-projection specialized_register16=register16 \
  --devices cpu mps --rounds 8 --samples 9 --warmup 50 --burst 4 --output "$NEW_RESULT"
```

The initial six-round recipe registers only before/after executables in both
projection modes, using labels `before_scalar`, `after_scalar`,
`before_register16`, `after_register16` in that order. All other timing options
are unchanged. Both runs use order seed 23.

Full results, rendered browser reports and source/binary/fixture hashes are
published here: `initial-comparison.json` retains the first endpoint;
`comparison.json` is the three-way follow-up. `initial-browser.json`,
`browser-failure.json`, `specialized-browser.json` and `browser.json` preserve
successive browser states. `provenance.json` distinguishes measured revisions
from the final opt-in API. Executables, WASM bundles, generated fixtures and complete logs
remain local; no model weights are included. Verify published records with
`shasum -a 256 -c SHA256SUMS`.

Single Apple M4, fixed synthetic inputs and no clock isolation: no universal
speedup, browser-throughput, CUDA or model-quality conclusion follows. The
post-hoc follow-up retains the initial endpoint rather than replacing history.
