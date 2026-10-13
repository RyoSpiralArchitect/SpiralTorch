# Learned Headwise Geometry Mixtures: Native And Browser Qualification

## Result And Scope

The actual Rust resident byte learner now trains a softmax mixture of existing
Poincare, origin-scaled Euclidean chord and categorical Fisher-Rao pair metrics
per block/head. The same implementation ran on native Metal/WGPU and actual
browser WebGPU, independently compared against one-thread CPU-f32 Torch 2.12.1.
All frozen numerical gates passed on both surfaces.

This qualifies a connected learning mechanism, not a language-quality win.
There is no held-out corpus result, speed claim, preferred geometry, advantage
over fixed routing, token-conditioned router or manifold QKV projection here.
The component biases are still materialized; no fusion/optimal-memory claim.

| Maximum across the two cases | Native WGPU | Browser WebGPU |
|---|---:|---:|
| Initial logit absolute error | 7.46e-8 | 7.46e-8 |
| All-step loss absolute error | 9.54e-7 | 4.77e-7 |
| All-step parameter absolute error | 4.77e-7 | 4.77e-7 |
| Geometry VJP relative L2 | 6.90e-5 | 5.49e-5 |
| Router VJP relative L2 | 1.32e-4 | 6.06e-5 |
| Router mass absolute error | 5.97e-8 | 5.97e-8 |
| Exact same-runtime resumed steps per case | 16/16 | 16/16 |

Bounds were fixed before full-model measurement: `3e-6 + 5e-5 * abs(reference)`
and relative geometry/router L2 `<= 0.002`, with nonzero, sufficiently sensitive
reference gradients. Router checks cover each head and each component history,
so a large neighboring gradient cannot hide a disconnected component.
Exact resume is within each runtime, not bitwise native/browser equivalence.
The browser reports `BrowserWebGpu` without an adapter name; native reports
Apple M4/Metal. The report is an observation, not hardware attestation.

## What Was Exercised

- A one-block zero-QK geometry-isolation case and a two-block QKV case with
  external attention bias and Topos, using seeds 1761 and 1863 respectively.
- All model parameters, embedding gradients, external-bias gradients, geometry
  and router gradients through 16 full-model SGD updates; 24/39 parameter
  tensors, 2522/2690 scalars. These are two tiny synthetic topologies, not a
  multi-seed corpus-quality study.
- Fresh-owner checkpoint v4 resume after update 7, including exact pre-update
  router masses, weights, loss, geometry/embedding VJPs and revision identity.
- A zero-mass component that fails numerically still guards the full model and
  rejects even zero-rate SGD without partial mutation.
- Four post-review analytic controls: large but finite final derivatives,
  mixed-scale centering across positions and batches, and identical components
  with exactly zero router derivative. Truly nonfinite final gradients reject.
- The previous headwise fixture also passed again on the reviewed native
  binary, preserving fixed-routing learning and canonical checkpoint behavior.

The router masses move, but only slightly in these fixtures (roughly `1e-7`
to `1e-5`). This is gradient-connectivity evidence, not preference discovery.
Pre-update losses at steps 1 and 16 are approximately `5.5511 -> 3.9425` and
`5.5237 -> 3.9158`, on the same synthetic training windows, not held-out text.

Tests: 6 CPU contract tests; 6 WGPU backend tests (4 actual GPU tests plus
shader/limit checks); 24 byte-model tests with GPU runtime opt-in; 71 Python
verifier tests. The Python tests use fabricated records to test rejection
boundaries and are not themselves GPU evidence. Formal CI was not run here.

A read-only independent local reviewer found and re-reviewed two numerical
issues: early float32 narrowing and centering after reduction. Both were fixed
without weakening gates. No remaining concrete P1/P2 was reported. This is
local source review, not human review or exhaustive numerical proof.
See the [post-protocol addendum](review-regressions.md) for chronology.

## Evidence And Reproduction

- [Frozen original protocol](pre-registered.md).
- [Native scalar comparison](native-comparison.json) and
  [browser scalar comparison](browser-comparison.json).
- [Fixed-head native regression](headwise-regression-comparison.json).
- [Manifest](manifest.json), [source file hashes](source-files.json) and
  [published file hashes](SHA256SUMS).
- [Rust API and numerical contract](../../../docs/learned_geometry_mixture.md).

The complete fixture, parameter/VJP/mass histories, checkpoint JSON, binaries,
logs and superseded attempts are retained locally. Only scalar evidence,
verification records, fixed settings and hashes are published. An initial
backend test failure (guard-buffer COPY_DST usage) and both superseded review
stages remain in the manifest; successful ordinary fixtures do not erase them.

From the repository root, use the recorded source revision and an isolated
local output directory. The generator refuses to overwrite an existing fixture.

```sh
export RAW=/path/to/new-local-validation
mkdir -p "$RAW"
python3 -I -B tools/generate_resident_byte_geometry_torch_fixture.py \
  "$RAW/reference.json" --mixture
cargo build --locked --release -p st-nn --no-default-features --features wgpu \
  --example resident_byte_flat_metric
"${CARGO_TARGET_DIR:-target}/release/examples/resident_byte_flat_metric" \
  "$RAW/reference.json" --mixture > "$RAW/native.json"
python3 -I -S -B tools/verify_byte_flat_metric.py \
  "$RAW/reference.json" "$RAW/native.json" "$RAW/native-comparison.json" --mixture
```

Follow the [WASM build recipe](../../../docs/resident_byte_decoder.md#reproducible-verification)
for `resident_byte_decoder_browser`; use a compatible wasm-bindgen CLI
(this run used 0.2.129). Serve the resulting JS/WASM at
`target/resident-byte-decoder-web/`, place the same opaque fixture bytes there
as `mixture-reference.json`, and open
`crates/st-nn/tests/byte_mixture_browser.html` through a loopback server.
Download the browser observations and run the same independent comparator with
`--mixture`. Run GPU jobs serially. The browser's compact success summary is
not a substitute for the independent all-step comparison.

Next efficacy study: retain the existing frozen corpus v1-v6 records, then
compare individual metrics, frozen mixtures and learned mixtures with the same
data order, training budget, predeclared seeds and train-only strength
calibration. This qualification alone does not select a winning mixture.
