# Headwise Geometry In Resident Attention

The resident byte decoder can assign a fixed pair metric to each block/head.
This composes the existing Poincare, origin-scaled Euclidean and categorical
Fisher-Rao operations in the real QKV learner, not a reporting-only selector.

```rust
use st_nn::resident::ByteDecoderPairMetric::{
    CategoricalFisherRaoSquared as Fisher,
    EuclideanChordSquared as Flat,
    PoincareSquared as Poincare,
};

// Two blocks, two heads each. Must match the plan's complete raw-gain grid.
let geometry = geometry.with_head_pair_metrics(&[
    vec![Poincare, Fisher],
    vec![Flat, Poincare],
])?;
let plan = plan.with_causal_geometry(geometry)?;
let model = plan.compile_training_wgpu(runtime)?;
```

## Semantics And Ownership

Projection, causal wave and bounded coordinates remain one shared encoder.
Each head adds `-softplus(raw_gain[block, head]) * distance_squared` to its own
QK scores; the structural causal mask remains mandatory. A route selects one
distance, not a mixture of distances within that head. No new trainable scalars,
parameter owner, optimizer slots or host activation readback are introduced.

Consecutive equal heads share a metric invocation. Its `[B,H,T,T]` result is
concatenated on the resident queue in original head order. Backward slices the
same head ranges, joins gain gradients in that order and sums coordinate
cotangents before wave BPTT and the shared projection/embedding pullback. Metric
groups inherit guards; a failed group must reject the complete model update.
Disjoint groups of the same metric are not accidentally merged or reordered.

A uniform block keeps the previous direct metric path without concatenation.
Heterogeneous routing has extra dispatch/copy costs and makes no speed claim.
This is not a separate coordinate projection per head or a learned router.
The input chart's negative curvature still does not become the intrinsic
curvature of [Fisher-Rao](fisher_rao_attention.md).

## Plan And Checkpoint Contract

`with_head_pair_metrics` freezes an exact block-major/head-major grid. Missing,
extra or empty heads/blocks fail before GPU allocation. `with_pair_metric`
resets every entry to the same metric. `head_pair_metrics()` returns
`Some(grid)` for fixed routing and `None` for a learned mixture.

Rust API migration: `pair_metric()` now returns `Option<ByteDecoderPairMetric>`.
It is `Some(metric)` only when the whole model is uniform, otherwise `None`.
Callers must not silently treat a heterogeneous model as Poincare.

Uniform plans keep their original canonical v1 (Poincare) or v2 (flat/Fisher)
checkpoint bytes. Heterogeneous plans use
`spiraltorch.nn.byte_decoder_checkpoint.v3` with `head_pair_metrics` and no
`pair_metric`. Conflicting fields, null/unknown metrics, incomplete grids,
legacy-schema relabeling and redundant uniform-v3 encodings are rejected.
Changing any route changes topology identity without changing tensor ownership.
Restoring creates a fresh owner and preserves every route and raw gain.

Corpus studies v1-v5 remain their frozen homogeneous experiments. They do not
silently gain headwise arms. A routed corpus comparison needs a separate fixed
study with strength controls before drawing language-quality conclusions.

## Qualification

The shared native/WASM harness uses two independently computed CPU-f32 Torch
fixtures: one block with four heads `[Poincare,Poincare,Fisher,Poincare]` and
zero Q/K; two blocks with `[Fisher,Flat]` and `[Flat,Poincare]`, QKV content,
external biases and Topos. Both have batch 2, four positions and width 4.
They cover unequal consecutive groups, repeated nonadjacent metrics, all three
metrics, both causal endpoints and accumulation across consuming blocks.

All initial logits/VJPs and 16 SGD updates at 0.125 use the existing
`3e-6 + 5e-5 * abs(reference)` gate. Geometry families keep the `0.002` relative
L2 gate and reference norm greater than `1e-8`. Each head's initial gain VJP
and its complete SGD gain-gradient history are additionally checked separately
under that same relative gate, so a larger neighboring head cannot hide a
disconnected small one. Resume after step 7 must preserve every loss, parameter,
geometry derivative and embedding derivative bit-for-bit in the same runtime.

An added fault-injection control changes only the final Poincare group's gain
to finite `f32::MAX`, after checking a finite, sensitive baseline. The joined
bias, output and complete VJP family must become guarded; even a zero-rate SGD
attempt must reject without modifying any model parameter. This extra control
is distinct from the fixed Torch numerical fixtures.

```sh
python3 -I -B tools/generate_resident_byte_geometry_torch_fixture.py \
  "$RAW/reference.json" --headwise
cargo build --locked --release -p st-nn --no-default-features --features wgpu \
  --example resident_byte_flat_metric
target/release/examples/resident_byte_flat_metric "$RAW/reference.json" \
  --headwise > "$RAW/native.json"
python3 -I -S -B tools/verify_byte_flat_metric.py \
  "$RAW/reference.json" "$RAW/native.json" "$RAW/native-comparison.json" --headwise
```

Build `resident_byte_decoder_browser` following the
[browser recipe](resident_byte_decoder.md#reproducible-verification).
Place the exact same reference at
`target/resident-byte-decoder-web/headwise-reference.json` and serve
`crates/st-nn/tests/byte_headwise_browser.html` on loopback. The page calls
`run_resident_byte_headwise(fixture_json)`; all learning remains Rust/WGPU.
Download its raw observations and use the same independent comparator.

These are synthetic correctness checks, not a corpus-quality or speed result.
Raw weights/checkpoints remain local; scalar evidence and hashes are published
separately. [Learned mixtures](learned_geometry_mixture.md) have a separate
model contract and qualification, not an outcome of this fixed-routing study.
Large-scale geometry selection remains separate work.

For fixed, calibrated combinations in the actual corpus learner, see
[the v6 headwise controls](byte_corpus_headwise_controls.md). The study retains
all nine homogeneous controls before adding six ordered heterogeneous routes
with trained/frozen twins.
