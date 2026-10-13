# Complex Projective Attention Geometry

The Rust-owned `ComplexProjectiveChordSquared` pair metric adds a fourth
geometry to the causal byte decoder: Poincare squared distance, scaled Euclidean
squared distance, categorical Fisher-Rao squared distance, and complex
projective chord squared. It is opt-in, not a replacement for QKV attention.

**Experimental:** the fixed six-case learning qualification is not fully
passing. See the [recorded study and unresolved gates](../benchmarks/results/2026-10-13-complex-projective-v1/README.md)
before using this path in a larger geometry sweep. Primitive tests passing and
successful checkpoint resume do not waive failed learning-gradient checks.

## Exact Meaning

Consecutive real/imaginary chart components form a complex vector `z`.
The regular homogeneous chart is `u=(1,z)/sqrt(1+|z|^2)`; zero input is valid.
For two normalized states, `d^2=1-|u^H v|^2`, equivalently half the squared
Frobenius distance of their rank-one projectors. Squared overlap uses the
[pure-state fidelity convention](https://quantum.cloud.ibm.com/docs/en/api/qiskit/quantum_info#state_fidelity).
This is **not** Fubini-Study geodesic distance, a quantum computer, or evidence
of quantum advantage. The square root is the chord distance.

The pure contract evaluates a sum of squared complex wedges, avoiding the
near-identity cancellation in `1-overlap`. The analytic pullback uses a
factored quotient derivative rather than subtracting saturated terms.
Compensated sums retain individual real products, the homogeneous anchor and
base endpoints before cancellation. Analytic regression cases cover those
specific failure modes; this is not an exact-arithmetic guarantee for every
ill-conditioned finite chart.
WGPU retains extended intermediates, including the reduced head cotangent,
until each final f32 output. No optional GPU capability or CPU fallback is
required. The wedge implementation costs quadratically in complex chart width;
it is not an optimized speed claim.

`st_core::theory::stv::PauliSpinor::projective_chord_squared` calls the same
backend-neutral projective contract. For CP1, it equals one quarter of the
squared Euclidean Bloch-vector chord. Separate global phase/scaling of
homogeneous states does not change the result. Rotating one anchored chart's
`z` alone is not generally a global phase of `(1,z)`.

## Learning Path

The existing tokenwise encoder and causal complex wave produce shared real
coordinates. The projective metric interprets adjacent coordinates as complex
pairs, applies its anchored chart, and contributes
`-softplus(head_gain)*d^2` to the ordinary QKV scores before causal softmax.
Future bias entries are zero; the attention consumer must still enforce its
structural causal mask.

```rust
use st_nn::resident::ByteDecoderPairMetric::ComplexProjectiveChordSquared;

let geometry = geometry.with_pair_metric(ComplexProjectiveChordSquared);
let plan = byte_plan.with_causal_geometry(geometry)?;
let mut model = plan.compile_training_wgpu(runtime)?;
```

Alternatively, `with_mixture` accepts two to four distinct metrics; fixed
positive per-head/component scales declare units, not a fair parameter or
compute budget. `with_query_projections` makes mixture weights causal and
position-dependent. Both metric endpoints, head gains, shared/router parameters,
wave recurrence and embeddings receive gradients through the same model owner.
The router still reads shared wave coordinates, not each block's hidden states.

The shared wave's [batch-gradient reduction](causal_wave_batch_gradients.md)
retains small decay/phase signals through batch cancellation. This fixes a
specific gradient-aggregation error; it does not close the projective
trajectory qualification failure.

An explicit [query-coordinate pullback control](query_coordinate_pullback.md)
can stop only the router's input gradient for paired experiments or selective
training. Ordinary backward remains fully coupled; this control does not
supersede the failed projective learning qualification.

The serialized metric identity is `complex_projective_chord_squared.v1`.
Existing checkpoint encodings remain v2 (uniform), v3 (headwise), v4 (shared
mixture) and v5 (query mixture). The explicit metric identity binds topology;
unknown identities fail closed. Old readers reject the new metric rather than
silently treating it as Poincare.

## Qualification

`tools/generate_resident_byte_projective_torch_fixture.py` freezes six
synthetic cases: projective-only, four-way shared and four-way query routing,
each with one and two decoder blocks. The one-block cases zero Q/K to isolate
geometry; two-block cases retain ordinary QKV, external biases and a Topos
gate. Each has 16 complete SGD updates and same-runtime checkpoint resume.
An independent Torch oracle uses explicit complex projectors and autograd,
not the Rust wedge/analytic VJP implementation.

The gates retain the existing absolute/relative output checks, a 0.002 relative
geometry-gradient threshold, geometry-off/detached controls, causal-prefix
checks, exact same-runtime resume, and nonzero router update controls.
New fixtures do not replace the earlier contextual v1 browser failure or any
v8 corpus result. Passing these checks establishes numerical wiring on a tiny
synthetic model, not useful language learning, an LLM FT benefit, or speed.

Only this projective qualification retains finite learning-comparison failures
in `qualification_failures` and continues collecting the trajectory. Such
failures still set `passed=false`, and the native example exits unsuccessfully.
Malformed/nonfinite contextual controls are rejected separately. Legacy
three-metric qualification entry points keep their previous strict behavior.

Use a private `RAW` directory. For clean CPU reference generation on machines
with automatic site hooks, the command below disables site initialization:

```sh
python3 -I -S -B -c 'import sys,sysconfig,runpy; sys.path.append(sysconfig.get_path("purelib")); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name="__main__")' \
  tools/generate_resident_byte_projective_torch_fixture.py "$RAW/reference.json"
cargo build --locked --release -p st-nn --no-default-features --features wgpu \
  --example resident_byte_flat_metric
"$CARGO_TARGET_DIR/release/examples/resident_byte_flat_metric" \
  "$RAW/reference.json" --projective > "$RAW/native.json"
```

Build `resident_byte_decoder_browser` with the same features for
`wasm32-unknown-unknown`, run matching `wasm-bindgen --target web`, and serve
the generated JS/WASM and the **same** fixture under
`target/resident-byte-projective-web/` on loopback. Open
`crates/st-nn/tests/byte_projective_browser.html`. Raw outputs contain local
parameter snapshots and checkpoints: publish scalar summaries and hashes,
not the payloads.
