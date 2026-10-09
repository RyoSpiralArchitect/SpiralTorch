# Fisher-Rao Geometry In Resident Attention

The existing narrative `InformationGeometryMetric` and concept-diffusion
comparison use categorical Fisher-Rao geometry. The resident byte decoder now
also exposes it as an explicit trainable pair metric:

```rust
use st_nn::resident::ByteDecoderPairMetric;

let geometry = geometry.with_pair_metric(
    ByteDecoderPairMetric::CategoricalFisherRaoSquared,
);
let plan = plan.with_causal_geometry(geometry)?;
```

This extends the existing Rust learner, rather than introducing a Python or
JavaScript training surrogate. The ordinary QKV attention algorithm remains
unchanged; this is a learned, causal addition to its score matrix.

## Geometry And Pullback

The same tokenwise projection, causal complex wave, bounded coordinates and
parameter owner are retained. Interpret each coordinate row `z` as logits of a
categorical distribution `p = softmax(z)`. For two rows:

```text
r = sqrt(p), s = sqrt(q)
t = sum((r - s)^2)
d_FR(p,q)^2 = 16 * asin(sqrt(t) / 2)^2
bias[b,h,q,k] = -softplus(raw_gain[h]) * d_FR(p_q,p_k)^2, k <= q
```

This is the radius-two square-root embedding, equivalent to
`d_FR = 2 * acos(sum(sqrt(p*q)))`; see
[Nielsen, equation 4 and section 6.1](https://arxiv.org/html/2403.10089v4).
It is not the Hellinger chord distance. Squared distance, rather than distance,
has a smooth identity pullback. A small-chord analytic series preserves nearby
distances without a constant epsilon floor. The CPU distance helper is shared
with `st-core::inference::concept_diffusion::compare_fisher_rao`, including its
nearby-distribution and tiny-affinity boundary handling.

The root-map pullback is `dz_i = 0.5*r_i*(dr_i - r_i*sum(r_j*dr_j))`.
Both causal endpoints contribute. All consuming residual blocks accumulate
before the wave BPTT and token/position embedding pullback. Projection weights,
wave decay/phase and per-block/head gains remain normal model parameters. No
geometry branch is detached and no extra parameter owner is introduced.

The configured negative curvature still controls the existing **input chart**;
it is not the intrinsic curvature of the categorical Fisher metric. Softmax adds
invariance to rowwise logit shifts and changes the effective function class.
Equal parameter count therefore does not imply equal functions, capacity or
compute. This is a distribution over latent coordinates, not over vocabulary
tokens and not the attention probability distribution itself.

## Native And Browser Contract

`ResidentTensor::causal_fisher_rao_bias` accepts `[B,T,C]` logits and `[H]` gains.
The root map, pair distances, head gains and both VJPs stay on the owning WGPU
queue. Strided inputs are materialized on-device; only explicit snapshots read
back. Invalid upstream guards propagate through both coordinate and gain
gradients. Saved forward tapes are immutable across repeated backward calls.
The root-space cotangents remain in the shared wide-arithmetic representation
until the final simplex pullback: a root cotangent outside float32 range must
not reject an otherwise finite final logit gradient. Final-output overflow and
invalid upstream tensors still reject the whole gradient family.

Future entries are **zero**, not an attention mask. Consumers must retain their
structural causal mask. Float32 root storage has normal finite-precision and
saturation limits; the analytic series does not claim arbitrary-precision
recovery of differences lost during softmax/root evaluation.

Checkpoints explicitly store `categorical_fisher_rao_squared.v1` in the existing
metric-aware model schema v2; schema v1 cannot silently load this metric.
Corpus-study v1-v4 remain their predeclared ordinary/Poincare/flat experiments.
They do not silently expand to this metric or recalibrate it.

## Qualification Recipe

Use a fresh local `$RAW` directory. The initial correctness recipe retains the
two existing one/two-block fixtures, seeds 1761/1863, 23/37 parameter tensors,
16 SGD updates at 0.125, and the existing numerical/gradient gates. One case
zeros Q/K to isolate geometry; the other combines QKV, external biases and
Topos. It also checks metric-off versus detached pullbacks, causal isolation,
guarded rejection, checkpoint import and exact same-runtime resume at step 7.
The shared native/browser runner also exercises a large-cotangent regression
whose root derivative overflows float32 but final logit/gain derivatives do not.
The comparator independently recomputes that fixed case in float64; this
boundary check is distinct from the frozen full-model Torch reference.

```sh
python3 -I -B tools/generate_resident_byte_geometry_torch_fixture.py \
  "$RAW/reference.json" --fisher-rao
shasum -a 256 "$RAW/reference.json"
CARGO_INCREMENTAL=0 cargo build --locked --release -p st-nn \
  --no-default-features --features wgpu --example resident_byte_flat_metric
target/release/examples/resident_byte_flat_metric "$RAW/reference.json" \
  --fisher-rao > "$RAW/native.json"
python3 -I -S -B tools/verify_byte_flat_metric.py \
  "$RAW/reference.json" "$RAW/native.json" "$RAW/native-comparison.json" \
  --fisher-rao
```

Build `resident_byte_decoder_browser` as in the existing byte-decoder recipe.
The new WASM export is `run_resident_byte_fisher_rao(fixture_json)`.
Serve `crates/st-nn/tests/byte_fisher_rao_browser.html` on loopback, with the exact
frozen fixture at `target/resident-byte-decoder-web/fisher-reference.json`.
Download opaque Rust JSON and run the same comparator with `--fisher-rao`.
Native/browser GPU runs must be sequential, not competing workloads.

Full raw weights, checkpoints and logs stay local. Publish scalar observations,
source hashes and reproduction, retaining failed attempts. These synthetic
checks establish neither language-quality gains nor performance gains.
Strength-matched corpus controls and geometric mixtures are subsequent
experiments, not conclusions of this qualification.

The [initial native/browser qualification](../benchmarks/results/2026-10-10-fisher-rao-attention-v1/README.md)
retains scalar traces, independent comparisons and the failed-then-repaired
large-cotangent review case.
