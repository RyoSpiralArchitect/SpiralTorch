# Learned Headwise Geometry Mixtures

The resident byte decoder can learn a convex mixture of two or three distinct
existing pair metrics: Poincare squared, origin-scaled Euclidean chord squared,
and categorical Fisher-Rao squared. The router is a per-block/head parameter
shared over positions, not a token-conditioned gate or a hard selector.

## Model Contract

```text
w[b,h,:] = softmax(router_logits[b,h,:])
bias[b,h,t,u] = -softplus(raw_gain[b,h])
                 * sum_k w[b,h,k] * fixed_scale[b,h,k] * distance_k(x_t,x_u)^2
attention = softmax(QK^T / sqrt(d) + bias + external_bias, causal_mask) V
```

The shared tokenwise projection, causal complex wave and bounded chart remain
the coordinate encoder. All component cotangents sum before wave BPTT and the
projection/embedding pullback. Positive finite scales separate component units
from normalized mixture weights and the shared gain; they are frozen topology,
not learned parameters or an implicit calibration. The Euclidean component
still uses the nonlinear chart. Q/K/V themselves are not manifold projections.

```rust
use st_nn::resident::ByteDecoderPairMetric::{
    PoincareSquared as P, EuclideanChordSquared as E,
    CategoricalFisherRaoSquared as F,
};

// One block, two heads; each table is flattened [head,component].
// Illustrative scales, not a universal calibration recipe.
let geometry = geometry.with_mixture(
    &[P, E, F],
    &[vec![-0.4, 0.2, 0.6, 0.7, -0.1, 0.3]],
    &[vec![1.0, 0.8, 24.0, 1.0, 1.2, 48.0]],
)?;
let plan = plan.with_causal_geometry(geometry)?;
let model = plan.compile_training_wgpu(runtime)?;
```

The geometry parameter layout adds one `[heads,components]` router tensor per
block after the gain tensors. `mixture_logits()` returns their absolute slot
range, and `all()` includes them. One complete-model owner and atomic SGD
decision cover embeddings, geometry, routers, blocks and output head.
`forward.geometry_mixture_weights(block)` returns that forward's resident
softmax masses, without host readback or Python/JavaScript recomputation.

For mixtures, both `pair_metric()` and `head_pair_metrics()` return `None`.
The latter now returns `Option`; fixed routes return `Some(grid)`.
`with_pair_metric` and `with_head_pair_metrics` explicitly replace a mixture
with fixed routing, removing router slots from the new plan. Duplicate metrics,
incomplete tables, nonfinite logits, nonpositive scales and an oversized packed
component volume fail before GPU allocation.

## Execution And Saving

`st-kernel-contracts::mixture` owns the backend-neutral shape contract and CPU
oracle. The resident primitive mixes `[B,H,P,K]` with `[H,K]` logits for
`2 <= K <= 8`; this model currently supplies two or three selected metrics.
Values, weights and VJPs remain on the owning GPU queue. The baseline
materializes component biases and uses a wide router reduction; it makes no
throughput or memory-optimality claim.

Router reductions remain wide through per-position pairwise centering and weighting:
`d_logit[k] = w[k] * sum_p seed[p] * sum_j w[j] * (x[p,k] - x[p,j])`.
Centering precedes position/batch reduction, so huge common components cannot
erase another position's small learning signal.
Only the final requested gradients narrow to float32. A large unweighted
reduction is not itself an error when the final derivative is finite.
Conversely, component failure guards the full forward/VJP family even when its
softmax mass is zero; even a zero-rate update must reject without mutation.

Checkpoint `spiraltorch.nn.byte_decoder_checkpoint.v4` stores `mixture` with
ordered `metrics`, fixed `scales` and current `raw_logits`, and neither fixed
route field. Null/conflicting fields, duplicate metrics, invalid tables and
legacy-schema relabeling are rejected. Topology includes scales/component
order but excludes learned logit values. Fixed plans retain canonical v1/v2/v3
bytes, and restoration creates a fresh owner.

## Qualification

The [frozen protocol](../benchmarks/results/2026-10-13-learned-geometry-mixture-v1/pre-registered.md)
covers zero-QK isolation and two QKV/Topos blocks with external biases: initial
VJPs, 16 SGD updates, fresh-owner resume after update 7, router mass history,
and atomic rejection of an invalid zero-mass component. Fabricated verifier
records only test rejection boundaries, not GPU correctness.

```sh
python3 -I -B tools/generate_resident_byte_geometry_torch_fixture.py \
  "$RAW/reference.json" --mixture
cargo build --locked --release -p st-nn --no-default-features --features wgpu \
  --example resident_byte_flat_metric
target/release/examples/resident_byte_flat_metric "$RAW/reference.json" \
  --mixture > "$RAW/native.json"
python3 -I -S -B tools/verify_byte_flat_metric.py \
  "$RAW/reference.json" "$RAW/native.json" "$RAW/native-comparison.json" --mixture
```

Build `resident_byte_decoder_browser` with the existing
[WASM recipe](resident_byte_decoder.md#reproducible-verification), serve the
same fixture bytes at `target/resident-byte-decoder-web/mixture-reference.json`,
and open `crates/st-nn/tests/byte_mixture_browser.html` on loopback. Download the
observations and use the same comparator. Production learning is Rust/WGPU;
Python is the independent oracle and JavaScript transports fixture/report bytes.

Corpus studies v1-v6 remain unchanged experiments. Learned-mixture corpus arms,
frozen-router controls and matched strength calibration need a new study.
Synthetic loss reduction is not a language-quality, preferred-geometry,
tokenizer-free advantage or improvement-over-fixed-routing result.
