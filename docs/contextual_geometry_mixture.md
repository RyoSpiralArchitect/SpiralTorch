# Contextual Geometry Routing

`ByteDecoderGeometryPlan::with_query_projections` extends an existing mixture
with one tokenwise graph per block. Each graph reads the shared causal wave
coordinates `[B,T,D]` and emits `[B,T,H*K]`. Its output plus that block's shared
`[H,K]` logit table produces query-local softmax weights `[B,H,T,K]`.

```rust
let geometry = geometry
    .with_mixture(&metrics, &initial_logits, &fixed_scales)?
    .with_query_projections(&query_projections)?;
let plan = byte_plan.with_causal_geometry(geometry)?;
let mut model = plan.compile_training_wgpu(runtime)?;
```

Projections must preserve batch/time, consume exactly the causal chart layout,
contain trainable parameters and use the validated tokenwise graph contract.
They are not graphs over an entire future window or the current block's hidden
states. All attention blocks consume the same causal coordinate encoder.

The resident graph binds the shared table as its final broadcast-add parameter.
Backward therefore returns its exact reduction, the query graph's parameter
VJPs and an input VJP added to all metric-coordinate contributions before wave
BPTT. Nothing is detached or copied to the host during this path. The one model
parameter owner updates all slots atomically, including failure propagation.

The order is encoder parameters, wave parameters, gains, shared logit tables,
then each query graph's parameters, followed by normal decoder blocks/head.
`query_projections()` on the geometry parameter layout identifies each range.
`geometry_mixture_weights()` returns `[H,K]` for shared routing and `[B,H,T,K]`
for contextual routing. A zero-output projection retains the original initial
mixed function, not an identical future optimization trajectory.

Checkpoint v5 stores the query graphs and their current values alongside the
shared tables. Shared-only mixtures retain v4. Dropping query graphs or changing
schema is rejected instead of silently restoring a shared-only model.

## Comparison Boundaries

Generic projections may include biases, as the linear qualification examples
do. Such a bias and the shared intercept are separate trainable slots, with
redundant constant directions. This qualification does not claim parameter-
budget or shared-intercept-update equivalence. A corpus comparison must declare
its parameter counts and any frozen-slot policy rather than hiding this effect.

The qualification combines three existing metrics, not new geometric families.
It checks the complete byte-LM learning path and resume, not language quality.
Large-corpus comparisons and Python exposure remain separate work.

## Reproduce

Use a new local directory for `RAW`; never publish its tensor/checkpoint payloads.

```sh
python3 -I -B tools/generate_resident_byte_geometry_torch_fixture.py \
  "$RAW/reference.json" --query-mixture
cargo build --locked --release -p st-nn --no-default-features --features wgpu \
  --example resident_byte_flat_metric
"$CARGO_TARGET_DIR/release/examples/resident_byte_flat_metric" \
  "$RAW/reference.json" --query-mixture > "$RAW/native.json"
```

For WASM, build `resident_byte_decoder_browser` with those features for
`wasm32-unknown-unknown`, run matching `wasm-bindgen --target web`, and serve its
JS/WASM plus `reference.json` under `target/resident-byte-query-web/` on loopback.
Open `crates/st-nn/tests/byte_query_mixture_browser.html`. Native and browser
must share the exact fixture, gates and Rust harness. Raw reports intentionally
include all parameter snapshots for local verification; publish projections only.
