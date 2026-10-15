# Resident AdamW For Byte-Language Learning

Source API after PyPI 0.4.31. This is a learning-path extension, not a claim that
the earlier SGD language pilot now has better generated text. The existing
`ByteCorpusStudy` request/reference protocol still specifies SGD and rejects
AdamW checkpoints. A future matched study must declare its optimizer explicitly.

## One Owner, Including Geometry

`ResidentParameters` can explicitly enable AdamW. A byte model uses that same
owner for byte/position tables, every block/head, and learned geometry/router
parameters. Moments stay in GPU buffers; an update does not map gradients,
weights, moments or clocks back to the host. The update has one shared decision:

- Accepted active tensors commit weights, first/second moments and their `u32`
  accepted-update clocks together. Bias correction uses those clocks.
- Zero-rate tensors preserve all those bits, while their derivatives still
  participate in the model-wide finite/validity checks. This is explicit freeze,
  equivalent to skipping a parameter's AdamW update, **not** PyTorch `lr=0` with
  a present gradient. Clocks can consequently differ between parameter groups.
- A rejected candidate preserves every tensor and every optimizer slot. The
  attempted model revision still advances and invalidates old gradient tokens.
- Only reading the returned receipt establishes device acceptance. Submitting
  a step, or reading the attempted revision, does not establish acceptance.

AdamW updates raw parameter coordinates. This is not Riemannian Adam or manifold
momentum transport. Existing geometric forward and VJP operations remain the
source of geometric behavior; Python does not reconstruct them.

The configuration owns fixed betas and epsilon. Rates and decoupled weight decay
are explicit per-step inputs. No clipping, AMSGrad, gradient scaling or implicit
averaging is added. Enabling twice rejects instead of silently resetting history;
SGD rejects on an AdamW owner instead of discarding its moments.

## Rust

Starting with a composed `ByteDecoderPlan`, a runtime and a next-byte batch:

```rust
use st_nn::resident::{AdamWConfig, AdamWStep};

let mut model = plan.compile_training_wgpu(runtime.clone())?;
model.enable_adamw(AdamWConfig::default())?;
let batch = model.prepare_batch(&host_batch)?;
let forward = model.forward(&batch)?;
let loss = forward.next_byte_loss(loss_spec)?;
let gradients = model.backward(&forward, loss.prediction_gradient())?;
let mut steps = vec![AdamWStep::new(0.001, 0.01)?; model.parameter_layout().len()];
if let Some(geometry) = model.parameter_layout().geometry() {
    for index in geometry.all() {
        steps[index] = AdamWStep::new(0.001, 0.0)?;
    }
}
let receipt = model.adamw_with_steps(&gradients, &steps)?;
let accepted_revision = receipt.snapshot()?.read()?;
```

For a uniform policy, `model.adamw(&gradients, AdamWStep::new(rate, decay)?)`
uses the same transaction. On WASM, receipt and checkpoint reads use
`read_async().await`; there is no separate browser optimizer implementation.

## Python

Use an installed source-built WGPU wheel. With an existing plan and
`st.nn.ByteLmBatch`, the wrapper only validates configuration and calls Rust:

```python
model = plan.compile_training_wgpu()
model.enable_adamw(beta1=0.9, beta2=0.999, epsilon=1e-8)
batch = model.prepare_batch(host_batch)
forward = model.forward(batch)
loss = forward.next_byte_loss()
gradients = model.backward(forward, loss.prediction_gradient_tensor())

layout = model.parameter_layout()
rates = [0.001] * layout["count"]
decays = [0.01] * layout["count"]
if layout["geometry"]:
    for index in range(*layout["geometry"]["all"]):
        decays[index] = 0.0
        # rates[index] = 0.0  # Explicit frozen-geometry control, including history.
model.adamw_with_rates(gradients, rates, weight_decays=decays).read()

payload = model.checkpoint_snapshot().read().to_json()
resumed = st.nn.ByteDecoderCheckpoint.from_json(payload).restore_wgpu()
assert resumed.adamw_config == model.adamw_config
```

The uniform form is `model.adamw(gradients, rate, weight_decay=0.0)`. Omitted
per-tensor decay means zero, not a guessed policy for biases or geometry.
`adamw_config` is `(beta1, beta2, epsilon)` or `None` on both the live model and
portable checkpoint. Python rejects nonzero controls that would underflow to
float32 zero before they could accidentally become a freeze.

## Resume And Portability

AdamW checkpoints use `spiraltorch.nn.byte_decoder_checkpoint.v6`, with
`update_rule="adamw_zero_rate_freeze.v1"`. They include all moments/shapes,
accepted tensor clocks, configuration and the model's attempted revision. A
required `model_schema` retains the topology-specific v1-v5 discriminator:
removing a metric cannot silently fall back to Poincare while keeping moments.
Legacy SGD checkpoint encodings remain unchanged. Null/omitted/mismatched state,
invalid moments and clocks ahead of the model reject before GPU allocation.

Restore creates a fresh owner. Reapply the same data cursor, rate/decay schedule,
query-pullback control and numerical kernel options. Those are application state,
not inferred from moments. This is not a complete experiment replay or training
attestation. Preserve the exact Rust JSON string through JavaScript; parsing and
reserializing floats can lose negative zero.

The shared Rust/WGSL rule uses integer exponentiation for bias correction and
checks unsafe arithmetic before it executes. WGSL does not guarantee that an
overflow becomes a detectable infinity, and may flush subnormals; see the
[WGSL floating-point rules](https://www.w3.org/TR/WGSL/#floating-point-evaluation).
Controls must be zero or normal finite float32 values; epsilon must be positive
and normal. Conservative preflight bounds reserve 15 ULPs below float32 MAX, so
an extreme finite candidate may be rejected. Division denominators outside
WGSL's guaranteed normal range (above `2^126`) reject before division, including
the multiplication guard's own divisor. Near-MAX division numerators reject even
when the denominator is at least one, retaining quotient headroom for its allowed
rounding error. There is no promise of bitwise
equality between different devices. A tensor clock at `u32::MAX` rejects an
active update instead of wrapping; a frozen clock remains unchanged.

## Verification

The [2026-10-15 validation record](../benchmarks/results/2026-10-15-resident-adamw/README.md)
contains native, installed-wheel and actual browser results, source hashes and
explicit scope limits.

`tools/generate_resident_adamw_fixture.py` generates the independent CPU PyTorch
oracle, with `foreach=False`, `fused=False` and `grad=None` for frozen tensors.
Its 12-update fixture checks values and both moments under `atol=3e-6, rtol=5e-5`.
The shared native/browser check also covers exact same-device resume, retained
snapshots, frozen bits, late whole-model rejection, clock overflow and near-one
betas. The NN test exercises real CE/backward with contextual geometric mixtures,
freezing, rejection and full checkpoint resume. These are mechanics tests, not
a corpus-quality benchmark.

```bash
cargo test --locked -p st-kernel-contracts
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked -p st-backend-wgpu \
  --lib resident_training::parameters -- --test-threads=1
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked -p st-nn \
  --no-default-features --features wgpu --lib resident::byte_ -- --test-threads=1
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 python -I -B bindings/st-py/tests/test_nn_byte_decoder.py
```

For the browser check, build `resident_parameters_browser` for
`wasm32-unknown-unknown`, generate its web wrapper with the pinned wasm-bindgen
version, and serve only those generated files beside
`crates/st-backend-wgpu/tests/adamw_browser.html` over loopback. The downloaded
receipt must say `backend: BrowserWebGpu` and `status: passed`; a WASM build alone
does not prove GPU execution. Test wheels retaining 0.4.31 metadata are private
validation artifacts and must not be uploaded to PyPI.
