# Geometric Byte Language Models From Python

Source API added for **0.4.30**. Version metadata is not proof of PyPI
publication; use a wheel built from this source until the release is verified.
This is explicit full-window WGPU learning, not the ordinary host Tensor route,
HF fine-tuning, a streaming decoder or a KV cache.

## One Shared Model

Python handles call `st_nn::resident::ByteDecoderPlan` and
`ResidentByteDecoder` directly. The same Rust model and checkpoint schemas are
used by the native and WASM qualification examples. Python does not implement
another attention rule, geometric derivative, optimizer or resume validator.
The standalone WASM qualification is not a claim that every Python class has
a corresponding JavaScript binding in the packaged `st-wasm` client.

The nine new `spiraltorch.nn` exports are `ByteLmBatch`,
`ByteDecoderGeometryPlan`, `ByteDecoderPlan`, `ByteDecoderCheckpoint`,
`ByteDecoderCheckpointSnapshot`, `ResidentByteBatch`, `ResidentByteDecoder`,
`ByteDecoderForward` and `ByteDecoderGradients`. Existing `InferencePlan`,
`ResidualAttentionPlan`, `ResidentLoss`, `ResidentParameterUpdate` and
`WgpuTensor` handles compose with them.

The complete [runnable example](../../bindings/st-py/examples/resident_byte_geometry_lm.py)
constructs immutable CPU plans and then explicitly compiles the model:

```bash
python -I -m examples.resident_byte_geometry_lm --mode query_mixture --updates 16
```

That command runs the example included in the installed wheel. From a source
checkout, the equivalent script is
`bindings/st-py/examples/resident_byte_geometry_lm.py`.

No Torch, NumPy, tokenizer, model download or provider key is required. A
compatible WGPU device is required to execute; a CPU-only wheel can construct
and validate plans/checkpoints but rejects resident compilation explicitly.

## Training And Resume

Given a `ByteDecoderPlan` constructed as in the example:

```python
import spiraltorch as st

model = plan.compile_training_wgpu()
host_batch = st.nn.ByteLmBatch.from_documents(
    [b"SpiralTorch learns bytes.", b"Geometry changes attention."],
    [(0, 0), (1, 0)], 4,
)
batch = model.prepare_batch(host_batch)  # Explicit upload, reusable across steps.

forward = model.forward(batch)
loss = forward.next_byte_loss()
gradients = model.backward(forward, loss.prediction_gradient_tensor())
update = model.sgd(gradients, 0.125)
revision = update.read()  # A submitted update is not proof of acceptance.
print(loss.loss_tensor().snapshot().read_values())

capture = model.checkpoint_snapshot()  # Immutable capture, no implicit mapping.
checkpoint = capture.read()             # Single-use, explicit readback.
saved = checkpoint.to_json()
restored = st.nn.ByteDecoderCheckpoint.from_json(saved).restore_wgpu()
```

Each document selection is a byte offset and reads `T+1` bytes. Rows do not
attend to one another; no concatenation, padding, EOS, text normalization or
UTF-8 encoding is implicit. Pass `bytes`, not text. A byte window can cut a
UTF-8 code point; encoding/decoding policy belongs to the caller.

Restore preserves values and attempted update revision under a fresh owner.
Old forwards and gradients cannot update the restored model. By contrast,
`checkpoint.plan().compile_training_wgpu()` explicitly starts a new revision
zero from the saved values. Data cursor, learning schedule, external biases,
kernel options and query-stop policy are caller state. These are stateless-SGD
model checkpoints, not Adam/momentum or complete experiment checkpoints.

## Geometry And Interventions

`ByteDecoderGeometryPlan` composes a tokenwise `InferencePlan` projection,
trainable causal-wave decay and phase, a bounded chart and per-block/head gains.
Curvature is fixed. A raw gain of zero is **not** geometry-off: omit geometry
from the decoder plan for the no-geometry control.
Projection width is `2 * complex_channels`; decay and phase each contain one
value per complex channel, not one per real coordinate. The gain grid contains
one vector of attention-head gains per block.

| Configuration | Builder |
| --- | --- |
| One metric | `with_pair_metric(name)` |
| Fixed metric per block/head | `with_head_pair_metrics(grid)` |
| Learned shared mixture | `with_mixture(metrics, raw_logits, scales)` |
| Position-conditioned mixture | `with_query_projections(plans)` after mixture |

The canonical Rust protocol names are `poincare_squared.v1`,
`euclidean_chord_squared.v1`, `categorical_fisher_rao_squared.v1` and
`complex_projective_chord_squared.v1`. The projective operator is squared
chord, not geodesic distance. Builders return new plans without mutating old
plans or compiled models. Mixture components must be distinct; scales are
fixed, finite and positive. Router parameters remain trainable.
Query projections reserve one graph stage and one parameter slot for the shared
logit table. Oversized augmented graphs are rejected during plan construction
and checkpoint import, before any device allocation.

The contextual router consumes causal geometry coordinates at the query
position, **not the transformer's Q tensor**. `backward(...,
query_pullback="stop_coordinates")` is an explicit ablation of the router-input
pullback, not a forward change or a freeze of router parameters. It requires
a contextual router and defaults to `"full"`.

`parameter_layout()` returns half-open parameter ranges. Use them to construct
one rate per tensor for `sgd_with_rates(gradients, rates)`. Zero freezes values,
not derivatives: geometric influence can still reach the embedding, and
frozen derivatives still participate in finite-value validation.

The forward exposes `geometry_pair_bias_tensor(block)` and
`geometry_mixture_weights_tensor(block)` without mapping GPU memory.
Shared weights have shape `[heads, components]`; contextual weights have
`[batch, heads, query, components]`. Read them explicitly with snapshots.
Optional external `z_biases`/`pair_biases` must cover every block. A causal
attention mask cannot undo future information already encoded by their producer.

## Validation Scope

The installed-wheel learning-stack smoke covers CPU plan construction, all
eight example routes and exact checkpoint round-trips without dispatching a GPU.
The opt-in `test_nn_byte_decoder.py` exercises actual resident training,
retained ownership, query controls, atomic rejection, per-parameter rates and
checkpoint continuation. The separate matched-graph fixture checks Python
against the same six native/WASM cases. Its verifier requires `--wheel` and
rejects a checkout-shadowed or mismatched native extension; the receipt binds
the imported distribution to the wheel hash. It checks sixteen updates and
exact split-at-seven continuation, including full parameter derivatives,
embedding derivatives and mixture weights, not just the resumed loss.

Small repeated-window loss reduction and gradient agreement establish
mechanics, not a winning geometry or LLM-quality improvement. Compare held-out
byte losses with fixed seeds/data/budgets before selecting configurations.
Backend speed comparisons need equivalent mathematics and transfers.
