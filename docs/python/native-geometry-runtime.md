# Native Geometry Runtime Without Torch

Source API for 0.4.32. Check the actual published package version before using
it. `pip install spiraltorch` does not require Torch, Transformers or NumPy.
The elliptic, Topos and fractional extensions below are source-only after the
0.4.32 candidate; they are not covered by that candidate's recorded wheel checks.

`NativeGeometryRuntime` executes **selected placements** from the same
`GeometryAdapterArtifact` used by optional HF adapters. Residual composition,
pullbacks, mean squared error, SGD and updated-artifact validation live in Rust.
Python transports values and chooses calls; it does not reimplement the math.

## Saved Artifact to Learning

```python
from array import array
import spiraltorch as st

# These pins come from the saved artifact's trusted local manifest.
artifact = st.GeometryAdapterArtifact.from_json(
    payload, expected_artifact_id, expected_base_model_identity
)
runtime = st.NativeGeometryRuntime(artifact)
batch = runtime.forward_buffer(path, array("f", activation_values), rows)
input_gradient, parameter_gradients = batch.vjp(upstream_values)

# Atomic stateless SGD, rejecting a snapshot from a different/currently stale
# artifact. Fixed buffers and requires_grad=False parameters are not updated.
runtime.step_sgd(batch, upstream_values, learning_rate=0.01)
updated = runtime.artifact()
```

`forward`/`vjp` accept flat numeric sequences; `forward_buffer`, `vjp_buffer` and
`output_buffer` use owned native-endian contiguous float32 buffers. An
`array('f')` is enough. Leading batch/sequence axes may be flattened only for
these feature-wise operators; the feature width is fixed by the artifact.
Changing a caller's input/output buffer cannot mutate a saved snapshot.

For a direct native learning loop, `train_mse` or `train_mse_buffer` returns the
pre-update mean-MSE and performs one plain SGD step using the existing Rust loss
and its pullback. No Python loss, Torch optimizer, scheduler or hidden adaptive
state is involved. A failed target/gradient/update leaves the artifact intact.

Run the self-contained synthetic learning and exact-reload example:

```sh
python -I -B bindings/st-py/examples/native_geometry_training.py
```

This exercises activation fitting, not whole-language-model quality.

## Scope and Next Recovery Work

| Surface | Ownership and current dependency |
| --- | --- |
| Native tensors/autograd/NN and resident byte decoder | Existing Rust execution; no Torch required |
| Affine, WaveGate v1, fixed/learned-radius WaveGate v2 artifacts | Native selected-placement forward/VJP/SGD/MSE in this API |
| Pointwise and anchored elliptic artifacts (source-only extension) | Native input projection, chart, geometric map, readout and complete first-order VJP/SGD |
| `GeometryAdapterStack`, `WaveGateAdapter`, `ChannelAffineAdapter` | Existing optional `torch.nn.Module` transport/integration, not removed or silently substituted |
| Existing `EllipticResidualAdapter` / HF wrappers | Optional Torch compositions remain available for compatibility/comparison; the same pointwise/anchored saved artifact can now execute through the native runtime |
| Causal and gated-causal elliptic artifacts (source-only extension) | Native full-context execution through the explicit sequence API; the row-only API still rejects them |
| Topos resonance (source-only extension) | Native shared-feature gates, finite resonance recurrence and complete residual VJP/SGD; row and sequence entries |
| Fractional memory/history, L2 history, learned-gain and angle/gain history (source-only extension) | Native full-context GL composition, feature gates and parameter charts; explicit sequence entry only |
| HF Trainer, datasets and model execution | Optional interoperability path; not a Torch-free HF checkpoint executor |

All recipes currently defined by `GeometryAdapterArtifact` have a native
selected-placement route. This is not a whole-model executor: a pretrained HF
model loader, placement scheduling, cache behavior and full-model native FT
need their own implementation and validation. No Python wrapper or foreign
model is silently replaced by this API.

`execution_backend` is `rust_f32_cpu`. No GPU residency, zero-copy foreign
ownership, higher-order differentiation, throughput gain or language-quality
claim is made. Stored artifacts contain adapter weights/recipes, not a complete
HF optimizer, dataset cursor or RNG checkpoint. Plain native SGD has no optimizer
state, so reproducing a continuation also requires the caller's next inputs,
targets, learning rate and, for sequence execution, the same batch/sequence partition.

The native Rust API is `st_nn::geometry_runtime::NativeGeometryRuntime`; Python
uses the same type through PyO3, not an alternate numerical implementation.

## Pointwise Elliptic Extension

The source-only extension reads the existing `orientation.weight`,
`orientation.bias`, `readout.weight` and optional `raw_mix` tensors without
changing the artifact schema. Rust Tensor primitives own both linear maps and
their derivatives; `st-core` owns the original chart map or fixed-anchor
correction and its saved pullback. The fixed first chart coordinate is not a
parameter. Weight storage remains `[output, input]` for artifact compatibility.
Forward packs the transposed weights directly instead of materializing an
extra transposed tensor. This is not a measured throughput claim.

Admission is at most 65,536 rows and 1,048,576 input values, checked before
Python sequence/buffer extraction. Rows are independent, including across
flattened batch/sequence axes. Frozen projection tensors do not update; a
zero readout starts at exact identity but has a nonzero learning path. The
causal variants require the separate sequence-aware entry point below and remain
explicitly unsupported by `forward`/`forward_buffer`. The native SGD and artifact-continuation API is
unchanged; no new Python numerical implementation or Torch fallback is added.

Pointwise checkpoint `5716e981f` qualification on 2026-10-18: 8 Rust runtime tests, 8 installed
native Python tests without optional ML packages, and 52 optional Torch/artifact
tests passed. Elliptic comparisons cover widths 1, 7, 64 and 768 with `[2,4,F]`
inputs; forward/input/parameter gradients use `atol=rtol=1e-5` for those widths.
Tiny complete SGD comparisons retain the tighter `1e-6` tolerance. Both elliptic
variants also pass finite differences, identity-start learning, exact artifact
continuation, frozen projections, row admission and atomic invalid-update tests.
The WASM core compiles; browser execution was not tested. Independent source
review found no actionable issues, but did not independently execute the tests.

The isolated source-branch arm64 wheel has SHA-256
`b0889df5bf4e84f625f2001710eb79c627362834fb47efdbbb0850520b0ea04f`.
It retains development metadata `0.4.32` and is **not** the fixed 0.4.32 release
candidate or a publishable official wheel. It was installed in a separate local
environment without replacing the earlier candidate or experiment packages.

See the [local 0.4.32 candidate checks](native-geometry-runtime-validation.md)
for installed-wheel tests and their limits. They do not establish publication.

## Full-Context Elliptic Extension

`forward_sequence` and `forward_sequence_buffer` take an explicit
`(batch, sequence)` pair. Feature width comes from the saved artifact. The native
kernel preserves each sequence: it never treats all rows as one token stream.
The existing Rust causal map uses tied `Q=K=V=phi(orientation)` on nine elliptic
features; this runtime composes that map with the learned projection/readout and
residual, without adding a second attention formula. The pullback includes all
three tied paths. The gated variant preserves a learning signal for `raw_mix`
even when its initial zero value gives exactly pointwise features.

```python
# runtime holds a compatible saved artifact; values are flattened [B,T,F].
capture = runtime.forward_sequence_buffer(path, array("f", activation_values), (B, T))
assert capture.sequence_shape == (B, T, F)
assert capture.shape == (B * T, F)  # Physical host-Tensor storage, not lost context.
input_gradient, parameter_gradients = capture.vjp(upstream_values)
runtime.step_sgd(capture, upstream_values, learning_rate=0.01)
```

The sequence API also accepts supported pointwise recipes, keeping the same
explicit shape without introducing token mixing. Row-only snapshots keep
`sequence_shape=None`. The source API is CPU f32 and full-prefix, **nonempty,
unpadded** execution: no padding mask, dropout or KV cache. Calling it separately
on single-token chunks is not equivalent to a full-context causal call.

Rust checks `B*T <= 65,536`, `B*T*T <= 1,048,576` potential score pairs (including
masked pairs), and `B*T*F <= 1,048,576` values for causal recipes before Python
input extraction or Rust input copies. Zero residual strength does not bypass
these admission limits. Pointwise recipes retain their existing value budgets.

`train_sequence_mse` / `train_sequence_mse_buffer` perform one native mean-MSE
SGD step across all values in the full unpadded context. Save/reload semantics
are unchanged: reproducing the next step requires the same inputs, targets,
learning rate and context partition, not just the adapter artifact.

Run the no-Torch synthetic learning, exact continuation and causal-boundary example:

```sh
python -I -B bindings/st-py/examples/native_geometry_sequence_training.py
```

This is an activation-fit mechanics example, not pretrained-HF-model FT or
language-quality evidence. It does not change the optional Torch/HF wrappers.

### Local Sequence Qualification

On 2026-10-18, the sequence extension passed 13 Rust runtime tests and 11
installed-wheel native Python tests in a fresh environment without Torch,
Transformers, NumPy or Accelerate. The optional Torch/artifact suite passed 62
tests, including forward, input/parameter VJP and SGD comparisons for eight
recipes. All four elliptic variants cover widths 1, 7, 64 and 768 with `[2,4,F]`
inputs. Tiny comparisons use `atol=rtol=1e-6`; wide comparisons use `1e-5`.
Finite differences independently check the causal and gated-causal pullbacks.

Tests reject future-token and cross-batch influence while confirming nonzero
past-token gradients. A zero context gate preserves the pointwise result and
still receives a learning signal. Invalid updates leave the artifact unchanged,
and admission rejects excessive contexts before extracting foreign input values.

The synthetic example above, both from source and from the installed wheel,
reduced mean-MSE from `0.0795833319` to `0.0545243919` across 40 updates. Its next
step after artifact reload matched exactly. This only establishes the declared
activation-fitting mechanics, not language-model quality or a speed advantage.

Three existing core smoke programs and 31 release/compatibility tests also
passed. WASM core compilation and strict WGPU rustdoc checks passed; this is not
browser execution evidence. Clippy completed with the same 24 existing `st-nn`
warnings and no warning in this runtime. Independent source review found no
actionable issues but did not execute these tests itself.

The isolated source-branch arm64 wheel has SHA-256
`69572def1f307877ec7444bba066f409d24d0778692b0406ec02ee874d177a62`.
Like the pointwise checkpoint above, it retains development metadata `0.4.32`
and is **not** the fixed release candidate or a publishable official wheel.
The candidate and prior experiment environments were not replaced. No
cross-platform execution or PyPI publication is established by these checks.

## Resonance and Fractional Memory

The source-only runtime also accepts Topos resonance and all five existing
fractional adapter recipes. Python still passes the same saved artifact, flat
numeric sequence or owned f32 buffer; there is no new Python numerical path.
Use `forward_sequence[_buffer]` / `train_sequence_mse[_buffer]` with `(B, T)`
for fractional recipes, even when a particular history window is empty. The
row-only entry rejects these recipes rather than guessing token boundaries.
Topos is pointwise and supports both entries, including empty row batches.

Rust composes the existing kernels with the learned controls:

- Topos uses the saved shared-feature gate, coupling, finite iteration count
  and open-topos envelope. It retains the native unrolled pullback, not a
  new recurrence or a fixed-point approximation.
- Fractional memory gates the existing zero-padded causal GL difference.
  History recipes instead gate the strictly-past term and add an independent
  local feature gate. All feature gates use `tanh`, including its pullback.
- Log-order recipes differentiate `alpha=exp(log_alpha)` in Rust; angle/order
  and learned log-gain use the existing `st-frac` chart and differentials.
  Invalid updates are rejected atomically, without clipping or projecting
  parameters back into the valid domain.
- L2/gain recipes normalize over the full declared kernel. Optional lag
  windows are applied after that normalization, exactly as in `st-frac`.
  An empty window has zero history, order and gain gradients while retaining
  the independent local gate. Full-context, nonempty, unpadded execution and
  the no-cache boundary remain unchanged.

Before extracting Python values, Rust checks the saved `max_values` against
`B*T*F`, and `max_products` against `B*T*F*kernel_len`. The latter remains a
conservative full-kernel budget even for short prefixes or empty lag windows.
Zero residual strength does not bypass either check. Topos uses its saved
volume limit. Captures own the forward values and controls; mutating a caller's
buffer or updating the live artifact cannot change an old capture's pullback.

Run a synthetic identity-start learning and exact-reload example without Torch:

```sh
python -I -B bindings/st-py/examples/native_geometry_memory_training.py --recipe history_angle
python -I -B bindings/st-py/examples/native_geometry_memory_training.py --recipe topos
```

The other example choices are `memory`, `history`, `history_l2` and
`history_gain`. These fit synthetic activation targets with native mean-MSE and
plain SGD; they do not download a model or establish language-quality gains.

### Local Memory Qualification

On 2026-10-18, 19 native Rust runtime tests, 14 installed-wheel Python tests
without optional ML packages, and 100 optional Torch/artifact tests passed.
The comparisons cover all 14 configurations (fixed and learned WaveGate radius
are separate configurations), including Topos and all five fractional recipes.
Memory/resonance forward and input/parameter gradients cover widths 1, 7, 64
and 768 with `[2,4,F]` contexts at `atol=rtol=1e-5`. Tiny forward/VJP/SGD and
history-window comparisons retain `1e-6`. Rust finite differences also check
every input and trainable parameter in the six new compositions.

The six identity-start synthetic examples use `[2,3,3]` inputs, 40 native SGD
steps and the same mean-MSE target. They all start at `0.0795833319` and finish
as follows; the subsequent update matches exactly after artifact reload:

| Example recipe | Last mean-MSE |
| --- | --- |
| `topos` | 0.0611702800 |
| `memory` | 0.0758749843 |
| `history` | 0.0625096411 |
| `history_l2` | 0.0641992092 |
| `history_gain` | 0.0613429956 |
| `history_angle` | 0.0610081293 |

Both source and wheel-packaged examples passed. These numbers exercise learning
mechanics, not a model-quality ranking or a matched capacity comparison. Three
existing core smoke programs and 31 release/compatibility tests passed, as did
WASM core compilation, strict WGPU rustdoc, pinned formatting and Python lint.
Clippy completed with 24 existing `st-nn` warnings and none in this runtime.
Browser execution and cross-platform numerics were not tested here.
Independent source review found no actionable issues; it did not independently
rerun the tests or verify the wheel and numerical results.

The separately installed local arm64 wheel has SHA-256
`c01fd1105bf1ec26d603fb58636056911a4b378030e1a1bcb77cb0a58bb23d22`.
It is a source-extension test wheel with development metadata `0.4.32`, **not**
the fixed candidate or an official release asset. Existing candidate and
experiment environments were left unchanged; PyPI publication is not implied.
