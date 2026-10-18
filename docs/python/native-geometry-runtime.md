# Native Geometry Runtime Without Torch

Source API for 0.4.32. Check the actual published package version before using
it. `pip install spiraltorch` does not require Torch, Transformers or NumPy.

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
| `GeometryAdapterStack`, `WaveGateAdapter`, `ChannelAffineAdapter` | Existing optional `torch.nn.Module` transport/integration, not removed or silently substituted |
| Elliptic orientation/readout and composite adapters | Their high-level Python compositions still use Torch; next native recovery work |
| Topos/fractional artifact execution | Native kernels exist; this selected-placement runtime explicitly rejects these recipes for now |
| HF Trainer, datasets and model execution | Optional interoperability path; not a Torch-free HF checkpoint executor |

The next useful extension is to carry the other composite recipes into this
same Rust-owned artifact/runtime contract rather than create independent Python
semantics. A pretrained HF model loader, cache behavior and full-model native FT
need their own implementation and validation; this API does not imply them.

`execution_backend` is `rust_f32_cpu`. No GPU residency, zero-copy foreign
ownership, higher-order differentiation, throughput gain or language-quality
claim is made. Stored artifacts contain adapter weights/recipes, not a complete
HF optimizer, dataset cursor or RNG checkpoint. Plain native SGD has no optimizer
state, so reproducing a continuation also requires the caller's next inputs,
targets and learning rate.

The native Rust API is `st_nn::geometry_runtime::NativeGeometryRuntime`; Python
uses the same type through PyO3, not an alternate numerical implementation.

See the [local 0.4.32 candidate checks](native-geometry-runtime-validation.md)
for installed-wheel tests and their limits. They do not establish publication.
