# Native Geometry AdamW

Source-only extension after the fixed **0.4.32** release. Installing that release
does not expose this new API yet. Optional HF/Torch wrappers remain available;
they are not silently replaced by this runtime.

`NativeGeometryAdamW` brings shared Rust AdamW and global-norm clipping to both
standalone geometry and [frozen GPT-2 language-loss learning](native-causal-lm.md).
Python only transports inputs and owns handles. It does not compute optimizer
updates, gradients, clipping rules or checkpoint validation.

## Language Learning

```python
import spiraltorch as st

# base: NativeCausalLm loaded from a supported local checkpoint.
# artifact: GeometryAdapterArtifact bound to base.base_model_identity.
model = base.with_geometry(artifact)
optimizer = st.NativeGeometryAdamW(
    model.geometry_artifact(), max_gradient_norm=1.0
)
for windows, shape in batches:  # full, unpadded token-ID windows [B,T+1]
    gradients = model.geometry_loss_gradients(windows, shape)
    model, optimizer = model.step_geometry_adamw(
        gradients, optimizer, learning_rate=0.001, weight_decay=0.01
    )
```

The base decoder remains frozen. Only trainable parameters of attached
geometry placements are updated. All supported native recipe configurations
share this owner, including scalar radius/order/gain parameters. This is not
full-model FT, an arbitrary HF architecture importer or a GPU optimizer path.

The defaults are `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`, no clipping, and
zero weight decay. Specify hyperparameters on construction; learning rate and
weight decay are explicit step arguments. No scheduler or parameter-group
decay heuristic is inferred from parameter names.

## Standalone Geometry

```python
runtime = st.NativeGeometryRuntime(artifact)
optimizer = st.NativeGeometryAdamW(artifact, max_gradient_norm=0.1)
batch = runtime.forward_sequence("site", values, (batch_size, sequence_length))
runtime, optimizer = runtime.step_adamw(
    batch, output_adjoint, optimizer, 0.001, weight_decay=0.01
)
```

`step_adamw_buffer` accepts an owned contiguous native-endian float32 buffer
such as `array('f')` for the adjoint. Both entries request the parameter-only
VJP. The adjoint must describe the loss being optimized; this API does not
invent a language loss for an isolated activation.

Unlike the existing in-place `step_sgd`, AdamW returns **two new snapshots**.
Assign both together. The original runtime/model and optimizer remain usable
and unchanged on either success or failure. A stale batch, mismatched optimizer,
non-finite candidate, exhausted clock or invalid geometry domain rejects the
whole update, never just its last parameter.

## Shared Update Rules

- The owner calls `st-kernel-contracts::adamw`; there is no second AdamW formula.
- Frozen parameters and buffers have no moment slots. Their values never change.
- Parameters in an absent placement keep their values, moments and clocks;
  they are not treated as present zero gradients.
- A present zero gradient still advances moments and permits decoupled decay.
  A zero learning rate instead freezes **values, moments and clocks**. This is
  an intentional ST contract, unlike Torch AdamW's `lr=0` behavior.
- All supplied parameter gradients, including frozen inspection gradients,
  must be finite and shape-correct. Freezing cannot hide invalid data.
- Global norm includes only selected trainable parameters, across all selected
  placements, before AdamW moments. It uses the shared f64 norm and ordered f32
  scale factors. Small-norm and near-one no-ops follow `GlobalNormClip`, not
  Torch's denominator-epsilon rule. Comparisons use explicit tolerances.

Positive AdamW configuration values and rates follow the shared portable
normal-f32 domain; zero betas/rate/decay are permitted as documented by that
kernel. Extreme finite candidates can reject conservatively. Python booleans
are not accepted as numeric hyperparameters.

## Exact Pair Resume

```python
geometry = model.geometry_artifact()
geometry_json, geometry_id = geometry.to_json(), geometry.artifact_id
optimizer_json, optimizer_id = optimizer.to_json(), optimizer.checkpoint_id
# Persist both byte strings and their separately pinned IDs.

restored_geometry = st.GeometryAdapterArtifact.from_json(
    geometry_json, geometry_id, base.base_model_identity
)
restored_optimizer = st.NativeGeometryAdamW.from_json(
    optimizer_json, optimizer_id, restored_geometry
)
restored_model = base.with_geometry(restored_geometry)
```

Optimizer checkpoints bind the **exact current geometry artifact**, parameter
names/order/shapes, betas/epsilon, clipping policy, moments and integer clocks.
They are bounded to 64 MiB, with the artifact's existing parameter budget.
Unknown fields, malformed state, divergent clocks within a placement and
inconsistent revision accounting fail before a restored owner is returned.
`revision` counts accepted nonzero-rate calls since optimizer creation; it is
not model lineage. Hashes pin bytes, not the truth of training history.

Replaying the same next input and step policy on a saved pair is exact on the
tested CPU runtime. Save the data cursor, next batch, schedule and any external
RNG separately. Native optimizer checkpoints are not Torch optimizer-state
files, and they are not automatically translated into them.

## Checks

`test_native_geometry_adamw.py` runs with imports of Torch, Transformers, NumPy,
Accelerate and Python safetensors forbidden. It checks all 14 recipes, exact
pair continuation, invalid-input atomicity and toy GPT-2 language-loss learning.
The optional `test_native_geometry_adamw_parity.py` compares eight consecutive
updates, moments and parameters over 14 recipes, two beta/epsilon settings and
clipping on/off (56 cases). Rust tests also cover subset placement updates,
frozen derivatives, scalar state, overflow, domain rejection and clock limits.

These are implementation and continuation checks, not held-out language-quality
or speedup evidence. The core compiles for WASM; no browser or GPU execution of
this new optimizer is claimed.

Local checks on 2026-10-18 passed 43 relevant Rust tests, 35 isolated Python
tests and 192 optional comparison/regression tests (including the 56 new
AdamW cases). A read-only independent review found an inconsistent-clock
restore case; the retained regression failed before the fix and passed after
adding owner-level placement/revision validation. WASM core compilation and
binding Clippy passed, with existing nonfatal warnings, not a clean whole-tree
lint claim.

The [bounded real-GPT-2 probe](../../benchmarks/results/2026-10-18-native-gpt2-adamw/README.md)
also completed three updates with Wave, fractional history and Topos together,
then reproduced the next update after restoring the artifact/optimizer pair.
Raw learned values stay local; all window results and checkpoint hashes are
published. The next model-quality study remains separate work.
