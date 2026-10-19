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

## Token-Weighted Microbatches

```python
for group in microbatch_groups:
    accumulation = model.geometry_accumulator()
    for windows, shape in group:  # unpadded [B,T], T >= 2; lengths may differ
        accumulation.accumulate_loss(model, windows, shape)
    print(accumulation.loss, accumulation.target_tokens, accumulation.microbatches)
    model, optimizer = accumulation.step_adamw(
        model, optimizer, learning_rate=0.001, weight_decay=0.01
    )
```

`NativeCausalLmAccumulator` has no public constructor or caller-supplied
gradient setter. `model.geometry_accumulator()` seals it to the current base
and geometry IDs. `accumulate_loss` performs native forward/backward and
reduction without exporting parameter arrays. Alternatively,
`accumulation.add(model.geometry_loss_gradients(windows, shape))` accepts a
sealed native gradient batch. `step_sgd(model, learning_rate)` uses the same
mean with the existing stateless update path.

Without explicit labels, each window contributes `B * (T - 1)` targets. With
[selected causal labels](native-causal-lm.md#selected-targets-and-label-smoothing),
only non-ignored shifted targets count. The group loss and
gradient are target-weighted means, **not** equally weighted microbatch means.
For lengths three and five with batch size one, the weights are two and four.
There is no fabricated concatenated shape or inferred padding. SGD runs once,
or AdamW clips the combined gradient once and advances
its clocks once, irrespective of microbatch count.

The Rust owner retains f64 parameter/loss means and integer counters, not token
windows, logits or decoder tapes. Each window still has its own admitted
forward/replay memory. An addition stages new means before committing, so a
bad token, stale snapshot, invalid gradient or exhausted budget cannot partly
modify the accumulation. The cumulative target count is capped at `2**53` for
exact integer-to-f64 conversion. Online means reduce f32 sum-overflow risk;
they are not cancellation-proof or bitwise equivalent to a full-batch
reduction, and order can affect rounding.

An empty accumulation has `loss is None`; inspection or update of its gradients
fails. `parameter_gradients()` returns detached f32 copies rounded at that
boundary. Updates do not clear or consume the accumulation. Start a new one
for each new model snapshot; applying an old one to changed geometry rejects.
Pending accumulations are **not serialized**: checkpoint the model/optimizer
pair at completed update boundaries and save the external data cursor/schedule
and objective/label policy separately. Mid-group fault recovery must replay
that whole group. A group seals its smoothing policy at construction;
`add()` rejects a batch with a different policy before modifying means.

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

The accumulation follow-up adds five Rust and four isolated Python tests for
weighted/ragged means, full-batch agreement within tolerance, atomic rejection,
pre-extraction budgets and exact next-group continuation. Its optional 28-case
reference test covers all 14 recipes with SGD and clipped AdamW over three
updates. A fresh local run passed 48 Rust, 39 isolated Python and 249 optional
comparison/regression cases. A separate read-only source review found no
actionable correctness issue; this is not a proof of arbitrary numerical
stability or long-run model quality.

The [real GPT-2 accumulation probe](../../benchmarks/results/2026-10-18-native-gpt2-accumulation/README.md)
then checks three target-weighted groups, one optimizer tick per group and
exact next-group continuation with simultaneous Wave/fractional/Topos learning.
All short-window outcomes and raw-state hashes are retained; no new matched
real-model Torch AdamW arm, speed measurement or quality claim is implied.

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
