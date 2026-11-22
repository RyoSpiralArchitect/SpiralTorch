# Fixed-length steps from a sealed native gradient

The Rust and Python method
`NativeCausalLm.step_geometry_gradient_at_distance(gradients, target_l2)`
uses the **unrounded negative gradient**, not an ordinary SGD proposal.
For example, with an attached trainable geometry:

```python
g = model.geometry_declared_rival_gradients(
    [*prefix, gold], [1, len(prefix) + 1], labels=[-100] * len(prefix) + [gold],
    contrasts=[(len(prefix), gold, [rival])], mix=1.0,
)
next_model, step = model.step_geometry_gradient_at_distance(g, target_l2=1 / 256)
```

Let `L` be the L2 norm across **all trainable geometry coordinates together**.
For a positive distance `d`, each trainable f32 value becomes
`f32(f64(value) + d * (-f64(gradient) / L))`. Frozen coordinates, base and
LoRA stay unchanged. The returned receipt measures the actual rounded update.
Its gradient dot product is a first-order prediction, not a new loss evaluation.
A zero distance still validates gradient ownership/policy; a zero gradient
cannot supply a positive step. If f32 cannot represent the requested length
or direction to relative tolerance 1e-4, the call fails without a new model.

Use `step_geometry_sgd_at_distance` instead if the experiment is explicitly
about scaling the **already rounded** displacement of a particular SGD step.
It is intentional that a tiny learning rate can erase this direction even
when the direct-gradient operation succeeds. Neither method silently falls
back to the other.

## Source-built equivalence

`check_source_step.py` accepts only the pinned **artificial** two-radius
rehearsal. It reconstructs the native toy decoder and repeats its eight
training transitions without scoring any assessment inputs. The saved
training wheel and the rebuilt CPU wheel both returned the exact recorded
gate/bias gradient bytes, resulting artifact bytes, and every receipt field
at all eight steps. The two check streams were byte-identical
(`736f275863b2447427758de0bee347b09247d5bfc1f4657162cfd8a0f5388a6a`).
All 299 Rust causal-LM and geometry-artifact tests, and the installed binding's
63 synthetic causal-LM tests, passed. Coverage includes a two-site global
norm and bit-preservation of frozen parameters.

```sh
cargo test --release --locked -p st-nn --lib gradient_ray_recovers
"$NEW_PYTHON" -IB bindings/st-py/tests/test_native_causal_lm.py
# From this directory; use the extension hash from the built wheel's member.
python3 -ISB ../parameter-vjp/bounded.py "$FRESH_OUTPUT" 60 "$NEW_PYTHON" -IB check_source_step.py "$TOY" "$EXTENSION_SHA256" "$FRESH_OUTPUT"
```

The [real study's frozen protocol](README.md) still requires the original
training extension and its explicit readout-storage selector. This restores
the direct update, not every storage backend in that wheel. No pretrained
updates or predictions are implied by the synthetic comparison.
