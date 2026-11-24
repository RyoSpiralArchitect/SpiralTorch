# Full CE and a declared choice from one decoder forward

An update that improves a verb's odds against a named rival can lower its
probability in the full vocabulary. The [archived training anchors](TRAINING_ANCHORS.md)
contain one such conflict in the gradients, not a measured post-step loss.
`NativeCausalLm.geometry_full_and_declared_gradients` exposes both native
geometry gradients on the **same** logits. Python does not implement an
objective, a backward pass or an optimizer.

```python
# model already has trainable geometry. Only the last label is supervised.
ids = prefix + [gold]
full, choice = model.geometry_full_and_declared_gradients(
    ids, [1, len(ids)], labels=[-100] * len(prefix) + [gold],
    contrasts=[(len(prefix), gold, [rival])], mix=1.0,
)
proposal, _ = model.step_geometry_gradient_at_distance(choice, 1 / 256)
next_model, receipt = model.guard_geometry_step(proposal, [[choice], [full]])
```

The first handle is ordinary full-vocabulary CE. The second uses the declared
mixture on named targets and ordinary CE on other supervised targets. Positions
are unshifted in each flattened batch; the first column has no causal predictor.
Each handle belongs to the original model's base, LoRA, geometry and readout
precision. Attached LoRA participates in the forward but is not updated. A
rejected guard returns an error, not the unsafe proposal. An accepted guard
checks the loss-gradient dots on the rounded parameter displacement: these
are **first-order** checks, not guarantees on finite-step or held-out losses.

The Rust Tensor kernel projects the full CE seed **before** it blends the
declared rows in place. The decoder shares checkpoints, the vocabulary
forward and each reverse block replay. The two vocabulary pullbacks and
parameter VJPs are still distinct. No second vocabulary-sized seed needs to
be kept alive. Two hidden adjoints and two gradient groups do have to fit.
No runtime speedup is asserted from counts alone.

## CPU evidence

- On a two-layer artificial decoder, counters on the actual block and head
  boundaries gave 2 vs 4 forward blocks, 1 vs 2 vocabulary forwards, and
  1 vs 2 (last-layer site) or 2 vs 4 (first-layer site) block replays.
  The per-objective losses, reports, parameter bits and owners matched the
  separate APIs across all three readout precision modes, masked and dense
  labels, a frozen LoRA, and geometry at multiple placements.
- Eight shared-objective geometry steps on the pinned artificial R=1/R=4
  trajectories matched eight independent reference steps: all 16 parameter
  gradients, step receipts and resulting artifact bytes agreed. The finite
  choice and full losses on the same training prefix both fell on all eight
  steps. **None required repair.** A separate three-class fixture was built
  to force conflict: the unguarded step improved choice and worsened full CE.
  With an independent bias direction the guard repaired it and both measured
  losses fell; with exactly opposed gradients it refused the step.
- The installed CPU wheel passed 67 native-LM Python tests. Rust passed 307
  selected LM/geometry tests and 4 declared-CE Tensor tests. A separate
  legacy replay matched all 19 raw numerical artifacts (8 updates,
  16 predictions) from the original artificial run.

Reproduce the shared and legacy trajectory checks with the source-built CPU
wheel installed and the original **artificial** plan. Both demand fresh
output directories. Neither accepts a pretrained-model path.

```sh
"$VENV/bin/python" -I -B check_shared_choice.py "$SEALED_TOY" "$EXTENSION_SHA256" "$NEW_SHARED_OUTPUT"
"$VENV/bin/python" -I -B check_source_storage.py "$SEALED_TOY" "$EXTENSION_SHA256" "$NEW_LEGACY_OUTPUT"
```

Shared journal SHA-256: `183c4be26d59209f2374ed3ebbfc9a56d5f10a10668164c7e059949d5b63de9c`.
The pinned pretrained comparison and its wheel are unchanged. No new GPT-2
calls, held-out scores, GPU runs or release are part of this check.

For a mechanism check using only archived training-site adjoints, see
[the paired local geometry comparison](objective-metric/RESULTS.md).
