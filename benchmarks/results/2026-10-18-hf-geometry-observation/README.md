# Read-Only Geometry Observations Of Saved Book States

This post-hoc mechanism diagnostic follows the completed
[12-run book comparison](../2026-10-18-hf-book-controls/README.md). It is not a
new quality confirmation, radius search or training run. The frozen plan names
all four shared identity-start artifacts and both 256/512 states of every run:
28 states, four MLP sites, every one of the same 159 development windows. No
reserve chapter is opened. No weights, optimizer or defaults are changed.

The installed source-candidate `GeometryAdapterStack.observe()` wraps the
ordinary public `HfGeometryTrainer.evaluate()` call. Each state is evaluated
without observation and then with observation from the same RNG state. Loss
and resulting RNG must match exactly; initial/final loss must also match the
published book result. A separate check requires exact first-development-window
logits, not all-development logits. Model/adapter states must remain unchanged.

Actual residual energy is pooled as `sqrt(sum(delta_l2**2)/sum(input_l2**2))`,
not as a mean of per-batch ratios. Rust conditioning means are weighted by row
count; affine-outside counts are divided by all input elements. Native local
radial/tangential gains are not end-to-end LM gradients. Ordinary affine gets
residual measurements but no invented geometric Jacobian. Observation overhead
is deliberately excluded from any performance claim.

## Replay

Keep the original private book directory intact, with `chapters-balanced`,
`preflight-bound` and `runs-fixed`. Rebuild/install the reviewed source wheel in
a separate environment with the book study's pinned runtime versions. The
runner requires the locally cached GPT-2 snapshot used by that study.

```sh
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PYTHON" -I -B benchmarks/results/2026-10-18-hf-geometry-observation/replay.py \
  --expected-plan-id "$INDEPENDENTLY_RETAINED_PLAN_ID" \
  --book-private "$PRIVATE_BOOK_ROOT" --model-dir "$PINNED_LOCAL_MODEL" \
  --wheel "$INSTALLED_SOURCE_WHEEL" --output "$NEW_PRIVATE_OUTPUT"
```

Run with a bounded supervisor, for example the existing `tools/owned_process.py`
`run()` helper with a 1,800-second timeout. The execution record is written
before model loading/measurement and binds implementation, installed package,
wheel and all named saved-state identities. The retained case manifest includes
not-started/failed cases if anything fails; replay never silently resumes or
selects a successful subset. A rerun must use a new directory.

Raw text, tokens, activations, model/adapter weights and full per-call traces
stay local. Published case/layer aggregates and retained trace hashes are not a
replacement for those private artifacts. There is no PyPI release in this slice.
