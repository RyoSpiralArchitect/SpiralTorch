# Read-Only Geometry Observations Of Saved Book States

This post-hoc mechanism diagnostic follows the completed
[12-run book comparison](../2026-10-18-hf-book-controls/README.md). It is not a
new quality confirmation, radius search or training run. The frozen plan names
all four shared identity-start artifacts and both 256/512 states of every run:
28 states, four MLP sites, every one of the same 159 development windows. No
reserve chapter is opened. No weights, optimizer or defaults are changed.

**Completed: all 28 states, 8,960 observed evaluation adapter calls, 112 layer
aggregates.** Every case passed exact observed/unobserved CE, RNG and
first-window-logit checks, complete coverage and unchanged frozen/adapter state.
All initial/final CE values exactly reproduce the earlier book run. These checks
include zero optimizer updates and no reserve access, not a new FT/resume run.

## Results

Final-step ranges span all three seeds and four sites per arm. Residual energy
is relative to that state's actual input, expressed as a percentage; native
gain is a row-weighted mean per site, then displayed as a range across sites
and seeds. Inputs are not held identical across different trained stacks.

| Final arm | Mean development CE | Relative residual energy | Local radial gain |
| --- | ---: | ---: | ---: |
| Ordinary affine | 4.065289 | 0.8857-1.9000% | Not provided |
| Legacy WaveGate | 4.110879 | 0.0701-0.3030% | 0.07896-0.09231 |
| WaveGate radius 1 | 4.110879 | 0.0701-0.3030% | 0.07896-0.09231 |
| WaveGate radius 4 | 4.096002 | 0.2171-0.9438% | 0.40885-0.46111 |

The completed snapshot measurements show a substantially smaller residual and
stronger local radial contraction at radius 1 than radius 4. They identify a
scale/conditioning question worth testing, **not** the cause of the CE gap.
All arms have their own learned states and upstream hidden values; this is not
an equal-residual intervention, full LM Jacobian measurement, or intervention
that isolates curvature from radius. Ordinary affine remains better on CE in
every seed. No default or deployment policy is promoted. Legacy and radius-1
paths are close but not bitwise identical; full precision stays in the results.

[All results](results.json), [execution identity](execution.json),
[validation](validation.json), and [independent review](REVIEW.md) retain the
complete declared set, including intermediate states rather than selected wins.

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

The retained plan ID is
`sha256:9053c62d8c72b763fd74f7f315525d212b1f0ed63168f8260a47157a539bd3d5`.

Run with a bounded supervisor, for example the existing `tools/owned_process.py`
`run()` helper with a 1,800-second timeout. The execution record is written
before model loading/measurement and binds implementation, installed package,
wheel and all named saved-state identities. The retained case manifest includes
not-started/failed cases if anything fails; replay never silently resumes or
selects a successful subset. A rerun must use a new directory.

Raw text, tokens, activations, model/adapter weights and full per-call traces
stay local. Published case/layer aggregates and retained trace hashes are not a
replacement for those private artifacts. There is no PyPI release in this slice.
