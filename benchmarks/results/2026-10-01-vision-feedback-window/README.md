# Rust Loss Windows: Quieter Frozen Gate, No Learning Advantage

The optional 80-observation window removes gate activity on these six frozen
models, but **does not demonstrate a learning advantage over nominal SGD or
the matched controls**. Synthetic abrupt-regression detection is
delayed by 39 to 80 observations in the three tested positions. Keep the default
width at one; the wider window is an explicit experiment, not a recommended default.

## Implementation And Frozen Observations

Source capture `dc957fdbea85181cf0058da3ce6a2bad9bc9f13a` adds non-overlapping,
equal-weight loss means in `st-core`. Window width, partial count/mean and the
previous completed mean are checkpointed. Python/WASM only transport this state.
Accepted losses update raw telemetry every step; the gate changes only at a
completed-window boundary. This does not remove per-step scalar readbacks.

The [earlier frozen-model probe](../2026-10-01-vision-feedback-stationarity/README.md)
recorded 400 losses for each initial/final model at seeds 17, 29 and 43. The new
native binary reproduces all 7,200 recorded default shadow observations exactly,
including reversed-order and constant-loss controls. Replaying the same losses
with width 80 yields no halt actions or nonidentity controls in any of those
streams. The recorded-order default produced 67/77/83/92/71/113 halt actions.
The data are retained inference observations, not newly measured model outputs.

The window starts with a closed gate; stable population means leave it closed.
There are only five completed means and four comparisons per 400 observations.
The two-order comparison cannot establish invariance for arbitrary permutations
that change window membership. See [all cases and latency probes](frozen-summary.json).

In a separate synthetic loss sequence, an open gate sees an abrupt change from
2 to 8 after an earlier improvement from 4 to 2:

| Position inside the 80-observation window | Default additional observations | Window additional observations |
| --- | ---: | ---: |
| First observation | 0 | 79 |
| Observation 41 | 0 | 39 |
| Last observation | 0 | 80 |

The last case also reflects the retained relative-delta EMA. These are observed
delays, not a general worst-case bound. All partial-state restarts at observation
37 are exact. A separate [actual wasm32 replay](frozen-wasm-verification.json)
matches all 14,400 default/windowed states across 36 streams, restoring at 37.
This is not browser GPU execution or browser real-image learning.

## Matched Real-Image Learning

The same three seeds, 1,280 CIFAR-10 training images, 320 development images,
batch 16 and five epochs are used with the existing 5,530-parameter ConvNeXt.
The four arms remain nominal SGD, a fixed half-rate proposal, windowed feedback,
and a retrospective constant rate matched to the new feedback trajectory.
The model, window, gate, sampler and parameter update remain Rust-owned.
PyTorch MPS independently computes model gradients under the Rust-selected
inputs/rates; it is not an independent implementation of the feedback policy.

Final development accuracy (%):

| Seed | SGD | Fixed half-rate | Previous default gate | Window 80 | New dose control |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 | 21.5625 | 19.3750 | 21.8750 | 21.8750 | 22.1875 |
| 29 | 24.0625 | 19.0625 | 24.0625 | 23.4375 | 24.0625 |
| 43 | 29.6875 | 26.5625 | 29.0625 | 29.0625 | 29.6875 |
| Mean | 25.1042 | 21.6667 | 25.0000 | 24.7917 | 25.3125 |

Mean final CE is 2.034476 for SGD, 2.037988 for the previous gate, 2.041282
for window 80 and 2.041592 for its dose control. The window has slightly lower
CE but lower accuracy than its dose control; neither comparison establishes
superiority. The small, repeatedly inspected development set is not a final test.

In all three windowed runs, the gate changes 240/400 actual rates, never halts,
and reaches 0.5 after the final observation. The mean realized rate is about
0.00925; its retrospective f32 constant-rate control differs in integrated rate
by about `1.49e-7`. A quieter gate admits more of the prescribed harmful half-rate
proposal than the old default. Better separation of batch composition is not
the same as identifying a useful proposal. No thresholds or seeds were retuned.

All 12 new arm/seed combinations finish with 400 accepted updates, no rejections,
and exact fresh-process 37/363 continuation of every record and bound checkpoint.
All epoch evaluation accuracies match PyTorch; maximum final-parameter scaled
error is `1.70e-7`. The independent [learning verification](learning-verification.json)
checks retained records, source identity, checkpoints and saved reference weights.
[Matched comparison](matched-comparison.json) also confirms that the unmodified
SGD/fixed-proposal controls preserve 4,800 phase records and 42 checkpoint files
exactly across the old and new builds. This is bounded equivalence, not a general
cross-device guarantee. Every epoch and arm is in [the summary](learning-summary.json).

## Reproduce And Boundaries

[Conditions](conditions.json) record the source, binary, dataset identities,
architecture and environment. The native wheel and WebGPU WASM use Rust 1.98;
native execution is WGPU/Metal and the reference is PyTorch MPS on Apple M4.
[Provenance](provenance.json) hashes locally retained raw records, weights,
fixtures and validation logs. Images and checkpoints are not republished.
Earlier results are unchanged. `SHA256SUMS` covers this public leaf.

```bash
python -I tools/compare_vision_feedback_windows.py \
  --stationarity "$FROZEN_PROBE_DIR" --verification "$FROZEN_VERIFICATION_JSON" \
  --output "$NEW_FROZEN_WINDOW_DIR"
node tools/verify_vision_feedback_windows_wasm.cjs "$WASM_NODE_MODULE" \
  "$NEW_FROZEN_WINDOW_DIR" "$NEW_WASM_VERIFICATION_JSON"
PYTORCH_ENABLE_MPS_FALLBACK=0 python -I tools/run_vision_feedback_ablation.py \
  --data-root "$CIFAR_ROOT" --output "$NEW_WINDOW_LEARNING_DIR" \
  --torch-device mps --seeds 17 29 43 --train-per-class 128 \
  --test-per-class 32 --batch-size 16 --epochs 5 --restart-at 37 \
  --feedback-window-observations 80
python -I tools/run_vision_feedback_ablation.py --verify "$NEW_WINDOW_LEARNING_DIR" \
  --source-ref dc957fdbea85181cf0058da3ce6a2bad9bc9f13a --output "$NEW_VERIFICATION_JSON"
```

Use the captured source/runtime for measurements. The additional WASM verifier
is included in this result's publication commit and its digest is recorded in
the replay receipt; it targets this half-rate-proposal experiment. The width follows recorded input coverage, not
accuracy selection; rejected updates would break a simple width-to-epoch mapping.
Per-step observations, sequential frameworks and checkpoint mapping exclude speed
and memory claims. Neither a model-derived proposal producer nor geometric
parameter updates are tested. The next execution gate is transfer-inclusive
throughput, not selecting a more flattering window on these development seeds.
