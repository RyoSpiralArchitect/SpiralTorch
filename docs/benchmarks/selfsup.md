# Self-Supervised Benchmarks

This note tracks the expected ranges and reproduction steps for the new self-supervised
objectives exposed by `spiral-selfsup`. The InfoNCE instrumentation is wired into the shared
`st_metrics::registry`, making the same metric descriptors available to Rust callers and any
external observability bridge.

## Core metrics

The registry exposes the following metrics for each [`InfoNCEResult`](../../crates/spiral-selfsup/src/contrastive.rs):

| Metric key | Description | Direction | Typical range |
| ---------- | ----------- | --------- | ------------- |
| `selfsup.info_nce.loss` | Batch-averaged InfoNCE loss. | Lower is better. | 0.4 – 1.0 |
| `selfsup.info_nce.top1_accuracy` | Share of anchors whose positives win the top-1 logit. | Higher is better. | 0.65 – 0.95 |
| `selfsup.info_nce.margin` | Margin between positive and hardest negative logit. | Higher is better. | 0.8 – 1.6 |
| `selfsup.info_nce.positive_log_prob` | Mean log probability of the positives (nats). | Higher is better. | −0.8 – −0.1 |
| `selfsup.info_nce.positive_logits` | Distribution of positive logits for histogramming. | Diagnostic. | — |
| `selfsup.info_nce.negative_logits` | Distribution of negative logits for histogramming. | Diagnostic. | — |

When running the example training loop these metrics are emitted both via the `MonitoringHub`
extra fields and via the tensorboard exporter, so either the CLI or TensorBoard can be used for
inspection.

## Reproduction recipe

1. Ensure the workspace has been formatted and dependencies are ready.
2. Launch the synthetic contrastive training loop:

   ```bash
   just selfsup-train
   ```

   The binary seeds a deterministic RNG and writes TensorBoard logs under `runs/selfsup` by
   default. Set `SELF_SUP_LOGDIR=/path/to/logdir` to override the location.

3. (Optional) Run the dedicated unit tests and doctests:

   ```bash
   just selfsup-eval
   ```

   This exercises the InfoNCE metric summaries alongside the existing objective tests.

4. Point TensorBoard at the chosen log directory to visualise `selfsup/*` and `chrono/*`
   series. The logged `ChronoSummary` traces are refreshed every iteration so drift and energy
   stability are visible alongside the contrastive scores.

## Regression expectations

On the synthetic workload shipped with the example the following aggregates should stabilise
within ~20 steps:

- `selfsup.info_nce.loss` converges near `0.55 ± 0.05`.
- `selfsup.info_nce.top1_accuracy` sits between `0.80` and `0.9` once the noise floor settles.
- `selfsup.info_nce.margin` remains above `1.1` while negatives cluster tightly.

Significant excursions (e.g. loss > 1.2 or margin < 0.5 for sustained periods) usually
indicate an augmentation bug or a regression in the similarity head. Capture the exported
TensorBoard run and attach it to any investigation so the full distribution histograms are
available.
