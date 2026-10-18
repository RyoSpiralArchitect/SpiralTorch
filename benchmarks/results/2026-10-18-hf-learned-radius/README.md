# Learned Radius Through Actual HF Fine-Tuning

This declared follow-up uses the existing Rust `WaveGateKernel` radius VJP,
portable geometry artifacts and public HF Trainer. It introduces no alternate
geometry formula, optimizer loop or adaptive controller. The motivating
[saved-state observations](../2026-10-18-hf-geometry-observation/README.md)
showed different residual magnitudes and local contraction at fixed radii 1/4,
but did not establish the cause of the ordinary affine control's better CE.

## Fixed Treatment

All original book data/window/model/training settings stay unchanged. The four
arms are fixed radius 1, learned radius initialized at 1, fixed radius 4 and
learned radius initialized at 4, with seeds 41/43/47 and 512 updates. Within each
fixed/learned pair, only four existing `log_radius` buffers become parameters.
The same HF AdamW/scheduler/clipping rule then owns them. Initial outputs and
common gate/bias gradients must match exactly; radius gradients start at zero.

There are 6,144 trainable values in a fixed arm and **6,148** in a learned arm.
This is not equal-parameter-budget or matched-intervention-energy training.
It tests the existing radius-learning treatment as a whole, including its
interaction with optimizer grouping and global gradient clipping, not a causal
isolation of hyperbolic geometry. No radius-specific learning rate is introduced.

All six fixed-control replays must reproduce the previous book's exact final
artifact IDs and CE under the qualified installed wheel. After all twelve full
runs, two preselected learned-radius seed-41 runs resume from 256 to 512 in fresh
processes. They must match uninterrupted final weights, optimizer/scheduler/MPS
RNG tensor states and HF RNG bytes. These are controlled checkpoint resumes,
not power-loss or partially-written-bundle recovery tests.

The original six reserve chapters remain unopened. No data acquisition, best-
checkpoint selection, adaptive tuning, silent retries or deployment promotion.
All cases, including failures/not-started cases, remain in the retained manifest.
Full learned tensors and per-site learned radius values stay private; only
aggregate radius-change statistics and artifact hashes are published.

## Replay

Use the separately qualified observation environment and original private book
root, retaining `chapters-balanced`, `preflight-bound` and `runs-fixed`. The new
runner verifies the complete installed Python/native inventory against that
study's pinned execution record before training. A new output directory is
mandatory. No wheel build or PyPI release is required by this experiment.

```sh
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PYTHON" -I -B benchmarks/results/2026-10-18-hf-learned-radius/replay.py \
  --expected-plan-id "$INDEPENDENTLY_RETAINED_PLAN_ID" \
  --book-private "$PRIVATE_BOOK_ROOT" --model-dir "$PINNED_LOCAL_MODEL" \
  --output "$NEW_PRIVATE_OUTPUT"
```

The runner owns each public-example child with a 900-second deadline. Use the
existing `tools/owned_process.py` outer supervisor with a 3,600-second deadline
and five-second cleanup grace. Total declared updates, counting the two replayed
tails, are 6,656. Generation from the inherited public example stays private.
The finite corpus may have appeared in GPT-2 pretraining; no unseen-data or
general-language-quality claim follows from this developmental comparison.
