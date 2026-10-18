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
The public projection also removes each site's exact maximum parameter change
from every run: when a scalar radius dominates, that maximum indirectly exposes
its individual update. Private source results remain byte-identical and pinned;
all cases, losses, checkpoint identities and non-maximal aggregates are retained.

## Actual Results

All twelve full runs and both fresh-process resumes completed. Development CE
is lower-is-better; the three-seed means below do not include the repeated tails.

| Arm | Seed 41 | Seed 43 | Seed 47 | Mean CE |
| --- | ---: | ---: | ---: | ---: |
| Fixed radius 1 | 4.110890388 | 4.110875607 | 4.110872269 | 4.110879421 |
| Learned radius from 1 | 4.108411312 | 4.108377934 | 4.108366013 | 4.108385086 |
| Fixed radius 4 | 4.096088409 | 4.095949650 | 4.095967770 | 4.096001943 |
| Learned radius from 4 | 4.090939999 | 4.090726376 | 4.090740681 | 4.090802352 |

Learning radius improved development CE in all six within-seed contrasts:
mean learned-minus-fixed differences were -0.002494335 (initial radius 1) and
-0.005199591 (initial radius 4). All four radii increased in every learned run;
the mean of the per-run geometric-mean radius multipliers was 1.396966 and
1.453605 respectively. Each learned checkpoint had four nonzero radius Adam
first moments. Thus the Rust radius VJP is not merely exposed: the existing HF
optimizer updated it during actual pretrained-model FT, and its state survived
the tested continuation path.

The original ordinary affine mean CE was **4.065288544**, still lower than all
these geometry means. That arm was not rerun here; all six fixed-radius arms
did exactly reproduce the earlier book's CE and final artifact IDs. This is a
small treatment effect on one developmental corpus with four extra trainable
values, not a general quality win or proof of geometry-specific superiority.

Both predetermined learned-radius resumes reproduced final export IDs and CE,
optimizer/scheduler/MPS RNG tensors and HF RNG bytes. All 26 checkpoint bundles
were retained, including the two replayed final states. A separate read-only
verification rechecks their 156 companion hashes, optimizer state, parameter
aggregates and resume equality without loading GPT-2 or performing updates.

Full-precision all-case results are in [results.json](results.json), the
before-training declaration in [execution.json](execution.json), and validation
details in [validation.json](validation.json). The deterministic projection in
[publication.py](publication.py) is rechecked against the pinned private source.
No arm, seed or failed attempt
was discarded; this study completed its declared case set without retries.

## Replay

Use the separately qualified observation environment and original private book
root, retaining `chapters-balanced`, `preflight-bound` and `runs-fixed`. The new
runner verifies the installed SpiralTorch Python/native inventory against that
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

To recheck the retained outcomes without rerunning FT:

```sh
"$PYTHON" -I -B benchmarks/results/2026-10-18-hf-learned-radius/verify_saved_state.py \
  --private-runs "$PRIVATE_RUNS" \
  --expected-results-id "$INDEPENDENTLY_RETAINED_RESULTS_ID" \
  --expected-manifest-id "$INDEPENDENTLY_RETAINED_CASE_MANIFEST_ID"
```

The verifier requires the same qualified installed SpiralTorch inventory.
`rng_state.pth` is checked as bytes, not deserialized; optimizer/scheduler/MPS
RNG companions use `torch.load(weights_only=True)`. The initial and exported
geometry records are validated by the existing Rust artifact parser.
