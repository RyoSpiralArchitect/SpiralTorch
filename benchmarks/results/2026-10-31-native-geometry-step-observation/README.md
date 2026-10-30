# Where the matched native book trajectories diverge

The [original study](../2026-10-31-native-book-ablation/README.md) established
an endpoint on released 0.4.33. This is a retrospective localization exercise
using a separate development wheel, not another release, new test corpus or
hyperparameter search. See [plan.json](plan.json) for the prospective replay
rules. Original model, code, data-order and case pins are not replaced.

Every update uses Rust `NativeCausalLmTrainer::step_observed` on the actual
training microbatch. Its ordinary nested report must match the original
published-wheel step bytes, including pre/post artifact IDs and loss. Each
final checkpoint must be byte-identical. The six development chapters are
scored at cursors 0, 1, 8, 32, 128 and 256 with 3,024 targets each; at cursors
0 and 256 every chapter score must reproduce the original. No external-book
score, reserve data, GPU job or base-weight update is used.

Pre-update residuals are summarized in fixed update bands: 1, 2-8, 9-32,
33-128, 129-256. For each site/seed/band pool the squared effective residual
and input across all observed elements, take the square-root ratio, then
average these ratios across the three seeds. Radius gradients are with respect
to **log radius** on the actual token-weighted objective before clipping,
not weight changes or gradients of the validation loss. Fixed/absent radius
gradients are null. Exact individual radius values, raw traces, checkpoints
and tokens stay private. These are matched batches *within* each update, not
identical batches *between* updates, and a correlation is not a mediation test.

## Reproduce

Use the original private reader-verifiable corpus/fit directory, local pinned
GPT-2 files and a fresh isolated venv containing the development wheel built
from the native source in the plan. Never publish that wheel as 0.4.33.
The original released wheel alone does not expose the observer. Supply
independently recorded SHA-256 pins and a new private output path:

```sh
python3 -I -S -B test_analysis.py -v
python3 -I -S -B replay.py --python "$DEV_VENV/bin/python" --wheel "$DEV_WHEEL" \
  --wheel-pin "$WHEEL_SHA" --plan-pin "$PLAN_SHA" --model "$MODEL" \
  --original "$ORIGINAL" --output "$OUTPUT"
python3 -I -S -B analysis.py "$OUTPUT" "$ORIGINAL"
```

One local CPU process per arm/seed, in original order, with the original four
Rayon threads. The supervisor stops on the first mismatch/error or deadline;
partial output is not a passed trajectory. Analysis of saved files never
reruns learning. Read the aggregate results separately from runtime or broader
model-quality claims.
