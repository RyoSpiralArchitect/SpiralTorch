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

## First attempt: incomplete

The first local CPU attempt ended without a supervisor status. The cause is
unknown. The [inspection](interrupted-inspection.json) certifies what was
retained, **not** a successful nine-case replay:

| Cases | Retained evidence |
| --- | --- |
| 101-affine, 101-fixed4, 101-learned4, 103-fixed4 | 256 matched ordinary steps, six development scores and a worker receipt for the byte-identical final checkpoint in each case |
| 103-learned4 | 83 contiguous decoded steps and scores at 0, 1, 8, 32; no completion receipt or intermediate checkpoint |
| 103-affine and all three seed-107 arms | Never started |

All 1,107 retained observation records matched the corresponding original step
*values* and four-site schema. The worker wrote the observation before checking
the ordinary step bytes. Decoded agreement in the interrupted prefix is not
a byte-level completion receipt. The 49 original replay files have a private
byte-count/hash inventory; no missing scores, updates or status were filled in.

Only seed 101 has a complete learned4/affine development pair. The NLL
differences (learned4 minus affine; positive favors affine) at updates
0, 1, 8, 32, 128 and 256 were **0, +0.000024, +0.000258, +0.001849,
+0.012078 and +0.028957**. The first sampled crossing of +0.001 was at 32
for this seed; the first **all-seed** crossing is **not evaluable**.
No cross-seed residual or radius-gradient projection was produced.

Separately, the original completed fits still verify: nine checkpoints and
2,304 updates. A post-hoc comparison of their **training batches**, not the
development chapters, shows a learned4-minus-affine mean NLL of +0.02021 to
+0.02081 across the three seeds in updates 129-256; learned4-minus-fixed4
was -0.00352 to -0.00333. This uses paired windows and 126 targets per update.
It neither fills the missing development curves nor identifies a causal mechanism.

Check or reproduce the inspection without loading a model, resuming training,
or touching the first attempt:

```sh
python3 -I -S -B -m unittest discover -s . -p 'test_*.py'
python3 -I -S -B inspect_interrupted.py "$OUTPUT" "$ORIGINAL" "$DEV_WHEEL" | diff - interrupted-inspection.json
```

## Original protocol (not a retry)

Use the original private reader-verifiable corpus/fit directory, local pinned
GPT-2 files and a fresh isolated venv containing the development wheel built
from the native source in the plan. Never publish that wheel as 0.4.33.
The original released wheel alone does not expose the observer. Supply
independently recorded SHA-256 pins and a new private output path. The commands
below document the frozen protocol; do not restart the interrupted first attempt:

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
