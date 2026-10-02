# Chart-Step Training Study

**Status: completed and verified.** All twelve runs finished 512 updates and
exact next-update continuation checks before endpoint evaluation. The mean chart
correction changes the optimizer direction substantially, but does **not** close
the elliptic map's gap to the ordinary tangent control. It is not promoted to
the default optimizer. No geometric quality or throughput advantage is claimed.

The [Rust operator and protocol](../../../docs/elliptic_chart_step.md) compares
ordinary anchored tangent and native anchored elliptic, each with ordinary Adam
and chart-corrected Adam proposals. All four arms have 8451 parameters and paired
initialization/minibatches. Fixed recipe: three seeds, 512 updates per run,
learning rate 0.001, relative damping 0.1, CPU f32, frozen local GPT-2. Total
primary updates: 6144, plus two continuation-only updates per run.

Rust owns the mean chart metric and the norm-preserving correction. Adam consumes
unchanged raw gradients. The correction changes only the orientation projection
proposal, not the gate/readout updates. Input covariance and downstream readout/
loss curvature are omitted; this is not a full natural-gradient optimizer.

## Results

Mean next-token cross-entropy across seeds 41, 43 and 47; lower is better.
Evaluation uses 120 unused within-book Pride blocks and 32 Alice transfer blocks.
All arms improve over the frozen base, but the tangent map remains better than
elliptic in every seed, both with ordinary Adam and with chart correction.

| Arm | Pride CE | Alice CE |
| --- | ---: | ---: |
| Frozen base | 4.073713 | 4.011807 |
| Adam tangent | 3.563489 | 3.673359 |
| Adam elliptic | 3.679653 | 3.738315 |
| Chart tangent | 3.564011 | 3.672017 |
| Chart elliptic | 3.680922 | 3.735024 |

The primary contrast is chart elliptic minus Adam elliptic. Negative favors
chart correction. Preserve both losing seeds rather than treating the Alice
mean alone as a consistent improvement:

| Seed | Pride CE difference | Alice CE difference |
| --- | ---: | ---: |
| 41 | +0.005085 | +0.006071 |
| 43 | -0.001830 | +0.000807 |
| 47 | +0.000554 | -0.016749 |
| Mean | +0.001270 | -0.003290 |

Only one seed improves on each endpoint, and they are different seeds. The
paired sample standard deviations are 0.003512 and 0.011949 respectively. Three
seeds with shared evaluation blocks do not support a significance claim.

The tangent correction changes mean CE by +0.000522 on Pride and -0.001341 on
Alice. The paired interaction, subtracting this ordinary-control effect from
the elliptic effect, is +0.000748 and -0.001949. Every per-seed contrast, including
both geometry gaps, appears in `summary.json`.

## Did the Correction Execute?

Each corrected arm makes 511 nonzero orientation proposals. The first step is
zero because the identity-initialized readout initially blocks its gradient.
Mean cosine with the original Adam proposal is 0.531-0.561 for elliptic versus
0.9968 for tangent. This is not a no-op, despite the limited endpoint effect.
The largest relative difference between the proposed norm and the actual f32
parameter displacement is 1.942e-6. Mean damped chart condition is 19.53-20.17
for elliptic versus 1.1762 for tangent.

These are descriptive update diagnostics, not proof that chart anisotropy
causes the quality gap. A mean two-coordinate feature metric still omits input
covariance and downstream loss curvature. The experiment supports neither
silently replacing Adam nor claiming that this correction fixes LM learning.

## Verification

The original process exited 0 after all 6144 primary and 24 continuation-only
updates. All saved adapter/Adam tensors are finite; checkpoint hashes, cursor,
recipe, paired initialization, update history and endpoint receipts agree.
All twelve runs preserve the frozen base and reproduce the next adapter/Adam
update exactly. Completed `--resume` exits 0 without reevaluating endpoints or
changing the plan, journal or result bytes.

All six ordinary-Adam controls reproduce the prior anchored study exactly:
adapter tensors, Adam moments, update/development history and per-block endpoint
losses. Their new optimizer receipt/recipe fields are excluded explicitly from
the old-format comparisons. These are historical replays, not six independent
replications.

`plan.json.gz` and `preflight.json` are the original, unchanged launch records.
The latter still says "running" because it is a historical preflight, not the
completion receipt. It records 16 Rust and 204 Python tests, strict TypeScript,
native builds and actual Node-hosted WASM updates. The main-integrated checkout
passes 209 postrun Python tests with no skips. `checkpoint-verification.json`
records saved-content checks; `validation.json` records process termination,
resume immutability, runtime/source/log hashes and public summary reproduction.

Training source: `76ccee3649bcadd407539de6e14b832cc1e480e0`.
Summary source: `48199ca5163cc1a3ced8662f2f82c6a429e1b328`.
The Python/native package and transitive client files are frozen separately
locally; changing the checkout does not change the running experiment.
Models, corpus text, checkpoints and raw logs remain private/local. Only numeric
outcomes and hash-bound verification are published here. The two earlier deleted
large benchmark archives are not restored or included in this change.

## Reproduction

Use the pinned source/runtime described in `preflight.json`, build/install its
native Python binding, and provide the same local model/corpora:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  python bindings/st-py/examples/hf_elliptic_chart_step_study.py \
  --config bindings/st-py/examples/hf_elliptic_pride_chart_step.json \
  --model-dir "$LOCAL_GPT2" --corpus "$LOCAL_PRIDE" \
  --transfer-corpus "$LOCAL_ALICE" --output-dir "$NEW_STUDY_DIRECTORY"
```

Resume only with the same frozen code/native package and `--resume`. Do not
overwrite or regenerate the launch plan. The study locks endpoint evaluation
until all runs finish and their next updates reproduce exactly. Development
scores do not select settings or checkpoints. Pride/Alice were inspected in
prior experiments, so this is exploratory rather than untouched confirmation.

The read-only summarizer reports all five factorial contrasts, every losing
seed, gate trajectories and per-run proposal norms/direction cosines. Published
compressed inputs reproduce `summary.json` byte for byte without model access:

```sh
python tools/summarize_wave_gate_long_horizon.py \
  --plan benchmarks/results/2026-10-03-elliptic-chart-step-study/plan.json.gz \
  --results benchmarks/results/2026-10-03-elliptic-chart-step-study/results.json.gz \
  --journal benchmarks/results/2026-10-03-elliptic-chart-step-study/journal.json \
  --output "$NEW_SUMMARY_JSON"
```

Do not use the summarizer on a partial run. Checksums bind the publication's
files; saved-state and execution verification are separate evidence above.
