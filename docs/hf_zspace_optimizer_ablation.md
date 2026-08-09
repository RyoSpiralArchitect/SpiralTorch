# HF Z-Space Optimizer Factorized Ablation

This workflow asks a narrow question: when Z-Space changes a Hugging Face
Trainer learning-rate trajectory, is an observed loss difference caused by the
integrated learning-rate dose, by the time-varying shape, or by both?

It does not claim that three seeds or one short fine-tune establish general
model quality. It produces a matched, replayable diagnostic before a larger
study is justified.

## Semantic boundary

`st-core::runtime::zspace_optimizer` owns the trajectory contract. Given the
Rust-derived raw scales `r_i` and scheduler-owned nominal rates, it sets
`w_i = sum_g(lr_i,g)` across optimizer parameter groups and emits:

| Arm | Applied scale | Purpose |
| --- | --- | --- |
| `observe` | `1` | Baseline update plus calibration only |
| `dose_matched_constant` | `sum(w_i r_i) / sum(w_i)` | Raw integrated dose without its shape |
| `raw` | `r_i` | Original Z-Space control, dose and shape together |
| `dose_normalized` | `clamp(c r_i, min, max)` | Raw shape rescaled to baseline dose |

Rust solves `c` so the bounded normalized schedule satisfies
`sum(w_i clamp(c r_i, min, max)) = sum(w_i)`. The report includes the complete
step table, saturation counts, invariant residuals, expected non-identity update
counts, identity tolerances, and a SHA-256 trajectory identity. Python transports
the report, checks each live scheduler row and measured intervention count
against it, temporarily actuates `torch.optim.Optimizer.step`, and restores
nominal rates before the scheduler advances. A constant or dose-normalized arm
may legitimately be an identity control; the raw arm must still contain at least
one non-identity update for the factorized comparison to be ready.

If another Trainer callback stops a run early, SpiralTorch seals the realized
schedule instead of turning successful training into an exception. Generated
trajectories use the realized update count; a partially consumed calibrated
trajectory remains explicit `blocked` evidence and cannot enter the comparator.

This dose is the sum of parameter-group learning rates over optimizer updates.
It is not parameter-count weighted and does not claim to measure gradient norm,
parameter displacement, or useful learning by itself.

## Run one matched seed

Run `observe` first. Keep every non-intervention argument identical across all
four runs, including model and tokenizer identity, corpus, selection, seed,
batching, optimizer, scheduler, precision, and step horizon.

The following abbreviated example assumes those shared arguments are in a
shell array. Use explicit values suitable for the model under test.

```bash
COMMON=(
  --model-name /path/to/local-model
  --tokenizer-name /path/to/local-model
  --train --train-file data/corpus.txt
  --finetune-mode lora --lora-rank 4 --lora-alpha 8
  --max-steps 16 --learning-rate 0.00005
  --eval-before-train --eval-after-train-policy always
  --seed 13
)

spiral-hf-finetune "${COMMON[@]}" \
  --output-dir runs/s13-observe \
  --run-card runs/s13-observe/run-card.json \
  --zspace-optimizer-control observe

TRAJECTORY=runs/s13-observe/spiraltorch-hf-zspace-optimizer-trajectory.json

spiral-hf-finetune "${COMMON[@]}" \
  --output-dir runs/s13-constant \
  --run-card runs/s13-constant/run-card.json \
  --zspace-optimizer-control apply \
  --zspace-optimizer-trajectory-arm dose_matched_constant \
  --zspace-optimizer-trajectory-json "$TRAJECTORY"

spiral-hf-finetune "${COMMON[@]}" \
  --output-dir runs/s13-raw \
  --run-card runs/s13-raw/run-card.json \
  --zspace-optimizer-control apply \
  --zspace-optimizer-trajectory-arm raw \
  --zspace-optimizer-trajectory-json "$TRAJECTORY"

spiral-hf-finetune "${COMMON[@]}" \
  --output-dir runs/s13-normalized \
  --run-card runs/s13-normalized/run-card.json \
  --zspace-optimizer-control apply \
  --zspace-optimizer-trajectory-arm dose_normalized \
  --zspace-optimizer-trajectory-json "$TRAJECTORY"
```

Calibrate independently for every seed. Do not reuse an `observe` artifact
after changing the scheduler, horizon, parameter groups, or control recipe.

## Compare run cards

```bash
spiral-hf-zspace-optimizer-factorized-compare \
  runs/s13-observe/run-card.json \
  runs/s13-constant/run-card.json \
  runs/s13-raw/run-card.json \
  runs/s13-normalized/run-card.json \
  --out runs/s13-factorized.json
```

Pass all four run cards for every seed in one invocation. Three matched seeds
allow the report to describe a directionally consistent bounded trend, not
statistical significance. Even a consistent improvement is reported separately
from `efficacy_claim_ready`, which remains false until a prespecified, powered,
multi-model evaluation exists.

## Run a resumable multi-seed study

The installed study runner generates the four commands for every seed, runs
`observe` before its three calibrated arms, and invokes the same comparator at
the end. First omit `--run` to create a recovery anchor without training:

```bash
spiral-hf-zspace-optimizer-factorized-study \
  --study-dir models/runs/zspace-factorized-64 \
  --seed 13 --seed 17 --seed 23 \
  --min-free-disk-gb 5 \
  -- \
  --model-name /path/to/local-model \
  --tokenizer-name /path/to/local-model \
  --train --train-file data/corpus.txt \
  --finetune-mode lora --lora-rank 4 --lora-alpha 8 \
  --max-steps 64 --learning-rate 0.00005 \
  --eval-before-train --eval-after-train-policy always
```

Inspect `study-plan.json`, then repeat the identical command with `--run` before
the separator. The plan is immutable inside one study directory and contains a
SHA-256 study identity over the scientific arguments, bridge content, package
source/native-extension fingerprint, available Git head/status, seeds, and arm
order. If the study directory is inside the repository, that generated subtree
is explicitly excluded from Git status so writing the plan does not change its
own recovery identity.

Each child completion is accepted only after the runner verifies its launch
command, seed and arm, before/after eval losses, complete optimizer horizon,
Rust trajectory identity, trainer and optimizer trace hashes, output directory,
and path-independent execution/runtime/input identities. Those identities must
also remain constant across every seed. The append-only `study-events.jsonl`
journal is fsynced before and after child execution, so a restart can recover a
child that finished immediately before the parent stopped without silently
adopting unrelated artifacts. `study-summary.json` records live progress and
`factorized-report.json` is written only after every verified arm is present.

If a failed attempt left artifacts that cannot be verified, the study fails
closed. `--retry-failed` preserves them under `quarantine/` before launching a
new attempt; it never overwrites them in place.

## Compare control gains

Run otherwise identical studies with an explicit
`--zspace-optimizer-control-gain`, then compare the completed study directories:

```bash
spiral-hf-zspace-optimizer-factorized-gain-compare \
  models/runs/zspace-factorized-gain-025 \
  models/runs/zspace-factorized-gain-050 \
  models/runs/zspace-factorized-gain-100 \
  --out models/runs/zspace-factorized-gain-response.json
```

The gain comparator verifies each plan identity, hash-chained completion
journal, factorized-report SHA-256, non-gain scientific arguments, seed set,
execution/runtime/input anchor, and exact observe losses. It rejects fewer than
three gains. The output reports per-contrast ordinary least-squares slope,
intercept, and `R²`; this is a descriptive response curve, not a significance
test.

### Audited 64-step result (2026-08-09)

The checked-in [gain-response artifact](benchmarks/hf_zspace_optimizer_gain_response_64step_20260809.json)
records one local GPT-2 LoRA run family on
`models/samples/spiral_corpus_en/06_spiral_longform.txt`: 64 optimizer updates,
seeds 13/17/23, CPU float32, and gains 0.25/0.5/1.0. Observe before/after losses
were exactly reproduced across all three studies.

| Contrast (left minus right) | gain 0.25 | gain 0.5 | gain 1.0 | `R²` |
| --- | ---: | ---: | ---: | ---: |
| raw minus observe | +0.000943 | +0.001855 | +0.003599 | 0.99988 |
| constant-dose minus observe | +0.000737 | +0.001457 | +0.002862 | 0.99996 |
| dose-normalized minus observe | +0.000217 | +0.000442 | +0.000902 | 0.99998 |
| raw minus constant-dose | +0.000206 | +0.000398 | +0.000737 | 0.99903 |

Lower validation loss is better, so every positive value favors the right arm.
For this single recipe the current progress-only, non-feedback control therefore
produced gain-correlated loss degradation. This is useful actuator diagnosis,
not evidence that Z-Space controls are generally harmful. It argues against
making this open-loop policy a default and motivates a separately ablated,
Rust-owned feedback gate before longer scale-up.

That gate is now available as the explicit
`--zspace-optimizer-feedback loss_guard` mode. It consumes the selected raw or
trajectory-arm proposal but keeps all loss projection, EMA, warmup, staleness,
halt/recovery state, and the final blend in
`st-core::runtime::zspace_optimizer_feedback`. Python and WASM expose the same
state-machine checkpoint/report contract; the HF receipt additionally seals
its observation and control lineage. It is deliberately not a fifth arm in
this frozen four-arm study: compare guarded and unguarded recipes in a new
matched multi-seed run, and retain the receipt's
`within_run_loss_guard_not_counterfactual_efficacy` boundary.

The frozen four-arm comparator fails closed unless each seed has exactly one
arm of every kind and all four share:

- the non-intervention training recipe and before-train loss anchor;
- training input, materialized dataset, tokenized blocks, model, and execution
  identities;
- one Rust trajectory, raw control sequence, and nominal scheduler sequence;
- complete optimizer actuation/restoration counts and a sealed actuation hash;
- measured nominal and effective-LR doses matching the selected Rust arm.

## Run the guarded feedback study

The feedback study freezes one Rust raw-control trajectory per seed and executes
three arms in this order:

| Arm | Raw trajectory | Loss guard | Question |
| --- | --- | --- | --- |
| `observe` | recorded, not applied | off | What does ordinary FT do? |
| `raw_unguarded` | applied | off | What is the total open-loop effect? |
| `raw_loss_guard` | applied | Rust `loss_guard` | Does the guard improve that same intervention? |

Create the immutable recovery anchor before training by omitting `--run`:

```bash
spiral-hf-zspace-optimizer-feedback-study \
  --study-dir models/runs/zspace-feedback-64 \
  --seed 13 --seed 17 --seed 23 \
  --min-free-disk-gb 5 \
  -- \
  --model-name /path/to/local-model \
  --tokenizer-name /path/to/local-model \
  --train --train-file data/corpus.txt \
  --validation-fraction 0.1 \
  --finetune-mode lora --lora-rank 4 --lora-alpha 8 \
  --max-steps 64 --learning-rate 0.00005 \
  --model-train-dtype float32 --training-use-cpu \
  --eval-before-train --eval-after-train-policy always
```

Inspect `feedback-study-plan.json`, then repeat the exact command with `--run`
before the separator. The runner owns `--seed`, output/run-card/trace paths,
trajectory transport, `--logging-steps 1`, feedback mode, and every feedback
parameter. Optional `--feedback-config-json config.json` accepts only a JSON
object; Rust validates the overrides and the plan seals the complete resolved
13-field config, contract owner, native-extension hash, and evidence boundary.

The final `feedback-report.json` exposes three lower-is-better contrasts:

- `unguarded_total_effect`: `raw_unguarded - observe`;
- `guarded_total_effect`: `raw_loss_guard - observe`;
- `guard_benefit`: `raw_loss_guard - raw_unguarded`.

The comparator additionally requires complete per-update loss/control lineage,
fail-closed warmup, no stale observations, a gate that actually became active,
and a guarded schedule distinct from the unguarded schedule. A short calibration
where the gate never actuates is preserved as useful `blocked` evidence rather
than being promoted into a vacuous comparison. Three consistent seeds permit a
bounded single-model, single-corpus trend only; `efficacy_claim_ready` remains
false by construction.

## Read the contrasts

All contrasts use `left_arm loss change - right_arm loss change`; lower is
better.

- `dose_effect`: constant-dose arm minus observe.
- `shape_effect_at_raw_dose`: raw minus constant-dose.
- `dose_normalized_shape_effect`: normalized-shape minus observe.
- `raw_total_effect`: raw minus observe.

A positive `dose_normalized_shape_effect` means this particular normalized
shape was worse than its matched baseline. It does not establish that Z-Space,
other controls, longer horizons, or other models are worse. Treat it as a
calibrated signal for the next control design.

## Direct Python API

The trajectory operation is available without Trainer orchestration:

```python
import spiraltorch as st

trajectory = st.zspace_parameter_trajectory(
    raw_learning_rate_scales=[0.8, 1.1, 0.7],
    nominal_learning_rates=[[5e-5], [3e-5], [1e-5]],
)
assert trajectory["semantic_backend"] == "rust"

validated = st.validate_zspace_parameter_trajectory(trajectory)
assert validated["trajectory_id"] == trajectory["trajectory_id"]
```

Changing any serialized report field causes Rust validation to reject the
artifact rather than accepting Python-reconstructed semantics.

## Resume compatibility

New checkpoints use optimizer-state contract v2 and retain every scheduler row
needed to reproduce the trajectory. A raw-control v1 checkpoint still resumes,
but v1 did not store historical nominal/effective LR rows. SpiralTorch records
that missing prefix explicitly, preserves optimizer actuation, and leaves the
trajectory and integrated-dose receipt fields unavailable instead of inventing
history. Such a migrated run cannot enter the four-arm comparator.
