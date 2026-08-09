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
step table, saturation counts, invariant residuals, and a SHA-256 trajectory
identity. Python transports the report, checks each live scheduler row against
it, temporarily actuates `torch.optim.Optimizer.step`, and restores nominal
rates before the scheduler advances.

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

The comparator fails closed unless each seed has exactly one arm of every
kind and all four share:

- the non-intervention training recipe and before-train loss anchor;
- training input, materialized dataset, tokenized blocks, model, and execution
  identities;
- one Rust trajectory, raw control sequence, and nominal scheduler sequence;
- complete optimizer actuation/restoration counts and a sealed actuation hash;
- measured nominal and effective-LR doses matching the selected Rust arm.

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
