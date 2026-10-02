# Repetition Schedule Study

This is the first long-horizon comparison for the
[Rust-owned objective control](hf_repetition_objective_control.md), not a new
decoding intervention or a claim of language-quality improvement.
The [prespecified protocol](benchmarks/hf_repetition_schedule_256step_prespec_20261002.json)
must be committed before training. Its byte SHA-256 is the generation protocol
identity, also recorded in each private execution plan.

## Fixed Comparison

- Cached pretrained GPT-2, the existing complete *Pride and Prejudice* text,
  three new paired seeds (137, 139, 149), and 256 update slots per arm.
- Ordinary LoRA FT, periodic unlikelihood at constant strength 0.1, and the
  same periodic objective with linear strength decay from slot 0 to 256.
- CPU float32, rank 4 / alpha 8, learning rate 5e-5 and the same linear learning
  rate scheduler. Batch size 1 with accumulation 16 avoids unequal last-batch
  sizes; every collated row must have 127 valid shifted labels.
- Full selected held-out split, not the development smoke's two-block subset.
  The initial data-only preflight found 1,133-1,138 training blocks and 124-129
  evaluation blocks across seeds. Rows are randomly split within the same book;
  this does not establish cross-book or contiguous-document generalization.
- Twelve existing frozen prompts, greedy generation without inference controls,
  96-token limit, and checkpoints at 64/128/192/256. No best-checkpoint selection.

Only the schedule differs between treatment arms. The constant control uses the
same active-position normalization as the earlier frozen objective. Historical
MPS outcomes are not pooled with this CPU comparison. Development seed 7,
short wiring smokes and preflight generation are excluded from all study results.

## Execution

Use the isolated environment containing the tested native wheel and HF/PEFT
dependencies. Point `MODEL_SNAPSHOT` and `CORPUS` at existing local inputs; the
runner checks the pinned snapshot directory, corpus and prompt hashes. It never
downloads models or datasets.
Each child preloads the tested installed wheel before opening the source example;
an unrelated native extension in the checkout cannot shadow the study runtime.

```bash
"$PYTHON" -I tools/run_hf_repetition_schedule_study.py \
  --model "$MODEL_SNAPSHOT" --corpus "$CORPUS" --output "$OUTPUT"
```

The output directory must not exist. `--preflight-only` performs data checks and
seals the plan without training; use a separate fresh output for the actual
study. The runner uses the existing generic HF bridge and generation client.
Objective coefficients and generation metrics still come from Rust.

Before the first update, `sealed-plan.json` records exact commands, dataset
identities, package versions, source/input hashes and the current Git commit.
Before every subprocess, changed sources/inputs and less than 8 GiB free space
stop execution. Training also receives the expected tokenized dataset identity.
Runs are sequential, with arm order rotated by seed. Existing output is never
automatically reused or restarted. A partial study keeps its logs and checkpoints
for explicit recovery; it cannot produce `completed.json`.

Completed run cards must prove the full horizon, saved checkpoint steps, finite
held-out loss, paired initial evaluation/runtime identity, and active treatment
with the exact objective policy. Each generation report is revalidated by the
Rust evidence API. `completed.json` means **execution complete, assessment still
pending**, not that the treatment passed its scientific gate.

```bash
"$PYTHON" -I -m pytest --import-mode=importlib --confcutdir=tools --rootdir=tools \
  -q tools/test_hf_repetition_schedule_study.py
```

## Assessment Boundary

Evaluate all nine runs at the frozen final endpoint using the protocol's paired
loop-score, CE-safety and ordinary-FT sanity gates. Keep intermediate curves
descriptive. Read the continuation texts and generated lengths as well as the
numeric scores: natural EOS or shorter output can reduce observed loops without
improving language. Three seeds do not establish statistical significance.

Retain complete raw logs, cards and adapters locally. Publish all conditions,
derived verification/results, hashes, representative text and reproduction
instructions, but not private machine paths or model weights. The earlier
negative long-horizon artifacts remain unchanged whatever this study finds.
