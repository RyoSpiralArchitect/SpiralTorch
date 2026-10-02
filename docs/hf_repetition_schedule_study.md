# Repetition Schedule Study

This is the first long-horizon comparison for the
[Rust-owned objective control](hf_repetition_objective_control.md), not a new
decoding intervention or a claim of language-quality improvement.
The [aligned v2 protocol](benchmarks/hf_repetition_schedule_aligned_256step_prespec_20261002.json)
must be committed before training. Its byte SHA-256 is the generation protocol
identity, also recorded in each private execution plan.
The original v1 attempt was stopped for a confirmed causal-label denominator
confound, not an efficacy result. Its protocol and raw artifacts remain intact;
see the [invalidation record](../benchmarks/results/2026-10-02-llm-schedule-v1-invalidated.md).
It is neither resumed nor pooled with v2. The horizon and acceptance gates did
not change.

## Fixed Comparison

- Cached pretrained GPT-2, the existing complete *Pride and Prejudice* text,
  three new paired seeds (151, 157, 163), and 256 update slots per arm.
- Ordinary LoRA FT, periodic unlikelihood at constant strength 0.1, and the
  same periodic objective with linear strength decay from slot 0 to 256.
- CPU float32, rank 4 / alpha 8, learning rate 5e-5 and the same linear learning
  rate scheduler. Batch size 1 with accumulation 16 avoids unequal last-batch
  sizes; every collated row must have 127 valid labels both before and after
  causal shifting. All three arms use `--causal-lm-mask-first-label` through the
  shared `st.HfCausalLabelAlignmentCollator`. Fixed lengths alone are not enough.
- Full selected held-out split, not the development smoke's two-block subset.
  The data-only preflight records exact training/evaluation block counts for
  each new seed: 1,139/124, 1,137/127 and 1,134/128 for seeds 151, 157 and 163.
  Rows are randomly split within the same book;
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
export SPIRALTON_MAGIC=0 SPIRALTON_TORCH=0
export SPIRALTON_MODEL_PATCHES=0 SPIRALTON_NUMPY=0
"$PYTHON" -I tools/run_hf_repetition_schedule_study.py \
  --model "$MODEL_SNAPSHOT" --corpus "$CORPUS" --output "$OUTPUT"
```

The output directory must not exist. `--preflight-only` performs data checks and
seals the plan without training; use a separate fresh output for the actual
study. The runner uses the existing generic HF bridge and generation client.
Objective coefficients and generation metrics still come from Rust.
Disable automatic device/model patches before Python starts, including when
running the tests or assessor. Preflight rejects a non-CPU default device.

Before the first update, `sealed-plan.json` records exact commands, dataset
identities, package versions, source/input hashes and the current Git commit.
Before every subprocess, changed sources/inputs and less than 8 GiB free space
stop execution. Training also receives the expected tokenized dataset identity.
Runs are sequential, with arm order rotated by seed. Existing output is never
automatically reused or restarted. A partial study keeps its logs and checkpoints
for explicit recovery; it cannot produce `completed.json`.

Completed run cards must prove the full horizon, saved checkpoint steps, finite
held-out loss, paired initial evaluation/runtime identity, the shared alignment
collator contract, and active treatment
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

After all nine runs and 36 generation reports complete, assess without changing
the frozen execution sources or inputs:

```bash
"$PYTHON" -I tools/assess_hf_repetition_schedule_study.py "$OUTPUT" \
  --output "$ASSESSMENT_JSON"
```

This rejects missing/duplicate conditions, incomplete horizons, mismatched
artifact hashes and altered source/input files. It revalidates Rust generation
evidence against the actual saved adapter fingerprint, then decodes the committed
tokens to check every published continuation text. All 432 prompt continuations
and their token lengths are exported, not a selected showcase. The output omits
private raw paths and contains the frozen final paired effects and gate result.
No assessment is emitted for a still-running study.

```bash
"$PYTHON" -I -m pytest --import-mode=importlib --confcutdir=tools --rootdir=tools \
  -q tools/test_assess_hf_repetition_schedule_study.py
```

Retain complete raw logs, cards and adapters locally. Publish all conditions,
derived verification/results, hashes, representative text and reproduction
instructions, but not private machine paths or model weights. The earlier
negative long-horizon artifacts remain unchanged whatever this study finds.
