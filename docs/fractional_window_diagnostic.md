# Saved-Model History-Window Diagnostic

This is a new, read-only intervention on the completed angular study, not a
replacement for its endpoints or a matched-training experiment. Every saved
`history_angle_full` seed and every original endpoint block is evaluated.
No duration, checkpoint, seed, endpoint or hyperparameter is selected.

The diagnostic uses the Rust [window operator](fractional_history_window.md):

| Mode | Retained lags | Normalization |
| --- | --- | --- |
| full | all past taps | declared K |
| retained_short | [1,3) | declared K, not K=3 |
| retained_tail | [3,K) | declared K |
| local_only | empty [1,1) | declared K |

Angle, log gain, local gate and history gate remain bitwise unchanged. The
local-only mode keeps the learned local gate; it is NOT the unadapted base
model. The source checkpoint is loaded only with its original full recipe.
Windowed targets receive an explicit parameter-only copy with a distinct
recipe hash. No saved `_extra_state` is rewritten and no optimizer is made.

Before any windowed scoring, the candidate runtime must reproduce all saved
full-history block losses AND their means exactly, for ALL seeds. Tolerances
are fixed at zero. Any mismatch produces `blocked_full_replay`, retains all
full-score observations, and withholds every partial-window score. Do not
relax the gate after observing a failure. Each model insertion is undone even
if scoring fails. Frozen base and adapter parameters are checked for mutation.

## Offline Execution

Use the original eight-file client, original sealed study and both runtime
manifests. The original runtime is verified as files; only the candidate
runtime is imported. These identities are separate in the diagnostic receipt:
the original plan's native hash is never silently changed or relaxed.

```sh
env PYTHONNOUSERSITE=1 \
  SPIRALTON_MAGIC=0 SPIRALTON_TORCH=0 SPIRALTON_MODEL_PATCHES=0 SPIRALTON_NUMPY=0 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
  PYTHONPATH="$FROZEN_CLIENT:$CANDIDATE_PACKAGE:$REPO/tools" \
  "$PYTHON" -P -B "$REPO/tools/diagnose_fractional_history_windows.py" \
  --study "$ORIGINAL/study" \
  --study-manifest "$ORIGINAL/sealed-output-sha256.json" \
  --client-manifest "$ORIGINAL/client-sha256.json" \
  --original-runtime "$ORIGINAL_PACKAGE" \
  --original-runtime-manifest "$ORIGINAL/frozen-runtime-sha256.json" \
  --candidate-runtime-manifest "$CANDIDATE/frozen-runtime-sha256.json" \
  --model-dir "$MODEL_SNAPSHOT" --corpus "$PRIDE" --transfer-corpus "$ALICE" \
  --diagnostic-source-revision "$SOURCE_REVISION" \
  --output-dir "$NEW_DIAGNOSTIC/run"
```

Use the frozen training framework versions, CPU float32, two threads, complete
unpadded contexts, no packed boundaries and no KV cache. The tool reconstructs
and verifies all token partitions and the base model identity. Output must be
new and outside all protected input roots. There is no automatic resume or
overwrite; keep interrupted/failed receipts instead of silently retrying.
`completed` is a scoring state; validation additionally records process exit
and unchanged source/client/runtime/study hashes. Weights, text and private
logs remain local. Publish numeric block scores/deltas, hashes and checks.

## Interpretation

Positive CE delta relative to full means removing the other taps hurt this
already-trained model. This isolates post-training dependence while keeping
full-K normalization fixed; it does not show that training the retained taps
alone would fail. Partial operators add up to the full filter (modulo f32
rounding), but nonlinear model losses do not have to be additive.

These are reused exploratory Pride/Alice endpoints. Do not infer significance,
new independent seeds, general LLM quality or speed. The preceding study's
failed ordinary/GL-short final-parameter parity remains failed and unchanged.
The next causal comparison would train full versus retained-short windows
from the same initialization/budget, not train an initially dormant tail alone.

The [completed three-seed diagnostic](../benchmarks/results/2026-10-07-fractional-window-diagnostic/README.md)
publishes all four modes after exact reproduction of the 456 original full
block losses. Removing either short or long taps hurt all three saved models
on both reused endpoints; this remains post-training dependence, not a result
of the proposed matched-training comparison.

The [matched-learning recipe](fractional_window_study.md) now fixes both arms
to the same K=32 normalization and pairs their training schedules.
