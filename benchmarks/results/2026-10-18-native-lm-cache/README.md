# Native GPT-2 KV Context

**Cache mechanics passed; the complete PyTorch comparison did not pass its
predeclared raw-logit gate.** All conditions, including failures and warmups,
are retained in [comparison-initial.json](comparison-initial.json). No threshold
was changed and no replacement benchmark was selected. This is a source-only
extension after the immutable PyPI 0.4.32 release, not a new published wheel.

## What Runs Natively

Rust owns prompt prefill, per-layer K/V, absolute-position appends and last-row
readout. Python uses `NativeCausalLmContext` or the generator's explicit
`execution={"kind":"kv_cache","max_values":16777216}` policy. No native run
imports Torch, Transformers, NumPy, Accelerate or Python safetensors.

All 14 existing geometry recipe configurations remain supported. Row-local
geometry evaluates only the newest row; causal geometry retains its raw inputs
and replays the existing Rust operator over that prefix. K/V still copies into
transaction buffers on append. This is not a zero-copy cache, an incremental
algorithm for every geometry, resident-GPU generation or batched serving.

On the pinned GPT-2 snapshot `607a30d783dfa663caf39e06633721c8d4cfcd7e`:

- Base and previously learned three-site geometry models, input lengths 8/32/64,
  each followed by eight fixed token appends: all **54** cached/full-prefix
  vocabulary-logit comparisons were exactly equal in this runtime.
- Ordinary greedy, seeded ordinary sampling and Z-Space sampling, three fixed
  prompts, both models: **18** cases / **144** generated tokens completed.
- Fresh-process cached continuation after token three was exact in all 18 cases,
  including step reports and final checkpoint bytes. Another **18** previously
  issued full-prefix checkpoints retained exact serialization and continuation.
- The source tests cover all recipe/site combinations, explicit work budgets,
  invalid inputs, frozen ownership and a numerical append failure after successful
  appends; the preceding state and later valid continuation remain unchanged.

The learned artifact is the earlier corpus trainer's step-20 checkpoint, not a
newly fitted model. Inputs come only from that already-admitted training chapter.
There is no holdout, generalization or geometry-quality claim. Exact equality
on these inputs is not a cross-hardware/kernel bitwise guarantee.

## Ordinary CPU Timing

Apple M4 / macOS 26.4.1, Python 3.12.6, Torch 2.12.1 / Transformers 4.57.6;
CPU float32, batch one, requested two threads. Geometry is **disabled** for this
speed comparison. Every arm projects only the last hidden row to vocabulary
logits, including the PyTorch full-prefix reference. Loading/packing, caller
input preparation and logit inspection are outside the clocks. Prefill is timed
separately; the table is the following eight teacher-forced appends, in ms.
Two warmup rounds precede seven measured repetitions per condition; arm order
alternates forward/reverse. Each repetition starts with a new cache.

| Prompt | ST full prefix | ST cached | Torch full prefix | Torch cached |
| --- | ---: | ---: | ---: | ---: |
| 16 | 1219.279 | 450.422 | 164.923 | 75.717 |
| 64 | 2642.589 | 462.620 | 284.463 | 82.733 |

ST's internal cached/full comparison is about **2.71x / 5.71x** faster here,
with exactly equal final logits. PyTorch remains faster on this machine.
Do not present the full cross-runtime table as a qualified speed win:
the sealed maximum-absolute-logit tolerance was `3e-4`. Both ST arms show
`0.000213623046875` at prompt 16 (passes) and `0.000701904296875` at prompt 64
(fails), against the full-prefix Torch reference. Torch cached differs by at
most `0.00006103515625`. Top-1 IDs match in every measured/warmup condition.
The identical error in both ST arms is not introduced by caching; it still
needs investigation rather than a post-result tolerance increase.

[diagnostic.json](diagnostic.json) is explicitly post-result and not a new gate.
Subtracting the mean logit shift does not remove the long-input discrepancy.
Its maximum probability difference is only `9.562561853471152e-9`, but that
does **not** retroactively pass the failed raw-logit contract. All failed
measurements, first test-fixture failures and subsequent fixes are retained
through [verification.json](verification.json).

## Reproduce Without Downloading

Use the same development wheel in a no-foreign-ML `NATIVE_PY` environment and a
separate `REFERENCE_PY` environment with the reference versions above. The
qualification scripts require Python 3.11+; the package API still supports its
declared Python floor. Set offline mode and CPU thread limits as in the receipt.
`MODEL_DIR`, `DATA` and `CHECKPOINT` must match the public hashes. `PREVIOUS`
is the retained `real-generation` directory from the preceding generation
qualification; old raw files are checked against its public hashes.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 RAYON_NUM_THREADS=2 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
SCRIPT=benchmarks/results/2026-10-18-native-lm-cache
COMMON=(--model-dir "$MODEL_DIR" --data "$DATA" --checkpoint "$CHECKPOINT" --cohort "$COHORT")
"$NATIVE_PY" -I "$SCRIPT/qualify.py" prepare "${COMMON[@]}"
"$NATIVE_PY" -I "$SCRIPT/qualify.py" native "${COMMON[@]}"
"$NATIVE_PY" -I "$SCRIPT/qualify.py" resume "${COMMON[@]}" --previous-cohort "$PREVIOUS"
"$REFERENCE_PY" -I "$SCRIPT/qualify.py" benchmark "${COMMON[@]}"
# The recorded benchmark exits nonzero at its numerical gate. Keep its files.
"$NATIVE_PY" -I "$SCRIPT/retain_failure.py" "${COMMON[@]}" --output "$RUN/comparison-initial.json"
"$REFERENCE_PY" -I "$SCRIPT/diagnose.py" "${COMMON[@]}" --output "$RUN/diagnostic.json"
```

Run the shell fragment with Bash or Zsh (it uses arrays). All output directories
and files must be new. `prepare` seals source/config/data/model hashes before
measurement. On a different runtime where all gates pass, use `qualify.py report`
instead; the failure archiver refuses a passing cohort. Private token sequences,
checkpoint bodies and model weights are not copied into the repository.
No remote compute, download or provider request was made.
