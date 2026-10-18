# Native corpus training: first qualification

Source-only after PyPI 0.4.32. This is frozen-GPT-2 geometry training, not
base-weight FT, general HF architecture support, speed evidence or language
quality. Rust owns document windows, selected-token means, accumulation,
gradients, AdamW and the checkpoint cursor. The public Python example performs
local file I/O and calls those owners; no Torch fallback is involved.

## Initial Sealed Run

One already admitted training chapter from the earlier
[book control](../2026-10-18-hf-book-controls/README.md), 1,303 encoded tokens,
40 disjoint 32-token windows and 23 dropped tail tokens. No development or
reserve data was read. Two epochs, microbatch 2, accumulation 2, learning rate
0.001, decay 0.01, smoothing 0.1 and clip limit 0.1 give 20 updates and 2,480
scored targets. The same local GPT-2 and three initial geometry placements
(WaveGate with radius, Fractional angle/gain/history, Topos) are hash-pinned.

`comparison-initial.json` retains every update, both engines' runtime records,
the predeclared gates, input/code hashes and hashes of private checkpoints.
Native execution forbids Torch, NumPy, Transformers, Accelerate and Python
safetensors imports. A new native process resumed at update 2: all remaining
18 step records and final checkpoint bytes matched uninterrupted execution.

**The original strict f32 comparison failed.** Maximum loss error was
3.5047531e-5 (within 2e-4), but final parameter error reached 0.00326771 and
exceeded the predeclared per-element `3e-5 + 1e-3 * abs(reference)` gate.
Neither this failure nor its thresholds are replaced by later diagnostics.

## Post-Failure Localization

The first accumulated update already differed in gradients. `diagnose.py`
localizes it without changing native math. `diagnose_head.py` compares terminal
operations on identical reference inputs; this diagnostic uses optional Torch
and is not a native-only run or performance measurement.

On the same logits, Rust CE's maximum gradient error against a float64 oracle
was 3.0154e-10; Torch f32 CE's was 2.5148e-6 in this installed runtime. Computing
only the reference loss in float64 reduced the first whole-decoder parameter
gradient maximum error to 1.3970e-8. `head-f64-diagnostic.json` records the
terminal measurements. This identifies a substantial numerical contribution,
not a general claim that Torch is incorrect or native is always more accurate.
The first-update diagnostic alone does not qualify a complete training run.

## Separate Precision Control

`comparison-f64-control.json` is a **post-failure diagnostic control**, not a
replacement for the initial report. Before its run, the new contract fixed
`reference_ce_dtype: f64`. Only reference CE received float64 logits; the Torch
decoder, geometry and AdamW stayed f32. Native math, initial geometry, encoded
data, recipe, update count and original error thresholds were unchanged.

All 20 updates passed: maximum loss difference 1.9663438e-6, final parameter
difference 3.7430e-6. Fresh-process continuation again reproduced the remaining
18 reports and final checkpoint bytes exactly. Both cohorts' native checkpoint
bytes also agree. This qualifies this bounded configuration against the stated
precision control, not ordinary all-f32 Torch parameter parity. It gives no
language-quality, speed, unseen-corpus or cross-platform conclusion.

Tests now include vocabulary-sized smoothed gradients against an independent
f64 distribution and the optional f64 Torch oracle. Native arithmetic was not
changed to imitate the f32 difference.

## Replay And Retention

Keep raw encoded documents, checkpoints, gradients and model weights private.
Only results, hashes, verification records and executable procedures are public.
Use new output paths for every invocation; preserve failed attempts.

Use the source revision matching `contract.source_sha256` when checking an old
cohort (the first protocol is retained in commit `4abd57c4e`). For a new cohort,
`prepare` pins the current script/example bytes before
training. `MODEL` must contain the exact local config, tokenizer and safetensors
hashes in the contract. `CHAPTER` is the pinned existing train/chapter-01.txt,
and `ARTIFACT` is the pinned initial geometry. No download is performed.

```sh
SCRIPT="$PWD/benchmarks/results/2026-10-18-native-corpus-training/qualify.py"
"$TOKENIZERS_PYTHON" -I "$SCRIPT" prepare --model-dir "$MODEL" \
  --chapter "$CHAPTER" --artifact "$ARTIFACT" --output "$COHORT"
"$NATIVE_PYTHON" -I "$SCRIPT" native --model-dir "$MODEL" \
  --cohort "$COHORT" --output "$COHORT/native-full"
"$NATIVE_PYTHON" -I "$SCRIPT" native --model-dir "$MODEL" \
  --cohort "$COHORT" --resume-run "$COHORT/native-full" --output "$COHORT/native-resumed"
"$REFERENCE_PYTHON" -I "$SCRIPT" reference --model-dir "$MODEL" \
  --cohort "$COHORT" --output "$COHORT/reference"
python3 -I -S "$SCRIPT" compare --cohort "$COHORT" --output "$COHORT/comparison.json"
```

For the separate precision control, add `--reference-ce-dtype f64` to `prepare`
and choose a new cohort directory. The precision is sealed before model execution
and cannot be overridden on an individual run. The default remains the original
f32 comparison, whose failure is retained above. New cohort/run directories use
POSIX mode `0700`; that is not encryption or a Windows ACL guarantee.

The stdlib comparator regression tests need neither model files nor Torch:

```sh
python3 -I -S benchmarks/results/2026-10-18-native-corpus-training/test_qualification.py
```

Both Python environments need this source build; only the comparison environment
needs Torch/Transformers. Tokenization uses the standalone Rust-backed tokenizers
package, not a model framework. A development wheel still reporting 0.4.32 is
not the published 0.4.32 binary and must never be uploaded under that version.
These qualification/diagnostic scripts require Python 3.11+ (`hashlib.file_digest`);
the public training example and installed-wheel contract retain Python 3.8 support.
The local execution recorded here used Python 3.12, not a simulated 3.8 interpreter.
