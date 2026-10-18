# Native Repetition Learning Qualification

Source-only after PyPI 0.4.32. The local development wheel was not uploaded.
See [API and ownership](../../../docs/python/native-repetition.md).

## What Was Exercised

- Shared Rust negative-token loss/VJP, canonical history selection and native
  deterministic top-k, connected through frozen GPT-2 to trainable geometry.
- CE target-count and UL active-position means reduced independently before a
  shared optimizer update. Regression cases include CE counts `[5, 3]` versus
  nonempty UL counts `[1, 2]`, plus candidate-free microbatches.
- Fully masked work-budget admission before the first corpus update, exact
  zero-strength behavior, immutable ownership and pinned continuation.
- Default loss/gradient/trainer bytes and legacy HF v3 plans match the previous
  installed binaries exactly for the retained compatibility fixtures.

Local qualification passed 3 tensor, 19 candidate and 57 native LM Rust tests;
54 installed Python tests with foreign ML imports forbidden; and 430 optional
Torch reference cases. The new 84 reference cases cover 14 geometry recipes,
3 candidate sources and both readout policies, comparing loss, parameter VJP
and SGD updates. These use small deterministic models, not pretrained-model
quality evaluation. Core WASM compilation and strict Rustdoc also passed.
Binding Clippy completed with existing unrelated warnings.

Two independent read-only reviews found admission, buffer-lifetime, binding
type and normalization-coverage issues. They were fixed and re-reviewed.
The first admission regression attempt hit a fixture geometry-capacity limit;
after correcting only the fixture, it failed at the intended constructor check.
Both failure logs and the successful repair runs are retained locally, with
hashes in [verification.json](verification.json). No failed attempt was erased.
A supplemental release smoke initially selected a stale global Python package,
not the qualified environment. The explicit development-wheel rerun passed all
9 tests without code/gate changes; both attempts are recorded in
[verification-followup.json](verification-followup.json).

## Pinned GPT-2 Pilot

[comparison.json](comparison.json) records every declared condition. The same
existing local pretrained GPT-2 and saved geometry initialize each arm, with
fresh native AdamW, rate 0.001, clip 0.1 and wide vocabulary readout. The two
32-ID documents are one fixed training-corpus window and one explicitly
synthetic periodic stress window derived from the same admitted token stream.
Each update accumulates two one-window microbatches: 62 CE targets.

| Candidate source | UL positions per update | UL mean, step 1 | UL mean, step 2 | Fresh-process resume |
| --- | ---: | ---: | ---: | --- |
| Prior continuation | 7 | 0.38980170 | 0.38920796 | Exact |
| Model top-k history | 51 | 0.11905463 | 0.11904670 | Exact |
| Model top-k periodic | 1 | 2.45967221 | 2.45266151 | Exact |

All three arms completed two geometry updates; saved reports and final
checkpoint bytes matched after a separate-process continuation from update 1.
The different UL means are **not a ranking of candidate sources**: their
negative sets and denominators differ. This is six accepted updates on training
and synthetic inputs, not held-out perplexity, improved generation, whole-model
FT, throughput, or memory-usage evidence. The pretrained pilot is native-only;
Torch numerical comparisons are the separate tiny-model suite above.

Raw checkpoints, encoded inputs, logs and local wheels stay private. Public
records contain conditions, aggregate reports, source/model/data hashes and
checkpoint receipts, not weight arrays or token streams.

## Reproduce

Use Python 3.12 with the matching source-built wheel and existing, authorized
local assets. This probe downloads nothing and refuses different model,
corpus or original-checkpoint digests. It creates a new private cohort and
refuses source/binary changes after preparation.

```sh
export OMP_NUM_THREADS=2 RAYON_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PROBE=benchmarks/results/2026-10-19-native-repetition/qualify.py
for MODE in prepare native resume; do
  python -I -B "$PROBE" --mode "$MODE" --model-dir "$LOCAL_GPT2" \
    --data "$ADMITTED_DATA" --checkpoint "$ORIGINAL_CHECKPOINT" --cohort "$NEW_PRIVATE_RUN"
done
python -I -B "$PROBE" --mode public --model-dir "$LOCAL_GPT2" \
  --data "$ADMITTED_DATA" --checkpoint "$ORIGINAL_CHECKPOINT" --cohort "$NEW_PRIVATE_RUN" \
  --output "$NEW_PUBLIC_RESULT"
```

Each stage was supervised by `tools/owned_process.py`: file logs, bounded
timeouts and owned process-group cleanup. Native execution had a 300-second
limit; preparation, replay and publication each had 180 seconds. These are
safety limits, not benchmark timings. Existing records must not be overwritten.
Linux, Windows and CPython 3.8 installed-wheel gates are wired into CI/release
workflows but were not executed locally; official publication remains separate.

## Public Example Integration

[integration-reviewed.json](integration-reviewed.json) adds a separate default-f32
study through the public corpus and generation examples: two periodic-repetition
updates with smoothing 0.1 and weight decay 0.01, then four Z-Space-controlled
tokens in each of full-prefix and KV-cache modes. Fresh-process partial/resumed
runs match complete runs byte-for-byte, including every step and checkpoint.
The verifier independently re-executes the same sealed inputs through Rust and
compares all summaries, intermediate/final states and ordered reports. This is
same-runtime mechanics verification, not an independent numerical oracle or a
language-quality/speed result. The synthetic stress window remains explicit.

Independent review found three initial verifier gaps: empty logs, unbound training
conditions and unbound generation conditions. All were fixed; negative tests use
coherent wrong-data, no-repetition, wrong-optimizer, wrong-seed and wrong-prompt
runs, not just malformed JSON. The [initial result](integration-initial.json) is
retained, not silently replaced. All 37 input/runtime file hashes and aggregate
results match the separate reviewed rerun; only the sealed verifier source
contract changed. Raw inputs, checkpoints and logs remain private.
[integration-verification.json](integration-verification.json) retains execution
receipts, expected failing regressions and a failed fixture attempt whose
re-serialized safetensors required a newly bound base identity. The broad native
Rust suite also passed 1,032 library and 9 gradient-reduction tests.

With the same authorized local assets and matching source-built native wheel,
run each command as its own process. The probe refuses source/binary/input
changes after preparation and forbids optional foreign ML imports:

```bash
PROBE=benchmarks/results/2026-10-19-native-repetition/integration.py
COMMON=(--model-dir "$LOCAL_GPT2" --data "$ADMITTED_DATA"
        --checkpoint "$ORIGINAL_CHECKPOINT" --cohort "$NEW_PRIVATE_RUN")
python -I -B "$PROBE" --mode prepare "${COMMON[@]}"
for ARM in full part resume; do
  python -I -B "$PROBE" --mode train --arm "$ARM" "${COMMON[@]}"
done
for EXECUTION in full_prefix kv_cache; do
  for ARM in full part resume; do
    python -I -B "$PROBE" --mode generate --execution "$EXECUTION" \
      --arm "$ARM" "${COMMON[@]}"
  done
done
python -I -B "$PROBE" --mode report --output "$NEW_PUBLIC_RESULT" "${COMMON[@]}"
```

The recorded runs use the same two-thread/offline environment above and
`tools/owned_process.py`, with 180-second training/report and 120-second
preparation/generation limits. The report's additional native re-execution is
qualification overhead, not part of a throughput measurement.
