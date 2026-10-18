# Native vocabulary projection: compatibility, precision and CPU timing

Development source, newer than immutable PyPI 0.4.32. The development wheels
used here must **not** be uploaded under that published version.

## Intervention and boundaries

Two separate changes, with no Torch dependency in the native path:

- Ordinary packed row tails compute eight independent output columns together.
  Each dot retains its original sequential f32 reduction. The CPU autotune key
  advances to revision 7 because timing choices can change.
- Explicit `model.with_readout_accumulator("f64")` uses four fixed f64 partial
  sums for vocabulary readout and its input-gradient projection. Weights,
  storage and results remain f32; the rest of the decoder is unchanged.

The latter policy travels through selected-target loss, geometry learning,
K/V contexts, sampled/Z-Space generation and trainer/generator checkpoints.
Cross-policy gradients, accumulations and checkpoint restoration reject.
An artifact alone does not save execution arithmetic. See the
[API contract](../../../docs/python/native-causal-lm.md#explicit-readout-arithmetic).

This experiment uses the same pinned local GPT-2 checkpoint, already-admitted
train chapter and learned geometry artifact as the earlier
[KV study](../2026-10-18-native-lm-cache/README.md). No new dataset, provider,
download, held-out evaluation or base-weight FT is introduced.

## Diagnosis

`probe.py` compares native primitives on **identical actual reference inputs**
and separately replaces reference linears/norms as a diagnostic intervention.
Those hybrid graphs are not a native runtime or quality evidence.
At 72 evaluated tokens, the output head alone had max error about 6.71e-4
against Torch f32 and 6.37e-4 against an f64 dot-product control.
Replacing only norms did not explain the original full-model 7.02e-4 discrepancy.

The first probe failed because a contiguous safetensors-backed DLPack view did
not satisfy the native alignment contract. An owned aligned transport copy
fixed the probe; the failed execution is retained, not relabeled as a model failure.
The raw-logit gate remains **3e-4 plus exact argmax**, not a probability-error
substitute or a post-result relaxed tolerance.

## Measurement protocol

Native-only processes forbid Torch, Transformers, NumPy, Accelerate and Python
safetensors imports. An old development binary is measured before replacement.
For base and learned geometry, lengths 16/64 each test prefill and eight
teacher-forced appends. Both numerical policies are exercised after rebuilding.
Raw f32-vector hashes compare cached/full results without publishing logits.
Default-policy projection hashes, generated step/checkpoint identities and two
geometry-learning update identities must equal the old binary's records.

Each model/policy runs greedy, sample and Z-Space generation. Fresh processes
resume after three generated tokens. The learned model/policy also performs
two native AdamW updates on two fixed 16-token windows, resuming after one.
This is continuation/integration evidence, not a convergence or quality result;
the two update losses belong to different windows.

Ordinary geometry-free CPU timing uses six arms: native f32/full, f32/KV,
f64-readout/full, f64-readout/KV and Torch full/KV. Every arm projects only the
last hidden row. Inputs are prepared before timing; loading is excluded.
There are two warmups and seven retained repetitions per length, with reversed
arm order on alternate rounds. Prefill and eight-token decode durations are
reported separately; no GPU, compilation or diagnostic hooks are timed.
The unchanged f32 path is retained even where its logit gate fails.
Wide-readout timing is a precision tradeoff, not an identical-arithmetic speed win.

Raw tokens, checkpoint payloads, weights and complete local logs remain private.
Public output contains all conditions, scalar metrics, hashes, source identities
and verification records. No failed condition is dropped from the aggregates.

## Results

[Reviewed v2 results](comparison.json) retain all 108 timing trials and all
108 separate prefix-accuracy checks. Both saved protocol seals, transitive
executable-helper hashes, candidate binary and raw continuation hashes are checked
before publication. Runtime free-text fields stay in hashed private records.
Measurements used Apple M4/macOS 26.4.1, Python 3.12.6, Torch 2.12.1 and
Transformers 4.57.6; requested CPU threads were two (Torch interop one).

| Prefix workload | Default f32 max error | Wide-readout max error | Wide gate |
| --- | ---: | ---: | --- |
| 16 through 24 tokens | 2.594e-4 | 7.629e-5 | Pass |
| 64 through 72 tokens | 7.019e-4 | 1.221e-4 | Pass |

These are maximum errors over **all prefixes**, not just the final output;
full and cached paths agree. The unchanged f32 path still fails at the long
condition. The earlier final-only numbers are not reused as workload qualification.

Median milliseconds for **eight teacher-forced decode calls**, excluding prefill:

| Arm | 16-token prompt | 64-token prompt |
| --- | ---: | ---: |
| Native f32, full prefix | 857.01 | 1820.47 |
| Native f32, K/V | 220.55 | 230.41 |
| Native wide readout, full prefix | 847.99 | 1814.71 |
| Native wide readout, K/V | 215.08 | 223.47 |
| Torch f32, full prefix | 179.60 | 291.75 |
| Torch f32, K/V | 82.38 | 87.72 |

Torch remains faster. The f32 rows are timing observations, not an accuracy-qualified
replacement at the longer length. Wide readout meets the scoped numerical gate
but has a different accumulator contract. Historical native K/V decode was about
459/470 ms in the previous study; that is an **unpaired historical comparison**,
not a controlled before/after speedup claim.

Native checks: 72 cached/full-prefix vector comparisons are bitwise exact;
default-policy outputs, six generated trajectories/checkpoints and two training
updates match a freshly recaptured old-binary baseline. Twelve generation cases
and two trainer cases resume exactly in fresh processes. This does not establish
cross-architecture or cross-version bitwise guarantees.

The [initial cohort](comparison-initial.json) is retained with status
`UNQUALIFIED_FULL_SEQUENCE_ACCURACY_NOT_CHECKED`: it checked only final reference
outputs and did not seal helper dependencies. Its original protocol is retained
in commit `ae4fb67d6`. The v2 baseline was recaptured, not re-hashed or silently
accepted. [Attribution](attribution.json) is diagnostic-only.

[Verification](verification.json) records source/wheel/log hashes and unsuccessful
attempts. Checks include 50 native LM release tests, 16 packed-kernel release tests,
five wide-projection release tests, 63 active native Python tests, a discovery
isolation subprocess, 346 optional reference tests and ten publication guards.
The wide test filter overlaps one packed-kernel test; these counts are not a sum.
WASM core compilation, strict Rust documentation and scoped formatting/lint also
passed. Independent read-only core and publication reviews closed all findings.

## Reproduce

Requires Python 3.11+ for the qualification tools, the exact admitted local
model/data/checkpoint, the pinned earlier development wheel for the baseline,
and a wheel built from this source for the candidate. The Python library itself
retains its existing supported-version floor.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 RAYON_NUM_THREADS=2 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

# Run with the old native-only environment before replacing that wheel.
"$OLD_NATIVE_PYTHON" -I experiment.py --mode baseline \
  --model-dir "$MODEL" --data "$DATA" --checkpoint "$CHECKPOINT" --output "$BASELINE"

# Run with the candidate native-only environment. Output directories must be new.
"$NATIVE_PYTHON" -I experiment.py --mode native --baseline "$BASELINE/native.json" \
  --model-dir "$MODEL" --data "$DATA" --checkpoint "$CHECKPOINT" --output "$CANDIDATE"
"$NATIVE_PYTHON" -I experiment.py --mode resume \
  --model-dir "$MODEL" --data "$DATA" --checkpoint "$CHECKPOINT" --output "$CANDIDATE"

# A separate reference environment has the SAME candidate native wheel plus Torch.
"$REFERENCE_PYTHON" -I experiment.py --mode benchmark \
  --model-dir "$MODEL" --data "$DATA" --checkpoint "$CHECKPOINT" --output "$CANDIDATE"
python3 -I test_publication.py -v
python3 -I publish.py --baseline "$BASELINE/native.json" \
  --cohort "$CANDIDATE" --output comparison.json
```

Run in this directory or pass absolute script paths. Keep model IDs, policy,
thread counts, all trials and original tolerances. A successful source build or
this one-machine comparison does not qualify release wheels on other platforms.
