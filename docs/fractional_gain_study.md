# Learned Gain And Ordinary Short History

The completed [length/energy factorial](fractional_history_factorial.md)
showed that fixing filter energy changed both the preferred order and the
benefit of a longer available history. This follow-up gives amplitude its own
parameter and asks whether that mechanism does more than an ordinary learned
short filter. It is a quality experiment, not a throughput comparison.

The [completed nine-run comparison](../benchmarks/results/2026-10-05-fractional-gain-study/README.md)
retains all outcomes: ordinary learned short beats both GL arms on both
endpoint sets in all three seeds, while GL full beats GL short. This is not
evidence of a distinctive GL quality advantage; shape-chart differences
remain an explicit follow-up question.

## Matched Arms

| Arm | Strictly Past Filter | Trainable Parameters |
| --- | --- | ---: |
| `ordinary_gain_short` | gain * [-cos(angle), sin(angle)] | 1538 |
| `history_gain_short` | Rust normalized GL, K=3 | 1538 |
| `history_gain_full` | Rust normalized GL, K=32 | 1538 |

All three use independent zero local/history feature gates, gain initialized
to sqrt(5), and the same initial [-2, 1] filter up to float32 rounding. GL
starts at alpha=2; the ordinary angle starts at the fixed protocol literal
approximately atan(1/2). Initialization consumes no RNG and the adapter is
exactly identity. Check nonzero-gate outputs and input/gate VJPs as well:
identity alone cannot validate the history branch.

GL short/full share parameter names, registration order and initial hashes.
The ordinary control uses an angle instead of log-order, so its full parameter
hash is intentionally different. Every arm has `2*F+2` trainable parameters
with F=768. The ordinary control is independent Torch trigonometry and two
zero-padded shifts, not Python GL coefficients or a new production backend.

The ordinary unrestricted angle and GL positive log-order have different
optimizer charts and reachable shapes. Equal capacity does not equalize
optimization geometry. Gain and feature gates are also redundant amplitude
controls. Filter norm is not output variance on correlated hidden states, and
a fit does not identify a unique order. In particular, only the GL full vs
GL short comparison shares a chart; even that does not uniquely isolate a
long-memory mechanism because normalization changes short taps too.

## Fixed Protocol

`bindings/st-py/examples/hf_fractional_pride_gain.json` fixes the preceding
local GPT-2 snapshot and Pride/Alice corpus hashes, first-MLP insertion,
seeds 41/43/47, Adam 0.001, strength 0.1, 512 updates per run, batch 2,
context 128 and two CPU threads. It keeps the prior data partition: 1204
training, 16 diagnostic development, and delayed 120 Pride / 32 Alice
endpoint blocks. There are 4608 primary updates and 18 continuation-only
updates across nine runs. Complete every run before scoring endpoints.

Primary CE contrasts are GL short minus ordinary short, GL full minus
ordinary short, and GL full minus GL short. Negative favors the first named
arm. Report every seed and losing outcome; development cannot choose the
duration, checkpoints, hyperparameters or which arms are reported. These
reused exploratory books may be in pretraining and are not pristine held-out
confirmation. Three minibatch-order seeds do not establish significance or
general LLM superiority. Do not pool prior experiments as new seeds.

Record signed log-gain gradients and both coordinate/effective gain before
and after every update. Native adapters read gain through the same checked
Rust f32 exponential as the learning map, with no history allocation. Also
record the norm of `strength * gain * tanh(history_gate)`, which describes
gate/gain coupling, not hidden-state variance. Invalid updated gain is rejected
before a new checkpoint can be saved, including on the final primary step.
The ordinary control reports its own Torch f32 exponential independently.

The summary validates complete schedules, trajectory continuity, final
coordinates, learned capacity, shared GL initialization and first-update
receipts. First losses must match exactly; gate-gradient norms use fixed
rtol=2e-6, atol=2e-7. A mismatch is preserved as failed, not erased by a later
quality improvement. Norm receipts do not prove full tensor equality. The
actual learning/resume tests compare saved parameters and Adam state.

## Run And Resume

Freeze the client, shared driver/helpers, config and isolated native runtime
first. Use existing hash-matching local assets in offline mode and a new
study directory, never a previous completed study:

First run the training-only preflight against the completed factorial study.
It verifies the same base/data/schedule, actual BTF shape and initial filters,
two training updates per arm, plus one reference/replay continuation per arm.
It does not score held-out endpoints or save weights:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  python -P -B tools/preflight_fractional_gain_study.py \
  --client-root "$FROZEN_CLIENT" --package-root "$FROZEN_PACKAGE" \
  --config "$FROZEN_CLIENT/hf_fractional_pride_gain.json" \
  --model-dir "$LOCAL_GPT2" --corpus "$LOCAL_PRIDE" \
  --transfer-corpus "$LOCAL_ALICE" --previous-study "$PREVIOUS_FACTORIAL_STUDY" \
  --output "$NEW_PREFLIGHT_JSON"
```

Then use the frozen client for the primary run:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  python -P -B "$FROZEN_CLIENT/hf_fractional_gain_study.py" \
  --config "$FROZEN_CLIENT/hf_fractional_pride_gain.json" \
  --model-dir "$LOCAL_GPT2" --corpus "$LOCAL_PRIDE" \
  --transfer-corpus "$LOCAL_ALICE" --output-dir "$NEW_STUDY_DIRECTORY"
```

With `-P`, put the frozen client directory and native package on `PYTHONPATH`
so helper imports resolve to the sealed versions. Repeat identical arguments
with `--resume` to continue. The shared driver owns writer locking, immutable
checkpoints, source/runtime/data identity and exact cursors. Completed resume
verifies existing results without rescoring endpoints. Summarize read-only
with `tools/summarize_wave_gate_long_horizon.py` using plan/results/journal
and a fresh output; the receipt summary is Torch-free.

After terminal success and a no-op completed resume, verify the actual saved
tensors as a separate read-only step. This reconstructs the summary byte for
byte, restores each arm's exact recipe, validates parameter dtype/shape and
registration-bound Adam state, and compares the saved gain, shape coordinate
and effective gate norm with endpoint receipts. Native gain observation stays
in Rust. It hashes each parameter and named Adam tensor without publishing
weights. Use the frozen seven-file client and runtime inventories:

```sh
PYTHONPATH="$FROZEN_CLIENT:$FROZEN_PACKAGE:$REPOSITORY/tools" \
  python -P -B "$REPOSITORY/tools/verify_fractional_gain_study.py" \
  --study "$NEW_STUDY_DIRECTORY" --summary "$SUMMARY_JSON" \
  --client-manifest "$CLIENT_SHA256_JSON" \
  --runtime-manifest "$RUNTIME_SHA256_JSON" --output "$NEW_VERIFICATION_JSON"
```

The client manifest is a filename-to-SHA256 object. The runtime manifest is
`{"source_revision": "<build commit>", "files": {"spiraltorch/...": "<SHA256>"}}`,
relative to the package root. Build and training-launch revisions can differ
by documentation commits; executable byte identities are verified against
the plan. Place verification output outside the sealed client/runtime/study.
The verifier reads continuation receipts but does not rerun those updates,
score the model, or claim state equivalence between unlike arms. Frozen file
inventories tolerate generated `__pycache__` only; other extras are rejected.

Float32, complete unpadded prefixes and no KV cache or packed documents;
history resets at each block. All paths are host/CPU, not resident GPU.
Publish numeric outcomes, verification records, hashes and reproduction only,
not weights, corpus, native packages or raw private logs. The included tiny
random-HF fixtures validate the learning path, not pretrained quality. The
separate completed comparison above contains the real-model outcomes.
