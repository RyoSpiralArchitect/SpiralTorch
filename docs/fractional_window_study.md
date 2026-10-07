# Full-Normalization Matched Learning

The [saved-model intervention](fractional_window_diagnostic.md) found that
removing long taps hurts the completed full-history models. This new study
trains both conditions from the same initialization instead of changing an
already-trained model's inputs.

## Fixed Comparison

`hf_fractional_pride_window.json` declares two native arms:

- `history_window_short`: retain [1,3) after normalization over K=32.
- `history_window_full`: retain all past taps after normalization over K=32.

Both use Rust's angular GL chart, full-K coefficient normalization and its
derivative, learned positive log gain, local/history feature gates, first-order
autograd and the same 1538 parameter registration order. Python orchestrates
the experiment; it does not reimplement the filter or substitute Torch math.
The short arm is not a separately normalized K=3 kernel. Do not silently load
prior full/short recipes: this study has its own exact saved-state schema.

The order-two initialization has identical nonzero first two taps and zero
remaining taps. Both start as identity through zero feature gates. The zero
tail still has a nonzero order differential, so full and short learning may
diverge after the feature gate opens. Do not detach these zeros, add jitter or
train an initially dormant tail-only branch as a replacement comparison.

Use the same frozen local GPT-2, Pride/Alice token partitions, seeds 41/43/47,
paired minibatch schedules, Adam 0.001, strength 0.1, batch two, context 128,
512 updates per arm, CPU float32/two threads, no padding/packing and no KV
cache. All six runs must finish before endpoints, followed by 12 separate
next-update/Adam continuation checks. Development never selects settings,
duration, checkpoint or seeds. Any angle-domain exit terminally locks the
whole study and its endpoint evaluation; no clipping, wrapping or projection.

The primary contrast is full minus retained-short CE (negative favors full).
Learned amplitude can compensate scale, but support changes its optimization
trajectory too. This tests this declared learning mechanism and budget, not
every short-filter parameterization or unique long-memory causation.

## Run And Verify

Freeze the new client, preflight and summary source files plus the reused
native runtime manifest before admission. Use a new output directory and
leave all previous clients, checkpoints, summaries and runtimes untouched.
The shared tools now accept `--coordinate window`; old modes are unchanged.

1. Run `tools/preflight_fractional_gain_study.py --coordinate window` with the
   fixed window config and previous completed angular study. This uses only
   two training batches from seed 41 per arm plus separate exact continuation
   checks (four auxiliary + four validation-only updates); no heldout scoring.
2. Run `bindings/st-py/examples/hf_fractional_window_study.py` with `--config`,
   `--model-dir`, `--corpus`, `--transfer-corpus`, `--output-dir`. An interrupted
   nonterminal run can resume with the identical binding and `--resume`.
3. Use `tools/summarize_wave_gate_long_horizon.py` on the completed plan,
   result and journal. It verifies paired budgets, initialization, angle/gain
   trajectories and the declared support/normalization recipe without Torch.
4. Use `tools/verify_fractional_gain_study.py --coordinate window` with the
   frozen client/runtime manifests and derived summary. It reads saved final
   parameters and named Adam moments, not new model scores. No final-state
   equality between different-support arms is required or claimed.

Keep `PYTHONNOUSERSITE=1`, `SPIRALTON_MAGIC=0`, `SPIRALTON_TORCH=0`,
`SPIRALTON_MODEL_PATCHES=0`, `SPIRALTON_NUMPY=0`, `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `OMP_NUM_THREADS=2` and
`TOKENIZERS_PARALLELISM=false`. Import only the frozen client and candidate
package using explicit `PYTHONPATH` and Python `-P -B`.

## Evidence Boundaries

These are reused exploratory books and sample-order seeds, not pristine
confirmation or new independent replicas. The new full arm may reproduce the
earlier full arm; verify any replay separately and retain failures. Do not
count it as independent evidence or overwrite old records. The old
ordinary/GL-short final-parameter parity failure stays failed.

Publish every condition, numeric block loss, relevant checkpoint/runtime/source
hash and process outcome, including failures. Weights, corpus text, native
packages and raw logs stay local. Equal updates are not equal arithmetic;
this different-support quality comparison is not a Torch speed benchmark.
