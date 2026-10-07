# Fractional input VJPs in multi-site HF learning

The [active-support optimization](fractional_active_support.md) reduces the
cost of a Rust history input VJP. A single adapter on a frozen base does not
need that input gradient. This auxiliary check uses **two** public
`spiraltorch.FractionalAngleGainHistoryAdapter` instances: the second one's
input depends on the first one's trainable parameters, through frozen HF
layers. Rust still owns the history operator, normalization, angle chart and
their differentials; this tool adds no Python replacement geometry.

## Observed result

The [numeric archive](../benchmarks/results/2026-10-07-fractional-multi-adapter-replay/README.md)
records a local pretrained GPT-2, CPU float32, two Torch threads, `[2,128,768]`
hidden states and 3,076 trainable adapter parameters. The original base weights
and buffers remain bit-identical, with no base gradients. The model runs in
evaluation mode to disable dropout, but autograd remains enabled for adapters.

For each of full history and retained lags `[1,3)`, with full-K=32 normalization:

1. The old native runtime performs four auxiliary training updates and saves
   a fresh checkpoint after update two.
2. The optimized runtime repeats all four updates from identical initialization.
3. A fresh optimized-runtime process loads the old midpoint and executes updates
   three and four. It does not rerun the prefix.

All losses, per-parameter gradients, adapter states, named Adam states and Torch
RNG states match bitwise, including signed-zero tensor bytes. Both windows pass:
six successful processes, 20 executed auxiliary updates, only eight unique
trajectory updates. These are not additional independent seeds or a new study.

The native route is observed, not inferred from configuration:

| Site | Backward operation | Input gradient required? |
| --- | --- | --- |
| First MLP adapter | `vjp_parameters_buffer` | No, the preceding base is frozen |
| Second MLP adapter | `vjp_buffer` | Yes, to reach the first adapter |

Both gates start at zero. The second input VJP is consequently zero on update
one, then nonzero on updates two through four. All eight trainable parameter
groups receive nonzero gradients after the initial update. The public archive
includes their per-update hashes and norms, not hidden-state or weight values.

Regression tests also replace only the later native input VJP with zeros: the
forward loss and later adapter gradients stay identical, but earlier adapter
gradients change. Another test compares observation enabled/disabled and finds
identical learning states, so instrumentation does not supply a derivative rule.

## Reproduce Offline

Use `tools/probe_fractional_multi_adapter.py` and its adjacent verifier helper
from the source revision in `summary.json`. The sample configuration is
`bindings/st-py/examples/hf_fractional_dual_replay.json`. Insertion module paths
are configuration, not hardcoded GPT-2 logic; this test nevertheless validates
only the named GPT-2 snapshot. Targets must accept a single positional tensor
and return float32 `[batch,time,feature]` tensors. Use full unpadded prefixes and
disable KV cache; tuple/keyword-based block interfaces need a separate wrapper.

Provide two isolated package roots with versioned runtime manifests of the form
`{"source_revision": "<40 hex>", "files": {"relative/path": "<sha256>"}}`.
The tested package inventories differ only in the native binary; the Python
bridge is identical. Native source revisions and file hashes are in the archive.
Use fresh output directories on each invocation. Do not edit frozen packages or
reuse any completed study checkpoint for this new two-site recipe.

```bash
export PYTHONNOUSERSITE=1
export SPIRALTON_MAGIC=0 SPIRALTON_TORCH=0 SPIRALTON_MODEL_PATCHES=0 SPIRALTON_NUMPY=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false

# Set PYTHON, REPO, OUT, MODEL_DIR, CORPUS, BEFORE_PACKAGE, AFTER_PACKAGE,
# BEFORE_MANIFEST and AFTER_MANIFEST to the local pinned inputs.
CONFIG="$REPO/bindings/st-py/examples/hf_fractional_dual_replay.json"
PROBE="$REPO/tools/probe_fractional_multi_adapter.py"
OLD_NATIVE_SHA=18f4c1c2bb4b76fa0e8beeca59f7fbfa0dffbc85efd72b96c45c119a63befa58

for WINDOW in full short; do
  PYTHONPATH="$BEFORE_PACKAGE:$REPO/tools" "$PYTHON" -P -B "$PROBE" run \
    --config "$CONFIG" --model-dir "$MODEL_DIR" --corpus "$CORPUS" \
    --package-root "$BEFORE_PACKAGE" --runtime-manifest "$BEFORE_MANIFEST" \
    --window "$WINDOW" --output "$OUT/$WINDOW-before"
  PYTHONPATH="$AFTER_PACKAGE:$REPO/tools" "$PYTHON" -P -B "$PROBE" run \
    --config "$CONFIG" --model-dir "$MODEL_DIR" --corpus "$CORPUS" \
    --package-root "$AFTER_PACKAGE" --runtime-manifest "$AFTER_MANIFEST" \
    --window "$WINDOW" --output "$OUT/$WINDOW-after"
  CHECKPOINT="$OUT/$WINDOW-before/midpoint.pt"
  CHECKPOINT_SHA=$(shasum -a 256 "$CHECKPOINT" | cut -d ' ' -f1)
  PYTHONPATH="$AFTER_PACKAGE:$REPO/tools" "$PYTHON" -P -B "$PROBE" run \
    --config "$CONFIG" --model-dir "$MODEL_DIR" --corpus "$CORPUS" \
    --package-root "$AFTER_PACKAGE" --runtime-manifest "$AFTER_MANIFEST" \
    --window "$WINDOW" --output "$OUT/$WINDOW-resumed" \
    --resume "$CHECKPOINT" --resume-sha256 "$CHECKPOINT_SHA" \
    --allow-source-native-sha256 "$OLD_NATIVE_SHA"
  PYTHONPATH="$AFTER_PACKAGE:$REPO/tools" "$PYTHON" -P -B "$PROBE" compare \
    --before "$OUT/$WINDOW-before" --after "$OUT/$WINDOW-after" \
    --resumed "$OUT/$WINDOW-resumed" --output "$OUT/$WINDOW-comparison.json"
done
```

`run` verifies package inventory and hashes before and after execution. Resume
requires the checkpoint hash, explicit source-native admission, identical
model/data/recipe/source binding, ordered parameter names, finite states and
matching Adam configuration/cursor. Loading uses `weights_only=True`. The
comparison checks actual saved tensor bytes, not just JSON success flags.
This is a bounded migration test, not permission to admit arbitrary runtimes.

## Boundaries

- No heldout scoring, generation, long-training convergence or model-quality
  claim. Training batches differ between updates; their loss sequence is not
  a fixed-batch learning curve. Full/short differences are not ranked here.
- No end-to-end throughput claim. Observation hashes and norms deliberately
  add overhead. The earlier matched operator timings remain separate evidence.
- This HF check is CPU only; it adds no CUDA, WGPU, WASM-HF or KV-cache proof.
- An initial manifest-envelope mismatch stopped before model load or training.
  It was fixed with an envelope/inventory regression test; the failure log and
  original client remain preserved privately. No previous study was rerun.

The next learning experiment can now compare single versus multiple insertions
under a fixed total budget, rather than assuming that a faster primitive helps
a training path which never requests it.
