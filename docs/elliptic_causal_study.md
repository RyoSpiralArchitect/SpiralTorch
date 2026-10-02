# Separating Geometry From Token Mixing

This fixed-budget, offline study crosses two factors rather than treating a
richer adapter as evidence for geometry by itself:

| Arm | Features | Token mixing |
| --- | --- | --- |
| `tangent` | Rust anchor + local Jacobian | None |
| `elliptic` | Rust elliptic map | None |
| `causal_tangent` | Rust anchor + local Jacobian | Ordinary causal attention |
| `causal_elliptic` | Rust elliptic map | Rust causal attention + composed VJP |

All arms have `11*F+2` trainable parameters (8450 at GPT-2 width 768), paired
initial parameter hashes, zero residual readout, Adam settings, sample order and
update count. This does **not** equalize effective feature rank, expressivity,
conditioning or compute. The first MLP output receives the adapter; all original
model parameters are frozen. The pointwise arms are rerun, not spliced from the
previous study's results.

Causal mixing ties Q/K/V to the nine features with scores `dot(f_i,f_j)/3` for
`j <= i`. The ordinary control uses explicit Torch operations with f64
intermediates; native geometric attention uses the existing checked f32 scores
and wide gradient accumulation. Tests verify forward/VJP tolerance agreement,
not bitwise arithmetic equivalence. The chart anchor and differential come from
Rust in both controls; no elliptic formula is reconstructed in Python.

## Frozen Protocol

The checked-in recipe uses seeds 41/43/47, 512 updates per arm (6144 primary
updates total), batch 2, 128-token contexts, Adam 0.001 and residual strength 0.1.
Only adapter parameters are optimized, with exact optimizer continuation checks.
Checkpoints occur every 64 updates and a fixed development probe every 128.
All twelve runs must complete and pass continuation/base-freeze checks before
endpoint evaluation. A completed resume verifies hashes without scoring again.

The model is local GPT-2 snapshot `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
The recipe pins the same complete Pride/Alice file hashes as earlier studies;
tokenized train, development and endpoint block identities are bound in the plan.
Those endpoints have already been inspected: this is an **exploratory reused
benchmark**, not pristine held-out confirmation. Do not select a winning seed or
replace the fixed final checkpoint with an endpoint-selected checkpoint.

Set `PYTHONPATH` to the installed native package and this checkout's
`bindings/st-py/examples` when using Python's safe-path mode. Disable any local
auto-patching helpers and keep Hugging Face offline. With local paths supplied:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
python -P bindings/st-py/examples/hf_elliptic_causal_study.py \
  --config bindings/st-py/examples/hf_elliptic_pride_causal.json \
  --model-dir "$MODEL_SNAPSHOT_DIR" \
  --corpus "$PRIDE_TEXT" --transfer-corpus "$ALICE_TEXT" \
  --output-dir "$NEW_STUDY_DIRECTORY"
```

Reuse the exact configuration/runtime/source files and add `--resume` after an
interruption. The plan binds all adapter source files, the shared driver/helper,
Python bridge, native extension, model state and data. Keep checkpoints, model
and source corpus local; publish numeric results and verification receipts.

## Interpretation

For each seed and evaluation set, report these CE differences (negative is
better for the first named arm):

- Geometry without mixing: `elliptic - tangent`.
- Geometry with mixing: `causal_elliptic - causal_tangent`.
- Mixing on ordinary features: `causal_tangent - tangent`.
- Mixing on geometric features: `causal_elliptic - elliptic`.
- Interaction: `(causal_elliptic - causal_tangent) - (elliptic - tangent)`.

An interaction does not by itself establish a useful absolute improvement; also
report all arms and the unchanged base model. Three seeds sharing evaluation
blocks do not justify a significance or generalization claim.

After completion, `tools/summarize_wave_gate_long_horizon.py` validates the sealed
numeric results and emits all five paired contrasts for this protocol, including
opposing seeds. It additionally checks paired initial hashes and parameter counts.
It does not import Torch, rerun endpoints or treat individual blocks as new seeds.

## Validation And Limits

`test_elliptic_causal_study.py` exercises all four paired initializations, CPU and
accelerator RNG isolation, the ordinary control's tangent map, equivalent Torch
attention versus native geometric VJP, future/batch isolation, pair budgets,
cross-arm checkpoint rejection, source binding, and actual tiny-HF learning with
an interrupted causal run resumed exactly. The CI geometry suite includes it.

The [causal bridge contract](elliptic_causal_learning.md) still applies: full
unpadded sequences only, `use_cache=False`, first-order float32 inputs, bounded
rows/pairs and explicit Rust CPU transport. Neither speed, resident WGPU, cached
generation nor pretrained quality improvement follows from these tests. The
tests alone do not establish pretrained-model quality; the completed experiment
is recorded separately below.

## Completed Run

The [12-run numeric record](../benchmarks/results/2026-10-02-elliptic-causal-factorial/README.md)
now contains all final checkpoints' receipts and sealed endpoints. Both geometry
and causal mixing lose their matched contrasts in every seed on both evaluation
sets; all active adapters improve the unchanged base. The pointwise tangent arm
remains best. Six rerun pointwise results exactly match the previous study and
must not be counted as additional independent evidence.

This supports the learning/continuation contract, not promoting causal elliptic
mixing as a better default. A next architectural hypothesis is to preserve the
current-token feature path and add a learnable contextual correction rather than
replace it with mixed features. That would require new paired ordinary controls
and a new fixed protocol; this run does not establish the cause of the loss.
