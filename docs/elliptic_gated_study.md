# Paired Learned-Context Study

The [causal factorial](elliptic_causal_study.md) found that unconditional mixing
hurt both tangent and elliptic feature paths on the reused Pride/Alice endpoints.
The [Rust gated correction](elliptic_gated_context.md) now preserves local
features at initialization and lets training choose signed context contribution.
This study asks whether that change helps actual language learning, rather than
treating a nonzero gate or passing derivative tests as a quality improvement.

## Fixed Design

`bindings/st-py/examples/hf_elliptic_pride_gated.json` fixes four arms and three
seeds (41,43,47), each for 512 updates of a frozen local float32 GPT-2. The same
first-MLP output insertion, zero readout, strength 0.1, Adam learning rate 0.001,
batch 2, full unpadded context 128, data split and CPU thread count are retained.
No recipe changes are made from interim development losses. Endpoints are scored
only after all twelve arms complete and their checkpoints pass continuation gates.

| Arm | Features | Context | Trainable parameters at F=768 |
| --- | --- | --- | --- |
| tangent | ordinary linearization from Rust anchor/Jacobian | pointwise | 8450 |
| elliptic | Rust nonlinear elliptic/Lie map | pointwise | 8450 |
| gated_tangent | same ordinary linearization | learned signed correction | 8451 |
| gated_elliptic | same Rust nonlinear map | Rust learned signed correction | 8451 |

The gated pair shares parameter count and zero raw-gate initialization. All four
arms share the initial projection/readout tensors and mini-batch order within a
seed; hashes excluding only `raw_mix` establish that cross-group pairing. Full
parameter hashes also pair each within-group comparison. Pointwise controls do
not receive a fake unused parameter. Thus gated-minus-local effects include one
additional learned scalar, and neither expressivity nor computation is matched.

The ordinary control uses explicit Torch attention and the exact same signed
blend formula. Its anchor/Jacobian comes from Rust; it does not reimplement the
nonlinear geometry. Tests compare that formula to the native path at B=2,T=128
for negative, zero and positive gates. No timing comparison or acceleration claim
is made. Production geometry/blending/derivatives remain Rust-owned.

## Receipts And Interpretation

The existing restartable study loop is reused, with immutable checkpoint files,
an exclusive writer lock, a hash-sealed plan and atomic journal. Gated checkpoints
add a paired projection hash; every update records raw gate **signed** gradient,
before/after values and existing gradient norms. Endpoint reports record the
saved final raw gate. The offline summarizer rejects missing/nonfinite values,
discontinuous trajectories, endpoint mismatches, and unpaired initialization.
It reports the final effective `tanh(raw_mix)` descriptively, not as a quality win.

Primary contrast: gated elliptic minus gated tangent. Also report pointwise
geometry, each gate versus its own pointwise control, and the paired interaction.
Negative cross-entropy differences favor the named arm. Preserve losing seeds,
baseline comparisons and all conditions. Three seeds do not establish significance;
evaluation blocks are shared, not independent experimental replications.

Pride and Alice have already been inspected, and the gate was designed after a
negative result on them. This is exploratory reuse, not pristine confirmation.
Pointwise arms are rerun with this exact package/source binding; exact historical
replays, if found, are reproducibility evidence rather than independent successes.

## Run Offline

Use a package built from the gated-context implementation, plus matching client,
Torch and Transformers 4.57.6. No model download, API request or hosted training
is required. Models, corpus text, raw logs and checkpoints stay local; publish
the numeric results, hashes and reproduction receipts separately.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
python bindings/st-py/examples/hf_elliptic_gated_study.py \
  --config bindings/st-py/examples/hf_elliptic_pride_gated.json \
  --model-dir /path/to/gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --corpus /path/to/pride_and_prejudice.txt \
  --transfer-corpus /path/to/alices_adventure_in_wonderland.txt \
  --output-dir /path/to/new-study
```

Use `--resume` only against the original directory with unchanged frozen source,
package, data and recipe. A completed resume checks receipts without rerunning
endpoints. Summarize with `tools/summarize_wave_gate_long_horizon.py`; it reads
sealed JSON (or gzip JSON) and never imports Torch or evaluates the model again.

Preflight tests are `test_elliptic_gated_study.py` and the shared geometry/study
suite. The miniature HF interrupt/resume test verifies full adapter/Adam/record
equality, native gradients and signed gate endpoint records. This preflight is
not itself the twelve-run pretrained-model study.

## Completed Outcome

The [completed twelve-run record](../benchmarks/results/2026-10-03-elliptic-gated-study/README.md)
shows a modest mean gain for gated versus pointwise elliptic (Pride -0.009083 CE,
Alice -0.009992), but no geometry advantage: gated elliptic still loses to gated
tangent in every seed on both sets. Pride seed 43 is a retained elliptic regression.
All elliptic gates finish negative, while ordinary tangent gates finish slightly
positive. The role of local gain/centering versus true contextual benefit remains
unresolved. The record includes actual interruption recovery, clean resumed
completion, exact final resume, checkpoint checks and six exact historical
pointwise replays; those replays are not independent confirmations.
